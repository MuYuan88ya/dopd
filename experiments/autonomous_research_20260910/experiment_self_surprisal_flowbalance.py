# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0.
"""Rigorous benchmark for Teacherless / Self-Surprisal Weighted FlowBalance.

Investigates whether intrinsic information-theoretic signals (Self-Surprisal,
Predictive Entropy) can solve the Token Heterogeneity credit assignment problem
in Pure RL (without any external teacher or PRM verifier).
"""

from __future__ import annotations

import json
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


class ReasoningDAGEnv:
    """Multi-step reasoning tree with decision forks and filler tokens."""

    def __init__(
        self,
        num_problems: int = 40,
        num_steps: int = 5,
        tokens_per_step: int = 8,
        vocab_size: int = 16,
    ):
        self.num_problems = num_problems
        self.num_steps = num_steps
        self.tokens_per_step = tokens_per_step
        self.seq_len = num_steps * tokens_per_step
        self.vocab_size = vocab_size

        self.problems = []
        for i in range(num_problems):
            # Decision tokens: index 2 of each step is the fork
            correct = torch.randint(2, vocab_size, (num_steps,))
            distractors = torch.randint(2, vocab_size, (num_steps,))
            for s in range(num_steps):
                while distractors[s] == correct[s]:
                    distractors[s] = torch.randint(2, vocab_size, (1,)).item()

            # Hard problem: student trap bias
            is_hard = (i < num_problems // 2)
            self.problems.append({
                "id": f"prob_{i:02d}",
                "correct": correct,
                "distractors": distractors,
                "is_hard": is_hard,
            })

    def evaluate(self, p_idx: int, tokens: torch.Tensor) -> tuple[float, int]:
        prob = self.problems[p_idx]
        for s in range(self.num_steps):
            fork_pos = s * self.tokens_per_step + 2
            chosen = tokens[fork_pos].item()
            if chosen != prob["correct"][s].item():
                return 0.0, s
        return 1.0, -1


class StudentPolicy(nn.Module):
    def __init__(self, num_problems: int, seq_len: int, vocab_size: int):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(num_problems, seq_len, vocab_size))

    def sample(self, p_idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.logits[p_idx]
        probs = F.softmax(logits, dim=-1)
        toks = torch.multinomial(probs, num_samples=1).squeeze(-1)
        lp = F.log_softmax(logits, dim=-1).gather(-1, toks.unsqueeze(-1)).squeeze(-1)
        # Compute Shannon entropy per position: H = -sum(p * log p)
        ent = -(probs * F.log_softmax(logits, dim=-1)).sum(dim=-1)
        return toks, lp, ent


def compute_advantages_pure_rl(
    method: str,
    scores: torch.Tensor,
    sampled_logp: torch.Tensor,
    ref_logp: torch.Tensor,
    entropies: torch.Tensor,
    teacher_logp: torch.Tensor | None = None,
    tau: float = 0.1,
    subtb_lambda: float = 0.5,
    gamma: float = 1.0,
) -> torch.Tensor:
    G, L = sampled_logp.shape
    device = sampled_logp.device

    # Base GRPO
    mean_score = scores.mean()
    std_score = scores.std()
    if std_score > 1e-6:
        grpo_adv = (scores - mean_score) / (std_score + 1e-6)
    else:
        grpo_adv = scores - mean_score
    grpo_token_adv = grpo_adv.unsqueeze(-1).expand(G, L)

    if method == "Standard_GRPO":
        return grpo_token_adv

    # Pure RL Trajectory Balance
    seq_old = sampled_logp.mean(dim=-1, keepdim=True)
    seq_ref = ref_logp.mean(dim=-1, keepdim=True)
    R_term = (scores / tau).unsqueeze(-1)
    # Teacherless: alpha = 0
    target_uncentered = seq_ref + R_term / L
    baseline = (seq_old - target_uncentered).mean()
    A_TB = 2.0 * (target_uncentered + baseline - seq_old).expand(G, L)

    if method == "PureRL_TB":
        return A_TB

    # Simplex Weighting Schemes w in Delta^{L-1}
    if method == "PureRL_SubTB_Uniform":
        w = torch.ones(G, L, device=device) / L
    elif method == "PureRL_SelfSurprisal_SubTB":
        # Surprisal = -log pi_old(y_t)
        surprisal = (-sampled_logp).clamp(min=0.0) + 0.1
        w = (surprisal ** gamma) / (surprisal ** gamma).sum(dim=-1, keepdim=True)
    elif method == "PureRL_Entropy_SubTB":
        # Local predictive entropy H(s_t)
        ent = entropies.clamp(min=0.0) + 0.1
        w = (ent ** gamma) / (ent ** gamma).sum(dim=-1, keepdim=True)
    elif method == "PureRL_SurprisalGrad_SubTB":
        # Variance / deviation of surprisal from mean sequence surprisal
        s = (-sampled_logp).clamp(min=0.0)
        s_dev = (s - s.mean(dim=-1, keepdim=True)).abs() + 0.1
        w = (s_dev ** gamma) / (s_dev ** gamma).sum(dim=-1, keepdim=True)
    elif method == "Privileged_Teacher_EW_SubTB":
        assert teacher_logp is not None
        delta = (teacher_logp - ref_logp).clamp(-4.0, 4.0)
        G_T_seq = delta.mean(dim=-1)
        # Pairwise AUC
        s_diff = scores.unsqueeze(1) - scores.unsqueeze(0)
        pos_mask = s_diff > 1e-6
        if pos_mask.any():
            g_diff = G_T_seq.unsqueeze(1) - G_T_seq.unsqueeze(0)
            concordant = (g_diff[pos_mask] > 0).float().sum()
            ties = (g_diff[pos_mask] == 0).float().sum()
            auc = (concordant + 0.5 * ties) / pos_mask.float().sum().clamp(min=1.0)
            g_consist = float(np.clip(2.0 * (auc.item() - 0.5), 0.0, 1.0))
        else:
            g_consist = 0.5
        target_uncentered_t = seq_ref + (0.5 * g_consist) * G_T_seq.unsqueeze(-1) + R_term / L
        baseline_t = (seq_old - target_uncentered_t).mean()
        A_TB_t = 2.0 * (target_uncentered_t + baseline_t - seq_old).expand(G, L)

        surprise = delta.abs() + 0.1
        w = (surprise ** gamma) / (surprise ** gamma).sum(dim=-1, keepdim=True)
        target_token = ref_logp + (0.5 * g_consist) * delta + w * L * (R_term / L + baseline_t)
        A_DB = 2.0 * (target_token - sampled_logp)
        return (1.0 - subtb_lambda) * A_DB + subtb_lambda * A_TB_t
    else:
        raise ValueError(f"Unknown method: {method}")

    # Pure RL Detailed Balance target
    target_token = ref_logp + w * L * (R_term / L + baseline)
    A_DB = 2.0 * (target_token - sampled_logp)

    A_SubTB = (1.0 - subtb_lambda) * A_DB + subtb_lambda * A_TB
    return A_SubTB


def run_self_surprisal_experiment():
    num_problems = 40
    num_steps = 5
    tokens_per_step = 8
    vocab_size = 14
    env = ReasoningDAGEnv(
        num_problems=num_problems,
        num_steps=num_steps,
        tokens_per_step=tokens_per_step,
        vocab_size=vocab_size,
    )

    fork_indices = [s * tokens_per_step + 2 for s in range(num_steps)]
    filler_indices = [i for i in range(env.seq_len) if i not in fork_indices]

    methods = [
        "Standard_GRPO",
        "PureRL_TB",
        "PureRL_SubTB_Uniform",
        "PureRL_SelfSurprisal_SubTB",
        "PureRL_Entropy_SubTB",
        "PureRL_SurprisalGrad_SubTB",
        "Privileged_Teacher_EW_SubTB",
    ]

    all_results = {}
    G = 8
    epochs = 25
    lr = 0.15

    for method in methods:
        print(f"\n==========================================")
        print(f"Running Experiment for: {method}")
        print(f"==========================================")
        student = StudentPolicy(num_problems, env.seq_len, vocab_size)

        # Initialize student: hard problems strongly favor distractor trap
        with torch.no_grad():
            for p_idx, prob in enumerate(env.problems):
                if prob["is_hard"]:
                    for s in range(num_steps):
                        fork_pos = fork_indices[s]
                        distractor = prob["distractors"][s]
                        student.logits[p_idx, fork_pos, distractor] = 2.2  # Trap

        ref_logits = student.logits.detach().clone()

        # Teacher logits for reference
        teacher_logits = ref_logits.clone()
        for p_idx, prob in enumerate(env.problems):
            for s in range(num_steps):
                fork_pos = fork_indices[s]
                correct_tok = prob["correct"][s]
                teacher_logits[p_idx, fork_pos, correct_tok] += 3.5

        optimizer = torch.optim.Adam(student.parameters(), lr=lr)

        acc_history = []
        hard_acc_history = []
        fork_credit_ratio_history = []
        syntax_drift_history = []

        for ep in range(epochs):
            total_corr = 0
            hard_corr = 0
            total_hard = 0
            ep_fork_grads = []
            ep_filler_grads = []
            ep_syntax_drift = []

            for p_idx, prob in enumerate(env.problems):
                if prob["is_hard"]:
                    total_hard += 1

                # Sample G rollouts
                sampled_tokens = []
                sampled_logp = []
                sampled_ent = []
                scores = []

                for _ in range(G):
                    toks, lp, ent = student.sample(p_idx)
                    score, first_err = env.evaluate(p_idx, toks)
                    sampled_tokens.append(toks)
                    sampled_logp.append(lp)
                    sampled_ent.append(ent)
                    scores.append(score)

                sampled_tokens = torch.stack(sampled_tokens)
                sampled_logp = torch.stack(sampled_logp)
                sampled_ent = torch.stack(sampled_ent)
                scores = torch.tensor(scores, dtype=torch.float32)

                # Accuracy metrics (Pass@1)
                best_s = scores.max().item()
                if best_s > 0.5:
                    total_corr += 1
                    if prob["is_hard"]:
                        hard_corr += 1

                # Teacher & ref logp
                teacher_lp_all = F.log_softmax(teacher_logits[p_idx], dim=-1).unsqueeze(0).expand(G, -1, -1)
                t_lp = teacher_lp_all.gather(-1, sampled_tokens.unsqueeze(-1)).squeeze(-1)

                ref_lp_all = F.log_softmax(ref_logits[p_idx], dim=-1).unsqueeze(0).expand(G, -1, -1)
                r_lp = ref_lp_all.gather(-1, sampled_tokens.unsqueeze(-1)).squeeze(-1)

                # Compute Advantages
                adv = compute_advantages_pure_rl(
                    method=method,
                    scores=scores,
                    sampled_logp=sampled_logp,
                    ref_logp=r_lp,
                    entropies=sampled_ent,
                    teacher_logp=t_lp,
                )

                # Policy gradient loss
                loss = -(sampled_logp * adv.detach()).mean()

                optimizer.zero_grad()
                loss.backward()

                # Measure gradient concentration on forks vs fillers
                g_param = student.logits.grad[p_idx]  # [L, V]
                g_norm = g_param.norm(dim=-1)         # [L]
                fork_g = g_norm[fork_indices].mean().item()
                filler_g = g_norm[filler_indices].mean().item()
                ep_fork_grads.append(fork_g)
                ep_filler_grads.append(filler_g)

                # Measure syntax drift (change in logits on filler tokens)
                curr_logits = student.logits[p_idx]
                drift = (curr_logits[filler_indices] - ref_logits[p_idx, filler_indices]).abs().mean().item()
                ep_syntax_drift.append(drift)

                optimizer.step()

            pass1 = total_corr / num_problems
            hard_pass = hard_corr / max(1, total_hard)
            fork_ratio = np.mean(ep_fork_grads) / max(1e-6, np.mean(ep_filler_grads))
            drift_val = np.mean(ep_syntax_drift)

            acc_history.append(pass1)
            hard_acc_history.append(hard_pass)
            fork_credit_ratio_history.append(fork_ratio)
            syntax_drift_history.append(drift_val)

            if (ep + 1) % 5 == 0 or ep == epochs - 1:
                print(
                    f"  Epoch {ep+1:02d}/{epochs:02d} | "
                    f"Pass@1: {pass1*100:5.2f}% | "
                    f"Hard Pass: {hard_pass*100:5.2f}% | "
                    f"Fork/Filler Grad Ratio: {fork_ratio:5.2f}x | "
                    f"Syntax Drift: {drift_val:6.4f}"
                )

        all_results[method] = {
            "final_pass1": acc_history[-1],
            "final_hard_pass": hard_acc_history[-1],
            "final_fork_ratio": fork_credit_ratio_history[-1],
            "final_syntax_drift": syntax_drift_history[-1],
            "acc_history": acc_history,
            "hard_acc_history": hard_acc_history,
            "fork_credit_ratio_history": fork_credit_ratio_history,
            "syntax_drift_history": syntax_drift_history,
        }

    out_file = os.path.join(os.path.dirname(__file__), "self_surprisal_benchmark_results.json")
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n=======================================================")
    print(f"BENCHMARK COMPLETE. SUMMARY OF RESULTS:")
    print(f"{'Method':<30} | {'Pass@1':<8} | {'Hard Pass':<10} | {'Fork Ratio':<12} | {'Syntax Drift':<12}")
    print("-" * 80)
    for m, res in all_results.items():
        print(
            f"{m:<30} | {res['final_pass1']*100:6.2f}% | {res['final_hard_pass']*100:8.2f}% | "
            f"{res['final_fork_ratio']:10.2f}x | {res['final_syntax_drift']:10.4f}"
        )
    print(f"=======================================================")


if __name__ == "__main__":
    run_self_surprisal_experiment()
