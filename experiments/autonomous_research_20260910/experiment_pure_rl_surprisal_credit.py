# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0.
"""Pure RL Credit Assignment: Self-Surprisal vs Uniform Token Weighting.

Investigates whether Self-Surprisal Flow Weighting solves the Token Heterogeneity
pathology in Pure RL (when positive rewards exist, but are diluted across long sequences of filler tokens).
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


class SolvableDAGEnv:
    """Multi-step reasoning tree where exploration can discover the solution."""

    def __init__(
        self,
        num_problems: int = 30,
        num_steps: int = 3,
        tokens_per_step: int = 8,
        vocab_size: int = 8,
    ):
        self.num_problems = num_problems
        self.num_steps = num_steps
        self.tokens_per_step = tokens_per_step
        self.seq_len = num_steps * tokens_per_step
        self.vocab_size = vocab_size

        self.problems = []
        for i in range(num_problems):
            correct = torch.randint(2, vocab_size, (num_steps,))
            distractors = torch.randint(2, vocab_size, (num_steps,))
            for s in range(num_steps):
                while distractors[s] == correct[s]:
                    distractors[s] = torch.randint(2, vocab_size, (1,)).item()

            is_hard = (i < num_problems // 3)
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
        ent = -(probs * F.log_softmax(logits, dim=-1)).sum(dim=-1)
        return toks, lp, ent


def compute_pure_rl_adv(
    method: str,
    scores: torch.Tensor,
    sampled_logp: torch.Tensor,
    ref_logp: torch.Tensor,
    entropies: torch.Tensor,
    tau: float = 0.1,
    subtb_lambda: float = 0.5,
    gamma: float = 1.0,
) -> torch.Tensor:
    G, L = sampled_logp.shape
    device = sampled_logp.device

    # 1. Base GRPO
    mean_s = scores.mean()
    std_s = scores.std()
    if std_s > 1e-6:
        grpo_adv = (scores - mean_s) / (std_s + 1e-6)
    else:
        grpo_adv = scores - mean_s

    if method == "Standard_GRPO":
        return grpo_adv.unsqueeze(-1).expand(G, L)

    # 2. Pure RL Trajectory Balance
    seq_old = sampled_logp.mean(dim=-1, keepdim=True)
    seq_ref = ref_logp.mean(dim=-1, keepdim=True)
    R_term = (scores / tau).unsqueeze(-1)
    target_uncentered = seq_ref + R_term / L
    baseline = (seq_old - target_uncentered).mean()
    A_TB = 2.0 * (target_uncentered + baseline - seq_old).expand(G, L)

    if method == "PureRL_TB":
        return A_TB

    # 3. Simplex weights
    if method == "PureRL_SubTB_Uniform":
        w = torch.ones(G, L, device=device) / L
    elif method == "PureRL_SelfSurprisal_SubTB":
        # w proportional to (-log p_old + eps)^gamma
        surp = (-sampled_logp).clamp(min=0.0) + 0.1
        w = (surp ** gamma) / (surp ** gamma).sum(dim=-1, keepdim=True)
    elif method == "PureRL_Entropy_SubTB":
        # w proportional to predictive entropy H(s_t)
        ent = entropies.clamp(min=0.0) + 0.1
        w = (ent ** gamma) / (ent ** gamma).sum(dim=-1, keepdim=True)
    elif method == "PureRL_Step_Uniform":
        # Uniform across 3 steps: each step of 8 tokens gets 1/3 total weight
        # so each token in step gets (1/3) / 8 = 1/24
        w = torch.ones(G, L, device=device) / L
    else:
        raise ValueError(f"Unknown method: {method}")

    target_token = ref_logp + w * L * (R_term / L + baseline)
    A_DB = 2.0 * (target_token - sampled_logp)

    A_SubTB = (1.0 - subtb_lambda) * A_DB + subtb_lambda * A_TB
    return A_SubTB


def run_experiment():
    num_problems = 30
    num_steps = 3
    tokens_per_step = 8
    vocab_size = 8
    env = SolvableDAGEnv(
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
    ]

    all_results = {}
    G = 12
    epochs = 30
    lr = 0.12

    for method in methods:
        print(f"\n--- Testing Method: {method} ---")
        student = StudentPolicy(num_problems, env.seq_len, vocab_size)

        with torch.no_grad():
            # For filler tokens: initialize strong preference for token 0 (mimicking real text boilerplate)
            for p_idx in range(num_problems):
                for fill_pos in filler_indices:
                    student.logits[p_idx, fill_pos, 0] = 3.0
                # For hard problems: bias fork to distractor
                prob = env.problems[p_idx]
                if prob["is_hard"]:
                    for s in range(num_steps):
                        fork_pos = fork_indices[s]
                        student.logits[p_idx, fork_pos, prob["distractors"][s]] = 1.8

        ref_logits = student.logits.detach().clone()
        optimizer = torch.optim.Adam(student.parameters(), lr=lr)

        acc_history = []
        hard_acc_history = []
        fork_ratio_history = []
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

                sampled_tokens = []
                sampled_logp = []
                sampled_ent = []
                scores = []

                for _ in range(G):
                    toks, lp, ent = student.sample(p_idx)
                    score, _ = env.evaluate(p_idx, toks)
                    sampled_tokens.append(toks)
                    sampled_logp.append(lp)
                    sampled_ent.append(ent)
                    scores.append(score)

                sampled_tokens = torch.stack(sampled_tokens)
                sampled_logp = torch.stack(sampled_logp)
                sampled_ent = torch.stack(sampled_ent)
                scores = torch.tensor(scores, dtype=torch.float32)

                if scores.max().item() > 0.5:
                    total_corr += 1
                    if prob["is_hard"]:
                        hard_corr += 1

                ref_lp_all = F.log_softmax(ref_logits[p_idx], dim=-1).unsqueeze(0).expand(G, -1, -1)
                r_lp = ref_lp_all.gather(-1, sampled_tokens.unsqueeze(-1)).squeeze(-1)

                adv = compute_pure_rl_adv(
                    method=method,
                    scores=scores,
                    sampled_logp=sampled_logp,
                    ref_logp=r_lp,
                    entropies=sampled_ent,
                )

                loss = -(sampled_logp * adv.detach()).mean()

                optimizer.zero_grad()
                loss.backward()

                g_param = student.logits.grad[p_idx]
                g_norm = g_param.norm(dim=-1)
                fork_g = g_norm[fork_indices].mean().item()
                filler_g = g_norm[filler_indices].mean().item()
                ep_fork_grads.append(fork_g)
                ep_filler_grads.append(filler_g)

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
            fork_ratio_history.append(fork_ratio)
            syntax_drift_history.append(drift_val)

            if (ep + 1) % 5 == 0 or ep == epochs - 1:
                print(
                    f"  Epoch {ep+1:02d}/{epochs:02d} | "
                    f"Pass@1: {pass1*100:5.2f}% | "
                    f"Hard Pass: {hard_pass*100:5.2f}% | "
                    f"Fork/Filler Ratio: {fork_ratio:5.2f}x | "
                    f"Syntax Drift: {drift_val:6.4f}"
                )

        all_results[method] = {
            "final_pass1": acc_history[-1],
            "final_hard_pass": hard_acc_history[-1],
            "final_fork_ratio": fork_ratio_history[-1],
            "final_syntax_drift": syntax_drift_history[-1],
            "acc_history": acc_history,
            "hard_acc_history": hard_acc_history,
            "fork_ratio_history": fork_ratio_history,
            "syntax_drift_history": syntax_drift_history,
        }

    out_file = os.path.join(os.path.dirname(__file__), "pure_rl_credit_results.json")
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
    run_experiment()
