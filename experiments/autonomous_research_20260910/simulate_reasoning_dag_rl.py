# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0.
"""Rigorous Reasoning DAG Simulation Benchmark.

Evaluates credit assignment and policy convergence on multi-step reasoning trees:
- Decision forks (Correct branch vs distractor trap)
- Filler tokens (low-entropy boilerplate)
- Compares:
  1. Standard GRPO (Outcome-only)
  2. FlowBalance (Trajectory Balance, TB)
  3. C-FlowBalance (SubTB Uniform)
  4. EW-SubTB (Entropy/Surprise-Weighted SubTB)
  5. Step-SubTB (Step-Boundary SubTB)
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
        num_problems: int = 30,
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
            # Step decision tokens: index 2 of each step is the fork
            # Choice between correct_token and distractor_token
            correct = torch.randint(2, vocab_size, (num_steps,))
            distractors = torch.randint(2, vocab_size, (num_steps,))
            # Ensure distractor != correct
            for s in range(num_steps):
                while distractors[s] == correct[s]:
                    distractors[s] = torch.randint(2, vocab_size, (1,)).item()

            # Hard problem: student initial logits favor the distractor (trap!)
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

    def sample(self, p_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.logits[p_idx]
        probs = F.softmax(logits, dim=-1)
        toks = torch.multinomial(probs, num_samples=1).squeeze(-1)
        lp = F.log_softmax(logits, dim=-1).gather(-1, toks.unsqueeze(-1)).squeeze(-1)
        return toks, lp


def compute_advantages(
    method: str,
    scores: torch.Tensor,
    sampled_logp: torch.Tensor,
    ref_logp: torch.Tensor,
    teacher_logp: torch.Tensor,
    step_masks: list[torch.Tensor],
    alpha: float = 0.5,
    tau: float = 0.1,
    subtb_lambda: float = 0.5,
) -> torch.Tensor:
    G, L = sampled_logp.shape
    device = sampled_logp.device

    # 1. Base GRPO
    mean_score = scores.mean()
    std_score = scores.std()
    if std_score > 1e-6:
        grpo_adv = (scores - mean_score) / (std_score + 1e-6)
    else:
        grpo_adv = scores - mean_score
    grpo_token_adv = grpo_adv.unsqueeze(-1).expand(G, L)

    if method == "GRPO":
        return grpo_token_adv

    # 2. Privileged teacher delta
    delta = (teacher_logp - ref_logp).clamp(-4.0, 4.0)
    G_T_seq = delta.mean(dim=-1)

    # 3. Pairwise AUC Gating
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

    # 4. Target & Trajectory Balance
    seq_old = sampled_logp.mean(dim=-1, keepdim=True)
    seq_ref = ref_logp.mean(dim=-1, keepdim=True)
    R_term = (scores / tau).unsqueeze(-1)
    target_uncentered = seq_ref + (alpha * g_consist) * G_T_seq.unsqueeze(-1) + R_term / L
    baseline = (seq_old - target_uncentered).mean()

    A_TB = 2.0 * (target_uncentered + baseline - seq_old).expand(G, L)
    if method == "FlowBalance_TB":
        return A_TB

    # 5. Token weightings for Detailed Balance
    if method == "SubTB_Uniform":
        w = torch.ones(G, L, device=device) / L
    elif method == "EW_SubTB":
        # Surprise/Entropy weighting: weight decision forks where teacher has distinct signal
        surprise = delta.abs() + 0.1
        w = surprise / surprise.sum(dim=-1, keepdim=True)
    elif method == "Step_SubTB":
        # Step-level weighting: uniform mass per step
        w = torch.zeros(G, L, device=device)
        for s_m in step_masks:
            w += s_m.unsqueeze(0) / (len(step_masks) * s_m.sum().clamp(min=1.0))
    else:
        raise ValueError(f"Unknown method: {method}")

    target_token = ref_logp + (alpha * g_consist) * delta + w * L * (R_term / L + baseline)
    A_DB = 2.0 * (target_token - sampled_logp)

    A_SubTB = (1.0 - subtb_lambda) * A_DB + subtb_lambda * A_TB
    return A_SubTB


def run_dag_benchmark():
    num_problems = 30
    num_steps = 4
    tokens_per_step = 8
    vocab_size = 12
    env = ReasoningDAGEnv(num_problems=num_problems, num_steps=num_steps, tokens_per_step=tokens_per_step, vocab_size=vocab_size)

    step_masks = []
    for s in range(num_steps):
        m = torch.zeros(env.seq_len)
        m[s * tokens_per_step : (s + 1) * tokens_per_step] = 1.0
        step_masks.append(m)

    methods = ["GRPO", "FlowBalance_TB", "SubTB_Uniform", "EW_SubTB", "Step_SubTB"]
    all_results = {}
    G = 8
    epochs = 20
    lr = 0.2

    for method in methods:
        print(f"\n--- Testing Method: {method} ---")
        student = StudentPolicy(num_problems, env.seq_len, vocab_size)

        # Initialize student: for hard problems, bias toward distractor (trap!)
        with torch.no_grad():
            for p_idx, prob in enumerate(env.problems):
                if prob["is_hard"]:
                    for s in range(num_steps):
                        fork_pos = s * tokens_per_step + 2
                        distractor = prob["distractors"][s]
                        student.logits[p_idx, fork_pos, distractor] = 2.5  # Student trap

        ref_logits = student.logits.detach().clone()

        # Privileged Teacher: knows the correct decision token
        teacher_logits = ref_logits.clone()
        for p_idx, prob in enumerate(env.problems):
            for s in range(num_steps):
                fork_pos = s * tokens_per_step + 2
                correct_tok = prob["correct"][s]
                teacher_logits[p_idx, fork_pos, correct_tok] += 3.5

        optimizer = torch.optim.Adam(student.parameters(), lr=lr)
        acc_curve = []
        hard_acc_curve = []
        credit_precision_curve = []
        snr_curve = []

        for ep in range(epochs):
            total_corr = 0
            hard_corr = 0
            total_hard = 0
            epoch_grads = []
            credit_hits = []

            for p_idx in range(num_problems):
                is_hard = env.problems[p_idx]["is_hard"]
                if is_hard:
                    total_hard += G

                tok_list, lp_list, rew_list, err_list = [], [], [], []
                for g in range(G):
                    toks, lp = student.sample(p_idx)
                    r, err_s = env.evaluate(p_idx, toks)
                    tok_list.append(toks)
                    lp_list.append(lp)
                    rew_list.append(r)
                    err_list.append(err_s)
                    if r > 0.5:
                        total_corr += 1
                        if is_hard:
                            hard_corr += 1

                scores = torch.tensor(rew_list, dtype=torch.float32)
                sampled_logp = torch.stack(lp_list)
                tok_tensor = torch.stack(tok_list)

                ref_lp_all = F.log_softmax(ref_logits[p_idx], dim=-1).unsqueeze(0).expand(G, -1, -1)
                ref_lp = ref_lp_all.gather(-1, tok_tensor.unsqueeze(-1)).squeeze(-1)

                teacher_lp_all = F.log_softmax(teacher_logits[p_idx], dim=-1).unsqueeze(0).expand(G, -1, -1)
                teacher_lp = teacher_lp_all.gather(-1, tok_tensor.unsqueeze(-1)).squeeze(-1)

                adv = compute_advantages(
                    method=method,
                    scores=scores,
                    sampled_logp=sampled_logp,
                    ref_logp=ref_lp,
                    teacher_logp=teacher_lp,
                    step_masks=step_masks,
                )

                # Check credit attribution on error steps
                for g in range(G):
                    err_s = err_list[g]
                    if err_s != -1:
                        err_pos = err_s * tokens_per_step + 2
                        if adv[g, err_pos].item() < -0.05:
                            credit_hits.append(1.0)
                        else:
                            credit_hits.append(0.0)

                optimizer.zero_grad()
                curr_lp_all = F.log_softmax(student.logits[p_idx], dim=-1).unsqueeze(0).expand(G, -1, -1)
                curr_lp = curr_lp_all.gather(-1, tok_tensor.unsqueeze(-1)).squeeze(-1)
                loss = -(curr_lp * adv.detach()).mean()
                loss.backward()

                epoch_grads.append(student.logits.grad[p_idx].detach().clone())
                optimizer.step()

            pass1 = total_corr / (num_problems * G)
            hard_pass1 = hard_corr / max(total_hard, 1)
            credit_prec = float(np.mean(credit_hits)) if credit_hits else 0.0

            stacked_grads = torch.stack(epoch_grads)
            g_mean = stacked_grads.mean(dim=0).abs().mean().item()
            g_std = stacked_grads.std(dim=0).mean().item()
            snr = g_mean / (g_std + 1e-6)

            acc_curve.append(pass1)
            hard_acc_curve.append(hard_pass1)
            credit_precision_curve.append(credit_prec)
            snr_curve.append(snr)

            if (ep + 1) % 5 == 0 or ep == epochs - 1:
                print(f"  Epoch {ep+1:02d}: Pass@1 = {pass1*100:.1f}%, Hard Pass@1 = {hard_acc_curve[-1]*100:.1f}%, Credit Prec = {credit_prec*100:.1f}%, SNR = {snr:.3f}")

        all_results[method] = {
            "final_pass1": acc_curve[-1],
            "final_hard_pass1": hard_acc_curve[-1],
            "credit_precision": credit_precision_curve[-1],
            "avg_snr": float(np.mean(snr_curve)),
            "acc_curve": acc_curve,
            "hard_acc_curve": hard_acc_curve,
        }

    out_file = "experiments/autonomous_research_20260910/dag_benchmark_results.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 88)
    print(" REASONING DAG BENCHMARK RESULTS: COMPARATIVE ANALYSIS")
    print("=" * 88)
    print(f"{'Method':<20} | {'Pass@1 (%)':<12} | {'Hard Pass@1 (%)':<16} | {'Credit Precision (%)':<22} | {'Avg SNR':<10}")
    print("-" * 88)
    for m in methods:
        r = all_results[m]
        print(f"{m:<20} | {r['final_pass1']*100:<12.2f} | {r['final_hard_pass1']*100:<16.2f} | {r['credit_precision']*100:<22.2f} | {r['avg_snr']:<10.3f}")
    print("=" * 88)


if __name__ == "__main__":
    run_dag_benchmark()
