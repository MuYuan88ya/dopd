# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0.
"""Benchmark: Flow-GAE (Geometric Multi-Horizon SubTB) vs 2-Point SubTB.

Compares:
1. 2-Point SubTB (standard linear mixture: (1 - lambda)*DB + lambda*TB)
2. Flow-GAE SubTB (exponential multi-horizon span discounting)
Across lambda in [0.0, 0.3, 0.5, 0.7, 0.9].
"""

from __future__ import annotations

import json
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from simulate_reasoning_dag_rl import ReasoningDAGEnv, StudentPolicy

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


def compute_flow_gae_subtb(
    A_DB: torch.Tensor,       # [G, L]
    A_TB: torch.Tensor,       # [G, L]
    response_mask: torch.Tensor,
    lam: float = 0.5,
) -> torch.Tensor:
    """Compute normalized Flow-GAE advantage via backward recursion."""
    if abs(lam) < 1e-6:
        return A_DB
    if abs(lam - 1.0) < 1e-6:
        return A_TB

    G, L = A_DB.shape
    device = A_DB.device
    dtype = A_DB.dtype

    adv = torch.zeros_like(A_DB)
    running_future = torch.zeros(G, device=device, dtype=dtype)

    for t in reversed(range(L)):
        # Flow-GAE recursion: (1 - lam)*A_DB_t + lam * future
        # Terminal boundary: future at end is A_TB
        running_future = (1.0 - lam) * A_DB[:, t] + lam * (running_future if t < L - 1 else A_TB[:, t])
        running_future = running_future * response_mask[:, t]
        adv[:, t] = running_future

    return adv


def run_comparison():
    lambdas = [0.0, 0.3, 0.5, 0.7, 0.9]
    num_problems = 25
    num_steps = 4
    tokens_per_step = 8
    vocab_size = 12
    env = ReasoningDAGEnv(num_problems=num_problems, num_steps=num_steps, tokens_per_step=tokens_per_step, vocab_size=vocab_size)

    G = 8
    epochs = 15
    lr = 0.2
    results = {}

    for mode in ["2Point_SubTB", "Flow_GAE_SubTB"]:
        for lam in lambdas:
            tag = f"{mode}_lam{lam:.1f}"
            print(f"\nEvaluating: {tag} ...")
            student = StudentPolicy(num_problems, env.seq_len, vocab_size)
            with torch.no_grad():
                for p_idx, prob in enumerate(env.problems):
                    if prob["is_hard"]:
                        for s in range(num_steps):
                            fork_pos = s * tokens_per_step + 2
                            distractor = prob["distractors"][s]
                            student.logits[p_idx, fork_pos, distractor] = 2.5

            ref_logits = student.logits.detach().clone()
            teacher_logits = ref_logits.clone()
            for p_idx, prob in enumerate(env.problems):
                for s in range(num_steps):
                    fork_pos = s * tokens_per_step + 2
                    correct_tok = prob["correct"][s]
                    teacher_logits[p_idx, fork_pos, correct_tok] += 3.5

            optimizer = torch.optim.Adam(student.parameters(), lr=lr)
            acc_curve = []
            hard_acc_curve = []
            snr_curve = []

            for ep in range(epochs):
                total_corr = 0
                hard_corr = 0
                total_hard = 0
                epoch_grads = []

                for p_idx in range(num_problems):
                    is_hard = env.problems[p_idx]["is_hard"]
                    if is_hard:
                        total_hard += G

                    tok_list, lp_list, rew_list = [], [], []
                    for g in range(G):
                        toks, lp = student.sample(p_idx)
                        r, _ = env.evaluate(p_idx, toks)
                        tok_list.append(toks)
                        lp_list.append(lp)
                        rew_list.append(r)
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

                    # Compute base DB and TB
                    delta = (teacher_lp - ref_lp).clamp(-4.0, 4.0)
                    G_T_seq = delta.mean(dim=-1)

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

                    L = env.seq_len
                    mask = torch.ones(G, L)
                    seq_old = sampled_logp.mean(dim=-1, keepdim=True)
                    seq_ref = ref_lp.mean(dim=-1, keepdim=True)
                    R_term_seq = (scores / 0.1).unsqueeze(-1) / L
                    target_uncentered = seq_ref + (0.5 * g_consist) * G_T_seq.unsqueeze(-1) + R_term_seq
                    baseline = (seq_old - target_uncentered).mean()

                    # Surprise weighting gamma=1.5
                    surprise = (delta.abs() + 0.05).pow(1.5)
                    w = surprise / surprise.sum(dim=-1, keepdim=True)

                    target_token = ref_lp + (0.5 * g_consist) * delta + w * L * (R_term_seq + baseline)
                    A_DB = 2.0 * (target_token - sampled_logp)
                    A_TB = 2.0 * (target_uncentered + baseline - seq_old).expand(G, L)

                    if mode == "2Point_SubTB":
                        adv = (1.0 - lam) * A_DB + lam * A_TB
                    else:
                        adv = compute_flow_gae_subtb(A_DB, A_TB, mask, lam=lam)

                    optimizer.zero_grad()
                    curr_lp_all = F.log_softmax(student.logits[p_idx], dim=-1).unsqueeze(0).expand(G, -1, -1)
                    curr_lp = curr_lp_all.gather(-1, tok_tensor.unsqueeze(-1)).squeeze(-1)
                    loss = -(curr_lp * adv.detach()).mean()
                    loss.backward()

                    epoch_grads.append(student.logits.grad[p_idx].detach().clone())
                    optimizer.step()

                pass1 = total_corr / (num_problems * G)
                hard_pass1 = hard_corr / max(total_hard, 1)

                stacked_grads = torch.stack(epoch_grads)
                g_mean = stacked_grads.mean(dim=0).abs().mean().item()
                g_std = stacked_grads.std(dim=0).mean().item()
                snr = g_mean / (g_std + 1e-6)

                acc_curve.append(pass1)
                hard_acc_curve.append(hard_pass1)
                snr_curve.append(snr)

                if (ep + 1) % 5 == 0 or ep == epochs - 1:
                    print(f"  Epoch {ep+1:02d}: Pass@1 = {pass1*100:.1f}%, Hard Pass@1 = {hard_acc_curve[-1]*100:.1f}%, SNR = {snr:.3f}")

            results[tag] = {
                "mode": mode,
                "lambda": lam,
                "final_pass1": acc_curve[-1],
                "final_hard_pass1": hard_acc_curve[-1],
                "avg_snr": float(np.mean(snr_curve)),
                "acc_curve": acc_curve,
                "hard_acc_curve": hard_acc_curve,
            }

    out_file = "experiments/autonomous_research_20260910/flow_gae_benchmark_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 90)
    print(" FLOW-GAE VS 2-POINT SUBTB COMPARATIVE RESULTS")
    print("=" * 90)
    print(f"{'Method / Setting':<28} | {'Pass@1 (%)':<14} | {'Hard Pass@1 (%)':<18} | {'Avg SNR':<10}")
    print("-" * 90)
    for tag, r in results.items():
        print(f"{tag:<28} | {r['final_pass1']*100:<14.2f} | {r['final_hard_pass1']*100:<18.2f} | {r['avg_snr']:<10.3f}")
    print("=" * 90)


if __name__ == "__main__":
    run_comparison()
