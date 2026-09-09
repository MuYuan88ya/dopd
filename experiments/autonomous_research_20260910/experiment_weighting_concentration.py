# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0.
"""Experiment: Concentration Exponent Sensitivity in EW-SubTB.

Investigates how the concentration power gamma in:
    w_t proportional to (|delta_t| + eps)^gamma
affects:
1. Pass@1 accuracy
2. Hard Problem recovery rate
3. Filler token stability (drift on non-decision tokens)
4. Gradient Signal-to-Noise Ratio (SNR)
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


def compute_gamma_advantages(
    gamma: float,
    scores: torch.Tensor,
    sampled_logp: torch.Tensor,
    ref_logp: torch.Tensor,
    teacher_logp: torch.Tensor,
    alpha: float = 0.5,
    tau: float = 0.1,
    subtb_lambda: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    G, L = sampled_logp.shape
    device = sampled_logp.device

    delta = (teacher_logp - ref_logp).clamp(-4.0, 4.0)
    G_T_seq = delta.mean(dim=-1)

    # Pairwise AUC Gating
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

    seq_old = sampled_logp.mean(dim=-1, keepdim=True)
    seq_ref = ref_logp.mean(dim=-1, keepdim=True)
    R_term_seq = (scores / tau).unsqueeze(-1) / L
    target_uncentered = seq_ref + (alpha * g_consist) * G_T_seq.unsqueeze(-1) + R_term_seq
    baseline = (seq_old - target_uncentered).mean()

    # Weight distribution
    if abs(gamma) < 1e-6:
        w = torch.ones(G, L, device=device) / L
    else:
        surprise = (delta.abs() + 0.05).pow(gamma)
        w = surprise / surprise.sum(dim=-1, keepdim=True)

    target_token = ref_logp + (alpha * g_consist) * delta + w * L * (R_term_seq + baseline)
    A_DB = 2.0 * (target_token - sampled_logp)
    A_TB = 2.0 * (target_uncentered + baseline - seq_old).expand(G, L)

    A_SubTB = (1.0 - subtb_lambda) * A_DB + subtb_lambda * A_TB
    return A_SubTB, w


def run_gamma_sweep():
    gammas = [0.0, 0.5, 1.0, 1.5, 2.0]
    num_problems = 30
    num_steps = 4
    tokens_per_step = 8
    vocab_size = 12
    env = ReasoningDAGEnv(num_problems=num_problems, num_steps=num_steps, tokens_per_step=tokens_per_step, vocab_size=vocab_size)

    G = 8
    epochs = 20
    lr = 0.2
    results = {}

    for gamma in gammas:
        print(f"\n--- Testing Gamma = {gamma} ---")
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
        filler_drift_curve = []
        snr_curve = []

        for ep in range(epochs):
            total_corr = 0
            hard_corr = 0
            total_hard = 0
            epoch_grads = []
            filler_drifts = []

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

                adv, w = compute_gamma_advantages(
                    gamma=gamma,
                    scores=scores,
                    sampled_logp=sampled_logp,
                    ref_logp=ref_lp,
                    teacher_logp=teacher_lp,
                )

                # Measure filler token drift: KL divergence from reference on filler tokens
                curr_logits = student.logits[p_idx]
                ref_logits_p = ref_logits[p_idx]
                fork_indices = [s * tokens_per_step + 2 for s in range(num_steps)]
                filler_indices = [idx for idx in range(env.seq_len) if idx not in fork_indices]
                filler_kl = F.kl_div(
                    F.log_softmax(curr_logits[filler_indices], dim=-1),
                    F.softmax(ref_logits_p[filler_indices], dim=-1),
                    reduction="batchmean",
                ).item()
                filler_drifts.append(filler_kl)

                optimizer.zero_grad()
                curr_lp_all = F.log_softmax(student.logits[p_idx], dim=-1).unsqueeze(0).expand(G, -1, -1)
                curr_lp = curr_lp_all.gather(-1, tok_tensor.unsqueeze(-1)).squeeze(-1)
                loss = -(curr_lp * adv.detach()).mean()
                loss.backward()

                epoch_grads.append(student.logits.grad[p_idx].detach().clone())
                optimizer.step()

            pass1 = total_corr / (num_problems * G)
            hard_pass1 = hard_corr / max(total_hard, 1)
            filler_drift = float(np.mean(filler_drifts))

            stacked_grads = torch.stack(epoch_grads)
            g_mean = stacked_grads.mean(dim=0).abs().mean().item()
            g_std = stacked_grads.std(dim=0).mean().item()
            snr = g_mean / (g_std + 1e-6)

            acc_curve.append(pass1)
            hard_acc_curve.append(hard_pass1)
            filler_drift_curve.append(filler_drift)
            snr_curve.append(snr)

            if (ep + 1) % 5 == 0 or ep == epochs - 1:
                print(f"  Epoch {ep+1:02d}: Pass@1 = {pass1*100:.1f}%, Hard Pass@1 = {hard_pass1*100:.1f}%, Filler KL = {filler_drift:.4f}, SNR = {snr:.3f}")

        results[f"gamma_{gamma}"] = {
            "gamma": gamma,
            "final_pass1": acc_curve[-1],
            "final_hard_pass1": hard_acc_curve[-1],
            "final_filler_drift": filler_drift_curve[-1],
            "avg_snr": float(np.mean(snr_curve)),
            "acc_curve": acc_curve,
            "hard_acc_curve": hard_acc_curve,
        }

    out_file = "experiments/autonomous_research_20260910/gamma_sweep_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 88)
    print(" CONCENTRATION EXPONENT GAMMA SWEEP: BENCHMARK RESULTS")
    print("=" * 88)
    print(f"{'Gamma':<12} | {'Pass@1 (%)':<14} | {'Hard Pass@1 (%)':<18} | {'Filler KL (Drift)':<18} | {'Avg SNR':<10}")
    print("-" * 88)
    for k, r in results.items():
        print(f"{r['gamma']:<12.1f} | {r['final_pass1']*100:<14.2f} | {r['final_hard_pass1']*100:<18.2f} | {r['final_filler_drift']:<18.4f} | {r['avg_snr']:<10.3f}")
    print("=" * 88)


if __name__ == "__main__":
    run_gamma_sweep()
