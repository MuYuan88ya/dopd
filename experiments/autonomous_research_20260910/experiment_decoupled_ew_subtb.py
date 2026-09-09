# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0.
"""Experiment: Decoupled Entropy-Weighted SubTB (DEW-SubTB).

Testing:
- Decoupling terminal reward credit (weighted by w_t) from baseline b_group (uniform across tokens).
- Compares:
  1. FlowBalance (TB)
  2. C-FlowBalance (SubTB Uniform)
  3. Fully-Coupled EW-SubTB
  4. Decoupled EW-SubTB (DEW-SubTB)
  5. Step-Decoupled SubTB (Step-DEW-SubTB)
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


def compute_decoupled_advantages(
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

    # Trajectory Balance advantage
    A_TB = 2.0 * (target_uncentered + baseline - seq_old).expand(G, L)
    if method == "FlowBalance_TB":
        return A_TB

    # Weights
    if method == "SubTB_Uniform":
        w = torch.ones(G, L, device=device) / L
        # Standard uniform target
        target_token = ref_logp + (alpha * g_consist) * delta + (R_term_seq + baseline)
    elif method == "Coupled_EW_SubTB":
        surprise = delta.abs() + 0.1
        w = surprise / surprise.sum(dim=-1, keepdim=True)
        # Coupled: both return and baseline scaled by w * L
        target_token = ref_logp + (alpha * g_consist) * delta + w * L * (R_term_seq + baseline)
    elif method == "Decoupled_EW_SubTB":
        surprise = delta.abs() + 0.1
        w = surprise / surprise.sum(dim=-1, keepdim=True)
        # Decoupled: only return scaled by w * L, baseline remains uniform Z-factor
        target_token = ref_logp + (alpha * g_consist) * delta + (w * L * R_term_seq) + baseline
    elif method == "Step_DEW_SubTB":
        # Step-level weights
        w = torch.zeros(G, L, device=device)
        for s_m in step_masks:
            w += s_m.unsqueeze(0) / (len(step_masks) * s_m.sum().clamp(min=1.0))
        target_token = ref_logp + (alpha * g_consist) * delta + (w * L * R_term_seq) + baseline
    else:
        raise ValueError(f"Unknown method {method}")

    A_DB = 2.0 * (target_token - sampled_logp)
    A_SubTB = (1.0 - subtb_lambda) * A_DB + subtb_lambda * A_TB
    return A_SubTB


def run_experiment():
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

    methods = [
        "FlowBalance_TB",
        "SubTB_Uniform",
        "Coupled_EW_SubTB",
        "Decoupled_EW_SubTB",
        "Step_DEW_SubTB",
    ]
    results = {}
    G = 8
    epochs = 20
    lr = 0.2

    for method in methods:
        print(f"\nEvaluating: {method} ...")
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

                adv = compute_decoupled_advantages(
                    method=method,
                    scores=scores,
                    sampled_logp=sampled_logp,
                    ref_logp=ref_lp,
                    teacher_logp=teacher_lp,
                    step_masks=step_masks,
                )

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
                print(f"  Epoch {ep+1:02d}: Pass@1 = {pass1*100:.1f}%, Hard Pass@1 = {hard_pass1*100:.1f}%, Credit Prec = {credit_prec*100:.1f}%, SNR = {snr:.3f}")

        results[method] = {
            "final_pass1": acc_curve[-1],
            "final_hard_pass1": hard_acc_curve[-1],
            "credit_precision": credit_precision_curve[-1],
            "avg_snr": float(np.mean(snr_curve)),
            "acc_curve": acc_curve,
            "hard_acc_curve": hard_acc_curve,
        }

    out_file = "experiments/autonomous_research_20260910/decoupled_ew_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 90)
    print(" DECOUPLED ENTROPY-WEIGHTED SUBTB: BENCHMARK RESULTS")
    print("=" * 90)
    print(f"{'Method':<22} | {'Pass@1 (%)':<12} | {'Hard Pass@1 (%)':<16} | {'Credit Precision (%)':<22} | {'Avg SNR':<10}")
    print("-" * 90)
    for m in methods:
        r = results[m]
        print(f"{m:<22} | {r['final_pass1']*100:<12.2f} | {r['final_hard_pass1']*100:<16.2f} | {r['credit_precision']*100:<22.2f} | {r['avg_snr']:<10.3f}")
    print("=" * 90)


if __name__ == "__main__":
    run_experiment()
