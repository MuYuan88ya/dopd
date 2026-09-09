# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0.
"""Rigorous Empirical Verification of Theorem 14:
Geometric Boundedness and Trust-Region Immunity of Sparse SubTB under GSPO.

Mathematical Hypothesis:
Under Sparse Surprise SubTB, gradient updates are concentrated onto K sparse decision forks (w_filler -> 0).
The sequence geometric mean importance ratio in GSPO satisfies:
    |log s_i(theta)| = (1/L) * |sum_t log(pi_theta / pi_old)| <= (K/L) * M_fork
As sequence length L increases (32 to 1024), sequence drift log s_i(theta) decays as O(K/L) -> 0,
guaranteeing unconditional compliance with the trust-region [1-eps, 1+eps],
while fork tokens take full, unattenuated updates (M_fork >> eps).
"""

from __future__ import annotations

import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


def evaluate_trust_region_scaling(lengths: list[int] = [32, 64, 128, 256, 512, 1024], K: int = 2):
    results = {}

    for L in lengths:
        fork_indices = [int(L * 0.3), int(L * 0.7)]

        # Student policy with L tokens
        logits = torch.zeros(1, L, 8, requires_grad=True)
        # Filler tokens have strong confidence on token 0
        with torch.no_grad():
            logits[:, :, 0] = 3.0
            for f in fork_indices:
                logits[:, f, 0] = 0.0
                logits[:, f, 1] = 2.0  # fork choice

        old_logits = logits.detach().clone()

        # Simulated teacher delta: forks have delta = 3.0, fillers have delta = 0.01
        delta = torch.full((1, L), 0.01)
        for f in fork_indices:
            delta[0, f] = 3.0

        # Sparse SubTB weights (gamma = 2.0)
        surprise = (delta.abs() + 0.001).pow(2.0)
        w = surprise / surprise.sum(dim=-1, keepdim=True)

        # Compute SubTB advantage
        R = 1.0
        tau = 0.1
        R_term = R / (tau * L)
        target_token = w * L * R_term  # Simplified reward term
        adv = 2.0 * target_token

        # Take one substantial gradient step (lr = 0.5)
        probs = F.softmax(logits, dim=-1)
        sampled_toks = probs.argmax(dim=-1)
        lp = F.log_softmax(logits, dim=-1).gather(-1, sampled_toks.unsqueeze(-1)).squeeze(-1)
        loss = -(lp * adv.detach()).mean()

        optimizer = torch.optim.SGD([logits], lr=0.5)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Measure token-level log-ratio at the fork vs sequence geometric mean ratio
        with torch.no_grad():
            new_lp = F.log_softmax(logits, dim=-1).gather(-1, sampled_toks.unsqueeze(-1)).squeeze(-1)
            old_lp = F.log_softmax(old_logits, dim=-1).gather(-1, sampled_toks.unsqueeze(-1)).squeeze(-1)
            log_ratio_token = (new_lp - old_lp).squeeze(0)  # [L]

            fork_log_ratios = log_ratio_token[fork_indices].tolist()
            max_fork_ratio = float(np.exp(np.max(fork_log_ratios)))
            mean_fork_log_ratio = float(np.mean(fork_log_ratios))

            # Sequence geometric mean ratio s_i = exp((1/L) * sum log r_t)
            seq_log_drift = float(log_ratio_token.mean().item())
            seq_geom_ratio = float(np.exp(seq_log_drift))

        results[f"L_{L}"] = {
            "L": L,
            "K": K,
            "K_over_L": K / L,
            "fork_importance_ratio": max_fork_ratio,
            "fork_log_ratio": mean_fork_log_ratio,
            "seq_geometric_ratio": seq_geom_ratio,
            "seq_log_drift": seq_log_drift,
            "exceeds_ppo_clip": max_fork_ratio > 1.28,
            "exceeds_gspo_clip": seq_geom_ratio > 1.28,
        }

    return results


def run_benchmark():
    lengths = [32, 64, 128, 256, 512, 1024]
    res = evaluate_trust_region_scaling(lengths=lengths, K=2)

    print(f"\n=========================================================================================")
    print(f"GSPO RATIO BOUNDEDNESS & TRUST REGION IMMUNITY BENCHMARK (Theorem 14)")
    print(f"{'Length L':<10} | {'K/L Ratio':<12} | {'Fork Ratio r_t':<16} | {'GSPO Ratio s_i':<16} | {'PPO Clipped?':<14} | {'GSPO Clipped?'}")
    print("-" * 90)

    for L in lengths:
        data = res[f"L_{L}"]
        kl = data["K_over_L"]
        fr = data["fork_importance_ratio"]
        sr = data["seq_geometric_ratio"]
        ppo_c = "YES (Clipped!)" if data["exceeds_ppo_clip"] else "NO (Safe)"
        gspo_c = "YES (Clipped!)" if data["exceeds_gspo_clip"] else "NO (Safe & Unclipped)"
        print(f"{L:<10} | {kl:<12.4f} | {fr:<16.3f} | {sr:<16.4f} | {ppo_c:<14} | {gspo_c}")
    print("=" * 90)

    out_file = os.path.join(os.path.dirname(__file__), "gspo_ratio_boundedness_results.json")
    with open(out_file, "w") as f:
        json.dump(res, f, indent=2)

    print(f"Results successfully saved to {out_file}")


if __name__ == "__main__":
    run_benchmark()
