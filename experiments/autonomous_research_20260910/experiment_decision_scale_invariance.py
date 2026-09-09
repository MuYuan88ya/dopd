# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0.
"""Rigorous Empirical Verification of Theorem 9:
Decision-Scale Invariance of Sparse Surprise SubTB vs Length Starvation in Uniform Credit Assignment.

Mathematical Hypothesis:
In long-chain reasoning traces where sequence length L varies from 32 to 1024 tokens,
uniform credit assignment (GRPO, Uniform SubTB) dilutes the decision fork advantage
as O(1/L), causing severe 'Length Starvation' on long derivations.

Sparse Surprise-Weighted SubTB (EW-SubTB) concentrates simplex mass w_fork ~ O(1/K),
where K is the number of decision forks. The factor of L in w_t * L cancels 1/L in R/(tau * L),
yielding scale-invariant fork advantages: A_fork ~ O(1/K), independent of L.
"""

from __future__ import annotations

import json
import os
import numpy as np
import torch
import torch.nn.functional as F

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


def evaluate_length_scaling(
    lengths: list[int] = [32, 64, 128, 256, 512, 1024],
    num_forks: int = 2,
    gamma_values: list[float] = [0.0, 1.0, 2.0],  # 0.0 is uniform
    tau: float = 0.1,
    rho: float = 1.0,
):
    results = {}

    for L in lengths:
        results[f"L_{L}"] = {}
        fork_indices = [int(L * 0.25), int(L * 0.75)]

        # Simulate teacher delta: forks have high surprise (delta = 2.5), fillers have low surprise (delta = 0.02)
        delta = torch.full((1, L), 0.02, dtype=torch.float32)
        for f_idx in fork_indices:
            delta[0, f_idx] = 2.5

        R = 1.0
        # Under intensive normalization rho = 1.0:
        R_term = R / (tau * (L ** rho))  # R / (tau * L)

        for gamma in gamma_values:
            mode_name = "Uniform_SubTB" if gamma == 0.0 else f"EW_SubTB_gamma_{gamma}"

            if gamma == 0.0:
                w = torch.ones(1, L) / L
            else:
                surprise = (delta.abs() + 0.001).pow(gamma)
                w = surprise / surprise.sum(dim=-1, keepdim=True)

            # Token reward target contribution: w_t * L * R_term
            reward_target_contribution = w * L * R_term
            fork_adv_contrib = reward_target_contribution[0, fork_indices].mean().item()
            filler_mask = torch.ones(L, dtype=torch.bool)
            for f_idx in fork_indices:
                filler_mask[f_idx] = False
            filler_adv_contrib = reward_target_contribution[0, filler_mask].mean().item()

            results[f"L_{L}"][mode_name] = {
                "L": L,
                "gamma": gamma,
                "fork_adv_contrib": fork_adv_contrib,
                "filler_adv_contrib": filler_adv_contrib,
                "fork_to_filler_ratio": fork_adv_contrib / max(filler_adv_contrib, 1e-12),
            }

    return results


def run_benchmark():
    lengths = [32, 64, 128, 256, 512, 1024]
    gammas = [0.0, 1.0, 2.0]
    res = evaluate_length_scaling(lengths=lengths, gamma_values=gammas)

    print(f"\n=========================================================================================")
    print(f"DECISION-SCALE INVARIANCE BENCHMARK (Theorem 9): Fork vs Filler Update Strength")
    print(f"{'Length L':<10} | {'Method':<22} | {'Fork Signal':<14} | {'Filler Signal':<14} | {'Fork/Filler Ratio'}")
    print("-" * 80)

    for L in lengths:
        for gamma in gammas:
            m_name = "Uniform_SubTB" if gamma == 0.0 else f"EW_SubTB_gamma_{gamma}"
            data = res[f"L_{L}"][m_name]
            f_sig = data["fork_adv_contrib"]
            fl_sig = data["filler_adv_contrib"]
            ratio = data["fork_to_filler_ratio"]
            print(f"{L:<10} | {m_name:<22} | {f_sig:14.4f} | {fl_sig:14.6f} | {ratio:14.1f}x")
        print("-" * 80)

    out_file = os.path.join(os.path.dirname(__file__), "decision_scale_invariance_results.json")
    with open(out_file, "w") as f:
        json.dump(res, f, indent=2)

    print(f"Results successfully saved to {out_file}")


if __name__ == "__main__":
    run_benchmark()
