# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0.
"""Comprehensive Test and Validation of Surprise-Weighted SubTB & Mixed Batch Dynamics."""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import importlib.util

module_path = os.path.abspath("verl/verl/trainer/ppo/c_flowbalance_adv.py")
spec = importlib.util.spec_from_file_location("c_flowbalance_adv", module_path)
c_flowbalance_adv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c_flowbalance_adv)


def test_ew_subtb_mean_flow_conservation():
    """Verify that Surprise-Weighted SubTB strictly preserves the Mean Flow Conservation theorem."""
    print("Testing EW-SubTB Mean Flow Conservation across gammas [0.5, 1.0, 1.5, 2.0]...")
    B, L = 6, 40
    rewards = torch.tensor([[1.0]*L, [0.0]*L, [1.0]*L, [0.0]*L, [0.5]*L, [0.0]*L])
    ref_lp = torch.randn(B, L)
    old_lp = ref_lp + torch.randn(B, L) * 0.1
    teacher_lp = ref_lp + torch.randn(B, L) * 1.5
    mask = torch.ones(B, L)
    uids = ["p1", "p1", "p2", "p2", "p3", "p3"]

    for gamma in [0.0, 0.5, 1.0, 1.5, 2.0]:
        for subtb_lambda in [0.0, 0.3, 0.5, 0.7, 1.0]:
            adv, ret, met = c_flowbalance_adv.compute_c_flowbalance_advantage(
                token_level_rewards=rewards,
                ref_log_prob=ref_lp,
                old_log_prob=old_lp,
                response_mask=mask,
                index=uids,
                teacher_log_prob=teacher_lp,
                alpha=0.5,
                tau=0.1,
                subtb_lambda=subtb_lambda,
                token_weight_mode="surprise" if gamma > 0 else "uniform",
                token_weight_gamma=gamma,
            )
            # Compare sequence mean of SubTB advantages against pure TB advantages (lambda=1.0)
            adv_tb, _, _ = c_flowbalance_adv.compute_c_flowbalance_advantage(
                token_level_rewards=rewards,
                ref_log_prob=ref_lp,
                old_log_prob=old_lp,
                response_mask=mask,
                index=uids,
                teacher_log_prob=teacher_lp,
                alpha=0.5,
                tau=0.1,
                subtb_lambda=1.0,
            )
            lengths = mask.sum(dim=-1)
            mean_subtb = (adv * mask).sum(dim=-1) / lengths
            mean_tb = (adv_tb * mask).sum(dim=-1) / lengths
            max_diff = (mean_subtb - mean_tb).abs().max().item()
            assert max_diff < 1e-4, f"Mean flow conservation failed for gamma={gamma}, lambda={subtb_lambda}: max_diff={max_diff}"

    print("  [PASS] Mean flow conservation verified across all gammas and lambdas (max diff < 1e-4)")


def test_mixed_batch_edge_dynamics():
    """Verify that a mixed batch containing good, toxic, tie, and unprivileged prompts runs safely."""
    print("Testing mixed batch edge dynamics (good, toxic, ties, unprivileged)...")
    B, L = 8, 25
    # 4 groups of 2 rollouts:
    # Group 1: Good teacher (AUC = 1.0)
    # Group 2: Toxic teacher (AUC = 0.0)
    # Group 3: All-zero failure group (Pass@G = 0)
    # Group 4: Unprivileged fallback (mask = 0)
    rewards = torch.zeros(B, L)
    rewards[0, -1] = 1.0  # G1 rollout 0 correct, rollout 1 wrong
    rewards[2, -1] = 1.0  # G2 rollout 2 correct, rollout 3 wrong
    # G3: all 0
    # G4: rollout 6 correct, rollout 7 wrong
    rewards[6, -1] = 1.0

    ref_lp = torch.randn(B, L)
    old_lp = ref_lp.clone()
    teacher_lp = ref_lp.clone()
    # G1: teacher prefers rollout 0 (good)
    teacher_lp[0] += 2.0
    teacher_lp[1] -= 2.0
    # G2: teacher prefers rollout 3 (toxic!)
    teacher_lp[2] -= 2.0
    teacher_lp[3] += 2.0
    # G3: teacher gives constructive signal
    teacher_lp[4] += 1.0
    teacher_lp[5] -= 1.0

    mask = torch.ones(B, L)
    sd_mask = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0])  # G4 unprivileged
    uids = ["g1_good", "g1_good", "g2_toxic", "g2_toxic", "g3_tie", "g3_tie", "g4_fallback", "g4_fallback"]

    adv, ret, met = c_flowbalance_adv.compute_c_flowbalance_advantage(
        token_level_rewards=rewards,
        ref_log_prob=ref_lp,
        old_log_prob=old_lp,
        response_mask=mask,
        index=uids,
        teacher_log_prob=teacher_lp,
        self_distillation_mask=sd_mask,
        alpha=0.5,
        tau=0.1,
        subtb_lambda=0.5,
        token_weight_mode="surprise",
        token_weight_gamma=1.5,
    )

    assert torch.isfinite(adv).all(), "All advantages must be finite"
    # G2 (toxic teacher) should be muted, so g_consist = 0.0
    # G4 (fallback) should match GRPO
    print(f"  Diagnostics: {met}")
    print("  [PASS] Mixed batch edge dynamics executed cleanly with zero errors")


if __name__ == "__main__":
    print("=" * 80)
    print(" RUNNING COMPREHENSIVE SUITE FOR SURPRISE-WEIGHTED C-FLOWBALANCE")
    print("=" * 80)
    test_ew_subtb_mean_flow_conservation()
    test_mixed_batch_edge_dynamics()
    print("=" * 80)
    print(" ALL TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 80)
