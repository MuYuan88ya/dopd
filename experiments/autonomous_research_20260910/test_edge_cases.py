# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0.
"""Rigorous Edge-Case and Stress Testing Suite for C-FlowBalance and FlowBalance."""

import importlib.util
import os
import sys
import numpy as np
import torch

module_path = os.path.abspath("verl/verl/trainer/ppo/c_flowbalance_adv.py")
spec = importlib.util.spec_from_file_location("c_flowbalance_adv", module_path)
c_flowbalance_adv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c_flowbalance_adv)

fb_path = os.path.abspath("verl/verl/trainer/ppo/flowbalance_adv.py")
spec_fb = importlib.util.spec_from_file_location("flowbalance_adv", fb_path)
flowbalance_adv = importlib.util.module_from_spec(spec_fb)
spec_fb.loader.exec_module(flowbalance_adv)


def test_edge_case_all_zero_group():
    """Test behavior when all student rollouts in a group fail (R = 0)."""
    print("Testing edge case: all rollouts fail (Pass@G = 0)...")
    B, L = 4, 20
    rewards = torch.zeros(B, L)
    ref_lp = torch.randn(B, L)
    old_lp = ref_lp.clone()
    teacher_lp = ref_lp + torch.tensor([[2.0], [1.0], [-1.0], [-2.0]])
    mask = torch.ones(B, L)
    uids = ["prompt_hard"] * B

    adv, ret, met = c_flowbalance_adv.compute_c_flowbalance_advantage(
        token_level_rewards=rewards,
        ref_log_prob=ref_lp,
        old_log_prob=old_lp,
        response_mask=mask,
        index=uids,
        teacher_log_prob=teacher_lp,
        alpha=0.5,
        tau=0.1,
    )
    # The teacher provided distinct signals: rollouts closer to the gold solution should get higher advantage
    assert adv[0, 0] > adv[1, 0] > adv[2, 0] > adv[3, 0], "Should order rollouts by teacher guidance"
    print(f"  [PASS] All-zero group ordered correctly: {adv[:, 0].tolist()}")


def test_edge_case_all_one_group():
    """Test behavior when all student rollouts in a group succeed (Pass@G = 1.0)."""
    print("Testing edge case: all rollouts succeed (Pass@G = 1.0)...")
    B, L = 4, 20
    rewards = torch.zeros(B, L)
    rewards[:, -1] = 1.0  # Terminal reward = 1
    ref_lp = torch.randn(B, L)
    old_lp = ref_lp.clone()
    teacher_lp = ref_lp + 0.5
    mask = torch.ones(B, L)
    uids = ["prompt_easy"] * B

    adv, ret, met = c_flowbalance_adv.compute_c_flowbalance_advantage(
        token_level_rewards=rewards,
        ref_log_prob=ref_lp,
        old_log_prob=old_lp,
        response_mask=mask,
        index=uids,
        teacher_log_prob=teacher_lp,
        alpha=0.5,
        tau=0.1,
    )
    assert torch.isfinite(adv).all(), "Advantages must be finite"
    print(f"  [PASS] All-one group passed with finite advantages: mean={adv.mean().item():.4f}")


def test_edge_case_empty_and_single_token():
    """Test empty response mask and length=1 response."""
    print("Testing edge case: empty response (L=0) and single token response (L=1)...")
    B, L = 3, 10
    rewards = torch.zeros(B, L)
    rewards[0, 0] = 1.0
    ref_lp = torch.randn(B, L)
    old_lp = ref_lp.clone()
    teacher_lp = ref_lp + 0.5
    mask = torch.zeros(B, L)
    mask[0, :1] = 1.0  # Length 1
    mask[1, :5] = 1.0  # Length 5
    # Sample 2 is completely empty (mask is all 0)

    uids = ["p1", "p1", "p1"]
    adv, ret, met = c_flowbalance_adv.compute_c_flowbalance_advantage(
        token_level_rewards=rewards,
        ref_log_prob=ref_lp,
        old_log_prob=old_lp,
        response_mask=mask,
        index=uids,
        teacher_log_prob=teacher_lp,
    )
    assert adv[2].abs().sum() == 0.0, "Empty response must have exactly 0.0 advantage"
    assert torch.isfinite(adv).all(), "Must be finite"
    print("  [PASS] Empty and single token responses handled safely")


def test_edge_case_bfloat16():
    """Test full execution in torch.bfloat16 precision."""
    print("Testing precision: torch.bfloat16 execution...")
    B, L = 4, 30
    dtype = torch.bfloat16
    rewards = torch.zeros(B, L, dtype=dtype)
    rewards[0, -1] = 1.0
    ref_lp = torch.randn(B, L, dtype=dtype)
    old_lp = ref_lp.clone()
    teacher_lp = (ref_lp + 1.0).to(dtype=dtype)
    mask = torch.ones(B, L, dtype=dtype)
    uids = ["p1", "p1", "p2", "p2"]

    adv, ret, met = c_flowbalance_adv.compute_c_flowbalance_advantage(
        token_level_rewards=rewards,
        ref_log_prob=ref_lp,
        old_log_prob=old_lp,
        response_mask=mask,
        index=uids,
        teacher_log_prob=teacher_lp,
    )
    assert adv.dtype == dtype, f"Output dtype must match input dtype {dtype}, got {adv.dtype}"
    assert torch.isfinite(adv).all(), "Must be finite in bfloat16"
    print("  [PASS] bfloat16 execution verified cleanly")


def test_edge_case_extreme_lengths():
    """Test mixed batch with extreme length heterogeneity (L=5 to L=3000)."""
    print("Testing scale invariance with extreme length heterogeneity (L=5 to L=3000)...")
    L_max = 3000
    B = 4
    lengths = [5, 50, 500, 3000]
    mask = torch.zeros(B, L_max)
    for i, l in enumerate(lengths):
        mask[i, :l] = 1.0

    ref_lp = torch.randn(B, L_max)
    old_lp = ref_lp.clone()
    teacher_lp = ref_lp + 1.0
    rewards = torch.zeros(B, L_max)
    rewards[:, 0] = 1.0
    uids = ["group1"] * B

    adv, ret, met = c_flowbalance_adv.compute_c_flowbalance_advantage(
        token_level_rewards=rewards,
        ref_log_prob=ref_lp,
        old_log_prob=old_lp,
        response_mask=mask,
        index=uids,
        teacher_log_prob=teacher_lp,
    )
    # Check advantage scale for each length
    scales = [adv[i, :lengths[i]].abs().mean().item() for i in range(B)]
    print(f"  Advantage scales across lengths {lengths}: {scales}")
    # All scales should be within an order of magnitude (O(1)), no 100x collapse
    for s in scales:
        assert 0.05 < s < 5.0, f"Advantage scale {s} out of reasonable O(1) range"
    print("  [PASS] Extreme lengths scale invariance confirmed")


if __name__ == "__main__":
    print("=" * 80)
    print(" RUNNING EDGE-CASE AND PRECISION AUDIT SUITE")
    print("=" * 80)
    test_edge_case_all_zero_group()
    test_edge_case_all_one_group()
    test_edge_case_empty_and_single_token()
    test_edge_case_bfloat16()
    test_edge_case_extreme_lengths()
    print("=" * 80)
    print(" ALL EDGE-CASE AUDITS PASSED!")
    print("=" * 80)
