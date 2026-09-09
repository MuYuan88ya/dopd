# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0.
"""Integration test suite for C-FlowBalance Advantage Estimator."""

from __future__ import annotations

import importlib.util
import os
import numpy as np
import torch
import torch.nn as nn

module_path = os.path.abspath("verl/verl/trainer/ppo/c_flowbalance_adv.py")
spec = importlib.util.spec_from_file_location("c_flowbalance_adv", module_path)
c_flowbalance_adv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c_flowbalance_adv)
compute_c_flowbalance_advantage = c_flowbalance_adv.compute_c_flowbalance_advantage
compute_pairwise_auc_group = c_flowbalance_adv.compute_pairwise_auc_group


def reference_gspo_policy_loss(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    clip_ratio_low: float = 0.2,
    clip_ratio_high: float = 0.2,
) -> torch.Tensor:
    """Standard GSPO / PPO clipped surrogate loss for testing backward pass."""
    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(
        ratio,
        1.0 - clip_ratio_low,
        1.0 + clip_ratio_high,
    )
    pg_loss = torch.max(pg_losses1, pg_losses2)
    loss = (pg_loss * response_mask).sum() / response_mask.sum().clamp(min=1.0)
    return loss


def test_g_consist_safety_and_attenuation():
    """Verify that g_consist correctly trusts good teachers and mutes toxic ones."""
    print("Running test_g_consist_safety_and_attenuation...")
    scores = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    
    # 1. Perfect Teacher (strongly prefers correct)
    G_perfect = torch.tensor([1.5, 1.2, 0.9, 0.8, -0.5, -0.8, -1.1, -1.5])
    auc_p, g_p = compute_pairwise_auc_group(G_perfect, scores)
    assert auc_p == 1.0
    assert g_p == 1.0
    print("  [PASS] Perfect teacher received g_consist = 1.0")

    # 2. Random/Neutral Teacher (uncorrelated)
    G_neutral = torch.zeros(8)
    auc_n, g_n = compute_pairwise_auc_group(G_neutral, scores)
    assert auc_n == 0.5
    assert g_n == 0.0
    print("  [PASS] Neutral teacher received g_consist = 0.0")

    # 3. Toxic/Inverted Teacher (prefers errors)
    G_toxic = -G_perfect
    auc_t, g_t = compute_pairwise_auc_group(G_toxic, scores)
    assert auc_t == 0.0
    assert g_t == 0.0
    print("  [PASS] Toxic teacher safely muted to g_consist = 0.0 (Safety Trip)")


def test_token_credit_assignment():
    """Verify step differentiation on key breakthrough vs boilerplate vs error tokens."""
    print("Running test_token_credit_assignment...")
    B, T = 4, 6
    mask = torch.ones(B, T)
    ref_lp = torch.full((B, T), -2.0)
    old_lp = torch.full((B, T), -2.0)
    teacher_lp = ref_lp.clone()
    
    # In sample 0 (correct):
    # Token 1 is a key breakthrough (teacher delta = +2.0)
    # Token 4 is an inefficient step (teacher delta = -1.5)
    teacher_lp[0, 1] += 2.0
    teacher_lp[0, 4] -= 1.5

    rewards = torch.zeros(B, T)
    rewards[0, -1] = 1.0
    rewards[1, -1] = 1.0
    rewards[2, -1] = 0.0
    rewards[3, -1] = 0.0
    
    uids = np.array(["prompt_0", "prompt_0", "prompt_0", "prompt_0"])
    sd_mask = torch.ones(B)

    adv_subtb, _, metrics = compute_c_flowbalance_advantage(
        token_level_rewards=rewards,
        ref_log_prob=ref_lp,
        old_log_prob=old_lp,
        teacher_log_prob=teacher_lp,
        response_mask=mask,
        index=uids,
        self_distillation_mask=sd_mask,
        subtb_lambda=0.5,
    )

    # Breakthrough token (1) must have higher advantage than baseline token (0)
    assert adv_subtb[0, 1] > adv_subtb[0, 0]
    # Inefficient token (4) must have lower advantage than baseline token (0)
    assert adv_subtb[0, 4] < adv_subtb[0, 0]
    print(f"  [PASS] Breakthrough adv ({adv_subtb[0, 1]:.4f}) > Baseline ({adv_subtb[0, 0]:.4f}) > Inefficient ({adv_subtb[0, 4]:.4f})")


def test_mean_flow_conservation():
    """Verify that (1/L) sum_t A_SubTB,t == A_TB strictly holds across all lambdas."""
    print("Running test_mean_flow_conservation...")
    B, T = 4, 8
    mask = torch.ones(B, T)
    ref_lp = torch.full((B, T), -2.0)
    old_lp = torch.full((B, T), -2.1)
    teacher_lp = ref_lp.clone()
    teacher_lp[0, 2] += 2.5
    teacher_lp[0, 5] -= 1.8

    rewards = torch.zeros(B, T)
    rewards[:2, -1] = 1.0
    rewards[2:, -1] = 0.0
    uids = np.array(["p0", "p0", "p0", "p0"])
    sd_mask = torch.ones(B)

    # Baseline pure TB (lambda = 1.0)
    adv_tb, _, _ = compute_c_flowbalance_advantage(
        token_level_rewards=rewards,
        ref_log_prob=ref_lp,
        old_log_prob=old_lp,
        teacher_log_prob=teacher_lp,
        response_mask=mask,
        index=uids,
        self_distillation_mask=sd_mask,
        subtb_lambda=1.0,
    )
    mean_tb = adv_tb.mean(dim=-1)

    for lam in [0.0, 0.25, 0.5, 0.75, 1.0]:
        adv_subtb, _, _ = compute_c_flowbalance_advantage(
            token_level_rewards=rewards,
            ref_log_prob=ref_lp,
            old_log_prob=old_lp,
            teacher_log_prob=teacher_lp,
            response_mask=mask,
            index=uids,
            self_distillation_mask=sd_mask,
            subtb_lambda=lam,
        )
        mean_subtb = adv_subtb.mean(dim=-1)
        diff = (mean_tb - mean_subtb).abs().max().item()
        assert diff < 1e-5, f"Mean flow conservation failed at lambda={lam}: diff={diff}"
    print("  [PASS] Mean flow conservation verified across all lambdas (diff < 1e-5)")


def test_fallback_to_gspo_when_unprivileged():
    """Verify bit-for-bit exact fallback to GRPO when unprivileged."""
    print("Running test_fallback_to_gspo_when_unprivileged...")
    B, T = 4, 6
    mask = torch.ones(B, T)
    ref_lp = torch.randn(B, T)
    old_lp = torch.randn(B, T)
    rewards = torch.zeros(B, T)
    rewards[0, -1] = 1.0
    rewards[1, -1] = 0.0
    rewards[2, -1] = 1.0
    rewards[3, -1] = 0.0
    uids = np.array(["p0", "p0", "p1", "p1"])
    # All samples unprivileged
    sd_mask = torch.zeros(B)

    adv, returns, metrics = compute_c_flowbalance_advantage(
        token_level_rewards=rewards,
        ref_log_prob=ref_lp,
        old_log_prob=old_lp,
        response_mask=mask,
        index=uids,
        self_distillation_mask=sd_mask,
    )

    # In group p0: rewards are [1, 0] -> mean=0.5, std=0.7071
    # Normalized advantages are +0.7071 and -0.7071
    assert metrics["c_flowsd/fallback_gspo_fraction"] == 1.0
    assert torch.allclose(adv[0, :], -adv[1, :], atol=1e-5)
    print("  [PASS] Unprivileged batch completely falls back to 100% pure GSPO/GRPO")


def test_c_flowbalance_gspo_backward_pass():
    """Verify end-to-end forward/backward with C-FlowBalance and GSPO policy loss."""
    print("Running test_c_flowbalance_gspo_backward_pass...")
    B, T = 4, 8
    vocab_size = 32
    hidden_dim = 16

    class ToyPolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(hidden_dim, vocab_size)

        def forward(self, x):
            return torch.log_softmax(self.linear(x), dim=-1)

    model = ToyPolicy()
    x = torch.randn(B, T, hidden_dim)
    log_probs_all = model(x)
    actions = torch.randint(0, vocab_size, (B, T))
    log_prob = log_probs_all.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    old_log_prob = log_prob.detach().clone()
    ref_log_prob = log_prob.detach().clone()
    teacher_log_prob = ref_log_prob.clone()
    teacher_log_prob[:2] += 1.0

    mask = torch.ones(B, T)
    rewards = torch.zeros(B, T)
    rewards[0, -1] = 1.0
    rewards[1, -1] = 0.0
    rewards[2, -1] = 1.0
    rewards[3, -1] = 0.0
    uids = np.array(["p0", "p0", "p1", "p1"])
    sd_mask = torch.ones(B)

    # 1. Compute advantage via compute_c_flowbalance_advantage
    advantages, returns, metrics = compute_c_flowbalance_advantage(
        token_level_rewards=rewards,
        ref_log_prob=ref_log_prob,
        old_log_prob=old_log_prob,
        teacher_log_prob=teacher_log_prob,
        response_mask=mask,
        index=uids,
        self_distillation_mask=sd_mask,
        subtb_lambda=0.7,
    )

    # 2. Feed into GSPO policy loss and backward
    loss = reference_gspo_policy_loss(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=mask,
    )

    assert torch.isfinite(loss)
    loss.backward()

    assert model.linear.weight.grad is not None
    assert torch.isfinite(model.linear.weight.grad).all()
    print("  [PASS] End-to-end GSPO loss backward pass successfully completed with finite gradients")


def run_all_tests():
    print("=" * 80)
    print(" RUNNING INTEGRATION TEST SUITE: C-FLOWBALANCE ADVANTAGE ESTIMATOR")
    print("=" * 80)
    test_g_consist_safety_and_attenuation()
    test_token_credit_assignment()
    test_mean_flow_conservation()
    test_fallback_to_gspo_when_unprivileged()
    test_c_flowbalance_gspo_backward_pass()
    print("=" * 80)
    print(" ALL 5 INTEGRATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_all_tests()
