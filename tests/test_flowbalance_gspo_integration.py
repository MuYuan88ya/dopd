# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0.
"""Unit and integration test for FlowBalance Advantage Estimator with GSPO fallback."""

import importlib.util
import os
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn

# Dynamically import compute_flowbalance_advantage to avoid requiring full Ray cluster dependencies
module_path = os.path.abspath("verl/verl/trainer/ppo/flowbalance_adv.py")
spec = importlib.util.spec_from_file_location("flowbalance_adv", module_path)
flowbalance_adv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(flowbalance_adv)
compute_flowbalance_advantage = flowbalance_adv.compute_flowbalance_advantage


def reference_grpo_outcome_advantage(token_level_rewards, response_mask, uids, epsilon=1e-6):
    """Reference implementation of GRPO outcome advantage (VeRL standard)."""
    scores = token_level_rewards.sum(dim=-1)
    id2score = defaultdict(list)
    bsz = scores.shape[0]
    for i in range(bsz):
        id2score[uids[i]].append(scores[i])

    grpo_adv = torch.zeros_like(scores)
    for i in range(bsz):
        group_scores = torch.stack(id2score[uids[i]])
        mean_score = group_scores.mean()
        if len(group_scores) > 1:
            std_score = group_scores.std()
            grpo_adv[i] = (scores[i] - mean_score) / (std_score + epsilon)
        else:
            grpo_adv[i] = scores[i] - mean_score
    return grpo_adv.unsqueeze(-1) * response_mask


def reference_gspo_policy_loss(old_log_prob, log_prob, advantages, response_mask, clip_ratio_low=0.2, clip_ratio_high=0.2):
    """Reference implementation of GSPO policy loss (VeRL standard compute_policy_loss_gspo)."""
    negative_approx_kl = log_prob - old_log_prob
    seq_lengths = torch.sum(response_mask, dim=-1).clamp(min=1)
    negative_approx_kl_seq = torch.sum(negative_approx_kl * response_mask, dim=-1) / seq_lengths

    log_seq_importance_ratio = log_prob - log_prob.detach() + negative_approx_kl_seq.detach().unsqueeze(-1)
    log_seq_importance_ratio = torch.clamp(log_seq_importance_ratio, max=10.0)
    seq_importance_ratio = torch.exp(log_seq_importance_ratio)

    pg_losses1 = -advantages * seq_importance_ratio
    pg_losses2 = -advantages * torch.clamp(seq_importance_ratio, 1.0 - clip_ratio_low, 1.0 + clip_ratio_high)
    pg_losses = torch.maximum(pg_losses1, pg_losses2)

    # seq-mean-token-mean
    loss = (pg_losses * response_mask).sum(dim=-1) / seq_lengths
    return loss.mean()


def test_privileged_flowbalance_branch():
    """Verify FlowBalance advantage computation when privileged demonstrations exist."""
    print("Running test_privileged_flowbalance_branch...")
    torch.manual_seed(42)
    B, T = 4, 8
    mask = torch.ones(B, T)
    mask[0, 6:] = 0  # length 6
    mask[1, 7:] = 0  # length 7
    mask[2, :] = 1   # length 8
    mask[3, :] = 1   # length 8

    ref_lp = torch.full((B, T), -2.0)
    old_lp = torch.full((B, T), -2.1)
    teacher_lp = torch.full((B, T), -1.6)
    rewards = torch.zeros(B, T)
    # Group 1 (indices 0, 1): one correct, one wrong
    rewards[0, -1] = 1.0
    rewards[1, -1] = 0.0
    # Group 2 (indices 2, 3): one correct, one wrong
    rewards[2, -1] = 1.0
    rewards[3, -1] = 0.0

    uids = np.array(["prompt_0", "prompt_0", "prompt_1", "prompt_1"])
    sd_mask = torch.ones(B)  # All have valid privileged demonstrations

    adv, ret, metrics = compute_flowbalance_advantage(
        token_level_rewards=rewards,
        ref_log_prob=ref_lp,
        old_log_prob=old_lp,
        teacher_log_prob=teacher_lp,
        response_mask=mask,
        index=uids,
        self_distillation_mask=sd_mask,
        beta_q=1.0,
        eta_R=15.0,
        rho=1.0,
        clip_B=4.0,
    )

    assert adv.shape == (B, T)
    assert ret.shape == (B, T)
    assert metrics["flowsd/tb_advantage_fraction"] == 1.0
    assert metrics["flowsd/fallback_gspo_fraction"] == 0.0
    # Check padding has zero advantage
    assert (adv[0, 6:] == 0.0).all()
    assert (adv[1, 7:] == 0.0).all()
    # Correct solutions should have positive advantage
    assert (adv[0, :6] > 0.0).all()
    assert (adv[2, :] > 0.0).all()
    # Incorrect solutions should have negative advantage
    assert (adv[1, :7] < 0.0).all()
    assert (adv[3, :] < 0.0).all()
    print("  [PASS] test_privileged_flowbalance_branch passed")


def test_fallback_to_gspo():
    """Verify that when no privileged demonstration exists, advantage falls back to exact GRPO."""
    print("Running test_fallback_to_gspo...")
    torch.manual_seed(42)
    B, T = 4, 8
    mask = torch.ones(B, T)
    ref_lp = torch.full((B, T), -2.0)
    old_lp = torch.full((B, T), -2.1)
    teacher_lp = torch.full((B, T), -1.6)

    rewards = torch.zeros(B, T)
    rewards[0, -1] = 1.0
    rewards[1, -1] = 0.0
    rewards[2, -1] = 1.0
    rewards[3, -1] = 0.0
    uids = np.array(["prompt_0", "prompt_0", "prompt_1", "prompt_1"])

    # CASE A: self_distillation_mask is all zeros (no demonstration available)
    sd_mask_zero = torch.zeros(B)
    adv_fb, _, metrics_fb = compute_flowbalance_advantage(
        token_level_rewards=rewards,
        ref_log_prob=ref_lp,
        old_log_prob=old_lp,
        teacher_log_prob=teacher_lp,
        response_mask=mask,
        index=uids,
        self_distillation_mask=sd_mask_zero,
        gate_no_context="fallback_gspo",
    )

    # Standard GRPO advantage
    adv_grpo = reference_grpo_outcome_advantage(
        token_level_rewards=rewards,
        response_mask=mask,
        uids=uids,
    )

    # They must match bit-for-bit!
    max_diff = (adv_fb - adv_grpo).abs().max().item()
    assert max_diff < 1e-6, f"Fallback advantage differs from GRPO: max_diff={max_diff}"
    assert metrics_fb["flowsd/fallback_gspo_fraction"] == 1.0
    assert metrics_fb["flowsd/tb_advantage_fraction"] == 0.0

    # CASE B: teacher_log_prob is None -> should also cleanly fall back to GRPO
    adv_fb_none, _, metrics_none = compute_flowbalance_advantage(
        token_level_rewards=rewards,
        ref_log_prob=ref_lp,
        old_log_prob=old_lp,
        teacher_log_prob=None,
        response_mask=mask,
        index=uids,
    )
    max_diff_none = (adv_fb_none - adv_grpo).abs().max().item()
    assert max_diff_none < 1e-6
    assert metrics_none["flowsd/fallback_gspo_fraction"] == 1.0

    print("  [PASS] test_fallback_to_gspo passed (100% exact match with GRPO/GSPO baseline)")


def test_mixed_privileged_and_fallback():
    """Verify mixed batch: Group 0 has demonstrations (TB), Group 1 has none (GSPO fallback)."""
    print("Running test_mixed_privileged_and_fallback...")
    torch.manual_seed(42)
    B, T = 4, 6
    mask = torch.ones(B, T)
    ref_lp = torch.full((B, T), -2.0)
    old_lp = torch.full((B, T), -2.1)
    teacher_lp = torch.full((B, T), -1.5)

    rewards = torch.zeros(B, T)
    rewards[0, -1] = 1.0
    rewards[1, -1] = 0.0
    rewards[2, -1] = 1.0
    rewards[3, -1] = 0.0
    uids = np.array(["p0", "p0", "p1", "p1"])
    # p0 has demonstration; p1 has none
    sd_mask = torch.tensor([1.0, 1.0, 0.0, 0.0])

    adv, _, metrics = compute_flowbalance_advantage(
        token_level_rewards=rewards,
        ref_log_prob=ref_lp,
        old_log_prob=old_lp,
        teacher_log_prob=teacher_lp,
        response_mask=mask,
        index=uids,
        self_distillation_mask=sd_mask,
    )

    # Check metrics
    assert metrics["flowsd/tb_advantage_fraction"] == 0.5
    assert metrics["flowsd/fallback_gspo_fraction"] == 0.5

    # Group 1 (indices 2, 3) must match GRPO advantage exactly
    adv_grpo = reference_grpo_outcome_advantage(rewards, mask, uids)
    diff_g1 = (adv[2:] - adv_grpo[2:]).abs().max().item()
    assert diff_g1 < 1e-6, f"Group 1 fallback differs from GRPO: diff={diff_g1}"

    print("  [PASS] test_mixed_privileged_and_fallback passed")


def test_end_to_end_gspo_loss_integration():
    """Verify end-to-end integration: FlowBalance Advantage -> GSPO Policy Loss."""
    print("Running test_end_to_end_gspo_loss_integration...")
    torch.manual_seed(42)
    B, T = 4, 10
    vocab_size = 30

    class DummyPolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(16, vocab_size)

        def forward(self, x):
            return torch.log_softmax(self.linear(x), dim=-1)

    model = DummyPolicy()
    x = torch.randn(B, T, 16)
    log_probs_all = model(x)
    dummy_tokens = torch.randint(0, vocab_size, (B, T))
    log_prob = torch.gather(log_probs_all, dim=-1, index=dummy_tokens.unsqueeze(-1)).squeeze(-1)
    old_log_prob = log_prob.detach().clone()
    ref_log_prob = old_log_prob.clone()
    teacher_log_prob = old_log_prob + 0.3  # Teacher likes it slightly more

    mask = torch.ones(B, T)
    mask[:, 8:] = 0.0  # Last 2 tokens are padding

    rewards = torch.zeros(B, T)
    rewards[0, -1] = 1.0
    rewards[1, -1] = 0.0
    rewards[2, -1] = 1.0
    rewards[3, -1] = 0.0
    uids = np.array(["p0", "p0", "p1", "p1"])
    sd_mask = torch.tensor([1.0, 1.0, 0.0, 0.0])  # p0 has privileged context, p1 falls back to GSPO

    # 1. Compute FlowBalance advantage
    advantages, returns, metrics = compute_flowbalance_advantage(
        token_level_rewards=rewards,
        ref_log_prob=ref_log_prob,
        old_log_prob=old_log_prob,
        teacher_log_prob=teacher_log_prob,
        response_mask=mask,
        index=uids,
        self_distillation_mask=sd_mask,
    )

    # 2. Feed into GSPO policy loss
    loss = reference_gspo_policy_loss(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=mask,
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
    )

    assert torch.isfinite(loss)
    loss.backward()

    assert model.linear.weight.grad is not None
    assert torch.isfinite(model.linear.weight.grad).all()
    print("  [PASS] test_end_to_end_gspo_loss_integration passed")


def test_subtb_mean_invariance_and_credit_assignment():
    """Verify that SubTB satisfies mean flow conservation and differentiates token quality."""
    print("Running test_subtb_mean_invariance_and_credit_assignment...")
    torch.manual_seed(42)
    B, T = 2, 6
    mask = torch.ones(B, T)
    ref_lp = torch.full((B, T), -2.0)
    old_lp = torch.full((B, T), -2.1)
    teacher_lp = ref_lp.clone()
    # Teacher gives extra +2.0 to token 1, -2.0 to token 4
    teacher_lp[:, 1] += 2.0
    teacher_lp[:, 4] -= 2.0

    rewards = torch.zeros(B, T)
    rewards[0, -1] = 1.0
    rewards[1, -1] = 0.0
    uids = np.array(["p0", "p0"])
    sd_mask = torch.ones(B)

    # 1. Check Trajectory Balance (lambda=1.0)
    adv_tb, _, metrics_tb = compute_flowbalance_advantage(
        token_level_rewards=rewards,
        ref_log_prob=ref_lp,
        old_log_prob=old_lp,
        teacher_log_prob=teacher_lp,
        response_mask=mask,
        index=uids,
        self_distillation_mask=sd_mask,
        subtb_lambda=1.0,
    )
    # In pure TB, all tokens in sequence have identical advantage
    assert torch.allclose(adv_tb[:, 0], adv_tb[:, 1])

    # 2. Check Detailed Balance (lambda=0.0)
    adv_db, _, metrics_db = compute_flowbalance_advantage(
        token_level_rewards=rewards,
        ref_log_prob=ref_lp,
        old_log_prob=old_lp,
        teacher_log_prob=teacher_lp,
        response_mask=mask,
        index=uids,
        self_distillation_mask=sd_mask,
        subtb_lambda=0.0,
    )
    # For positive-advantage sample 0:
    # Token 1 (teacher favored) must have strictly higher advantage than Token 4 (teacher disliked)
    assert adv_db[0, 1] > adv_db[0, 4]

    # 3. Check SubTB Invariant: for any lambda in [0, 1], mean over tokens is invariant
    for lam in [0.0, 0.25, 0.5, 0.75, 1.0]:
        adv_subtb, _, _ = compute_flowbalance_advantage(
            token_level_rewards=rewards,
            ref_log_prob=ref_lp,
            old_log_prob=old_lp,
            teacher_log_prob=teacher_lp,
            response_mask=mask,
            index=uids,
            self_distillation_mask=sd_mask,
            subtb_lambda=lam,
        )
        mean_tb = adv_tb.mean(dim=-1)
        mean_subtb = adv_subtb.mean(dim=-1)
        diff = (mean_tb - mean_subtb).abs().max().item()
        assert diff < 1e-5, f"SubTB mean invariance failed at lambda={lam}: diff={diff}"

    print("  [PASS] test_subtb_mean_invariance_and_credit_assignment passed")


def test_subtb_gspo_backward_pass():
    """Verify backward pass with SubTB advantage (lambda=0.5)."""
    print("Running test_subtb_gspo_backward_pass...")
    torch.manual_seed(42)
    B, T = 4, 8
    vocab_size = 20

    class DummyPolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(8, vocab_size)

        def forward(self, x):
            return torch.log_softmax(self.linear(x), dim=-1)

    model = DummyPolicy()
    x = torch.randn(B, T, 8)
    log_probs_all = model(x)
    dummy_tokens = torch.randint(0, vocab_size, (B, T))
    log_prob = torch.gather(log_probs_all, dim=-1, index=dummy_tokens.unsqueeze(-1)).squeeze(-1)
    old_log_prob = log_prob.detach().clone()
    ref_log_prob = old_log_prob.clone()
    teacher_log_prob = old_log_prob + 0.5

    mask = torch.ones(B, T)
    mask[:, 6:] = 0.0  # padding

    rewards = torch.zeros(B, T)
    rewards[0, -1] = 1.0
    rewards[1, -1] = 0.0
    rewards[2, -1] = 1.0
    rewards[3, -1] = 0.0
    uids = np.array(["p0", "p0", "p1", "p1"])
    sd_mask = torch.tensor([1.0, 1.0, 1.0, 1.0])

    adv_subtb, _, _ = compute_flowbalance_advantage(
        token_level_rewards=rewards,
        ref_log_prob=ref_log_prob,
        old_log_prob=old_log_prob,
        teacher_log_prob=teacher_log_prob,
        response_mask=mask,
        index=uids,
        self_distillation_mask=sd_mask,
        subtb_lambda=0.5,
    )

    loss = reference_gspo_policy_loss(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=adv_subtb,
        response_mask=mask,
    )

    assert torch.isfinite(loss)
    loss.backward()
    assert model.linear.weight.grad is not None
    assert torch.isfinite(model.linear.weight.grad).all()
    print("  [PASS] test_subtb_gspo_backward_pass passed")


if __name__ == "__main__":
    test_privileged_flowbalance_branch()
    test_fallback_to_gspo()
    test_mixed_privileged_and_fallback()
    test_end_to_end_gspo_loss_integration()
    test_subtb_mean_invariance_and_credit_assignment()
    test_subtb_gspo_backward_pass()
    print("\nALL 6 INTEGRATION TESTS PASSED SUCCESSFULLY!")

