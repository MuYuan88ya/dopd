# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0.
"""Unit tests for Curriculum EMA AUC tracking across batches in C-FlowBalance."""

from __future__ import annotations

import importlib.util
import os
import torch

module_path = os.path.abspath("verl/verl/trainer/ppo/c_flowbalance_adv.py")
spec = importlib.util.spec_from_file_location("c_flowbalance_adv", module_path)
c_flowbalance_adv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c_flowbalance_adv)
compute_c_flowbalance_advantage = c_flowbalance_adv.compute_c_flowbalance_advantage
compute_pairwise_auc_group = c_flowbalance_adv.compute_pairwise_auc_group


def test_pairwise_auc_ema_tracker():
    """Verify compute_pairwise_auc_group updates running EMA on valid groups and uses it on degenerate groups."""
    tracker = {}

    # 1. Informative group with perfect correlation: scores [1.0, 0.0], G_T [2.0, -1.0]
    scores = torch.tensor([1.0, 0.0])
    G_T = torch.tensor([2.0, -1.0])
    auc, g_consist = compute_pairwise_auc_group(
        G_seq=G_T,
        scores=scores,
        ema_auc_tracker=tracker,
        ema_beta=0.5,
    )
    assert auc == 1.0
    assert g_consist == 1.0
    assert "running_auc" in tracker
    assert tracker["running_auc"] == 1.0

    # 2. Informative group with imperfect correlation: scores [1.0, 0.0], G_T [-1.0, 2.0] -> auc = 0.0
    scores2 = torch.tensor([1.0, 0.0])
    G_T2 = torch.tensor([-1.0, 2.0])
    auc2, g_consist2 = compute_pairwise_auc_group(
        G_seq=G_T2,
        scores=scores2,
        ema_auc_tracker=tracker,
        ema_beta=0.5,
    )
    assert auc2 == 0.0
    assert g_consist2 == 0.0
    # EMA: 0.5 * 1.0 + 0.5 * 0.0 = 0.5
    assert abs(tracker["running_auc"] - 0.5) < 1e-5

    # 3. Degenerate group (all zero rewards): scores [0.0, 0.0]
    # Should recall running EMA (0.9) rather than uninformative default
    tracker["running_auc"] = 0.9  # Set high historical trust
    scores_degen = torch.tensor([0.0, 0.0])
    G_T_degen = torch.tensor([1.5, 0.5])
    auc_degen, g_consist_degen = compute_pairwise_auc_group(
        G_seq=G_T_degen,
        scores=scores_degen,
        ema_auc_tracker=tracker,
        ema_beta=0.1,
    )
    assert abs(auc_degen - 0.9) < 1e-5
    assert abs(g_consist_degen - 0.8) < 1e-5  # 2.0 * (0.9 - 0.5) = 0.8


def test_c_flowbalance_ema_auc_integration():
    """Verify compute_c_flowbalance_advantage tracks EMA and reports metric."""
    B, L = 4, 8
    scores = torch.tensor([[0.0] * 7 + [1.0],
                           [0.0] * 7 + [0.0],
                           [0.0] * 7 + [1.0],
                           [0.0] * 7 + [0.0]], dtype=torch.float32)
    ref_lp = torch.full((B, L), -1.0)
    old_lp = torch.full((B, L), -1.0)
    resp_mask = torch.ones((B, L))
    uids = ["prompt_0", "prompt_0", "prompt_1", "prompt_1"]
    teacher_lp = torch.full((B, L), -0.5)
    priv_mask = torch.ones(B)

    tracker = {}
    adv, ret, metrics = compute_c_flowbalance_advantage(
        token_level_rewards=scores,
        ref_log_prob=ref_lp,
        old_log_prob=old_lp,
        response_mask=resp_mask,
        index=uids,
        teacher_log_prob=teacher_lp,
        self_distillation_mask=priv_mask,
        ema_auc_tracker=tracker,
        ema_beta=0.2,
    )

    assert "running_auc" in tracker
    assert "c_flowsd/running_ema_auc" in metrics
    assert 0.0 <= metrics["c_flowsd/running_ema_auc"] <= 1.0
    assert adv.shape == (B, L)
