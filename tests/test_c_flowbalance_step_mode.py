# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0.
"""Unit tests for Step-Boundary Vectorized SubTB in Consistent FlowBalance."""

import importlib.util
import os
import unittest
import torch

module_path = os.path.abspath("verl/verl/trainer/ppo/c_flowbalance_adv.py")
spec = importlib.util.spec_from_file_location("c_flowbalance_adv", module_path)
c_flowbalance_adv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c_flowbalance_adv)

compute_c_flowbalance_advantage = c_flowbalance_adv.compute_c_flowbalance_advantage
_compute_step_weights_vectorized = c_flowbalance_adv._compute_step_weights_vectorized


class TestStepModeFlowBalance(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device("cpu")
        self.dtype = torch.float32

    def test_vectorized_step_weights_simplex_constraint(self):
        """Verify that _compute_step_weights_vectorized produces exact simplex weights."""
        B, L = 6, 30
        response_mask = torch.zeros(B, L)
        lengths = torch.randint(10, L + 1, (B,))
        for b in range(B):
            response_mask[b, :lengths[b]] = 1.0

        step_delimiters = torch.zeros(B, L)
        for b in range(B):
            pos = 3
            while pos < lengths[b]:
                step_delimiters[b, pos] = 1.0
                pos += 4

        # Uniform step weighting
        w_uniform, step_ids = _compute_step_weights_vectorized(
            step_delimiter_mask=step_delimiters,
            response_mask=response_mask,
            token_surprise=None,
        )

        sum_w = (w_uniform * response_mask).sum(dim=-1)
        for b in range(B):
            self.assertAlmostEqual(sum_w[b].item(), 1.0, places=5)

        # Intra-step uniformity check
        for b in range(B):
            steps_in_seq = torch.unique(step_ids[b, :lengths[b]])
            for s in steps_in_seq:
                mask = (step_ids[b] == s) & (response_mask[b] == 1.0)
                step_weights = w_uniform[b, mask]
                max_deviation = (step_weights - step_weights[0]).abs().max().item()
                self.assertLess(max_deviation, 1e-6)

    def test_step_mode_exact_flow_conservation(self):
        """Verify that compute_c_flowbalance_advantage preserves sequence flow conservation in step mode."""
        B, L = 4, 24
        token_rewards = torch.zeros(B, L)
        response_mask = torch.zeros(B, L)
        for b in range(B):
            response_mask[b, :18] = 1.0
            token_rewards[b, 17] = 1.0 if b % 2 == 0 else 0.0

        old_lp = torch.randn(B, L) * response_mask
        ref_lp = torch.randn(B, L) * response_mask
        teacher_lp = torch.randn(B, L) * response_mask

        # Step delimiters at positions 5, 11, 17
        step_delimiters = torch.zeros(B, L)
        step_delimiters[:, 5] = 1.0
        step_delimiters[:, 11] = 1.0
        step_delimiters[:, 17] = 1.0

        uids = ["group_0", "group_0", "group_1", "group_1"]

        # Run with SubTB lambda = 0.0 (Pure Detailed Balance) in step mode
        adv_step_db, _, _ = compute_c_flowbalance_advantage(
            token_level_rewards=token_rewards,
            ref_log_prob=ref_lp,
            old_log_prob=old_lp,
            response_mask=response_mask,
            index=uids,
            teacher_log_prob=teacher_lp,
            alpha=0.5,
            subtb_lambda=0.0,
            token_weight_mode="step",
            token_weight_gamma=1.5,
            step_delimiter_mask=step_delimiters,
        )

        # Run with SubTB lambda = 1.0 (Pure Trajectory Balance)
        adv_tb, _, _ = compute_c_flowbalance_advantage(
            token_level_rewards=token_rewards,
            ref_log_prob=ref_lp,
            old_log_prob=old_lp,
            response_mask=response_mask,
            index=uids,
            teacher_log_prob=teacher_lp,
            alpha=0.5,
            subtb_lambda=1.0,
        )

        # By Theorem 1 (Generalized Mean Flow Conservation), the sequence-mean advantage
        # of step-weighted SubTB must identically equal the Trajectory Balance advantage!
        lengths = response_mask.sum(dim=-1)
        mean_adv_step = (adv_step_db * response_mask).sum(dim=-1) / lengths
        mean_adv_tb = (adv_tb * response_mask).sum(dim=-1) / lengths

        diff = (mean_adv_step - mean_adv_tb).abs().max().item()
        print(f"Step Mode Mean Flow Conservation max diff: {diff:.8e}")
        self.assertLess(diff, 1e-5, f"Step mode violated flow conservation: {diff}")

    def test_step_mode_graceful_fallback_when_mask_none(self):
        """Verify that when step_delimiter_mask is None, step mode falls back safely to uniform."""
        B, L = 4, 16
        token_rewards = torch.zeros(B, L)
        response_mask = torch.ones(B, L)
        token_rewards[:, -1] = 1.0
        old_lp = torch.randn(B, L)
        ref_lp = torch.randn(B, L)
        uids = ["g1", "g1", "g2", "g2"]

        adv, returns, metrics = compute_c_flowbalance_advantage(
            token_level_rewards=token_rewards,
            ref_log_prob=ref_lp,
            old_log_prob=old_lp,
            response_mask=response_mask,
            index=uids,
            token_weight_mode="step",
            step_delimiter_mask=None,
        )
        self.assertEqual(adv.shape, (B, L))
        self.assertFalse(torch.isnan(adv).any())


if __name__ == "__main__":
    unittest.main()
