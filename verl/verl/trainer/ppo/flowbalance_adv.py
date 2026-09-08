# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0.
"""FlowBalance advantage estimator with seamless fallback to GSPO/GRPO."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Optional
import numpy as np
import torch

try:
    from verl.trainer.ppo.core_algos import register_adv_est
except Exception:
    def register_adv_est(name_or_enum):
        def decorator(fn):
            return fn
        return decorator


def _uids_to_list(uids: Any) -> list[Any]:
    if isinstance(uids, np.ndarray):
        return uids.tolist()
    if hasattr(uids, "tolist"):
        return uids.tolist()
    return list(uids)


@register_adv_est("flow_balance")
def compute_flowbalance_advantage(
    token_level_rewards: torch.Tensor,
    ref_log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    response_mask: torch.Tensor,
    index: Any,
    teacher_log_prob: Optional[torch.Tensor] = None,
    self_distillation_mask: Optional[torch.Tensor] = None,
    beta_q: float = 1.0,
    eta_R: float = 15.0,
    clip_B: float = 4.0,
    rho: float = 1.0,
    min_group_valid: int = 2,
    gate_no_context: str = "fallback_gspo",
    norm_adv_by_std_in_grpo: bool = True,
    subtb_lambda: float = 1.0,
    epsilon: float = 1e-6,
    config: Optional[Any] = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Compute length-normalized FlowBalance advantage with SubTB/Detailed Balance and GSPO fallback.

    Parameters:
        subtb_lambda (float): Sub-Trajectory Balance interpolation knob.
            - subtb_lambda = 1.0 (default): Exact Trajectory Balance (FlowBalance standard).
            - subtb_lambda = 0.0: Pure Detailed Balance (fine-grained token-level credit assignment).
            - 0.0 < subtb_lambda < 1.0: Sub-Trajectory Balance (SubTB) multi-scale interpolation.
            For any lambda in [0, 1], the sequence mean flow advantage is mathematically invariant.

    When a rollout group contains valid privileged demonstrations (self_distillation_mask > 0.5),
    the advantage matches the FlowBalance Trajectory Balance target:
        A_TB = 2 * (target - seq_logp_old) / L^rho
    When a rollout group lacks valid demonstrations (all rollouts failed in the group)
    or gate_no_context == "fallback_gspo", the advantage seamlessly falls back to the
    standard GRPO outcome advantage:
        A_fallback = A_GRPO
    When paired with VeRL's `loss_mode="gspo"`, the unprivileged samples are updated
    via 100% pure GSPO without dropping any rollout data.
    """
    device = ref_log_prob.device
    dtype = ref_log_prob.dtype

    response_mask = response_mask.to(device=device, dtype=dtype)
    ref_log_prob = ref_log_prob.to(device=device, dtype=dtype)
    old_log_prob = old_log_prob.to(device=device, dtype=dtype)
    token_level_rewards = token_level_rewards.to(device=device, dtype=dtype)

    batch_size = response_mask.shape[0]
    uid_list = _uids_to_list(index)
    if len(uid_list) != batch_size:
        raise ValueError(f"index length {len(uid_list)} does not match batch size {batch_size}")

    if self_distillation_mask is None:
        self_distillation_mask = torch.ones(batch_size, device=device, dtype=dtype)
    else:
        self_distillation_mask = self_distillation_mask.to(device=device, dtype=dtype)

    beta_zero = abs(beta_q) <= 1e-12
    if teacher_log_prob is None or beta_zero:
        has_teacher = False
        teacher_log_prob = ref_log_prob
    else:
        has_teacher = True
        teacher_log_prob = teacher_log_prob.to(device=device, dtype=dtype)

    with torch.no_grad():
        # 1. Length normalization factor
        raw_lengths = response_mask.sum(dim=-1)
        lengths = raw_lengths.clamp(min=1.0)
        length_norm = lengths.pow(rho)

        # 2. Sequence-level length-normalized log-probs
        seq_logp_ref = (ref_log_prob * response_mask).sum(dim=-1) / length_norm
        seq_logp_old = (old_log_prob * response_mask).sum(dim=-1) / length_norm

        # 3. Base GRPO outcome advantage calculation
        scores = token_level_rewards.sum(dim=-1)
        id2score = defaultdict(list)
        for i in range(batch_size):
            id2score[uid_list[i]].append(scores[i])

        grpo_adv = torch.zeros_like(scores)
        for i in range(batch_size):
            group_scores = torch.stack(id2score[uid_list[i]])
            mean_score = group_scores.mean()
            if len(group_scores) > 1 and norm_adv_by_std_in_grpo:
                std_score = group_scores.std()
                grpo_adv[i] = (scores[i] - mean_score) / (std_score + epsilon)
            else:
                grpo_adv[i] = scores[i] - mean_score

        # GRPO token-level advantage (pure GSPO fallback baseline)
        grpo_token_adv = grpo_adv.unsqueeze(-1) * response_mask

        # If beta_q == 0 or teacher_log_prob was not provided, full fallback to GRPO/GSPO
        if beta_zero or not has_teacher:
            metrics = {
                "flowsd/fallback_gspo_fraction": 1.0,
                "flowsd/tb_advantage_fraction": 0.0,
                "flowsd/grpo_adv_mean": grpo_adv.mean().item() if batch_size else 0.0,
            }
            return grpo_token_adv, grpo_token_adv, metrics

        # 4. Privileged teacher gain with clipping
        delta_raw = (teacher_log_prob - ref_log_prob) * response_mask
        delta = delta_raw.clamp(min=-clip_B, max=clip_B) * response_mask
        G_q_raw = delta.sum(dim=-1) / length_norm

        # 5. Sign-gated energy: verifier advantage dictates the direction
        advantage_sign = torch.sign(grpo_adv)
        G_q = G_q_raw * advantage_sign

        # Composite target log-flow
        log_R_tilde = seq_logp_ref + beta_q * G_q + eta_R * grpo_adv
        b_old = seq_logp_old - log_R_tilde

        # 6. Group valid masking
        valid_priv = (self_distillation_mask > 0.5) & (raw_lengths > 0)

        groups: dict[Any, list[int]] = defaultdict(list)
        for idx, uid in enumerate(uid_list):
            groups[uid].append(idx)

        baseline = torch.zeros(batch_size, device=device, dtype=dtype)
        is_tb_eligible = torch.zeros(batch_size, device=device, dtype=torch.bool)
        degenerate_groups = 0

        for indices in groups.values():
            idx_tensor = torch.tensor(indices, device=device, dtype=torch.long)
            group_valid = valid_priv[idx_tensor]
            if int(group_valid.sum().item()) < min_group_valid:
                degenerate_groups += 1
                # Group lacks enough privileged demonstration -> mark not eligible for TB
                continue
            valid_indices = idx_tensor[group_valid]
            group_baseline = b_old[valid_indices].mean()
            baseline[idx_tensor] = group_baseline
            is_tb_eligible[valid_indices] = True

        # 7. Construct FlowBalance Target & Trajectory Balance Advantage
        flowsd_target = log_R_tilde + baseline
        tb_seq_adv = 2.0 * (flowsd_target - seq_logp_old)
        tb_token_adv = (tb_seq_adv / length_norm).unsqueeze(-1) * response_mask

        # 7.1 SubTB / Detailed Balance Extension (when subtb_lambda < 1.0)
        if subtb_lambda < 1.0 - 1e-6:
            adv_sign = advantage_sign.unsqueeze(-1)
            G_q_token = delta * adv_sign
            target_token = (
                ref_log_prob
                + beta_q * G_q_token
                + (eta_R * grpo_adv + baseline).unsqueeze(-1)
            ) * response_mask

            db_token_adv = (2.0 * (target_token - old_log_prob) / length_norm.unsqueeze(-1)) * response_mask
            # SubTB interpolation: lambda=1.0 -> pure TB; lambda=0.0 -> pure DB
            effective_tb_adv = (1.0 - subtb_lambda) * db_token_adv + subtb_lambda * tb_token_adv
        else:
            effective_tb_adv = tb_token_adv

        # 8. Seamless Fallback Selection:
        # Eligible samples -> SubTB/FlowBalance advantage
        # Ineligible samples -> Standard GRPO advantage (pure GSPO) or drop
        if gate_no_context == "drop":
            final_token_adv = torch.where(
                is_tb_eligible.unsqueeze(-1),
                effective_tb_adv,
                torch.zeros_like(effective_tb_adv),
            )
        else:  # "fallback_gspo" (default)
            final_token_adv = torch.where(
                is_tb_eligible.unsqueeze(-1),
                effective_tb_adv,
                grpo_token_adv,
            )

        tb_fraction = is_tb_eligible.float().mean().item() if batch_size else 0.0
        metrics = {
            "flowsd/tb_advantage_fraction": tb_fraction,
            "flowsd/fallback_gspo_fraction": 1.0 - tb_fraction,
            "flowsd/subtb_lambda": float(subtb_lambda),
            "flowsd/degenerate_group_fraction": degenerate_groups / max(len(groups), 1),
            "flowsd/G_q_raw_mean": G_q_raw[is_tb_eligible].mean().item() if is_tb_eligible.any() else 0.0,
            "flowsd/tb_seq_adv_mean": tb_seq_adv[is_tb_eligible].mean().item() if is_tb_eligible.any() else 0.0,
            "flowsd/grpo_adv_mean": grpo_adv.mean().item() if batch_size else 0.0,
            "flowsd/G_q_mean": G_q[is_tb_eligible].mean().item() if is_tb_eligible.any() else 0.0,
            "flowsd/R_mean": grpo_adv.mean().item() if batch_size else 0.0,
            "flowsd/target_mean": flowsd_target[is_tb_eligible].mean().item() if is_tb_eligible.any() else 0.0,
            "flowsd/logZ_hat_mean": (-baseline[is_tb_eligible]).mean().item() if is_tb_eligible.any() else 0.0,
            "flowsd/final_adv_mean": final_token_adv[response_mask.bool()].mean().item() if response_mask.any() else 0.0,
            "flowsd/final_adv_std": final_token_adv[response_mask.bool()].std().item() if response_mask.any() else 0.0,
        }

    return final_token_adv, final_token_adv, metrics
