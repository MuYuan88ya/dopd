# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0.
"""C-FlowBalance: Consistent Sub-Trajectory Balance Advantage Estimator with Pairwise AUC Gating."""

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


def compute_pairwise_auc_group(
    G_seq: torch.Tensor,
    scores: torch.Tensor,
    g_consist_prior: float = 0.5,
    ema_auc_tracker: Optional[dict[str, float]] = None,
    ema_beta: float = 0.1,
) -> tuple[float, float]:
    """Computes pairwise concordance AUC of teacher guidance against outcome rewards.

    When ema_auc_tracker is provided:
        - Updates the running teacher AUC on groups with non-zero score variance.
        - Uses the historical running EMA on degenerate groups (e.g. all rollouts 0), transferring curriculum trust.

    Args:
        G_seq: [G] sequence-level teacher gain
        scores: [G] sequence-level outcome rewards
        g_consist_prior: float in [0.0, 1.0], default prior confidence when all rollouts in the group tie
        ema_auc_tracker: optional dict storing {"running_auc": float} across groups/batches
        ema_beta: float, EMA momentum parameter

    Returns:
        auc: float in [0.0, 1.0]
        g_consist: float in [0.0, 1.0]
    """
    if len(scores) < 2:
        if ema_auc_tracker is not None and "running_auc" in ema_auc_tracker:
            auc_val = ema_auc_tracker["running_auc"]
            g_consist_val = float(np.clip(2.0 * (auc_val - 0.5), 0.0, 1.0))
            return auc_val, g_consist_val
        return 0.5, float(g_consist_prior)

    # Pairs (i, j) where scores[i] > scores[j]
    s_diff = scores.unsqueeze(1) - scores.unsqueeze(0)
    pos_mask = s_diff > 1e-6
    if not pos_mask.any():
        # All completions in group received identical scores -> no contrastive truth
        if ema_auc_tracker is not None and "running_auc" in ema_auc_tracker:
            auc_val = ema_auc_tracker["running_auc"]
            g_consist_val = float(np.clip(2.0 * (auc_val - 0.5), 0.0, 1.0))
            return auc_val, g_consist_val
        return 0.5, float(g_consist_prior)

    g_diff = G_seq.unsqueeze(1) - G_seq.unsqueeze(0)
    concordant = (g_diff[pos_mask] > 0).float().sum()
    ties = (g_diff[pos_mask] == 0).float().sum()
    total_pairs = pos_mask.float().sum()
    auc = (concordant + 0.5 * ties) / total_pairs.clamp(min=1.0)
    auc_val = float(auc.item())

    # Update running EMA
    if ema_auc_tracker is not None:
        cur_ema = ema_auc_tracker.get("running_auc", auc_val)
        ema_auc_tracker["running_auc"] = (1.0 - ema_beta) * cur_ema + ema_beta * auc_val

    # Map [0.5, 1.0] -> [0.0, 1.0]. Toxic/hallucinating teacher (AUC <= 0.5) is muted to 0.0.
    g_consist_val = float(np.clip(2.0 * (auc_val - 0.5), 0.0, 1.0))
    return auc_val, g_consist_val


def _compute_step_weights_vectorized(
    step_delimiter_mask: torch.Tensor,
    response_mask: torch.Tensor,
    token_surprise: Optional[torch.Tensor] = None,
    gamma: float = 1.0,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes normalized token weights w in Delta^{L-1} aligned to semantic reasoning steps."""
    B, L = response_mask.shape
    device = response_mask.device
    dtype = response_mask.dtype

    delimiters = (step_delimiter_mask.bool() & response_mask.bool()).long()
    step_id_shifted = torch.zeros_like(delimiters)
    step_id_shifted[:, 1:] = delimiters[:, :-1]
    step_ids = torch.cumsum(step_id_shifted, dim=-1) * response_mask.long()

    max_steps = int(step_ids.max().item()) + 1
    step_token_counts = torch.zeros(B, max_steps, device=device, dtype=dtype)
    step_token_counts.scatter_add_(1, step_ids, response_mask)

    if token_surprise is not None:
        raw_surprise = (token_surprise.clamp(min=0.0) + eps).pow(gamma) * response_mask
        step_surprise = torch.zeros(B, max_steps, device=device, dtype=dtype)
        step_surprise.scatter_add_(1, step_ids, raw_surprise)
        has_tokens = (step_token_counts > 0).float()
        step_mass = (step_surprise + eps) * has_tokens
        step_weights = step_mass / step_mass.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    else:
        has_tokens = (step_token_counts > 0).float()
        num_steps_per_seq = has_tokens.sum(dim=-1, keepdim=True).clamp(min=1.0)
        step_weights = has_tokens / num_steps_per_seq

    step_token_density = step_weights / step_token_counts.clamp(min=1.0)
    token_weights = torch.gather(step_token_density, 1, step_ids) * response_mask
    w_sum = token_weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    token_weights = (token_weights / w_sum) * response_mask

    return token_weights, step_ids


@register_adv_est("c_flow_balance")
def compute_c_flowbalance_advantage(
    token_level_rewards: torch.Tensor,
    ref_log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    response_mask: torch.Tensor,
    index: Any,
    teacher_log_prob: Optional[torch.Tensor] = None,
    self_distillation_mask: Optional[torch.Tensor] = None,
    alpha: float = 0.5,
    tau: float = 0.1,
    clip_B: float = 4.0,
    rho: float = 1.0,
    min_group_valid: int = 2,
    gate_no_context: str = "fallback_gspo",
    norm_adv_by_std_in_grpo: bool = True,
    subtb_lambda: float = 1.0,
    token_weight_mode: str = "uniform",
    token_weight_gamma: float = 1.0,
    g_consist_prior: float = 0.5,
    vac_mode: bool = False,
    vac_sigma_0: float = 0.25,
    vac_alpha_min: float = 0.2,
    vac_alpha_max: float = 0.8,
    flow_gae_mode: bool = False,
    step_delimiter_mask: Optional[torch.Tensor] = None,
    ema_auc_tracker: Optional[dict[str, float]] = None,
    ema_beta: float = 0.1,
    epsilon: float = 1e-6,
    config: Optional[Any] = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Compute C-FlowBalance advantage with pairwise AUC consistency gating and SubTB.

    Features:
        1. g_consist in [0, 1]: Pairwise AUC consistency gate between teacher gain and outcome reward.
           Safely mutes toxic/hallucinating teachers (g_consist -> 0) without erroneous sign-flipping.
        2. Bounded Confidence & Temperature:
           - alpha in [0, 1]: strictly bounded teacher confidence coordinate on the simplex.
           - tau: exploration temperature scale.
        3. SubTB (Sub-Trajectory Balance) / Detailed Balance:
           - subtb_lambda = 1.0 (default): Trajectory Balance.
           - subtb_lambda in [0, 1): Detailed Balance & SubTB token-level credit assignment.
           - Strictly satisfies Mean Flow Conservation: (1/L) sum_t A_SubTB,t == A_TB.
        4. Seamless GSPO Fallback: Unprivileged samples gracefully fall back to standard GRPO advantage.

    Returns:
        advantages: shape [B, L]
        returns: shape [B, L]
        metrics: scalar diagnostics
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
        raise ValueError(f"uid length {len(uid_list)} does not match batch size {batch_size}")

    if self_distillation_mask is None:
        self_distillation_mask = torch.ones(batch_size, device=device, dtype=dtype)
    else:
        self_distillation_mask = self_distillation_mask.to(device=device, dtype=dtype)

    alpha_zero = abs(alpha) <= 1e-12
    has_teacher = teacher_log_prob is not None

    # 1. Sequence Lengths
    raw_lengths = response_mask.sum(dim=-1)
    lengths = raw_lengths.clamp(min=1.0)
    length_norm = lengths.pow(rho)

    # 2. Sequence Log-Probs
    seq_logp_ref = (ref_log_prob * response_mask).sum(dim=-1) / length_norm
    seq_logp_old = (old_log_prob * response_mask).sum(dim=-1) / length_norm

    # 3. Standard GRPO Advantage (Baseline and Fallback)
    scores = (token_level_rewards * response_mask).sum(dim=-1)
    id2score = defaultdict(list)
    for i in range(batch_size):
        id2score[uid_list[i]].append(scores[i])

    id2mean = {}
    id2std = {}
    for uid, group_s in id2score.items():
        if len(group_s) == 1:
            id2mean[uid] = group_s[0]
            id2std[uid] = torch.tensor(1.0, device=device, dtype=dtype)
        else:
            s_tensor = torch.stack(group_s)
            id2mean[uid] = s_tensor.mean()
            id2std[uid] = s_tensor.std()

    grpo_adv = torch.zeros(batch_size, device=device, dtype=dtype)
    for i in range(batch_size):
        uid = uid_list[i]
        if norm_adv_by_std_in_grpo:
            grpo_adv[i] = (scores[i] - id2mean[uid]) / (id2std[uid] + epsilon)
        else:
            grpo_adv[i] = scores[i] - id2mean[uid]

    grpo_token_adv = grpo_adv.unsqueeze(-1) * response_mask

    # If alpha == 0 or no teacher provided, complete graceful fallback to pure GSPO
    if alpha_zero or not has_teacher:
        metrics = {
            "c_flowsd/fallback_gspo_fraction": 1.0,
            "c_flowsd/c_flowbalance_fraction": 0.0,
            "c_flowsd/g_consist_mean": 0.0,
            "c_flowsd/grpo_adv_mean": grpo_adv.mean().item() if batch_size else 0.0,
        }
        return grpo_token_adv, grpo_token_adv, metrics

    teacher_log_prob = teacher_log_prob.to(device=device, dtype=dtype)

    # 4. Privileged Teacher Gain with Clipping
    delta_raw = (teacher_log_prob - ref_log_prob) * response_mask
    delta = delta_raw.clamp(min=-clip_B, max=clip_B) * response_mask
    G_T_seq = delta.sum(dim=-1) / length_norm

    # 5. Grouping, Consistency Gating, and Baseline Calculation
    valid_priv = (self_distillation_mask > 0.5) & (raw_lengths > 0)
    groups: dict[Any, list[int]] = defaultdict(list)
    for idx, uid in enumerate(uid_list):
        groups[uid].append(idx)

    baseline = torch.zeros(batch_size, device=device, dtype=dtype)
    g_consist_tensor = torch.zeros(batch_size, device=device, dtype=dtype)
    alpha_tensor = torch.full((batch_size,), fill_value=alpha, device=device, dtype=dtype)
    is_eligible = torch.zeros(batch_size, device=device, dtype=torch.bool)
    degenerate_groups = 0
    muted_toxic_teachers = 0
    all_aucs = []

    # Scaled reward term: R / (tau * length_norm)
    tau_safe = max(float(tau), 1e-4)
    R_term_seq = (scores / tau_safe) / length_norm

    for indices in groups.values():
        idx_tensor = torch.tensor(indices, device=device, dtype=torch.long)
        group_valid = valid_priv[idx_tensor]
        if int(group_valid.sum().item()) < min_group_valid:
            degenerate_groups += 1
            continue

        valid_indices = idx_tensor[group_valid]
        group_scores = scores[valid_indices]
        group_G_T = G_T_seq[valid_indices]

        # Compute Pairwise AUC and g_consist for this prompt group
        auc_val, g_consist_val = compute_pairwise_auc_group(
            group_G_T,
            group_scores,
            g_consist_prior=g_consist_prior,
            ema_auc_tracker=ema_auc_tracker,
            ema_beta=ema_beta,
        )
        all_aucs.append(auc_val)
        if auc_val < 0.5 - 1e-6:
            muted_toxic_teachers += 1

        # Variance-Adaptive Confidence (VAC): scale alpha dynamically based on outcome variance
        if vac_mode:
            group_std = group_scores.std().item() if len(group_scores) > 1 else 0.0
            alpha_group = vac_alpha_min + (vac_alpha_max - vac_alpha_min) * float(np.exp(-group_std / max(vac_sigma_0, 1e-4)))
        else:
            alpha_group = alpha

        alpha_tensor[valid_indices] = alpha_group
        g_consist_tensor[valid_indices] = g_consist_val
        is_eligible[valid_indices] = True

        # Target uncentered: seq_logp_ref + (alpha_group * g_consist) * G_T + R_term_seq
        effective_teacher_gain = (alpha_group * g_consist_val) * G_T_seq[valid_indices]
        target_uncentered = seq_logp_ref[valid_indices] + effective_teacher_gain + R_term_seq[valid_indices]

        # Zero-residual baseline b_group
        b_old = seq_logp_old[valid_indices] - target_uncentered
        group_baseline = b_old.mean()
        baseline[idx_tensor] = group_baseline

    # 6. Construct C-FlowBalance Target and Detailed Balance Advantage
    effective_alpha_token = (alpha_tensor * g_consist_tensor).unsqueeze(-1)

    if token_weight_mode == "surprise" and abs(token_weight_gamma) > 1e-6:
        # Surprise-weighted SubTB: w_t proportional to (|delta_t| + eps)^gamma
        # Concentrates macro flow updates on decision-critical tokens
        surprise = (delta.abs() + 0.05).pow(token_weight_gamma) * response_mask
        w = surprise / surprise.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        w_factor = w * lengths.unsqueeze(-1)
        macro_flow_term = w_factor * (R_term_seq + baseline).unsqueeze(-1)
    elif token_weight_mode == "step" and step_delimiter_mask is not None:
        # Step-weighted SubTB: semantic step boundaries (e.g. newline tokens)
        w, _ = _compute_step_weights_vectorized(
            step_delimiter_mask=step_delimiter_mask.to(device=device, dtype=dtype),
            response_mask=response_mask,
            token_surprise=delta.abs() if abs(token_weight_gamma) > 1e-6 else None,
            gamma=token_weight_gamma,
        )
        w_factor = w * lengths.unsqueeze(-1)
        macro_flow_term = w_factor * (R_term_seq + baseline).unsqueeze(-1)
    else:
        # Standard uniform SubTB (default)
        macro_flow_term = (R_term_seq + baseline).unsqueeze(-1)

    target_token = (
        ref_log_prob
        + effective_alpha_token * delta
        + macro_flow_term
    ) * response_mask

    A_DB = 2.0 * (target_token - old_log_prob) * response_mask

    # 7. Trajectory Balance (TB) and SubTB Combination
    A_TB = (A_DB.sum(dim=-1) / lengths).unsqueeze(-1).expand_as(A_DB) * response_mask

    if subtb_lambda < 1.0 - 1e-6:
        if flow_gae_mode:
            # Flow-GAE: recursive multi-horizon span discounting along the trajectory
            adv_gae = torch.zeros_like(A_DB)
            running_future = torch.zeros(batch_size, device=device, dtype=dtype)
            terminal_indices = (raw_lengths.long() - 1).clamp(min=-1)
            seq_dim = A_DB.shape[1]
            for t in reversed(range(seq_dim)):
                is_terminal_token = (t == terminal_indices)
                future_target = torch.where(is_terminal_token, A_TB[:, t], running_future)
                running_future = (
                    (1.0 - subtb_lambda) * A_DB[:, t] + subtb_lambda * future_target
                ) * response_mask[:, t]
                adv_gae[:, t] = running_future
            effective_adv = adv_gae
        else:
            # Standard 2-point SubTB convex combination: (1 - lambda) * DB + lambda * TB
            effective_adv = (1.0 - subtb_lambda) * A_DB + subtb_lambda * A_TB
    else:
        # Pure TB
        effective_adv = A_TB

    # 8. Seamless Fallback Execution
    if gate_no_context == "drop":
        final_token_adv = torch.where(
            is_eligible.unsqueeze(-1),
            effective_adv,
            torch.zeros_like(effective_adv),
        )
    else:  # "fallback_gspo" (default)
        final_token_adv = torch.where(
            is_eligible.unsqueeze(-1),
            effective_adv,
            grpo_token_adv,
        )

    # 9. Diagnostic Metrics
    eligible_count = int(is_eligible.sum().item())
    total_groups = max(len(groups), 1)
    metrics: dict[str, float] = {
        "c_flowsd/c_flowbalance_fraction": float(eligible_count / max(batch_size, 1)),
        "c_flowsd/fallback_gspo_fraction": float(1.0 - (eligible_count / max(batch_size, 1))),
        "c_flowsd/degenerate_group_fraction": float(degenerate_groups / total_groups),
        "c_flowsd/toxic_teacher_muted_fraction": float(muted_toxic_teachers / total_groups),
        "c_flowsd/g_consist_mean": float(g_consist_tensor[is_eligible].mean().item()) if eligible_count else 0.0,
        "c_flowsd/auc_mean": float(np.mean(all_aucs)) if all_aucs else 0.5,
        "c_flowsd/alpha": float(alpha),
        "c_flowsd/vac_alpha_mean": float(alpha_tensor[is_eligible].mean().item()) if eligible_count else float(alpha),
        "c_flowsd/flow_gae_mode": float(flow_gae_mode),
        "c_flowsd/tau": float(tau),
        "c_flowsd/subtb_lambda": float(subtb_lambda),
        "c_flowsd/advantage_mean": float(final_token_adv[response_mask.bool()].mean().item()) if response_mask.any() else 0.0,
    }
    if ema_auc_tracker is not None and "running_auc" in ema_auc_tracker:
        metrics["c_flowsd/running_ema_auc"] = float(ema_auc_tracker["running_auc"])

    return final_token_adv, final_token_adv, metrics
