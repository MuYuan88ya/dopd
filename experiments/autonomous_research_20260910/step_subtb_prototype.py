# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0.
"""Step-Boundary Vectorized SubTB (Step-SubTB) Prototype.

Implements fully vectorized, O(1)-kernel GPU step-level credit assignment
using scatter_add_ and gather operations with exact Mean Flow Conservation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_step_weights_vectorized(
    step_delimiter_mask: torch.Tensor,
    response_mask: torch.Tensor,
    token_surprise: torch.Tensor | None = None,
    gamma: float = 1.0,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes normalized token weights w in Delta^{L-1} aligned to semantic reasoning steps.

    Args:
        step_delimiter_mask: [B, L] boolean/float tensor, 1 at tokens that end a step (e.g. newline).
        response_mask: [B, L] float tensor, 1 for active response tokens, 0 for padding.
        token_surprise: [B, L] optional float tensor (e.g. |delta_t| or surprisal). If None, uniform across steps.
        gamma: power exponent for step weighting.
        eps: small positive constant for numerical stability.

    Returns:
        w: [B, L] normalized token weights such that (w * response_mask).sum(-1) == 1.0.
        step_ids: [B, L] step index (0, 1, ..., K-1) for each token.
    """
    B, L = response_mask.shape
    device = response_mask.device
    dtype = response_mask.dtype

    # Ensure step delimiter occurs only within active response tokens
    delimiters = (step_delimiter_mask.bool() & response_mask.bool()).long()

    # Step ID calculation:
    # A step ID increases after each delimiter.
    # To have the delimiter token itself belong to the current step, we shift cumsum by 1 or use cumsum with exclusive shift.
    # Specifically: step_id at token t is the number of delimiters *before* token t.
    step_id_shifted = torch.zeros_like(delimiters)
    step_id_shifted[:, 1:] = delimiters[:, :-1]
    step_ids = torch.cumsum(step_id_shifted, dim=-1) * response_mask.long()

    # Total number of steps across the batch
    max_steps = int(step_ids.max().item()) + 1

    # 1. Compute Step Lengths L_k = number of tokens in step k for each sequence
    # Shape: [B, max_steps]
    step_token_counts = torch.zeros(B, max_steps, device=device, dtype=dtype)
    step_token_counts.scatter_add_(1, step_ids, response_mask)

    # 2. Compute Step Mass / Surprise S_k
    if token_surprise is not None:
        raw_surprise = (token_surprise.clamp(min=0.0) + eps).pow(gamma) * response_mask
        step_surprise = torch.zeros(B, max_steps, device=device, dtype=dtype)
        step_surprise.scatter_add_(1, step_ids, raw_surprise)
        # Step weight W_k proportional to aggregate surprise in step k
        has_tokens = (step_token_counts > 0).float()
        step_mass = (step_surprise + eps) * has_tokens
        step_weights = step_mass / step_mass.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    else:
        # Uniform weight per step: W_k = 1 / K_b
        has_tokens = (step_token_counts > 0).float()
        num_steps_per_seq = has_tokens.sum(dim=-1, keepdim=True).clamp(min=1.0)
        step_weights = has_tokens / num_steps_per_seq

    # 3. Intra-step uniform distribution: w_t = W_k / L_k for all t in step k
    step_token_density = step_weights / step_token_counts.clamp(min=1.0)

    # 4. Gather back to token level: [B, L]
    token_weights = torch.gather(step_token_density, 1, step_ids) * response_mask

    # Renormalize to ensure exact numerical simplex constraint sum_t w_t == 1.0
    w_sum = token_weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    token_weights = (token_weights / w_sum) * response_mask

    return token_weights, step_ids


def test_step_subtb_vectorized():
    """Verify exact simplex conservation and step-granularity credit distribution."""
    print("Testing Step-SubTB Vectorized Implementation...")
    torch.manual_seed(42)

    B = 5
    L = 32
    # Generate random response lengths
    lengths = torch.randint(12, L + 1, (B,))
    response_mask = torch.zeros(B, L)
    for b in range(B):
        response_mask[b, :lengths[b]] = 1.0

    # Create synthetic step delimiters (e.g. every 5 to 8 tokens)
    step_delimiter_mask = torch.zeros(B, L)
    for b in range(B):
        pos = torch.randint(3, 7, (1,)).item()
        while pos < lengths[b]:
            step_delimiter_mask[b, pos] = 1.0
            pos += torch.randint(4, 8, (1,)).item()
        # Always end the final step at length - 1
        step_delimiter_mask[b, lengths[b] - 1] = 1.0

    # Case 1: Uniform Step Weighting
    w_uniform, step_ids = compute_step_weights_vectorized(
        step_delimiter_mask=step_delimiter_mask,
        response_mask=response_mask,
        token_surprise=None,
    )

    # Check simplex constraint
    sum_w = (w_uniform * response_mask).sum(dim=-1)
    diff_simplex = (sum_w - 1.0).abs().max().item()
    print(f"Simplex sum test (Uniform): max deviation = {diff_simplex:.8e}")
    assert diff_simplex < 1e-6, f"Simplex constraint violated: {diff_simplex}"

    # Verify intra-step uniformity
    for b in range(B):
        unique_steps = torch.unique(step_ids[b, :lengths[b]])
        for s in unique_steps:
            mask_s = (step_ids[b] == s) & (response_mask[b] == 1.0)
            step_w_vals = w_uniform[b, mask_s]
            assert (step_w_vals - step_w_vals[0]).abs().max().item() < 1e-6, "Intra-step weights must be identical!"

    # Case 2: Surprise-Weighted Step Weighting
    token_surprise = torch.rand(B, L) * response_mask
    w_surprise, _ = compute_step_weights_vectorized(
        step_delimiter_mask=step_delimiter_mask,
        response_mask=response_mask,
        token_surprise=token_surprise,
        gamma=1.5,
    )

    sum_w_surp = (w_surprise * response_mask).sum(dim=-1)
    diff_simplex_surp = (sum_w_surp - 1.0).abs().max().item()
    print(f"Simplex sum test (Surprise): max deviation = {diff_simplex_surp:.8e}")
    assert diff_simplex_surp < 1e-6, f"Simplex constraint violated: {diff_simplex_surp}"

    # Case 3: Verify Exact Flow Conservation in Detailed Balance vs Trajectory Balance
    # Let A_DB = 2.0 * (target - old_logp), where target has w * L * (R/L + b)
    old_logp = torch.randn(B, L) * response_mask
    ref_logp = torch.randn(B, L) * response_mask
    R = torch.tensor([1.0, 0.0, 1.0, 0.5, 0.0])
    baseline = torch.tensor([0.2, -0.1, 0.3, 0.0, 0.1])
    L_tensor = lengths.float().unsqueeze(-1)

    macro_flow = w_surprise * L_tensor * (R.unsqueeze(-1) / L_tensor + baseline.unsqueeze(-1))
    target = (ref_logp + macro_flow) * response_mask
    A_DB = 2.0 * (target - old_logp) * response_mask

    # Mean token advantage along sequence
    mean_A_seq = (A_DB * response_mask).sum(dim=-1) / lengths.float()

    # Trajectory Balance advantage
    target_TB = (ref_logp.sum(-1) / lengths.float() + R / lengths.float() + baseline)
    A_TB = 2.0 * (target_TB - old_logp.sum(-1) / lengths.float())

    flow_conservation_diff = (mean_A_seq - A_TB).abs().max().item()
    print(f"Exact Flow Conservation (mean A_DB == A_TB): max diff = {flow_conservation_diff:.8e}")
    assert flow_conservation_diff < 1e-6, f"Flow conservation violated: {flow_conservation_diff}"

    print("ALL STEP-SUBTB VECTORIZED TESTS PASSED SUCCESSFULLY!\n")


if __name__ == "__main__":
    test_step_subtb_vectorized()
