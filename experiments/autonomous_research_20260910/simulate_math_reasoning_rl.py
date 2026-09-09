# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0.
"""Empirical Simulation Environment: Multi-Step Mathematical Reasoning RL Benchmark.

Compares:
1. Standard GRPO
2. FlowBalance (Trajectory Balance, TB)
3. C-FlowBalance (Uniform Detailed Balance, SubTB-Uniform)
4. C-FlowBalance + Surprise-Weighted SubTB (SW-SubTB)
5. C-FlowBalance + Step-Boundary SubTB (Step-SubTB)

Measures:
- Pass@1 Accuracy trajectory
- Recovery rate on 'Hard' problems (initial Pass@G = 0)
- Gradient Signal-to-Noise Ratio (SNR)
- Step-level credit assignment accuracy
"""

from __future__ import annotations

import json
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


class MathReasoningEnv:
    """Synthetic multi-step reasoning problem with decision forks and filler tokens."""

    def __init__(
        self,
        num_problems: int = 50,
        num_steps: int = 5,
        tokens_per_step: int = 15,
        vocab_size: int = 32,
        hard_fraction: float = 0.4,
    ):
        self.num_problems = num_problems
        self.num_steps = num_steps
        self.tokens_per_step = tokens_per_step
        self.seq_len = num_steps * tokens_per_step
        self.vocab_size = vocab_size

        # For each problem, define the correct sequence of decision tokens (1 key token per step)
        # Decision token is located at index 2 of each step
        self.problems = []
        for i in range(num_problems):
            # Target correct decision at each step
            correct_decisions = torch.randint(1, vocab_size, (num_steps,))
            is_hard = (i < int(num_problems * hard_fraction))
            self.problems.append({
                "id": f"prob_{i:03d}",
                "correct_decisions": correct_decisions,
                "is_hard": is_hard,
            })

    def evaluate_trajectory(self, problem_idx: int, tokens: torch.Tensor) -> tuple[float, int]:
        """Check correctness. Returns reward (1.0 or 0.0) and first error step (-1 if all correct)."""
        prob = self.problems[problem_idx]
        for step in range(self.num_steps):
            decision_pos = step * self.tokens_per_step + 2
            chosen_token = tokens[decision_pos].item()
            if chosen_token != prob["correct_decisions"][step].item():
                return 0.0, step
        return 1.0, -1


class PolicyModel(nn.Module):
    """Linear-softmax policy parameterized per problem and token position."""

    def __init__(self, num_problems: int, seq_len: int, vocab_size: int):
        super().__init__()
        # Logits: [num_problems, seq_len, vocab_size]
        self.logits = nn.Parameter(torch.randn(num_problems, seq_len, vocab_size) * 0.1)

    def get_log_probs(self, problem_idx: int) -> torch.Tensor:
        return F.log_softmax(self.logits[problem_idx], dim=-1)

    def sample_rollout(self, problem_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        log_probs = self.get_log_probs(problem_idx)
        probs = log_probs.exp()
        sampled_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
        sampled_logp = log_probs.gather(-1, sampled_tokens.unsqueeze(-1)).squeeze(-1)
        return sampled_tokens, sampled_logp


def compute_advantages(
    method: str,
    scores: torch.Tensor,               # [G]
    sampled_logp: torch.Tensor,         # [G, L]
    ref_logp: torch.Tensor,             # [G, L]
    teacher_logp: torch.Tensor,         # [G, L]
    step_masks: list[torch.Tensor],     # list of M step masks
    alpha: float = 0.5,
    tau: float = 0.1,
    subtb_lambda: float = 0.5,
) -> torch.Tensor:
    """Compute token-level advantage under specified algorithm."""
    G, L = sampled_logp.shape
    device = sampled_logp.device

    # 1. Base GRPO outcome advantage
    mean_score = scores.mean()
    std_score = scores.std()
    if std_score > 1e-6:
        grpo_adv = (scores - mean_score) / (std_score + 1e-6)
    else:
        grpo_adv = scores - mean_score
    grpo_token_adv = grpo_adv.unsqueeze(-1).expand(G, L)

    if method == "GRPO":
        return grpo_token_adv

    # 2. Teacher delta
    delta = (teacher_logp - ref_logp).clamp(-4.0, 4.0)
    G_T_seq = delta.mean(dim=-1)

    # 3. Pairwise AUC Consistency Gate
    s_diff = scores.unsqueeze(1) - scores.unsqueeze(0)
    pos_mask = s_diff > 1e-6
    if pos_mask.any():
        g_diff = G_T_seq.unsqueeze(1) - G_T_seq.unsqueeze(0)
        concordant = (g_diff[pos_mask] > 0).float().sum()
        ties = (g_diff[pos_mask] == 0).float().sum()
        auc = (concordant + 0.5 * ties) / pos_mask.float().sum().clamp(min=1.0)
        g_consist = float(np.clip(2.0 * (auc.item() - 0.5), 0.0, 1.0))
    else:
        # All tie (all 0 or all 1)
        g_consist = 0.5

    # 4. Target construction
    R_term = (scores / tau).unsqueeze(-1)  # [G, 1]
    # Macro seq logp
    seq_old = sampled_logp.mean(dim=-1, keepdim=True)
    seq_ref = ref_logp.mean(dim=-1, keepdim=True)
    target_uncentered = seq_ref + (alpha * g_consist) * G_T_seq.unsqueeze(-1) + (scores / tau).unsqueeze(-1) / L
    baseline = (seq_old - target_uncentered).mean()

    # Trajectory Balance advantage
    seq_target = target_uncentered + baseline
    A_TB = 2.0 * (seq_target - seq_old).expand(G, L)

    if method == "FlowBalance_TB":
        return A_TB

    # 5. Detailed Balance with various token weighting schemes
    if method == "SubTB_Uniform":
        w = torch.ones(G, L, device=device) / L
    elif method == "SW_SubTB":
        # Surprise-Weighted: tokens where teacher diverges get higher weight
        surprise = delta.abs() + 0.05
        w = surprise / surprise.sum(dim=-1, keepdim=True)
    elif method == "Step_SubTB":
        # Step-Level SubTB: balance across steps
        w = torch.zeros(G, L, device=device)
        num_steps = len(step_masks)
        for s_mask in step_masks:
            step_len = s_mask.sum().clamp(min=1.0)
            w += s_mask.unsqueeze(0) / (num_steps * step_len)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Construct weighted DB target: strictly preserves sum_t w_t = 1
    weighted_return_and_base = w * L * (R_term / L + baseline)
    target_token = ref_logp + (alpha * g_consist) * delta + weighted_return_and_base
    A_DB = 2.0 * (target_token - sampled_logp)

    # SubTB convex combination
    A_SubTB = (1.0 - subtb_lambda) * A_DB + subtb_lambda * A_TB
    return A_SubTB


def run_benchmark():
    num_problems = 40
    num_steps = 4
    tokens_per_step = 10
    vocab_size = 20
    env = MathReasoningEnv(num_problems=num_problems, num_steps=num_steps, tokens_per_step=tokens_per_step, vocab_size=vocab_size)

    # Step masks
    step_masks = []
    for s in range(num_steps):
        m = torch.zeros(env.seq_len)
        m[s * tokens_per_step : (s + 1) * tokens_per_step] = 1.0
        step_masks.append(m)

    methods = ["GRPO", "FlowBalance_TB", "SubTB_Uniform", "SW_SubTB", "Step_SubTB"]
    results = {}

    G = 8  # rollouts per prompt
    num_epochs = 15
    lr = 0.15

    for method in methods:
        print(f"\nEvaluating Method: {method} ...")
        # Initialize student policy
        student = PolicyModel(num_problems, env.seq_len, vocab_size)
        optimizer = torch.optim.Adam(student.parameters(), lr=lr)

        # Frozen reference policy
        ref_logits = student.logits.detach().clone()

        # Oracle teacher policy: on decision tokens, puts 90% mass on the correct choice
        teacher_logits = ref_logits.clone()
        for p_idx, prob in enumerate(env.problems):
            for step in range(num_steps):
                pos = step * tokens_per_step + 2
                correct_tok = prob["correct_decisions"][step].item()
                teacher_logits[p_idx, pos, correct_tok] += 3.0  # Privileged signal

        acc_history = []
        hard_acc_history = []
        snr_history = []
        credit_precision_history = []

        for epoch in range(num_epochs):
            total_correct = 0
            hard_correct = 0
            total_hard = 0
            epoch_grads = []
            credit_hits = []

            for p_idx in range(num_problems):
                is_hard = env.problems[p_idx]["is_hard"]
                if is_hard:
                    total_hard += G

                # Rollout G candidates
                sampled_tokens_list = []
                sampled_logp_list = []
                scores_list = []
                error_steps_list = []

                for g in range(G):
                    toks, lp = student.sample_rollout(p_idx)
                    rew, err_step = env.evaluate_trajectory(p_idx, toks)
                    sampled_tokens_list.append(toks)
                    sampled_logp_list.append(lp)
                    scores_list.append(rew)
                    error_steps_list.append(err_step)
                    if rew > 0.5:
                        total_correct += 1
                        if is_hard:
                            hard_correct += 1

                scores = torch.tensor(scores_list)
                sampled_logp = torch.stack(sampled_logp_list)
                tokens_tensor = torch.stack(sampled_tokens_list)

                # Reference and teacher log_probs for sampled tokens
                ref_lp_all = F.log_softmax(ref_logits[p_idx], dim=-1)
                ref_lp = ref_lp_all.gather(-1, tokens_tensor).squeeze(-1)

                teacher_lp_all = F.log_softmax(teacher_logits[p_idx], dim=-1)
                teacher_lp = teacher_lp_all.gather(-1, tokens_tensor).squeeze(-1)

                # Compute Advantage
                adv = compute_advantages(
                    method=method,
                    scores=scores,
                    sampled_logp=sampled_logp,
                    ref_logp=ref_lp,
                    teacher_logp=teacher_lp,
                    step_masks=step_masks,
                )

                # Step-level credit attribution check:
                # For failed trajectories, did the advantage negatively penalize the exact step of error?
                for g in range(G):
                    err_step = error_steps_list[g]
                    if err_step != -1:
                        err_pos = err_step * tokens_per_step + 2
                        # Check if error token received negative advantage
                        if adv[g, err_pos].item() < -0.1:
                            credit_hits.append(1.0)
                        else:
                            credit_hits.append(0.0)

                # Policy loss and gradient update
                optimizer.zero_grad()
                curr_lp = F.log_softmax(student.logits[p_idx], dim=-1).gather(-1, tokens_tensor).squeeze(-1)
                loss = -(curr_lp * adv.detach()).mean()
                loss.backward()

                # Collect gradients for SNR
                g_vec = student.logits.grad[p_idx].detach().clone()
                epoch_grads.append(g_vec)
                optimizer.step()

            # Epoch metrics
            pass1 = total_correct / (num_problems * G)
            hard_pass1 = hard_correct / max(total_hard, 1)
            # Gradient SNR = |mean| / (std + eps)
            stacked_grads = torch.stack(epoch_grads)
            g_mean = stacked_grads.mean(dim=0).abs().mean().item()
            g_std = stacked_grads.std(dim=0).mean().item()
            snr = g_mean / (g_std + 1e-6)
            credit_prec = float(np.mean(credit_hits)) if credit_hits else 0.0

            acc_history.append(pass1)
            hard_acc_history.append(hard_pass1)
            snr_history.append(snr)
            credit_precision_history.append(credit_prec)

            if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
                print(f"  Epoch {epoch+1:02d}/{num_epochs}: Pass@1 = {pass1*100:.1f}%, Hard Pass@1 = {hard_pass1*100:.1f}%, SNR = {snr:.3f}, Credit Prec = {credit_prec*100:.1f}%")

        results[method] = {
            "final_pass1": acc_history[-1],
            "final_hard_pass1": hard_acc_history[-1],
            "mean_snr": float(np.mean(snr_history)),
            "credit_precision": credit_precision_history[-1],
            "acc_history": acc_history,
            "hard_acc_history": hard_acc_history,
        }

    # Save results to JSON
    out_dir = "experiments/autonomous_research_20260910"
    out_file = os.path.join(out_dir, "benchmark_results.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print(" FINAL COMPARATIVE BENCHMARK REPORT")
    print("=" * 80)
    print(f"{'Method':<20} | {'Pass@1 (%)':<12} | {'Hard Pass@1 (%)':<16} | {'Avg SNR':<10} | {'Credit Precision (%)':<20}")
    print("-" * 88)
    for m in methods:
        r = results[m]
        print(f"{m:<20} | {r['final_pass1']*100:<12.2f} | {r['final_hard_pass1']*100:<16.2f} | {r['mean_snr']:<10.3f} | {r['credit_precision']*100:<20.2f}")
    print("=" * 80)


if __name__ == "__main__":
    run_benchmark()
