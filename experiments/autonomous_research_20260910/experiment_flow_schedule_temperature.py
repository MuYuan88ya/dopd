# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0.
"""Benchmark: Synergy between Surprise Weighting and Temperature Scheduling."""

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


class MultiPhaseReasoningDAG:
    def __init__(self, num_problems: int = 30, num_steps: int = 4, tokens_per_step: int = 8, vocab_size: int = 12):
        self.num_problems = num_problems
        self.num_steps = num_steps
        self.tokens_per_step = tokens_per_step
        self.seq_len = num_steps * tokens_per_step
        self.vocab_size = vocab_size

        self.problems = []
        for i in range(num_problems):
            correct = torch.randint(2, vocab_size, (num_steps,))
            distractors = torch.randint(2, vocab_size, (num_steps,))
            for s in range(num_steps):
                while distractors[s] == correct[s]:
                    distractors[s] = torch.randint(2, vocab_size, (1,)).item()
            is_hard = (i < num_problems // 2)
            self.problems.append({
                "id": f"prob_{i:02d}",
                "correct": correct,
                "distractors": distractors,
                "is_hard": is_hard,
            })

    def evaluate(self, p_idx: int, tokens: torch.Tensor) -> tuple[float, int]:
        prob = self.problems[p_idx]
        for s in range(self.num_steps):
            fork_pos = s * self.tokens_per_step + 2
            if tokens[fork_pos].item() != prob["correct"][s].item():
                return 0.0, s
        return 1.0, -1


class StudentPolicy(nn.Module):
    def __init__(self, num_problems: int, seq_len: int, vocab_size: int):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(num_problems, seq_len, vocab_size))

    def sample(self, p_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.logits[p_idx]
        probs = F.softmax(logits, dim=-1)
        toks = torch.multinomial(probs, num_samples=1).squeeze(-1)
        lp = F.log_softmax(logits, dim=-1).gather(-1, toks.unsqueeze(-1)).squeeze(-1)
        return toks, lp


def compute_advantages_scheduled_ew(
    mode: str,
    scores: torch.Tensor,
    sampled_logp: torch.Tensor,
    ref_logp: torch.Tensor,
    teacher_logp: torch.Tensor,
    gamma: float = 1.2,
    alpha: float = 0.5,
    subtb_lambda: float = 0.5,
) -> torch.Tensor:
    G, L = sampled_logp.shape
    device = sampled_logp.device

    delta = (teacher_logp - ref_logp).clamp(-4.0, 4.0)
    G_T_seq = delta.mean(dim=-1)

    s_diff = scores.unsqueeze(1) - scores.unsqueeze(0)
    pos_mask = s_diff > 1e-6
    if pos_mask.any():
        g_diff = G_T_seq.unsqueeze(1) - G_T_seq.unsqueeze(0)
        concordant = (g_diff[pos_mask] > 0).float().sum()
        ties = (g_diff[pos_mask] == 0).float().sum()
        auc = (concordant + 0.5 * ties) / pos_mask.float().sum().clamp(min=1.0)
        g_consist = float(np.clip(2.0 * (auc.item() - 0.5), 0.0, 1.0))
    else:
        g_consist = 0.5

    t_indices = torch.linspace(0.0, 1.0, L, device=device)

    if mode == "Uniform_SubTB":
        w = torch.ones(G, L, device=device) / L
        harmonic_mean_tau = 0.1
    elif mode == "EW_SubTB_Constant_Tau":
        # Constant tau = 0.1
        surprise = (delta.abs() + 0.1).pow(gamma)
        w = surprise / surprise.sum(dim=-1, keepdim=True)
        harmonic_mean_tau = 0.1
    elif mode == "EW_SubTB_Linear_Decay_Tau":
        # Tau decays from 0.25 (early exploration) to 0.05 (late calculation rigor)
        tau_t = 0.25 - (0.25 - 0.05) * t_indices
        inv_tau = 1.0 / tau_t
        harmonic_mean_tau = L / inv_tau.sum().item()
        # Scale surprise by 1 / tau_t
        surprise = (delta.abs() + 0.1).pow(gamma) * inv_tau.unsqueeze(0)
        w = surprise / surprise.sum(dim=-1, keepdim=True)
    elif mode == "EW_SubTB_Linear_Warmup_Tau":
        # Tau warms up from 0.05 to 0.25
        tau_t = 0.05 + (0.25 - 0.05) * t_indices
        inv_tau = 1.0 / tau_t
        harmonic_mean_tau = L / inv_tau.sum().item()
        surprise = (delta.abs() + 0.1).pow(gamma) * inv_tau.unsqueeze(0)
        w = surprise / surprise.sum(dim=-1, keepdim=True)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    seq_old = sampled_logp.mean(dim=-1, keepdim=True)
    seq_ref = ref_logp.mean(dim=-1, keepdim=True)
    R_term = (scores / harmonic_mean_tau).unsqueeze(-1)
    target_uncentered = seq_ref + (alpha * g_consist) * G_T_seq.unsqueeze(-1) + R_term / L
    baseline = (seq_old - target_uncentered).mean()
    A_TB = 2.0 * (target_uncentered + baseline - seq_old).expand(G, L)

    target_token = ref_logp + (alpha * g_consist) * delta + w * L * (R_term / L + baseline)
    A_DB = 2.0 * (target_token - sampled_logp)

    return (1.0 - subtb_lambda) * A_DB + subtb_lambda * A_TB


def run_experiment():
    num_problems = 30
    num_steps = 4
    tokens_per_step = 8
    vocab_size = 12
    env = MultiPhaseReasoningDAG(
        num_problems=num_problems,
        num_steps=num_steps,
        tokens_per_step=tokens_per_step,
        vocab_size=vocab_size,
    )

    fork_indices = [s * tokens_per_step + 2 for s in range(num_steps)]

    modes = [
        "Uniform_SubTB",
        "EW_SubTB_Constant_Tau",
        "EW_SubTB_Linear_Decay_Tau",
        "EW_SubTB_Linear_Warmup_Tau",
    ]

    all_results = {}
    G = 8
    epochs = 20
    lr = 0.15

    for mode in modes:
        print(f"\n--- Testing Mode: {mode} ---")
        student = StudentPolicy(num_problems, env.seq_len, vocab_size)

        with torch.no_grad():
            for p_idx, prob in enumerate(env.problems):
                if prob["is_hard"]:
                    for s in range(num_steps):
                        fork_pos = fork_indices[s]
                        student.logits[p_idx, fork_pos, prob["distractors"][s]] = 2.5

        ref_logits = student.logits.detach().clone()
        teacher_logits = ref_logits.clone()
        for p_idx, prob in enumerate(env.problems):
            for s in range(num_steps):
                fork_pos = fork_indices[s]
                teacher_logits[p_idx, fork_pos, prob["correct"][s]] += 3.5

        optimizer = torch.optim.Adam(student.parameters(), lr=lr)

        acc_history = []
        hard_acc_history = []

        for ep in range(epochs):
            total_corr = 0
            hard_corr = 0
            total_hard = 0

            for p_idx, prob in enumerate(env.problems):
                if prob["is_hard"]:
                    total_hard += 1

                sampled_tokens = []
                sampled_logp = []
                scores = []

                for _ in range(G):
                    toks, lp = student.sample(p_idx)
                    score, _ = env.evaluate(p_idx, toks)
                    sampled_tokens.append(toks)
                    sampled_logp.append(lp)
                    scores.append(score)

                sampled_tokens = torch.stack(sampled_tokens)
                sampled_logp = torch.stack(sampled_logp)
                scores = torch.tensor(scores, dtype=torch.float32)

                if scores.max().item() > 0.5:
                    total_corr += 1
                    if prob["is_hard"]:
                        hard_corr += 1

                teacher_lp_all = F.log_softmax(teacher_logits[p_idx], dim=-1).unsqueeze(0).expand(G, -1, -1)
                t_lp = teacher_lp_all.gather(-1, sampled_tokens.unsqueeze(-1)).squeeze(-1)

                ref_lp_all = F.log_softmax(ref_logits[p_idx], dim=-1).unsqueeze(0).expand(G, -1, -1)
                r_lp = ref_lp_all.gather(-1, sampled_tokens.unsqueeze(-1)).squeeze(-1)

                adv = compute_advantages_scheduled_ew(
                    mode=mode,
                    scores=scores,
                    sampled_logp=sampled_logp,
                    ref_logp=r_lp,
                    teacher_logp=t_lp,
                ).detach()

                loss = -(sampled_logp * adv).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            pass1 = total_corr / num_problems
            hard_pass = hard_corr / max(1, total_hard)
            acc_history.append(pass1)
            hard_acc_history.append(hard_pass)

            if (ep + 1) % 5 == 0 or ep == epochs - 1:
                print(
                    f"  Epoch {ep+1:02d}/{epochs:02d} | "
                    f"Pass@1: {pass1*100:5.2f}% | "
                    f"Hard Pass: {hard_pass*100:5.2f}%"
                )

        all_results[mode] = {
            "final_pass1": acc_history[-1],
            "final_hard_pass": hard_acc_history[-1],
            "acc_history": acc_history,
            "hard_acc_history": hard_acc_history,
        }

    out_file = os.path.join(os.path.dirname(__file__), "scheduled_ew_results.json")
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n=======================================================")
    print(f"BENCHMARK COMPLETE. SUMMARY:")
    print(f"{'Mode':<30} | {'Pass@1':<8} | {'Hard Pass':<10}")
    print("-" * 55)
    for k, v in all_results.items():
        print(f"{k:<30} | {v['final_pass1']*100:6.2f}% | {v['final_hard_pass']*100:8.2f}%")
    print(f"=======================================================")


if __name__ == "__main__":
    run_experiment()
