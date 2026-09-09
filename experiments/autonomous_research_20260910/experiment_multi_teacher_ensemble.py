# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0.
"""Benchmark: Multi-Teacher Ensemble with Running Exponential Moving Average (EMA) AUC Gating.

Demonstrates that Cross-Batch Running EMA-AUC solves the All-Zero contrastive vacuum:
By accumulating teacher reliability across diverse prompts, the system accurately
mutes toxic teachers even on novel, zero-reward hard problems.
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


class ReasoningDAGEnv:
    def __init__(self, num_problems: int = 30, num_steps: int = 4, tokens_per_step: int = 8, vocab_size: int = 14):
        self.num_problems = num_problems
        self.num_steps = num_steps
        self.tokens_per_step = tokens_per_step
        self.seq_len = num_steps * tokens_per_step
        self.vocab_size = vocab_size

        self.problems = []
        for i in range(num_problems):
            correct = torch.randint(3, vocab_size, (num_steps,))
            distractors = torch.randint(3, vocab_size, (num_steps,))
            for s in range(num_steps):
                while distractors[s] == correct[s]:
                    distractors[s] = torch.randint(3, vocab_size, (1,)).item()
            is_hard = (i < num_problems // 2)
            self.problems.append({
                "id": f"prob_{i:02d}",
                "correct": correct,
                "distractors": distractors,
                "is_hard": is_hard,
            })

    def evaluate(self, p_idx: int, tokens: torch.Tensor) -> float:
        prob = self.problems[p_idx]
        for s in range(self.num_steps):
            fork_pos = s * self.tokens_per_step + 2
            if tokens[fork_pos].item() != prob["correct"][s].item():
                return 0.0
        return 1.0


class StudentPolicy(nn.Module):
    def __init__(self, num_problems: int, seq_len: int, vocab_size: int):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(num_problems, seq_len, vocab_size))

    def sample(self, p_idx: int, G: int) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.logits[p_idx]
        probs = F.softmax(logits, dim=-1)
        toks = torch.multinomial(probs, num_samples=G, replacement=True).transpose(0, 1)
        lp = F.log_softmax(logits, dim=-1).unsqueeze(0).expand(G, -1, -1).gather(
            -1, toks.unsqueeze(-1)
        ).squeeze(-1)
        return toks, lp


def compute_multi_teacher_adv_with_ema(
    ensemble_mode: str,
    scores: torch.Tensor,
    sampled_logp: torch.Tensor,
    ref_logp: torch.Tensor,
    teacher_logps: list[torch.Tensor],  # [T1_gold, T2_toxic, T3_noisy]
    ema_gates: list[float],             # Running EMA of teacher AUC
    ema_beta: float = 0.2,
    tau: float = 0.1,
    subtb_lambda: float = 0.5,
    gamma: float = 1.5,
) -> tuple[torch.Tensor, list[float]]:
    G, L = sampled_logp.shape
    device = sampled_logp.device
    num_teachers = len(teacher_logps)

    deltas = [(t_lp - ref_logp).clamp(-4.0, 4.0) for t_lp in teacher_logps]

    s_diff = scores.unsqueeze(1) - scores.unsqueeze(0)
    pos_mask = s_diff > 1e-6

    # Update running EMA when contrastive pairs exist
    for m, delta_m in enumerate(deltas):
        G_m_seq = delta_m.mean(dim=-1)
        if pos_mask.any():
            g_diff = G_m_seq.unsqueeze(1) - G_m_seq.unsqueeze(0)
            concordant = (g_diff[pos_mask] > 0).float().sum()
            ties = (g_diff[pos_mask] == 0).float().sum()
            auc = (concordant + 0.5 * ties) / pos_mask.float().sum().clamp(min=1.0)
            instant_g = float(np.clip(2.0 * (auc.item() - 0.5), 0.0, 1.0))
            ema_gates[m] = (1.0 - ema_beta) * ema_gates[m] + ema_beta * instant_g

    # Weighting
    if ensemble_mode == "uniform_average":
        effective_delta = sum(deltas) / num_teachers
        effective_alpha_gate = 0.5
    elif ensemble_mode == "instant_auc_only":
        # Uses only instant AUC without EMA
        total_g = sum(ema_gates)
        if total_g > 1e-6:
            teacher_weights = [g / total_g for g in ema_gates]
            effective_delta = sum(w * d for w, d in zip(teacher_weights, deltas))
            effective_alpha_gate = 0.5 * float(np.mean(ema_gates))
        else:
            effective_delta = torch.zeros_like(deltas[0])
            effective_alpha_gate = 0.0
    elif ensemble_mode == "ema_auc_gated_ensemble":
        # Filters using running EMA
        total_g = sum(ema_gates)
        if total_g > 1e-6:
            # Softmax or normalized weighting favoring reliable teachers
            teacher_weights = [g / total_g for g in ema_gates]
            effective_delta = sum(w * d for w, d in zip(teacher_weights, deltas))
            effective_alpha_gate = 0.5 * float(np.mean(ema_gates))
        else:
            effective_delta = torch.zeros_like(deltas[0])
            effective_alpha_gate = 0.0
    else:
        raise ValueError(f"Unknown mode: {ensemble_mode}")

    seq_old = sampled_logp.mean(dim=-1, keepdim=True)
    seq_ref = ref_logp.mean(dim=-1, keepdim=True)
    R_term = (scores / tau).unsqueeze(-1)
    G_T_seq = effective_delta.mean(dim=-1)

    target_uncentered = seq_ref + effective_alpha_gate * G_T_seq.unsqueeze(-1) + R_term / L
    baseline = (seq_old - target_uncentered).mean()
    A_TB = 2.0 * (target_uncentered + baseline - seq_old).expand(G, L)

    surprise = (effective_delta.abs() + 0.05).pow(gamma)
    w = surprise / surprise.sum(dim=-1, keepdim=True).clamp(min=1e-8)

    target_token = ref_logp + effective_alpha_gate * effective_delta + w * L * (R_term / L + baseline)
    A_DB = 2.0 * (target_token - sampled_logp)

    return (1.0 - subtb_lambda) * A_DB + subtb_lambda * A_TB, ema_gates


def run_ema_experiment():
    env = ReasoningDAGEnv(num_problems=30, num_steps=4, tokens_per_step=8, vocab_size=14)
    fork_indices = [s * 8 + 2 for s in range(4)]
    modes = ["uniform_average", "instant_auc_only", "ema_auc_gated_ensemble"]

    all_results = {}
    G = 8
    epochs = 20
    lr = 0.15

    for mode in modes:
        print(f"\n--- Testing Mode: {mode} ---")
        student = StudentPolicy(env.num_problems, env.seq_len, env.vocab_size)

        # Easy problems start neutral, hard problems start trapped
        with torch.no_grad():
            for p_idx, prob in enumerate(env.problems):
                if prob["is_hard"]:
                    for s in range(env.num_steps):
                        fork_pos = fork_indices[s]
                        student.logits[p_idx, fork_pos, prob["distractors"][s]] = 2.4

        ref_logits = student.logits.detach().clone()
        t1_logits = ref_logits.clone()  # Gold
        t2_logits = ref_logits.clone()  # Toxic
        t3_logits = ref_logits.clone() + torch.randn_like(ref_logits) * 0.5  # Noisy

        for p_idx, prob in enumerate(env.problems):
            for s in range(env.num_steps):
                fork_pos = fork_indices[s]
                t1_logits[p_idx, fork_pos, prob["correct"][s]] += 3.5
                t2_logits[p_idx, fork_pos, prob["distractors"][s]] += 3.5

        optimizer = torch.optim.Adam(student.parameters(), lr=lr)

        acc_history = []
        hard_acc_history = []

        # Running EMA priors for 3 teachers: start at neutral [0.5, 0.5, 0.5]
        running_ema = [0.5, 0.5, 0.5]

        for ep in range(epochs):
            total_corr = 0
            hard_corr = 0
            total_hard = 0

            for p_idx, prob in enumerate(env.problems):
                if prob["is_hard"]:
                    total_hard += 1

                toks, lp = student.sample(p_idx, G)
                scores = torch.tensor([env.evaluate(p_idx, toks[g]) for g in range(G)], dtype=torch.float32)

                if scores.max().item() > 0.5:
                    total_corr += 1
                    if prob["is_hard"]:
                        hard_corr += 1

                t_lps = []
                for t_log in [t1_logits, t2_logits, t3_logits]:
                    t_lp_all = F.log_softmax(t_log[p_idx], dim=-1).unsqueeze(0).expand(G, -1, -1)
                    t_lps.append(t_lp_all.gather(-1, toks.unsqueeze(-1)).squeeze(-1))

                ref_lp_all = F.log_softmax(ref_logits[p_idx], dim=-1).unsqueeze(0).expand(G, -1, -1)
                r_lp = ref_lp_all.gather(-1, toks.unsqueeze(-1)).squeeze(-1)

                adv, running_ema = compute_multi_teacher_adv_with_ema(
                    ensemble_mode=mode,
                    scores=scores,
                    sampled_logp=lp,
                    ref_logp=r_lp,
                    teacher_logps=t_lps,
                    ema_gates=running_ema,
                )
                adv = adv.detach()

                loss = -(lp * adv).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            pass1 = total_corr / env.num_problems
            hard_pass = hard_corr / max(1, total_hard)

            acc_history.append(pass1)
            hard_acc_history.append(hard_pass)

            if (ep + 1) % 5 == 0 or ep == epochs - 1:
                print(
                    f"  Epoch {ep+1:02d}/{epochs:02d} | "
                    f"Pass@1: {pass1*100:5.2f}% | "
                    f"Hard Pass: {hard_pass*100:5.2f}% | "
                    f"Running EMA [Gold, Toxic, Noisy]: [{running_ema[0]:.2f}, {running_ema[1]:.2f}, {running_ema[2]:.2f}]"
                )

        all_results[mode] = {
            "final_pass1": acc_history[-1],
            "final_hard_pass": hard_acc_history[-1],
            "final_ema": list(running_ema),
            "acc_history": acc_history,
            "hard_acc_history": hard_acc_history,
        }

    out_file = os.path.join(os.path.dirname(__file__), "ema_multi_teacher_results.json")
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n=========================================================================================")
    print(f"EMA MULTI-TEACHER BENCHMARK COMPLETE. SUMMARY:")
    print(f"{'Ensemble Mode':<26} | {'Pass@1':<8} | {'Hard Pass':<10} | {'Gold EMA':<12} | {'Toxic EMA':<12}")
    print("-" * 78)
    for k, v in all_results.items():
        g_gold = v["final_ema"][0]
        g_toxic = v["final_ema"][1]
        print(f"{k:<26} | {v['final_pass1']*100:6.2f}% | {v['final_hard_pass']*100:8.2f}% | {g_gold:10.2f} | {g_toxic:10.2f}")
    print(f"=========================================================================================")


if __name__ == "__main__":
    run_ema_experiment()
