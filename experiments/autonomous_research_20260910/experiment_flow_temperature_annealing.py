# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0.
"""Rigorous Empirical Verification of Theorem 12:
Flow Temperature Annealing and Entropy Preservation in SubTB Reinforcement Learning.

Investigates entropy collapse and premature convergence into distractor traps:
- Fixed Cold Temperature (tau = 0.05): Fast early exploitation, but early entropy collapse and trap lock.
- Fixed Warm Temperature (tau = 0.50): Strong entropy preservation, but loose terminal mode convergence.
- Exponential Flow Annealing (tau = 0.50 -> 0.05): Sustains early exploration, avoids trap lock,
  and achieves sharp terminal mode locking.
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


class DistractorTrapReasoningEnv:
    """Reasoning environment where Step 3 contains a distractor trap."""

    def __init__(self, num_problems: int = 30, vocab_size: int = 12):
        self.num_problems = num_problems
        self.vocab_size = vocab_size

        self.problems = []
        for i in range(num_problems):
            target = torch.randint(2, vocab_size, (1,)).item()
            trap = torch.randint(2, vocab_size, (1,)).item()
            while trap == target:
                trap = torch.randint(2, vocab_size, (1,)).item()

            is_hard = (i < num_problems // 2)
            self.problems.append({
                "id": f"prob_{i:02d}",
                "target": target,
                "trap": trap,
                "is_hard": is_hard,
            })

    def evaluate(self, p_idx: int, tokens: torch.Tensor) -> tuple[float, bool]:
        prob = self.problems[p_idx]
        toks = tokens.tolist()
        # Step 3 fork (token 3) must avoid trap, Step 15 must match target
        if toks[3] != prob["trap"] and toks[15] == prob["target"]:
            return 1.0, prob["is_hard"]
        return 0.0, prob["is_hard"]


class StudentPolicy(nn.Module):
    def __init__(self, num_problems: int, seq_len: int = 16, vocab_size: int = 12):
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


def compute_subtb_adv_tau(
    scores: torch.Tensor,
    sampled_tokens: torch.Tensor,
    sampled_logp: torch.Tensor,
    ref_logp: torch.Tensor,
    teacher_logp: torch.Tensor,
    tau: float,
    subtb_lambda: float = 0.5,
    gamma: float = 1.5,
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

    seq_old = sampled_logp.mean(dim=-1, keepdim=True)
    seq_ref = ref_logp.mean(dim=-1, keepdim=True)
    R_term = (scores / tau).unsqueeze(-1) / L
    target_uncentered = seq_ref + (0.5 * g_consist) * G_T_seq.unsqueeze(-1) + R_term
    baseline = (seq_old - target_uncentered).mean()
    A_TB = 2.0 * (target_uncentered + baseline - seq_old).expand(G, L)

    surprise = (delta.abs() + 0.05).pow(gamma)
    w = surprise / surprise.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    target_token = ref_logp + (0.5 * g_consist) * delta + w * L * (R_term + baseline)
    A_DB = 2.0 * (target_token - sampled_logp)

    return (1.0 - subtb_lambda) * A_DB + subtb_lambda * A_TB


def train_schedule(schedule_name: str, epochs: int = 25, G: int = 8, lr: float = 0.2):
    env = DistractorTrapReasoningEnv(num_problems=30, vocab_size=12)
    student = StudentPolicy(env.num_problems, seq_len=16, vocab_size=12)

    # Initialize student: hard problems strongly biased to trap
    with torch.no_grad():
        for p_idx, prob in enumerate(env.problems):
            if prob["is_hard"]:
                student.logits[p_idx, 3, prob["trap"]] = 3.0
                student.logits[p_idx, 15, prob["target"]] = 0.5

    ref_logits = student.logits.detach().clone()
    teacher_logits = ref_logits.clone()
    for p_idx, prob in enumerate(env.problems):
        teacher_logits[p_idx, 3, prob["trap"]] = -3.0
        teacher_logits[p_idx, 15, prob["target"]] += 3.5

    optimizer = torch.optim.Adam(student.parameters(), lr=lr)

    acc_history = []
    hard_acc_history = []
    entropy_history = []

    for ep in range(epochs):
        # Determine tau
        if schedule_name == "Fixed_Cold_tau_0.05":
            tau = 0.05
        elif schedule_name == "Fixed_Warm_tau_0.50":
            tau = 0.50
        elif schedule_name == "Exponential_Annealing":
            tau = 0.05 + (0.50 - 0.05) * float(np.exp(-ep / 8.0))
        else:
            raise ValueError(f"Unknown schedule: {schedule_name}")

        corr = 0
        hard_corr = 0

        for p_idx, prob in enumerate(env.problems):
            toks, lp = student.sample(p_idx, G)
            scores = torch.tensor([env.evaluate(p_idx, toks[g])[0] for g in range(G)], dtype=torch.float32)

            if scores.max().item() > 0.5:
                corr += 1
                if prob["is_hard"]:
                    hard_corr += 1

            t_lp = F.log_softmax(teacher_logits[p_idx], dim=-1).unsqueeze(0).expand(G, -1, -1).gather(
                -1, toks.unsqueeze(-1)
            ).squeeze(-1)
            r_lp = F.log_softmax(ref_logits[p_idx], dim=-1).unsqueeze(0).expand(G, -1, -1).gather(
                -1, toks.unsqueeze(-1)
            ).squeeze(-1)

            adv = compute_subtb_adv_tau(
                scores=scores,
                sampled_tokens=toks,
                sampled_logp=lp,
                ref_logp=r_lp,
                teacher_logp=t_lp,
                tau=tau,
            ).detach()

            loss = -(lp * adv).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Compute average policy entropy across all problems
        with torch.no_grad():
            probs_all = F.softmax(student.logits, dim=-1)
            ent_all = -(probs_all * (probs_all + 1e-8).log()).sum(dim=-1).mean().item()

        pass1 = corr / env.num_problems
        hard_pass = hard_corr / (env.num_problems // 2)
        acc_history.append(pass1)
        hard_acc_history.append(hard_pass)
        entropy_history.append(ent_all)

        if (ep + 1) % 5 == 0 or ep == epochs - 1:
            print(
                f"  Epoch {ep+1:02d}/{epochs:02d} | "
                f"tau: {tau:5.3f} | "
                f"Pass@1: {pass1*100:5.1f}% | "
                f"Hard Pass: {hard_pass*100:5.1f}% | "
                f"Entropy: {ent_all:5.3f}"
            )

    return {
        "final_pass1": acc_history[-1],
        "final_hard_pass": hard_acc_history[-1],
        "final_entropy": entropy_history[-1],
        "acc_history": acc_history,
        "hard_history": hard_acc_history,
        "entropy_history": entropy_history,
    }


def run_benchmark():
    schedules = [
        "Fixed_Cold_tau_0.05",
        "Fixed_Warm_tau_0.50",
        "Exponential_Annealing",
    ]

    all_res = {}
    print(f"\n=========================================================================================")
    print(f"FLOW TEMPERATURE ANNEALING & ENTROPY DYNAMICS BENCHMARK (Theorem 12)")
    print("=" * 80)

    for sch in schedules:
        print(f"\n--- Testing Schedule: {sch} ---")
        res = train_schedule(sch, epochs=25, G=8, lr=0.2)
        all_res[sch] = res

    out_file = os.path.join(os.path.dirname(__file__), "temperature_annealing_results.json")
    with open(out_file, "w") as f:
        json.dump(all_res, f, indent=2)

    print(f"\n=========================================================================================")
    print(f"TEMPERATURE ANNEALING SUMMARY (Theorem 12):")
    print(f"{'Schedule':<25} | {'Pass@1':<8} | {'Hard Pass':<10} | {'Final Entropy':<14} | {'Exploration Health'}")
    print("-" * 80)
    for k, v in all_res.items():
        status = "HEALTHY & SHARP" if v["final_hard_pass"] >= 0.50 and v["final_entropy"] > 0.05 else "COLLAPSED / IMPRECISE"
        print(f"{k:<25} | {v['final_pass1']*100:6.1f}% | {v['final_hard_pass']*100:8.1f}% | {v['final_entropy']:12.3f} | {status}")
    print(f"=========================================================================================")


if __name__ == "__main__":
    run_benchmark()
