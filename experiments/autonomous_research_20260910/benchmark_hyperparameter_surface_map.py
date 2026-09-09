# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0.
"""Comprehensive 3D Hyperparameter Response Surface Benchmark.

Explores the interaction grid across:
- gamma in [0.5, 1.0, 1.5, 2.0] (Credit concentration power)
- subtb_lambda in [0.0, 0.25, 0.5, 0.75, 1.0] (Detailed vs Trajectory Balance horizon)
- alpha in [0.2, 0.4, 0.6, 0.8] (Teacher confidence weight)

Maps the empirical Pareto frontier of Pass@1, Hard Trap Recovery, and Gradient Variance.
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
    def __init__(self, num_problems: int = 24, num_steps: int = 4, tokens_per_step: int = 8, vocab_size: int = 12):
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


def compute_surface_adv(
    alpha: float,
    subtb_lambda: float,
    gamma: float,
    scores: torch.Tensor,
    sampled_logp: torch.Tensor,
    ref_logp: torch.Tensor,
    teacher_logp: torch.Tensor,
    tau: float = 0.1,
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
    R_term = (scores / tau).unsqueeze(-1)
    target_uncentered = seq_ref + (alpha * g_consist) * G_T_seq.unsqueeze(-1) + R_term / L
    baseline = (seq_old - target_uncentered).mean()
    A_TB = 2.0 * (target_uncentered + baseline - seq_old).expand(G, L)

    surprise = (delta.abs() + 0.05).pow(gamma)
    w = surprise / surprise.sum(dim=-1, keepdim=True).clamp(min=1e-8)

    target_token = ref_logp + (alpha * g_consist) * delta + w * L * (R_term / L + baseline)
    A_DB = 2.0 * (target_token - sampled_logp)

    return (1.0 - subtb_lambda) * A_DB + subtb_lambda * A_TB


def run_surface_sweep():
    gammas = [0.5, 1.0, 1.5, 2.0]
    lambdas = [0.0, 0.25, 0.5, 0.75, 1.0]
    alphas = [0.2, 0.4, 0.6, 0.8]

    total_runs = len(gammas) * len(lambdas) * len(alphas)
    print(f"Starting 3D Response Surface Sweep: {total_runs} configuration points...")

    env = ReasoningDAGEnv(num_problems=24, num_steps=4, tokens_per_step=8, vocab_size=12)
    fork_indices = [s * 8 + 2 for s in range(4)]
    G = 8
    epochs = 15
    lr = 0.15

    results_grid = []
    run_idx = 0

    for gamma in gammas:
        for subtb_lambda in lambdas:
            for alpha in alphas:
                run_idx += 1
                torch.manual_seed(42)
                student = StudentPolicy(env.num_problems, env.seq_len, env.vocab_size)

                with torch.no_grad():
                    for p_idx, prob in enumerate(env.problems):
                        if prob["is_hard"]:
                            for s in range(env.num_steps):
                                fork_pos = fork_indices[s]
                                student.logits[p_idx, fork_pos, prob["distractors"][s]] = 2.4

                ref_logits = student.logits.detach().clone()
                teacher_logits = ref_logits.clone()
                for p_idx, prob in enumerate(env.problems):
                    for s in range(env.num_steps):
                        fork_pos = fork_indices[s]
                        teacher_logits[p_idx, fork_pos, prob["correct"][s]] += 3.5

                optimizer = torch.optim.Adam(student.parameters(), lr=lr)

                total_corr = 0
                hard_corr = 0
                total_hard = 0
                adv_variances = []

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

                        teacher_lp_all = F.log_softmax(teacher_logits[p_idx], dim=-1).unsqueeze(0).expand(G, -1, -1)
                        t_lp = teacher_lp_all.gather(-1, toks.unsqueeze(-1)).squeeze(-1)

                        ref_lp_all = F.log_softmax(ref_logits[p_idx], dim=-1).unsqueeze(0).expand(G, -1, -1)
                        r_lp = ref_lp_all.gather(-1, toks.unsqueeze(-1)).squeeze(-1)

                        adv = compute_surface_adv(
                            alpha=alpha,
                            subtb_lambda=subtb_lambda,
                            gamma=gamma,
                            scores=scores,
                            sampled_logp=lp,
                            ref_logp=r_lp,
                            teacher_logp=t_lp,
                        ).detach()

                        adv_variances.append(adv.var().item())

                        loss = -(lp * adv).mean()
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                pass1 = total_corr / env.num_problems
                hard_pass = hard_corr / max(1, total_hard)
                avg_var = float(np.mean(adv_variances))

                grid_point = {
                    "gamma": gamma,
                    "lambda": subtb_lambda,
                    "alpha": alpha,
                    "pass1": pass1,
                    "hard_pass": hard_pass,
                    "adv_variance": avg_var,
                }
                results_grid.append(grid_point)

                if run_idx % 10 == 0 or run_idx == total_runs:
                    print(
                        f"  [{run_idx:02d}/{total_runs:02d}] "
                        f"γ={gamma:.1f}, λ={subtb_lambda:.2f}, α={alpha:.1f} | "
                        f"Pass@1: {pass1*100:5.2f}% | Hard Pass: {hard_pass*100:5.2f}% | Var: {avg_var:.4f}"
                    )

    out_file = os.path.join(os.path.dirname(__file__), "hyperparameter_surface_results.json")
    with open(out_file, "w") as f:
        json.dump(results_grid, f, indent=2)

    # Find Optimal Pareto Configuration
    best_overall = max(results_grid, key=lambda x: x["pass1"])
    best_hard = max(results_grid, key=lambda x: x["hard_pass"])

    print(f"\n=========================================================================================")
    print(f"HYPERPARAMETER SURFACE MAPPING COMPLETE.")
    print(f"Optimal Pass@1 Config: γ={best_overall['gamma']}, λ={best_overall['lambda']}, α={best_overall['alpha']} -> Pass@1 = {best_overall['pass1']*100:.2f}%")
    print(f"Optimal Hard Trap Config: γ={best_hard['gamma']}, λ={best_hard['lambda']}, α={best_hard['alpha']} -> Hard Pass = {best_hard['hard_pass']*100:.2f}%")
    print(f"=========================================================================================")


if __name__ == "__main__":
    run_surface_sweep()
