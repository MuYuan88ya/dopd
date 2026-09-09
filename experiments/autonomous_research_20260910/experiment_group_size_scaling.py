# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0.
"""Rigorous Benchmark: Group Size (G) Scaling and Sample Complexity.

Investigates how group size G in {2, 4, 8, 16} affects:
1. AUC consistency gating reliability
2. Sample efficiency (Pass@1 as a function of total rollout tokens)
3. Distractor trap escape under constrained rollout budgets
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
        toks = torch.multinomial(probs, num_samples=G, replacement=True).transpose(0, 1)  # [G, L]
        lp = F.log_softmax(logits, dim=-1).unsqueeze(0).expand(G, -1, -1).gather(
            -1, toks.unsqueeze(-1)
        ).squeeze(-1)
        return toks, lp


def compute_advantages(
    method: str,
    scores: torch.Tensor,
    sampled_logp: torch.Tensor,
    ref_logp: torch.Tensor,
    teacher_logp: torch.Tensor,
    tau: float = 0.1,
    subtb_lambda: float = 0.5,
    gamma: float = 1.5,
    g_consist_prior: float = 0.5,
) -> torch.Tensor:
    G, L = sampled_logp.shape
    device = sampled_logp.device

    mean_s = scores.mean()
    std_s = scores.std()
    grpo_adv = (scores - mean_s) / (std_s + 1e-6) if std_s > 1e-6 else (scores - mean_s)
    if method == "GRPO":
        return grpo_adv.unsqueeze(-1).expand(G, L)

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
        g_consist = g_consist_prior

    seq_old = sampled_logp.mean(dim=-1, keepdim=True)
    seq_ref = ref_logp.mean(dim=-1, keepdim=True)
    R_term = (scores / tau).unsqueeze(-1)
    target_uncentered = seq_ref + (0.5 * g_consist) * G_T_seq.unsqueeze(-1) + R_term / L
    baseline = (seq_old - target_uncentered).mean()
    A_TB = 2.0 * (target_uncentered + baseline - seq_old).expand(G, L)

    surprise = (delta.abs() + 0.05).pow(gamma)
    w = surprise / surprise.sum(dim=-1, keepdim=True).clamp(min=1e-8)

    target_token = ref_logp + (0.5 * g_consist) * delta + w * L * (R_term / L + baseline)
    A_DB = 2.0 * (target_token - sampled_logp)

    return (1.0 - subtb_lambda) * A_DB + subtb_lambda * A_TB


def run_group_size_benchmark():
    group_sizes = [2, 4, 8, 16]
    methods = ["GRPO", "C_FlowBalance"]
    env = ReasoningDAGEnv(num_problems=30, num_steps=4, tokens_per_step=8, vocab_size=12)
    fork_indices = [s * 8 + 2 for s in range(4)]

    # We equalize total rollout budget across G:
    # total_samples = epochs * G -> epochs = budget // G
    budget_per_problem = 160  # total rollouts per problem
    lr = 0.15

    all_results = {}

    for method in methods:
        for G in group_sizes:
            epochs = budget_per_problem // G
            config_name = f"{method}_G{G}"
            print(f"\n--- Testing {config_name} (Epochs: {epochs}, Budget: {budget_per_problem}) ---")

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

            acc_history = []
            hard_acc_history = []

            for ep in range(epochs):
                total_corr = 0
                hard_corr = 0
                total_hard = 0

                for p_idx, prob in enumerate(env.problems):
                    if prob["is_hard"]:
                        total_hard += 1

                    toks, lp = student.sample(p_idx, G)
                    scores = []
                    for g in range(G):
                        score = env.evaluate(p_idx, toks[g])
                        scores.append(score)
                    scores = torch.tensor(scores, dtype=torch.float32)

                    if scores.max().item() > 0.5:
                        total_corr += 1
                        if prob["is_hard"]:
                            hard_corr += 1

                    teacher_lp_all = F.log_softmax(teacher_logits[p_idx], dim=-1).unsqueeze(0).expand(G, -1, -1)
                    t_lp = teacher_lp_all.gather(-1, toks.unsqueeze(-1)).squeeze(-1)

                    ref_lp_all = F.log_softmax(ref_logits[p_idx], dim=-1).unsqueeze(0).expand(G, -1, -1)
                    r_lp = ref_lp_all.gather(-1, toks.unsqueeze(-1)).squeeze(-1)

                    adv = compute_advantages(
                        method=method,
                        scores=scores,
                        sampled_logp=lp,
                        ref_logp=r_lp,
                        teacher_logp=t_lp,
                    ).detach()

                    loss = -(lp * adv).mean()

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                pass1 = total_corr / env.num_problems
                hard_pass = hard_corr / max(1, total_hard)
                acc_history.append(pass1)
                hard_acc_history.append(hard_pass)

            print(
                f"  [{config_name}] Final Pass@1: {acc_history[-1]*100:5.2f}% | "
                f"Hard Pass: {hard_acc_history[-1]*100:5.2f}%"
            )

            all_results[config_name] = {
                "method": method,
                "G": G,
                "epochs": epochs,
                "final_pass1": acc_history[-1],
                "final_hard_pass": hard_acc_history[-1],
                "acc_history": acc_history,
                "hard_acc_history": hard_acc_history,
            }

    out_file = os.path.join(os.path.dirname(__file__), "group_size_scaling_results.json")
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n=========================================================================================")
    print(f"GROUP SIZE SCALING SUMMARY (EQUAL SAMPLE BUDGET = {budget_per_problem} ROLLOUTS):")
    print(f"{'Config':<20} | {'G':<4} | {'Epochs':<8} | {'Pass@1':<8} | {'Hard Pass':<10}")
    print("-" * 65)
    for k, v in all_results.items():
        print(f"{k:<20} | {v['G']:<4} | {v['epochs']:<8} | {v['final_pass1']*100:6.2f}% | {v['final_hard_pass']*100:8.2f}%")
    print(f"=========================================================================================")


if __name__ == "__main__":
    run_group_size_benchmark()
