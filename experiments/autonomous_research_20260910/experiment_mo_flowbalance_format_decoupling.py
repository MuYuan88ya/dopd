# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0.
"""Rigorous Empirical Verification of Theorem 11:
Multi-Objective Decoupled FlowBalance (MO-FlowBalance) vs Format Hallucination in Multi-Verifier RL.

Theoretical Problem:
In reasoning models (e.g. DeepSeek-R1 style), rewards combine Math Correctness (R_math in {0, 1})
and Structural Formatting (R_format in {0, 1}).
Under standard scalarized GRPO (R = R_math + beta * R_format), rollouts with correct format but
wrong math receive positive reward, causing Cross-Objective Credit Contamination:
the model reinforces the WRONG math because the format was praised ('Format Hallucination').

MO-FlowBalance Solution:
Orthogonalizes flow simplex weights: <w_math, w_format> = 0.
Format reward flows exclusively to syntax tokens; Math reward flows exclusively to reasoning tokens.
Strictly satisfies Joint Mean Flow Conservation while eliminating format hacking.
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


class MultiObjectiveReasoningEnv:
    """Environment with heterogeneous verifiers:

    - Total length L = 16 tokens.
    - Format tokens: Indices [0, 1, 14, 15] (headers/tags).
      Must match format target [1, 2, 3, 4] -> R_format in {0.0, 1.0}.
    - Math reasoning tokens: Indices [2..13].
      Step 5 is decision fork (trap vs target).
      Step 11 is final answer calculation.
      Must avoid trap at Step 5 and match target at Step 11 -> R_math in {0.0, 1.0}.
    """

    def __init__(self, num_problems: int = 30, vocab_size: int = 14):
        self.num_problems = num_problems
        self.vocab_size = vocab_size
        self.format_indices = [0, 1, 14, 15]
        self.math_indices = [i for i in range(16) if i not in self.format_indices]
        self.format_pattern = [1, 2, 3, 4]

        self.problems = []
        for i in range(num_problems):
            target_fork = torch.randint(5, vocab_size, (1,)).item()
            trap_fork = torch.randint(5, vocab_size, (1,)).item()
            while trap_fork == target_fork:
                trap_fork = torch.randint(5, vocab_size, (1,)).item()

            target_ans = torch.randint(5, vocab_size, (1,)).item()
            is_hard = (i < num_problems // 2)
            self.problems.append({
                "id": f"prob_{i:02d}",
                "target_fork": target_fork,
                "trap_fork": trap_fork,
                "target_ans": target_ans,
                "is_hard": is_hard,
            })

    def evaluate(self, p_idx: int, tokens: torch.Tensor) -> tuple[float, float, bool]:
        """Returns (R_math, R_format, is_hard)."""
        prob = self.problems[p_idx]
        toks = tokens.tolist()

        # Format verification
        format_actual = [toks[idx] for idx in self.format_indices]
        format_ok = (format_actual == self.format_pattern)
        r_format = 1.0 if format_ok else 0.0

        # Math verification
        took_trap = (toks[5] == prob["trap_fork"])
        got_fork = (toks[5] == prob["target_fork"])
        got_ans = (toks[11] == prob["target_ans"])

        math_ok = (not took_trap) and got_fork and got_ans
        r_math = 1.0 if math_ok else 0.0

        return r_math, r_format, prob["is_hard"]


class StudentPolicy(nn.Module):
    def __init__(self, num_problems: int, seq_len: int = 16, vocab_size: int = 14):
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


def compute_mo_advantages(
    method: str,
    r_math: torch.Tensor,
    r_format: torch.Tensor,
    sampled_tokens: torch.Tensor,
    sampled_logp: torch.Tensor,
    ref_logp: torch.Tensor,
    teacher_logp: torch.Tensor,
    env: MultiObjectiveReasoningEnv,
    beta_format: float = 0.5,
    tau: float = 0.1,
    subtb_lambda: float = 0.5,
    gamma: float = 1.5,
) -> torch.Tensor:
    G, L = sampled_logp.shape
    device = sampled_logp.device

    # Scalarized total outcome reward
    r_total = r_math + beta_format * r_format

    if method == "Scalarized_GRPO":
        mean_s = r_total.mean()
        std_s = r_total.std()
        grpo_adv = (r_total - mean_s) / (std_s + 1e-6) if std_s > 1e-6 else (r_total - mean_s)
        return grpo_adv.unsqueeze(-1).expand(G, L)

    delta = (teacher_logp - ref_logp).clamp(-4.0, 4.0)
    G_T_seq = delta.mean(dim=-1)

    s_diff = r_math.unsqueeze(1) - r_math.unsqueeze(0)
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

    if method == "Scalarized_SubTB":
        # Standard SubTB using total scalarized reward
        R_term = (r_total / tau).unsqueeze(-1) / L
        target_uncentered = seq_ref + (0.5 * g_consist) * G_T_seq.unsqueeze(-1) + R_term
        baseline = (seq_old - target_uncentered).mean()
        A_TB = 2.0 * (target_uncentered + baseline - seq_old).expand(G, L)

        surprise = (delta.abs() + 0.05).pow(gamma)
        w = surprise / surprise.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        target_token = ref_logp + (0.5 * g_consist) * delta + w * L * (R_term + baseline)
        A_DB = 2.0 * (target_token - sampled_logp)
        return (1.0 - subtb_lambda) * A_DB + subtb_lambda * A_TB

    elif method == "MO_Decoupled_FlowBalance":
        # Theorem 11: Orthogonal Multi-Objective Flow Decomposition
        # Split credit by verifier domain:
        format_mask = torch.zeros(G, L, device=device)
        format_mask[:, env.format_indices] = 1.0

        math_mask = torch.zeros(G, L, device=device)
        math_mask[:, env.math_indices] = 1.0

        # Math flow weight: surprise-weighted within math tokens
        math_surprise = (delta.abs() + 0.05).pow(gamma) * math_mask
        w_math = math_surprise / math_surprise.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        # Format flow weight: uniform within format tokens
        w_format = format_mask / format_mask.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        # Sequence level targets for each objective
        R_math_term = (r_math / tau).unsqueeze(-1) / L
        R_format_term = (beta_format * r_format / tau).unsqueeze(-1) / L

        target_math_seq = seq_ref + (0.5 * g_consist) * G_T_seq.unsqueeze(-1) + R_math_term
        target_format_seq = seq_ref + R_format_term

        b_math = (seq_old - target_math_seq).mean()
        b_format = (seq_old - target_format_seq).mean()

        A_TB_math = 2.0 * (target_math_seq + b_math - seq_old)
        A_TB_format = 2.0 * (target_format_seq + b_format - seq_old)
        A_TB = A_TB_math.expand(G, L) * math_mask + A_TB_format.expand(G, L) * format_mask

        # Token level targets: orthogonal flows
        target_token_math = ref_logp + (0.5 * g_consist) * delta + w_math * L * (R_math_term + b_math)
        target_token_format = ref_logp + w_format * L * (R_format_term + b_format)

        target_token = target_token_math * math_mask + target_token_format * format_mask
        A_DB = 2.0 * (target_token - sampled_logp)

        return (1.0 - subtb_lambda) * A_DB + subtb_lambda * A_TB

    else:
        raise ValueError(f"Unknown method: {method}")


def run_benchmark():
    env = MultiObjectiveReasoningEnv(num_problems=30, vocab_size=14)
    methods = [
        "Scalarized_GRPO",
        "Scalarized_SubTB",
        "MO_Decoupled_FlowBalance",
    ]

    all_results = {}
    G = 8
    epochs = 25
    lr = 0.2

    for method in methods:
        print(f"\n--- Testing Method: {method} ---")
        student = StudentPolicy(env.num_problems, seq_len=16, vocab_size=14)

        # Initialize student:
        # Format is easy to discover: weak positive bias on format
        with torch.no_grad():
            for p_idx, prob in enumerate(env.problems):
                # Strong bias on format target
                for i, idx in enumerate(env.format_indices):
                    student.logits[p_idx, idx, env.format_pattern[i]] = 2.0

                if prob["is_hard"]:
                    # Hard math: strongly biased to Math Trap at Step 5
                    student.logits[p_idx, 5, prob["trap_fork"]] = 3.0
                    student.logits[p_idx, 11, prob["target_ans"]] = 0.5

        ref_logits = student.logits.detach().clone()
        teacher_logits = ref_logits.clone()
        for p_idx, prob in enumerate(env.problems):
            # Teacher knows correct math
            teacher_logits[p_idx, 5, prob["target_fork"]] += 4.0
            teacher_logits[p_idx, 11, prob["target_ans"]] += 3.0

        optimizer = torch.optim.Adam(student.parameters(), lr=lr)

        math_pass_history = []
        format_pass_history = []
        trap_history = []
        format_hallucination_history = []

        for ep in range(epochs):
            math_corr = 0
            format_corr = 0
            format_hallucinated = 0

            for p_idx, prob in enumerate(env.problems):
                toks, lp = student.sample(p_idx, G)
                r_math_list = []
                r_format_list = []
                for g in range(G):
                    rm, rf, _ = env.evaluate(p_idx, toks[g])
                    r_math_list.append(rm)
                    r_format_list.append(rf)
                    if rf > 0.5 and rm < 0.5:
                        format_hallucinated += 1

                r_math = torch.tensor(r_math_list, dtype=torch.float32)
                r_format = torch.tensor(r_format_list, dtype=torch.float32)

                if r_math.max().item() > 0.5:
                    math_corr += 1
                if r_format.max().item() > 0.5:
                    format_corr += 1

                t_lp = F.log_softmax(teacher_logits[p_idx], dim=-1).unsqueeze(0).expand(G, -1, -1).gather(
                    -1, toks.unsqueeze(-1)
                ).squeeze(-1)
                r_lp = F.log_softmax(ref_logits[p_idx], dim=-1).unsqueeze(0).expand(G, -1, -1).gather(
                    -1, toks.unsqueeze(-1)
                ).squeeze(-1)

                adv = compute_mo_advantages(
                    method=method,
                    r_math=r_math,
                    r_format=r_format,
                    sampled_tokens=toks,
                    sampled_logp=lp,
                    ref_logp=r_lp,
                    teacher_logp=t_lp,
                    env=env,
                ).detach()

                loss = -(lp * adv).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            math_pass = math_corr / env.num_problems
            format_pass = format_corr / env.num_problems
            halluc_rate = format_hallucinated / (env.num_problems * G)
            trap_logits = [student.logits[p_idx, 5, env.problems[p_idx]["trap_fork"]].item() for p_idx in range(env.num_problems // 2)]
            mean_trap_logit = float(np.mean(trap_logits))

            math_pass_history.append(math_pass)
            format_pass_history.append(format_pass)
            trap_history.append(mean_trap_logit)
            format_hallucination_history.append(halluc_rate)

            if (ep + 1) % 5 == 0 or ep == epochs - 1:
                print(
                    f"  Epoch {ep+1:02d}/{epochs:02d} | "
                    f"Math Pass: {math_pass*100:5.1f}% | "
                    f"Format Pass: {format_pass*100:5.1f}% | "
                    f"Format Hallucination: {halluc_rate*100:4.1f}% | "
                    f"Math Trap Logit: {mean_trap_logit:5.2f}"
                )

        all_results[method] = {
            "final_math_pass": math_pass_history[-1],
            "final_format_pass": format_pass_history[-1],
            "final_hallucination_rate": format_hallucination_history[-1],
            "final_trap_logit": trap_history[-1],
            "math_history": math_pass_history,
            "format_history": format_pass_history,
            "hallucination_history": format_hallucination_history,
            "trap_history": trap_history,
        }

    out_file = os.path.join(os.path.dirname(__file__), "mo_flowbalance_results.json")
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n=========================================================================================")
    print(f"MULTI-OBJECTIVE FLOW DECOUPLING BENCHMARK SUMMARY (Theorem 11):")
    print(f"{'Method':<25} | {'Math Pass':<10} | {'Format Pass':<12} | {'Format Hallucination':<22} | {'Math Trap Logit'}")
    print("-" * 90)
    for k, v in all_results.items():
        print(f"{k:<25} | {v['final_math_pass']*100:8.1f}% | {v['final_format_pass']*100:10.1f}% | {v['final_hallucination_rate']*100:20.1f}% | {v['final_trap_logit']:14.2f}")
    print(f"=========================================================================================")


if __name__ == "__main__":
    run_benchmark()
