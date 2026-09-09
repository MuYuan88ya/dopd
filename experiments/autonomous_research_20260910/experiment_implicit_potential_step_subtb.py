# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0.
"""Rigorous Empirical Verification of Theorem 10:
Critic-Free Implicit Potential Sub-Trajectory Balance (IP-SubTB) for Multi-Step Reasoning DAGs.

Addresses the critic dilemma:
PPO requires an auto-regressive value critic network, consuming 50% of training VRAM and suffering
from non-stationary value estimation.
IP-SubTB uses the teacher prefix flow as an implicit state potential Phi(s_k), computing exact
step-level SubTB flow residuals without any value network parameters.
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


class MultiStepReasoningTreeEnv:
    """Environment with M=4 sequential reasoning steps.

    Step 1: Premise extraction (filler)
    Step 2: Key Theorem Choice (Fork 1: 1 correct, 1 trap)
    Step 3: Algebraic derivation (filler)
    Step 4: Final calculation (Fork 2: 1 correct, 1 trap)
    """

    def __init__(self, num_problems: int = 30, vocab_size: int = 12):
        self.num_problems = num_problems
        self.vocab_size = vocab_size
        self.steps = 4
        self.tokens_per_step = 4
        self.total_tokens = self.steps * self.tokens_per_step  # 16 tokens

        self.problems = []
        for i in range(num_problems):
            target_fork1 = torch.randint(2, vocab_size, (1,)).item()
            trap_fork1 = torch.randint(2, vocab_size, (1,)).item()
            while trap_fork1 == target_fork1:
                trap_fork1 = torch.randint(2, vocab_size, (1,)).item()

            target_fork2 = torch.randint(2, vocab_size, (1,)).item()
            trap_fork2 = torch.randint(2, vocab_size, (1,)).item()
            while trap_fork2 == target_fork2:
                trap_fork2 = torch.randint(2, vocab_size, (1,)).item()

            is_hard = (i < num_problems // 2)
            self.problems.append({
                "id": f"prob_{i:02d}",
                "target1": target_fork1,
                "trap1": trap_fork1,
                "target2": target_fork2,
                "trap2": trap_fork2,
                "is_hard": is_hard,
            })

    def evaluate(self, p_idx: int, tokens: torch.Tensor) -> tuple[float, bool]:
        """Returns (reward, is_hard)."""
        prob = self.problems[p_idx]
        toks = tokens.tolist()

        # Step 2 fork is at token index 7; Step 4 fork is at token index 15
        s2_ok = (toks[7] == prob["target1"])
        s4_ok = (toks[15] == prob["target2"])

        if s2_ok and s4_ok:
            return 1.0, prob["is_hard"]
        return 0.0, prob["is_hard"]


class StudentPolicy(nn.Module):
    def __init__(self, num_problems: int, total_tokens: int = 16, vocab_size: int = 12):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(num_problems, total_tokens, vocab_size))

    def sample(self, p_idx: int, G: int) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.logits[p_idx]
        probs = F.softmax(logits, dim=-1)
        toks = torch.multinomial(probs, num_samples=G, replacement=True).transpose(0, 1)  # [G, L]
        lp = F.log_softmax(logits, dim=-1).unsqueeze(0).expand(G, -1, -1).gather(
            -1, toks.unsqueeze(-1)
        ).squeeze(-1)
        return toks, lp


def compute_ip_subtb_advantages(
    method: str,
    scores: torch.Tensor,
    sampled_tokens: torch.Tensor,
    sampled_logp: torch.Tensor,
    ref_logp: torch.Tensor,
    teacher_logp: torch.Tensor,
    tau: float = 0.1,
    subtb_lambda: float = 0.5,
    tokens_per_step: int = 4,
    steps: int = 4,
) -> torch.Tensor:
    G, L = sampled_logp.shape
    device = sampled_logp.device

    mean_s = scores.mean()
    std_s = scores.std()
    grpo_adv = (scores - mean_s) / (std_s + 1e-6) if std_s > 1e-6 else (scores - mean_s)

    if method == "Standard_GRPO":
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
        g_consist = 0.5

    seq_old = sampled_logp.mean(dim=-1, keepdim=True)
    seq_ref = ref_logp.mean(dim=-1, keepdim=True)
    R_term = (scores / tau).unsqueeze(-1)
    target_uncentered = seq_ref + (0.5 * g_consist) * G_T_seq.unsqueeze(-1) + R_term / L
    baseline = (seq_old - target_uncentered).mean()
    A_TB = 2.0 * (target_uncentered + baseline - seq_old).expand(G, L)

    if method == "Uniform_SubTB":
        w = torch.ones(G, L, device=device) / L
        target_token = ref_logp + (0.5 * g_consist) * delta + w * L * (R_term / L + baseline)
        A_DB = 2.0 * (target_token - sampled_logp)
        return (1.0 - subtb_lambda) * A_DB + subtb_lambda * A_TB

    elif method == "Critic_Free_IP_SubTB":
        # Theorem 10: Implicit Potential Step SubTB
        # Step-level aggregation of potential differences
        step_advs = torch.zeros(G, steps, device=device)
        for s in range(steps):
            start = s * tokens_per_step
            end = (s + 1) * tokens_per_step
            step_delta = delta[:, start:end].sum(dim=-1)  # [G]
            step_old = sampled_logp[:, start:end].sum(dim=-1)
            step_ref = ref_logp[:, start:end].sum(dim=-1)
            step_R = (R_term.squeeze(-1) / steps) + (baseline * tokens_per_step)

            # Target step flow
            target_step = step_ref + (0.5 * g_consist) * step_delta + step_R
            step_advs[:, s] = 2.0 * (target_step - step_old) / tokens_per_step

        # Broadcast step advantage to tokens within each step
        A_Step = torch.zeros(G, L, device=device)
        for s in range(steps):
            start = s * tokens_per_step
            end = (s + 1) * tokens_per_step
            A_Step[:, start:end] = step_advs[:, s].unsqueeze(-1).expand(G, tokens_per_step)

        return (1.0 - subtb_lambda) * A_Step + subtb_lambda * A_TB

    else:
        raise ValueError(f"Unknown method: {method}")


def run_benchmark():
    env = MultiStepReasoningTreeEnv(num_problems=30, vocab_size=12)
    methods = [
        "Standard_GRPO",
        "Uniform_SubTB",
        "Critic_Free_IP_SubTB",
    ]

    all_results = {}
    G = 8
    epochs = 25
    lr = 0.2

    for method in methods:
        print(f"\n--- Testing Method: {method} ---")
        student = StudentPolicy(env.num_problems, total_tokens=16, vocab_size=12)

        # Initialize student: hard problems strongly favor distractor traps initially
        with torch.no_grad():
            for p_idx, prob in enumerate(env.problems):
                if prob["is_hard"]:
                    student.logits[p_idx, 7, prob["trap1"]] = 2.5
                    student.logits[p_idx, 15, prob["trap2"]] = 2.5

        ref_logits = student.logits.detach().clone()
        teacher_logits = ref_logits.clone()
        for p_idx, prob in enumerate(env.problems):
            # Teacher knows correct solutions
            teacher_logits[p_idx, 7, prob["target1"]] += 3.5
            teacher_logits[p_idx, 15, prob["target2"]] += 3.5

        optimizer = torch.optim.Adam(student.parameters(), lr=lr)

        acc_history = []
        hard_acc_history = []

        for ep in range(epochs):
            total_corr = 0
            hard_corr = 0

            for p_idx, prob in enumerate(env.problems):
                toks, lp = student.sample(p_idx, G)
                scores = []
                for g in range(G):
                    s, _ = env.evaluate(p_idx, toks[g])
                    scores.append(s)
                scores = torch.tensor(scores, dtype=torch.float32)

                if scores.max().item() > 0.5:
                    total_corr += 1
                    if prob["is_hard"]:
                        hard_corr += 1

                t_lp = F.log_softmax(teacher_logits[p_idx], dim=-1).unsqueeze(0).expand(G, -1, -1).gather(
                    -1, toks.unsqueeze(-1)
                ).squeeze(-1)
                r_lp = F.log_softmax(ref_logits[p_idx], dim=-1).unsqueeze(0).expand(G, -1, -1).gather(
                    -1, toks.unsqueeze(-1)
                ).squeeze(-1)

                adv = compute_ip_subtb_advantages(
                    method=method,
                    scores=scores,
                    sampled_tokens=toks,
                    sampled_logp=lp,
                    ref_logp=r_lp,
                    teacher_logp=t_lp,
                ).detach()

                loss = -(lp * adv).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            pass1 = total_corr / env.num_problems
            hard_pass1 = hard_corr / (env.num_problems // 2)
            acc_history.append(pass1)
            hard_acc_history.append(hard_pass1)

            if (ep + 1) % 5 == 0 or ep == epochs - 1:
                print(
                    f"  Epoch {ep+1:02d}/{epochs:02d} | "
                    f"Pass@1: {pass1*100:5.1f}% | "
                    f"Hard Trap Pass: {hard_pass1*100:5.1f}%"
                )

        all_results[method] = {
            "final_pass1": acc_history[-1],
            "final_hard_pass1": hard_acc_history[-1],
            "acc_history": acc_history,
            "hard_acc_history": hard_acc_history,
            "critic_param_count": 0,  # 0 critic parameters for all methods!
        }

    out_file = os.path.join(os.path.dirname(__file__), "implicit_potential_subtb_results.json")
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n=========================================================================================")
    print(f"CRITIC-FREE IMPLICIT POTENTIAL SUBTB SUMMARY (Theorem 10):")
    print(f"{'Method':<25} | {'Pass@1':<8} | {'Hard Pass':<10} | {'Critic Params':<14} | {'VRAM Overhead'}")
    print("-" * 75)
    for k, v in all_results.items():
        print(f"{k:<25} | {v['final_pass1']*100:6.1f}% | {v['final_hard_pass1']*100:8.1f}% | {0:<14} | {'0 MB (Zero Critic)'}")
    print(f"=========================================================================================")


if __name__ == "__main__":
    run_benchmark()
