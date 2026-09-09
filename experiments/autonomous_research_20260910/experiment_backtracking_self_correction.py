# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0.
"""Rigorous Benchmark: Backtracking and Self-Correction Credit Assignment in FlowBalance.

Addresses the 'False Positive' dilemma in reasoning traces with self-correction:
When a model explores a dead-end, backtracks, and eventually solves the problem,
standard GRPO rewards the dead-end tokens equally with the solution.
Backtracking-Aware FlowBalance reallocates macro flow to the rescue branch
while attenuating credit on the abandoned dead-end branch.
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


class BacktrackingDAGEnv:
    """Reasoning environment where trajectories can contain:

    - Path A: Direct correct derivation (8 tokens, correct)
    - Path B: Dead-end branch (8 tokens, wrong) followed by Backtracking token (token 1)
              and a Rescue branch (8 tokens, correct). Total length = 17 tokens.
    - Path C: Complete failure (dead-end without recovery).
    """

    def __init__(self, num_problems: int = 30, vocab_size: int = 14):
        self.num_problems = num_problems
        self.vocab_size = vocab_size

        self.problems = []
        for i in range(num_problems):
            # Target solution token
            target_token = torch.randint(3, vocab_size, (1,)).item()
            trap_token = torch.randint(3, vocab_size, (1,)).item()
            while trap_token == target_token:
                trap_token = torch.randint(3, vocab_size, (1,)).item()

            is_hard = (i < num_problems // 2)
            self.problems.append({
                "id": f"prob_{i:02d}",
                "target": target_token,
                "trap": trap_token,
                "is_hard": is_hard,
            })

    def evaluate(self, p_idx: int, tokens: torch.Tensor) -> tuple[float, bool]:
        """Returns (reward, has_backtracking).

        Backtracking token is token id 1 ('Wait/Undo').
        """
        prob = self.problems[p_idx]
        tokens_list = tokens.tolist()

        if 1 in tokens_list:
            # Contains backtracking token
            bt_pos = tokens_list.index(1)
            # Inspect token after backtracking
            if bt_pos + 1 < len(tokens_list):
                if tokens_list[-1] == prob["target"]:
                    return 1.0, True
            return 0.0, True
        else:
            # Direct attempt
            if tokens_list[-1] == prob["target"]:
                return 1.0, False
            return 0.0, False


class BacktrackingStudentPolicy(nn.Module):
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


def compute_advantages_backtracking(
    method: str,
    scores: torch.Tensor,
    sampled_tokens: torch.Tensor,
    sampled_logp: torch.Tensor,
    ref_logp: torch.Tensor,
    teacher_logp: torch.Tensor,
    tau: float = 0.1,
    subtb_lambda: float = 0.5,
    gamma: float = 1.5,
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
    elif method == "EW_SubTB":
        surprise = (delta.abs() + 0.05).pow(gamma)
        w = surprise / surprise.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    elif method == "Backtracking_Aware_FlowBalance":
        # Attenuate credit on tokens before the backtracking marker (token id 1)
        raw_surprise = (delta.abs() + 0.05).pow(gamma)
        w_mask = torch.ones_like(raw_surprise)
        for g in range(G):
            toks_g = sampled_tokens[g].tolist()
            if 1 in toks_g:
                bt_idx = toks_g.index(1)
                # Attenuate dead-end tokens before backtracking marker to 5% mass
                w_mask[g, :bt_idx] = 0.05

        mod_surprise = raw_surprise * w_mask
        w = mod_surprise / mod_surprise.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    else:
        raise ValueError(f"Unknown method: {method}")

    target_token = ref_logp + (0.5 * g_consist) * delta + w * L * (R_term / L + baseline)
    A_DB = 2.0 * (target_token - sampled_logp)

    return (1.0 - subtb_lambda) * A_DB + subtb_lambda * A_TB


def run_backtracking_experiment():
    env = BacktrackingDAGEnv(num_problems=30, vocab_size=14)
    methods = [
        "Standard_GRPO",
        "Uniform_SubTB",
        "EW_SubTB",
        "Backtracking_Aware_FlowBalance",
    ]

    all_results = {}
    G = 8
    epochs = 25
    lr = 0.15

    for method in methods:
        print(f"\n--- Testing Method: {method} ---")
        student = BacktrackingStudentPolicy(env.num_problems, seq_len=16, vocab_size=14)

        # Initialize student: hard problems strongly favor distractor trap initially
        with torch.no_grad():
            for p_idx, prob in enumerate(env.problems):
                if prob["is_hard"]:
                    # Trap token at step 1
                    student.logits[p_idx, 3, prob["trap"]] = 2.2
                    # Backtracking token (1) at step 2
                    student.logits[p_idx, 7, 1] = 1.5
                    # Target token at step 3
                    student.logits[p_idx, 15, prob["target"]] = 0.5

        ref_logits = student.logits.detach().clone()
        teacher_logits = ref_logits.clone()
        for p_idx, prob in enumerate(env.problems):
            # Teacher knows correct target and does not need trap
            teacher_logits[p_idx, 15, prob["target"]] += 3.5

        optimizer = torch.optim.Adam(student.parameters(), lr=lr)

        acc_history = []
        dead_end_persistence = []  # Logit on trap token

        for ep in range(epochs):
            total_corr = 0

            for p_idx, prob in enumerate(env.problems):
                toks, lp = student.sample(p_idx, G)
                scores = []
                for g in range(G):
                    s, _ = env.evaluate(p_idx, toks[g])
                    scores.append(s)
                scores = torch.tensor(scores, dtype=torch.float32)

                if scores.max().item() > 0.5:
                    total_corr += 1

                teacher_lp_all = F.log_softmax(teacher_logits[p_idx], dim=-1).unsqueeze(0).expand(G, -1, -1)
                t_lp = teacher_lp_all.gather(-1, toks.unsqueeze(-1)).squeeze(-1)

                ref_lp_all = F.log_softmax(ref_logits[p_idx], dim=-1).unsqueeze(0).expand(G, -1, -1)
                r_lp = ref_lp_all.gather(-1, toks.unsqueeze(-1)).squeeze(-1)

                adv = compute_advantages_backtracking(
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
            acc_history.append(pass1)

            # Measure average logit on the trap token across hard problems
            trap_logits = [student.logits[p_idx, 3, env.problems[p_idx]["trap"]].item() for p_idx in range(env.num_problems // 2)]
            dead_end_persistence.append(float(np.mean(trap_logits)))

            if (ep + 1) % 5 == 0 or ep == epochs - 1:
                print(
                    f"  Epoch {ep+1:02d}/{epochs:02d} | "
                    f"Pass@1: {pass1*100:5.2f}% | "
                    f"Trap Logit Persistence: {dead_end_persistence[-1]:5.2f}"
                )

        all_results[method] = {
            "final_pass1": acc_history[-1],
            "final_trap_persistence": dead_end_persistence[-1],
            "acc_history": acc_history,
            "trap_history": dead_end_persistence,
        }

    out_file = os.path.join(os.path.dirname(__file__), "backtracking_results.json")
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n=========================================================================================")
    print(f"BACKTRACKING CREDIT REALLOCATION SUMMARY:")
    print(f"{'Method':<35} | {'Pass@1':<8} | {'Trap Logit Persistence':<22}")
    print("-" * 75)
    for k, v in all_results.items():
        print(f"{k:<35} | {v['final_pass1']*100:6.2f}% | {v['final_trap_persistence']:20.2f}")
    print(f"=========================================================================================")


if __name__ == "__main__":
    run_backtracking_experiment()
