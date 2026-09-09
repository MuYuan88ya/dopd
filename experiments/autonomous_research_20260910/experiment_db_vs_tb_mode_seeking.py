# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0.
"""Rigorous Empirical Verification of Theorem 8:
The Reference Prior Plateau Theorem: Why Pure Detailed Balance Fails at Mode-Seeking RL
and Trajectory Balance (SubTB) is Necessary for Policy Optimization.

Investigates convergence under biased reference priors:
pi_ref(trap) / pi_ref(clean) in {1, 5, 20, 50}.
Compares:
- Pure Detailed Balance (lambda = 0.0)
- SubTB (lambda = 0.25)
- SubTB (lambda = 0.50)
- Trajectory Balance (lambda = 1.0)
- Standard GRPO
"""

from __future__ import annotations

import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


def simulate_mode_seeking_convergence(bias_ratio: float = 20.0, num_steps: int = 50, lr: float = 0.15):
    """Simulates a decision fork with 1 trap token and 1 clean token.

    Reward: Clean gives R=1.0, Trap gives R=0.0.
    Reference policy: pi_ref(trap) / pi_ref(clean) = bias_ratio.
    """
    vocab_size = 2  # [trap, clean]
    ref_logits = torch.tensor([np.log(bias_ratio), 0.0], dtype=torch.float32)
    ref_lp = F.log_softmax(ref_logits, dim=-1)

    # Teacher knows clean is correct
    teacher_logits = torch.tensor([-2.0, 2.0], dtype=torch.float32)
    teacher_lp = F.log_softmax(teacher_logits, dim=-1)
    delta = teacher_lp - ref_lp

    methods = {
        "Pure_DB_lambda_0.0": 0.0,
        "SubTB_lambda_0.25": 0.25,
        "SubTB_lambda_0.50": 0.50,
        "Pure_TB_lambda_1.0": 1.0,
        "Standard_GRPO": -1.0,
    }

    results = {}
    tau = 0.1
    L = 4  # short sequence length

    for name, lam in methods.items():
        # Initialize student matching reference prior
        logits = ref_logits.clone().detach()
        logits.requires_grad = True
        optimizer = torch.optim.SGD([logits], lr=lr)

        prob_clean_history = []

        for step in range(num_steps):
            probs = F.softmax(logits, dim=-1)
            prob_clean = probs[1].item()
            prob_clean_history.append(prob_clean)

            # Rollout G=16 samples
            G = 16
            sampled = torch.multinomial(probs, num_samples=G, replacement=True)
            scores = (sampled == 1).float()  # R=1.0 if clean, 0.0 if trap
            sampled_lp = F.log_softmax(logits, dim=-1)[sampled]

            if name == "Standard_GRPO":
                s_mean = scores.mean()
                s_std = scores.std()
                adv = (scores - s_mean) / (s_std + 1e-6) if s_std > 1e-6 else (scores - s_mean)
                loss = -(sampled_lp * adv).mean()
            else:
                # SubTB family
                R_term = scores / tau
                # Sequence level target
                G_T = delta.mean()
                seq_old = sampled_lp.mean()
                seq_ref = ref_lp.mean()
                target_seq = seq_ref + 0.5 * G_T + R_term / L
                baseline = (seq_old - target_seq).mean()
                A_TB = 2.0 * (target_seq + baseline - seq_old)

                # Token level target
                w = 1.0 / L
                target_token = ref_lp[sampled] + 0.5 * delta[sampled] + w * L * (R_term / L + baseline)
                A_DB = 2.0 * (target_token - sampled_lp)

                adv = (1.0 - lam) * A_DB + lam * A_TB
                loss = -(sampled_lp * adv).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        results[name] = {
            "initial_prob_clean": prob_clean_history[0],
            "final_prob_clean": prob_clean_history[-1],
            "max_prob_clean": max(prob_clean_history),
            "history": prob_clean_history[::5],
        }

    return results


def run_benchmark():
    bias_ratios = [1.0, 5.0, 20.0, 50.0]
    all_experiments = {}

    print(f"{'Bias Ratio':<12} | {'Method':<22} | {'Init Clean %':<14} | {'Final Clean %':<14} | {'Mode Reached?'}")
    print("=" * 80)

    for b in bias_ratios:
        res = simulate_mode_seeking_convergence(bias_ratio=b, num_steps=60, lr=0.2)
        all_experiments[f"bias_{b}"] = res
        for m, d in res.items():
            mode_ok = "YES (Mode Locked)" if d["final_prob_clean"] > 0.90 else "NO (Plateaued/Trapped)"
            print(f"{b:<12.1f} | {m:<22} | {d['initial_prob_clean']*100:12.1f}% | {d['final_prob_clean']*100:12.1f}% | {mode_ok}")
        print("-" * 80)

    out_path = os.path.join(os.path.dirname(__file__), "mode_seeking_convergence_results.json")
    with open(out_path, "w") as f:
        json.dump(all_experiments, f, indent=2)


if __name__ == "__main__":
    run_benchmark()
