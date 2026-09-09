# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0.
"""Rigorous Benchmark: Length Exponent rho and Reasoning Horizon Dynamics.

Investigates the theoretical and empirical impact of length normalization exponent rho in [0, 1]
on credit assignment, gradient scale stability, and length exploitation resistance.
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


class VariableLengthReasoningEnv:
    """Environment where reasoning paths can have varying lengths (short concise vs long detailed).

    Rewards:
    - Correct solution: +1.0
    - Incorrect solution: 0.0
    - Distractor trap penalty: student bias
    - Variable steps: 3 to 6 steps
    """

    def __init__(self, num_problems: int = 30, vocab_size: int = 12):
        self.num_problems = num_problems
        self.vocab_size = vocab_size

        self.problems = []
        for i in range(num_problems):
            # Problem can be solved in 3 steps or 6 steps
            num_steps = 3 if i % 2 == 0 else 6
            tokens_per_step = 8
            seq_len = num_steps * tokens_per_step

            correct = torch.randint(2, vocab_size, (num_steps,))
            distractors = torch.randint(2, vocab_size, (num_steps,))
            for s in range(num_steps):
                while distractors[s] == correct[s]:
                    distractors[s] = torch.randint(2, vocab_size, (1,)).item()
            is_hard = (i < num_problems // 2)
            self.problems.append({
                "id": f"prob_{i:02d}",
                "num_steps": num_steps,
                "tokens_per_step": tokens_per_step,
                "seq_len": seq_len,
                "correct": correct,
                "distractors": distractors,
                "is_hard": is_hard,
            })

    def evaluate(self, p_idx: int, tokens: torch.Tensor) -> float:
        prob = self.problems[p_idx]
        for s in range(prob["num_steps"]):
            fork_pos = s * prob["tokens_per_step"] + 2
            if tokens[fork_pos].item() != prob["correct"][s].item():
                return 0.0
        return 1.0


class VariableStudentPolicy(nn.Module):
    def __init__(self, max_len: int = 48, vocab_size: int = 12):
        super().__init__()
        self.max_len = max_len
        self.vocab_size = vocab_size
        # One parameter per problem for controlled simulation
        self.logits_dict = nn.ParameterDict()

    def get_logits(self, p_id: str, seq_len: int) -> torch.Tensor:
        if p_id not in self.logits_dict:
            self.logits_dict[p_id] = nn.Parameter(torch.zeros(seq_len, self.vocab_size))
        return self.logits_dict[p_id]

    def sample(self, p_id: str, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.get_logits(p_id, seq_len)
        probs = F.softmax(logits, dim=-1)
        toks = torch.multinomial(probs, num_samples=1).squeeze(-1)
        lp = F.log_softmax(logits, dim=-1).gather(-1, toks.unsqueeze(-1)).squeeze(-1)
        return toks, lp


def compute_advantages_with_rho(
    rho: float,
    scores: torch.Tensor,
    sampled_logp: torch.Tensor,
    ref_logp: torch.Tensor,
    teacher_logp: torch.Tensor,
    response_mask: torch.Tensor,
    alpha: float = 0.5,
    tau: float = 0.1,
    subtb_lambda: float = 0.5,
    gamma: float = 1.5,
) -> torch.Tensor:
    G, L = sampled_logp.shape
    device = sampled_logp.device

    lengths = response_mask.sum(dim=-1).clamp(min=1.0)
    length_norm = lengths.pow(rho)

    delta = ((teacher_logp - ref_logp).clamp(-4.0, 4.0)) * response_mask
    G_T_seq = delta.sum(dim=-1) / length_norm

    # AUC Gating
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

    seq_old = (sampled_logp * response_mask).sum(dim=-1, keepdim=True) / length_norm.unsqueeze(-1)
    seq_ref = (ref_logp * response_mask).sum(dim=-1, keepdim=True) / length_norm.unsqueeze(-1)
    R_term = (scores / tau).unsqueeze(-1)

    target_uncentered = seq_ref + (alpha * g_consist) * G_T_seq.unsqueeze(-1) + R_term / length_norm.unsqueeze(-1)
    baseline = (seq_old - target_uncentered).mean()

    # Surprise weighting
    surprise = (delta.abs() + 0.05).pow(gamma) * response_mask
    w = surprise / surprise.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    w_factor = w * lengths.unsqueeze(-1)

    macro_flow = w_factor * (R_term / length_norm.unsqueeze(-1) + baseline)
    target_token = (ref_logp + (alpha * g_consist) * delta + macro_flow) * response_mask

    A_DB = 2.0 * (target_token - sampled_logp) * response_mask
    A_TB = (A_DB.sum(dim=-1, keepdim=True) / lengths.unsqueeze(-1)).expand_as(A_DB) * response_mask

    return (1.0 - subtb_lambda) * A_DB + subtb_lambda * A_TB


def run_rho_sweep():
    env = VariableLengthReasoningEnv(num_problems=30, vocab_size=12)
    rhos = [0.0, 0.5, 0.75, 1.0]

    all_results = {}
    G = 8
    epochs = 20
    lr = 0.15

    for rho in rhos:
        print(f"\n--- Testing Length Exponent rho = {rho} ---")
        student = VariableStudentPolicy(max_len=48, vocab_size=12)

        # Initialize student: hard problems favor distractor trap
        with torch.no_grad():
            for p_idx, prob in enumerate(env.problems):
                logits = student.get_logits(prob["id"], prob["seq_len"])
                if prob["is_hard"]:
                    for s in range(prob["num_steps"]):
                        fork_pos = s * prob["tokens_per_step"] + 2
                        logits[fork_pos, prob["distractors"][s]] = 2.4

        # Clone ref and teacher
        ref_logits = {k: v.detach().clone() for k, v in student.logits_dict.items()}
        teacher_logits = {k: v.clone() for k, v in ref_logits.items()}
        for p_idx, prob in enumerate(env.problems):
            for s in range(prob["num_steps"]):
                fork_pos = s * prob["tokens_per_step"] + 2
                teacher_logits[prob["id"]][fork_pos, prob["correct"][s]] += 3.5

        optimizer = torch.optim.Adam(student.parameters(), lr=lr)

        acc_history = []
        hard_acc_history = []
        short_acc_history = []
        long_acc_history = []
        adv_scale_ratios = []

        for ep in range(epochs):
            total_corr = 0
            hard_corr = 0
            total_hard = 0
            short_corr = 0
            short_total = 0
            long_corr = 0
            long_total = 0
            ep_short_advs = []
            ep_long_advs = []

            for p_idx, prob in enumerate(env.problems):
                p_id = prob["id"]
                seq_len = prob["seq_len"]
                is_short = (prob["num_steps"] == 3)

                if is_short:
                    short_total += 1
                else:
                    long_total += 1

                if prob["is_hard"]:
                    total_hard += 1

                sampled_tokens = []
                sampled_logp = []
                scores = []

                for _ in range(G):
                    toks, lp = student.sample(p_id, seq_len)
                    score = env.evaluate(p_idx, toks)
                    sampled_tokens.append(toks)
                    sampled_logp.append(lp)
                    scores.append(score)

                sampled_tokens = torch.stack(sampled_tokens)
                sampled_logp = torch.stack(sampled_logp)
                scores = torch.tensor(scores, dtype=torch.float32)
                response_mask = torch.ones(G, seq_len)

                if scores.max().item() > 0.5:
                    total_corr += 1
                    if prob["is_hard"]:
                        hard_corr += 1
                    if is_short:
                        short_corr += 1
                    else:
                        long_corr += 1

                t_lp = F.log_softmax(teacher_logits[p_id], dim=-1).unsqueeze(0).expand(G, -1, -1).gather(
                    -1, sampled_tokens.unsqueeze(-1)
                ).squeeze(-1)
                r_lp = F.log_softmax(ref_logits[p_id], dim=-1).unsqueeze(0).expand(G, -1, -1).gather(
                    -1, sampled_tokens.unsqueeze(-1)
                ).squeeze(-1)

                adv = compute_advantages_with_rho(
                    rho=rho,
                    scores=scores,
                    sampled_logp=sampled_logp,
                    ref_logp=r_lp,
                    teacher_logp=t_lp,
                    response_mask=response_mask,
                ).detach()

                adv_mag = adv.abs().mean().item()
                if is_short:
                    ep_short_advs.append(adv_mag)
                else:
                    ep_long_advs.append(adv_mag)

                loss = -(sampled_logp * adv).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            pass1 = total_corr / env.num_problems
            hard_pass = hard_corr / max(1, total_hard)
            short_pass = short_corr / max(1, short_total)
            long_pass = long_corr / max(1, long_total)
            adv_scale_ratio = np.mean(ep_long_advs) / max(1e-6, np.mean(ep_short_advs))

            acc_history.append(pass1)
            hard_acc_history.append(hard_pass)
            short_acc_history.append(short_pass)
            long_acc_history.append(long_pass)
            adv_scale_ratios.append(adv_scale_ratio)

            if (ep + 1) % 5 == 0 or ep == epochs - 1:
                print(
                    f"  Epoch {ep+1:02d}/{epochs:02d} | "
                    f"Pass@1: {pass1*100:5.2f}% | "
                    f"Short Pass: {short_pass*100:5.2f}% | "
                    f"Long Pass: {long_pass*100:5.2f}% | "
                    f"Long/Short Adv Ratio: {adv_scale_ratio:5.2f}x"
                )

        all_results[f"rho_{rho}"] = {
            "final_pass1": acc_history[-1],
            "final_hard_pass": hard_acc_history[-1],
            "final_short_pass": short_acc_history[-1],
            "final_long_pass": long_acc_history[-1],
            "final_adv_ratio": adv_scale_ratios[-1],
            "acc_history": acc_history,
            "hard_acc_history": hard_acc_history,
            "adv_scale_ratios": adv_scale_ratios,
        }

    out_file = os.path.join(os.path.dirname(__file__), "rho_sweep_results.json")
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n=========================================================================================")
    print(f"RHO SWEEP BENCHMARK COMPLETE. SUMMARY:")
    print(f"{'Config':<12} | {'Pass@1':<8} | {'Hard Pass':<10} | {'Short Pass':<12} | {'Long Pass':<12} | {'Adv Ratio (Long/Short)':<22}")
    print("-" * 85)
    for k, v in all_results.items():
        print(
            f"{k:<12} | {v['final_pass1']*100:6.2f}% | {v['final_hard_pass']*100:8.2f}% | "
            f"{v['final_short_pass']*100:10.2f}% | {v['final_long_pass']*100:10.2f}% | "
            f"{v['final_adv_ratio']:18.2f}x"
        )
    print(f"=========================================================================================")


if __name__ == "__main__":
    run_rho_sweep()
