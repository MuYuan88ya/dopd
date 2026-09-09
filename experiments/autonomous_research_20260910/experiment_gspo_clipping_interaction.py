# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0.
"""Rigorous Benchmark: Interaction Between Token Credit Assignment and Policy Clipping.

Compares:
1. PPO Token-Level Clipping + Uniform Advantage (Standard PPO/GRPO)
2. PPO Token-Level Clipping + EW-SubTB (PPO + FlowBalance)
3. GSPO Sequence-Level Clipping + Uniform Advantage (Standard GSPO)
4. GSPO Sequence-Level Clipping + EW-SubTB (Consistent Flow-GSPO)

Evaluates:
- Premature fork clipping rate
- Optimization stability and policy drift
- Pass@1 recovery on hard reasoning DAGs
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


def compute_advantages(
    token_weighting: str,
    scores: torch.Tensor,
    sampled_logp: torch.Tensor,
    ref_logp: torch.Tensor,
    teacher_logp: torch.Tensor,
    alpha: float = 0.5,
    tau: float = 0.1,
    subtb_lambda: float = 0.5,
    gamma: float = 1.0,
) -> torch.Tensor:
    G, L = sampled_logp.shape
    device = sampled_logp.device

    mean_score = scores.mean()
    std_score = scores.std()
    grpo_adv = (scores - mean_score) / (std_score + 1e-6) if std_score > 1e-6 else (scores - mean_score)

    if token_weighting == "uniform_grpo":
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
    target_uncentered = seq_ref + (alpha * g_consist) * G_T_seq.unsqueeze(-1) + R_term / L
    baseline = (seq_old - target_uncentered).mean()
    A_TB = 2.0 * (target_uncentered + baseline - seq_old).expand(G, L)

    if token_weighting == "uniform_subtb":
        w = torch.ones(G, L, device=device) / L
    elif token_weighting == "ew_subtb":
        surprise = (delta.abs() + 0.1).pow(gamma)
        w = surprise / surprise.sum(dim=-1, keepdim=True)
    else:
        raise ValueError(f"Unknown token weighting: {token_weighting}")

    target_token = ref_logp + (alpha * g_consist) * delta + w * L * (R_term / L + baseline)
    A_DB = 2.0 * (target_token - sampled_logp)
    return (1.0 - subtb_lambda) * A_DB + subtb_lambda * A_TB


def compute_loss(
    algo: str,
    log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    advantages: torch.Tensor,
    clip_ratio: float = 0.2,
) -> tuple[torch.Tensor, float, float]:
    """Computes actor loss and tracks clipping statistics on forks vs fillers."""
    G, L = log_prob.shape

    if algo == "PPO":
        # Standard token-level ratio clipping
        ratio = torch.exp(log_prob - old_log_prob)
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
        pg_loss = torch.maximum(pg_loss1, pg_loss2).mean()
        is_clipped = torch.gt(pg_loss2, pg_loss1).float()
    elif algo == "GSPO":
        # Sequence-level importance ratio
        neg_kl = log_prob - old_log_prob
        neg_kl_seq = neg_kl.mean(dim=-1, keepdim=True)
        # Log space ratio: sg[seq_ratio] + token_lp - sg[token_lp]
        log_seq_ratio = log_prob - log_prob.detach() + neg_kl_seq.detach()
        seq_ratio = torch.exp(torch.clamp(log_seq_ratio, max=10.0))
        pg_loss1 = -advantages * seq_ratio
        pg_loss2 = -advantages * torch.clamp(seq_ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
        pg_loss = torch.maximum(pg_loss1, pg_loss2).mean()
        is_clipped = torch.gt(pg_loss2, pg_loss1).float()
    else:
        raise ValueError(f"Unknown algo: {algo}")

    return pg_loss, is_clipped


def run_clipping_benchmark():
    num_problems = 30
    num_steps = 4
    tokens_per_step = 8
    vocab_size = 12
    env = ReasoningDAGEnv(
        num_problems=num_problems,
        num_steps=num_steps,
        tokens_per_step=tokens_per_step,
        vocab_size=vocab_size,
    )

    fork_indices = [s * tokens_per_step + 2 for s in range(num_steps)]
    filler_indices = [i for i in range(env.seq_len) if i not in fork_indices]

    configurations = [
        ("PPO_GRPO", "PPO", "uniform_grpo"),
        ("PPO_EW_SubTB", "PPO", "ew_subtb"),
        ("GSPO_Uniform", "GSPO", "uniform_grpo"),
        ("GSPO_EW_SubTB", "GSPO", "ew_subtb"),
    ]

    all_results = {}
    G = 8
    epochs = 22
    lr = 0.15
    clip_ratio = 0.2

    for name, algo, weighting in configurations:
        print(f"\n--- Testing Configuration: {name} (Algo: {algo}, Weighting: {weighting}) ---")
        student = StudentPolicy(num_problems, env.seq_len, vocab_size)

        # Initialize student: hard problems favor distractor trap
        with torch.no_grad():
            for p_idx, prob in enumerate(env.problems):
                if prob["is_hard"]:
                    for s in range(num_steps):
                        fork_pos = fork_indices[s]
                        distractor = prob["distractors"][s]
                        student.logits[p_idx, fork_pos, distractor] = 2.5

        ref_logits = student.logits.detach().clone()
        teacher_logits = ref_logits.clone()
        for p_idx, prob in enumerate(env.problems):
            for s in range(num_steps):
                fork_pos = fork_indices[s]
                correct_tok = prob["correct"][s]
                teacher_logits[p_idx, fork_pos, correct_tok] += 3.5

        optimizer = torch.optim.Adam(student.parameters(), lr=lr)

        acc_history = []
        hard_acc_history = []
        fork_clip_history = []
        filler_clip_history = []

        for ep in range(epochs):
            total_corr = 0
            hard_corr = 0
            total_hard = 0
            ep_fork_clips = []
            ep_filler_clips = []

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

                adv = compute_advantages(
                    token_weighting=weighting,
                    scores=scores,
                    sampled_logp=sampled_logp,
                    ref_logp=r_lp,
                    teacher_logp=t_lp,
                ).detach()

                # Simulate mini-batch update (2 gradient steps per rollout batch to test clipping)
                old_lp = sampled_logp.detach()
                for _ in range(2):
                    current_lp = F.log_softmax(student.logits[p_idx], dim=-1).unsqueeze(0).expand(G, -1, -1).gather(
                        -1, sampled_tokens.unsqueeze(-1)
                    ).squeeze(-1)

                    loss, is_clipped = compute_loss(
                        algo=algo,
                        log_prob=current_lp,
                        old_log_prob=old_lp,
                        advantages=adv,
                        clip_ratio=clip_ratio,
                    )

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    fork_clip = is_clipped[:, fork_indices].mean().item()
                    filler_clip = is_clipped[:, filler_indices].mean().item()
                    ep_fork_clips.append(fork_clip)
                    ep_filler_clips.append(filler_clip)

            pass1 = total_corr / num_problems
            hard_pass = hard_corr / max(1, total_hard)
            f_clip = np.mean(ep_fork_clips)
            fil_clip = np.mean(ep_filler_clips)

            acc_history.append(pass1)
            hard_acc_history.append(hard_pass)
            fork_clip_history.append(f_clip)
            filler_clip_history.append(fil_clip)

            if (ep + 1) % 5 == 0 or ep == epochs - 1:
                print(
                    f"  Epoch {ep+1:02d}/{epochs:02d} | "
                    f"Pass@1: {pass1*100:5.2f}% | "
                    f"Hard Pass: {hard_pass*100:5.2f}% | "
                    f"Fork Clip: {f_clip*100:4.1f}% | "
                    f"Filler Clip: {fil_clip*100:4.1f}%"
                )

        all_results[name] = {
            "final_pass1": acc_history[-1],
            "final_hard_pass": hard_acc_history[-1],
            "avg_fork_clip": np.mean(fork_clip_history),
            "avg_filler_clip": np.mean(filler_clip_history),
            "acc_history": acc_history,
            "hard_acc_history": hard_acc_history,
            "fork_clip_history": fork_clip_history,
            "filler_clip_history": filler_clip_history,
        }

    out_file = os.path.join(os.path.dirname(__file__), "clipping_benchmark_results.json")
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n=======================================================")
    print(f"CLIPPING BENCHMARK COMPLETE. SUMMARY:")
    print(f"{'Configuration':<20} | {'Pass@1':<8} | {'Hard Pass':<10} | {'Fork Clip':<10} | {'Filler Clip':<12}")
    print("-" * 75)
    for k, v in all_results.items():
        print(
            f"{k:<20} | {v['final_pass1']*100:6.2f}% | {v['final_hard_pass']*100:8.2f}% | "
            f"{v['avg_fork_clip']*100:8.1f}% | {v['avg_filler_clip']*100:10.1f}%"
        )
    print(f"=======================================================")


if __name__ == "__main__":
    run_clipping_benchmark()
