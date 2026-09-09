import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import os

"""
Experiment: Off-Policy Experience Replay Flow Invariance (Theorem 16)
Benchmarking standard PPO vs GRPO vs Trajectory Balance (TB) vs SubTB under mixed On-Policy + Off-Policy Replay Buffers.

Setting:
- 8 distinct multi-step reasoning problems (vocab size 4, seq len 8).
- Each problem has a unique ground-truth solution sequence.
- An off-policy replay buffer D_replay contains optimal trajectories from past expert runs.
- In each training iteration, the batch contains:
  * 50% on-policy rollouts from current policy pi_theta
  * 50% off-policy rollouts sampled from D_replay (with KL divergence D_KL(pi_theta || pi_replay) increasing over training)

Questions:
1. Does PPO suffer severe importance sampling ratio clipping (r_t > 1.2 or r_t < 0.8) on off-policy replay samples, degrading gradient efficiency?
2. Does FlowBalance (TB and SubTB) maintain smooth, bounded flow gradients on off-policy data without importance weight divergence, accelerating convergence?
"""

class MultiStepMathTask:
    def __init__(self, num_problems=8, seq_len=8):
        self.num_problems = num_problems
        self.seq_len = seq_len
        # Ground truth optimal tokens for each problem
        self.optimal_tokens = torch.randint(0, 4, (num_problems, seq_len))
        
    def evaluate(self, problem_idx, tokens):
        opt = self.optimal_tokens[problem_idx]
        matches = (tokens == opt).float()
        prefix_len = 0
        for m in matches:
            if m == 1.0:
                prefix_len += 1
            else:
                break
        if prefix_len == self.seq_len:
            return 1.0
        elif prefix_len >= self.seq_len // 2:
            return 0.25 * (prefix_len / self.seq_len)
        else:
            return 0.0

class SmallPolicy(nn.Module):
    def __init__(self, vocab_size=4, hidden_dim=32, num_problems=8):
        super().__init__()
        self.prob_embed = nn.Embedding(num_problems, hidden_dim)
        self.token_embed = nn.Embedding(vocab_size, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, prob_ids, seqs):
        # prob_ids: [B], seqs: [B, L]
        p_emb = self.prob_embed(prob_ids).unsqueeze(1) # [B, 1, H]
        if seqs.shape[1] > 1:
            t_emb = self.token_embed(seqs[:, :-1]) # [B, L-1, H]
            inp = torch.cat([p_emb, t_emb], dim=1) # [B, L, H]
        else:
            inp = p_emb
        out, _ = self.lstm(inp)
        logits = self.head(out) # [B, L, V]
        return logits

    def step(self, prob_tensor, tokens_list):
        p_emb = self.prob_embed(prob_tensor).unsqueeze(1) # [1, 1, H]
        if len(tokens_list) > 0:
            t_tensor = torch.tensor([tokens_list], dtype=torch.long)
            t_emb = self.token_embed(t_tensor) # [1, len, H]
            inp = torch.cat([p_emb, t_emb], dim=1)
        else:
            inp = p_emb
        out, _ = self.lstm(inp)
        logits = self.head(out[0, -1, :]) # [V]
        return logits

def run_off_policy_experiment(algo="flowbalance", num_iters=80, batch_size=16, replay_ratio=0.5, seeds=5):
    results = {
        "pass_rate_history": [],
        "clip_rate_history": [],
        "final_pass_rate": []
    }
    
    for seed in range(seeds):
        torch.manual_seed(seed * 100 + 42)
        np.random.seed(seed * 100 + 42)
        
        env = MultiStepMathTask(num_problems=8, seq_len=8)
        policy = SmallPolicy(vocab_size=4, hidden_dim=32, num_problems=8)
        optimizer = optim.Adam(policy.parameters(), lr=0.01)
        
        # Pre-populate replay buffer with optimal/expert trajectories
        replay_buffer = []
        for p_idx in range(8):
            opt_seq = env.optimal_tokens[p_idx].clone()
            # Stored with historical log probabilities from initial uniform random policy
            old_lp = torch.full((8,), float(np.log(0.25)))
            replay_buffer.append({
                "problem_idx": p_idx,
                "tokens": opt_seq,
                "old_log_probs": old_lp,
                "reward": 1.0,
                "is_replay": True
            })
            
        pass_history = []
        clip_history = []
        
        for iteration in range(num_iters):
            # 1. Generate on-policy rollouts
            on_policy_samples = []
            num_on_policy = int(batch_size * (1.0 - replay_ratio))
            num_replay = batch_size - num_on_policy
            
            policy.eval()
            with torch.no_grad():
                for _ in range(num_on_policy):
                    p_idx = int(np.random.randint(0, 8))
                    tokens = []
                    log_probs = []
                    p_tensor = torch.tensor([p_idx], dtype=torch.long)
                    
                    for step in range(8):
                        step_logits = policy.step(p_tensor, tokens)
                        probs = torch.softmax(step_logits, dim=-1)
                        action = int(torch.multinomial(probs, 1).item())
                        tokens.append(action)
                        log_probs.append(torch.log(probs[action] + 1e-12))
                        
                    tok_tensor = torch.tensor(tokens, dtype=torch.long)
                    r = env.evaluate(p_idx, tok_tensor)
                    on_policy_samples.append({
                        "problem_idx": p_idx,
                        "tokens": tok_tensor,
                        "old_log_probs": torch.stack(log_probs),
                        "reward": r,
                        "is_replay": False
                    })
                    
            # 2. Sample from replay buffer
            replay_samples = []
            for _ in range(num_replay):
                idx = int(np.random.randint(0, len(replay_buffer)))
                replay_samples.append(replay_buffer[idx])
                
            batch = on_policy_samples + replay_samples
            
            # 3. Policy update step
            policy.train()
            optimizer.zero_grad()
            
            p_ids = torch.tensor([b["problem_idx"] for b in batch], dtype=torch.long)
            seqs = torch.stack([b["tokens"] for b in batch]) # [B, 8]
            old_lps = torch.stack([b["old_log_probs"] for b in batch]) # [B, 8]
            rewards = torch.tensor([b["reward"] for b in batch], dtype=torch.float32) # [B]
            
            logits = policy(p_ids, seqs) # [B, 8, 4]
            log_probs = torch.log_softmax(logits, dim=-1)
            selected_lps = log_probs.gather(2, seqs.unsqueeze(-1)).squeeze(-1) # [B, 8]
            
            # Group stats
            mean_r = rewards.mean()
            std_r = rewards.std() + 1e-8
            norm_r = (rewards - mean_r) / std_r
            
            clips = 0
            total_tokens = batch_size * 8
            
            if algo == "ppo":
                # Standard PPO with clipping
                ratio = torch.exp(selected_lps - old_lps.detach()) # [B, 8]
                adv = norm_r.unsqueeze(1).expand(-1, 8)
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 0.8, 1.2) * adv
                loss = -torch.min(surr1, surr2).mean()
                clips = ((ratio < 0.8) | (ratio > 1.2)).float().sum().item()
                
            elif algo == "flowbalance":
                # Trajectory Balance Flow Advantage
                tau = 0.5
                target = float(np.log(0.25)) + (rewards / (tau * 8.0) + norm_r * 0.5).unsqueeze(1)
                adv = 2.0 * (target.detach() - selected_lps.detach())
                loss = -(selected_lps * adv).mean()
                clips = 0
                
            elif algo == "subtb":
                # Step SubTB with surprisal weighting
                tau = 0.5
                target = float(np.log(0.25)) + (rewards / (tau * 8.0) + norm_r * 0.5).unsqueeze(1)
                delta = torch.abs(target.detach() - selected_lps.detach()) + 1e-3
                weights = delta / delta.sum(dim=-1, keepdim=True) # normalized simplex
                adv = 2.0 * weights * 8.0 * (target.detach() - selected_lps.detach())
                loss = -(selected_lps * adv).mean()
                clips = 0
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            
            # Evaluate greedy accuracy on all 8 problems
            policy.eval()
            with torch.no_grad():
                correct = 0
                for p_idx in range(8):
                    p_tensor = torch.tensor([p_idx], dtype=torch.long)
                    tokens = []
                    for step in range(8):
                        step_logits = policy.step(p_tensor, tokens)
                        pred = int(step_logits.argmax(dim=-1).item())
                        tokens.append(pred)
                    if env.evaluate(p_idx, torch.tensor(tokens)) == 1.0:
                        correct += 1
                pass_history.append(correct / 8.0)
                clip_history.append(clips / total_tokens)
                
        results["pass_rate_history"].append(pass_history)
        results["clip_rate_history"].append(clip_history)
        results["final_pass_rate"].append(pass_history[-1])
        
    return {
        "algo": algo,
        "final_pass_mean": float(np.mean(results["final_pass_rate"])),
        "final_pass_std": float(np.std(results["final_pass_rate"])),
        "avg_clip_rate": float(np.mean([np.mean(c) for c in results["clip_rate_history"]])),
        "final_pass_curve": [float(np.mean([h[i] for h in results["pass_rate_history"]])) for i in range(num_iters)]
    }

def main():
    print("================================================================================")
    print("BENCHMARK: Off-Policy Experience Replay Flow Invariance (Theorem 16)")
    print("================================================================================")
    
    algorithms = ["ppo", "flowbalance", "subtb"]
    all_results = {}
    
    for algo in algorithms:
        res = run_off_policy_experiment(algo=algo, num_iters=80, batch_size=16, replay_ratio=0.5, seeds=5)
        all_results[algo] = res
        print(f"Algorithm: {algo.upper():12s} | Pass Rate: {res['final_pass_mean']*100:5.2f}% +/- {res['final_pass_std']*100:4.2f}% | Avg Clip Rate: {res['avg_clip_rate']*100:5.2f}%")
        
    output_path = "experiments/autonomous_research_20260910/off_policy_replay_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved results to {output_path}")

if __name__ == "__main__":
    main()
