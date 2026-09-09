import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import os

"""
Experiment: Flow-Curiosity Active Exploration in Deep Multi-Step Reasoning (Theorem 17)
Benchmarking standard GRPO, PPO+Entropy, FlowBalance (TB), and Flow-Curiosity FlowBalance (FC-FlowBalance).

Setting:
- A needle-in-a-haystack reasoning task with D=6 consecutive decision forks (each with 4 choices).
- Random exploration chance of solving the problem = (1/4)^6 = 1 / 4096 (~0.024%).
- The model starts with a strong prior distractor trap at Step 1 (leading to dead ends).
- Outcome reward: R = 1.0 ONLY if all 6 steps are correct. Otherwise R = 0.0.

Question:
Can Flow-Residual Variance (Flow-Curiosity) guide the agent to escape the initial trap
and discover the deep reasoning chain when all initial group rollouts yield R = 0?
"""

class DeepReasoningTreeEnv:
    def __init__(self, depth=6, branching=4):
        self.depth = depth
        self.branching = branching
        # Fixed optimal path: e.g. [2, 1, 3, 0, 2, 3]
        self.optimal_path = [2, 1, 3, 0, 2, 3]
        
    def evaluate(self, path):
        if list(path) == self.optimal_path:
            return 1.0
        return 0.0

class BranchingPolicy(nn.Module):
    def __init__(self, depth=6, branching=4):
        super().__init__()
        self.depth = depth
        self.branching = branching
        # Logits for each step: shape [depth, branching]
        # Initialized with a strong distractor trap at step 0 (option 0 has +3.0 logit bias)
        init_logits = torch.zeros(depth, branching)
        init_logits[0, 0] = 3.0 # Trap at step 0 (optimal is 2)
        self.step_logits = nn.Parameter(init_logits)
        
    def get_probs(self):
        return torch.softmax(self.step_logits, dim=-1)

def run_curiosity_experiment(algo="fc_flowbalance", num_epochs=120, group_size=16, lr=0.08, seeds=5):
    results = {
        "discovered": [],
        "epochs_to_discover": [],
        "final_pass_rate": []
    }
    
    for seed in range(seeds):
        torch.manual_seed(seed * 100 + 13)
        np.random.seed(seed * 100 + 13)
        
        env = DeepReasoningTreeEnv(depth=6, branching=4)
        policy = BranchingPolicy(depth=6, branching=4)
        optimizer = optim.Adam(policy.parameters(), lr=lr)
        
        discovered = False
        discovery_epoch = num_epochs
        
        for epoch in range(num_epochs):
            probs = policy.get_probs() # [6, 4]
            
            # Rollout group_size trajectories
            trajectories = []
            for _ in range(group_size):
                path = []
                log_probs = []
                for d in range(env.depth):
                    action = int(torch.multinomial(probs[d], 1).item())
                    path.append(action)
                    log_probs.append(torch.log(probs[d, action] + 1e-12))
                r = env.evaluate(path)
                if r == 1.0 and not discovered:
                    discovered = True
                    discovery_epoch = epoch
                trajectories.append({
                    "path": path,
                    "log_probs": torch.stack(log_probs),
                    "reward": r
                })
                
            rewards = [t["reward"] for t in trajectories]
            mean_r = np.mean(rewards)
            std_r = np.std(rewards) + 1e-8
            
            optimizer.zero_grad()
            loss = 0.0
            
            if algo == "grpo":
                # Standard GRPO
                for t in trajectories:
                    adv = (t["reward"] - mean_r) / std_r
                    loss -= adv * t["log_probs"].sum()
                loss = loss / group_size
                
            elif algo == "ppo_entropy":
                # PPO with Entropy Bonus
                entropy = -(probs * torch.log(probs + 1e-12)).sum()
                for t in trajectories:
                    adv = (t["reward"] - mean_r) / std_r
                    loss -= adv * t["log_probs"].sum()
                loss = (loss / group_size) - 0.05 * entropy
                
            elif algo == "flowbalance":
                # Standard Trajectory Balance Flow Advantage
                tau = 0.5
                uniform_lp = float(np.log(0.25))
                for t in trajectories:
                    target = uniform_lp + (t["reward"] / (tau * 6.0) + (t["reward"] - mean_r) / std_r * 0.5)
                    adv = 2.0 * (target - t["log_probs"].sum() / 6.0)
                    loss -= adv * t["log_probs"].sum()
                loss = loss / group_size
                
            elif algo == "fc_flowbalance":
                # Flow-Curiosity FlowBalance:
                # 1. Compute flow residual for each trajectory and step
                # delta_{i, d} = log pi_theta(a_{i, d}) - log pi_ref(a_{i, d})
                # Epistemic flow variance across the group for step d:
                stacked_lps = torch.stack([t["log_probs"] for t in trajectories]) # [G, 6]
                step_variance = torch.var(stacked_lps, dim=0) # [6]
                
                # Flow-Curiosity bonus: states with high decision variance get exploration encouragement
                tau = 0.5
                uniform_lp = float(np.log(0.25))
                
                for i, t in enumerate(trajectories):
                    # Intrinsic curiosity reward based on step variance
                    curiosity_bonus = 0.3 * step_variance.sum().item()
                    total_r = t["reward"] + curiosity_bonus
                    
                    # Simplex weighting based on local variance
                    w = (step_variance + 1e-3) / (step_variance.sum() + 6e-3)
                    target = uniform_lp + w * 6.0 * (total_r / (tau * 6.0) + (t["reward"] - mean_r) / std_r * 0.5)
                    adv = 2.0 * (target.detach() - t["log_probs"])
                    loss -= (adv * t["log_probs"]).sum()
                loss = loss / group_size
                
            elif algo == "adaptive_fc_flowbalance":
                # Adaptive Phase-Switching Flow-Curiosity FlowBalance:
                stacked_lps = torch.stack([t["log_probs"] for t in trajectories]) # [G, 6]
                step_variance = torch.var(stacked_lps, dim=0) # [6]
                
                tau = 0.5
                uniform_lp = float(np.log(0.25))
                has_success = (max(rewards) > 0.0)
                
                for i, t in enumerate(trajectories):
                    # Phase switch: when success is discovered, exploration collapses to 0 for instant mode locking
                    curiosity_bonus = 0.0 if has_success else 0.4 * step_variance.sum().item()
                    total_r = t["reward"] + curiosity_bonus
                    
                    w = (step_variance + 1e-3) / (step_variance.sum() + 6e-3)
                    target = uniform_lp + w * 6.0 * (total_r / (tau * 6.0) + (t["reward"] - mean_r) / std_r * 0.5)
                    adv = 2.0 * (target.detach() - t["log_probs"])
                    loss -= (adv * t["log_probs"]).sum()
                loss = loss / group_size
                
            loss.backward()
            optimizer.step()
            
        # Final pass rate: probability of generating optimal path under greedy / current policy
        final_probs = policy.get_probs()
        greedy_path = [int(final_probs[d].argmax().item()) for d in range(env.depth)]
        final_pass = 1.0 if greedy_path == env.optimal_path else 0.0
        
        results["discovered"].append(1.0 if discovered else 0.0)
        results["epochs_to_discover"].append(discovery_epoch)
        results["final_pass_rate"].append(final_pass)
        
    return {
        "algo": algo,
        "discovery_rate": float(np.mean(results["discovered"])),
        "avg_discovery_epoch": float(np.mean(results["epochs_to_discover"])),
        "final_pass_rate": float(np.mean(results["final_pass_rate"]))
    }

def main():
    print("================================================================================")
    print("BENCHMARK: Flow-Curiosity Active Exploration in Deep Multi-Step Reasoning (Theorem 17)")
    print("================================================================================")
    
    algorithms = ["grpo", "ppo_entropy", "flowbalance", "fc_flowbalance", "adaptive_fc_flowbalance"]
    all_results = {}
    
    for algo in algorithms:
        res = run_curiosity_experiment(algo=algo, num_epochs=120, group_size=16, lr=0.08, seeds=6)
        all_results[algo] = res
        print(f"Algorithm: {algo.upper():24s} | Discovery Rate: {res['discovery_rate']*100:5.2f}% | Avg Discovery Epoch: {res['avg_discovery_epoch']:5.1f} | Final Pass: {res['final_pass_rate']*100:5.2f}%")
        
    output_path = "experiments/autonomous_research_20260910/flow_curiosity_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved results to {output_path}")

if __name__ == "__main__":
    main()
