import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import os

"""
Experiment: Heterogeneous Multi-Teacher Consensus & Hallucination Isolation (Theorem 19)
Benchmarking Uniform Teacher Averaging vs Majority Voting vs Bayesian Concordance Consensus (C-FlowBalance Multi-Teacher).

Setting:
- A reasoning domain with 2 problem categories: Algebra (60%) and Geometry (40%).
- 3 Teachers:
  * Teacher 1 (Oracle / Code Verifier): Sparse (only active on 30% of problems), 100% accurate.
  * Teacher 2 (Domain-Specific LLM): 85% accurate on Algebra, 20% on Geometry (hallucinates on geometry).
  * Teacher 3 (Adversarial / Hallucinating): 25% accurate across all problems (toxic).

Student is initialized with 0% initial knowledge (uniform random policy over 4 choices across 4 steps).
Can C-FlowBalance Multi-Teacher dynamically discover domain reliability, mute toxic guidance,
and achieve high pass rates on BOTH Algebra and Geometry?
"""

class MultiDomainMathEnv:
    def __init__(self):
        # 10 problems: 6 Algebra (0..5), 4 Geometry (6..9)
        self.num_problems = 10
        self.problem_types = ["algebra"] * 6 + ["geometry"] * 4
        # Optimal paths (4 steps, 4 choices each)
        self.optimal_paths = [
            [1, 2, 0, 3], [0, 3, 1, 2], [2, 1, 3, 0], [3, 0, 2, 1], [1, 1, 2, 2], [0, 2, 1, 3], # Algebra
            [2, 3, 0, 1], [3, 1, 2, 0], [1, 0, 3, 2], [0, 1, 2, 3]                               # Geometry
        ]
        
    def evaluate(self, prob_id, path):
        return 1.0 if list(path) == self.optimal_paths[prob_id] else 0.0

def get_teacher_recommendation(teacher_id, prob_id, env):
    opt = env.optimal_paths[prob_id]
    p_type = env.problem_types[prob_id]
    
    if teacher_id == 1: # Oracle: active only on 30% of problems
        if prob_id in [0, 3, 7]:
            return opt, True
        return None, False
    elif teacher_id == 2: # Domain LLM: 85% on Algebra, 20% on Geometry
        p_correct = 0.85 if p_type == "algebra" else 0.20
        if np.random.rand() < p_correct:
            return opt, True
        else:
            # Toxic hallucination: wrong step at fork 1
            hallucinated = opt.copy()
            hallucinated[1] = (opt[1] + 1) % 4
            return hallucinated, True
    elif teacher_id == 3: # Toxic Teacher: 25% correct (essentially random/misleading)
        if np.random.rand() < 0.25:
            return opt, True
        else:
            hallucinated = [(x + 2) % 4 for x in opt]
            return hallucinated, True

class StudentPolicy(nn.Module):
    def __init__(self, num_problems=10, depth=4, branching=4):
        super().__init__()
        # Parameter tensor [num_problems, depth, branching]
        self.logits = nn.Parameter(torch.zeros(num_problems, depth, branching))
        
    def get_probs(self, prob_id):
        return torch.softmax(self.logits[prob_id], dim=-1)

def run_experiment(algo="consensus_flowbalance", num_epochs=90, group_size=16, lr=0.08, seeds=5):
    results = {
        "algebra_acc": [],
        "geometry_acc": [],
        "overall_acc": [],
        "poison_rate": []
    }
    
    for seed in range(seeds):
        torch.manual_seed(seed * 100 + 42)
        np.random.seed(seed * 100 + 42)
        
        env = MultiDomainMathEnv()
        student = StudentPolicy()
        optimizer = optim.Adam(student.parameters(), lr=lr)
        
        # Historical EMA AUC per teacher and per domain
        ema_auc = {
            1: {"algebra": 0.5, "geometry": 0.5},
            2: {"algebra": 0.5, "geometry": 0.5},
            3: {"algebra": 0.5, "geometry": 0.5},
        }
        
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            batch_loss = 0.0
            
            for p_id in range(env.num_problems):
                p_type = env.problem_types[p_id]
                probs = student.get_probs(p_id) # [4, 4]
                
                # Rollout group
                trajectories = []
                for _ in range(group_size):
                    path = []
                    log_probs = []
                    for step in range(4):
                        action = int(torch.multinomial(probs[step], 1).item())
                        path.append(action)
                        log_probs.append(torch.log(probs[step, action] + 1e-12))
                    r = env.evaluate(p_id, path)
                    trajectories.append({
                        "path": path,
                        "log_probs": torch.stack(log_probs),
                        "reward": r
                    })
                    
                rewards = [t["reward"] for t in trajectories]
                mean_r = np.mean(rewards)
                std_r = np.std(rewards) + 1e-8
                
                # Fetch teacher paths
                t_paths = {}
                for t_id in [1, 2, 3]:
                    path_rec, active = get_teacher_recommendation(t_id, p_id, env)
                    if active:
                        t_paths[t_id] = path_rec
                        
                # Update teacher AUC based on current group if reward variance exists
                if std_r > 1e-4:
                    for t_id, rec in t_paths.items():
                        # Student paths concordant with teacher?
                        concordance = [1.0 if t["path"] == rec else 0.0 for t in trajectories]
                        # Pairwise AUC with outcome rewards
                        pos_pairs = 0
                        concordant_pairs = 0
                        for i in range(group_size):
                            for j in range(group_size):
                                if rewards[i] > rewards[j]:
                                    pos_pairs += 1
                                    if concordance[i] > concordance[j]:
                                        concordant_pairs += 1
                                    elif concordance[i] == concordance[j]:
                                        concordant_pairs += 0.5
                        if pos_pairs > 0:
                            emp_auc = concordant_pairs / pos_pairs
                            # Update EMA
                            cur = ema_auc[t_id][p_type]
                            ema_auc[t_id][p_type] = 0.8 * cur + 0.2 * emp_auc
                            
                # Compute loss
                loss = 0.0
                tau = 0.5
                uniform_lp = float(np.log(0.25))
                
                if algo == "uniform_teacher_avg":
                    # Simple average of all available teacher paths
                    # Gives equal weight 1/3 to Teacher 1, 2, and 3
                    for t in trajectories:
                        adv = (t["reward"] - mean_r) / std_r
                        # Teacher delta guidance
                        t_guidance = torch.zeros(4)
                        for t_id, rec in t_paths.items():
                            for step in range(4):
                                if t["path"][step] == rec[step]:
                                    t_guidance[step] += 1.0 / len(t_paths)
                        total_adv = adv + 0.5 * t_guidance.sum().item()
                        loss -= total_adv * t["log_probs"].sum()
                        
                elif algo == "majority_voting":
                    # Majority vote across active teachers
                    vote_rec = []
                    for step in range(4):
                        choices = [rec[step] for rec in t_paths.values()]
                        majority_action = max(set(choices), key=choices.count)
                        vote_rec.append(majority_action)
                    for t in trajectories:
                        adv = (t["reward"] - mean_r) / std_r
                        t_match = sum(1.0 for step in range(4) if t["path"][step] == vote_rec[step])
                        total_adv = adv + 0.5 * (t_match / 4.0)
                        loss -= total_adv * t["log_probs"].sum()
                        
                elif algo == "consensus_flowbalance":
                    # Bayesian Concordance Consensus (Theorem 19):
                    # Weight each teacher strictly by its domain-specific concordance gate g_consist = max(0, 2*(AUC - 0.5))
                    g_weights = {}
                    for t_id in t_paths.keys():
                        auc_val = ema_auc[t_id][p_type]
                        g_weights[t_id] = max(0.0, 2.0 * (auc_val - 0.5))
                    total_g = sum(g_weights.values())
                    
                    for t in trajectories:
                        adv_tb = (t["reward"] - mean_r) / std_r
                        # Token-level consensus flow delta
                        step_deltas = torch.zeros(4)
                        if total_g > 1e-4:
                            for t_id, rec in t_paths.items():
                                weight = g_weights[t_id] / total_g
                                for step in range(4):
                                    if t["path"][step] == rec[step]:
                                        step_deltas[step] += weight * 1.5
                                    else:
                                        step_deltas[step] -= weight * 0.5
                                        
                        # SubTB advantage strictly combining outcome and consensus teacher flow
                        token_adv = 0.5 * adv_tb + 0.5 * step_deltas
                        loss -= (token_adv * t["log_probs"]).sum()
                        
                batch_loss += (loss / group_size)
                
            batch_loss = batch_loss / env.num_problems
            batch_loss.backward()
            optimizer.step()
            
        # Evaluation
        alg_correct = 0
        geom_correct = 0
        poisoned = 0
        
        for p_id in range(env.num_problems):
            probs = student.get_probs(p_id)
            greedy_path = [int(probs[s].argmax().item()) for s in range(4)]
            is_correct = (greedy_path == env.optimal_paths[p_id])
            if env.problem_types[p_id] == "algebra":
                if is_correct: alg_correct += 1
            else:
                if is_correct: geom_correct += 1
                
            # Check if student fell into Teacher 3's toxic hallucination
            toxic_path = [(x + 2) % 4 for x in env.optimal_paths[p_id]]
            if greedy_path == toxic_path:
                poisoned += 1
                
        results["algebra_acc"].append(alg_correct / 6.0)
        results["geometry_acc"].append(geom_correct / 4.0)
        results["overall_acc"].append((alg_correct + geom_correct) / 10.0)
        results["poison_rate"].append(poisoned / 10.0)
        
    return {
        "algo": algo,
        "overall_acc": float(np.mean(results["overall_acc"])),
        "algebra_acc": float(np.mean(results["algebra_acc"])),
        "geometry_acc": float(np.mean(results["geometry_acc"])),
        "poison_rate": float(np.mean(results["poison_rate"]))
    }

def main():
    print("================================================================================")
    print("BENCHMARK: Heterogeneous Multi-Teacher Consensus & Hallucination Isolation (Theorem 19)")
    print("================================================================================")
    
    algorithms = ["uniform_teacher_avg", "majority_voting", "consensus_flowbalance"]
    all_results = {}
    
    for algo in algorithms:
        res = run_experiment(algo=algo, num_epochs=90, group_size=16, lr=0.08, seeds=6)
        all_results[algo] = res
        print(f"Algorithm: {algo.upper():24s} | Overall Acc: {res['overall_acc']*100:5.2f}% | Algebra: {res['algebra_acc']*100:5.2f}% | Geometry: {res['geometry_acc']*100:5.2f}% | Poison Rate: {res['poison_rate']*100:5.2f}%")
        
    output_path = "experiments/autonomous_research_20260910/multi_teacher_consensus_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved results to {output_path}")

if __name__ == "__main__":
    main()
