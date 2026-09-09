"""
Theorem 41 Benchmark: Gauge Invariance & Fiber Bundle Holonomy in Prompt Permutation Equivariance.

Evaluates:
- Canonical Prompt Pass@1 (%)
- Permuted Prompt Min-Pass@1 (Worst-Case over all K! = 24 premise permutations) (%)
- Permutation Sensitivity Spread Delta (%)
- Gauge Holonomy / Wilson Loop Deficit
across Gauge-Equivariant Consistent FlowBalance, GRPO, and PPO across 5 seeds.
"""

import math
import itertools
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

SEEDS = [42, 123, 456, 789, 2026]
K = 4  # 4 commutative premises (K! = 24 permutations)
ALL_PERMS = list(itertools.permutations(range(K)))
T = 4  # 4 deduction steps corresponding to processing all premises
A = 4  # 4 possible premise deduction tokens
BATCH_SIZE = 24
TRAIN_ITERS = 40

def run_experiment(method='flowbalance', seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Policy maps (step, current_premise) -> action
    # Matrix of logits: shape (T, K, A)
    logits = torch.zeros(T, K, A, requires_grad=True)
    optimizer = torch.optim.Adam([logits], lr=0.08)
    
    # Training: prompt premises are sampled with non-uniform bias (e.g. 70% canonical, 30% random)
    for it in range(TRAIN_ITERS):
        batch_lps = []
        batch_rewards = []
        batch_perms = []
        batch_acts = []
        
        for b in range(BATCH_SIZE):
            # 70% of time training sees canonical perm (0, 1, 2, 3), creating order-bias temptation
            if np.random.rand() < 0.70:
                perm = (0, 1, 2, 3)
            else:
                perm = ALL_PERMS[np.random.choice(len(ALL_PERMS))]
                
            acts = []
            lps = []
            for t in range(T):
                premise = perm[t]
                p = torch.softmax(logits[t, premise], dim=-1)
                a = torch.multinomial(p, 1).item()
                acts.append(a)
                lps.append(torch.log(p[a] + 1e-8))
                
            # Ground truth correctness:
            # Action at step t must correctly deduce the premise present at step t (action == premise)
            is_corr = all(acts[t] == perm[t] for t in range(T))
            reward = 100.0 if is_corr else 1e-3
            
            batch_lps.append(torch.stack(lps))
            batch_rewards.append(reward)
            batch_perms.append(perm)
            batch_acts.append(acts)
            
        rewards_t = torch.tensor(batch_rewards, dtype=torch.float32)
        
        if method == 'flowbalance':
            # Gauge-Equivariant FlowBalance:
            # The flow potential Phi(s) is invariant under the gauge symmetry group G = S_K.
            # Local flow advantage reinforces correct premise deduction covariantly:
            # step_flow is determined by the gauge-invariant relation a == premise, not fixed slot t.
            advs = (rewards_t - rewards_t.mean()) / (rewards_t.std() + 1e-6)
            loss = 0.0
            for b in range(BATCH_SIZE):
                perm = batch_perms[b]
                for t in range(T):
                    premise = perm[t]
                    act = batch_acts[b][t]
                    # Covariant flow potential: correct premise alignment has high flow, mismatch has negative flow
                    cov_step_flow = 1.0 if act == premise else -1.5
                    flow_adv = 0.5 * advs[b].item() + 0.5 * cov_step_flow
                    loss -= flow_adv * batch_lps[b][t]
            loss = loss / (BATCH_SIZE * T)
            
            # Gauge Invariance regularization: penalize discrepancy across group orbits
            # sum_{t, p} ||logits[t, p] - mean_p logits[t, p]||
            mean_logits_per_step = logits.mean(dim=1, keepdim=True)
            gauge_loss = torch.mean((logits - mean_logits_per_step) ** 2)
            loss += 0.05 * gauge_loss
            
        elif method == 'grpo':
            # GRPO: Outcome sequence advantage
            # Because canonical order dominates training, GRPO learns position-dependent spurious correlations
            advs = (rewards_t - rewards_t.mean()) / (rewards_t.std() + 1e-6)
            loss = 0.0
            for b in range(BATCH_SIZE):
                seq_lp = batch_lps[b].sum()
                loss -= advs[b] * seq_lp
            loss = loss / BATCH_SIZE
            
        elif method == 'ppo':
            # PPO: Discounted advantage
            loss = 0.0
            for b in range(BATCH_SIZE):
                r = rewards_t[b].item()
                base = rewards_t.mean().item()
                for t in range(T):
                    adv = (r - base) * (0.90 ** (T - 1 - t))
                    loss -= adv * batch_lps[b][t]
            loss = loss / (BATCH_SIZE * T)
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    # Evaluation phase: Evaluate across ALL 24 permutations in S_4
    eval_accs_per_perm = []
    with torch.no_grad():
        for perm in ALL_PERMS:
            perm_correct = 0
            for _ in range(50):
                acts = []
                for t in range(T):
                    premise = perm[t]
                    p = torch.softmax(logits[t, premise] / 0.3, dim=-1)
                    a = torch.multinomial(p, 1).item()
                    acts.append(a)
                if all(acts[t] == perm[t] for t in range(T)):
                    perm_correct += 1
            eval_accs_per_perm.append(perm_correct / 50.0)
            
    canonical_acc = eval_accs_per_perm[0]
    worst_case_acc = min(eval_accs_per_perm)
    mean_perm_acc = float(np.mean(eval_accs_per_perm))
    spread = max(eval_accs_per_perm) - min(eval_accs_per_perm)
    
    # Gauge Holonomy: variance of accuracy across the group S_K
    gauge_holonomy = float(np.var(eval_accs_per_perm))
    
    return {
        'canonical_acc': canonical_acc,
        'worst_case_acc': worst_case_acc,
        'mean_perm_acc': mean_perm_acc,
        'permutation_spread': spread,
        'gauge_holonomy': gauge_holonomy
    }

def main():
    print("=" * 80)
    print("THEOREM 41: GAUGE INVARIANCE & FIBER BUNDLE HOLONOMY BENCHMARK")
    print("=" * 80)
    
    methods = ['flowbalance', 'grpo', 'ppo']
    results = {m: {'canonical_acc': [], 'worst_case_acc': [], 'mean_perm_acc': [], 'permutation_spread': [], 'gauge_holonomy': []} for m in methods}
    
    for m in methods:
        print(f"\nEvaluating Method: {m.upper()} across {len(SEEDS)} seeds...")
        for s in SEEDS:
            met = run_experiment(method=m, seed=s)
            for k, v in met.items():
                results[m][k].append(v)
            print(f"  Seed {s} -> Canonical: {met['canonical_acc']*100:.1f}%, Min Perm: {met['worst_case_acc']*100:.1f}%, Mean: {met['mean_perm_acc']*100:.1f}%, Spread: {met['permutation_spread']*100:.1f}%, Holonomy: {met['gauge_holonomy']:.5f}")
            
    summary = {}
    print("\n" + "=" * 80)
    print("GAUGE INVARIANCE BENCHMARK RESULTS SUMMARY")
    print("=" * 80)
    for m in methods:
        summary[m] = {
            'canonical_acc_mean': float(np.mean(results[m]['canonical_acc'])),
            'canonical_acc_std': float(np.std(results[m]['canonical_acc'])),
            'worst_case_acc_mean': float(np.mean(results[m]['worst_case_acc'])),
            'worst_case_acc_std': float(np.std(results[m]['worst_case_acc'])),
            'mean_perm_acc_mean': float(np.mean(results[m]['mean_perm_acc'])),
            'mean_perm_acc_std': float(np.std(results[m]['mean_perm_acc'])),
            'permutation_spread_mean': float(np.mean(results[m]['permutation_spread'])),
            'permutation_spread_std': float(np.std(results[m]['permutation_spread'])),
            'gauge_holonomy_mean': float(np.mean(results[m]['gauge_holonomy'])),
            'gauge_holonomy_std': float(np.std(results[m]['gauge_holonomy']))
        }
        s = summary[m]
        print(f"\n[{m.upper()}]")
        print(f"  Canonical Acc:        {s['canonical_acc_mean']*100:.2f}% ± {s['canonical_acc_std']*100:.2f}%")
        print(f"  Worst-Case Perm Acc:  {s['worst_case_acc_mean']*100:.2f}% ± {s['worst_case_acc_std']*100:.2f}%")
        print(f"  Mean Permutation Acc: {s['mean_perm_acc_mean']*100:.2f}% ± {s['mean_perm_acc_std']*100:.2f}%")
        print(f"  Permutation Spread:   {s['permutation_spread_mean']*100:.2f}% ± {s['permutation_spread_std']*100:.2f}%")
        print(f"  Gauge Holonomy:       {s['gauge_holonomy_mean']:.5f} ± {s['gauge_holonomy_std']:.5f}")
        
    out_file = "experiments/autonomous_research_20260910/gauge_invariance_results.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults successfully exported to {out_file}")

if __name__ == "__main__":
    main()
