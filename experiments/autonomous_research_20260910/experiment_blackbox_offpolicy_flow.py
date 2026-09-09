"""
Empirical Benchmark: Theorem 23 - Black-Box Off-Policy Flow Invariance (BBO-FlowBalance)
========================================================================================
Demonstrates training a reasoning model on an offline dataset produced by an external
black-box model (where teacher token log-probs are completely unavailable).
Shows how BBO-FlowBalance optimizes reward without importance sampling ratios,
avoids cloning suboptimal fluff (unlike SFT), and converges to optimal reasoning modes.
"""

import math
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)

class BlackBoxDataset:
    """
    Generates an offline dataset of trajectories from an unknown black-box generator:
    - 40% Mode A: Clean optimal proof (Action 0 at fork, Length 4, R = 1.0)
    - 30% Mode B: Suboptimal fluffy proof (Action 1 at fork, Length 12, R = 1.0)
    - 30% Mode C: Flawed hallucination (Action 2 at fork, Length 8, R = 0.0)
    Notice: The dataset provides ONLY the token sequence and the scalar reward R.
    It does NOT provide pi_ext(a|s)!
    """
    def __init__(self, n_samples=600, seed=42):
        np.random.seed(seed)
        self.data = []
        for _ in range(n_samples):
            u = np.random.rand()
            if u < 0.40:
                # Mode A: Optimal clean
                self.data.append({
                    'action': 0,
                    'length': 4,
                    'reward': 1.0,
                    'is_optimal': True
                })
            elif u < 0.70:
                # Mode B: Suboptimal fluffy
                self.data.append({
                    'action': 1,
                    'length': 12,
                    'reward': 1.0,
                    'is_optimal': False
                })
            else:
                # Mode C: Flawed
                self.data.append({
                    'action': 2,
                    'length': 8,
                    'reward': 0.0,
                    'is_optimal': False
                })

    def get_batch(self, batch_size=32):
        indices = np.random.choice(len(self.data), batch_size, replace=False)
        return [self.data[i] for i in indices]

def run_experiment(method='bbo_flow', num_epochs=150, batch_size=32, lr=0.06, seed=42):
    set_seed(seed)
    dataset = BlackBoxDataset(n_samples=600, seed=seed)

    # Student model logits: 3 choices [optimal_mode, fluffy_mode, flawed_mode]
    logits = nn.Parameter(torch.tensor([0.0, 0.0, 0.0], requires_grad=True))
    log_z = nn.Parameter(torch.tensor(0.0, requires_grad=True))
    optimizer = optim.Adam([logits, log_z], lr=lr)

    history = {
        'epoch': [],
        'p_optimal': [],
        'p_fluffy': [],
        'p_flawed': [],
        'expected_length': [],
        'expected_reward': []
    }

    for epoch in range(num_epochs):
        batch = dataset.get_batch(batch_size=batch_size)
        loss = 0.0
        
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log(probs + 1e-8)

        if method == 'sft_filtered':
            # SFT on R=1 demonstrations: maximizes log pi(action) for R=1
            # Does not distinguish optimal from fluffy!
            count = 0
            for b in batch:
                if b['reward'] > 0.5:
                    loss -= log_probs[b['action']]
                    count += 1
            if count > 0:
                loss = loss / count
            else:
                loss = torch.tensor(0.0)

        elif method == 'bbo_flow':
            # Black-Box Off-Policy FlowBalance (BBO-FlowBalance)
            # Trajectory Balance Loss: (log Z + sum log pi_theta - log R)^2
            # Notice: NO teacher probabilities needed!
            # With length-aware reward flow: R_eff = R * exp(-0.05 * length)
            # which penalizes unnecessary filler tokens in Mode B.
            for b in batch:
                act = b['action']
                L = b['length']
                r = b['reward']
                
                # Length-regularized reward flow
                r_eff = r * math.exp(-0.05 * L)
                log_r = math.log(max(r_eff, 1e-4))
                
                # Forward log-prob along trajectory:
                # Fork action log_probs[act] + neutral continuation tokens (assumed balanced)
                log_pf = log_probs[act]
                
                # TB error
                delta = log_z + log_pf - log_r
                loss += 0.5 * (delta ** 2)
            loss = loss / batch_size

        elif method == 'unweighted_tb':
            # Standard unweighted TB without length flow regularization
            for b in batch:
                act = b['action']
                r = b['reward']
                log_r = math.log(max(r, 1e-4))
                log_pf = log_probs[act]
                delta = log_z + log_pf - log_r
                loss += 0.5 * (delta ** 2)
            loss = loss / batch_size

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        p = torch.softmax(logits, dim=-1).detach().numpy()
        exp_len = p[0] * 4 + p[1] * 12 + p[2] * 8
        exp_r = p[0] * 1.0 + p[1] * 1.0 + p[2] * 0.0

        history['epoch'].append(epoch)
        history['p_optimal'].append(float(p[0]))
        history['p_fluffy'].append(float(p[1]))
        history['p_flawed'].append(float(p[2]))
        history['expected_length'].append(float(exp_len))
        history['expected_reward'].append(float(exp_r))

    return {
        'final_p_optimal': float(np.mean(history['p_optimal'][-20:])),
        'final_p_fluffy': float(np.mean(history['p_fluffy'][-20:])),
        'final_p_flawed': float(np.mean(history['p_flawed'][-20:])),
        'final_expected_length': float(np.mean(history['expected_length'][-20:])),
        'final_expected_reward': float(np.mean(history['expected_reward'][-20:])),
        'history': history
    }

def main():
    print("=" * 80)
    print("Theorem 23: Black-Box Off-Policy Flow Invariance (BBO-FlowBalance) Benchmark")
    print("=" * 80)

    seeds = [42, 101, 2024, 777, 999]
    methods = ['sft_filtered', 'unweighted_tb', 'bbo_flow']
    results = {m: {} for m in methods}

    for m in methods:
        print(f"\nEvaluating Method: {m.upper()} across {len(seeds)} seeds...")
        res_list = []
        for s in seeds:
            res = run_experiment(method=m, seed=s)
            res_list.append(res)

        p_opts = [r['final_p_optimal'] for r in res_list]
        p_fluffs = [r['final_p_fluffy'] for r in res_list]
        p_flaws = [r['final_p_flawed'] for r in res_list]
        lengths = [r['final_expected_length'] for r in res_list]
        rewards = [r['final_expected_reward'] for r in res_list]

        results[m] = {
            'optimal_mean': float(np.mean(p_opts)),
            'optimal_std': float(np.std(p_opts)),
            'fluffy_mean': float(np.mean(p_fluffs)),
            'fluffy_std': float(np.std(p_fluffs)),
            'flawed_mean': float(np.mean(p_flaws)),
            'flawed_std': float(np.std(p_flaws)),
            'length_mean': float(np.mean(lengths)),
            'length_std': float(np.std(lengths)),
            'reward_mean': float(np.mean(rewards)),
            'reward_std': float(np.std(rewards))
        }

        print(f"  -> Optimal Clean Rate:      {results[m]['optimal_mean']*100:.2f}% ± {results[m]['optimal_std']*100:.2f}%")
        print(f"  -> Suboptimal Fluffy Rate:  {results[m]['fluffy_mean']*100:.2f}% ± {results[m]['fluffy_std']*100:.2f}%")
        print(f"  -> Flawed Hallucination:    {results[m]['flawed_mean']*100:.2f}% ± {results[m]['flawed_std']*100:.2f}%")
        print(f"  -> Average Output Length:   {results[m]['length_mean']:.2f} tokens")
        print(f"  -> Expected Solution Reward:{results[m]['reward_mean']*100:.2f}%")

    out_file = "experiments/autonomous_research_20260910/blackbox_offpolicy_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved empirical results to {out_file}")

if __name__ == "__main__":
    main()
