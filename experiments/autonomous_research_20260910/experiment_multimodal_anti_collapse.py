"""
Empirical Benchmark: Theorem 26 - Multi-Mode Coverage & Anti-Collapse Invariance
================================================================================
Demonstrates how standard RL (GRPO / PPO) suffers from Mode Collapse, driving minor
valid reasoning modes to extinction, whereas FlowBalance intrinsically generates a
restorative self-balancing force across equally valid reasoning modes (Mode 1, 2, 3),
converging to maximum reasoning entropy while completely suppressing errors.
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

class MultiModalEnv:
    """
    Reasoning environment with 3 equally valid derivation modes:
    - Mode 0: Algebraic proof (R = 1.0)
    - Mode 1: Geometric proof (R = 1.0)
    - Mode 2: Inductive proof (R = 1.0)
    - Mode 3: Flawed shortcut trap (R = 0.0)
    """
    def __init__(self):
        pass

    def rollout(self, logits):
        p = torch.softmax(logits, dim=-1)
        action = torch.multinomial(p, 1).item()
        reward = 1.0 if action in [0, 1, 2] else 0.0
        return {
            'action': action,
            'reward': reward,
            'log_prob': torch.log(p[action] + 1e-8),
            'is_valid_mode': (action in [0, 1, 2])
        }

def run_experiment(method='flow_tb', num_epochs=160, batch_size=32, lr=0.06, seed=42):
    set_seed(seed)
    env = MultiModalEnv()

    # Initial logits: Mode 0 dominates (1.0), Mode 1 (0.0), Mode 2 (-1.0), Trap (0.0)
    # Probs: ~55% Mode 0, ~20% Mode 1, ~7% Mode 2, ~20% Trap
    logits = nn.Parameter(torch.tensor([1.0, 0.0, -1.0, 0.0], requires_grad=True))
    log_z = nn.Parameter(torch.tensor(0.0, requires_grad=True))
    optimizer = optim.Adam([logits, log_z] if method == 'flow_tb' else [logits], lr=lr)

    history = {
        'epoch': [],
        'p0': [],
        'p1': [],
        'p2': [],
        'p_trap': [],
        'mode_entropy': [],
        'overall_acc': []
    }

    for epoch in range(num_epochs):
        batch = [env.rollout(logits) for _ in range(batch_size)]
        rewards = torch.tensor([b['reward'] for b in batch], dtype=torch.float32)

        loss = 0.0
        p = torch.softmax(logits, dim=-1)

        if method == 'grpo':
            # GRPO: Outcome scalar advantage
            r_mean = rewards.mean()
            r_std = rewards.std() + 1e-6
            advs = (rewards - r_mean) / r_std
            for i, b in enumerate(batch):
                loss -= advs[i] * b['log_prob']
            loss = loss / batch_size

        elif method == 'ppo_entropy':
            # PPO with entropy regularization (beta = 0.05)
            r_mean = rewards.mean()
            r_std = rewards.std() + 1e-6
            advs = (rewards - r_mean) / r_std
            loss_actor = 0.0
            for i, b in enumerate(batch):
                loss_actor -= advs[i] * b['log_prob']
            loss_actor = loss_actor / batch_size
            entropy = -torch.sum(p * torch.log(p + 1e-8))
            loss = loss_actor - 0.05 * entropy

        elif method == 'flow_tb':
            # FlowBalance (TB): (log Z + log pi(y) - log R)^2
            # When R = 1 for all valid modes, delta = log Z + log pi(y).
            # Overrepresented modes have log pi > log Z -> positive loss gradient reduces them!
            # Underrepresented modes have log pi < log Z -> negative loss gradient increases them!
            # Automatically converges to uniform pi(m) = 1/3!
            for b in batch:
                r = b['reward']
                log_r = math.log(max(r, 1e-4))
                delta = log_z + b['log_prob'] - log_r
                loss += 0.5 * (delta ** 2)
            loss = loss / batch_size

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        p_det = torch.softmax(logits, dim=-1).detach().numpy()
        # Mode entropy over the 3 valid modes
        valid_mass = p_det[0] + p_det[1] + p_det[2]
        if valid_mass > 1e-6:
            p_valid = p_det[:3] / valid_mass
            h = -np.sum(p_valid * np.log(p_valid + 1e-8))
        else:
            h = 0.0

        history['epoch'].append(epoch)
        history['p0'].append(float(p_det[0]))
        history['p1'].append(float(p_det[1]))
        history['p2'].append(float(p_det[2]))
        history['p_trap'].append(float(p_det[3]))
        history['mode_entropy'].append(float(h))
        history['overall_acc'].append(float(valid_mass))

    return {
        'final_p0': float(np.mean(history['p0'][-20:])),
        'final_p1': float(np.mean(history['p1'][-20:])),
        'final_p2': float(np.mean(history['p2'][-20:])),
        'final_trap': float(np.mean(history['p_trap'][-20:])),
        'final_entropy': float(np.mean(history['mode_entropy'][-20:])),
        'final_acc': float(np.mean(history['overall_acc'][-20:])),
        'history': history
    }

def main():
    print("=" * 80)
    print("Theorem 26: Multi-Mode Coverage & Anti-Collapse Invariance Benchmark")
    print("=" * 80)

    seeds = [42, 101, 2024, 777, 999]
    methods = ['grpo', 'ppo_entropy', 'flow_tb']
    results = {m: {} for m in methods}

    for m in methods:
        print(f"\nEvaluating Method: {m.upper()} across {len(seeds)} seeds...")
        res_list = []
        for s in seeds:
            res = run_experiment(method=m, seed=s)
            res_list.append(res)

        p0s = [r['final_p0'] for r in res_list]
        p1s = [r['final_p1'] for r in res_list]
        p2s = [r['final_p2'] for r in res_list]
        traps = [r['final_trap'] for r in res_list]
        ents = [r['final_entropy'] for r in res_list]
        accs = [r['final_acc'] for r in res_list]

        # Theoretical maximum entropy for 3 modes is ln(3) ≈ 1.0986
        results[m] = {
            'p0_mean': float(np.mean(p0s)),
            'p0_std': float(np.std(p0s)),
            'p1_mean': float(np.mean(p1s)),
            'p1_std': float(np.std(p1s)),
            'p2_mean': float(np.mean(p2s)),
            'p2_std': float(np.std(p2s)),
            'trap_mean': float(np.mean(traps)),
            'trap_std': float(np.std(traps)),
            'entropy_mean': float(np.mean(ents)),
            'entropy_std': float(np.std(ents)),
            'acc_mean': float(np.mean(accs)),
            'acc_std': float(np.std(accs))
        }

        print(f"  -> Mode 0 (Algebraic):      {results[m]['p0_mean']*100:.2f}% ± {results[m]['p0_std']*100:.2f}%")
        print(f"  -> Mode 1 (Geometric):      {results[m]['p1_mean']*100:.2f}% ± {results[m]['p1_std']*100:.2f}%")
        print(f"  -> Mode 2 (Inductive):      {results[m]['p2_mean']*100:.2f}% ± {results[m]['p2_std']*100:.2f}%")
        print(f"  -> Trap Fallacy Rate:       {results[m]['trap_mean']*100:.2f}% ± {results[m]['trap_std']*100:.2f}%")
        print(f"  -> Reasoning Mode Entropy:  {results[m]['entropy_mean']:.4f} (Max: 1.0986)")
        print(f"  -> Total Mathematical Acc:  {results[m]['acc_mean']*100:.2f}%")

    out_file = "experiments/autonomous_research_20260910/multimodal_anti_collapse_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved empirical results to {out_file}")

if __name__ == "__main__":
    main()
