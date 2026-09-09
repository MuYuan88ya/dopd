"""
Empirical Benchmark: Theorem 24 - Total Log-Flow Decoupling & Reward Scale Invariance
====================================================================================
Demonstrates how FlowBalance completely absorbs dynamic reward scale shifts (inflation / deflation)
into the scalar partition function log Z, maintaining invariant policy gradients, whereas
PPO suffers catastrophic gradient explosion during reward inflation and freezes during deflation.
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

class DriftingRewardEnv:
    """
    Reasoning environment with dynamic reward scaling across epochs:
    - Phase 1 (Epochs 0-40): Scale = 1.0 (Normal)
    - Phase 2 (Epochs 40-80): Scale = 20.0 (Reward Inflation / Verifier Scaling)
    - Phase 3 (Epochs 80-120): Scale = 0.02 (Reward Deflation / Harsh Verifier)
    
    Decision Fork at s0:
    - Action 0: Correct proof (R = 1.0 * Scale)
    - Action 1: Incorrect proof (R = 1e-4 * Scale)
    """
    def __init__(self):
        pass

    def get_scale(self, epoch):
        if epoch < 40:
            return 1.0
        elif epoch < 80:
            return 20.0
        else:
            return 0.02

    def rollout(self, logits, scale):
        p = torch.softmax(logits, dim=-1)
        action = torch.multinomial(p, 1).item()
        base_r = 1.0 if action == 0 else 1e-4
        reward = base_r * scale
        return {
            'action': action,
            'reward': reward,
            'base_reward': base_r,
            'is_correct': (action == 0),
            'log_prob': torch.log(p[action] + 1e-8)
        }

def run_experiment(method='flow_tb', num_epochs=120, batch_size=16, lr=0.04, seed=42):
    set_seed(seed)
    env = DriftingRewardEnv()

    # Model parameters: initialize with slight bias towards action 1 (trap)
    logits = nn.Parameter(torch.tensor([0.0, 0.5], requires_grad=True))
    log_z = nn.Parameter(torch.tensor(0.0, requires_grad=True))
    critic = nn.Parameter(torch.tensor(0.5, requires_grad=True))

    if method == 'flow_tb':
        optimizer = optim.Adam([logits, log_z], lr=lr)
    elif method == 'ppo':
        optimizer = optim.Adam([logits, critic], lr=lr)
    elif method == 'grpo':
        optimizer = optim.Adam([logits], lr=lr)

    history = {
        'epoch': [],
        'accuracy': [],
        'grad_norm': [],
        'log_z': [],
        'reward_scale': []
    }

    for epoch in range(num_epochs):
        scale = env.get_scale(epoch)
        batch = [env.rollout(logits, scale) for _ in range(batch_size)]
        rewards = torch.tensor([b['reward'] for b in batch], dtype=torch.float32)

        loss = 0.0
        if method == 'ppo':
            # Standard PPO without group norm: advantage = R - V(s)
            advs = rewards - critic
            loss_actor = 0.0
            for i, b in enumerate(batch):
                loss_actor -= advs[i] * b['log_prob']
            loss_critic = 0.5 * torch.mean((rewards - critic) ** 2)
            loss = (loss_actor / batch_size) + loss_critic

        elif method == 'grpo':
            # GRPO: Group-normalized advantage
            r_mean = rewards.mean()
            r_std = rewards.std() + 1e-6
            advs = (rewards - r_mean) / r_std
            for i, b in enumerate(batch):
                loss -= advs[i] * b['log_prob']
            loss = loss / batch_size

        elif method == 'flow_tb':
            # Trajectory Balance: (log Z + log pi - log R)^2
            for b in batch:
                log_r = math.log(max(b['reward'], 1e-6))
                delta = log_z + b['log_prob'] - log_r
                loss += 0.5 * (delta ** 2)
            loss = loss / batch_size

        optimizer.zero_grad()
        loss.backward()
        
        # Measure gradient norm on policy logits
        grad_norm = logits.grad.norm().item() if logits.grad is not None else 0.0
        optimizer.step()

        p = torch.softmax(logits, dim=-1).detach().numpy()
        acc = p[0]

        history['epoch'].append(epoch)
        history['accuracy'].append(float(acc))
        history['grad_norm'].append(float(grad_norm))
        history['log_z'].append(float(log_z.item()))
        history['reward_scale'].append(float(scale))

    # Evaluate phase performance
    p1_acc = np.mean(history['accuracy'][25:40])
    p2_acc = np.mean(history['accuracy'][65:80])
    p3_acc = np.mean(history['accuracy'][105:120])
    max_grad = np.max(history['grad_norm'])

    return {
        'phase1_acc': float(p1_acc),
        'phase2_acc': float(p2_acc),
        'phase3_acc': float(p3_acc),
        'max_grad_norm': float(max_grad),
        'final_acc': float(history['accuracy'][-1]),
        'history': history
    }

def main():
    print("=" * 80)
    print("Theorem 24: Total Log-Flow Decoupling & Reward Scale Invariance Benchmark")
    print("=" * 80)

    seeds = [42, 101, 2024, 777, 999]
    methods = ['ppo', 'grpo', 'flow_tb']
    results = {m: {} for m in methods}

    for m in methods:
        print(f"\nEvaluating Method: {m.upper()} across {len(seeds)} seeds...")
        res_list = []
        for s in seeds:
            res = run_experiment(method=m, seed=s)
            res_list.append(res)

        p1s = [r['phase1_acc'] for r in res_list]
        p2s = [r['phase2_acc'] for r in res_list]
        p3s = [r['phase3_acc'] for r in res_list]
        max_grads = [r['max_grad_norm'] for r in res_list]

        results[m] = {
            'phase1_mean': float(np.mean(p1s)),
            'phase1_std': float(np.std(p1s)),
            'phase2_mean': float(np.mean(p2s)),
            'phase2_std': float(np.std(p2s)),
            'phase3_mean': float(np.mean(p3s)),
            'phase3_std': float(np.std(p3s)),
            'max_grad_mean': float(np.mean(max_grads)),
            'max_grad_std': float(np.std(max_grads))
        }

        print(f"  -> Phase 1 Acc (Scale 1.0):   {results[m]['phase1_mean']*100:.2f}% ± {results[m]['phase1_std']*100:.2f}%")
        print(f"  -> Phase 2 Acc (Scale 20.0):  {results[m]['phase2_mean']*100:.2f}% ± {results[m]['phase2_std']*100:.2f}%")
        print(f"  -> Phase 3 Acc (Scale 0.02):  {results[m]['phase3_mean']*100:.2f}% ± {results[m]['phase3_std']*100:.2f}%")
        print(f"  -> Max Policy Gradient Norm:  {results[m]['max_grad_mean']:.4f} ± {results[m]['max_grad_std']:.4f}")

    out_file = "experiments/autonomous_research_20260910/reward_scale_drift_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved empirical results to {out_file}")

if __name__ == "__main__":
    main()
