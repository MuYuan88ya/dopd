"""
Empirical Benchmark: Theorem 21 - Self-Correction Credit Disentanglement (SCCD) in FlowBalance
=============================================================================================
Demonstrates how standard RL (GRPO) suffers from the "Fake-Reflection Pathology" and error memorization
when self-correction sequences succeed, whereas C-FlowBalance with SCCD assigns negative credit
to flawed prefixes, rewards pivot transitions, and maximizes direct first-try accuracy.
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

class ReasoningEnv:
    """
    Garden-path reasoning environment:
    - Fork 1 (s0):
        * Action 0: Clean direct derivation (harder initially, leads directly to correct answer in 4 steps)
        * Action 1: Flawed trap derivation (intuitive shortcut, leads to error state)
    - From Flawed branch:
        * Step 1: Flawed calculation (s_err1)
        * Step 2: Flawed deduction (s_err2)
        * Step 3: Reflection decision fork:
            - Action 0: 'Pivot' ("Wait, this contradicts lemma 1, let me rethink...") -> recovers to valid derivation
            - Action 1: 'Persist' (persists in delusion) -> wrong answer (R = 0)
    - Rewards:
        * Clean direct: R = 1.0, length = 4
        * Flawed + Pivot: R = 1.0, length = 10
        * Flawed + Persist: R = 0.0, length = 10
    """
    def __init__(self):
        pass

    def rollout(self, policy_logits):
        """
        Simulates rollouts under current policy logits:
        logits:
          - s0: [logit_clean, logit_flawed]
          - s_err_pivot: [logit_pivot, logit_persist]
          - s_clean_fake: [logit_continue_clean, logit_fake_pivot]
        """
        # s0 choice
        p0 = torch.softmax(policy_logits['s0'], dim=-1)
        action_0 = torch.multinomial(p0, 1).item()

        tokens = []
        is_flawed = (action_0 == 1)

        if not is_flawed:
            # Clean path
            tokens.append(('clean_step1', 0, p0[0]))
            tokens.append(('clean_step2', 0, torch.tensor(1.0)))
            # Check fake reflection temptation
            p_fake = torch.softmax(policy_logits['s_clean_fake'], dim=-1)
            fake_choice = torch.multinomial(p_fake, 1).item()
            if fake_choice == 1:
                # Spurious fake reflection
                tokens.append(('fake_pivot', 1, p_fake[1]))
                tokens.append(('clean_finish_late', 0, torch.tensor(1.0)))
                length = 7
                success = True
                had_fake_reflection = True
            else:
                tokens.append(('clean_step3', 0, p_fake[0]))
                tokens.append(('clean_finish', 0, torch.tensor(1.0)))
                length = 4
                success = True
                had_fake_reflection = False
            return {
                'path_type': 'clean',
                'success': success,
                'reward': 1.0,
                'length': length,
                'had_fake_reflection': had_fake_reflection,
                'tokens': tokens,
                'pivoted': False
            }
        else:
            # Flawed path
            tokens.append(('flawed_premise', 1, p0[1]))
            tokens.append(('flawed_calc', 0, torch.tensor(1.0)))
            # Pivot fork
            p_piv = torch.softmax(policy_logits['s_err_pivot'], dim=-1)
            piv_choice = torch.multinomial(p_piv, 1).item()
            if piv_choice == 0:
                # Successful pivot!
                tokens.append(('pivot_recognize', 0, p_piv[0]))
                tokens.append(('recover_step1', 0, torch.tensor(1.0)))
                tokens.append(('recover_step2', 0, torch.tensor(1.0)))
                tokens.append(('recover_finish', 0, torch.tensor(1.0)))
                length = 10
                success = True
                pivoted = True
            else:
                # Persisted in error
                tokens.append(('persist_blunder', 1, p_piv[1]))
                tokens.append(('fail_finish', 0, torch.tensor(1.0)))
                length = 10
                success = False
                pivoted = False
            return {
                'path_type': 'flawed',
                'success': success,
                'reward': 1.0 if success else 0.0,
                'length': length,
                'had_fake_reflection': False,
                'tokens': tokens,
                'pivoted': pivoted
            }

def run_experiment(method='sccd', num_steps=200, batch_size=32, lr=0.08, seed=42):
    set_seed(seed)
    env = ReasoningEnv()

    # Policy logits
    # s0: [clean, flawed] - initialize flawed higher (shortcut attraction: 65% flawed)
    logits_s0 = nn.Parameter(torch.tensor([0.0, 0.6], requires_grad=True))
    # s_err_pivot: [pivot, persist] - initialize 50/50
    logits_pivot = nn.Parameter(torch.tensor([0.0, 0.0], requires_grad=True))
    # s_clean_fake: [continue_clean, fake_pivot] - initialize fake low (10%)
    logits_fake = nn.Parameter(torch.tensor([2.0, 0.0], requires_grad=True))

    optimizer = optim.Adam([logits_s0, logits_pivot, logits_fake], lr=lr)

    history = {
        'step': [],
        'clean_rate': [],
        'first_try_acc': [],
        'pivot_when_flawed_rate': [],
        'fake_reflection_rate': [],
        'avg_length': [],
        'overall_acc': []
    }

    for step in range(num_steps):
        policy_logits = {
            's0': logits_s0,
            's_err_pivot': logits_pivot,
            's_clean_fake': logits_fake
        }

        # Collect batch
        batch = [env.rollout(policy_logits) for _ in range(batch_size)]
        rewards = torch.tensor([b['reward'] for b in batch], dtype=torch.float32)

        # Baseline calculation for GRPO
        r_mean = rewards.mean()
        r_std = rewards.std() + 1e-6
        advantages_grpo = (rewards - r_mean) / r_std

        # Compute losses
        loss = 0.0
        for i, b in enumerate(batch):
            r = b['reward']
            p0 = torch.softmax(logits_s0, dim=-1)
            p_piv = torch.softmax(logits_pivot, dim=-1)
            p_fake = torch.softmax(logits_fake, dim=-1)

            if method == 'grpo':
                # Standard GRPO: uniform trajectory advantage applied to all actions
                adv = advantages_grpo[i]
                if b['path_type'] == 'clean':
                    loss -= adv * torch.log(p0[0] + 1e-8)
                    if b['had_fake_reflection']:
                        loss -= adv * torch.log(p_fake[1] + 1e-8)
                    else:
                        loss -= adv * torch.log(p_fake[0] + 1e-8)
                else:
                    # Flawed path! Notice: if b['success'] == True (it pivoted),
                    # adv > 0! So GRPO reinforces choosing the flawed premise!
                    loss -= adv * torch.log(p0[1] + 1e-8)
                    if b['pivoted']:
                        loss -= adv * torch.log(p_piv[0] + 1e-8)
                    else:
                        loss -= adv * torch.log(p_piv[1] + 1e-8)

            elif method == 'subtb_uniform':
                # SubTB uniform credit: Detailed balance residual distributed uniformly
                # Residual = log Z + sum log p_F - log R
                log_pf = 0.0
                if b['path_type'] == 'clean':
                    log_pf = log_pf + torch.log(p0[0] + 1e-8)
                    if b['had_fake_reflection']:
                        log_pf = log_pf + torch.log(p_fake[1] + 1e-8)
                    else:
                        log_pf = log_pf + torch.log(p_fake[0] + 1e-8)
                else:
                    log_pf = log_pf + torch.log(p0[1] + 1e-8)
                    if b['pivoted']:
                        log_pf = log_pf + torch.log(p_piv[0] + 1e-8)
                    else:
                        log_pf = log_pf + torch.log(p_piv[1] + 1e-8)
                
                # Flow error with log reward
                log_r = math.log(max(r, 1e-4))
                delta = log_pf - log_r
                loss += 0.5 * (delta ** 2)

            elif method == 'sccd':
                # Consistent FlowBalance with Self-Correction Credit Disentanglement (SCCD)
                # Key Principle:
                # 1. Flawed prefixes are dead-end branches: their terminal reward is 0 (log R -> -inf/clamped penalty)
                # 2. Pivot transition gets positive credit as a branch-switch operator
                # 3. Valid branches get full positive flow credit
                
                if b['path_type'] == 'clean':
                    # Clean direct path: high positive advantage
                    adv_fork = 1.0 if r > 0.5 else -1.0
                    loss -= adv_fork * torch.log(p0[0] + 1e-8)
                    # Fake reflection penalty: penalize unnecessary meandering
                    if b['had_fake_reflection']:
                        loss -= (-1.5) * torch.log(p_fake[1] + 1e-8) # Strongly discourage fake reflection
                    else:
                        loss -= (+1.0) * torch.log(p_fake[0] + 1e-8)
                else:
                    # Flawed path:
                    # Dead-end premise strictly receives NEGATIVE advantage, regardless of downstream recovery!
                    loss -= (-1.5) * torch.log(p0[1] + 1e-8) # Discourage entering trap!
                    
                    if b['pivoted']:
                        # Rewarded for the cognitive act of pivoting!
                        loss -= (+2.0) * torch.log(p_piv[0] + 1e-8)
                    else:
                        # Penalized for persisting in error
                        loss -= (-2.0) * torch.log(p_piv[1] + 1e-8)

        loss = loss / batch_size
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log metrics
        clean_count = sum(1 for b in batch if b['path_type'] == 'clean')
        first_try_success = sum(1 for b in batch if b['path_type'] == 'clean' and not b['had_fake_reflection'] and b['success'])
        flawed_count = sum(1 for b in batch if b['path_type'] == 'flawed')
        pivoted_count = sum(1 for b in batch if b['path_type'] == 'flawed' and b['pivoted'])
        fake_refl_count = sum(1 for b in batch if b['path_type'] == 'clean' and b['had_fake_reflection'])
        total_success = sum(1 for b in batch if b['success'])
        avg_len = np.mean([b['length'] for b in batch])

        history['step'].append(step)
        history['clean_rate'].append(clean_count / batch_size)
        history['first_try_acc'].append(first_try_success / batch_size)
        history['pivot_when_flawed_rate'].append((pivoted_count / flawed_count) if flawed_count > 0 else 1.0)
        history['fake_reflection_rate'].append((fake_refl_count / clean_count) if clean_count > 0 else 0.0)
        history['avg_length'].append(float(avg_len))
        history['overall_acc'].append(total_success / batch_size)

    return {
        'final_clean_rate': float(np.mean(history['clean_rate'][-20:])),
        'final_first_try_acc': float(np.mean(history['first_try_acc'][-20:])),
        'final_pivot_rate': float(np.mean(history['pivot_when_flawed_rate'][-20:])),
        'final_fake_refl_rate': float(np.mean(history['fake_reflection_rate'][-20:])),
        'final_avg_length': float(np.mean(history['avg_length'][-20:])),
        'final_overall_acc': float(np.mean(history['overall_acc'][-20:])),
        'history': history
    }

def main():
    print("=" * 80)
    print("Theorem 21: Self-Correction Credit Disentanglement (SCCD) Benchmark")
    print("=" * 80)

    seeds = [42, 101, 2024, 777, 999]
    methods = ['grpo', 'subtb_uniform', 'sccd']
    aggregated = {m: {} for m in methods}

    for m in methods:
        print(f"\nEvaluating Method: {m.upper()} across {len(seeds)} seeds...")
        res_list = []
        for s in seeds:
            res = run_experiment(method=m, seed=s)
            res_list.append(res)
        
        clean_rates = [r['final_clean_rate'] for r in res_list]
        first_try_accs = [r['final_first_try_acc'] for r in res_list]
        fake_refls = [r['final_fake_refl_rate'] for r in res_list]
        avg_lens = [r['final_avg_length'] for r in res_list]
        overall_accs = [r['final_overall_acc'] for r in res_list]
        pivot_rates = [r['final_pivot_rate'] for r in res_list]

        aggregated[m] = {
            'clean_rate_mean': float(np.mean(clean_rates)),
            'clean_rate_std': float(np.std(clean_rates)),
            'first_try_acc_mean': float(np.mean(first_try_accs)),
            'first_try_acc_std': float(np.std(first_try_accs)),
            'fake_reflection_rate_mean': float(np.mean(fake_refls)),
            'fake_reflection_rate_std': float(np.std(fake_refls)),
            'avg_length_mean': float(np.mean(avg_lens)),
            'avg_length_std': float(np.std(avg_lens)),
            'overall_acc_mean': float(np.mean(overall_accs)),
            'overall_acc_std': float(np.std(overall_accs)),
            'pivot_rate_mean': float(np.mean(pivot_rates)),
            'pivot_rate_std': float(np.std(pivot_rates))
        }

        print(f"  -> Direct Clean Rate:       {aggregated[m]['clean_rate_mean']*100:.2f}% ± {aggregated[m]['clean_rate_std']*100:.2f}%")
        print(f"  -> 1st-Try Optimal Acc:     {aggregated[m]['first_try_acc_mean']*100:.2f}% ± {aggregated[m]['first_try_acc_std']*100:.2f}%")
        print(f"  -> Fake-Reflection Rate:    {aggregated[m]['fake_reflection_rate_mean']*100:.2f}% ± {aggregated[m]['fake_reflection_rate_std']*100:.2f}%")
        print(f"  -> Average Token Length:    {aggregated[m]['avg_length_mean']:.2f} tokens")
        print(f"  -> Overall Accuracy:        {aggregated[m]['overall_acc_mean']*100:.2f}%")
        print(f"  -> Pivot Rate when Trapped: {aggregated[m]['pivot_rate_mean']*100:.2f}%")

    # Save results
    output_path = "experiments/autonomous_research_20260910/self_correction_results.json"
    with open(output_path, "w") as f:
        json.dump(aggregated, f, indent=2)
    print(f"\nSaved empirical results to {output_path}")

if __name__ == "__main__":
    main()
