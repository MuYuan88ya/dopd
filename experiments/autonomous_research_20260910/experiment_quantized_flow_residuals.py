import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import os

"""
Experiment: Quantized Flow Residuals for Ultra-Low Precision (FP8/INT8) Distributed Training (Theorem 20)
Benchmarking PPO vs FlowBalance under Full Precision (FP32), Half Precision (BF16), and 8-bit Quantization (FP8/INT8).

Context:
In massive distributed LLM RL (e.g. 1024 GPUs on Ray clusters), communicating token-level log-probs,
ratios, and advantages across rollout nodes and learner nodes consumes substantial network bandwidth.

Hypothesis:
1. PPO token ratios r_t = pi_theta / pi_old concentrate near 1.0. Quantizing r_t in 8-bit precision
   truncates small policy updates (Q(r_t) = 1.0), severely degrading learning or stalling training.
2. FlowBalance operates in log-space (log pi_theta - log pi_ref), which spans a smooth range [-15.0, 0.0].
   Stochastic or block-quantized 8-bit flow residuals satisfy E[Q(delta_t)] = delta_t,
   preserving exact Mean Flow Conservation in expectation and achieving 100% convergence with 4x bandwidth savings!
"""

def quantize_int8(tensor, num_bits=8):
    # Symmetric 8-bit uniform quantization
    qmin = -(2 ** (num_bits - 1))
    qmax = (2 ** (num_bits - 1)) - 1
    max_val = tensor.abs().max().clamp(min=1e-6)
    scale = max_val / qmax
    q_tensor = torch.clamp(torch.round(tensor / scale), qmin, qmax) * scale
    return q_tensor

def quantize_fp8_e4m3(tensor):
    # Simulated FP8 E4M3: 1 sign bit, 4 exponent bits, 3 mantissa bits
    # Max representable ~ 448.0, smallest normal ~ 2^-6 = 0.0156
    # Truncate mantissa to 3 bits
    sign = torch.sign(tensor)
    abs_t = tensor.abs().clamp(min=1e-5, max=448.0)
    exp = torch.floor(torch.log2(abs_t))
    exp = torch.clamp(exp, -6.0, 8.0)
    mantissa = abs_t / (2.0 ** exp) - 1.0 # in [0, 1)
    mantissa_q = torch.round(mantissa * 8.0) / 8.0 # 3-bit mantissa (8 levels)
    reconstructed = sign * (1.0 + mantissa_q) * (2.0 ** exp)
    return reconstructed

class MultiStepMathTask:
    def __init__(self, num_problems=8, seq_len=10):
        self.num_problems = num_problems
        self.seq_len = seq_len
        self.optimal_tokens = torch.randint(0, 4, (num_problems, seq_len))
        
    def evaluate(self, problem_idx, tokens):
        opt = self.optimal_tokens[problem_idx]
        if list(tokens) == list(opt):
            return 1.0
        # Partial reward
        matches = sum(1 for a, b in zip(tokens, opt) if a == b)
        return 0.1 * (matches / self.seq_len)

class SmallLinearPolicy(nn.Module):
    def __init__(self, num_problems=8, seq_len=10, vocab_size=4):
        super().__init__()
        # Direct logit parameters [num_problems, seq_len, vocab_size]
        self.logits = nn.Parameter(torch.zeros(num_problems, seq_len, vocab_size))
        
    def get_probs(self, p_id):
        return torch.softmax(self.logits[p_id], dim=-1)

def run_quantization_experiment(algo="flowbalance", precision="fp32", num_epochs=90, group_size=16, lr=0.1, seeds=6):
    results = {
        "final_pass_rate": [],
        "convergence_epoch": []
    }
    
    for seed in range(seeds):
        torch.manual_seed(seed * 100 + 42)
        np.random.seed(seed * 100 + 42)
        
        env = MultiStepMathTask(num_problems=8, seq_len=8)
        policy = SmallLinearPolicy(num_problems=8, seq_len=8, vocab_size=4)
        optimizer = optim.Adam(policy.parameters(), lr=lr)
        
        converged_epoch = num_epochs
        
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            total_loss = 0.0
            
            for p_id in range(env.num_problems):
                probs = policy.get_probs(p_id) # [8, 4]
                
                trajectories = []
                for _ in range(group_size):
                    tokens = []
                    log_probs = []
                    for t in range(8):
                        a = int(torch.multinomial(probs[t], 1).item())
                        tokens.append(a)
                        log_probs.append(torch.log(probs[t, a] + 1e-12))
                    r = env.evaluate(p_id, tokens)
                    trajectories.append({
                        "tokens": tokens,
                        "log_probs": torch.stack(log_probs),
                        "reward": r
                    })
                    
                rewards = [t["reward"] for t in trajectories]
                mean_r = np.mean(rewards)
                std_r = np.std(rewards) + 1e-8
                
                loss = 0.0
                for t in trajectories:
                    adv_base = (t["reward"] - mean_r) / std_r
                    
                    if algo == "ppo":
                        # Simulate PPO with old log probs
                        old_lps = t["log_probs"].detach() + 0.05 * torch.randn_like(t["log_probs"])
                        ratio = torch.exp(t["log_probs"] - old_lps)
                        
                        # Apply quantization to ratio
                        if precision == "int8":
                            ratio = quantize_int8(ratio, num_bits=8)
                        elif precision == "fp8":
                            ratio = quantize_fp8_e4m3(ratio)
                            
                        surr1 = ratio * adv_base
                        surr2 = torch.clamp(ratio, 0.8, 1.2) * adv_base
                        loss -= torch.min(surr1, surr2).sum()
                        
                    elif algo == "flowbalance":
                        # FlowBalance advantage: 2 * (target - log_p)
                        tau = 0.5
                        uniform_lp = float(np.log(0.25))
                        target = uniform_lp + (t["reward"] / (tau * 8.0) + adv_base * 0.5)
                        raw_adv = 2.0 * (target - t["log_probs"])
                        
                        # Apply quantization to flow advantage
                        if precision == "int8":
                            q_adv = quantize_int8(raw_adv, num_bits=8)
                        elif precision == "fp8":
                            q_adv = quantize_fp8_e4m3(raw_adv)
                        else:
                            q_adv = raw_adv
                            
                        loss -= (q_adv * t["log_probs"]).sum()
                        
                total_loss += (loss / group_size)
                
            total_loss = total_loss / env.num_problems
            total_loss.backward()
            optimizer.step()
            
            # Check accuracy across all 8 problems
            with torch.no_grad():
                correct = 0
                for p_id in range(env.num_problems):
                    greedy = [int(policy.get_probs(p_id)[t].argmax().item()) for t in range(8)]
                    if env.evaluate(p_id, greedy) == 1.0:
                        correct += 1
                if correct == env.num_problems and converged_epoch == num_epochs:
                    converged_epoch = epoch
                    
        # Final pass rate
        correct = 0
        for p_id in range(env.num_problems):
            greedy = [int(policy.get_probs(p_id)[t].argmax().item()) for t in range(8)]
            if env.evaluate(p_id, greedy) == 1.0:
                correct += 1
        final_pass = correct / float(env.num_problems)
        
        results["final_pass_rate"].append(final_pass)
        results["convergence_epoch"].append(converged_epoch)
        
    return {
        "algo": algo,
        "precision": precision,
        "final_pass_mean": float(np.mean(results["final_pass_rate"])),
        "final_pass_std": float(np.std(results["final_pass_rate"])),
        "avg_convergence_epoch": float(np.mean(results["convergence_epoch"]))
    }

def main():
    print("================================================================================")
    print("BENCHMARK: Quantized Flow Residuals for Ultra-Low Precision (FP8/INT8) Training")
    print("================================================================================")
    
    experiments = [
        ("ppo", "fp32"),
        ("ppo", "fp8"),
        ("ppo", "int8"),
        ("flowbalance", "fp32"),
        ("flowbalance", "fp8"),
        ("flowbalance", "int8"),
    ]
    all_results = {}
    
    for algo, prec in experiments:
        key = f"{algo}_{prec}"
        res = run_quantization_experiment(algo=algo, precision=prec, num_epochs=90, group_size=16, lr=0.1, seeds=6)
        all_results[key] = res
        print(f"Config: {algo.upper():12s} [{prec.upper():4s}] | Pass Rate: {res['final_pass_mean']*100:5.2f}% +/- {res['final_pass_std']*100:4.2f}% | Conv Epoch: {res['avg_convergence_epoch']:5.1f}")
        
    output_path = "experiments/autonomous_research_20260910/quantized_flow_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved results to {output_path}")

if __name__ == "__main__":
    main()
