"""
Experiment: Skorokhod Stochastic Differential Equations & Reflecting Boundary Invariance (Theorem 44)

Tests reasoning trajectory dynamics on verifier-constrained deduction polytopes D with boundary \partial D:
    dX_t = \nabla \Phi(X_t) dt + \sigma dW_t - \mathbf{n}(X_t) dL_t
    subject to zero boundary flux: \int_{\partial D} F \cdot \mathbf{n} \, dS = 0.

Compares Skorokhod FlowBalance against Monolithic GRPO and Actor-Critic PPO in a narrow
syntax/type-constrained deduction corridor where unconstrained exploration causes boundary crash.
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

DEVICE = torch.device("cpu")

class ConstrainedPolytopeCorridor:
    """
    Constrained reasoning corridor D = { (x, y) | -2.0 <= x <= 2.0, y_min(x) <= y <= y_max(x) }.
    A narrow corridor of width 0.4 representing valid type/syntax deductions.
    Any unconstrained step outside D triggers verifier syntax rejection (absorption crash).
    """
    def __init__(self, steps=8, width=0.4):
        self.steps = steps
        self.width = width
        self.start = torch.tensor([-1.5, 0.0], dtype=torch.float32)
        self.target = torch.tensor([1.5, 0.0], dtype=torch.float32)

    def corridor_bounds(self, x):
        """Returns upper and lower y-bounds for a given x: curved corridor."""
        # Central spine: y_spine(x) = 0.8 * cos(pi * x / 3.0)
        y_spine = 0.8 * torch.cos(np.pi * x / 3.0)
        y_min = y_spine - self.width / 2.0
        y_max = y_spine + self.width / 2.0
        return y_min, y_max

    def is_inside(self, pts):
        """pts: [..., 2] -> boolean tensor"""
        x = pts[..., 0]
        y = pts[..., 1]
        y_min, y_max = self.corridor_bounds(x)
        x_ok = (x >= -2.0) & (x <= 2.0)
        y_ok = (y >= y_min) & (y <= y_max)
        return x_ok & y_ok

    def project_skorokhod(self, pts):
        """Projects points back onto D along outward boundary normal (Skorokhod reflection)."""
        x = torch.clamp(pts[..., 0], -2.0, 2.0)
        y = pts[..., 1]
        y_min, y_max = self.corridor_bounds(x)
        # Reflect y if violated
        y_proj = torch.clamp(y, y_min + 1e-4, y_max - 1e-4)
        reflected = torch.stack([x, y_proj], dim=-1)
        # Local time increment dL = ||pts - reflected||
        dl = torch.norm(pts - reflected, dim=-1)
        return reflected, dl

    def evaluate_trajectories(self, trajs, is_reflected=False):
        """
        trajs: [B, steps+1, 2]
        Returns:
            clean_pass: [B] (boolean)
            boundary_crash_rate: [B] (boolean)
            final_distance: [B]
            mean_local_time: [B]
        """
        B = trajs.shape[0]
        crashes = torch.zeros(B, dtype=torch.bool)
        for s in range(self.steps + 1):
            inside = self.is_inside(trajs[:, s])
            crashes = crashes | (~inside)

        final_dist = torch.norm(trajs[:, -1] - self.target, dim=-1)
        # If unreflected and crashed, cannot pass
        if not is_reflected:
            success = (final_dist < 0.35) & (~crashes)
        else:
            success = (final_dist < 0.35)

        return success, crashes, final_dist

class CorridorPolicy(nn.Module):
    """Parameterizes transition velocity field v_theta(x, t)."""
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2)
        )
        self.log_f = nn.Sequential(
            nn.Linear(3, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1)
        )
        self.log_z = nn.Parameter(torch.tensor(0.0))

    def get_velocity(self, x, t):
        return self.net(torch.cat([x, t], dim=-1))

    def get_flow(self, x, t):
        return self.log_f(torch.cat([x, t], dim=-1))

def run_experiment_seed(seed, n_epochs=100, batch_size=32):
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = ConstrainedPolytopeCorridor(steps=8, width=0.4)
    dt = 0.45

    # 1. FlowBalance with Skorokhod Boundary Reflection
    fb_model = CorridorPolicy()
    opt_fb = optim.Adam(fb_model.parameters(), lr=0.008)

    for ep in range(n_epochs):
        opt_fb.zero_grad()
        x_curr = env.start.repeat(batch_size, 1) + 0.02 * torch.randn(batch_size, 2)
        # Ensure start is inside
        x_curr, _ = env.project_skorokhod(x_curr)
        traj = [x_curr]
        log_flows = []
        dls = []

        for s in range(env.steps):
            t_t = torch.full((batch_size, 1), s / env.steps)
            v = fb_model.get_velocity(x_curr, t_t)
            f = fb_model.get_flow(x_curr, t_t)
            log_flows.append(f)

            # SDE step with noise
            x_next_raw = x_curr + dt * v + 0.03 * torch.randn_like(x_curr)
            # Skorokhod reflection at boundary \partial D
            x_next, dl = env.project_skorokhod(x_next_raw)
            dls.append(dl)
            traj.append(x_next)
            x_curr = x_next

        traj = torch.stack(traj, dim=1) # [B, steps+1, 2]
        t_T = torch.full((batch_size, 1), 1.0)
        f_T = fb_model.get_flow(traj[:, -1], t_T)

        succ, crashes, final_d = env.evaluate_trajectories(traj, is_reflected=True)
        # Terminal reward
        rew = torch.where(succ, 10.0 - 2.0 * final_d, -1.0 - 5.0 * final_d)
        log_r = rew

        # SubTB Detailed Balance loss with zero boundary flux constraint
        step_losses = []
        for s in range(env.steps):
            # Detailed balance: F(s_t) = F(s_{t+1}) * exp(-cost - dl)
            step_cost = 0.5 * torch.sum((traj[:, s+1] - traj[:, s]) ** 2, dim=-1, keepdim=True)
            delta = log_flows[s] - fb_model.get_flow(traj[:, s+1], torch.full((batch_size, 1), (s+1)/env.steps)) - step_cost - dls[s].unsqueeze(-1)
            step_losses.append(torch.mean(delta ** 2))

        term_loss = torch.mean((f_T.squeeze(-1) - log_r) ** 2)
        # Guidance towards target
        guide = torch.mean(final_d ** 2)
        tot_fb_loss = sum(step_losses) + term_loss + 0.5 * guide
        tot_fb_loss.backward()
        torch.nn.utils.clip_grad_norm_(fb_model.parameters(), 1.0)
        opt_fb.step()

    # 2. Monolithic GRPO (Unconstrained policy, absorbed at boundary)
    grpo_model = CorridorPolicy()
    opt_grpo = optim.Adam(grpo_model.parameters(), lr=0.008)

    for ep in range(n_epochs):
        opt_grpo.zero_grad()
        x_curr = env.start.repeat(batch_size, 1) + 0.02 * torch.randn(batch_size, 2)
        x_curr, _ = env.project_skorokhod(x_curr)
        traj = [x_curr]
        velocities = []

        for s in range(env.steps):
            t_t = torch.full((batch_size, 1), s / env.steps)
            v = grpo_model.get_velocity(x_curr, t_t)
            velocities.append(v)
            # Unconstrained transition: NO reflection
            x_next = x_curr + dt * v + 0.03 * torch.randn_like(x_curr)
            traj.append(x_next)
            x_curr = x_next

        traj = torch.stack(traj, dim=1)
        succ, crashes, final_d = env.evaluate_trajectories(traj, is_reflected=False)
        # Verifier rejects all boundary crashes: 0.0 reward
        rews = torch.where(succ, 10.0 - 2.0 * final_d, -5.0 * crashes.float() - final_d)
        adv = (rews - rews.mean()) / (rews.std() + 1e-4)

        all_v = torch.stack(velocities, dim=1)
        loss_grpo = -torch.mean(adv.unsqueeze(-1).unsqueeze(-1) * (-0.5 * all_v ** 2))
        loss_grpo.backward()
        torch.nn.utils.clip_grad_norm_(grpo_model.parameters(), 1.0)
        opt_grpo.step()

    # 3. Actor-Critic PPO (Unconstrained, absorbed at boundary)
    ppo_model = CorridorPolicy()
    ppo_critic = nn.Sequential(nn.Linear(3, 64), nn.SiLU(), nn.Linear(64, 1))
    opt_ppo = optim.Adam(list(ppo_model.parameters()) + list(ppo_critic.parameters()), lr=0.008)

    for ep in range(n_epochs):
        opt_ppo.zero_grad()
        x_curr = env.start.repeat(batch_size, 1) + 0.02 * torch.randn(batch_size, 2)
        x_curr, _ = env.project_skorokhod(x_curr)
        traj = [x_curr]
        velocities = []
        values = []

        for s in range(env.steps):
            t_t = torch.full((batch_size, 1), s / env.steps)
            v = ppo_model.get_velocity(x_curr, t_t)
            val = ppo_critic(torch.cat([x_curr, t_t], dim=-1)).squeeze(-1)
            velocities.append(v)
            values.append(val)
            x_next = x_curr + dt * v + 0.03 * torch.randn_like(x_curr)
            traj.append(x_next)
            x_curr = x_next

        traj = torch.stack(traj, dim=1)
        succ, crashes, final_d = env.evaluate_trajectories(traj, is_reflected=False)
        rews = torch.where(succ, 10.0 - 2.0 * final_d, -5.0 * crashes.float() - final_d)

        values = torch.stack(values, dim=1)
        adv_ppo = rews.unsqueeze(-1) - values
        all_v = torch.stack(velocities, dim=1)
        loss_pol = -torch.mean(adv_ppo.detach().unsqueeze(-1) * (-0.5 * all_v ** 2))
        loss_val = torch.mean(adv_ppo ** 2)
        tot_ppo = loss_pol + 0.5 * loss_val
        tot_ppo.backward()
        torch.nn.utils.clip_grad_norm_(ppo_model.parameters(), 1.0)
        opt_ppo.step()

    # Evaluation Phase (N=200 rollouts)
    eval_n = 200
    with torch.no_grad():
        # Evaluate FlowBalance (with Skorokhod reflection)
        x_curr = env.start.repeat(eval_n, 1)
        traj_fb = [x_curr]
        fb_total_dl = torch.zeros(eval_n)
        for s in range(env.steps):
            t_t = torch.full((eval_n, 1), s / env.steps)
            v = fb_model.get_velocity(x_curr, t_t)
            x_next_raw = x_curr + dt * v
            x_next, dl = env.project_skorokhod(x_next_raw)
            fb_total_dl += dl
            traj_fb.append(x_next)
            x_curr = x_next
        traj_fb = torch.stack(traj_fb, dim=1)
        succ_fb, crash_fb, dist_fb = env.evaluate_trajectories(traj_fb, is_reflected=True)

        # Evaluate GRPO (Unconstrained)
        x_curr = env.start.repeat(eval_n, 1)
        traj_grpo = [x_curr]
        for s in range(env.steps):
            t_t = torch.full((eval_n, 1), s / env.steps)
            v = grpo_model.get_velocity(x_curr, t_t)
            x_next = x_curr + dt * v
            traj_grpo.append(x_next)
            x_curr = x_next
        traj_grpo = torch.stack(traj_grpo, dim=1)
        succ_grpo, crash_grpo, dist_grpo = env.evaluate_trajectories(traj_grpo, is_reflected=False)

        # Evaluate PPO (Unconstrained)
        x_curr = env.start.repeat(eval_n, 1)
        traj_ppo = [x_curr]
        for s in range(env.steps):
            t_t = torch.full((eval_n, 1), s / env.steps)
            v = ppo_model.get_velocity(x_curr, t_t)
            x_next = x_curr + dt * v
            traj_ppo.append(x_next)
            x_curr = x_next
        traj_ppo = torch.stack(traj_ppo, dim=1)
        succ_ppo, crash_ppo, dist_ppo = env.evaluate_trajectories(traj_ppo, is_reflected=False)

    return {
        "flow": {
            "clean_pass_at_1": succ_fb.float().mean().item(),
            "boundary_crash_rate": crash_fb.float().mean().item(),
            "final_distance": dist_fb.mean().item(),
            "local_time_dissipation": fb_total_dl.mean().item()
        },
        "grpo": {
            "clean_pass_at_1": succ_grpo.float().mean().item(),
            "boundary_crash_rate": crash_grpo.float().mean().item(),
            "final_distance": dist_grpo.mean().item(),
            "local_time_dissipation": 0.0
        },
        "ppo": {
            "clean_pass_at_1": succ_ppo.float().mean().item(),
            "boundary_crash_rate": crash_ppo.float().mean().item(),
            "final_distance": dist_ppo.mean().item(),
            "local_time_dissipation": 0.0
        }
    }

def main():
    print("Starting Theorem 44: Skorokhod SDE & Reflecting Boundary Invariance...")
    seeds = [42, 43, 44, 45, 46]
    all_results = {"flow": [], "grpo": [], "ppo": []}

    for seed in seeds:
        print(f"Running seed {seed}...")
        res = run_experiment_seed(seed, n_epochs=100)
        for k in ["flow", "grpo", "ppo"]:
            all_results[k].append(res[k])

    summary = {}
    for k in ["flow", "grpo", "ppo"]:
        summary[k] = {
            "clean_pass_at_1_mean": float(np.mean([r["clean_pass_at_1"] for r in all_results[k]])),
            "clean_pass_at_1_std": float(np.std([r["clean_pass_at_1"] for r in all_results[k]])),
            "boundary_crash_rate_mean": float(np.mean([r["boundary_crash_rate"] for r in all_results[k]])),
            "boundary_crash_rate_std": float(np.std([r["boundary_crash_rate"] for r in all_results[k]])),
            "final_distance_mean": float(np.mean([r["final_distance"] for r in all_results[k]])),
            "final_distance_std": float(np.std([r["final_distance"] for r in all_results[k]])),
            "local_time_dissipation_mean": float(np.mean([r["local_time_dissipation"] for r in all_results[k]])),
            "local_time_dissipation_std": float(np.std([r["local_time_dissipation"] for r in all_results[k]]))
        }

    print("\n--- Summary Results (5 Seeds) ---")
    print(json.dumps(summary, indent=2))

    out_file = os.path.join(os.path.dirname(__file__), "skorokhod_reflection_results.json")
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved to {out_file}")

if __name__ == "__main__":
    main()
