# Autonomous Research Plan: Deepening GFlowNet FlowBalance for Long-Chain Mathematical Reasoning

**Author**: Antigravity Autonomous Research Agent  
**Date**: September 10, 2026  
**Directory**: `experiments/autonomous_research_20260910/`  

---

## 1. Executive Research Objective

While `flow_balance` (Trajectory Balance) and `c_flow_balance` (Consistent Sub-Trajectory Balance) have addressed macro-scale convergence and basic sign-safety via Group Pairwise AUC ($g_{\text{consist}}$), several critical theoretical and practical challenges remain in Reinforcement Learning for long-chain reasoning models:

1. **Token Importance Heterogeneity**:
   Current Detailed Balance (DB) spreads the sequence-level reward $R$ and baseline $b$ uniformly across all tokens ($\frac{R}{\tau L} + \frac{b}{L}$). However, mathematical reasoning consists of high-entropy decision forks ("aha!" insight tokens) surrounded by low-entropy syntactic filler ("let", "then", "=", "\\frac"). Uniform spreading dilutes gradient signal at critical forks and introduces unnecessary noise on trivial tokens.
2. **Step-Level vs. Token-Level Granularity**:
   Single-token Detailed Balance ($\lambda = 0.0$) has high variance because individual token transitions have noisy log-probabilities. Sequence-level Trajectory Balance ($\lambda = 1.0$) lacks credit attribution. What is the optimal sub-trajectory scale? We hypothesize that natural reasoning steps ($\mathcal{S}_k$, separated by `\n\n` or punctuation) form the natural Markov boundaries for GFlowNet Sub-Trajectory Balance.
3. **All-Zero Group Failure Mode (Pass@G = 0)**:
   In the hardest mathematical problems, all $G$ student rollouts receive reward 0. In this regime, standard GRPO has zero variance and fails completely. We need to investigate how $g_{\text{consist}}$ should behave when contrastive outcome pairs are absent, ensuring that gold demonstration guidance is safely utilized rather than suppressed.
4. **Numerical Stability & Mixed-Precision Robustness**:
   Verifying that flow advantages remain bounded, scale-invariant, and numerically well-conditioned under bfloat16 / fp16 training.

---

## 2. Research Hypotheses

### Hypothesis 1: Entropy-Weighted SubTB (EW-SubTB) Preserves Flow Conservation with Superior SNR
- **Formulation**: Let $w_t \propto \mathcal{H}(\pi_{\text{old}}(\cdot \mid y_{<t}))^\gamma$ or $w_t \propto |\delta_t| + \epsilon$, with $\sum_{t=1}^L w_t = 1$.
- **Theorem (Generalized Mean Flow Conservation)**: For any non-negative weight distribution $w \in \Delta^{L-1}$, defining $\hat{A}_{\text{EW-TB}} = \sum_{t=1}^L w_t \hat{A}_{\text{DB}, t}$ preserves exact equivalence to the trajectory-level flow balance target while concentrating updates on decision boundaries.
- **Metric**: Signal-to-Noise Ratio (SNR) of policy gradients $\frac{|\mathbb{E}[\nabla_\theta \mathcal{L}]|}{\sigma(\nabla_\theta \mathcal{L})}$.

### Hypothesis 2: Step-Level SubTB (Step-SubTB) Outperforms Both Token-DB and Sequence-TB
- **Formulation**: Partition trajectory $\tau = (s_0, s_1, \dots, s_K)$ where each transition $s_{k-1} \to s_k$ corresponds to an entire reasoning step (e.g., sentence or derivation step).
- **Advantage**: Intermediate state flows $F(s_k)$ aggregate step-level log-probs $\sum_{t \in \text{step}_k} \log \pi(y_t)$, dampening token-level stochasticity while isolating buggy steps.

### Hypothesis 3: Adaptive Bayesian Prior for $g_{\text{consist}}$ in Degenerate (All-Zero) Groups
- When $\sum_i R_i = 0$ (all rollouts fail), pairwise verifier contrast is undefined. An adaptive prior based on teacher-reference KL divergence or external verification confidence should provide smooth regularization rather than arbitrary muting.

---

## 3. Planned Research Milestones

| Milestone | Description | Expected Output |
| :--- | :--- | :--- |
| **M1: Mathematical Proofs** | Formalize Generalized Flow Conservation & Variance Bounds | `THEORETICAL_PROOFS.md` |
| **M2: Edge-Case Audit** | Comprehensive stress-test of `c_flowbalance_adv.py` on edge cases | `test_edge_cases.py` + bug fixes |
| **M3: EW-SubTB Implementation** | Implement Entropy-Weighted & Surprise-Weighted SubTB | `ew_subtb_prototype.py` |
| **M4: Step-SubTB Implementation** | Implement Step-Boundary SubTB for long chain reasoning | `step_subtb_prototype.py` |
| **M5: Comparative Simulation** | Empirical benchmark comparing GRPO, TB, SubTB, EW-SubTB, Step-SubTB | Experiment logs, plots, analysis |
| **M6: Final Synthesis & Code Integration** | Synthesize findings and integrate verified breakthroughs into core codebase | Production PR ready code & report |

---
