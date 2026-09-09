# Theoretical Foundations of FlowBalance & Generalized Sub-Trajectory Balance

**Author**: Antigravity Autonomous Research Agent  
**Date**: September 10, 2026  
**Artifact Path**: `experiments/autonomous_research_20260910/THEORETICAL_PROOFS.md`  

---

## 1. Mathematical Formulation of Autoregressive GFlowNets

Let an autoregressive language model generate a reasoning trajectory $\tau = (y_1, y_2, \dots, y_L) \in \mathcal{Y}^L$ given prompt $x$.
The state space forms a prefix tree:
- Initial state: $s_0 = x$.
- State at step $t$: $s_t = (x, y_{1:t})$.
- Terminal state: $s_L = (x, y_{1:L})$.

### 1.1 Determinism of Backward Transitions
In a prefix tree, every state $s_t$ has a **unique parent state** $s_{t-1} = (x, y_{1:t-1})$.
Therefore, the backward transition policy $P_B$ is deterministic:
$$P_B(s_{t-1} \mid s_t) \equiv 1.0, \quad \forall t \ge 1$$
This is a remarkable property of causal language generation that eliminates the need to learn or parameterize a backward policy.

### 1.2 Detailed Balance in LLM Generation
In GFlowNets, Detailed Balance (DB) equates forward and backward flows along every transition $(s_{t-1} \to s_t)$:
$$F(s_{t-1}) P_F(y_t \mid s_{t-1}) = F(s_t) P_B(s_{t-1} \mid s_t)$$
Substituting $P_B = 1$:
$$F(s_{t-1}) P_F(y_t \mid s_{t-1}) = F(s_t)$$
Taking logarithms:
$$\log P_F(y_t \mid s_{t-1}) = \log F(s_t) - \log F(s_{t-1})$$
**Key Insight**: In an autoregressive LLM, the policy log-probability is the **step difference of log state flows**!

### 1.3 Telescoping to Trajectory Balance (TB)
Summing this relationship over the full sequence $t = 1, \dots, L$:
$$\sum_{t=1}^L \log P_F(y_t \mid s_{t-1}) = \sum_{t=1}^L \big( \log F(s_t) - \log F(s_{t-1}) \big) = \log F(s_L) - \log F(s_0)$$
At the terminal state, flow conservation requires $F(s_L) = R(\tau)$, and the root flow is the partition function $F(s_0) = Z(x)$.
Thus:
$$\sum_{t=1}^L \log P_F(y_t \mid s_{t-1}) = \log R(\tau) - \log Z(x)$$
which is the exact Trajectory Balance condition:
$$\mathcal{E}_{\text{TB}}(\tau) = \log Z(x) + \sum_{t=1}^L \log P_F(y_t \mid s_{t-1}) - \log R(\tau) = 0$$

---

## 2. Generalized Sub-Trajectory Balance with Token Weighting

### 2.1 The Token Heterogeneity Problem
In standard C-FlowBalance, the Detailed Balance token target is constructed as:
$$\text{target}_t = \log \pi_{\text{ref}}(y_t) + \alpha g_{\text{consist}} \delta_t + \left(\frac{R}{\tau L^\rho} + b_{\text{group}}\right)$$
where the scalar trajectory return and baseline $(R / \tau + b)$ are uniformly distributed across all $L$ tokens ($w_t = 1/L$).
However, reasoning tokens exhibit extreme variance in semantic significance:
- Low-entropy syntax tokens ($w_t \approx 0$): boilerplate tokens, punctuation, formatting.
- High-entropy decision forks ($w_t \gg 0$): theorem selection, arithmetic calculations, case splits.

### 2.2 Generalized Token Weighting
Let $w = (w_1, w_2, \dots, w_L)$ be any normalized weighting vector on the $(L-1)$-simplex:
$$w \in \Delta^{L-1} \iff w_t \ge 0, \quad \sum_{t=1}^L w_t = 1$$
We define the **Generalized Token Target**:
$$\text{target}_t^{(w)} \triangleq \log \pi_{\text{ref}}(y_t) + \alpha g_{\text{consist}} \delta_t + w_t \cdot L \cdot \left( \frac{R}{\tau L^\rho} + b_{\text{group}} \right)$$

---

## 3. Core Theorems and Proofs

### Theorem 1 (Generalized Mean Flow Conservation)
**Statement**: For *any* normalized token weighting $w \in \Delta^{L-1}$ and for any sequence length $L \ge 1$:
$$\frac{1}{L} \sum_{t=1}^L \hat{A}_{\text{DB}, t}^{(w)} \equiv \hat{A}_{\text{TB}}$$
That is, the mean advantage across the sequence is strictly invariant to the choice of token weights $w$, and exactly equals the macro Trajectory Balance advantage.

**Proof**:
By definition:
$$\hat{A}_{\text{DB}, t}^{(w)} = 2 \cdot (\text{target}_t^{(w)} - \log \pi_{\text{old}}(y_t))$$
Summing across all tokens $t = 1, \dots, L$:
$$\sum_{t=1}^L \hat{A}_{\text{DB}, t}^{(w)} = 2 \sum_{t=1}^L \text{target}_t^{(w)} - 2 \sum_{t=1}^L \log \pi_{\text{old}}(y_t)$$
Expanding $\sum_{t=1}^L \text{target}_t^{(w)}$:
$$\sum_{t=1}^L \text{target}_t^{(w)} = \sum_{t=1}^L \log \pi_{\text{ref}}(y_t) + \alpha g_{\text{consist}} \sum_{t=1}^L \delta_t + L \left( \frac{R}{\tau L^\rho} + b_{\text{group}} \right) \sum_{t=1}^L w_t$$
Since $w \in \Delta^{L-1}$, $\sum_{t=1}^L w_t = 1$. Therefore:
$$\sum_{t=1}^L \text{target}_t^{(w)} = \sum_{t=1}^L \log \pi_{\text{ref}}(y_t) + \alpha g_{\text{consist}} \sum_{t=1}^L \delta_t + L \left( \frac{R}{\tau L^\rho} + b_{\text{group}} \right)$$
Notice that the right-hand side is **completely independent of $w$**!
Dividing by $L$:
$$\frac{1}{L} \sum_{t=1}^L \text{target}_t^{(w)} = \text{seq\_logp}_{\text{ref}} + \alpha g_{\text{consist}} G_T + \left( \frac{R}{\tau L^\rho} + b_{\text{group}} \right) \equiv \text{target}_{\text{TB}}$$
Consequently:
$$\frac{1}{L} \sum_{t=1}^L \hat{A}_{\text{DB}, t}^{(w)} = 2 \cdot (\text{target}_{\text{TB}} - \text{seq\_logp}_{\text{old}}) \equiv \hat{A}_{\text{TB}} \quad \blacksquare$$

---

### Theorem 2 (Gradient Variance Reduction via Information Concentration)
**Statement**: Let $\mathcal{I}_t \ge 0$ denote the information content (or decision importance) of token $t$. If non-decision tokens ($t \notin \mathcal{D}$) have $\nabla_\theta \log \pi_\theta(y_t \mid y_{<t})$ that is independent zero-mean noise with variance $\sigma_{\text{noise}}^2$, then setting $w_t \to 0$ for $t \notin \mathcal{D}$ strictly reduces the policy gradient estimator variance:
$$\mathbb{V}\left[ \nabla_\theta \mathcal{L}_{\text{policy}}^{(w)} \right] < \mathbb{V}\left[ \nabla_\theta \mathcal{L}_{\text{policy}}^{(\text{uniform})} \right]$$

**Proof Sketch**:
In policy gradient under GSPO:
$$\nabla_\theta \mathcal{L} = - \frac{1}{L} \sum_{t=1}^L s_i(\theta) \hat{A}_t \nabla_\theta \log \pi_\theta(y_t)$$
For non-decision tokens $t \in \mathcal{N}$, $\hat{A}_t^{(\text{uniform})}$ assigns non-zero random credit from $(R/\tau + b)$, causing $\hat{A}_t \nabla_\theta \log \pi_\theta(y_t)$ to inject variance $\mathcal{O}\left(\frac{|\mathcal{N}|}{L^2} \sigma_{\text{noise}}^2\right)$.
Under concentrated weighting $w_t = 0$ for $t \in \mathcal{N}$, $\text{target}_t = \log \pi_{\text{ref}}(y_t)$, so $\hat{A}_t = 2 (\log \pi_{\text{ref}} - \log \pi_{\text{old}}) \approx 0$ when the reference and student policies are close, effectively filtering out noise on filler tokens. $\blacksquare$

---

## 4. Proposed Concrete Weighting Schemes

1. **Surprise-Weighted (SW-SubTB)**:
   $$w_t = \frac{|\delta_t|^\gamma + \epsilon}{\sum_{k=1}^L (|\delta_k|^\gamma + \epsilon)}, \quad \delta_t = \log \pi_{\text{teacher}}(y_t) - \log \pi_{\text{ref}}(y_t)$$
   - When $\gamma = 0$: Reverts to uniform SubTB ($w_t = 1/L$).
   - When $\gamma > 0$: Focuses return credit on steps where teacher diverges from reference.

2. **Surprisal-Weighted (Surprisal-SubTB)**:
   $$w_t = \frac{(-\log \pi_{\text{old}}(y_t))^\gamma + \epsilon}{\sum_{k=1}^L (-\log \pi_{\text{old}}(y_k))^\gamma + \epsilon}$$
   - Measures token information content without needing a teacher vocabulary distribution.

3. **Step-Boundary SubTB (Step-SubTB)**:
   Partitions the sequence into $K$ reasoning steps: $\tau = (\mathcal{S}_1, \mathcal{S}_2, \dots, \mathcal{S}_K)$ where step boundaries are detected via `\n\n` or punctuation.
   Step weights are assigned uniformly per step, meaning tokens within shorter steps receive proportionally higher weight per token than tokens in repetitive paragraphs:
   $$w_t = \frac{1}{K \cdot |\mathcal{S}_{k(t)}|}$$
