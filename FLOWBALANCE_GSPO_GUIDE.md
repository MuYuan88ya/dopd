# FlowBalance + GSPO 训练与评测全流程指南

本文档介绍如何在 FlowBalance 代码库中使用 **FlowBalance 优势估计器** 与 **GSPO 策略损失（Group Sequence Policy Optimization）** 进行长思维链数学推理强化学习训练，包含数据准备、训练配置参数详解以及训练后 7 大主流数学 Benchmark 的多随机种子评测流水线。

---

## 目录
- [一、核心算法与架构原理](#一核心算法与架构原理)
- [二、数据准备 (Data Preparation)](#二数据准备-data-preparation)
- [三、模型准备与启动训练 (Training Pipeline)](#三模型准备与启动训练-training-pipeline)
  - [1. 启动脚本使用](#1-启动脚本使用)
  - [2. 关键参数对照与深度解析](#2-关键参数对照与深度解析)
  - [3. 显存与并行规模规划](#3-显存与并行规模规划)
- [四、模型合并与 Benchmark 评测 (Evaluation Pipeline)](#四模型合并与-benchmark-评测-evaluation-pipeline)
  - [1. FSDP 权重合并 (Model Merger)](#1-fsdp-权重合并-model-merger)
  - [2. 7 大数学 Benchmark 规约](#2-7-大数学-benchmark-规约)
  - [3. 多随机种子评估协议 (5-Seed Protocol)](#3-多随机种子评估协议-5-seed-protocol)
  - [4. 自动生成报表与指标解读](#4-自动生成报表与指标解读)
- [五、调试建议与常见问题 (FAQ)](#五调试建议与常见问题-faq)

---

## 一、核心算法与架构原理

在传统的长序列数学强化学习中（如 GRPO），Token 级重要性采样比率容易累积高方差；且仅依据最终二值奖励（0/1）更新时，当组内候选探索全错时无法提供有效朝向正确分布的微观引导。

本项目深度融合 **FlowBalance (Trajectory Balance / SubTB 优势)** 与 **GSPO (序列级几何重要性加权策略损失)**，并针对原生 FlowSD 的缺陷进行了重大理论重构，提供两个优势估计器：
* **`flow_balance`**：经典 Trajectory Balance 优势估计器（支持 SubTB 连续流插值，已修复序列长度重归一化陷阱）。
* **`c_flow_balance`（新升级）**：**Consistent Sub-Trajectory Balance** 估计器，引入 Pairwise AUC 真实性一致性门控（$g_{\text{consist}} \in [0, 1]$）、单形有界置信度（$\alpha \in [0, 1]$）与细粒度 Token 信用分配，杜绝伪正向梯度与幻觉自教师干扰。

### 1. 架构总览与数据流

```
                    ┌──────────────────────────────────────┐
                    │        Prompt x & Rollouts G         │
                    └──────────────────┬───────────────────┘
                                       │
                     Has privileged demonstration y* ?
                                     /   \
                                Yes /     \ No (Fallback)
                                   v       v
         ┌───────────────────────────────┐   ┌─────────────────────────┐
         │ FlowBalance / C-FlowBalance   │   │ GRPO Advantage:         │
         │ • g_consist Pairwise AUC Gate │   │ Â_GRPO = (R - μ) / σ    │
         │ • SubTB Credit Assignment     │   │ (Pure GSPO Fallback)    │
         │ • Scale-Invariant Â ~ O(1)    │   └────────────┬────────────┘
         └──────────────┬────────────────┘                │
                        │                                 │
                        └────────────────┬────────────────┘
                                         │
                                         v
                        ┌─────────────────────────────────┐
                        │        GSPO Policy Loss         │
                        │    s_i(θ) = exp(1/|y| Σ Δlogp)  │
                        │    Dual-clipped objective       │
                        └─────────────────────────────────┘
```

---

### 2. 估计器深度剖析：`c_flow_balance` vs `flow_balance`

#### (1) `c_flow_balance`（推荐）：Consistent Sub-Trajectory Balance
原生 FlowSD 使用启发式加权（$\beta_q, \eta_R$）与符号乘积 $\delta_t \cdot \text{sign}(A)$，当组内解答全错（$A < 0$）且教师生成存在负增益（$\delta_t < 0$）时，负负得正（$(-2) \times (-1) = +2$）会导致模型对错误步骤产生**错误的正向奖励**。

针对此问题，`c_flow_balance` 实现了严格的数学重构：
1. **Pairwise AUC 真实性一致性门控（$g_{\text{consist}} \in [0, 1]$）**：
   在每个 Prompt 的 Rollout 组内，计算教师序列偏好 $G_T$ 与真实判题结果 $R$ 的序对一致性：
   $$\text{AUC} = \frac{\sum_{i, j: R_i > R_j} \left[\mathbb{I}(G_{T, i} > G_{T, j}) + 0.5 \cdot \mathbb{I}(G_{T, i} = G_{T, j})\right]}{\sum_{i, j: R_i > R_j} 1}$$
   $$g_{\text{consist}} = \text{clip}(2 \cdot (\text{AUC} - 0.5), 0.0, 1.0)$$
   * 当自教师质量高、与真实胜负高度一致（$\text{AUC} \to 1.0$）时，$g_{\text{consist}} \to 1.0$ 全力注入微观引导；
   * 当自教师发生幻觉、给出与真实答案相悖的错误偏好（$\text{AUC} \le 0.5$）时，$g_{\text{consist}} = 0.0$，**安全静音（Safety Trip）**，绝不翻转符号误导策略。
2. **单形有界置信度参数化（Simplex Parameterization）**：
   将无界的超参数重构为单形坐标 $\alpha \in [0, 1]$ 与温度 $\tau > 0$：
   $$\text{target}_t = \log \pi_{\text{ref}}(y_t) + (\alpha \cdot g_{\text{consist}}) \cdot \delta_t + \left( \frac{R}{\tau \cdot L^\rho} + b_{\text{group}} \right)$$
3. **SubTB 连续流插值与 Token 细粒度信用分配**：
   * **Detailed Balance（局部流平衡，$\lambda=0.0$）**：
     $$\hat{A}_{\text{DB}, t} = 2 \cdot (\text{target}_t - \log \pi_{\text{old}}(y_t))$$
   * **Trajectory Balance（全局轨迹平衡，$\lambda=1.0$）**：
     $$\hat{A}_{\text{TB}} = \frac{1}{L} \sum_{t=1}^L \hat{A}_{\text{DB}, t}$$
   * **SubTB 凸组合（$\lambda \in [0, 1]$）**：
     $$\hat{A}_{\text{SubTB}, t} = (1 - \lambda) \cdot \hat{A}_{\text{DB}, t} + \lambda \cdot \hat{A}_{\text{TB}}$$
   * **【核心数学定理：均值守恒定理】**：对任意 $\lambda \in [0, 1]$，序列内所有 Token 的优势均值恒等于宏观轨迹平衡优势：
     $$\frac{1}{L} \sum_{t=1}^L \hat{A}_{\text{SubTB}, t} \equiv \hat{A}_{\text{TB}}$$

#### (2) `flow_balance`：经典 Trajectory Balance + SubTB
保留了标准 FlowBalance 结构，参数为 `beta_q` 与 `eta_R`，支持 `subtb_lambda` 调节微观与宏观信用。无特权数据时自动回退为纯 GSPO。

---

### 3. 关键修正：强化学习尺度不变性与长度归一化

在旧版实现中，由于沿用了传统独立 Actor 时代 FlowSD 回归损失的求导结果：
$$\mathcal{L}_{\text{FlowSD}} = \left( \frac{1}{L} \sum_{t=1}^L \log \pi_\theta(y_t) - \text{target} \right)^2 \implies \frac{\partial \mathcal{L}}{\partial \log \pi_t} = \frac{2}{L} \left( \frac{1}{L} \sum_{t=1}^L \log \pi_\theta(y_t) - \text{target} \right)$$
公式内部包含了一个 $1/L$。

**陷阱所在**：
在现代强化学习架构（VeRL GSPO）中，策略梯度的外层损失函数**本身已经**执行了 Token 级序列平均（`seq-mean-token-mean`）：
$$\mathcal{L}_{\text{GSPO}}(\theta) = - \mathbb{E}_i \left[ \frac{1}{|y_i|} \sum_{t=1}^{|y_i|} \text{clip}(s_i(\theta), 1-\epsilon_{\text{low}}, 1+\epsilon_{\text{high}}) \cdot \hat{A}_{i, t} \right]$$

如果内部优势估计 $\hat{A}_{i, t}$ 再次除以 $L$，就会导致整体梯度被二次除以长度（$1 / L^2$）。在长思维链推理任务中（$L \approx 2000 \sim 4000$）：
* 原版错误计算：$\hat{A}$ 坍缩至 $0.0005$，长思维链策略几乎**冻结不更新**；
* 修正后计算：$\hat{A} \sim \mathcal{O}(1)$，保持长短序列之间尺度不变（Scale-Invariant），梯度稳定传递！

本项目已在 `flowbalance_adv.py` 与 `c_flowbalance_adv.py` 中彻底修复该问题。

---

### 4. 下游策略损失（GSPO Policy Loss）

GSPO 采用序列级几何平均重要性采样比率 $s_i(\theta)$ 代替传统 Token 积乘：
$$s_i(\theta) = \exp\left( \frac{1}{|y_i|} \sum_{t=1}^{|y_i|} (\log \pi_\theta(y_{i,t}) - \log \pi_{\theta_{\text{old}}}(y_{i,t})) \right)$$
双边裁剪：$\text{clip}(s_i(\theta), 1 - \epsilon_{\text{low}}, 1 + \epsilon_{\text{high}}) \cdot \hat{A}_{\text{SubTB}, t}$，与 SubTB 优势形成“宏观序列控漂移 + 微观步骤赋信用”的高效协同。

---

## 二、数据准备 (Data Preparation)

### 1. 数据来源
* **训练集**：`BytedTsinghua-SIA/DAPO-Math-17k`（17,000 道精选奥林匹克竞赛难度数学题）。
* **训练中验证集**：`BytedTsinghua-SIA/AIME-2024`（AIME 2024 全部 30 道难题，格式化为带 `\boxed{}` 答案）。

### 2. 自动化下载与 Prompt 格式化
执行仓库内置的预处理流水线：

```bash
# 设置数据落地主目录（支持本地硬盘或共享存储如 /data 或 Ceph）
export DATA_ROOT="/path/to/your/data"
mkdir -p "${DATA_ROOT}/rl"

# 运行准备脚本
python3 -m recipe.sdpo.data.prepare_math_data
```

> **提示**：若在国内服务器或无法直连 HuggingFace，可设置镜像环境变量：
> ```bash
> export HF_ENDPOINT="https://hf-mirror.com"
> ```

### 3. 生成数据格式结构
预处理后的 Parquet 文件包含如下核心字段：
* `prompt`：包含题目描述，并在末尾附加统一指令：
  `\nPlease reason step by step, and put your final answer within \boxed{}.`
* `extra_info`：
  * `answer`：题目最终数值/表达式答案（用于奖励判分）。
  * `solution`：参考解答思维链（用于特权自教师 $y^*$ 演示构造）。

---

## 三、模型准备与启动训练 (Training Pipeline)

### 1. 启动脚本使用
我们提供了对齐官方训练标准的全套启动脚本 [recipe/flowbalance/run_math_flowbalance_gspo.sh](file:///g:/project/FlowBalance/recipe/flowbalance/run_math_flowbalance_gspo.sh)。

```bash
# 激活环境并配置路径
export DATA_ROOT="/path/to/your/data"
export MODEL_ROOT="/path/to/your/models"
export OUTPUT_ROOT="/path/to/your/output"

# 指定模型基座（例如 Qwen2.5-Math-7B 或 DeepSeek-R1-Distill-Qwen-7B）
export MODEL_PATH="${MODEL_ROOT}/Qwen2.5-Math-7B"

# 执行训练
bash recipe/flowbalance/run_math_flowbalance_gspo.sh
```

### 2. 关键参数对照与深度解析

下表详细对比官方 `run_math_flowsd.sh` 与本项目 (`flow_balance` / `c_flow_balance` + GSPO) 的核心配置项：

| 参数项 | 官方 FlowSD 配置 | 本方案 (`flow_balance` / `c_flow_balance`) | 作用与深度解析 |
| :--- | :--- | :--- | :--- |
| `algorithm.adv_estimator` | `grpo` | **`c_flow_balance`** *(或 `flow_balance`)* | • **`c_flow_balance`（推荐）**：带 Pairwise AUC 真实性门控 $g_{\text{consist}}$、单形置信度 $\alpha$ 与 SubTB 连续插值的自洽流平衡估计器；<br>• **`flow_balance`**：经典 Trajectory Balance + SubTB 优势估计器。<br>两者在无特权数据时均 100% 自动回退至 GRPO/GSPO。 |
| `algorithm.alpha` | *(未定义)* | **`0.5`** *(取值区间 $[0, 1]$)* | **`c_flow_balance` 专属**：单形凸坐标，平衡 RL 结果导向与自教师微观流平衡的相对置信度。 |
| `algorithm.tau` | *(未定义)* | **`0.1`** *(或 `1.0`)* | **`c_flow_balance` 专属**：探索温度，缩放终局奖励项。 |
| `algorithm.subtb_lambda` | *(未定义)* | **`1.0`** *(可配 0.0 或 0.5)* | **SubTB 子轨迹连续流平衡插值旋钮**：<br>• 1.0 (默认)：纯 Trajectory Balance (宏观序列流)；<br>• 0.0：纯 Detailed Balance (开启逐 Token 细粒度步骤信用分配)；<br>• (0, 1)：SubTB 混合流平衡。满足**均值守恒定理**。 |
| `algorithm.clip_B` | *(未定义)* | **`4.0`** | 自教师单步差分截断半径，防止长尾离群 Token 引起数值溢出。 |
| `algorithm.gate_no_context`| *(未定义)* | **`fallback_gspo`** | 当 Prompt 组内无有效特权解答时的策略：<br>• `fallback_gspo` (默认)：使用 GRPO 优势计算 GSPO 损失，充分利用 Rollout 数据；<br>• `drop`：将该样本优势置零。 |
| `actor.policy_loss.loss_mode` | `flowsd` | **`gspo`** | 启用序列级几何重要性加权策略损失，替换容易方差爆炸的 Token 乘积。 |
| `actor.loss_agg_mode` | `token-mean` | **`seq-mean-token-mean`** | GSPO 官方推荐的损失聚合模式，在序列间与序列内双重均匀加权。 |
| `data.max_prompt_length` | `2048` | `2048` | Prompt 输入最大长度截断。 |
| `data.max_response_length`| `8192` | `8192` (可配 16384) | 限制长思维链最大生成 Tokens。 |
| `actor_rollout_ref.rollout.n`| `8` | `8` | 每个 Prompt 采样的候选响应数 $G$。 |
| `actor.clip_ratio_low` | `0.2` | `0.2` | GSPO 截断下界 ($1 - \epsilon_{\text{low}}$)。 |
| `actor.clip_ratio_high`| `0.28` | `0.28` (或 `0.2`) | GSPO 截断上界 ($1 + \epsilon_{\text{high}}$)。 |
| `reward_model.reward_manager`| `custom_dapo` | `custom_dapo` | 基于 SymPy/MathVerify 的数学符号等价答案判定引擎。 |
| `trainer.test_freq` | `1` | `1` | 训练中 AIME-24 在线验证频率（每 1 个 step 验证一次）。 |
| `step180_val_enable` | `1` | `1` | 到达 180 步时自动触发后台 7 大 Benchmark 5-Seed 完整评测。 |

#### 快速切换配置示例

在启动脚本中，只需修改 `algorithm.adv_estimator` 及相应超参即可一键切换：

```bash
# 方案 A: 使用升级版 C-FlowBalance (AUC Gating + SubTB 细粒度赋权)
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=c_flow_balance \
    algorithm.alpha=0.5 \
    algorithm.tau=0.1 \
    algorithm.subtb_lambda=0.5 \
    actor.policy_loss.loss_mode=gspo \
    ...

# 方案 B: 使用经典 Trajectory Balance (宏观序列流平衡)
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=flow_balance \
    algorithm.beta_q=0.5 \
    algorithm.eta_R=1.0 \
    algorithm.subtb_lambda=1.0 \
    actor.policy_loss.loss_mode=gspo \
    ...
```

### 3. 显存与并行规模规划（以单机 8 卡为基准）
本方案默认按照**单机 8 卡（如 8 × A100/H800/H20）**环境进行开箱即用配置：
* **节点与卡数**：`NNODES=1`, `N_GPUS_PER_NODE=8`（总计 8 张 GPU）。
* **序列并行（Sequence Parallelism）**：设置 `sp_size=2`，数据并行度 $DP = 8 / 2 = 4$。长序列 Attention 计算被拆分在 2 张卡间，极大释放长上下文显存。
* **批次分配**：默认 `train_prompt_bsz=128`，每个 Prompt 采样 $G=8$ 条回答，总 Rollout 样本数 $128 \times 8 = 1024$；4 个 DP 组每组分得 256 条样本，完美配合 `train_prompt_mini_bsz=64` 切分为 4 个 Mini-batch 进行稳定更新。
* **显存卸载（Offload）**：默认启用 FSDP 参数和优化器状态卸载：
  `fsdp_config.param_offload=True`，`fsdp_config.optimizer_offload=True`。
* **vLLM 推理利用率**：`rollout.gpu_memory_utilization=0.55`，为长思维链保留充裕的 KV Cache 空间。

---

## 四、模型合并与 Benchmark 评测 (Evaluation Pipeline)

训练保存的 Checkpoint 默认是分布式 FSDP 切片权重（例如 `actor/model_world_size_8_rank_*.pt`）。必须先将其合并为标准 HuggingFace 格式，才能被 `vLLM` 加载评估。

### 1. FSDP 权重合并 (Model Merger)
使用 VeRL 提供的 `verl.model_merger` 模块执行：

```bash
python3 -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "${CKPTS_DIR}/global_step_180/actor" \
    --target_dir "${CKPTS_DIR}/merged_hf_step180" \
    --use_cpu_initialization
```
> 执行后，`${CKPTS_DIR}/merged_hf_step180/` 将生成包含 `config.json`、`tokenizer.json` 及 `model.safetensors` 的完整 HuggingFace 模型。

---

### 2. 7 大数学 Benchmark 规约
所有评测集预先处理并存放在 `evaluation/math/data/processed/`：

1. **AIME 24** (`aime24.parquet`): 30 题，美国数学邀请赛 2024。
2. **AIME 25** (`aime25.parquet`): 30 题，最新邀请赛真题。
3. **AIME 26** (`aime26.parquet`): 盲测高难度竞赛题。
4. **HMMT 25** (`hmmt25.parquet`): 哈佛-麻省理工数学锦标赛 2025。
5. **Minerva Math** (`minerva_math.parquet`): 综合高等数学。
6. **MATH-500** (`math500.parquet`): 从 MATH 数据集抽取的 500 道经典难题。
7. **OlympiadBench** (`olympiadbench.parquet`): 中英双语奥林匹克竞赛集。

---

### 3. 多随机种子评估协议 (5-Seed Protocol)

为了消除高方差，FlowBalance 评测体系执行严格的 **5 种子（Seed 0, 1, 2, 3, 4）统计**：
* **单样本能力（Pass@1, $n=1$）**：在所有 7 个 Benchmark 上运行。
* **多次采样覆盖率（Pass@16, $n=16$）**：专门针对 AIME 24/25/26 运行。
* **生成参数**：`temperature=0.6`, `top_p=0.95`, `top_k=20`, `max_tokens=38912`。

#### 执行自动化评测 Watcher
启动脚本会自动在后台监听 step-180 checkpoint 完成并触发评测，亦可手动触发：

```bash
RUN_DIR="${CKPTS_DIR}" \
EXPERIMENT_NAME="${EXP_NAME}" \
VAL_STEP=180 \
VAL_SEEDS="0 1 2 3 4" \
bash recipe/flowsd/submit_step180_val.sh
```

---

### 4. 自动生成报表与指标解读

评测完成后，[ray_flowsd_step180_val.py](file:///g:/project/FlowBalance/evaluation/math/scripts/ray_flowsd_step180_val.py) 会在输出目录中输出聚合结果：

* **`pass1_n1_mean_std_percent.csv`**：Pass@1 在 5 个种子下的均值与标准差。
* **`aime_pass16_n16_mean_std_percent.csv`**：AIME 系列的 Pass@16 表现。
* **`results_mean_std.md`**：格式化 Markdown 报告。

**报告输出示例：**
| Dataset | Setting | Metric | Mean (%) | Std Dev (%) | Per-Seed Breakdown (0~4) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **AIME 24** | $n=1$ | Pass@1 | **77.33** | ± 2.11 | [76.7, 76.7, 80.0, 73.3, 80.0] |
| **AIME 24** | $n=16$| Pass@16| **93.33** | ± 0.00 | [93.3, 93.3, 93.3, 93.3, 93.3] |
| **AIME 25** | $n=1$ | Pass@1 | **63.33** | ± 1.82 | [63.3, 60.0, 66.7, 63.3, 63.3] |
| **MATH-500** | $n=1$ | Pass@1 | **94.60** | ± 0.49 | [94.2, 95.0, 94.4, 94.6, 94.8] |
| **HMMT 25** | $n=1$ | Pass@1 | **68.80** | ± 1.55 | [67.5, 70.0, 70.0, 67.5, 69.0] |

---

## 五、调试建议与常见问题 (FAQ)

### Q1: 训练初期出现 OOM（显存溢出）如何解决？
1. 将 `sp_size` 从 2 或 4 提升为 8（序列并行平摊长思维链的中间激活值）。
2. 调小 `train_prompt_mini_bsz`（例如由 128 减至 64）。
3. 调低 `rollout.gpu_memory_utilization`（例如由 0.60 降至 0.50，为 KV Cache 和梯度预留空间）。

### Q2: 如何验证估计器与策略损失是否正常工作？
运行仓库内置的端到端自动化集成单元测试集：
```bash
# 1. 验证经典 FlowBalance + GSPO 端到端闭环
python tests/test_flowbalance_gspo_integration.py

# 2. 验证 C-FlowBalance + AUC Gating + SubTB 均值守恒
python tests/test_c_flowbalance_integration.py
```
测试全面覆盖：
* 特权自教师分支计算与非特权回退至 GRPO/GSPO 的无缝切换；
* Pairwise AUC 一致性门控在好教师（$g_{\text{consist}}=1$）、中性教师（$g_{\text{consist}}=0$）和有毒教师（$g_{\text{consist}}=0$）下的**熔断安全机制**；
* 细粒度 Token 信用分配对关键突破步骤与无效步骤的区分能力；
* SubTB 在任意 $\lambda \in [0, 1]$ 下的严格**均值流守恒**（$\text{diff} < 10^{-5}$）；
* GSPO 策略损失反向传播与有效梯度的正常流动。

### Q3: 为什么移除了 Advantage 内部冗余的长度除法（除以 $L$）？
在原版 FlowSD 的独立 actor 脚本中，损失函数是均方误差回归：
$$\mathcal{L} = \left(\frac{1}{L}\sum_{t=1}^L \log \pi - \text{target}\right)^2$$
其对单个 Token 对数概率的导数必然包含链式法则带来的内部因子 $\frac{2}{L}$。
但在标准强化学习中，VeRL 的 `compute_policy_loss_gspo` 损失函数外层**已经统一执行了序列 Token 平均**（`loss_agg_mode="seq-mean-token-mean"`，即外层已有 $\frac{1}{|y_i|}\sum_t \dots$）。
如果优势估计器 $\hat{A}_t$ 内部也除以 $L$，总体梯度就会变为 $\mathcal{O}(1/L^2)$。当模型生成长思维链（例如 $L=2000$）时，优势值直接缩水至 $0.0005$，导致策略无法更新；而短序列（$L=50$）优势却为 $0.02$（相差 40 倍）。
**修正后的优势估计器 $\hat{A}_t$ 保持为 $\mathcal{O}(1)$ 量纲**，使得不同长度的思维链在策略梯度下获得公平的更新权重。

### Q4: 自教师生成错误或幻觉推理时，C-FlowBalance 如何确保策略不被误导？
原版 FlowSD 使用 $\delta_t \cdot \text{sign}(A)$。当组内全错（$A < 0$）且自教师也走偏（$\delta_t < 0$）时，两负相乘为正，策略反而会强化这个错误步骤。
`c_flow_balance` 引入 **Group Pairwise AUC 真实性一致性门控**：
* 只有当自教师在整组 Rollout 中的偏好排序与真实验题判分（$R \in \{0, 1\}$）呈现正相关（$\text{AUC} > 0.5$）时，才激活门控系数 $g_{\text{consist}} = 2 \cdot (\text{AUC} - 0.5)$；
* 一旦自教师偏好与真实答案不一致或甚至反转（$\text{AUC} \le 0.5$），门控自动置零（$g_{\text{consist}} = 0$），**直接熔断自教师信号**，只保留客观强化学习结果优势更新，从根源上杜绝了对错误步骤的正向奖励。

### Q5: SubTB 的连续插值参数 $\lambda$ 该如何选择？
* **$\lambda = 1.0$（Trajectory Balance）**：整条轨迹所有 Token 享有相同优势。方差最小，但在非常长的思维链中缺乏对单步推导好坏的辨别力。
* **$\lambda = 0.0$（Detailed Balance）**：完全依据每一步相对于自教师目标的差距计算 Token 级局部优势。归因能力最强，适合需要细粒度修正推理步骤的场景。
* **$\lambda = 0.5$（SubTB 混合平衡，推荐探索）**：兼顾宏观路径连贯性与微观步骤指导，由于严格满足均值守恒定理，不会破坏全局收敛性。

### Q6: 为什么训练后评测必须使用多种子（5 Seeds）？
长思维链模型在探索温度 $T=0.6$ 下具有较高随机性。单个种子在 30 题规模的 AIME 上，仅仅做对或做错 1 道题就会引起 $3.33\%$ 的剧烈百分比波动。采用 5-Seed 均值与方差（Mean ± Std）是目前大模型竞赛数学评测领域公认的防过拟合学术规范。
