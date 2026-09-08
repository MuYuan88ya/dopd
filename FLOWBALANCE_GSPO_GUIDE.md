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

在传统的长序列数学强化学习中（如 GRPO），Token 级重要性采样比率容易累积高方差，且仅依据二值奖励（0/1）更新时，当候选探索全错时无法提供有效的向正确分布靠近的梯度。

本项目提出的方案将 **FlowBalance (Trajectory Balance 优势)** 与 **GSPO (序列级几何重要性加权)** 深度结合：

```
                    ┌───────────────────────────────┐
                    │      Prompt x & Rollouts G    │
                    └───────────────┬───────────────┘
                                    │
                  Has privileged demonstration y* ?
                                  /   \
                             Yes /     \ No (Fallback)
                                v       v
              ┌─────────────────────┐   ┌─────────────────────┐
              │ TB Advantage:       │   │ GRPO Advantage:     │
              │ Â_TB = 2(target-logp│   │ Â_GRPO = (R-μ)/σ    │
              └──────────┬──────────┘   └──────────┬──────────┘
                         │                         │
                         └────────────┬────────────┘
                                      │
                                      v
                    ┌───────────────────────────────┐
                    │      GSPO Policy Loss         │
                    │   s_i(θ) = exp(1/|y| Σ Δlogp) │
                    │   Dual-clipped objective      │
                    └───────────────────────────────┘
```

1. **上游优势估计（FlowBalance & SubTB Advantage Estimator）**：
   * **特权自教师分支（Privileged Group）**：当组内包含参考解答或特权教师演示 $y^*$ 时，基于 GFlowNet 流平衡理论构建等价优势。
     本系统原生支持 **Sub-Trajectory Balance（SubTB）连续流插值**：
     $$\hat{A}_{\text{SubTB}, t} = (1 - \lambda) \cdot \hat{A}_{\text{DB}, t} + \lambda \cdot \hat{A}_{\text{TB}}$$
     * **$\lambda = 1.0$（默认值，纯 Trajectory Balance）**：全序列共享标量优势，与原版 FlowBalance 严格一致：
       $$\hat{A}_{\text{TB}} = 2 \cdot (\text{flowsd\_target} - \text{seq\_logp}_{\text{old}}) / L^\rho$$
     * **$\lambda = 0.0$（纯 Detailed Balance，单步局部流平衡）**：将目标能量展开至各个 Token，赋予各 Token 独立的局部信用：
       $$\hat{A}_{\text{DB}, t} = 2 \cdot (\text{target}_t - \log \pi_{\text{old}}(y_t)) / L^\rho$$
     * **$0.0 < \lambda < 1.0$（SubTB 混合流平衡，如 $\lambda=0.5$）**：兼顾宏观路径连贯性与局部步骤归因能力。
     * **【核心数学定理：均值守恒】**：对任意 $\lambda \in [0, 1]$，序列内 Token 优势的均值严格守恒且恒等于原版 TB 标量优势：
       $$\frac{1}{L} \sum_{t=1}^L \hat{A}_{\text{SubTB}, t} \equiv \hat{A}_{\text{TB}}$$
   * **无特权回退分支（Fallback Group）**：当当前 prompt 无参考答案（纯探索样本）时，自动回退为标准 GRPO 结果优势：
     $$\hat{A}_i = \frac{R_i - \text{mean}(R)}{\text{std}(R) + \epsilon}$$
2. **下游策略损失（GSPO Policy Loss）**：
   * 传统的 Token 级重要性采样在 $L \ge 8192$ 长度下容易数值不稳定；GSPO 改用**序列级几何平均重要性比率**：
     $$s_i(\theta) = \left( \frac{\pi_\theta(y_i \mid x)}{\pi_{\theta_{\text{old}}}(y_i \mid x)} \right)^{\frac{1}{|y_i|}} = \exp\left( \frac{1}{|y_i|} \sum_{t=1}^{|y_i|} (\log \pi_\theta(y_{i,t}) - \log \pi_{\theta_{\text{old}}}(y_{i,t})) \right)$$
   * 采用双边裁剪：$\text{clip}(s_i(\theta), 1 - \epsilon_{\text{low}}, 1 + \epsilon_{\text{high}}) \cdot \hat{A}_{\text{SubTB}, t}$，不仅消除了 Token 乘积爆炸，更与 SubTB 形成“宏观序列控漂移 + 微观局部控梯度”的强力协同。

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

下表详细对比官方 `run_math_flowsd.sh` 与我们的 `run_math_flowbalance_gspo.sh` 的核心配置项：

| 参数项 | 官方 FlowSD 配置 | 本方案 (FlowBalance + GSPO) | 作用与解析 |
| :--- | :--- | :--- | :--- |
| `algorithm.adv_estimator` | `grpo` | **`flow_balance`** | 启用 Trajectory Balance 优势估计器；对无特权样本自动回退至 GRPO。 |
| `actor.policy_loss.loss_mode` | `flowsd` | **`gspo`** | 启用序列级几何重要性加权策略损失，替换 Token 级损失。 |
| `actor.loss_agg_mode` | `token-mean` | **`seq-mean-token-mean`** | GSPO 官方推荐的损失聚合模式，均衡不同长度长思维链对梯度的贡献。 |
| `algorithm.flowbalance_coef` | *(未定义)* | **`1.0`** | Trajectory Balance 方差项的缩放系数。 |
| `algorithm.log_rf_init` | *(未定义)* | **`0.0`** | 配分函数估算初值 $\log \tilde{R}_F$。 |
| `algorithm.subtb_lambda` | *(未定义)* | **`1.0`** *(可配 0.0 或 0.5)* | **SubTB 子轨迹流平衡插值旋钮**。<br>• 1.0 (默认)：纯 Trajectory Balance (原版 FlowBalance)；<br>• 0.0：纯 Detailed Balance (开启逐 Token 细粒度信用分配)；<br>• (0, 1)：SubTB 混合流平衡。无论何值均满足**均值守恒定理**。 |
| `data.max_prompt_length` | `2048` | `2048` | Prompt 输入最大长度截断。 |
| `data.max_response_length`| `8192` | `8192` (可配 16384) | 限制长思维链最大生成 Tokens。 |
| `actor_rollout_ref.rollout.n`| `8` | `8` | 每个 Prompt 采样的候选响应数 $G$。 |
| `actor.clip_ratio_low` | `0.2` | `0.2` | GSPO 截断下界 ($1 - \epsilon_{\text{low}}$)。 |
| `actor.clip_ratio_high`| `0.28` | `0.28` (或 `0.2`) | GSPO 截断上界 ($1 + \epsilon_{\text{high}}$)。 |
| `reward_model.reward_manager`| `custom_dapo` | `custom_dapo` | 基于 SymPy/MathVerify 的数学符号等价答案判定引擎。 |
| `trainer.test_freq` | `1` | `1` | 训练中 AIME-24 在线验证频率（每 1 个 step 验证一次）。 |
| `step180_val_enable` | `1` | `1` | 到达 180 步时自动触发后台 7 大 Benchmark 5-Seed 完整评测。 |

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
1. 将 `sp_size` 从 4 提升为 8（进一步平摊激活值）。
2. 调小 `train_prompt_mini_bsz`（例如由 128 减至 64）。
3. 调低 `rollout.gpu_memory_utilization`（例如由 0.60 降至 0.50）。

### Q2: 如何验证 FlowBalance 估计器是否正常工作？
运行项目内的集成单元测试：
```bash
python -m pytest tests/test_flowbalance_gspo_integration.py -v
```
测试通过即保证：
* 特权分支计算 TB 方差优势。
* 无特权分支（`mask=0`）完全回退为标准 GRPO 组内优势。
* GSPO 策略损失反向传播梯度正常流动。

### Q3: 为什么训练后评测必须使用多种子（5 Seeds）？
长思维链模型在温度 $T=0.6$ 下具有随机性，单个种子在 30 题规模的 AIME 上每做对/错 1 题就会引起 $3.33\%$ 的剧烈波动。采用 5-Seed 均值与方差（Mean ± Std）是目前大模型竞赛数学评测领域公认的标准学术规范。
