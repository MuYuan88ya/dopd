#!/usr/bin/env bash
# ==============================================================================
# FlowBalance + GSPO Math Run Script
# 
# Combines:
#   1. FlowBalance Advantage Estimator (TB advantage for privileged demos,
#      graceful fallback to GRPO outcome advantage for unprivileged groups).
#   2. GSPO Policy Loss (Sequence-level geometric importance sampling ratio
#      with dual-side clipping).
#
# Aligned with the official FlowSD Run28 training setup and parameters.
# ==============================================================================
set -xeuo pipefail

# ------------------------------------------------------------------------------
# 1. Project & Experiment Metadata
# ------------------------------------------------------------------------------
project_name=${PROJECT_NAME:-verl-flowbalance}
exp_name=${EXP_NAME:-FlowBalance-GSPO-Qwen3-8B-math-dapo17k-lr1e6}

RAY_ADDRESS=${RAY_ADDRESS:-"http://localhost:8265"}
WORKING_DIR=${WORKING_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/runtime_env.yaml"}
NNODES=${NNODES:-1}
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-8}

# ------------------------------------------------------------------------------
# 2. Paths Configuration
# ------------------------------------------------------------------------------
DATA_ROOT=${DATA_ROOT:-"/apdcephfs_gy4/share_303378103/user/audenhuang/data"}
MODEL_ROOT=${MODEL_ROOT:-"/apdcephfs_gy4/share_303378103/user/audenhuang/models"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"/apdcephfs_gy4/share_303378103/user/audenhuang/output"}

TRAIN_FILE=${TRAIN_FILE:-"${DATA_ROOT}/rl/train_dapo17k.parquet"}
TEST_FILE=${TEST_FILE:-"${DATA_ROOT}/rl/aime24_30_boxed.parquet"}
MODEL_PATH=${MODEL_PATH:-"${MODEL_ROOT}/Qwen3-8B"}
CKPTS_DIR=${CKPTS_DIR:-"${OUTPUT_ROOT}/${project_name}/${exp_name}"}

# ------------------------------------------------------------------------------
# 3. Context Lengths & Parallelism
# ------------------------------------------------------------------------------
max_prompt_length=${MAX_PROMPT_LENGTH:-$((1024 * 2))}       # 2048
max_response_length=${MAX_RESPONSE_LENGTH:-$((1024 * 8))}   # 8192
max_model_len=${MAX_MODEL_LEN:-$((max_prompt_length + max_response_length))} # 10240
max_reprompt_len=${MAX_REPROMPT_LEN:-${max_model_len}}

sp_size=${SP_SIZE:-2}       # Ulysses sequence parallel size (8 GPUs / SP=2 -> DP=4)
gen_tp=${GEN_TP:-1}         # Rollout tensor parallel size
use_dynamic_bsz=True
offload=True

# ------------------------------------------------------------------------------
# 4. Batch Sizing & Optimization (Single-Machine 8-GPU Default)
# ------------------------------------------------------------------------------
n_resp_per_prompt=${N_RESP_PER_PROMPT:-8}
train_prompt_bsz=${TRAIN_PROMPT_BSZ:-128}
train_prompt_mini_bsz=${PPO_MINI_BATCH_SIZE:-64}
rollout_batch_size=$((train_prompt_bsz * n_resp_per_prompt))
lr=${LR:-1e-6}

# FlowBalance Advantage & GSPO Specific Hyperparameters
flowbalance_coef=${FLOWBALANCE_COEF:-1.0}
log_rf_init=${FLOWBALANCE_LOG_RF_INIT:-0.0}
subtb_lambda=${SUBTB_LAMBDA:-1.0}  # 1.0 = Trajectory Balance (default); 0.0 = Detailed Balance; (0, 1) = SubTB
clip_ratio_low=${CLIP_RATIO_LOW:-0.2}
clip_ratio_high=${CLIP_RATIO_HIGH:-0.28}

total_gpus=$((NNODES * N_GPUS_PER_NODE))
if (( total_gpus % sp_size != 0 )); then
    echo "ERROR: total_gpus=${total_gpus} must be divisible by SP_SIZE=${sp_size}"
    exit 1
fi
dp_size=$((total_gpus / sp_size))

echo "Distributed setup: total_gpus=${total_gpus}, SP=${sp_size}, DP=${dp_size}, train_prompt_bsz=${train_prompt_bsz}, rollout_batch=${rollout_batch_size}"

# ------------------------------------------------------------------------------
# 5. Training Loop, Validation & Checkpoint Frequencies
# ------------------------------------------------------------------------------
total_training_steps=${TOTAL_TRAINING_STEPS:-null}
test_freq=${TEST_FREQ:-1}
save_freq=${SAVE_FREQ:-10}
resume_mode=${RESUME_MODE:-disable}
resume_from_path=${RESUME_FROM_PATH:-null}
rollout_data_dir=${ROLLOUT_DATA_DIR:-null}
log_val_generations=${LOG_VAL_GENERATIONS:-4}
nccl_timeout=${NCCL_TIMEOUT:-600}

# Full external post-training benchmark validation (Step 180 by default)
step180_val_enable=${STEP180_VAL_ENABLE:-1}
step180_val_step=${STEP180_VAL_STEP:-180}
step180_val_output_dir=${STEP180_VAL_OUTPUT_DIR:-${CKPTS_DIR}/val/step_${step180_val_step}_math_benchmarks_5seeds}

cd "${WORKING_DIR}"
echo "Working dir: ${WORKING_DIR}"
echo "Model:       ${MODEL_PATH}"
echo "Train:       ${TRAIN_FILE}"
echo "Test:        ${TEST_FILE}"
echo "Ckpts:       ${CKPTS_DIR}"
echo "FlowBalance + GSPO Setup: adv_estimator=flow_balance, loss_mode=gspo, coef=${flowbalance_coef}, log_rf_init=${log_rf_init}, lr=${lr}"

if [[ "${step180_val_enable}" == "1" ]]; then
    if (( save_freq <= 0 || step180_val_step % save_freq != 0 )); then
        echo "ERROR: step-${step180_val_step} validation requires SAVE_FREQ to be a positive divisor of ${step180_val_step}; got SAVE_FREQ=${save_freq}"
        exit 1
    fi
    if [[ "${total_training_steps}" != "null" ]] && (( total_training_steps < step180_val_step )); then
        echo "ERROR: TOTAL_TRAINING_STEPS=${total_training_steps} is smaller than STEP180_VAL_STEP=${step180_val_step}"
        exit 1
    fi
fi

# Preflight check if enabled
if [[ "${SKIP_PREFLIGHT:-0}" != "1" && -f "${WORKING_DIR}/recipe/flowsd/preflight_math_flowsd.py" ]]; then
    python3 "${WORKING_DIR}/recipe/flowsd/preflight_math_flowsd.py" \
        --repo-root "${WORKING_DIR}" \
        --train-file "${TRAIN_FILE}" \
        --test-file "${TEST_FILE}" \
        --model-path "${MODEL_PATH}" \
        --reward-fn-path "core/utils/reward_score/sdpo_math_feedback_score.py" \
        --reward-fn-name "compute_score" \
        --nnodes "${NNODES}" \
        --gpus-per-node "${N_GPUS_PER_NODE}" \
        --sp-size "${sp_size}" \
        --train-batch-size "${train_prompt_bsz}" \
        --rollout-n "${n_resp_per_prompt}" \
        --ppo-mini-batch-size "${train_prompt_mini_bsz}" \
        --max-prompt-length "${max_prompt_length}" \
        --max-response-length "${max_response_length}" \
        --max-model-len "${max_model_len}" \
        --max-train-rows 50000
fi

# ------------------------------------------------------------------------------
# 6. Ray Training Job Submission
# ------------------------------------------------------------------------------
ray job submit --address="${RAY_ADDRESS}" --no-wait --runtime-env="${RUNTIME_ENV}" \
    -- python3 -m recipe.flowsd.main_flowsd \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    data.gen_batch_size=${train_prompt_bsz} \
    data.filter_overlong_prompts=True \
    data.shuffle=True \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.nccl_timeout=${nccl_timeout} \
    \
    algorithm.adv_estimator=flow_balance \
    algorithm.norm_adv_by_std_in_grpo=True \
    algorithm.use_kl_in_reward=False \
    algorithm.rollout_correction.rollout_is=null \
    ++algorithm.flowbalance_coef=${flowbalance_coef} \
    ++algorithm.log_rf_init=${log_rf_init} \
    ++algorithm.subtb_lambda=${subtb_lambda} \
    \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.policy_loss.loss_mode=gspo \
    actor_rollout_ref.actor.loss_agg_mode="seq-mean-token-mean" \
    \
    actor_rollout_ref.actor.self_distillation.success_reward_threshold=0.5 \
    actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=True \
    actor_rollout_ref.actor.self_distillation.max_reprompt_len=${max_reprompt_len} \
    \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.actor.optim.lr=${lr} \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${max_model_len} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${max_model_len} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${max_model_len} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=${max_model_len} \
    actor_rollout_ref.rollout.max_model_len=${max_model_len} \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    \
    reward_model.reward_manager=custom_dapo \
    reward_model.use_reward_loop=False \
    reward_manager.source=register \
    reward_manager.name=custom_dapo \
    reward_manager.module.path=core/workers/reward_manager/custom_dapo.py \
    reward_manager.module.name=CustomDAPORewardManager \
    custom_reward_function.path=core/utils/reward_score/sdpo_math_feedback_score.py \
    custom_reward_function.name=compute_score \
    ++reward_model.reward_kwargs.max_resp_len=${max_response_length} \
    \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=False \
    trainer.test_freq=${test_freq} \
    trainer.log_val_generations=${log_val_generations} \
    trainer.save_freq=${save_freq} \
    trainer.total_epochs=500 \
    trainer.total_training_steps=${total_training_steps} \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.validation_data_dir="${CKPTS_DIR}/val/" \
    trainer.rollout_data_dir=${rollout_data_dir} \
    trainer.resume_mode=${resume_mode} \
    trainer.resume_from_path=${resume_from_path}

# ------------------------------------------------------------------------------
# 7. Background Step Validation Watcher (7 Benchmarks, 5 Seeds)
# ------------------------------------------------------------------------------
if [[ "${step180_val_enable}" == "1" ]]; then
    echo "Submitting FlowBalance step-${step180_val_step} benchmark validation watcher"
    RUN_DIR="${CKPTS_DIR}" \
    EXPERIMENT_NAME="${exp_name}" \
    VAL_ALGORITHM="flowbalance_gspo" \
    VAL_MODEL_TAG_PREFIX="flowbalance_gspo" \
    VAL_STEP="${step180_val_step}" \
    VAL_OUTPUT_DIR="${step180_val_output_dir}" \
    VAL_SEEDS="${STEP180_VAL_SEEDS:-0 1 2 3 4}" \
    VAL_TIMEOUT_SECONDS="${STEP180_VAL_TIMEOUT_SECONDS:-604800}" \
    VAL_KEEP_MERGED_MODEL="${STEP180_VAL_KEEP_MERGED_MODEL:-0}" \
    VAL_TEMPERATURE="${STEP180_VAL_TEMPERATURE:-0.6}" \
    VAL_TOP_P="${STEP180_VAL_TOP_P:-0.95}" \
    VAL_TOP_K="${STEP180_VAL_TOP_K:-20}" \
    VAL_MAX_TOKENS="${STEP180_VAL_MAX_TOKENS:-38912}" \
    VAL_MAX_MODEL_LEN="${STEP180_VAL_MAX_MODEL_LEN:-40960}" \
    VAL_BATCH_SIZE="${STEP180_VAL_BATCH_SIZE:-16}" \
    VAL_GPU_MEMORY_UTILIZATION="${STEP180_VAL_GPU_MEMORY_UTILIZATION:-0.80}" \
    RAY_ADDRESS="${RAY_ADDRESS}" \
    RUNTIME_ENV="${RUNTIME_ENV}" \
    bash "${WORKING_DIR}/recipe/flowsd/submit_step180_val.sh"
fi
