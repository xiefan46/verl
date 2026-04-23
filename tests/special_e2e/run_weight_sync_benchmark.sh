#!/usr/bin/env bash
set -xeuo pipefail

# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Weight Sync Benchmark: Qwen3-30B-A3B on single node 8×H100
#
# Purpose: Measure parameter synchronization overhead between
#          training (Megatron) and inference (vLLM) engines.
#
# GPU allocation (6 GPUs):
#   - 4 GPU: Training (Megatron, tp2 ep2)
#   - 2 GPU: Inference (vLLM, tp2)
#
# Training and inference use DIFFERENT parallelism strategies
# to demonstrate resharding overhead and zero-redundancy potential.
#
# Usage:
#   cd /root/verl && bash tests/special_e2e/run_weight_sync_benchmark.sh

# Workaround for NVIDIA driver bug (r560-r575) SIGSEGV in ncclCuMemHostEnable()
export NCCL_CUMEM_ENABLE=0
export NCCL_CUMEM_HOST_ENABLE=0

############################ Quick Config ############################

ROLLOUT_NAME="vllm"
export VLLM_USE_V1=1

MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-30B-A3B"}

# Short sequences — we only care about weight sync, not training quality
MAX_PROMPT=512
MAX_RESPONSE_LENGTH=512
MAX_NUM_TOKENS=$(( MAX_PROMPT + MAX_RESPONSE_LENGTH + 1 ))

# Training: Megatron tp2 ep2 (4 GPU)
N_GPUS_TRAINING=4
TRAIN_TP=2
TRAIN_EP=2
TRAIN_CP=1
TRAIN_PP=1
TRAIN_VPP=null

# Inference: vLLM tp2 (2 GPU) — different parallelism from training
N_GPUS_ROLLOUT=2
INFER_TP=2

# Fully async parameters — minimal steps, focus on sync overhead
TOTAL_ROLLOUT_STEPS=256
STALENESS_THRESHOLD=0.5
TRIGGER_PARAMETER_SYNC_STEP=4

############################ Data Preparation ############################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

GSM8K_DIR="${HOME}/data/gsm8k"

if [ ! -f "${GSM8K_DIR}/train.parquet" ]; then
    echo "Preparing GSM8K dataset..."
    python3 "${VERL_ROOT}/examples/data_preprocess/gsm8k.py" --local_save_dir "$GSM8K_DIR"
fi

TRAIN_FILE="${GSM8K_DIR}/train.parquet"
TEST_FILE="${GSM8K_DIR}/test.parquet"

############################ Parameter Groups ############################

DATA=(
    data.train_files="${TRAIN_FILE}"
    data.val_files="${TEST_FILE}"
    data.prompt_key=prompt
    data.truncation='left'
    data.max_prompt_length=$MAX_PROMPT
    data.max_response_length=$MAX_RESPONSE_LENGTH
    data.train_batch_size=0
    data.gen_batch_size=1
    data.return_raw_chat=True
)

MODEL=(
    actor_rollout_ref.model.path="${MODEL_PATH}"
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.use_fused_kernels=False
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.optim.lr_warmup_steps=-1
    actor_rollout_ref.actor.optim.weight_decay=0.1
    actor_rollout_ref.actor.optim.lr_decay_style='constant'
    actor_rollout_ref.actor.ppo_mini_batch_size=32
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.loss_agg_mode="token-mean"
    actor_rollout_ref.actor.clip_ratio_low=0.2
    actor_rollout_ref.actor.clip_ratio_high=0.28
    actor_rollout_ref.actor.clip_ratio_c=10.0
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.kl_loss_coef=0.0
    actor_rollout_ref.actor.use_dynamic_bsz=True
    # Megatron parallelism
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${TRAIN_TP}
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${TRAIN_PP}
    actor_rollout_ref.actor.megatron.virtual_pipeline_model_parallel_size=${TRAIN_VPP}
    actor_rollout_ref.actor.megatron.context_parallel_size=${TRAIN_CP}
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=${TRAIN_EP}
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=1
    # Megatron optimizations
    actor_rollout_ref.actor.megatron.use_mbridge=True
    actor_rollout_ref.actor.megatron.use_dist_checkpointing=False
    # Offload everything to fit in 4 GPUs
    actor_rollout_ref.actor.megatron.param_offload=True
    actor_rollout_ref.actor.megatron.grad_offload=True
    actor_rollout_ref.actor.megatron.optimizer_offload=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=1.0
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True
    # Megatron kernel configs
    +actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_grouped_gemm=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_token_dispatcher_type="alltoall"
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_dtype=fp32
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=$ROLLOUT_NAME
    actor_rollout_ref.rollout.mode=async
    actor_rollout_ref.rollout.n=4
    actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.gpu_memory_utilization=0.70
    actor_rollout_ref.rollout.temperature=1.0
    actor_rollout_ref.rollout.top_p=1.0
    actor_rollout_ref.rollout.top_k=-1
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.disable_log_stats=False
    actor_rollout_ref.rollout.max_model_len=$MAX_NUM_TOKENS
    actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_NUM_TOKENS
    actor_rollout_ref.rollout.max_num_seqs=$MAX_NUM_TOKENS
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.tensor_model_parallel_size=${INFER_TP}
    actor_rollout_ref.rollout.enforce_eager=False
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.rollout.checkpoint_engine.backend='nccl'
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=2048
    actor_rollout_ref.hybrid_engine=False
)

REF=(
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.megatron.param_offload=True
    actor_rollout_ref.ref.megatron.use_dist_checkpointing=False
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${TRAIN_TP}
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${TRAIN_PP}
    actor_rollout_ref.ref.megatron.virtual_pipeline_model_parallel_size=${TRAIN_VPP}
    actor_rollout_ref.ref.megatron.context_parallel_size=${TRAIN_CP}
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=${TRAIN_EP}
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=1
)

ALGORITHM=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    algorithm.kl_ctrl.kl_coef=0.0
)

REWARD=(
    reward.reward_manager.name=dapo
    +reward.reward_kwargs.overlong_buffer_cfg.enable=False
    +reward.reward_kwargs.overlong_buffer_cfg.len=128
    +reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0
    +reward.reward_kwargs.overlong_buffer_cfg.log=False
    +reward.reward_kwargs.max_resp_len=${MAX_RESPONSE_LENGTH}
)

TRAINER=(
    trainer.logger='["console"]'
    trainer.project_name='weight-sync-benchmark'
    trainer.experiment_name="qwen3-30b-a3b-tp${TRAIN_TP}ep${TRAIN_EP}-to-tp${INFER_TP}"
    trainer.val_before_train=False
    trainer.save_freq=-1
    trainer.resume_mode=disable
    trainer.nnodes=1
    trainer.n_gpus_per_node=${N_GPUS_TRAINING}
    trainer.log_val_generations=0
    +trainer.use_legacy_worker_impl=disable
    trainer.total_epochs=1
    trainer.test_freq=-1
)

ASYNC_TRAINING=(
    rollout.nnodes=1
    rollout.n_gpus_per_node=${N_GPUS_ROLLOUT}
    rollout.total_rollout_steps=${TOTAL_ROLLOUT_STEPS}
    async_training.staleness_threshold=${STALENESS_THRESHOLD}
    async_training.partial_rollout=True
    async_training.trigger_parameter_sync_step=${TRIGGER_PARAMETER_SYNC_STEP}
    async_training.use_trainer_do_validate=False
)

############################ Launch ############################

echo "=============================================="
echo "Weight Sync Benchmark"
echo "=============================================="
echo "Model: ${MODEL_PATH}"
echo "Training: ${N_GPUS_TRAINING} GPU (Megatron tp${TRAIN_TP} ep${TRAIN_EP})"
echo "Inference: ${N_GPUS_ROLLOUT} GPU (vLLM tp${INFER_TP})"
echo "Total: $(( N_GPUS_TRAINING + N_GPUS_ROLLOUT )) GPUs"
echo "Sequence: ${MAX_PROMPT} prompt + ${MAX_RESPONSE_LENGTH} response"
echo "=============================================="

python3 -m verl.experimental.fully_async_policy.fully_async_main \
    --config-path=config \
    --config-name='fully_async_ppo_megatron_trainer.yaml' \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${ALGORITHM[@]}" \
    "${REWARD[@]}" \
    "${TRAINER[@]}" \
    "${ASYNC_TRAINING[@]}" \
    "$@"

echo "Weight sync benchmark completed."
