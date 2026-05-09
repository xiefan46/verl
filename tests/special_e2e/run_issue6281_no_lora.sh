#!/usr/bin/env bash
set -xeuo pipefail

# Variant of run_issue6281_lora_no_opd.sh with LoRA fully stripped.
# Goal: determine whether issue #6281's CUDA illegal memory access requires
# LoRA, or surfaces with just fully_async + Qwen3.5 + FSDP1 alone.
#
# Same components: fully_async + Qwen3.5-2B + FSDP1 + partial_rollout + GRPO
# Removed: ALL LoRA configs (lora_rank, lora_alpha, lora.merge, target_modules,
#          exclude_modules) — full-parameter training.
#
# GPU allocation (3 GPUs — same as no-LoRA bisect):
#   - 1 GPU: Rollout (vLLM async, TP=1)
#   - 2 GPU: Training (FSDP world_size=2, full-param)
#
# Note: full-param Qwen3.5-2B with Adam needs ~5GB params + ~5GB grads
# + ~16GB optimizer = ~26GB per replica. With FSDP world=2, ~13GB per GPU.
# Plus activations. Should fit on H100 (80GB). If OOM, enable offload via:
#   actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
#
# Usage:
#   cd /root/verl && bash tests/special_e2e/run_issue6281_no_lora.sh

# Workaround for NVIDIA driver bug (r560-r575) causing SIGSEGV in ncclCuMemHostEnable()
export NCCL_CUMEM_ENABLE=0
export NCCL_CUMEM_HOST_ENABLE=0

############################ Quick Config ############################

ROLLOUT_NAME="vllm"
export VLLM_USE_V1=1

MODEL_ID=${MODEL_ID:-Qwen/Qwen3.5-2B}
MODEL_PATH=${MODEL_PATH:-${HOME}/models/${MODEL_ID}}

MAX_PROMPT=2048
MAX_RESPONSE_LENGTH=1024
MAX_NUM_TOKENS=$(( MAX_PROMPT + MAX_RESPONSE_LENGTH + 1 ))

N_GPUS_ROLLOUT=1
N_GPUS_TRAINING=2
TOTAL_ROLLOUT_STEPS=${TOTAL_ROLLOUT_STEPS:-128}
N_RESP_PER_PROMPT=8                   # GRPO group size

STALENESS_THRESHOLD=0.5
TRIGGER_PARAMETER_SYNC_STEP=4

############################ Data Paths ############################

GSM8K_TRAIN="${HOME}/data/gsm8k/train.parquet"
GSM8K_TEST="${HOME}/data/gsm8k/test.parquet"

############################ Parameter Groups ############################

DATA=(
    data.train_files="$GSM8K_TRAIN"
    data.val_files="$GSM8K_TEST"
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
)

ACTOR=(
    actor_rollout_ref.actor.strategy=fsdp
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.optim.lr_warmup_steps=-1
    actor_rollout_ref.actor.optim.weight_decay=0.1
    actor_rollout_ref.actor.ppo_mini_batch_size=8
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.loss_agg_mode="token-mean"
    actor_rollout_ref.actor.clip_ratio_low=0.2
    actor_rollout_ref.actor.clip_ratio_high=0.28
    actor_rollout_ref.actor.clip_ratio_c=10.0
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.kl_loss_coef=0.0
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=$ROLLOUT_NAME
    actor_rollout_ref.rollout.mode=async
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.n=${N_RESP_PER_PROMPT}
    actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.gpu_memory_utilization=0.80
    actor_rollout_ref.rollout.temperature=1.0
    actor_rollout_ref.rollout.top_p=1.0
    actor_rollout_ref.rollout.top_k=-1
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.disable_log_stats=False
    actor_rollout_ref.rollout.max_model_len=$MAX_NUM_TOKENS
    actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_NUM_TOKENS
    actor_rollout_ref.rollout.max_num_seqs=$MAX_NUM_TOKENS
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7
    actor_rollout_ref.rollout.val_kwargs.top_k=-1
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
    actor_rollout_ref.rollout.val_kwargs.n=1
    actor_rollout_ref.rollout.agent.num_workers=1
    actor_rollout_ref.rollout.checkpoint_engine.backend='nccl'
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=1024
    actor_rollout_ref.rollout.enforce_eager=False
)

ALGORITHM=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    algorithm.kl_ctrl.kl_coef=0.0
)

REWARD=(
    reward.reward_manager.name=dapo
    +reward.reward_kwargs.overlong_buffer_cfg.enable=True
    +reward.reward_kwargs.overlong_buffer_cfg.len=128
    +reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0
    +reward.reward_kwargs.overlong_buffer_cfg.log=False
    +reward.reward_kwargs.max_resp_len=${MAX_RESPONSE_LENGTH}
)

CRITIC=(
    critic.enable=False
)

TRAINER=(
    trainer.logger='["console"]'
    trainer.project_name='verl-test-fully-async-no-lora'
    trainer.experiment_name="qwen3.5-2b-fully-async-no-lora"
    trainer.val_before_train=False
    trainer.save_freq=-1
    trainer.resume_mode=disable
    trainer.nnodes=1
    trainer.n_gpus_per_node=${N_GPUS_TRAINING}
    trainer.log_val_generations=10
    +trainer.use_legacy_worker_impl=disable
    trainer.total_epochs=2
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

echo "Running fully_async_policy + Qwen3.5 (no LoRA, no OPD) — issue #6281 bisect"
echo "Model: ${MODEL_PATH}"
echo "GPUs: ${N_GPUS_ROLLOUT} rollout + ${N_GPUS_TRAINING} training (full-param)"

python3 -m verl.experimental.fully_async_policy.fully_async_main \
    --config-path=config \
    --config-name='fully_async_ppo_trainer.yaml' \
    actor_rollout_ref.hybrid_engine=False \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${ALGORITHM[@]}" \
    "${REWARD[@]}" \
    "${CRITIC[@]}" \
    "${TRAINER[@]}" \
    "${ASYNC_TRAINING[@]}" \
    "$@"

echo "Fully async + Qwen3.5 (no LoRA) test completed"
