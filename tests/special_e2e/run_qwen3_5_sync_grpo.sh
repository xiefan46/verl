#!/usr/bin/env bash
set -xeuo pipefail

# Sync (colocate) GRPO test for Qwen3.5 + use_remove_padding=True.
#
# Purpose: validate that the flash_attn_varlen CUDA illegal memory access
# from 3D mRoPE position_ids is NOT a fully_async-specific bug — the same
# bug surfaces in standard sync/colocate training of Qwen3.5 with
# use_remove_padding=True (which is the default for verl).
#
# Without the fix in verl/models/transformers/monkey_patch.py
# (collapse 3D mRoPE position_ids to 2D), this script crashes with
# `CUDA error: an illegal memory access` in flash_attn_varlen during
# the first trainer forward.
#
# With the fix applied, this script runs to completion.
#
# GPU allocation (2 GPUs, colocate):
#   - Both GPUs: vLLM (TP=1, DP=2 via FSDP rank) + FSDP world_size=2 trainer
#   - vLLM uses sleep mode to release memory while trainer trains
#
# Usage:
#   cd /root/verl && bash tests/special_e2e/run_qwen3_5_sync_grpo.sh

# Workaround for NVIDIA driver bug (r560-r575)
export NCCL_CUMEM_ENABLE=0
export NCCL_CUMEM_HOST_ENABLE=0

############################ Quick Config ############################

ROLLOUT_NAME="vllm"
export VLLM_USE_V1=1

MODEL_ID=${MODEL_ID:-Qwen/Qwen3.5-2B}
MODEL_PATH=${MODEL_PATH:-${HOME}/models/${MODEL_ID}}

MAX_PROMPT=2048
MAX_RESPONSE=1024

N_GPUS=2

############################ Data Paths ############################

GSM8K_TRAIN="${HOME}/data/gsm8k/train.parquet"
GSM8K_TEST="${HOME}/data/gsm8k/test.parquet"

############################ Parameter Groups ############################

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    algorithm.kl_ctrl.kl_coef=0.0
    data.train_files="$GSM8K_TRAIN"
    data.val_files="$GSM8K_TEST"
    data.prompt_key=prompt
    data.train_batch_size=4
    data.max_prompt_length=$MAX_PROMPT
    data.max_response_length=$MAX_RESPONSE
    data.truncation='left'
    data.return_raw_chat=True
)

MODEL=(
    actor_rollout_ref.model.path="${MODEL_PATH}"
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.model.use_remove_padding=True   # ← bug trigger (varlen path)
)

ACTOR=(
    actor_rollout_ref.actor.strategy=fsdp
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.optim.lr_warmup_steps=-1
    actor_rollout_ref.actor.optim.weight_decay=0.1
    actor_rollout_ref.actor.ppo_mini_batch_size=4
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

REF=(
    actor_rollout_ref.ref.fsdp_config.param_offload=True
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=$ROLLOUT_NAME
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.n=4
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5
    actor_rollout_ref.rollout.temperature=1.0
    actor_rollout_ref.rollout.top_p=1.0
    actor_rollout_ref.rollout.top_k=-1
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.max_num_batched_tokens=$(( MAX_PROMPT + MAX_RESPONSE + 1 ))
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.rollout.enforce_eager=False
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
    actor_rollout_ref.rollout.val_kwargs.n=1
)

REWARD=(
    reward.reward_manager.name=dapo
    +reward.reward_kwargs.overlong_buffer_cfg.enable=True
    +reward.reward_kwargs.overlong_buffer_cfg.len=128
    +reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0
    +reward.reward_kwargs.overlong_buffer_cfg.log=False
    +reward.reward_kwargs.max_resp_len=${MAX_RESPONSE}
)

CRITIC=(
    critic.enable=False
)

TRAINER=(
    trainer.logger='["console"]'
    trainer.project_name='verl-test-qwen3-5-sync-grpo'
    trainer.experiment_name="qwen3.5-2b-sync-grpo-2gpu"
    trainer.val_before_train=False
    trainer.save_freq=-1
    trainer.resume_mode=disable
    trainer.nnodes=1
    trainer.n_gpus_per_node=${N_GPUS}
    trainer.test_freq=-1
    trainer.total_epochs=1
    trainer.total_training_steps=4
)

############################ Launch ############################

echo "Running sync GRPO + Qwen3.5 (colocate) — varlen mRoPE bug repro"
echo "Model: ${MODEL_PATH}"
echo "GPUs: ${N_GPUS} (vLLM + FSDP colocate, sleep mode)"

python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${REF[@]}" \
    "${ROLLOUT[@]}" \
    "${REWARD[@]}" \
    "${CRITIC[@]}" \
    "${TRAINER[@]}" \
    "$@"

echo "Sync GRPO + Qwen3.5 test completed"
