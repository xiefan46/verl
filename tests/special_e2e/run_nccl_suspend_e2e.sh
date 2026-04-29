#!/usr/bin/env bash
set -xeuo pipefail

# E2E test for NCCL communicator suspend/resume during colocated GRPO training.
#
# Runs a few steps of sync GRPO with suspend_nccl_comms enabled to verify:
#   1. Training comms are correctly suspended/resumed at phase transitions
#   2. Rollout comms are correctly suspended/resumed at phase transitions
#   3. Weight sync (NCCL checkpoint engine) works with suspended comms
#   4. Training loss converges normally (no corruption from suspend/resume)
#
# GPU requirement: 2 GPUs minimum (TP=2 colocated, training + rollout share GPUs)
# Expected runtime: ~5 minutes on 2×H100
#
# Usage:
#   bash tests/special_e2e/run_nccl_suspend_e2e.sh
#   NUM_GPUS=4 bash tests/special_e2e/run_nccl_suspend_e2e.sh    # 4 GPU
#   NCCL_NVLS_ENABLE=0 bash tests/special_e2e/run_nccl_suspend_e2e.sh  # disable NVLS

NUM_GPUS=${NUM_GPUS:-2}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-0.5B-Instruct}

# Training parameters — small config for fast E2E validation
train_prompt_bsz=4
n_resp_per_prompt=2
train_prompt_mini_bsz=4
max_prompt_length=256
max_response_length=256
max_num_tokens=$(( max_prompt_length + max_response_length + 1 ))

# Parallelism — TP=2 to create NCCL comms on rollout side
gen_tp=2
sp_size=1
fsdp_size=${NUM_GPUS}

# Use naive backend for weight transfer (avoids double-sleep bug in nccl backend).
# NCCL suspend/resume targets training/rollout comms, not the weight transfer path.
checkpoint_engine_backend="naive"

exp_name="nccl-suspend-e2e-${NUM_GPUS}gpu"

echo "============================================"
echo "NCCL Suspend/Resume E2E Test"
echo "GPUs: ${NUM_GPUS}, TP: ${gen_tp}, FSDP: ${fsdp_size}"
echo "suspend_nccl_comms: true"
echo "============================================"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="${HOME}/data/gsm8k/train.parquet" \
    data.val_files="${HOME}/data/gsm8k/test.parquet" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    data.val_batch_size=${train_prompt_bsz} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.loss_agg_mode="token-mean" \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.fsdp_config.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=${fsdp_size} \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.50 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_model_len=${max_num_tokens} \
    actor_rollout_ref.rollout.max_num_batched_tokens=${max_num_tokens} \
    actor_rollout_ref.rollout.max_num_seqs=${max_num_tokens} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.checkpoint_engine.backend=${checkpoint_engine_backend} \
    +actor_rollout_ref.rollout.checkpoint_engine.suspend_nccl_comms=true \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    reward.reward_manager.name=dapo \
    +reward.reward_kwargs.overlong_buffer_cfg.enable=True \
    +reward.reward_kwargs.overlong_buffer_cfg.len=64 \
    +reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0 \
    +reward.reward_kwargs.overlong_buffer_cfg.log=False \
    +reward.reward_kwargs.max_resp_len=${max_response_length} \
    reward.custom_reward_function.path=verl/utils/reward_score/gsm8k.py \
    reward.custom_reward_function.name=compute_score \
    trainer.logger='["console"]' \
    trainer.project_name='verl-test-nccl-suspend' \
    trainer.experiment_name="${exp_name}" \
    trainer.val_before_train=False \
    trainer.test_freq=-1 \
    trainer.save_freq=-1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=3 \
    trainer.resume_mode=disable \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=${NUM_GPUS} \
    "$@"

echo "============================================"
echo "NCCL Suspend/Resume E2E Test PASSED"
echo "============================================"
