#!/usr/bin/env bash
set -xeuo pipefail

# Baseline test: sync GRPO colocated training (no NCCL suspend)
# Validates that verl + vLLM 0.18 colocated mode works before adding NCCL suspend.
#
# Based on: examples/grpo_trainer/run_qwen2-7b.sh (simplified for 2 GPU + small model)
#
# GPU: 2 minimum (TP=2 colocated)
# Model: Qwen2.5-0.5B-Instruct
# Backend: naive (default, no NCCL checkpoint engine)

NUM_GPUS=${NUM_GPUS:-2}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-0.5B-Instruct}

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.train_batch_size=4 \
    data.val_batch_size=4 \
    data.max_prompt_length=256 \
    data.max_response_length=256 \
    data.filter_overlong_prompts=True \
    data.truncation='left' \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.loss_agg_mode="token-mean" \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.50 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${NUM_GPUS} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_model_len=513 \
    actor_rollout_ref.rollout.max_num_batched_tokens=513 \
    actor_rollout_ref.rollout.max_num_seqs=513 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    reward.custom_reward_function.path=verl/utils/reward_score/gsm8k.py \
    reward.custom_reward_function.name=compute_score \
    trainer.logger='["console"]' \
    trainer.project_name='verl-test-baseline' \
    trainer.experiment_name='colocate-baseline' \
    trainer.val_before_train=False \
    trainer.test_freq=-1 \
    trainer.save_freq=-1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=3 \
    trainer.resume_mode=disable \
    trainer.n_gpus_per_node=${NUM_GPUS} \
    trainer.nnodes=1 \
    "$@"

echo "Colocate baseline test PASSED"
