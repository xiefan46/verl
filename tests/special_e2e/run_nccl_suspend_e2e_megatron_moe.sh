#!/usr/bin/env bash
set -xeuo pipefail

# E2E test for NCCL communicator suspend/resume during colocated GRPO training
# with Megatron-LM as the training engine, using a MoE model (Qwen3-30B-A3B-Base).
#
# Compared to the Dense variant (run_nccl_suspend_e2e_megatron.sh), this exercises:
#   - Expert-parallel all_to_all collective (the largest single NCCL channel buffer
#     in our prior 8×H100 profile — ~2.9 GB per group vs ~480 MB for normal collectives)
#   - Expert-tensor-parallel comms (when ETP > 1)
#   - The full MoE weight loading path through mbridge
#
# GPU requirement: 8 GPUs (H100 80GB recommended).
# Memory: 30B model needs all_offload=True; vLLM uses 0.4 GMU + enforce_eager
#         to leave room for training activations.
# Expected runtime: ~20-30 min on 8×H100 (model init + dist convert + 3 steps).
#
# Default config: TP=2 PP=1 DP=4 EP=4 ETP=1 (mirrors upstream NPU script halved).
#
# Usage:
#   bash tests/special_e2e/run_nccl_suspend_e2e_megatron_moe.sh
#
#   # Pure-EP variant (TP=1 PP=1 DP=8 EP=8) — cleanest all_to_all signal:
#   MEGATRON_TP=1 MEGATRON_EP=8 GEN_TP=1 \
#     bash tests/special_e2e/run_nccl_suspend_e2e_megatron_moe.sh

NUM_GPUS=${NUM_GPUS:-8}
MODEL_PATH=${MODEL_PATH:-${HOME}/models/Qwen/Qwen3-30B-A3B-Base}

# Training parameters — small config for fast E2E validation
train_prompt_bsz=8
n_resp_per_prompt=2
train_prompt_mini_bsz=8
max_prompt_length=${MAX_PROMPT_LENGTH:-512}
max_response_length=${MAX_RESPONSE_LENGTH:-512}
max_num_tokens=$(( max_prompt_length + max_response_length + 1 ))

# Megatron parallelism: TP × PP × CP determines the base; DP fills remainder.
# EP is orthogonal — partitions experts within DP×TP space (must satisfy EP × ETP ≤ DP × TP).
megatron_tp=${MEGATRON_TP:-2}
megatron_pp=${MEGATRON_PP:-1}
megatron_cp=${MEGATRON_CP:-1}
megatron_ep=${MEGATRON_EP:-4}
megatron_etp=${MEGATRON_ETP:-1}
megatron_dp=$(( NUM_GPUS / (megatron_tp * megatron_pp * megatron_cp) ))

# vLLM rollout parallelism
gen_tp=${GEN_TP:-2}
gen_pp=${GEN_PP:-1}

# Use naive backend for weight transfer (in-process generator yield, no NCCL).
checkpoint_engine_backend="naive"

exp_name="nccl-suspend-megatron-moe-${NUM_GPUS}gpu-tp${megatron_tp}pp${megatron_pp}dp${megatron_dp}ep${megatron_ep}"

echo "============================================"
echo "NCCL Suspend/Resume E2E Test (Megatron MoE)"
echo "Model:          ${MODEL_PATH}"
echo "GPUs:           ${NUM_GPUS}"
echo "Megatron:       TP=${megatron_tp} PP=${megatron_pp} DP=${megatron_dp} CP=${megatron_cp} EP=${megatron_ep} ETP=${megatron_etp}"
echo "Rollout (vLLM): TP=${gen_tp} PP=${gen_pp}"
echo "suspend_nccl_comms: true"
echo "============================================"

python3 -m verl.trainer.main_ppo \
    --config-name=ppo_megatron_trainer \
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
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.loss_agg_mode="token-mean" \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${megatron_tp} \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${megatron_pp} \
    actor_rollout_ref.actor.megatron.context_parallel_size=${megatron_cp} \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=${megatron_ep} \
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=${megatron_etp} \
    actor_rollout_ref.actor.megatron.virtual_pipeline_model_parallel_size=null \
    actor_rollout_ref.actor.megatron.param_offload=True \
    actor_rollout_ref.actor.megatron.grad_offload=True \
    actor_rollout_ref.actor.megatron.optimizer_offload=True \
    actor_rollout_ref.actor.megatron.use_mbridge=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${megatron_tp} \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${megatron_pp} \
    actor_rollout_ref.ref.megatron.context_parallel_size=${megatron_cp} \
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=${megatron_ep} \
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=${megatron_etp} \
    actor_rollout_ref.ref.megatron.virtual_pipeline_model_parallel_size=null \
    actor_rollout_ref.ref.megatron.param_offload=True \
    actor_rollout_ref.ref.megatron.use_mbridge=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.40 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.pipeline_model_parallel_size=${gen_pp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.max_model_len=${max_num_tokens} \
    actor_rollout_ref.rollout.max_num_batched_tokens=${max_num_tokens} \
    actor_rollout_ref.rollout.max_num_seqs=${max_num_tokens} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
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
    trainer.project_name='verl-test-nccl-suspend-megatron-moe' \
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
echo "NCCL Suspend/Resume E2E Test (Megatron MoE) PASSED"
echo "============================================"
