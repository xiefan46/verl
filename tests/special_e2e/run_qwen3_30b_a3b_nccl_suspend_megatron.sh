#!/usr/bin/env bash
# GRPO | Qwen3-30B-A3B (MoE) | Megatron | NCCL suspend/resume memory release test
#
# Test recipe for the NCCL suspend/resume feature on 8xH200.
# Parallelism: TP=2 PP=2 DP=2 EP=2 ETP=2 (matches the 3D+MoE config that
# freed ~9.5 GB/rank on 8xH100). With ALL_OFFLOAD=True, trainer state is
# CPU-offloaded during rollout so the released NCCL comm memory goes to
# vLLM KV cache.
#
# Adapted from run_qwen3_30b_a3b_megatron.sh:
#   - Parallelism switched from TP=4 PP=2 DP=1 EP=2 to TP=2 PP=2 DP=2 EP=2 ETP=2
#     so the trainer NCCL comms include PP/DP/EP/ETP groups (more memory to free).
#   - rollout_gpu_mem_util bumped 0.60 -> 0.80 since H200 has more headroom.
#   - total_training_steps capped to 100 for a short measurement run.
#   - save_freq / test_freq disabled to keep the run purely about NCCL telemetry.
#   - suspend_nccl_comms defaults to True (the whole point of this script);
#     set SUSPEND_NCCL_COMMS=False to capture a baseline.
#
# Prerequisites on RunPod 8xH200:
#   - libnccl >= 2.29.7 in the env (RunPod default is 2.x.y < 2.29.7;
#     run `bash /root/verl-deploy/upgrade_nccl.sh` first).
#   - wandb logged in so the new nccl_suspend/* metrics get captured.
#   - NCCL_NVLS_ENABLE=0 (auto-set below; NCCL 2.29+ on H200 NVSwitch
#     can crash on cuMulticastBindMem without a Fabric Manager).
#
# Knobs (env vars):
#   SUSPEND_NCCL_COMMS   True|False                       (default: True)
#   TOTAL_TRAIN_STEPS    cap steps for the test run       (default: 100)
#   INFER_BACKEND        rollout backend                  (default: vllm)
#   MODEL_PATH           HF model path or local dir       (default: Qwen/Qwen3-30B-A3B-Base)

set -xeuo pipefail
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0}

########################### user-adjustable ###########################
INFER_BACKEND=${INFER_BACKEND:-vllm}
SUSPEND_NCCL_COMMS=${SUSPEND_NCCL_COMMS:-True}

DATA_DIR=${DATA_DIR:-"$PWD"}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-30B-A3B-Base}
MCORE_MODEL_PATH=${MCORE_MODEL_PATH:-}
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-${GPUS_PER_NODE:-8}}

train_files=${TRAIN_FILES:-${DAPO_MATH_TRAIN:-${DATA_DIR}/data/DAPO-Math-17k/data/dapo-math-17k.parquet}}
val_files=${VAL_FILES:-${AIME_VAL:-${DATA_DIR}/data/AIME-2024/data/aime-2024.parquet}}

train_batch_size=${TRAIN_BATCH_SIZE:-128}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-16}
max_prompt_length=${MAX_PROMPT_LENGTH:-2048}
max_response_length=${MAX_RESPONSE_LENGTH:-4096}
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-30720}
actor_ppo_micro_batch_size_per_gpu=${ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU:-2}

# TP=2 PP=2 DP=2 EP=2 ETP=2; world = 8.  EP*ETP = 4 = TP*DP (Megatron constraint).
actor_tp=${ACTOR_TP:-2}
actor_pp=${ACTOR_PP:-2}
actor_vpp=${ACTOR_VPP:-2}
actor_ep=${ACTOR_EP:-2}
actor_etp=${ACTOR_ETP:-2}
actor_cp=${ACTOR_CP:-1}
ref_tp=${REF_TP:-${actor_tp}}
ref_pp=${REF_PP:-${actor_pp}}
ref_vpp=${REF_VPP:-2}
ref_ep=${REF_EP:-${actor_ep}}
ref_etp=${REF_ETP:-${actor_etp}}
ref_cp=${REF_CP:-1}
all_offload=${ALL_OFFLOAD:-True}

rollout_tp=${ROLLOUT_TP:-2}
infer_tp=${INFER_TP:-${rollout_tp}}
gen_moe_tp=${GEN_MOE_TP:-2}
gen_moe_ep=${GEN_MOE_EP:-2}
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.80}
rollout_n=${ROLLOUT_N:-4}
rollout_max_num_batched_tokens=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-8192}
rollout_max_model_len=${ROLLOUT_MAX_MODEL_LEN:-8192}
rollout_temperature=${ROLLOUT_TEMPERATURE:-1.0}
rollout_top_p=${ROLLOUT_TOP_P:-1}

ref_log_prob_max_token_len_per_gpu=${REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-40960}
ref_log_prob_micro_batch_size_per_gpu=${REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-4}
rollout_log_prob_max_token_len_per_gpu=${ROLLOUT_LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-40960}
rollout_log_prob_micro_batch_size_per_gpu=${ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-4}

total_train_steps=${TOTAL_TRAIN_STEPS:-100}
save_freq=${SAVE_FREQ:--1}
test_freq=${TEST_FREQ:--1}

project_name=${PROJECT_NAME:-verl_nccl_suspend_h200}
suspend_tag=$(echo "${SUSPEND_NCCL_COMMS}" | tr '[:upper:]' '[:lower:]')
experiment_name=${EXPERIMENT_NAME:-qwen3_30b_a3b_suspend_${suspend_tag}}
########################### end user-adjustable ###########################

########################### derived defaults ###########################
if [ "${infer_tp}" = 4 ] || [ "${infer_tp}" = 2 ]; then
    rollout_max_num_seqs=${ROLLOUT_MAX_NUM_SEQS:-1024}
else
    rollout_max_num_seqs=${ROLLOUT_MAX_NUM_SEQS:-384}
fi

[ "${actor_pp}" -gt 1 ] && actor_vpp_override=${actor_vpp} || actor_vpp_override=null
[ "${ref_pp}" -gt 1 ] && ref_vpp_override=${ref_vpp} || ref_vpp_override=null

########################### parameter arrays ###########################

ALGORITHM=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    algorithm.kl_ctrl.kl_coef=0.0
)

REWARD=(
    reward_model.reward_manager=dapo
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=True
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=4096
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0
    +reward_model.reward_kwargs.overlong_buffer_cfg.log=False
    +reward_model.reward_kwargs.max_resp_len=${max_response_length}
)

DATA=(
    data.train_files="$train_files"
    data.val_files="$val_files"
    data.train_batch_size=${train_batch_size}
    data.prompt_key=prompt
    data.return_raw_chat=True
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.filter_overlong_prompts=False
    data.truncation=left
)

MODEL=(
    actor_rollout_ref.model.path="$MODEL_PATH"
    actor_rollout_ref.model.use_fused_kernels=True
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=1e-5
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${actor_ppo_micro_batch_size_per_gpu}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.loss_agg_mode=token-mean
    actor_rollout_ref.actor.suspend_nccl_comms=${SUSPEND_NCCL_COMMS}
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${actor_tp}
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${actor_pp}
    actor_rollout_ref.actor.megatron.virtual_pipeline_model_parallel_size=${actor_vpp_override}
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=${actor_ep}
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=${actor_etp}
    actor_rollout_ref.actor.megatron.context_parallel_size=${actor_cp}
    actor_rollout_ref.actor.megatron.param_offload=${all_offload}
    actor_rollout_ref.actor.megatron.optimizer_offload=${all_offload}
    actor_rollout_ref.actor.megatron.grad_offload=${all_offload}
    actor_rollout_ref.actor.megatron.use_mbridge=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1
    +actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.masked_softmax_fusion=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.bias_activation_fusion=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.bias_dropout_fusion=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.deallocate_pipeline_outputs=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.persist_layer_norm=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_grouped_gemm=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=True
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=${INFER_BACKEND}
    actor_rollout_ref.rollout.tensor_model_parallel_size=${infer_tp}
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util}
    actor_rollout_ref.rollout.n=${rollout_n}
    actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${rollout_log_prob_max_token_len_per_gpu}
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${rollout_log_prob_micro_batch_size_per_gpu}
    actor_rollout_ref.rollout.max_num_seqs=${rollout_max_num_seqs}
    actor_rollout_ref.rollout.max_num_batched_tokens=${rollout_max_num_batched_tokens}
    actor_rollout_ref.rollout.max_model_len=${rollout_max_model_len}
    actor_rollout_ref.rollout.prompt_length=${max_prompt_length}
    actor_rollout_ref.rollout.response_length=${max_response_length}
    actor_rollout_ref.rollout.temperature=${rollout_temperature}
    actor_rollout_ref.rollout.top_p=${rollout_top_p}
)

REF=(
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${ref_log_prob_max_token_len_per_gpu}
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${ref_log_prob_micro_batch_size_per_gpu}
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${ref_tp}
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${ref_pp}
    actor_rollout_ref.ref.megatron.virtual_pipeline_model_parallel_size=${ref_vpp_override}
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=${ref_ep}
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=${ref_etp}
    actor_rollout_ref.ref.megatron.context_parallel_size=${ref_cp}
    actor_rollout_ref.ref.megatron.param_offload=${all_offload}
    actor_rollout_ref.ref.megatron.use_mbridge=True
)

TRAINER=(
    trainer.balance_batch=True
    trainer.logger='["console","wandb"]'
    trainer.project_name=${project_name}
    trainer.experiment_name=${experiment_name}
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.save_freq=${save_freq}
    trainer.test_freq=${test_freq}
    trainer.total_training_steps=${total_train_steps}
    trainer.resume_mode=disable
    trainer.val_before_train=False
)

EXTRA=(
    model_engine=megatron
)

# rollout.moe_tensor_parallel_size + expert_parallel_size are only valid for the
# trtllm rollout backend; vllm asserts against them in RolloutConfig.__post_init__.
if [ "${INFER_BACKEND}" = "trtllm" ]; then
    EXTRA+=(
        +actor_rollout_ref.rollout.moe_tensor_parallel_size=${gen_moe_tp}
        actor_rollout_ref.rollout.expert_parallel_size=${gen_moe_ep}
    )
fi

if [ -n "$MCORE_MODEL_PATH" ]; then
    EXTRA+=(
        actor_rollout_ref.actor.megatron.dist_checkpointing_path=${MCORE_MODEL_PATH}
        actor_rollout_ref.actor.megatron.use_dist_checkpointing=True
        actor_rollout_ref.ref.megatron.dist_checkpointing_path=${MCORE_MODEL_PATH}
        actor_rollout_ref.ref.megatron.use_dist_checkpointing=True
    )
fi

########################### launch ###########################
python3 -m verl.trainer.main_ppo \
    "${ALGORITHM[@]}" \
    "${REWARD[@]}" \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${TRAINER[@]}" \
    "${EXTRA[@]}" \
    "$@"
