#!/usr/bin/env bash
# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
#
# E2E test for NCCL suspend/resume during colocated GRPO with Megatron.
# Defaults match tests/special_e2e/run_ppo_trainer_megatron.sh for A/B
# comparison. Requires 8 GPUs (TP=2 PP=2 DP=2).
#
# Usage:
#   bash tests/special_e2e/run_nccl_suspend_megatron.sh             # default: suspend ON, 3-step
#   SUSPEND_NCCL_COMMS=false TOTAL_STEPS=100 bash ...               # A/B baseline arm
#   MEGATRON_EP=2 MEGATRON_ETP=2 bash ...                           # MoE variant
#
# See RFC: verl-project/verl#6266.

set -xeuo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS=1
export VERL_LOGGING_LEVEL=${VERL_LOGGING_LEVEL:-INFO}

# Disable NVLink SHARP (NVLS) Multicast. NCCL >= 2.27 auto-enables NVLS on H100
# + NVSwitch hosts, but containerized 8xH100 environments where the Fabric
# Manager does not expose multicast (e.g. RunPod, several other cloud pods)
# fail `cuMulticastBindMem` with CUDA error 401 at the first collective, well
# before training starts. NVLS off falls back to regular NVLink — bandwidth
# loss is negligible for this e2e shape and irrelevant to the suspend/resume
# behavior under test. Bare-metal users with a working multicast fabric can
# override with `NCCL_NVLS_ENABLE=1 bash run_nccl_suspend_megatron.sh ...`.
export NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0}

NUM_GPUS=${NUM_GPUS:-8}
MODEL_ID=${MODEL_ID:-Qwen/Qwen2.5-0.5B-Instruct}
MODEL_PATH=${MODEL_PATH:-${HOME}/models/${MODEL_ID}}

TRAIN_FILES=${TRAIN_FILES:-${HOME}/data/gsm8k/train.parquet}
VAL_FILES=${VAL_FILES:-${HOME}/data/gsm8k/test.parquet}

# Megatron parallelism. TP × PP × DP × CP must equal NUM_GPUS.
MEGATRON_TP=${MEGATRON_TP:-2}
MEGATRON_PP=${MEGATRON_PP:-2}
MEGATRON_CP=${MEGATRON_CP:-1}
MEGATRON_EP=${MEGATRON_EP:-1}
MEGATRON_ETP=${MEGATRON_ETP:-1}
MEGATRON_DP=$(( NUM_GPUS / (MEGATRON_TP * MEGATRON_PP * MEGATRON_CP) ))

# Rollout (vLLM) parallelism.
GEN_TP=${GEN_TP:-2}

# Training shape — defaults match run_ppo_trainer_megatron.sh so an A/B
# comparison against that baseline is apples-to-apples.
TRAIN_BSZ=${TRAIN_BSZ:-16}
MINI_BSZ=${MINI_BSZ:-8}
MICRO_BSZ_PER_GPU=${MICRO_BSZ_PER_GPU:-2}
MAX_PROMPT_LEN=${MAX_PROMPT_LEN:-512}
MAX_RESPONSE_LEN=${MAX_RESPONSE_LEN:-512}
N_RESP=${N_RESP:-4}
TOTAL_STEPS=${TOTAL_STEPS:-3}

# Toggle the feature under test. Default on so a bare `bash run_nccl_suspend_megatron.sh`
# exercises the new path; flip to false for the baseline arm of an A/B comparison.
SUSPEND_NCCL_COMMS=${SUSPEND_NCCL_COMMS:-true}

EXP_NAME="nccl-suspend-megatron-${NUM_GPUS}gpu-tp${MEGATRON_TP}pp${MEGATRON_PP}dp${MEGATRON_DP}"

echo "================================================================"
echo "NCCL suspend/resume E2E (Megatron)"
echo "  GPUs:               ${NUM_GPUS}"
echo "  Megatron:           TP=${MEGATRON_TP} PP=${MEGATRON_PP} DP=${MEGATRON_DP}"
echo "                      CP=${MEGATRON_CP} EP=${MEGATRON_EP} ETP=${MEGATRON_ETP}"
echo "  Rollout (vLLM):     TP=${GEN_TP}"
echo "  Batch:              prompts=${TRAIN_BSZ} mini=${MINI_BSZ} micro/gpu=${MICRO_BSZ_PER_GPU} n_resp=${N_RESP}"
echo "  Sequence:           prompt<=${MAX_PROMPT_LEN} response<=${MAX_RESPONSE_LEN}"
echo "  Steps:              ${TOTAL_STEPS}"
echo "  suspend_nccl_comms: ${SUSPEND_NCCL_COMMS}"
echo "================================================================"

python3 -m verl.trainer.main_ppo \
    --config-name=ppo_megatron_trainer \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length=${MAX_PROMPT_LEN} \
    data.max_response_length=${MAX_RESPONSE_LEN} \
    data.train_batch_size=${TRAIN_BSZ} \
    data.val_batch_size=${TRAIN_BSZ} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${MINI_BSZ} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${MICRO_BSZ_PER_GPU} \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${MEGATRON_TP} \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${MEGATRON_PP} \
    actor_rollout_ref.actor.megatron.context_parallel_size=${MEGATRON_CP} \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=${MEGATRON_EP} \
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=${MEGATRON_ETP} \
    actor_rollout_ref.actor.megatron.use_mbridge=True \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${MEGATRON_TP} \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${MEGATRON_PP} \
    actor_rollout_ref.ref.megatron.context_parallel_size=${MEGATRON_CP} \
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=${MEGATRON_EP} \
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=${MEGATRON_ETP} \
    actor_rollout_ref.ref.megatron.param_offload=True \
    actor_rollout_ref.ref.megatron.use_mbridge=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${MICRO_BSZ_PER_GPU} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=${N_RESP} \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.50 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${GEN_TP} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${MICRO_BSZ_PER_GPU} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.checkpoint_engine.backend=naive \
    actor_rollout_ref.actor.suspend_nccl_comms=${SUSPEND_NCCL_COMMS} \
    trainer.logger='["console"]' \
    trainer.project_name='verl-test-nccl-suspend' \
    trainer.experiment_name="${EXP_NAME}" \
    trainer.val_before_train=False \
    trainer.test_freq=-1 \
    trainer.save_freq=-1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=${TOTAL_STEPS} \
    trainer.resume_mode=disable \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=${NUM_GPUS} \
    "$@"

echo "================================================================"
echo "NCCL suspend/resume E2E (Megatron) PASSED"
echo "================================================================"
