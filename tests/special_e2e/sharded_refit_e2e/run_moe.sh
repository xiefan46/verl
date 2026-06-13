#!/usr/bin/env bash
# Sharded-aware NCCL refit MoE e2e gate — Qwen3-30B-A3B-Instruct on 4×H100.
#
# Trainer Megatron: 2 GPU, PP=1 CP=1 TP=1 EP=2 ETP=1, param+grad+optim offload ON.
# Rollout vLLM:     2 GPU, TP=2 EP=2 DP=1, standalone, gpu_mem_util=0.85.
#
# Validates:
#   - multi-rank ParameterShardMeta enumeration (rollout EP-split experts,
#     TP-split attention)
#   - per-expert routing: trainer's local experts land on the right vLLM
#     EP rank via the M3 enricher's w1/w2/w3 + expert_id mapping
#   - DISTINGUISHING=1 confirms shards actually overwrite vLLM
set -xeuo pipefail

MODEL_ID=${MODEL_ID:-Qwen/Qwen3-30B-A3B-Instruct}
MODEL_PATH=${MODEL_PATH:-${HOME}/models/${MODEL_ID}}
OUT_DIR=${OUT_DIR:-/tmp/sharded_refit_moe_e2e}
PROMPT=${PROMPT:-"The capital of France is"}
ATOL=${ATOL:-5e-3}
DISTINGUISHING=${DISTINGUISHING:-0}

DRIVER_FLAGS=()
if [ "${DISTINGUISHING}" = "1" ]; then
    DRIVER_FLAGS+=(--zero-init-trainer)
    echo "[run_moe] DISTINGUISHING mode: zero-init trainer pre-update"
fi

mkdir -p "${OUT_DIR}"

if [ ! -f "${MODEL_PATH}/config.json" ]; then
    echo "[run_moe] model not at ${MODEL_PATH}, downloading via hf cli (~60 GB) ..."
    hf download "${MODEL_ID}" --local-dir "${MODEL_PATH}"
fi

REPO_ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
cd "${REPO_ROOT}"

# --- Pass 1: legacy broadcast (baseline) ---
python tests/special_e2e/sharded_refit_e2e/driver_moe.py \
    --backend nccl \
    --model-path "${MODEL_PATH}" \
    --out-path "${OUT_DIR}/dump_nccl.json" \
    --prompt "${PROMPT}" \
    "${DRIVER_FLAGS[@]}"

# --- Pass 2: sharded routing (the gate) ---
python tests/special_e2e/sharded_refit_e2e/driver_moe.py \
    --backend sharded_nccl \
    --model-path "${MODEL_PATH}" \
    --out-path "${OUT_DIR}/dump_sharded_nccl.json" \
    --prompt "${PROMPT}" \
    "${DRIVER_FLAGS[@]}"

# --- Compare ---
python tests/special_e2e/sharded_refit_e2e/compare.py \
    "${OUT_DIR}/dump_nccl.json" \
    "${OUT_DIR}/dump_sharded_nccl.json" \
    --atol "${ATOL}"

echo "[run_moe] ===== sharded_refit MoE e2e: ALL GATES PASSED ====="
