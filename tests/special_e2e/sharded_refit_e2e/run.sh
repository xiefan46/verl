#!/usr/bin/env bash
# Drive sharded-aware NCCL refit gate end-to-end.
#
# Runs the verl Megatron+vLLM stack once per backend (legacy ``nccl``
# broadcast, then the new ``sharded_nccl`` P2P routing), dumps each run's
# greedy completion + per-position logprobs, and asserts equivalence.
#
# Single 1xH100 (Qwen2.5-0.5B). Total wall time ~3-5 min on a healthy pod.
set -xeuo pipefail

MODEL_ID=${MODEL_ID:-Qwen/Qwen2.5-0.5B-Instruct}
MODEL_PATH=${MODEL_PATH:-${HOME}/models/${MODEL_ID}}
OUT_DIR=${OUT_DIR:-/tmp/sharded_refit_e2e}
PROMPT=${PROMPT:-"The capital of France is"}
ATOL=${ATOL:-5e-3}
# DISTINGUISHING=1 zeros trainer Megatron weights pre-update; both backends
# must then produce IDENTICAL garbage. If non-distinguishing PASSES and this
# FAILS, your sharded backend is a silent no-op.
DISTINGUISHING=${DISTINGUISHING:-0}

DRIVER_FLAGS=()
if [ "${DISTINGUISHING}" = "1" ]; then
    DRIVER_FLAGS+=(--zero-init-trainer)
    echo "[run] DISTINGUISHING mode: zero-init trainer pre-update"
fi

mkdir -p "${OUT_DIR}"

# Ensure model is local; fall back to ``hf download`` if missing.
if [ ! -f "${MODEL_PATH}/config.json" ]; then
    echo "[run] model not at ${MODEL_PATH}, downloading via hf cli ..."
    hf download "${MODEL_ID}" --local-dir "${MODEL_PATH}"
fi

REPO_ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
cd "${REPO_ROOT}"

# --- Pass 1: legacy broadcast (baseline) ---
python tests/special_e2e/sharded_refit_e2e/driver.py \
    --backend nccl \
    --model-path "${MODEL_PATH}" \
    --out-path "${OUT_DIR}/dump_nccl.json" \
    --prompt "${PROMPT}" \
    "${DRIVER_FLAGS[@]}"

# --- Pass 2: sharded routing (the gate) ---
python tests/special_e2e/sharded_refit_e2e/driver.py \
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

echo "[run] ===== sharded_refit e2e: ALL GATES PASSED ====="
