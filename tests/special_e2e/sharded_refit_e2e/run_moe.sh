#!/usr/bin/env bash
# Sharded-aware NCCL refit MoE e2e gate — Qwen3-30B-A3B-Instruct-2507 on 8×H200.
#
# Trainer Megatron: 4 GPU, PP=1 CP=1 TP=2 EP=2 ETP=1, offload ON.
# Rollout vLLM:     4 GPU, TP=4 EP=4 DP=1, standalone, gpu_mem_util=0.85.
#
# The asymmetric trainer-vs-rollout TP/EP is intentional — it forces the
# routing plan to do cross-TP (2→4) and cross-EP (2→4) shard
# redistribution, which is the actual value-add over broadcast.
#
# Validates:
#   - multi-rank ParameterShardMeta enumeration on BOTH sides
#   - cross-TP redistribution of attention rows + dense col splits
#   - cross-EP redistribution of routed experts
#   - per-expert routing: trainer's local experts land on the right vLLM
#     EP rank via the M3 enricher's w1/w2/w3 + expert_id mapping
#   - DISTINGUISHING=1 confirms shards actually overwrite vLLM
set -xeuo pipefail

MODEL_ID=${MODEL_ID:-Qwen/Qwen3-30B-A3B-Instruct-2507}
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

# Helper: drain Ray state between runs. Qwen3-30B-A3B spins up ~16 Ray
# actors per run (4 trainer + 4 CheckpointEngineWorker + 4 vLLM + 4 misc)
# plus detached NCCLUniqueIDStore actors from ray.util.collective; the
# next ray.init() routinely times out reading stale GCS state from
# /tmp/ray. Force-stop + scrub keeps the second driver from hitting
# "RPC error: Deadline Exceeded" on bootstrap.
drain_ray() {
    ray stop --force 2>/dev/null || true
    sleep 3
    rm -rf /tmp/ray /tmp/ray-* 2>/dev/null || true
    sleep 2
}

# --- Pass 1: legacy broadcast (baseline) ---
python tests/special_e2e/sharded_refit_e2e/driver_moe.py \
    --backend nccl \
    --model-path "${MODEL_PATH}" \
    --out-path "${OUT_DIR}/dump_nccl.json" \
    --prompt "${PROMPT}" \
    "${DRIVER_FLAGS[@]}"

drain_ray

# --- Pass 2: sharded routing (the gate) ---
python tests/special_e2e/sharded_refit_e2e/driver_moe.py \
    --backend sharded_nccl \
    --model-path "${MODEL_PATH}" \
    --out-path "${OUT_DIR}/dump_sharded_nccl.json" \
    --prompt "${PROMPT}" \
    "${DRIVER_FLAGS[@]}"

drain_ray

# --- Compare ---
python tests/special_e2e/sharded_refit_e2e/compare.py \
    "${OUT_DIR}/dump_nccl.json" \
    "${OUT_DIR}/dump_sharded_nccl.json" \
    --atol "${ATOL}"

echo "[run_moe] ===== sharded_refit MoE e2e: ALL GATES PASSED ====="
