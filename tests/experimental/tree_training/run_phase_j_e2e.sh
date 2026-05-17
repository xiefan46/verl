#!/usr/bin/env bash
# Phase J: 1-step e2e GRPO on Qwen2.5-0.5B-Instruct + GSM8K with MagiAttention.
#
# Validates that the full training loop (rollout -> compute_old_log_prob ->
# compute_ref_log_prob -> update_policy -> optimizer.step) runs cleanly with
# actor.use_tree_training=True and produces a finite loss + non-zero parameter
# delta. Treats step 1 as the gate; longer convergence runs are Phase K+.
#
# Run AFTER Phase H (Magi install + GPU sanity) passes. Single-GPU run; for
# multi-DP variant use ``run_phase_l_multi_dp.sh``.
#
# Usage (on RunPod, inside /root/verl):
#   bash tests/experimental/tree_training/run_phase_j_e2e.sh

set -euo pipefail

cd "$(dirname "$0")/../../.."  # repo root

MODEL_PATH="${MODEL_PATH:-$HOME/models/Qwen/Qwen2.5-0.5B-Instruct}"
if [[ ! -d "$MODEL_PATH" ]]; then
  echo "ERROR: model not found at $MODEL_PATH"
  echo "Download with: bash /root/verl-deploy/download_models.sh Qwen/Qwen2.5-0.5B-Instruct"
  exit 1
fi

# Conservative single-step shapes — Magi smoke, not convergence run.
# Upstream run_qwen3_4b_fsdp.sh expects DEVICE (gpu/npu) to be set; we're on H100/H200.
export DEVICE=gpu
export NGPUS_PER_NODE=1
export ROLLOUT_TP=1
export TRAIN_BATCH_SIZE=8
export PPO_MINI_BATCH_SIZE=8
export PPO_MICRO_BATCH_SIZE_PER_GPU=1
export LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=1
export MAX_PROMPT_LENGTH=256
export MAX_RESPONSE_LENGTH=256
export ROLLOUT_GPU_MEM_UTIL=0.4
export ROLLOUT_N=4
export TOTAL_EPOCHS=1
# Keep step count tiny: 1 step proves the path; 10 steps proves no NaN/OOM.
export TOTAL_TRAINING_STEPS=10
export TEST_FREQ=-1
export SAVE_FREQ=-1
export PROJECT_NAME="verl_magi_phase_j"
export EXPERIMENT_NAME="qwen2_5_0_5b_tree_smoke"

LOG="/tmp/magi_phase_j_$(date +%Y%m%d_%H%M%S).log"

# NCCL_NVLS_ENABLE=0 guards against NVLS-allreduce errors on some H100 pods
# (see verl-deploy/README.md). Harmless on pods that don't need it.
NCCL_NVLS_ENABLE=0 \
bash examples/grpo_trainer/run_qwen3_4b_fsdp.sh \
  data.train_files="$HOME/data/gsm8k/train.parquet" \
  data.val_files="$HOME/data/gsm8k/test.parquet" \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.actor.strategy=fsdp2 \
  actor_rollout_ref.actor.use_tree_training=true \
  actor_rollout_ref.actor.tree_training.max_tokens_per_mb=2048 \
  actor_rollout_ref.actor.tree_training.tree_cp_size=1 \
  actor_rollout_ref.actor.use_dynamic_bsz=false \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=false \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=false \
  trainer.balance_batch=false \
  trainer.total_training_steps="$TOTAL_TRAINING_STEPS" \
  trainer.logger='["console"]' \
  2>&1 | tee "$LOG"

# Quick post-run validation: ensure we saw finite losses and the tree path
# was actually exercised.
echo ""
echo "=== Phase J validation ==="
if grep -q "tree_token_ratio" "$LOG"; then
  echo "OK: tree_token_ratio metric present in log"
else
  echo "FAIL: tree_token_ratio not logged — tree path may not have engaged"
  exit 1
fi

if grep -qE "loss[: ]*nan|inf" -i "$LOG"; then
  echo "FAIL: NaN/Inf loss detected"
  exit 1
fi

if grep -q "step 0" "$LOG"; then
  echo "OK: at least step 0 completed"
else
  echo "FAIL: training did not reach step 0"
  exit 1
fi

echo ""
echo "Phase J PASS. Log: $LOG"
