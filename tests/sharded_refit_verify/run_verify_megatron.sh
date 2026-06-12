#!/usr/bin/env bash
# Verify Megatron MoE layout for Qwen3.5-35B-A3B (sharded-aware refit M3 prep).
#
# Triggers _verify_moe_layout_hook.print_layout_and_exit() at the end of
# verl/workers/engine/megatron/transformer_impl.py:_build_megatron_module.
# The hook prints the MoE storage layout and calls sys.exit(0) right after
# model init, so no training step ever runs.
#
# Default: 4×H100, Qwen3.5-35B-A3B, TP=2 EP=2 GEN_TP=2 (with ALL_OFFLOAD=True).
#
# Run:
#   bash tests/sharded_refit_verify/run_verify_megatron.sh
#
# Override examples:
#   # 8×H100 — exercise the canonical config (TP=2 EP=8 GEN_TP=8)
#   TP=2 EP=8 GEN_TP=8 NDEVICES_PER_NODE=8 \
#     bash tests/sharded_refit_verify/run_verify_megatron.sh
#
#   # Use local HF cache path
#   HF_MODEL_PATH=/root/models/Qwen3.5-35B-A3B \
#     bash tests/sharded_refit_verify/run_verify_megatron.sh
#
# Requirements (on top of verl base env, per run_qwen3_5_35b_megatron.sh):
#   pip install --upgrade transformers
#   pip install flash-linear-attention
#   pip install -U git+https://github.com/ISEEKYAN/mbridge.git
#   Megatron-LM==0.16.0
#
# Dataset: geo3k parquet at $HOME/data/geo3k/{train,test}.parquet
#   (Hook exits before data is loaded, but Hydra needs paths to parse.)
#   Override with: train_path=/path/to/x.parquet test_path=/path/to/y.parquet

set -xeuo pipefail

# Hook trigger
export VERIFY_MOE_LAYOUT=1

# 4×H100 minimal config (override-friendly)
export TP=${TP:-2}
export PP=${PP:-1}
export CP=${CP:-1}
export EP=${EP:-2}
export ETP=${ETP:-1}
export GEN_TP=${GEN_TP:-2}
export NDEVICES_PER_NODE=${NDEVICES_PER_NODE:-4}

# Offload during init is fine; we exit before any forward/backward.
export ALL_OFFLOAD=${ALL_OFFLOAD:-True}

# Model (defaults to local HF cache layout in the upstream launcher)
export HF_MODEL_PATH=${HF_MODEL_PATH:-Qwen/Qwen3.5-35B-A3B}

# Dummy small dataset paths (hook exits before data load, but Hydra parses these)
export train_path=${train_path:-$HOME/data/geo3k/train.parquet}
export test_path=${test_path:-$HOME/data/geo3k/test.parquet}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec bash "${REPO_ROOT}/examples/grpo_trainer/run_qwen3_5_35b_megatron.sh"
