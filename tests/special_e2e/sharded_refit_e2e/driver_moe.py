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
"""Sharded-aware NCCL weight-refit MoE e2e driver.

Same shape as ``driver.py`` (Qwen2.5-0.5B dense, single-GPU each side),
but configured for **Qwen3-30B-A3B-Instruct on 8×H100**:

* Megatron trainer: 4 GPU, PP=1 CP=1 **TP=2 EP=2** ETP=1.
  Each rank holds 1/2 attention + 64/128 routed experts (~15 GB raw).
  Param + grad + optim offload ON — without offload Adam fp32 state
  alone would blow past 80 GB on H100; with offload, GPU peak during
  the no-train forward walk is ~20 GB.
* vLLM rollout: 4 GPU, **TP=4 EP=4 DP=1**. vLLM enforces
  ``ep_size == tp_size * dp_size`` (see
  :class:`verl.workers.config.rollout.RolloutConfig`); we pair TP=4
  with EP=4 so each rank holds 1/4 attention + 32/128 experts
  (~15 GB weights). ``gpu_memory_utilization=0.7`` leaves headroom
  for the NCCL bucket buffer (2 GB) + CUDA IPC handles + KV cache.

The asymmetric trainer-vs-rollout split is the WHOLE point. It exercises
the routing algorithm's actual value-add over naive broadcast:

* **Cross-TP redistribution (2→4)** — each trainer TP rank holds half
  the attention rows; each rollout TP rank wants a quarter. Every
  trainer rank must split its half across two rollout ranks.
* **Cross-EP redistribution (2→4)** — trainer EP rank 0 owns experts
  [0..64); rollout EP rank 0 wants experts [0..32) and rank 1 wants
  [32..64). The routing plan must dispatch the right per-expert
  slices to the right rollout actor.
* MoE routed-expert routing — vLLM's ``RoutedExperts.weight_loader``
  must accept ``shard_id`` ∈ {"w1","w2","w3"} + ``expert_id: int``.
* TP-split QKV with no bias (Qwen3 has ``attention_bias=False``).

Without cross-TP/EP redistribution the test would collapse to trivial
1:1 routing — covered already by the dense Qwen2.5-0.5B driver.

Reuses ``driver.py``'s ``init_separated_stack`` and ``run_e2e`` to keep
the stack/dump/logprob plumbing in one place.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from omegaconf import DictConfig

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/../../.."))

# Reuse the dense driver's init + run helpers — only ``build_config`` differs.
from tests.special_e2e.sharded_refit_e2e import driver as base_driver  # noqa: E402


def build_config_moe(backend: str, model_path: str) -> DictConfig:
    """Compose ``ppo_megatron_trainer`` for Qwen3-30B-A3B sharded refit."""
    from hydra import compose, initialize_config_dir

    config_dir = os.path.abspath("verl/trainer/config")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        config = compose(config_name="ppo_megatron_trainer")

    # ---- cluster: 4+4 GPU (trainer TP=2 EP=2, rollout TP=4 EP=4 DP=1) ----
    config.trainer.n_gpus_per_node = 4
    config.trainer.nnodes = 1
    config.actor_rollout_ref.hybrid_engine = False
    config.trainer.total_epochs = 1
    config.trainer.total_training_steps = 1
    config.trainer.val_before_train = False
    config.trainer.test_freq = -1
    config.trainer.save_freq = -1
    config.trainer.resume_mode = "disable"
    config.trainer.logger = "console"
    config.trainer.project_name = "sharded-refit-moe-e2e"
    config.trainer.experiment_name = f"sharded-refit-moe-{backend}"

    # ---- model ----
    config.actor_rollout_ref.model.path = model_path
    config.actor_rollout_ref.model.use_remove_padding = False

    # ---- Megatron actor: PP=1 CP=1 TP=2 EP=2 ETP=1 ----
    actor_mc = config.actor_rollout_ref.actor.megatron
    actor_mc.pipeline_model_parallel_size = 1
    actor_mc.virtual_pipeline_model_parallel_size = None
    actor_mc.context_parallel_size = 1
    actor_mc.tensor_model_parallel_size = 2
    actor_mc.expert_model_parallel_size = 2
    actor_mc.expert_tensor_parallel_size = 1
    # 30B dense trunk + half experts per rank exceeds 80GB raw; offload.
    actor_mc.param_offload = True
    actor_mc.grad_offload = True
    actor_mc.optimizer_offload = True
    actor_mc.use_dist_checkpointing = False

    # ---- Megatron ref ----
    ref_mc = config.actor_rollout_ref.ref.megatron
    ref_mc.pipeline_model_parallel_size = 1
    ref_mc.virtual_pipeline_model_parallel_size = None
    ref_mc.context_parallel_size = 1
    ref_mc.tensor_model_parallel_size = 2
    ref_mc.expert_model_parallel_size = 2
    ref_mc.expert_tensor_parallel_size = 1
    ref_mc.param_offload = True

    # ---- batch sizes (tiny — no real training step) ----
    config.data.train_batch_size = 4
    config.data.max_prompt_length = 64
    config.data.max_response_length = 32
    config.actor_rollout_ref.actor.ppo_mini_batch_size = 2
    config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu = 1
    config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu = 1
    config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu = 1

    # ---- LR scheduler stub ----
    config.actor_rollout_ref.actor.optim.total_training_steps = 1
    config.actor_rollout_ref.actor.optim.lr_decay_steps = 1
    config.actor_rollout_ref.actor.optim.lr_warmup_steps = 0

    # ---- rollout: vLLM async, STANDALONE, TP=4 EP=4 DP=1 ----
    config.actor_rollout_ref.rollout.name = "vllm"
    config.actor_rollout_ref.rollout.mode = "async"
    config.actor_rollout_ref.rollout.nnodes = 1
    config.actor_rollout_ref.rollout.n_gpus_per_node = 4
    config.actor_rollout_ref.rollout.tensor_model_parallel_size = 4
    config.actor_rollout_ref.rollout.expert_parallel_size = 4
    config.actor_rollout_ref.rollout.data_parallel_size = 1
    # H100 80 GB: 15 GB model weight + 2 GB bucket buffer + NCCL/IPC
    # state already eats ~25 GB; leave half the GPU for KV cache.
    config.actor_rollout_ref.rollout.gpu_memory_utilization = 0.7
    config.actor_rollout_ref.rollout.enforce_eager = True
    config.actor_rollout_ref.rollout.n = 1
    config.actor_rollout_ref.rollout.checkpoint_engine.backend = backend

    # ---- disable critic + reward ----
    config.reward.reward_model.enable = False
    config.reward.reward_model.enable_resource_pool = False

    return config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=["nccl", "sharded_nccl"])
    ap.add_argument("--model-path", required=True, help="Local Qwen3-30B-A3B-Instruct directory")
    ap.add_argument("--out-path", required=True, help="JSON dump destination")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--zero-init-trainer", action="store_true")
    args = ap.parse_args()

    if not os.path.isdir(args.model_path):
        raise FileNotFoundError(
            f"--model-path must point at a local HF model directory, got {args.model_path!r}. "
            f"Run: hf download Qwen/Qwen3-30B-A3B-Instruct --local-dir {args.model_path}"
        )

    # Monkey-patch base_driver.build_config so run_e2e picks up the MoE
    # config without us having to copy run_e2e.
    base_driver.build_config = build_config_moe
    dump = base_driver.run_e2e(
        args.backend,
        args.model_path,
        args.prompt,
        zero_init_trainer=args.zero_init_trainer,
    )
    with open(args.out_path, "w") as f:
        json.dump(dump, f, indent=2)
    print(
        f"[driver_moe:{args.backend}] PASS — dumped {len(dump['tokens'])} tokens to {args.out_path}",
        flush=True,
    )
    print(f"[driver_moe:{args.backend}] generated text: {dump['text']!r}", flush=True)


if __name__ == "__main__":
    main()
