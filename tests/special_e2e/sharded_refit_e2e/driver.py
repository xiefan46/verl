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
"""Sharded-aware NCCL weight-refit end-to-end driver.

Stands up a minimal verl train+rollout colocated stack on a single GPU
(Qwen2.5-0.5B-Instruct on Megatron with everything = 1), runs the
checkpoint engine's ``update_weights`` once (which exercises the
selected backend's full data path: trainer engine → checkpoint engine
→ rollout-side bridge → vLLM worker weight loaders), then issues a
greedy completion against the freshly-loaded vLLM and dumps both the
generated tokens and their per-position logprobs to JSON.

The companion ``compare.py`` is run after this script has been invoked
twice with different ``--backend`` values; if the two backends produce
identical greedy tokens and per-position logprobs (within BF16
tolerance), the sharded backend is validated end-to-end.

This script is *intentionally* not a pytest case — pytest's per-test
fixture lifecycle struggles with multiple full Ray-cluster setups in
one process. Two cleanly-isolated subprocess invocations is the safer
shape.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import ray
from omegaconf import DictConfig

# Make verl root importable when invoked from the repo root.
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/../../.."))


def build_config(backend: str, model_path: str) -> DictConfig:
    """Compose ``ppo_megatron_trainer`` with single-GPU Qwen2.5-0.5B overrides.

    Mirrors the structure of ``run_ppo_trainer_megatron.sh`` but shrinks
    every parallel dimension to 1 and disables critic/reward — we only
    need actor + rollout + checkpoint engine.
    """
    from hydra import compose, initialize_config_dir

    config_dir = os.path.abspath("verl/trainer/config")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        config = compose(config_name="ppo_megatron_trainer")

    # ---- cluster: 2 GPUs total (1 trainer + 1 standalone rollout) ----
    # We CANNOT use the default hybrid mode (trainer+rollout colocated in one
    # WorkerDict actor) — that path has no separate CheckpointEngineWorker
    # and ``CheckpointEngineManager.build_process_group`` blows up looking
    # for ``execute_checkpoint_engine`` on the trainer's WorkerDict.
    # Standalone rollout (LLMServerManager.create with worker_group=None)
    # creates its own CheckpointEngineWorker actors, which is what every
    # NCCL-class checkpoint backend expects.
    config.trainer.n_gpus_per_node = 1
    config.trainer.nnodes = 1
    config.actor_rollout_ref.hybrid_engine = False
    config.trainer.total_epochs = 1
    config.trainer.total_training_steps = 1
    config.trainer.val_before_train = False
    config.trainer.test_freq = -1
    config.trainer.save_freq = -1
    config.trainer.resume_mode = "disable"
    config.trainer.logger = "console"
    config.trainer.project_name = "sharded-refit-e2e"
    config.trainer.experiment_name = f"sharded-refit-{backend}"

    # ---- model ----
    config.actor_rollout_ref.model.path = model_path
    config.actor_rollout_ref.model.use_remove_padding = False

    # ---- Megatron actor: every parallel dim = 1 ----
    actor_mc = config.actor_rollout_ref.actor.megatron
    actor_mc.pipeline_model_parallel_size = 1
    actor_mc.virtual_pipeline_model_parallel_size = None
    actor_mc.context_parallel_size = 1
    actor_mc.tensor_model_parallel_size = 1
    actor_mc.expert_model_parallel_size = 1
    actor_mc.expert_tensor_parallel_size = 1
    # Offloading is overhead we don't need at 0.5B.
    actor_mc.param_offload = False
    actor_mc.grad_offload = False
    actor_mc.optimizer_offload = False
    actor_mc.use_dist_checkpointing = False

    # ---- Megatron ref (mirror actor) ----
    ref_mc = config.actor_rollout_ref.ref.megatron
    ref_mc.pipeline_model_parallel_size = 1
    ref_mc.virtual_pipeline_model_parallel_size = None
    ref_mc.context_parallel_size = 1
    ref_mc.tensor_model_parallel_size = 1
    ref_mc.expert_model_parallel_size = 1
    ref_mc.expert_tensor_parallel_size = 1
    ref_mc.param_offload = False

    # ---- batch sizes (tiny — we never run an actual training step) ----
    config.data.train_batch_size = 4
    config.data.max_prompt_length = 64
    config.data.max_response_length = 32
    config.actor_rollout_ref.actor.ppo_mini_batch_size = 2
    config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu = 1
    config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu = 1
    config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu = 1

    # ---- LR scheduler — RayPPOTrainer.fit() normally resolves these from the
    # dataloader before init_model(); we bypass fit() so we have to set them
    # by hand. Without this Megatron's OptimizerParamScheduler hits
    # ``assert lr_decay_steps > 0`` because total_training_steps stays at -1.
    config.actor_rollout_ref.actor.optim.total_training_steps = 1
    config.actor_rollout_ref.actor.optim.lr_decay_steps = 1
    config.actor_rollout_ref.actor.optim.lr_warmup_steps = 0

    # ---- rollout: vLLM async, STANDALONE (own placement group, separate process) ----
    # Standalone mode requires rollout.nnodes >= 1 + rollout.n_gpus_per_node
    # so LLMServerManager can create its own resource pool.
    config.actor_rollout_ref.rollout.name = "vllm"
    config.actor_rollout_ref.rollout.mode = "async"
    config.actor_rollout_ref.rollout.nnodes = 1
    config.actor_rollout_ref.rollout.n_gpus_per_node = 1
    config.actor_rollout_ref.rollout.tensor_model_parallel_size = 1
    config.actor_rollout_ref.rollout.gpu_memory_utilization = 0.5
    config.actor_rollout_ref.rollout.enforce_eager = True
    config.actor_rollout_ref.rollout.n = 1
    # *** THE BACKEND UNDER TEST ***
    config.actor_rollout_ref.rollout.checkpoint_engine.backend = backend

    # ---- disable critic + reward (out of scope) ----
    config.reward.reward_model.enable = False
    config.reward.reward_model.enable_resource_pool = False
    # Critic is wired by main_ppo's RayPPOTrainer; init_agent_loop_manager
    # doesn't spin it up, so no further opt-out needed here.

    return config


def init_separated_stack(config: DictConfig):
    """Spin up trainer + standalone rollout + CheckpointEngineManager, sync once.

    Diverges from ``tests.experimental.agent_loop.agent_utils.init_agent_loop_manager``
    in one critical place: we pass ``worker_group=None`` to
    ``LLMServerManager.create`` so the rollout side runs as standalone
    ``CheckpointEngineWorker`` actors (not WorkerDict-wrapped colocated
    actors). That's the only configuration where the checkpoint engine's
    ``execute_checkpoint_engine`` RPCs land on the right method —
    ``WorkerDict`` would route everything through an ``actor_rollout_``
    prefix and the call fails on AttributeError.
    """
    from verl.checkpoint_engine import CheckpointEngineManager
    from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
    from verl.single_controller.ray.base import create_colocated_worker_cls
    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
    from verl.utils import omega_conf_to_dataclass
    from verl.utils.device import get_device_name
    from verl.workers.engine_workers import ActorRolloutRefWorker
    from verl.workers.rollout.llm_server import LLMServerManager

    # 1. trainer-only worker group (own GPU)
    role_worker_mapping = {Role.ActorRollout: ray.remote(ActorRolloutRefWorker)}
    trainer_pool_id = "trainer_pool"
    resource_pool_spec = {trainer_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes}
    mapping = {Role.ActorRollout: trainer_pool_id}
    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
    resource_pool_manager.create_resource_pool()
    resource_pool = resource_pool_manager.get_resource_pool(Role.ActorRollout)
    actor_cls = RayClassWithInitArgs(
        cls=role_worker_mapping[Role.ActorRollout],
        config=config.actor_rollout_ref,
        role="actor_rollout",
    )
    worker_dict_cls = create_colocated_worker_cls(class_dict={"actor_rollout": actor_cls})
    wg_dict = RayWorkerGroup(
        resource_pool=resource_pool,
        ray_cls_with_init=worker_dict_cls,
        device_name=get_device_name(),
    )
    actor_rollout_wg = wg_dict.spawn(prefix_set={"actor_rollout"})["actor_rollout"]
    actor_rollout_wg.init_model()

    # 2. STANDALONE rollout (own GPU, separate process, CheckpointEngineWorker actors)
    llm_server_manager = LLMServerManager.create(config=config, worker_group=None)

    # 3. checkpoint manager — this is the gate. update_weights() runs the full
    # routing-plan / NCCL P2P / bucket-IPC pipeline against the live actor
    # Megatron module on the trainer side and the live vLLM workers on the
    # rollout side.
    checkpoint_manager = CheckpointEngineManager(
        config=omega_conf_to_dataclass(config.actor_rollout_ref.rollout.checkpoint_engine),
        trainer=actor_rollout_wg,
        replicas=llm_server_manager.get_replicas(),
    )
    checkpoint_manager.sleep_replicas()
    checkpoint_manager.update_weights()

    return llm_server_manager


def run_e2e(backend: str, model_path: str, prompt: str) -> dict:
    """Spin up verl, sync weights once via ``backend``, generate, return dump."""
    from openai import OpenAI

    # Fresh Ray each invocation — the companion shell script runs this
    # as a subprocess per backend, so we own the cluster lifecycle.
    if ray.is_initialized():
        ray.shutdown()
    ray.init(
        runtime_env={
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "WARN",
                "VLLM_USE_V1": "1",
            }
        }
    )

    print(f"[driver:{backend}] initializing separated stack (model={model_path}) ...", flush=True)
    t0 = time.time()
    llm_server_manager = init_separated_stack(build_config(backend, model_path))
    init_seconds = time.time() - t0
    print(f"[driver:{backend}] init done in {init_seconds:.1f}s", flush=True)

    # Greedy completion with prompt_logprobs disabled (vLLM /completions
    # endpoint exposes per-position token logprobs via ``logprobs`` int).
    server_address = llm_server_manager.server_addresses[0]
    client = OpenAI(api_key="x", base_url=f"http://{server_address}/v1")
    response = client.completions.create(
        model=model_path,
        prompt=prompt,
        max_tokens=8,
        temperature=0.0,
        logprobs=1,  # top-1 per position; we compare against the chosen-token logprob anyway
        echo=False,
    )

    choice = response.choices[0]
    dump = {
        "backend": backend,
        "model_path": model_path,
        "prompt": prompt,
        "text": choice.text,
        # Per-position arrays. ``tokens`` are token strings; ``token_logprobs``
        # are floats for each generated position. ``top_logprobs`` is a list of
        # {token_str: float} maps, one per position.
        "tokens": list(choice.logprobs.tokens),
        "token_logprobs": [None if x is None else float(x) for x in choice.logprobs.token_logprobs],
        "top_logprobs": [{k: float(v) for k, v in tl.items()} for tl in choice.logprobs.top_logprobs],
        "init_seconds": init_seconds,
    }

    ray.shutdown()
    return dump


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=["nccl", "sharded_nccl"])
    ap.add_argument("--model-path", required=True, help="Local HF model directory")
    ap.add_argument("--out-path", required=True, help="JSON dump destination")
    ap.add_argument(
        "--prompt",
        default="The capital of France is",
        help="Deterministic prompt for the greedy completion gate",
    )
    args = ap.parse_args()

    if not os.path.isdir(args.model_path):
        raise FileNotFoundError(
            f"--model-path must point at a local HF model directory, got {args.model_path!r}. "
            f"Run: hf download <model_id> --local-dir {args.model_path}"
        )

    dump = run_e2e(args.backend, args.model_path, args.prompt)

    with open(args.out_path, "w") as f:
        json.dump(dump, f, indent=2)
    print(
        f"[driver:{args.backend}] PASS — dumped {len(dump['tokens'])} tokens to {args.out_path}",
        flush=True,
    )
    print(f"[driver:{args.backend}] generated text: {dump['text']!r}", flush=True)


if __name__ == "__main__":
    main()
