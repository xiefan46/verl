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
"""Sharded-aware refit M3 verify hook.

Triggered by env var ``VERIFY_MOE_LAYOUT=1`` from ``MegatronEngine._build_megatron_module``.
Prints Megatron MoE storage layout (grouped vs list, exact shapes, named_parameters keys)
then exits. NO-OP if env var unset.

This file is a one-shot debugging utility. It is safe to delete after M3 layout
assumptions are confirmed and prototype is built.
"""

from __future__ import annotations

import os
import sys

from megatron.core import parallel_state as mpu


def print_layout_and_exit(module, tf_config, hf_config) -> None:
    """Print MoE layout from a fully-initialized Megatron module, then sys.exit(0).

    Args:
        module: the wrapped megatron module list returned by ``make_megatron_module``.
                Each entry is typically a DDP-wrapped (or Float16Module-wrapped) model.
        tf_config: Megatron ``TransformerConfig`` instance.
        hf_config: HuggingFace ``PretrainedConfig`` instance for the model.
    """
    rank = mpu.get_data_parallel_rank() if mpu.is_initialized() else 0
    tp_rank = mpu.get_tensor_model_parallel_rank() if mpu.is_initialized() else 0
    ep_rank = mpu.get_expert_model_parallel_rank() if mpu.is_initialized() else 0
    pp_rank = mpu.get_pipeline_model_parallel_rank() if mpu.is_initialized() else 0

    tp_size = mpu.get_tensor_model_parallel_world_size() if mpu.is_initialized() else 1
    ep_size = mpu.get_expert_model_parallel_world_size() if mpu.is_initialized() else 1
    pp_size = mpu.get_pipeline_model_parallel_world_size() if mpu.is_initialized() else 1

    # Only rank 0 prints to keep output readable.
    if rank == 0 and tp_rank == 0 and ep_rank == 0 and pp_rank == 0:
        print("\n" + "=" * 78)
        print("=== Megatron MoE Layout Verify (sharded-aware refit M3 prep) ===")
        print("=" * 78)
        print(f"Parallel state:  TP={tp_size}, EP={ep_size}, PP={pp_size}")
        print("hf_config:")
        for key in [
            "num_experts",
            "num_experts_per_tok",
            "moe_intermediate_size",
            "shared_expert_intermediate_size",
            "intermediate_size",
            "hidden_size",
            "num_attention_heads",
            "num_key_value_heads",
        ]:
            if hasattr(hf_config, key):
                print(f"  {key:36s} = {getattr(hf_config, key)}")
        print("tf_config (MoE-relevant):")
        for key in [
            "num_moe_experts",
            "moe_grouped_gemm",
            "moe_ffn_hidden_size",
            "moe_shared_expert_intermediate_size",
            "moe_router_topk",
            "moe_token_dispatcher_type",
        ]:
            if hasattr(tf_config, key):
                print(f"  {key:36s} = {getattr(tf_config, key)}")

        # ``module`` is a list (Megatron returns list for VPP / single-stage).
        m = module[0] if isinstance(module, list) else module

        # Walk through wrappers: DDP -> Float16Module -> GPTModel.
        unwrapped = m
        wrappers = []
        while True:
            wrappers.append(type(unwrapped).__name__)
            if hasattr(unwrapped, "module"):
                unwrapped = unwrapped.module
            else:
                break
        print(f"\nWrapper chain: {' -> '.join(wrappers)}")
        print(f"Innermost type: {type(unwrapped).__name__}")

        # Find layer 0 MLP.
        print("\n=== layer 0 MLP structure ===")
        try:
            layer0 = unwrapped.decoder.layers[0]
            print(f"Layer 0 type:           {type(layer0).__name__}")
            print(f"Layer 0 .mlp type:      {type(layer0.mlp).__name__}")

            mlp = layer0.mlp
            if hasattr(mlp, "router"):
                print(f".router type:           {type(mlp.router).__name__}")
                if hasattr(mlp.router, "weight"):
                    print(f"  .router.weight shape: {tuple(mlp.router.weight.shape)}")

            if hasattr(mlp, "shared_experts") and mlp.shared_experts is not None:
                print(f".shared_experts type:   {type(mlp.shared_experts).__name__}")
                for n, p in mlp.shared_experts.named_parameters():
                    print(f"  .shared_experts.{n}\t{tuple(p.shape)}\t{p.dtype}")
            elif hasattr(mlp, "shared_expert") and mlp.shared_expert is not None:
                print(f".shared_expert type:    {type(mlp.shared_expert).__name__}")
                for n, p in mlp.shared_expert.named_parameters():
                    print(f"  .shared_expert.{n}\t{tuple(p.shape)}\t{p.dtype}")

            if hasattr(mlp, "experts"):
                print(f".experts type:          {type(mlp.experts).__name__}")
                experts = mlp.experts
                # Grouped layout: TEGroupedMLP / GroupedMLP / SequentialMLP-with-grouped-gemm
                # has linear_fc1.weight directly on .experts as a 3D tensor [E_local, ...]
                # List layout: ModuleList where each entry is an MLP module.
                grouped_attrs = ["linear_fc1", "linear_fc2"]
                grouped_found = []
                for attr in grouped_attrs:
                    if hasattr(experts, attr):
                        sub = getattr(experts, attr)
                        if hasattr(sub, "weight"):
                            print(f"  .experts.{attr}.weight\t{tuple(sub.weight.shape)}\t{sub.weight.dtype}")
                            grouped_found.append(attr)
                # Some grouped impls keep weights as Parameters at experts level.
                for n, p in experts.named_parameters(recurse=False):
                    if "weight" in n:
                        print(f"  .experts.{n} (direct)\t{tuple(p.shape)}\t{p.dtype}")
                # List style: check if experts is iterable of modules.
                if not grouped_found:
                    try:
                        n_local = len(experts)
                        print(f"  → list-style, len(.experts) = {n_local}")
                        if n_local > 0:
                            expert0 = experts[0]
                            print(f"  .experts[0] type:    {type(expert0).__name__}")
                            for n, p in expert0.named_parameters():
                                print(f"    .experts[0].{n}\t{tuple(p.shape)}\t{p.dtype}")
                    except TypeError:
                        print("  → not iterable; check above grouped attrs")
        except Exception as e:
            print(f"Layer walk failed: {e}")
            import traceback

            traceback.print_exc()

        print("\n=== full named_parameters() of layer 0 (after all wrappers) ===")
        try:
            count = 0
            for n, p in m.named_parameters():
                if ".layers.0." in n or "layers.0." in n:
                    print(f"  {n}\t{tuple(p.shape)}\t{p.dtype}")
                    count += 1
            print(f"  (total {count} params in layer 0)")
        except Exception as e:
            print(f"named_parameters walk failed: {e}")

        print("\n=== first 8 named_parameters() (embedding / first layer / etc.) ===")
        try:
            for i, (n, p) in enumerate(m.named_parameters()):
                if i >= 8:
                    break
                print(f"  {i:3d}: {n}\t{tuple(p.shape)}")
        except Exception:
            pass

        print("=" * 78)
        print("=== END VERIFY (sys.exit 0) ===")
        print("=" * 78 + "\n")
        sys.stdout.flush()

    # All ranks exit cleanly (avoid Ray actor hang).
    import torch

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    sys.exit(0)


def maybe_run_verify(module, tf_config, hf_config) -> None:
    """Entry point. Call this from MegatronEngine._build_megatron_module after
    model is built + weights loaded.

    No-op unless ``VERIFY_MOE_LAYOUT=1``.
    """
    if os.environ.get("VERIFY_MOE_LAYOUT") != "1":
        return
    print_layout_and_exit(module, tf_config, hf_config)
