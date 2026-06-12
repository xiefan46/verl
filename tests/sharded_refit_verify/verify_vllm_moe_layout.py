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
"""Verify vLLM MoE parameter layout for Qwen3.5 (sharded-aware refit M3 prep).

Loads a Qwen3.5 MoE model into vLLM and prints:
  - FusedMoE-related layer 0 module type + named_parameters
  - Full named_parameters keys + shapes for layer 0
  - Whether experts are grouped (single fused tensor) or list-of-modules
  - packed_modules_mapping (HF→vLLM fused param mapping — critical for M3)

Run on 2×H100 (Qwen3.5-35B-A3B at TP=2):
    cd verl
    python tests/sharded_refit_verify/verify_vllm_moe_layout.py \
        --model Qwen/Qwen3.5-35B-A3B \
        --tp 2

vLLM requirements: Qwen3.5 support landed in vLLM main ~2026-03. Use a
recent build (>= v0.10 or main). If your build is too old, fallback:
    python tests/sharded_refit_verify/verify_vllm_moe_layout.py \
        --model Qwen/Qwen3-30B-A3B \
        --tp 2
(Qwen3 and Qwen3.5 share the same MoE block structure in vLLM, so the
layout dump for Qwen3 is still informative for Qwen3.5 M3 prep.)
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-35B-A3B")
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument(
        "--gpu-mem-util",
        type=float,
        default=0.6,
        help="Lower this if running alongside other GPU jobs.",
    )
    args = parser.parse_args()

    print(f"\n{'=' * 78}")
    print("=== vLLM MoE Layout Verify (sharded-aware refit M3 prep) ===")
    print(f"{'=' * 78}")
    print(f"Model:            {args.model}")
    print(f"tensor_parallel:  {args.tp}")
    print(f"max_model_len:    {args.max_model_len}")
    print()

    from vllm import LLM

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        enforce_eager=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem_util,
        trust_remote_code=True,
    )

    # Walk to the underlying model. Path varies across vLLM versions.
    try:
        model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    except AttributeError:
        # vLLM v1 / newer path
        try:
            model = llm.llm_engine.engine_core.engine_core.model_executor.driver_worker.model_runner.model
        except AttributeError:
            # Fallback: collective_rpc to driver worker
            def _get_named(self):
                m = self.model_runner.model
                return [(n, tuple(p.shape), str(p.dtype)) for n, p in m.named_parameters()]

            result = llm.llm_engine.collective_rpc(_get_named)[0]
            print(f"=== Got params via collective_rpc ({len(result)} total) ===")
            _print_layer0(result)
            return 0

    print(f"\nModel class: {type(model).__name__}")
    print(f"MRO: {[c.__name__ for c in type(model).__mro__[:5]]}")

    # Try walking to layer 0 MoE module.
    print("\n=== layer 0 module structure ===")
    try:
        if hasattr(model, "model"):
            inner = model.model
        else:
            inner = model
        if hasattr(inner, "layers"):
            layer0 = inner.layers[0]
        elif hasattr(inner, "decoder"):
            layer0 = inner.decoder.layers[0]
        else:
            layer0 = None

        if layer0 is not None:
            print(f"Layer 0 type:       {type(layer0).__name__}")
            mlp = getattr(layer0, "mlp", None) or getattr(layer0, "feed_forward", None)
            if mlp is not None:
                print(f"Layer 0 mlp type:   {type(mlp).__name__}")
                for attr_name in ["experts", "gate", "shared_expert", "shared_experts"]:
                    sub = getattr(mlp, attr_name, None)
                    if sub is not None:
                        print(f"  .{attr_name} type: {type(sub).__name__}")
                        # Check for FusedMoE typical fused tensors
                        for fattr in ["w13_weight", "w2_weight", "weight", "linear_fc1", "linear_fc2"]:
                            v = getattr(sub, fattr, None)
                            if v is not None and hasattr(v, "shape"):
                                print(f"    .{attr_name}.{fattr}\t{tuple(v.shape)}\t{v.dtype}")
    except Exception as e:
        print(f"Layer walk failed: {e}")
        import traceback

        traceback.print_exc()

    # Dump all named_parameters for layer 0 (most useful output).
    print("\n=== layer 0 named_parameters (rank-0 driver_worker view) ===")
    layer0_params = [(n, tuple(p.shape), str(p.dtype)) for n, p in model.named_parameters() if "layers.0." in n]
    for n, s, d in layer0_params:
        print(f"  {n}\t{s}\t{d}")
    print(f"\n(total {len(layer0_params)} params in layer 0)")

    print("\n=== first 8 named_parameters() (embedding + first layer head) ===")
    for i, (n, p) in enumerate(model.named_parameters()):
        if i >= 8:
            break
        print(f"  {i:3d}: {n}\t{tuple(p.shape)}")

    # Dump packed_modules_mapping if exposed — this is the key info for
    # HF→vLLM fused offset computation in build_transfer_plan.
    print("\n=== packed_modules_mapping (HF→vLLM fused param mapping) ===")
    pmm = getattr(model, "packed_modules_mapping", None)
    if pmm is not None:
        for k, v in pmm.items():
            print(f"  {k!r}: {v}")
    else:
        print("  (model has no packed_modules_mapping attribute)")

    print(f"\n{'=' * 78}")
    print("=== END VERIFY ===")
    print(f"{'=' * 78}\n")
    return 0


def _print_layer0(params):
    layer0 = [(n, s, d) for n, s, d in params if "layers.0." in n]
    print("=== layer 0 named_parameters ===")
    for n, s, d in layer0:
        print(f"  {n}\t{s}\t{d}")
    print(f"(total {len(layer0)} layer-0 params)")


if __name__ == "__main__":
    sys.exit(main())
