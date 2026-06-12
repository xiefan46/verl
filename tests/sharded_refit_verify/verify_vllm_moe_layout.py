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
"""Verify vLLM MoE parameter layout (sharded-aware refit M3 prep).

Loads a Qwen MoE model into vLLM and prints:
  - FusedMoE-related layer 0 module type
  - Full named_parameters keys + shapes for layer 0
  - Whether experts are grouped (single fused tensor) or list-of-modules
  - packed_modules_mapping (HF→vLLM fused param mapping — critical for M3)

vLLM v1 runs the model in multiprocess engine workers, so we must walk it
via ``collective_rpc`` with a module-level function (lambdas/closures don't
serialize). We set ``VLLM_ALLOW_INSECURE_SERIALIZATION=1`` to allow
cloudpickle of the dump callback.

Run on 2×H100:
    cd verl
    python tests/sharded_refit_verify/verify_vllm_moe_layout.py \\
        --model /root/models/Qwen/Qwen3-30B-A3B --tp 2

(Qwen3 / Qwen3.5 share the same FusedMoE block structure in vLLM, so
Qwen3-30B-A3B is a good stand-in if Qwen3.5 not yet supported by your build.)
"""

from __future__ import annotations

import argparse
import os
import sys

# Must set before importing vllm so the engine workers inherit it.
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")


# Module-level (NOT lambda / closure) so vLLM v1's cloudpickle can serialize
# this callback across the engine subprocess boundary.
def _dump_model_info(self) -> dict:
    """Run inside the vLLM worker; serialize layout info back to driver."""
    m = self.model_runner.model

    # Walk inner model for module type info (best-effort).
    layer0_info: dict = {}
    try:
        inner = m.model if hasattr(m, "model") else m
        if hasattr(inner, "layers"):
            layer0 = inner.layers[0]
        elif hasattr(inner, "decoder"):
            layer0 = inner.decoder.layers[0]
        else:
            layer0 = None

        if layer0 is not None:
            layer0_info["layer0_type"] = type(layer0).__name__
            mlp = getattr(layer0, "mlp", None) or getattr(layer0, "feed_forward", None)
            if mlp is not None:
                layer0_info["mlp_type"] = type(mlp).__name__
                sub_info = {}
                for attr_name in ["experts", "gate", "shared_expert", "shared_experts"]:
                    sub = getattr(mlp, attr_name, None)
                    if sub is not None:
                        entry = {"type": type(sub).__name__, "attrs": {}}
                        for fattr in [
                            "w13_weight",
                            "w2_weight",
                            "weight",
                            "linear_fc1",
                            "linear_fc2",
                        ]:
                            v = getattr(sub, fattr, None)
                            if v is not None and hasattr(v, "shape"):
                                entry["attrs"][fattr] = (tuple(v.shape), str(v.dtype))
                        sub_info[attr_name] = entry
                layer0_info["mlp_children"] = sub_info
    except Exception as e:
        layer0_info["walk_error"] = repr(e)

    return {
        "model_class": type(m).__name__,
        "model_mro": [c.__name__ for c in type(m).__mro__[:5]],
        "params": [(n, tuple(p.shape), str(p.dtype)) for n, p in m.named_parameters()],
        "packed_modules_mapping": getattr(m, "packed_modules_mapping", None),
        "layer0_info": layer0_info,
    }


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

    # vLLM v1: model lives in engine subprocess; RPC into driver worker.
    info = llm.llm_engine.collective_rpc(_dump_model_info)[0]

    print(f"\nModel class: {info['model_class']}")
    print(f"MRO: {info['model_mro']}")

    layer0_info = info.get("layer0_info", {})
    if layer0_info.get("walk_error"):
        print(f"\n=== layer 0 walk error: {layer0_info['walk_error']} ===")
    else:
        print("\n=== layer 0 module structure ===")
        if "layer0_type" in layer0_info:
            print(f"Layer 0 type:       {layer0_info['layer0_type']}")
        if "mlp_type" in layer0_info:
            print(f"Layer 0 mlp type:   {layer0_info['mlp_type']}")
        for sub_name, entry in layer0_info.get("mlp_children", {}).items():
            print(f"  .{sub_name} type: {entry['type']}")
            for fattr, (shape, dtype) in entry["attrs"].items():
                print(f"    .{sub_name}.{fattr}\t{shape}\t{dtype}")

    print("\n=== first 8 named_parameters() (embedding + first layer head) ===")
    for i, (n, s, _d) in enumerate(info["params"][:8]):
        print(f"  {i:3d}: {n}\t{s}")

    print("\n=== layer 0 named_parameters ===")
    layer0_params = [(n, s, d) for n, s, d in info["params"] if "layers.0." in n]
    for n, s, d in layer0_params:
        print(f"  {n}\t{s}\t{d}")
    print(f"\n(total {len(layer0_params)} params in layer 0, {len(info['params'])} total)")

    print("\n=== packed_modules_mapping (HF→vLLM fused param mapping) ===")
    pmm = info["packed_modules_mapping"]
    if pmm is not None:
        for k, v in pmm.items():
            print(f"  {k!r}: {v}")
    else:
        print("  (model has no packed_modules_mapping attribute)")

    print(f"\n{'=' * 78}")
    print("=== END VERIFY ===")
    print(f"{'=' * 78}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
