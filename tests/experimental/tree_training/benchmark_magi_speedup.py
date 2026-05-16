# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""Phase K (GPU): speedup-vs-SPR benchmark for the MagiAttention tree path.

Measures forward+backward step time on a small instruct model across three
Shared Prefix Ratios (SPR), comparing the tree path (Magi) to dense FA2 on
the same model + batch. Emits a CSV + a matplotlib plot (PNG) for the
Monday meeting.

Output:
    /tmp/magi_speedup_<timestamp>.csv
    /tmp/magi_speedup_<timestamp>.png

Usage (RunPod 1xH100, after Phase H/I pass):
    cd /root/verl
    python tests/experimental/tree_training/benchmark_magi_speedup.py \\
        --model-path $HOME/models/Qwen/Qwen2.5-0.5B-Instruct

Skips automatically if MagiAttention or CUDA is missing.
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
import time
from dataclasses import dataclass

import torch


@dataclass
class BenchConfig:
    model_path: str
    prompt_len: int
    response_len: int
    rollouts_per_prompt: int
    num_prompts: int
    warmup_steps: int = 3
    measure_steps: int = 10
    max_tokens_per_mb: int = 4096

    @property
    def spr(self) -> float:
        return self.prompt_len / (self.prompt_len + self.response_len)


def _check_env():
    if not torch.cuda.is_available():
        print("CUDA not available, exiting.")
        sys.exit(0)
    try:
        import magi_attention  # noqa: F401
    except ImportError as exc:
        print(f"magi_attention not installed: {exc}")
        sys.exit(0)


def _build_model(model_path: str, attn_impl: str):
    """Load Qwen-style HF model with the requested attention implementation."""
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(model_path)
    config._attn_implementation = attn_impl
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.bfloat16,
    )
    return model.cuda().train(), config


def _make_batch(num_prompts: int, rollouts: int, prompt_len: int, response_len: int, vocab_size: int):
    from tests.experimental.tree_training.synthetic import make_prompt_sharing_batch

    return make_prompt_sharing_batch(
        num_prompts=num_prompts,
        rollouts_per_prompt=rollouts,
        prompt_len=prompt_len,
        response_len=response_len,
        vocab_size=min(vocab_size, 32000),
        device="cuda",
    )


def _bench_dense(model, config, batch, warmup, measure):
    """Per-sequence FA2 forward+backward. Returns list of step times (ms)."""
    input_ids = batch["input_ids"]
    times_ms: list[float] = []
    for step in range(warmup + measure):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        # Sum loss over all sequences then one backward.
        total_loss = torch.zeros((), device="cuda", dtype=torch.float32)
        for i in range(input_ids.size(0)):
            out = model(input_ids=input_ids[i : i + 1], use_cache=False)
            # Mean of logits as a stand-in loss (we're measuring kernel time, not training).
            total_loss = total_loss + out.logits.float().mean()
        total_loss.backward()
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        if step >= warmup:
            times_ms.append(elapsed_ms)
    return times_ms


def _bench_tree(model, config, batch, cfg: BenchConfig):
    """Magi tree-path forward+backward. Returns list of step times (ms)."""
    from verl.experimental.tree_training._areal_data import MicroBatchSpec
    from verl.experimental.tree_training._magi_backend import (
        TreeCPContext,
        register_tree_attention,
        tree_attn_scope,
    )
    from verl.experimental.tree_training._verl_adapter import build_tree_model_inputs
    from verl.experimental.tree_training.tree import build_packed_tree_batch

    register_tree_attention()
    tree_ctx = TreeCPContext(cp_size=1)
    tree_ctx.setup_model(model)

    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    data = {
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"].long(),
    }
    mb_list = build_packed_tree_batch(data, MicroBatchSpec(max_tokens_per_mb=cfg.max_tokens_per_mb))

    times_ms: list[float] = []
    for step in range(cfg.warmup_steps + cfg.measure_steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        total_loss = torch.zeros((), device="cuda", dtype=torch.float32)
        for mb in mb_list.padded_mbs:
            _, output_args, scope_args = build_tree_model_inputs(mb, "cuda")
            trie = output_args["trie"]
            if not trie.all_sequence_ids:
                continue
            packed_input_ids = output_args["packed_input_ids"].cuda()
            position_ids = mb["position_ids"].cuda()
            if position_ids.dim() == 1:
                position_ids = position_ids.unsqueeze(0)
            scope_args.update(
                {
                    "num_heads_q": config.num_attention_heads,
                    "num_heads_kv": getattr(config, "num_key_value_heads", config.num_attention_heads),
                    "head_dim": head_dim,
                    "cp_group": tree_ctx.cp_group,
                }
            )
            with tree_attn_scope(**scope_args):
                out = model(input_ids=packed_input_ids, position_ids=position_ids, use_cache=False)
            total_loss = total_loss + out.logits.float().mean()
        total_loss.backward()
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        if step >= cfg.warmup_steps:
            times_ms.append(elapsed_ms)
    return times_ms


def main():
    _check_env()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        default=os.environ.get("VERL_QWEN_INSTRUCT_PATH", os.path.expanduser("~/models/Qwen/Qwen2.5-0.5B-Instruct")),
    )
    parser.add_argument("--num-prompts", type=int, default=4)
    parser.add_argument("--rollouts", type=int, default=8)
    parser.add_argument("--response-len", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--measure", type=int, default=10)
    parser.add_argument("--out-dir", default="/tmp")
    args = parser.parse_args()

    if not os.path.isdir(args.model_path):
        print(f"model not found at {args.model_path}")
        sys.exit(0)

    # SPR sweep: prompt_len chosen so SPR ~ 0.33, 0.67, 0.89
    spr_configs = [
        (256, args.response_len),  # SPR ~ 0.33
        (1024, args.response_len),  # SPR ~ 0.67
        (4096, args.response_len),  # SPR ~ 0.89
    ]

    results = []
    print(f"\nBenchmarking {args.model_path}\n")
    print(f"{'SPR':>6} {'prompt':>7} {'resp':>5} | {'dense ms':>10} {'tree ms':>10} {'speedup':>9}")
    print("-" * 60)

    for prompt_len, response_len in spr_configs:
        cfg = BenchConfig(
            model_path=args.model_path,
            prompt_len=prompt_len,
            response_len=response_len,
            rollouts_per_prompt=args.rollouts,
            num_prompts=args.num_prompts,
            warmup_steps=args.warmup,
            measure_steps=args.measure,
        )

        # Dense
        model, config = _build_model(args.model_path, attn_impl="flash_attention_2")
        batch = _make_batch(
            cfg.num_prompts, cfg.rollouts_per_prompt, cfg.prompt_len, cfg.response_len, config.vocab_size
        )
        dense_times = _bench_dense(model, config, batch, cfg.warmup_steps, cfg.measure_steps)
        del model
        torch.cuda.empty_cache()

        # Tree (Magi)
        model, config = _build_model(args.model_path, attn_impl="Magi_Tree_Attention")
        # Reuse the batch object (same input_ids).
        tree_times = _bench_tree(model, config, batch, cfg)
        del model
        torch.cuda.empty_cache()

        dense_med = statistics.median(dense_times)
        tree_med = statistics.median(tree_times)
        speedup = dense_med / max(tree_med, 1e-6)
        results.append(
            {
                "spr": round(cfg.spr, 3),
                "prompt_len": prompt_len,
                "response_len": response_len,
                "rollouts": cfg.rollouts_per_prompt,
                "num_prompts": cfg.num_prompts,
                "dense_median_ms": round(dense_med, 2),
                "tree_median_ms": round(tree_med, 2),
                "speedup": round(speedup, 3),
            }
        )
        print(
            f"{cfg.spr:>6.2f} {prompt_len:>7} {response_len:>5} | {dense_med:>10.1f} {tree_med:>10.1f} {speedup:>8.2f}x"
        )

    # Write CSV
    ts = time.strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(args.out_dir, f"magi_speedup_{ts}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\nCSV: {csv_path}")

    # Write plot
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        spr_vals = [r["spr"] for r in results]
        dense_vals = [r["dense_median_ms"] for r in results]
        tree_vals = [r["tree_median_ms"] for r in results]

        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax1.plot(spr_vals, dense_vals, "o-", label="Dense FA2", color="tab:gray")
        ax1.plot(spr_vals, tree_vals, "s-", label="MagiAttention tree", color="tab:blue")
        ax1.set_xlabel("Shared Prefix Ratio (SPR)")
        ax1.set_ylabel("Step time (ms, median)")
        ax1.set_title(
            f"MagiAttention tree training speedup vs dense FA2\n"
            f"{os.path.basename(args.model_path)}  |  "
            f"n={args.rollouts} rollouts/prompt × {args.num_prompts} prompts"
        )
        ax1.legend(loc="upper left")
        ax1.grid(alpha=0.3)

        ax2 = ax1.twinx()
        speedups = [r["speedup"] for r in results]
        ax2.plot(spr_vals, speedups, "^--", label="Speedup", color="tab:red")
        ax2.set_ylabel("Speedup (dense / tree)", color="tab:red")
        ax2.tick_params(axis="y", labelcolor="tab:red")
        ax2.axhline(1.5, color="tab:red", linestyle=":", alpha=0.5, label="1.5x gate")

        plt.tight_layout()
        png_path = os.path.join(args.out_dir, f"magi_speedup_{ts}.png")
        plt.savefig(png_path, dpi=120)
        print(f"PNG: {png_path}")
    except ImportError:
        print("matplotlib not available, skipped plot")

    # Phase K gate
    print("\n=== Phase K gate ===")
    mid_spr_speedup = next((r["speedup"] for r in results if 0.5 < r["spr"] < 0.8), None)
    if mid_spr_speedup is not None:
        gate = "PASS" if mid_spr_speedup >= 1.5 else "SOFT FAIL"
        print(f"SPR ~0.67 speedup = {mid_spr_speedup:.2f}x ({gate}; target >= 1.5x)")


if __name__ == "__main__":
    main()
