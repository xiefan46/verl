# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Tree (dynamic-trie + Magi) vs dense (sdpa) accuracy + throughput benchmark.

Compares the two paths on a configurable GRPO-style workload:

  1 unique prompt × N rollouts (each rollout has same prompt + different
  response). This is the regime where tree training is supposed to win:
  the prompt computation is shared across all rollouts, so tree forward
  should be roughly (1 + N * R / (P + R)) cheaper than dense, where
  P = prompt length, R = response length, N = num rollouts.

The script reports:

  * Per-sample logits ``max_diff`` between tree forward and dense forward
    (bf16 noise floor ~ 0.3 - 0.5 for Qwen2.5-0.5B; FAIL if > 1.0).
  * Tree fwd+bwd wall time vs dense fwd+bwd wall time + speedup.
  * Token throughput for each path.

Tunables via env:

  P             prompt length (default 512)
  R             response length (default 256)
  N             rollouts per prompt (default 8)
  WARMUP        warmup iters (default 3)
  ITERS         measurement iters (default 10)
  USE_FSDP      ``1`` to wrap the model in FSDP2 (default 0 — plain HF on
                cuda; turn on to match production verl).
  MODEL_PATH    HF model dir (default ~/models/Qwen/Qwen2.5-0.5B-Instruct)

Usage
-----
    torchrun --standalone --nproc_per_node=1 \\
        tests/special_e2e/bench_dynamic_magi_vs_dense.py

Or with FSDP2 + larger workload:

    USE_FSDP=1 P=1024 R=512 N=16 ITERS=20 \\
        torchrun --standalone --nproc_per_node=1 \\
        tests/special_e2e/bench_dynamic_magi_vs_dense.py
"""

from __future__ import annotations

import os
import sys
import time

import torch
import torch.distributed as dist


def _env_int(name, default):
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


P = _env_int("P", 512)
R = _env_int("R", 256)
N = _env_int("N", 8)
WARMUP = _env_int("WARMUP", 3)
ITERS = _env_int("ITERS", 10)
USE_FSDP = os.environ.get("USE_FSDP", "0") == "1"


def _init_dist():
    if not dist.is_initialized():
        if "RANK" not in os.environ:
            raise RuntimeError("Run via torchrun")
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
        dist.init_process_group(backend="nccl")


def _build_fsdp_qwen(model_path):
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import MixedPrecisionPolicy
    from transformers import Qwen2ForCausalLM

    from verl.utils.fsdp_utils import apply_fsdp2, fsdp2_load_full_state_dict

    model = Qwen2ForCausalLM.from_pretrained(model_path, dtype=torch.float32).cuda()
    world_size = dist.get_world_size()
    mesh = init_device_mesh("cuda", mesh_shape=(world_size,), mesh_dim_names=("fsdp",))
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        cast_forward_inputs=True,
    )
    full_state = model.state_dict()
    apply_fsdp2(
        model, {"mesh": mesh, "mp_policy": mp_policy, "offload_policy": None, "reshard_after_forward": True}, config={}
    )
    fsdp2_load_full_state_dict(model, full_state, mesh, None)
    return model


def _make_grpo_batch(model, p_len, r_len, n_roll):
    """Build N rollouts that share a single P-token prompt."""
    torch.manual_seed(42)
    vocab = model.config.vocab_size
    prompt = torch.randint(0, vocab, (p_len,), device="cuda")
    samples = []
    for i in range(n_roll):
        torch.manual_seed(100 + i)
        resp = torch.randint(0, vocab, (r_len,), device="cuda")
        samples.append(torch.cat([prompt, resp]))
    nested = torch.nested.nested_tensor(samples, layout=torch.jagged)
    return samples, nested


def _tree_forward(model, nested, cp_group):
    """dynamic-trie + Magi packed forward; returns flat logits + pt_batch."""
    from verl.models.transformers.monkey_patch import set_magi_attention_key
    from verl.utils.prefix_tree_dynamic import build_prefix_tree_micro_batch_dynamic
    from verl.utils.prefix_tree_magi import restore_flat_to_nested

    pt_batch = build_prefix_tree_micro_batch_dynamic(
        model,
        nested,
        attention_type="magi",
        cp_group=cp_group,
        cp_size=1,
    )
    assert pt_batch is not None
    set_magi_attention_key(model, pt_batch.magi_key)
    flat_in = pt_batch.local_flat_input_ids.unsqueeze(0)
    flat_pos = pt_batch.local_flat_position_ids.unsqueeze(0)
    out = model(input_ids=flat_in, attention_mask=None, position_ids=flat_pos, use_cache=False)
    flat_logits = out.logits[0][: pt_batch.real_tokens]
    nested_logits = restore_flat_to_nested(flat_logits, pt_batch)
    per_sample = [nested_logits[i] for i in range(pt_batch.original_batch_size)]
    return per_sample, pt_batch


def _dense_forward(model, samples):
    """Per-sample dense (sdpa) forward."""
    model.config._attn_implementation = "sdpa"
    try:
        per_sample = []
        for s in samples:
            out = model(input_ids=s.unsqueeze(0), attention_mask=None, use_cache=False)
            per_sample.append(out.logits[0])
        return per_sample
    finally:
        model.config._attn_implementation = "Magi_Attention"


def _loss_from_logits(per_sample, samples):
    """Mean negative log-prob over each sample's response tokens."""
    losses = []
    for logits, s in zip(per_sample, samples, strict=False):
        f = logits.float()
        labels = s[1:]
        logp = torch.log_softmax(f[:-1], dim=-1).gather(1, labels.unsqueeze(1)).squeeze(1)
        losses.append(-logp.mean())
    return torch.stack(losses).mean()


def _time_iters(label, fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    per_iter = elapsed / iters
    if dist.get_rank() == 0:
        print(f"[BENCH] {label:30s} avg={per_iter * 1000:.2f} ms/iter  total={elapsed:.2f} s ({iters} iter)")
    return per_iter


def main() -> int:
    _init_dist()

    from transformers import Qwen2ForCausalLM

    from verl.models.transformers.monkey_patch import apply_magi_prefix_tree_backend

    apply_magi_prefix_tree_backend()

    model_path = os.environ.get("MODEL_PATH", os.path.expanduser("~/models/Qwen/Qwen2.5-0.5B-Instruct"))
    if dist.get_rank() == 0:
        print(f"[BENCH] Config: P={P} R={R} N={N}  USE_FSDP={USE_FSDP}")
        print(f"[BENCH] Loading {model_path}")

    if USE_FSDP:
        model = _build_fsdp_qwen(model_path)
        model.train()
    else:
        model = Qwen2ForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16).cuda().train()

    model.config._attn_implementation = "Magi_Attention"
    cp_group = dist.group.WORLD if dist.get_world_size() > 0 else None
    n_attached = 0
    for _, mod in model.named_modules():
        cls = mod.__class__.__name__.lower()
        if cls.endswith(("attention", "self_attn", "selfattention")):
            mod.cp_group = cp_group
            n_attached += 1
    if dist.get_rank() == 0:
        print(f"[BENCH] Attached cp_group to {n_attached} attention modules")

    # ── Build a reusable batch
    samples, nested = _make_grpo_batch(model, P, R, N)
    total_tokens = sum(s.shape[0] for s in samples)
    shared_prefix_tokens = P * (N - 1)  # tokens we'd avoid recomputing
    if dist.get_rank() == 0:
        print(
            f"[BENCH] Workload: total_tokens_dense={total_tokens}, "
            f"prefix_shared={shared_prefix_tokens} ({shared_prefix_tokens / total_tokens * 100:.1f}% reusable)"
        )

    # ── 1) Accuracy check: tree vs dense
    if dist.get_rank() == 0:
        print()
        print("[BENCH] Accuracy: tree (dynamic-trie + Magi) vs dense (sdpa)")
    with torch.no_grad():
        tree_logits, _ = _tree_forward(model, nested, cp_group)
        dense_logits = _dense_forward(model, samples)

    if dist.get_rank() == 0:
        max_diff = 0.0
        mean_diff = 0.0
        for t, d in zip(tree_logits, dense_logits, strict=False):
            diff = (t - d).abs()
            max_diff = max(max_diff, diff.max().item())
            mean_diff += diff.mean().item() / len(tree_logits)
        print(f"[BENCH]   max_diff = {max_diff:.4f}  mean_diff = {mean_diff:.4f}")
        if max_diff < 1.0:
            print("[BENCH]   accuracy: PASS (within bf16 noise floor)")
        else:
            print(f"[BENCH]   accuracy: WARN (max_diff={max_diff:.4f} > 1.0)")

    # ── 2) Timing: tree forward + backward
    if dist.get_rank() == 0:
        print()
        print(f"[BENCH] Timing (warmup={WARMUP} iters, measured={ITERS} iters)")

    def _tree_fwd_bwd():
        for p in model.parameters():
            if p.grad is not None:
                p.grad = None
        per_sample, _ = _tree_forward(model, nested, cp_group)
        loss = _loss_from_logits(per_sample, samples)
        loss.backward()

    def _dense_fwd_bwd():
        for p in model.parameters():
            if p.grad is not None:
                p.grad = None
        per_sample = _dense_forward(model, samples)
        loss = _loss_from_logits(per_sample, samples)
        loss.backward()

    tree_t = _time_iters("tree (dynamic-trie + Magi)  fwd+bwd", _tree_fwd_bwd, WARMUP, ITERS)
    dense_t = _time_iters("dense (sdpa)    fwd+bwd", _dense_fwd_bwd, WARMUP, ITERS)

    if dist.get_rank() == 0:
        speedup = dense_t / tree_t if tree_t > 0 else float("inf")
        tree_tput = total_tokens / tree_t
        dense_tput = total_tokens / dense_t
        print()
        print("[BENCH] Throughput:")
        print(f"[BENCH]   tree  : {tree_tput:.0f} tokens/s")
        print(f"[BENCH]   dense : {dense_tput:.0f} tokens/s")
        print(f"[BENCH]   speedup (dense_t / tree_t): {speedup:.2f}x")
        if speedup > 1.0:
            print(f"[BENCH]   tree training is FASTER by {(speedup - 1) * 100:.1f}%")
        else:
            print(f"[BENCH]   tree training is SLOWER by {(1 - speedup) * 100:.1f}%")
            print("[BENCH]   (small workloads or low prefix-sharing → tree overhead wins)")

    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
