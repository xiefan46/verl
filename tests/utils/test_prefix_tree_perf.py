# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Performance test for the dynamic-trie prefix-tree builder.

Runs the builder on representative shapes (varying batch size, context
length, and tree depth) and reports wall-clock per build. Also asserts each
scenario stays under a permissive ceiling so the test fails loudly if a
future refactor regresses performance by an order of magnitude.

Trie construction is CPU-only, so this runs on any host (no CUDA needed).

Usage:
    pytest -s tests/utils/test_prefix_tree_perf.py

The ``-s`` flag prints the timing table.
"""
from __future__ import annotations

import statistics
import time
from typing import Callable

import pytest
import torch
import torch.nested

from verl.utils.prefix_tree_magi import build_prefix_tree_micro_batch


def _make_nested(samples):
    return torch.nested.nested_tensor(
        [torch.tensor(s, dtype=torch.long) for s in samples],
        layout=torch.jagged,
    )


def _measure(fn: Callable[[], object], warmup: int = 2, iters: int = 5) -> float:
    """Run ``fn`` ``warmup`` times, then time ``iters`` runs and return median ms."""
    for _ in range(warmup):
        fn()
    times_ms: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        times_ms.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times_ms)


# ---------------------------------------------------------------------------
# Scenario generators
# ---------------------------------------------------------------------------


def _depth1_grpo_batch(B: int, P: int, R: int):
    """1 prompt × B rollouts, leaf is R unique tokens per sample."""
    prompt = list(range(100, 100 + P))
    return [prompt + list(range(1_000_000 * i, 1_000_000 * i + R)) for i in range(B)]


def _depth2_branched_batch(B: int, P: int, branch_len: int, leaf_len: int, n_branches: int = 2):
    """Root → ``n_branches`` branches → leaves under each."""
    assert B % n_branches == 0, f"batch {B} must be divisible by n_branches {n_branches}"
    prompt = list(range(100, 100 + P))
    per_branch = B // n_branches
    samples = []
    for b in range(n_branches):
        branch_seg = list(range(500_000 + b * 1000, 500_000 + b * 1000 + branch_len))
        for s in range(per_branch):
            sample_id = b * per_branch + s
            leaf = list(range(1_000_000 * (sample_id + 1), 1_000_000 * (sample_id + 1) + leaf_len))
            samples.append(prompt + branch_seg + leaf)
    return samples


def _depth4_branched_batch(B: int, P: int, mid_len: int = 32, leaf_len: int = 32):
    """Root → 2-way split → 2-way split → 2-way split → leaves.

    Produces a depth-4 tree (root + 3 layers of branching). Requires B
    divisible by 8 to fan out evenly.
    """
    assert B % 8 == 0
    prompt = list(range(100, 100 + P))
    samples = []
    sample_id = 0
    for a in range(2):
        seg_a = list(range(500_000 + a * 10, 500_000 + a * 10 + mid_len))
        for b in range(2):
            seg_b = list(range(600_000 + b * 10, 600_000 + b * 10 + mid_len))
            for c in range(2):
                seg_c = list(range(700_000 + c * 10, 700_000 + c * 10 + mid_len))
                per_leaf = B // 8
                for _ in range(per_leaf):
                    leaf = list(
                        range(1_000_000 * (sample_id + 1), 1_000_000 * (sample_id + 1) + leaf_len)
                    )
                    samples.append(prompt + seg_a + seg_b + seg_c + leaf)
                    sample_id += 1
    return samples


def _wide_deep_tree(depth: int, branch_factor: int, segment_len: int, prompt_len: int = 0):
    """Build a perfectly balanced tree: root + ``depth`` layers of
    ``branch_factor``-way fan-out. Total leaves = ``branch_factor ** depth``;
    each sample has length ``prompt_len + (depth + 1) * segment_len``.

    ``prompt_len > 0`` adds a longer shared root prefix on top of the first
    segment. ``segment_len`` is the per-level branch span (root and every
    interior node have this many tokens).
    """
    leaves_total = branch_factor ** depth
    samples: list[list[int]] = []

    # Each non-root level has its own (level, branch-idx-at-level) namespace
    # for token ids, so the trie can correctly tell branches apart.
    def _emit(level: int, ancestor_tokens: list[int], leaf_index_base: int) -> None:
        if level == depth:
            # leaf — append per-sample unique tail
            leaf_tail = list(
                range(
                    10_000_000 + leaf_index_base * segment_len,
                    10_000_000 + leaf_index_base * segment_len + segment_len,
                )
            )
            samples.append(ancestor_tokens + leaf_tail)
            return
        for b in range(branch_factor):
            # Per-level / per-branch namespace: 1e6 * level + 1e4 * b
            base = 1_000_000 * (level + 1) + 10_000 * b
            seg = list(range(base, base + segment_len))
            sub_base = leaf_index_base * branch_factor + b
            _emit(level + 1, ancestor_tokens + seg, sub_base)

    if prompt_len > 0:
        prompt = list(range(100, 100 + prompt_len))
    else:
        prompt = []
    _emit(0, prompt, 0)
    assert len(samples) == leaves_total
    return samples


# ---------------------------------------------------------------------------
# Perf table
# ---------------------------------------------------------------------------


# (label, samples_fn, ceiling_ms)
SCENARIOS = [
    # depth-1, small / medium / large
    ("depth1  B=8   P=512  R=256",      lambda: _depth1_grpo_batch(8, 512, 256),       50),
    ("depth1  B=32  P=512  R=256",      lambda: _depth1_grpo_batch(32, 512, 256),      150),
    ("depth1  B=8   P=4K   R=512",      lambda: _depth1_grpo_batch(8, 4096, 512),      150),
    ("depth1  B=16  P=16K  R=1K",       lambda: _depth1_grpo_batch(16, 16384, 1024),   800),
    ("depth1  B=8   P=64K  R=512",      lambda: _depth1_grpo_batch(8, 65536, 512),     1500),
    # depth-2
    ("depth2  B=8   P=512  br=64 lf=128",  lambda: _depth2_branched_batch(8, 512, 64, 128),    50),
    ("depth2  B=32  P=1K   br=128 lf=256", lambda: _depth2_branched_batch(32, 1024, 128, 256), 300),
    # depth-4 (binary branching)
    ("depth4  B=8   P=512  lf=64",      lambda: _depth4_branched_batch(8, 512, 32, 64),  100),
    ("depth4  B=16  P=2K   lf=128",     lambda: _depth4_branched_batch(16, 2048, 64, 128), 400),
    # Wide-and-deep trees — depth ≥ 5, branch factor ≥ 4 (MCTS / agentic RL territory).
    # Ceilings are ~8× observed Mac CPU median to absorb CI variance without
    # masking real regressions.
    # depth-5 × 4-way = 4^5 = 1024 leaves; per-sample = root + 5 segments
    ("deepwide depth=5 bf=4 seg=32 P=128", lambda: _wide_deep_tree(5, 4, 32, 128),    1500),
    # depth-6 × 4-way = 4096 leaves
    ("deepwide depth=6 bf=4 seg=32 P=128", lambda: _wide_deep_tree(6, 4, 32, 128),    6000),
    # depth-8 × 2-way = 256 leaves; pure depth stress (binary)
    ("deepwide depth=8 bf=2 seg=64 P=128", lambda: _wide_deep_tree(8, 2, 64, 128),     800),
    # depth-5 × 6-way = 7776 leaves — large fan-out, moderate depth
    # (push past realistic MCTS workloads to catch quadratic blow-ups)
    ("deepwide depth=5 bf=6 seg=16 P=64",  lambda: _wide_deep_tree(5, 6, 16, 64),    8000),
]


@pytest.mark.parametrize("label,samples_fn,ceiling_ms", SCENARIOS)
def test_perf(label, samples_fn, ceiling_ms):
    """Each scenario must stay under ``ceiling_ms``. Prints median ms with -s."""
    samples = samples_fn()
    nested = _make_nested(samples)
    total_tokens = sum(len(s) for s in samples)

    def _run():
        pt = build_prefix_tree_micro_batch(None, nested)
        assert pt is not None, f"{label}: builder returned None"
        return pt

    pt = _run()
    median_ms = _measure(_run)
    saved_ratio = 1.0 - pt.flat_input_ids.numel() / total_tokens

    print(
        f"[PERF] {label:42s}  B={len(samples):4d}  "
        f"total_tokens={total_tokens:8d}  "
        f"flat_tokens={pt.flat_input_ids.numel():8d}  "
        f"saved={saved_ratio*100:5.1f}%  "
        f"median={median_ms:7.2f}ms  "
        f"(ceiling={ceiling_ms}ms)",
        flush=True,
    )
    assert median_ms < ceiling_ms, (
        f"{label}: median {median_ms:.2f}ms exceeds ceiling {ceiling_ms}ms"
    )


# ---------------------------------------------------------------------------
# Standalone long-context stress (printed only, not asserted strictly)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "B,P,R,ceiling_ms",
    [
        (8, 16_000, 256, 1_500),
        (8, 64_000, 256, 3_000),
        (8, 128_000, 256, 5_000),
        # 1M per-sample, varying batch. trie cost scales ~linearly in B at
        # this prompt length (each new sample still walks the full prompt
        # once during insertion, even if every step is a hit).
        (2, 1_000_000, 512, 15_000),
        (8, 1_000_000, 512, 25_000),
        (16, 1_000_000, 512, 40_000),
    ],
)
def test_long_context_scaling(B: int, P: int, R: int, ceiling_ms: int):
    """Long-context scaling. ``P`` tokens of shared prompt, ``R`` of unique
    response per rollout, ``B`` rollouts.

    Prints throughput. Asserts a ceiling chosen so the test catches
    catastrophic regressions but doesn't flake on noisy CI machines.
    """
    samples = _depth1_grpo_batch(B=B, P=P, R=R)
    nested = _make_nested(samples)
    total_tokens = sum(len(s) for s in samples)

    def _run():
        pt = build_prefix_tree_micro_batch(None, nested)
        assert pt is not None
        return pt

    # 1M context: tolist() alone is ~half the cost. Single iter + no warmup
    # to keep test runtime sane.
    if P >= 500_000:
        warmup, iters = 0, 1
    else:
        warmup, iters = 1, 3
    median_ms = _measure(_run, warmup=warmup, iters=iters)
    throughput = total_tokens / (median_ms / 1000)

    print(
        f"[PERF-LC] B={B} P={P:7d}  R={R:5d}  total={total_tokens:9d} tokens  "
        f"median={median_ms:8.2f}ms  throughput={throughput / 1e6:6.2f}M tok/s",
        flush=True,
    )
    assert median_ms < ceiling_ms, (
        f"long-context B={B} P={P}: median {median_ms:.2f}ms > {ceiling_ms}ms"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
