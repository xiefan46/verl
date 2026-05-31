# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Dynamic-trie prefix tree construction benchmark — absolute timing only.

"""Measure absolute CPU time of build_prefix_tree_micro_batch_dynamic across scenarios.

Three scenario families:
  - B1.*: shallow (depth-3) trees, GRPO-style rollouts.
  - B2.*: deep trees + long-sequence cases (depth up to 8, contexts up to 1M).
  - B3.*: paper-derived realistic configurations (TreeRL, rStar-Math, DeepSearch).

Usage:
    python benchmark_main.py
    python benchmark_main.py --iters 50 --warmup 5
    python benchmark_main.py --b1-only
    python benchmark_main.py --b2-only
    python benchmark_main.py --b3-only

Outputs:
    - stdout: per-scenario median / p95 / mean timing table
    - benchmark_results.json: machine-readable raw results
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from typing import Optional

import torch


# ============================================================================
# Side-load verl modules without triggering verl/__init__.py heavy deps
# ============================================================================


def _load_verl_module(rel_path: str, mod_name: str):
    here = os.path.dirname(os.path.abspath(__file__))
    full = os.path.normpath(os.path.join(here, "../../..", rel_path))
    spec = importlib.util.spec_from_file_location(mod_name, full)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


# Load in dependency order (prefix_tree_dynamic imports from prefix_tree)
_load_verl_module("verl/utils/prefix_tree_params.py", "verl.utils.prefix_tree_params")
_load_verl_module("verl/utils/prefix_tree_utils.py", "verl.utils.prefix_tree_utils")
_load_verl_module("verl/utils/prefix_tree.py", "verl.utils.prefix_tree")
_dyn = _load_verl_module("verl/utils/prefix_tree_dynamic.py", "verl.utils.prefix_tree_dynamic")

build_dynamic = _dyn.build_prefix_tree_micro_batch_dynamic


# ============================================================================
# Scenarios
# ============================================================================


@dataclass
class Scenario:
    name: str
    samples: list[torch.Tensor]
    prefix_segments_batch: Optional[list[list[tuple[int, int]]]]
    expected_depth: int
    notes: str


def _hash(tokens: list[int]) -> int:
    return int(hashlib.sha256(b",".join(str(t).encode() for t in tokens)).hexdigest()[:8], 16)


def _gen_tokens(n: int, base: int, rng: torch.Generator) -> list[int]:
    return torch.randint(base, base + 10_000, (n,), generator=rng).tolist()


def _make_depth3_scenario(
    name: str,
    batch_size: int,
    prefix_len: int,
    mid_len: int,
    leaf_len: int,
    num_groups: int,
    seed: int,
    notes: str = "",
) -> Scenario:
    """root prefix (shared) → num_groups intermediate segments (each shared by group) → per-sample leaves."""
    assert batch_size % num_groups == 0
    samples_per_group = batch_size // num_groups
    assert samples_per_group >= 2, "Need ≥2 samples per group for depth-3"

    rng = torch.Generator().manual_seed(seed)
    prefix_tokens = _gen_tokens(prefix_len, base=1000, rng=rng)

    samples: list[torch.Tensor] = []
    prefix_segments_batch: list[list[tuple[int, int]]] = []
    sample_idx = 0
    for g in range(num_groups):
        mid_tokens = _gen_tokens(mid_len, base=10000 + g * 100000, rng=rng)
        for _ in range(samples_per_group):
            leaf_tokens = _gen_tokens(leaf_len, base=100000 + sample_idx * 1000, rng=rng)
            full = prefix_tokens + mid_tokens + leaf_tokens
            samples.append(torch.tensor(full, dtype=torch.long))
            seg1 = (_hash(prefix_tokens), prefix_len)
            seg2 = (_hash(prefix_tokens + mid_tokens), prefix_len + mid_len)
            seg3 = (_hash(prefix_tokens + mid_tokens + leaf_tokens), prefix_len + mid_len + leaf_len)
            prefix_segments_batch.append([seg1, seg2, seg3])
            sample_idx += 1

    return Scenario(
        name=name, samples=samples, prefix_segments_batch=prefix_segments_batch, expected_depth=3, notes=notes
    )


def _make_deep_scenario(
    name: str,
    batch_size: int,
    depth: int,
    segment_len: int,
    leaf_len: int,
    branch_factor: int,
    seed: int,
    notes: str = "",
) -> Scenario:
    """Balanced tree: root → level2 → ... → level_depth → leaves. branch_factor children per node."""
    rng = torch.Generator().manual_seed(seed)
    samples_tokens: dict[int, list[int]] = {i: [] for i in range(batch_size)}

    def _grow(level: int, sample_indices: list[int]):
        if level >= depth - 1:
            for s in sample_indices:
                leaf = _gen_tokens(leaf_len, base=500000 + s * 10000, rng=rng)
                samples_tokens[s].extend(leaf)
            return
        seg = _gen_tokens(segment_len, base=1000 + level * 100000 + sample_indices[0], rng=rng)
        for s in sample_indices:
            samples_tokens[s].extend(seg)
        n = len(sample_indices)
        if n <= branch_factor:
            for s in sample_indices:
                _grow(level + 1, [s])
        else:
            chunk_size = max(1, n // branch_factor)
            for i in range(0, n, chunk_size):
                _grow(level + 1, sample_indices[i : i + chunk_size])

    _grow(level=0, sample_indices=list(range(batch_size)))
    samples = [torch.tensor(samples_tokens[i], dtype=torch.long) for i in range(batch_size)]

    prefix_segments_batch: list[list[tuple[int, int]]] = []
    for s in samples:
        toks = s.tolist()
        t1_end = min(segment_len, len(toks))
        t2_end = min(2 * segment_len, len(toks))
        t3_end = len(toks)
        prefix_segments_batch.append(
            [
                (_hash(toks[:t1_end]), t1_end),
                (_hash(toks[:t2_end]), t2_end),
                (_hash(toks[:t3_end]), t3_end),
            ]
        )

    return Scenario(
        name=name, samples=samples, prefix_segments_batch=prefix_segments_batch, expected_depth=depth, notes=notes
    )


def _make_long_seq_scenario(
    name: str, batch_size: int, prefix_len: int, leaf_len: int, seed: int, notes: str = ""
) -> Scenario:
    """Depth-2: N samples share a long prefix, diverge at the leaf."""
    rng = torch.Generator().manual_seed(seed)
    prefix_tokens = _gen_tokens(prefix_len, base=1000, rng=rng)
    samples = []
    prefix_segments_batch = []
    for i in range(batch_size):
        leaf = _gen_tokens(leaf_len, base=500000 + i * 10000, rng=rng)
        full = prefix_tokens + leaf
        samples.append(torch.tensor(full, dtype=torch.long))
        prefix_segments_batch.append([(_hash(prefix_tokens), prefix_len), (_hash(full), len(full))])
    return Scenario(
        name=name, samples=samples, prefix_segments_batch=prefix_segments_batch, expected_depth=2, notes=notes
    )


def b1_scenarios() -> list[Scenario]:
    """Shallow depth-3 scenarios (GRPO-style rollouts)."""
    return [
        _make_depth3_scenario("B1.small", 8, 256, 64, 64, num_groups=2, seed=1, notes="P=256,M=64,L=64"),
        _make_depth3_scenario("B1.medium", 16, 512, 128, 128, num_groups=4, seed=2, notes="P=512,M=128,L=128"),
        _make_depth3_scenario("B1.long_prefix", 8, 4096, 256, 256, num_groups=2, seed=3, notes="P=4096,M=256,L=256"),
        _make_depth3_scenario("B1.long_ctx_8k", 8, 6000, 1000, 1000, num_groups=2, seed=4, notes="P=6000,M=1000,L=1000"),
        _make_depth3_scenario("B1.wide_batch", 32, 512, 128, 128, num_groups=8, seed=5, notes="B=32,8 groups"),
    ]


def b2_scenarios() -> list[Scenario]:
    """Deep trees + long sequences.

    For "real" depth-D, we need B = branch_factor^(D-1) samples to avoid dynamic-trie
    compressing single-sample chains at the bottom of the tree.
    """
    return [
        _make_deep_scenario("B2.depth4_B8", 8, 4, 128, 128, branch_factor=2, seed=11, notes="d=4,branch=2,seg=128"),
        _make_deep_scenario("B2.depth5_B16", 16, 5, 128, 128, branch_factor=2, seed=12, notes="d=5,branch=2,seg=128"),
        _make_deep_scenario("B2.depth7_B64", 64, 7, 64, 64, branch_factor=2, seed=13, notes="d=7,branch=2,seg=64"),
        _make_deep_scenario("B2.depth8_B128", 128, 8, 32, 32, branch_factor=2, seed=14, notes="d=8,branch=2,seg=32"),
        _make_long_seq_scenario("B2.long_64k_B8", 8, 60_000, 4_000, seed=21, notes="64K ctx (P=60K,L=4K)"),
        _make_long_seq_scenario("B2.long_128k_B4", 4, 120_000, 8_000, seed=22, notes="128K ctx (P=120K,L=8K)"),
        _make_long_seq_scenario("B2.long_256k_B4", 4, 240_000, 16_000, seed=23, notes="256K ctx (P=240K,L=16K)"),
        _make_long_seq_scenario("B2.long_1M_B2", 2, 950_000, 50_000, seed=24, notes="1M ctx (P=950K,L=50K)"),
        _make_deep_scenario(
            "B2.deep_long_d5_25k", 16, 5, 5_000, 5_000, branch_factor=2, seed=31, notes="d=5, ~25K/sample, 400K total"
        ),
        _make_deep_scenario(
            "B2.deep_long_d4_80k", 8, 4, 20_000, 20_000, branch_factor=2, seed=32, notes="d=4, ~80K/sample, 640K total"
        ),
        _make_deep_scenario(
            "B2.deep_long_d3_300k_1M",
            4,
            3,
            100_000,
            100_000,
            branch_factor=2,
            seed=33,
            notes="d=3, ~300K/sample, 1.2M total",
        ),
    ]


def b3_paper_scenarios() -> list[Scenario]:
    """Scenarios derived from published tree-search/tree-RL paper configurations.

    Sources:
      - TreeRL (arXiv 2506.11902): M=6, T=2, N=2, L=1, B=30 responses/prompt, max_seq=8K, 7B-14B
      - rStar-Math (2501.04519): MCTS depth=16, 8-16 candidates/node, 16 rollouts/problem, avg 5K (MATH)
      - DeepSearch (2509.25454): MCTS depth=64, 8 children/expansion, 256 tokens/node, max response 16K, B=256
    """
    return [
        _make_deep_scenario(
            "B3.TreeRL_B30",
            30,
            3,
            2500,
            3000,
            branch_factor=6,
            seed=41,
            notes="TreeRL: 30 rollouts, 6 sub-trees × 5 leaves, ~8K/sample, 240K total",
        ),
        _make_deep_scenario(
            "B3.rStarMath_B16_d5",
            16,
            5,
            1000,
            1000,
            branch_factor=2,
            seed=42,
            notes="rStar-Math: 16 rollouts × depth-5 from MCTS-16, ~5K/sample, 80K total",
        ),
        _make_long_seq_scenario(
            "B3.DeepSearch_chain_B8",
            8,
            15000,
            1000,
            seed=43,
            notes="DeepSearch: 8 siblings, 15K shared MCTS chain, 16K/sample, 128K total",
        ),
        _make_deep_scenario(
            "B3.DeepSearch_d5_B16",
            16,
            5,
            2000,
            8000,
            branch_factor=2,
            seed=44,
            notes="DeepSearch-style: 5-level MCTS, 16K/sample, 256K total",
        ),
    ]


def all_scenarios() -> list[Scenario]:
    return b1_scenarios() + b2_scenarios() + b3_paper_scenarios()


# ============================================================================
# Benchmark runner
# ============================================================================


@dataclass
class BenchmarkResult:
    scenario_name: str
    iters: int
    total_p50_ms: float
    total_p95_ms: float
    total_mean_ms: float
    batch_size: int
    total_tokens: int


def _run_one(scenario: Scenario, iters: int, warmup: int) -> BenchmarkResult:
    def runner():
        # model=None gates magi_key construction off (CPU-only path)
        return build_dynamic(None, scenario.samples, prefix_segments_batch=scenario.prefix_segments_batch)

    for _ in range(warmup):
        runner()

    total_ms = []
    for _ in range(iters):
        t0 = time.perf_counter()
        runner()
        total_ms.append((time.perf_counter() - t0) * 1000)

    def p(xs, q):
        s = sorted(xs)
        return s[int(q * (len(s) - 1))]

    return BenchmarkResult(
        scenario_name=scenario.name,
        iters=iters,
        total_p50_ms=p(total_ms, 0.5),
        total_p95_ms=p(total_ms, 0.95),
        total_mean_ms=statistics.mean(total_ms),
        batch_size=len(scenario.samples),
        total_tokens=sum(t.numel() for t in scenario.samples),
    )


def run_benchmark(scenarios: list[Scenario], iters: int, warmup: int) -> list[BenchmarkResult]:
    return [_run_one(s, iters, warmup) for s in scenarios]


def print_benchmark(results: list[BenchmarkResult]):
    print()
    print("=" * 90)
    print("DYNAMIC-TRIE BUILD TIMING (median / p95 / mean over iters, ms)")
    print("=" * 90)
    print(f"{'Scenario':<25} {'P50':>9} {'P95':>9} {'Mean':>9}  {'B':>4} {'Tokens':>10}")
    print("-" * 90)
    for r in sorted(results, key=lambda x: x.scenario_name):
        print(
            f"{r.scenario_name:<25} {r.total_p50_ms:>9.3f} {r.total_p95_ms:>9.3f} "
            f"{r.total_mean_ms:>9.3f}  {r.batch_size:>4} {r.total_tokens:>10}"
        )
    print()


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=30, help="# benchmark iterations (default 30)")
    parser.add_argument("--warmup", type=int, default=3, help="# warmup iterations (default 3)")
    parser.add_argument("--b1-only", action="store_true", help="Run only B1 (shallow / depth-3)")
    parser.add_argument("--b2-only", action="store_true", help="Run only B2 (deep + long-sequence)")
    parser.add_argument("--b3-only", action="store_true", help="Run only B3 (paper-derived)")
    parser.add_argument("--output", type=str, default="benchmark_results.json", help="JSON output path")
    args = parser.parse_args()

    if args.b1_only:
        scenarios = b1_scenarios()
    elif args.b2_only:
        scenarios = b2_scenarios()
    elif args.b3_only:
        scenarios = b3_paper_scenarios()
    else:
        scenarios = all_scenarios()

    print(f"Loaded {len(scenarios)} scenarios.")
    for s in scenarios:
        avg = sum(t.numel() for t in s.samples) // len(s.samples)
        print(f"  - {s.name}: B={len(s.samples)}, avg_seq={avg}")

    print(f"\nRunning {args.iters} iters per scenario ({args.warmup} warmup)…")
    t_start = time.perf_counter()
    results = run_benchmark(scenarios, iters=args.iters, warmup=args.warmup)
    print(f"Done in {time.perf_counter() - t_start:.1f}s.")

    print_benchmark(results)

    with open(args.output, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"Raw results written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
