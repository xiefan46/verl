# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Dynamic-trie prefix tree construction benchmark — absolute timing only.

"""Measure absolute CPU time of build_prefix_tree_micro_batch_dynamic across scenarios.

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
import importlib.util
import json
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass


def _load_verl_module(rel_path: str, mod_name: str):
    """Side-load a verl util module without triggering verl/__init__.py heavy deps."""
    here = os.path.dirname(os.path.abspath(__file__))
    full = os.path.normpath(os.path.join(here, "../../..", rel_path))
    spec = importlib.util.spec_from_file_location(mod_name, full)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


# Load in dependency order (prefix_tree_dynamic imports from prefix_tree_magi)
_load_verl_module("verl/utils/prefix_tree_params.py", "verl.utils.prefix_tree_params")
_load_verl_module("verl/utils/prefix_tree_utils.py", "verl.utils.prefix_tree_utils")
_load_verl_module("verl/utils/prefix_tree_magi.py", "verl.utils.prefix_tree_magi")
_dyn = _load_verl_module("verl/utils/prefix_tree_dynamic.py", "verl.utils.prefix_tree_dynamic")

build_dynamic = _dyn.build_prefix_tree_micro_batch_dynamic

from scenarios import all_scenarios, b1_scenarios, b2_scenarios, b3_paper_scenarios  # noqa: E402


@dataclass
class BenchmarkResult:
    scenario_name: str
    iters: int
    total_p50_ms: float
    total_p95_ms: float
    total_mean_ms: float
    batch_size: int
    total_tokens: int


def _run_one(scenario, iters: int, warmup: int) -> BenchmarkResult:
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
        k = int(q * (len(s) - 1))
        return s[k]

    return BenchmarkResult(
        scenario_name=scenario.name,
        iters=iters,
        total_p50_ms=p(total_ms, 0.5),
        total_p95_ms=p(total_ms, 0.95),
        total_mean_ms=statistics.mean(total_ms),
        batch_size=len(scenario.samples),
        total_tokens=sum(t.numel() for t in scenario.samples),
    )


def run_benchmark(scenarios, iters: int, warmup: int) -> list[BenchmarkResult]:
    return [_run_one(s, iters, warmup) for s in scenarios]


def print_benchmark(results: list[BenchmarkResult]):
    print()
    print("=" * 90)
    print("DYNAMIC-TRIE BUILD TIMING (median / p95 / mean over iters, ms)")
    print("=" * 90)
    header = f"{'Scenario':<25} {'P50':>9} {'P95':>9} {'Mean':>9}  {'B':>4} {'Tokens':>10}"
    print(header)
    print("-" * 90)
    for r in sorted(results, key=lambda x: x.scenario_name):
        print(
            f"{r.scenario_name:<25} {r.total_p50_ms:>9.3f} {r.total_p95_ms:>9.3f} "
            f"{r.total_mean_ms:>9.3f}  {r.batch_size:>4} {r.total_tokens:>10}"
        )
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=30, help="# benchmark iterations (default 30)")
    parser.add_argument("--warmup", type=int, default=3, help="# warmup iterations (default 3)")
    parser.add_argument("--b1-only", action="store_true", help="Run only B1 (shallow / depth-3 scenarios)")
    parser.add_argument("--b2-only", action="store_true", help="Run only B2 (deep + long-sequence scenarios)")
    parser.add_argument("--b3-only", action="store_true", help="Run only B3 (paper-derived scenarios)")
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
