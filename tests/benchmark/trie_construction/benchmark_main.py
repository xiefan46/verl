# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Trie construction benchmark — main entry point.
# Compares dynamic-trie (token-by-token) vs hash-based (hash-based detection) at
# 3-layer timing granularity. Sanity check is BLOCKING.

"""Main entry: run sanity check → if pass, run benchmarks → print + save report.

Usage:
    python benchmark_main.py
    python benchmark_main.py --iters 50 --warmup 5
    python benchmark_main.py --skip-sanity     # debug only
    python benchmark_main.py --b1-only         # depth-3 fair comparison only
    python benchmark_main.py --b2-only         # dynamic-trie absolute deep tree only

Outputs:
    - stdout: human-readable timing table + decomposition
    - benchmark_results.json: machine-readable raw results
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass

from dynamic_trie_wrapper import build_prefix_tree_micro_batch_dynamic_full
from hash_based_wrapper import build_prefix_tree_micro_batch_hash_based_full
from sanity_check import check_all, print_report
from scenarios import all_scenarios, b1_scenarios, b2_scenarios, b3_paper_scenarios


@dataclass
class BenchmarkResult:
    scenario_name: str
    impl: str  # "dynamic" or "hash-based"
    iters: int
    # All times in ms
    total_p50: float
    total_p95: float
    total_mean: float
    unpack_p50: float
    tree_detect_p50: float
    pack_p50: float
    # Sample count + total tokens for ratio computations
    batch_size: int
    total_tokens: int


def _run_one(runner, scenario, iters: int, warmup: int) -> BenchmarkResult:
    """Run a single (scenario × impl) for `iters` iterations after `warmup` warmups."""
    # Warmup
    for _ in range(warmup):
        runner(scenario)

    total_ms = []
    unpack_ms = []
    tree_detect_ms = []
    pack_ms = []
    for _ in range(iters):
        _, _, t = runner(scenario)
        total_ms.append(t.get("total_ms", 0))
        unpack_ms.append(t.get("unpack_ms", 0))
        tree_detect_ms.append(t.get("tree_detect_ms", 0))
        pack_ms.append(t.get("pack_ms", 0))

    def p(xs, q):
        n = len(xs)
        s = sorted(xs)
        k = int(q * (n - 1))
        return s[k]

    return BenchmarkResult(
        scenario_name=scenario.name,
        impl="?",  # caller fills
        iters=iters,
        total_p50=p(total_ms, 0.5),
        total_p95=p(total_ms, 0.95),
        total_mean=statistics.mean(total_ms),
        unpack_p50=p(unpack_ms, 0.5),
        tree_detect_p50=p(tree_detect_ms, 0.5),
        pack_p50=p(pack_ms, 0.5),
        batch_size=len(scenario.samples),
        total_tokens=sum(t.numel() for t in scenario.samples),
    )


def run_benchmark(scenarios, iters: int, warmup: int) -> list[BenchmarkResult]:
    """Run both implementations implementations on each scenario."""
    results = []
    for s in scenarios:
        # dynamic
        def dyn_run(scn=s):
            return build_prefix_tree_micro_batch_dynamic_full(
                None, scn.samples, prefix_segments_batch=scn.prefix_segments_batch
            )

        dyn_res = _run_one(dyn_run, s, iters, warmup)
        dyn_res.impl = "dynamic"
        results.append(dyn_res)

        # hash-based
        def mt_run(scn=s):
            return build_prefix_tree_micro_batch_hash_based_full(
                None, scn.samples, prefix_segments_batch=scn.prefix_segments_batch
            )

        mt_res = _run_one(mt_run, s, iters, warmup)
        mt_res.impl = "hash-based"
        results.append(mt_res)

    return results


def print_benchmark(results: list[BenchmarkResult]):
    """Print per-scenario side-by-side comparison."""
    # Group by scenario name
    by_scenario: dict[str, dict[str, BenchmarkResult]] = {}
    for r in results:
        by_scenario.setdefault(r.scenario_name, {})[r.impl] = r

    print()
    print("=" * 110)
    print("BENCHMARK RESULTS — 3-layer timing decomposition (median over iters, all times in ms)")
    print("=" * 110)
    header_fmt = (
        f"{'Scenario':<22} {'Impl':<8} {'Total':>8} {'P95':>8} "
        f"{'Unpack':>8} {'TreeDet':>8} {'Pack':>8}  {'B':>4} {'Tokens':>8}"
    )
    print(header_fmt)
    print("-" * 110)
    for name in sorted(by_scenario.keys()):
        grp = by_scenario[name]
        for impl in ["dynamic", "hash-based"]:
            if impl not in grp:
                continue
            r = grp[impl]
            print(
                f"{name:<22} {impl:<8} {r.total_p50:>8.3f} {r.total_p95:>8.3f} {r.unpack_p50:>8.3f} "
                f"{r.tree_detect_p50:>8.3f} {r.pack_p50:>8.3f}  {r.batch_size:>4} {r.total_tokens:>8}"
            )
        # Speedup row
        if "dynamic" in grp and "hash-based" in grp:
            dyn = grp["dynamic"]
            mt = grp["hash-based"]
            ratio = dyn.total_p50 / mt.total_p50 if mt.total_p50 > 0 else float("inf")
            td_ratio = dyn.tree_detect_p50 / mt.tree_detect_p50 if mt.tree_detect_p50 > 0 else float("inf")
            print(f"{'':<22} {'Dyn/Hash':<8} {ratio:>8.2f}x {'':>8} {'':>8} {td_ratio:>8.2f}x {'':>8}")
        print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=30, help="# benchmark iterations (default 30)")
    parser.add_argument("--warmup", type=int, default=3, help="# warmup iterations (default 3)")
    parser.add_argument("--skip-sanity", action="store_true", help="Skip sanity check (NOT recommended)")
    parser.add_argument("--b1-only", action="store_true", help="Run only B1 (depth-3 fair comparison)")
    parser.add_argument("--b2-only", action="store_true", help="Run only B2 (dynamic-trie absolute deep)")
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
        print(f"  - {s.name}: B={len(s.samples)}, avg_seq={sum(t.numel() for t in s.samples) // len(s.samples)}")

    # Sanity check (BLOCKING)
    if not args.skip_sanity:
        print("\n" + "=" * 80)
        print("SANITY CHECK (blocking — benchmark will not run if this fails)")
        print("=" * 80)

        def dyn_run(s):
            return build_prefix_tree_micro_batch_dynamic_full(
                None, s.samples, prefix_segments_batch=s.prefix_segments_batch
            )

        def mt_run(s):
            return build_prefix_tree_micro_batch_hash_based_full(
                None, s.samples, prefix_segments_batch=s.prefix_segments_batch
            )

        sanity_results = check_all(scenarios, dyn_run, mt_run)
        ok = print_report(sanity_results)

        if not ok:
            print("\nABORTING: sanity check failed. Fix correctness before benchmarking.")
            return 1
    else:
        print("\nSKIPPING sanity check (--skip-sanity).")

    # Benchmark
    print("\n" + "=" * 80)
    print(f"RUNNING BENCHMARK: {args.iters} iters per (scenario × impl), {args.warmup} warmup")
    print("=" * 80)

    t_start = time.perf_counter()
    results = run_benchmark(scenarios, iters=args.iters, warmup=args.warmup)
    elapsed = time.perf_counter() - t_start
    print(f"\nBenchmark complete in {elapsed:.1f}s.")

    print_benchmark(results)

    # Save raw results
    with open(args.output, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\nRaw results written to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
