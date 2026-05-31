# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Compare reward trajectories between two GRPO runs (dense vs dynamic prefix tree).

Parses console logs from ``verl.trainer.main_ppo``. Assertion fails when:
  - Either run has fewer than 2 reward points (training didn't proceed).
  - The two trajectories have wildly different shapes (corr < min_correlation).
  - Per-step reward differs more than max_step_diff (absolute) at any step.

Used by ``tests/special_e2e/run_grpo_dynamic_prefix_tree.sh``.
"""

from __future__ import annotations

import argparse
import re
import sys

import numpy as np

REWARD_KEYS = (
    "critic/rewards/mean",
    "actor/rewards/mean",
    "rewards/mean",
)


def _extract_step_rewards(log_path: str) -> list[tuple[int, float]]:
    """Walk a verl console log and return [(step, reward), ...] in order.

    Tolerates Ray's ``(TaskRunner pid=...)`` line prefixes that appear when
    verl runs under Ray. We strip everything up to and including the first
    ``)`` if the line starts with ``(``.
    """
    ray_prefix_re = re.compile(r"^\([^)]*\)\s*")
    step_line_re = re.compile(r"^step[ :]+(\d+)\b")
    rewards: list[tuple[int, float]] = []
    with open(log_path) as fh:
        for line in fh:
            # Strip Ray (TaskRunner pid=NNN) prefix if present
            line = ray_prefix_re.sub("", line)
            m = step_line_re.match(line)
            if not m:
                continue
            step_no = int(m.group(1))
            # Lines look like: step:1 - key1:val1 - key2:val2 - ...
            reward_val = None
            for chunk in line.split(" - "):
                if ":" not in chunk:
                    continue
                k, v = chunk.split(":", 1)
                k = k.strip()
                if k in REWARD_KEYS:
                    try:
                        reward_val = float(v.strip())
                    except ValueError:
                        continue
                    break
            if reward_val is not None:
                rewards.append((step_no, reward_val))
    return rewards


def _align_runs(
    a: list[tuple[int, float]],
    b: list[tuple[int, float]],
) -> tuple[np.ndarray, np.ndarray]:
    """Align two run trajectories by step number; keep only steps present in both."""
    a_map = dict(a)
    b_map = dict(b)
    common = sorted(set(a_map) & set(b_map))
    return (
        np.array([a_map[s] for s in common]),
        np.array([b_map[s] for s in common]),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense_log", required=True)
    parser.add_argument("--dyn_log", required=True)
    parser.add_argument("--max_step_diff", type=float, default=0.05, help="max allowed per-step abs reward diff")
    parser.add_argument(
        "--min_correlation", type=float, default=0.85, help="Pearson correlation lower bound between trajectories"
    )
    args = parser.parse_args()

    dense = _extract_step_rewards(args.dense_log)
    dyn = _extract_step_rewards(args.dyn_log)

    print(f"[CHECK] dense steps: {len(dense)}, dynamic-trie steps: {len(dyn)}")
    if len(dense) < 2 or len(dyn) < 2:
        print(f"[CHECK] FAIL — too few reward points (dense={len(dense)}, dyn={len(dyn)})")
        return 1

    dense_arr, dyn_arr = _align_runs(dense, dyn)
    if dense_arr.size < 2:
        print("[CHECK] FAIL — fewer than 2 overlapping steps between runs")
        return 1

    diff = np.abs(dense_arr - dyn_arr)
    max_diff = float(diff.max())
    mean_diff = float(diff.mean())
    final_diff = float(diff[-1])

    if dense_arr.std() < 1e-6 or dyn_arr.std() < 1e-6:
        # Constant trajectory — correlation undefined, only inspect magnitude
        corr = float("nan")
    else:
        corr = float(np.corrcoef(dense_arr, dyn_arr)[0, 1])

    print(f"[CHECK] reward.mean dense={dense_arr.mean():.4f} dyn={dyn_arr.mean():.4f}")
    print(f"[CHECK] per-step abs diff: max={max_diff:.4f} mean={mean_diff:.4f} final={final_diff:.4f}")
    print(f"[CHECK] correlation = {corr:.4f}  (min required {args.min_correlation:.2f})")

    failed: list[str] = []
    if max_diff > args.max_step_diff:
        failed.append(f"max per-step abs diff {max_diff:.4f} > tolerance {args.max_step_diff:.4f}")
    if not np.isnan(corr) and corr < args.min_correlation:
        failed.append(f"correlation {corr:.4f} < min {args.min_correlation:.2f}")

    if failed:
        print("[CHECK] FAIL:")
        for msg in failed:
            print(f"  - {msg}")
        return 1

    print("[CHECK] PASS — Dynamic prefix-tree reward trajectory matches dense baseline within tolerance")
    return 0


if __name__ == "__main__":
    sys.exit(main())
