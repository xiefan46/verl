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
"""Compare two ``driver.py`` dumps to gate sharded NCCL weight equivalence.

Gate 1 (hard): greedy token sequence MUST be identical. Any divergence
here means the sharded backend delivered structurally different weights
to vLLM (wrong shard at wrong position, missing tensor, wrong dtype).

Gate 2 (soft): per-position chosen-token logprobs must be within a BF16
tolerance. Small differences (~1e-3) are expected due to non-associative
floating-point accumulation order changes between the two NCCL routing
strategies — anything beyond that indicates a numerically-off shard.

Usage::

    python compare.py /tmp/dump_nccl.json /tmp/dump_sharded_nccl.json
"""

from __future__ import annotations

import argparse
import json
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_a")
    ap.add_argument("dump_b")
    ap.add_argument(
        "--atol",
        type=float,
        default=5e-3,
        help="Absolute tolerance on per-position chosen-token logprob (default 5e-3, BF16-tight)",
    )
    args = ap.parse_args()

    with open(args.dump_a) as f:
        a = json.load(f)
    with open(args.dump_b) as f:
        b = json.load(f)

    print(f"[compare] A: backend={a['backend']!r}  text={a['text']!r}")
    print(f"[compare] B: backend={b['backend']!r}  text={b['text']!r}")
    if a.get("zero_init_trainer") or b.get("zero_init_trainer"):
        print("[compare] DISTINGUISHING mode (zero-init trainer): both backends were forced to push zeros.")

    # ----- Gate 1: token sequence -----
    if a["tokens"] != b["tokens"]:
        first_diff = next(
            (i for i, (x, y) in enumerate(zip(a["tokens"], b["tokens"], strict=False)) if x != y),
            min(len(a["tokens"]), len(b["tokens"])),
        )
        ctx_lo = max(0, first_diff - 2)
        ctx_hi = min(len(a["tokens"]), first_diff + 3)
        print(f"[compare] FAIL: greedy token mismatch at position {first_diff}")
        print(f"[compare]   A[{ctx_lo}:{ctx_hi}] = {a['tokens'][ctx_lo:ctx_hi]}")
        print(f"[compare]   B[{ctx_lo}:{ctx_hi}] = {b['tokens'][ctx_lo:ctx_hi]}")
        return 1
    print(f"[compare] PASS gate 1 (tokens): {len(a['tokens'])} positions identical")

    # ----- Gate 2: chosen-token logprobs -----
    max_diff = 0.0
    max_idx = -1
    for i, (la, lb) in enumerate(zip(a["token_logprobs"], b["token_logprobs"], strict=False)):
        if la is None or lb is None:
            continue
        d = abs(la - lb)
        if d > max_diff:
            max_diff = d
            max_idx = i

    print(f"[compare] max chosen-token logprob diff: {max_diff:.6f} at position {max_idx}  (atol={args.atol})")
    if max_diff > args.atol:
        print("[compare] FAIL gate 2: logprob diff exceeds tolerance")
        # Print the full per-position diffs to make root-cause faster.
        for i, (la, lb) in enumerate(zip(a["token_logprobs"], b["token_logprobs"], strict=False)):
            if la is not None and lb is not None:
                print(f"[compare]   pos {i}: A={la:.6f}  B={lb:.6f}  diff={abs(la - lb):.6f}  tok={a['tokens'][i]!r}")
        return 1
    print("[compare] PASS gate 2 (logprobs)")

    print("[compare] ===== sharded_nccl e2e: PASSED =====")
    return 0


if __name__ == "__main__":
    sys.exit(main())
