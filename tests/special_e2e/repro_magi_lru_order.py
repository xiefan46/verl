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
"""Repro 7: MagiAttention LRU-order bug via ``get_most_recent_key`` cache-hit drift.

Setup-free: no verl, no FSDP, no vLLM, no Ray, no model. Just magi_attention
+ PyTorch. Runs on 1 GPU.

What this repro actually tests
------------------------------
Repros 1-6 all called ``calc_attn(q, k, v, KEY)`` with the **explicit key**
returned from ``magi_attn_flex_key`` — bypassing ``get_most_recent_key``.
None of them exercised verl's real attention path
(``_magi_prefix_tree_attention_forward`` in ``models/transformers/monkey_patch.py``),
which calls ``get_most_recent_key(cp_group)`` to retrieve the key from a
**global per-cp_group LRU**.

The hypothesis under test
-------------------------
``DistAttnRuntimeDict`` (an OrderedDict subclass) only reorders on
``__setitem__`` and ``.get(key)``. Inherited ``__contains__`` does NOT
move-to-end.

``magi_attn_flex_key`` does::

    if key not in dist_attn_runtime_dict_mgr:
        dist_attn_runtime_dict_mgr[key] = init_dist_attn_runtime_mgr(...)
    return key

⇒ on **cache HIT**, the just-built key's LRU position is **untouched**.

Magi's official trainer example (``examples/transformers/magi_trainer.py``)
calls ``dispatch(x, key)`` between build and forward, which invokes
``dist_attn_runtime_dict_mgr.get(key)`` and re-orders. So the official
pattern works.

Verl's dynamic-trie + Magi path SKIPS ``dispatch`` at ``cp_size=1`` (see
``verl/workers/engine/fsdp/transformer_impl.py`` around the
``if self.context_parallel_size > 1`` block). So on single-GPU runs:

  1. compute_log_prob processes N micro-batches with N unique shapes.
     Each ``magi_attn_flex_key`` is a cache MISS → insert at LRU end.
     Forward uses ``get_most_recent_key`` = last-inserted key ✓
  2. ref_log_prob (and update_actor) processes the same shapes again.
     Each ``magi_attn_flex_key`` is a cache HIT → LRU unchanged.
     ``get_most_recent_key`` keeps returning the LAST key from step 1
     for every micro-batch in the new phase. ✗ WRONG MGR.

That explains why ``cache_size=1`` masks the bug (every build is a miss
→ insert → most-recent = current), and why ``MAGI_FIX_CLEAR_CACHE`` also
worked (clears the dict → forced miss).

How this repro proves it
------------------------
Use two keys that have the **same total_seqlen** (so calc_attn won't
crash on shape mismatch) but **different mask types** (FULL vs CAUSAL).
The two mgrs produce visibly different outputs from the same q/k/v.

  1. Build key_A (FULL) and run calc_attn(qkv, key_A) → record o_A_correct.
  2. Build key_B (CAUSAL) and run calc_attn(qkv, key_B) → record o_B_correct.
     LRU end is now key_B.
  3. Re-build key_A. Cache HIT (no LRU reorder). LRU end is still key_B.
  4. Now mimic the verl path: call ``get_most_recent_key`` and forward
     with whatever it returns.
  5. Compare with explicit ``calc_attn(qkv, key_A)``.

Expected outcomes
-----------------
  - cache_size=1000:
        most_recent_key == key_B (BUG)
        out_via_most_recent ≈ o_B_correct
        out_via_most_recent != o_A_correct
        VERDICT: BUG REPRODUCED.
  - cache_size=1:
        Cache only holds 1 entry, so building key_B in step 2 evicts
        key_A. Re-building key_A in step 3 is a cache MISS → fresh
        insert at end → most_recent_key == key_A.
        out_via_most_recent ≈ o_A_correct.
        VERDICT: BUG NOT REPRODUCED (matches the empirical workaround).

Usage
-----
    # Reproduce the bug:
    MAGI_ATTENTION_DIST_ATTN_RUNTIME_DICT_SIZE=1000 \\
        torchrun --standalone --nproc_per_node=1 \\
        tests/special_e2e/repro_magi_lru_order.py

    # Validate the workaround:
    MAGI_ATTENTION_DIST_ATTN_RUNTIME_DICT_SIZE=1 \\
        torchrun --standalone --nproc_per_node=1 \\
        tests/special_e2e/repro_magi_lru_order.py
"""

from __future__ import annotations

import os
import sys

import torch
import torch.distributed as dist

# Qwen2.5-0.5B-Instruct-like shapes (GQA 14/2, head_dim=64) to stay
# consistent with repros 1-6.
NUM_HEADS_Q = 14
NUM_HEADS_KV = 2
HEAD_DIM = 64

SEQ_LEN = 1024
QKV_SEED = 42


def _build_key(mask: str):
    """Register a DistAttnRuntimeKey with the chosen mask type.

    Both keys share total_seqlen so calc_attn never sees a shape
    mismatch even if we feed q/k/v built for one mgr into another.
    Only the mask type differs ⇒ different mgrs ⇒ visibly different
    outputs.
    """
    from magi_attention.api import DistAttnConfig, magi_attn_flex_key
    from magi_attention.common import AttnRanges
    from magi_attention.common.enum import AttnMaskType
    from magi_attention.meta.solver.dispatch_solver import DispatchConfig

    if mask == "full":
        mask_type = AttnMaskType.FULL
    elif mask == "causal":
        mask_type = AttnMaskType.CAUSAL
    else:
        raise ValueError(f"unknown mask: {mask}")

    return magi_attn_flex_key(
        q_ranges=AttnRanges.from_ranges([(0, SEQ_LEN)]),
        k_ranges=AttnRanges.from_ranges([(0, SEQ_LEN)]),
        attn_mask_type=[mask_type],
        total_seqlen_q=SEQ_LEN,
        total_seqlen_k=SEQ_LEN,
        num_heads_q=NUM_HEADS_Q,
        num_heads_kv=NUM_HEADS_KV,
        head_dim=HEAD_DIM,
        pad_size=0,
        cp_group_or_mesh=dist.group.WORLD,
        dist_attn_config=DistAttnConfig(
            dispatch_config=DispatchConfig(uneven_shard=True),
        ),
    )


def _qkv():
    g = torch.Generator(device="cuda").manual_seed(QKV_SEED)
    q = torch.randn(SEQ_LEN, NUM_HEADS_Q, HEAD_DIM, dtype=torch.bfloat16, device="cuda", generator=g)
    k = torch.randn(SEQ_LEN, NUM_HEADS_KV, HEAD_DIM, dtype=torch.bfloat16, device="cuda", generator=g)
    v = torch.randn(SEQ_LEN, NUM_HEADS_KV, HEAD_DIM, dtype=torch.bfloat16, device="cuda", generator=g)
    return q, k, v


def _calc(q, k, v, key) -> tuple[float, float]:
    from magi_attention.api import calc_attn

    o = calc_attn(q, k, v, key)[0]
    of = o.detach().float()
    return float(of.sum().item()), float(of.abs().max().item())


def main() -> int:
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

    from magi_attention.api import get_most_recent_key

    cache_size_env = os.environ.get("MAGI_ATTENTION_DIST_ATTN_RUNTIME_DICT_SIZE", "default(1000)")

    if dist.get_rank() == 0:
        print(f"[REPRO7] MAGI_ATTENTION_DIST_ATTN_RUNTIME_DICT_SIZE={cache_size_env}")
        print(f"[REPRO7] SEQ_LEN={SEQ_LEN} QKV_SEED={QKV_SEED}")
        print()

    q, k, v = _qkv()

    # ── Phase 1: establish ground truth for both mgrs ───────────────
    # Each build is a cache miss → insert at LRU end → calc_attn → get()
    # → moves to end (already there). LRU end after this phase = key_B.
    key_a_first = _build_key("full")
    sum_a, max_a = _calc(q, k, v, key_a_first)

    key_b = _build_key("causal")
    sum_b, max_b = _calc(q, k, v, key_b)

    # ── Phase 2: re-build key_A. At default cache size both keys still
    # live in the cache → cache HIT → no LRU reorder → most-recent stays
    # key_B. At cache_size=1, building key_B evicted key_A, so this is
    # a MISS → insert → most-recent = key_A.
    key_a_again = _build_key("full")

    # Identity check on the key object itself. Magi uses __hash__/__eq__,
    # so cache hit returns the *same* key value (equal hash) even if the
    # Python object identity differs. What matters for the bug is whether
    # ``most_recent`` equals the just-built key.
    most_recent = get_most_recent_key(dist.group.WORLD)
    is_correct_key = most_recent == key_a_again
    is_stale_key = most_recent == key_b

    # ── Phase 3: forward via get_most_recent_key (what verl does) ───
    sum_mr, max_mr = _calc(q, k, v, most_recent)

    # ── Phase 4: forward via explicit key_A (what every prior repro did)
    sum_explicit, max_explicit = _calc(q, k, v, key_a_again)

    if dist.get_rank() == 0:
        print("[REPRO7] Ground truth:")
        print(f"  o_A_correct (key_A FULL):    sum={sum_a:.6f}  abs_max={max_a:.6f}")
        print(f"  o_B_correct (key_B CAUSAL):  sum={sum_b:.6f}  abs_max={max_b:.6f}")
        print()
        print("[REPRO7] After rebuild of key_A (cache hit at size>=2, miss at size=1):")
        print(f"  most_recent == key_A_again: {is_correct_key}")
        print(f"  most_recent == key_B:       {is_stale_key}")
        print()
        print("[REPRO7] Verl-path forward (uses get_most_recent_key):")
        print(f"  sum={sum_mr:.6f}  abs_max={max_mr:.6f}")
        print("[REPRO7] Explicit-key forward (uses key_A directly):")
        print(f"  sum={sum_explicit:.6f}  abs_max={max_explicit:.6f}")
        print()

        # Drift between verl-path output and the correct (explicit-key_A) one
        diff_mr_a = abs(sum_mr - sum_a)
        rel_mr_a = diff_mr_a / (abs(sum_a) + 1e-9)
        # Drift between verl-path output and the WRONG (key_B causal) one
        diff_mr_b = abs(sum_mr - sum_b)
        rel_mr_b = diff_mr_b / (abs(sum_b) + 1e-9)

        print(f"[REPRO7] |o_via_most_recent - o_A_correct| / |o_A_correct|  = {rel_mr_a * 100:.4f}%")
        print(f"[REPRO7] |o_via_most_recent - o_B_correct| / |o_B_correct|  = {rel_mr_b * 100:.4f}%")
        print()

        BOLD = "\033[1m"
        GREEN = "\033[0;32m"
        RED = "\033[0;31m"
        RESET = "\033[0m"

        # bf16 noise floor ~1e-3 relative.
        verl_matches_correct = rel_mr_a < 1e-3
        verl_matches_wrong = rel_mr_b < 1e-3

        if verl_matches_correct and not verl_matches_wrong:
            print(f"  {GREEN}{BOLD}VERDICT: verl path returns CORRECT mgr (matches key_A).{RESET}")
            print(f"  {GREEN}LRU ordering is fine at this cache size. No bug.{RESET}")
        elif verl_matches_wrong and not verl_matches_correct:
            print(f"  {RED}{BOLD}VERDICT: BUG REPRODUCED — verl path returns key_B mgr after rebuilding key_A.{RESET}")
            print(
                f"  {RED}Root cause: cache hit in magi_attn_flex_key skips LRU reorder, "
                f"get_most_recent_key returns stale tail.{RESET}"
            )
        else:
            print(f"  {RED}{BOLD}VERDICT: AMBIGUOUS — verl output matches neither cleanly. Look closer.{RESET}")
            print(f"  rel_to_correct={rel_mr_a * 100:.4f}%  rel_to_wrong={rel_mr_b * 100:.4f}%")

    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
