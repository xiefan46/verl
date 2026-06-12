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
"""Routing-plan algorithm for sharded-aware weight refit.

Given per-rank ``ParameterShardMeta`` lists (training side and rollout
side), produce a ``TransferPlan`` that lists every (src_rank, dst_rank,
src_slice, dst_slice) edge needed to migrate weights from training shards
to rollout shards via NCCL P2P.

The algorithm is pure CPU and DETERMINISTIC: same inputs → identical
output across runs and across ranks (so every worker computing the plan
independently arrives at the same answer, see design doc §3.3 / 5.4).

Determinism strategy:
- Inputs are pre-sorted by ``(param_name, global_rank)`` before grouping.
- All for-loops iterate sorted collections (sorted dict keys, sorted
  candidate lists).
- Tie-breaks on greedy load-balancing use ``global_rank`` ascending.
- No random tie-breaks; no hash-ordered iteration.

A ``TransferPlan.plan_hash`` is emitted so callers can cheaply sanity-check
cross-rank agreement via ``all_reduce(BAND)`` on the hash.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from verl.checkpoint_engine.parallel_meta import (
    GlobalRange,
    ParameterShardMeta,
    TransferEdge,
    TransferPlan,
)


def intersect(box_a: tuple[GlobalRange, ...], box_b: tuple[GlobalRange, ...]) -> tuple[GlobalRange, ...] | None:
    """N-D box intersection.

    Each input is a tuple of per-dim ``(start, end)`` half-open ranges.
    Returns the intersection box, or ``None`` if the boxes do not overlap
    on at least one dim. Both inputs must have the same number of dims;
    a ValueError is raised otherwise.
    """
    if len(box_a) != len(box_b):
        raise ValueError(f"box_a and box_b have different ndim: {len(box_a)} vs {len(box_b)}")
    result: list[GlobalRange] = []
    for (a_start, a_end), (b_start, b_end) in zip(box_a, box_b, strict=True):
        start = max(a_start, b_start)
        end = min(a_end, b_end)
        if start >= end:
            return None  # empty on this dim → empty overall
        result.append((start, end))
    return tuple(result)


def global_box_to_local_slice(
    global_box: tuple[GlobalRange, ...],
    meta_ranges: tuple[GlobalRange, ...],
) -> tuple[tuple[int | None, int | None, int | None], ...]:
    """Translate a global-coordinate box into a per-dim ``slice`` (encoded
    as a tuple) on the LOCAL tensor of a rank whose holdings are described
    by ``meta_ranges``.

    Subtracts the rank's per-dim shard offset. Caller must guarantee
    ``global_box`` ⊆ ``meta_ranges`` per dim (this is true if ``global_box``
    came from ``intersect(r_meta.ranges, meta_ranges)``).

    Returned tuples are ``(start, stop, None)`` to JSON-serialize cleanly.
    """
    encoded: list[tuple[int | None, int | None, int | None]] = []
    for (g_start, g_end), (m_start, _m_end) in zip(global_box, meta_ranges, strict=True):
        local_start = g_start - m_start
        local_end = g_end - m_start
        encoded.append((local_start, local_end, None))
    return tuple(encoded)


def _box_size(box: tuple[GlobalRange, ...]) -> int:
    n = 1
    for s, e in box:
        n *= e - s
    return n


def _bytes_for_box(box: tuple[GlobalRange, ...], dtype_str: str) -> int:
    from verl.checkpoint_engine.parallel_meta import _DTYPE_SIZE_BYTES  # local import to keep deps tight

    return _box_size(box) * _DTYPE_SIZE_BYTES[dtype_str]


def _group_by_param_name(metas: Iterable[ParameterShardMeta]) -> dict[str, list[ParameterShardMeta]]:
    """Return ``{param_name: sorted-by-global_rank metas}`` for determinism."""
    by_name: dict[str, list[ParameterShardMeta]] = defaultdict(list)
    for m in metas:
        by_name[m.param_name].append(m)
    return {name: sorted(by_name[name], key=lambda x: x.global_rank) for name in sorted(by_name)}


def build_transfer_plan(
    train_metas: Iterable[ParameterShardMeta],
    rollout_metas: Iterable[ParameterShardMeta],
    *,
    enrich_edge: callable | None = None,
) -> TransferPlan:
    """Compute the deterministic routing plan.

    Algorithm: for each param_name (sorted), for each rollout meta (sorted
    by global_rank), find all training metas that overlap its global box.
    For each unique overlap region, greedily pick the training rank that
    has sent the fewest bytes so far (tie-broken by smaller global_rank).

    DP-replica handling: if multiple training metas have the IDENTICAL
    ``ranges`` (DP replicas), they share the same overlap key and the
    greedy picker chooses one. Partial overlap across distinct shards is
    NOT supported in MVP (Megatron TP/EP shards are mutually exclusive;
    only the DP dim introduces full-replica duplicates, which the
    same-key grouping handles correctly).

    Args:
        train_metas: All training-side ParameterShardMetas across all ranks.
        rollout_metas: All rollout-side ParameterShardMetas.
        enrich_edge: Optional callback called per edge with the proto edge
            (``target_param_name``, ``shard_id``, ``expert_id`` all None);
            returns an enriched ``TransferEdge`` with vLLM-side metadata
            filled. If None, edges keep those fields as None.
    """
    train_by_name = _group_by_param_name(train_metas)
    rollout_by_name = _group_by_param_name(rollout_metas)

    # Validate role tags (cheap defensive check; would catch swapped inputs).
    for metas in train_by_name.values():
        for m in metas:
            if m.role != "train":
                raise ValueError(f"train_metas contains a non-train role: {m}")
    for metas in rollout_by_name.values():
        for m in metas:
            if m.role != "rollout":
                raise ValueError(f"rollout_metas contains a non-rollout role: {m}")

    send_bytes: dict[int, int] = defaultdict(int)  # per-train-rank running total
    edges: list[TransferEdge] = []

    for param_name in sorted(rollout_by_name):
        if param_name not in train_by_name:
            # Some HF params live only on trainer (e.g. lm_head when tied) or
            # only on rollout (rare). Skip with no error; sanity check is the
            # caller's job.
            continue
        train_list = train_by_name[param_name]
        rollout_list = rollout_by_name[param_name]

        # Validate shape/dtype agreement across all metas for this param.
        full_shape = train_list[0].full_shape
        dtype_str = train_list[0].dtype_str
        for m in train_list + rollout_list:
            if m.full_shape != full_shape:
                raise ValueError(f"full_shape mismatch for {param_name}: {m.full_shape} vs {full_shape}")
            if m.dtype_str != dtype_str:
                raise ValueError(f"dtype_str mismatch for {param_name}: {m.dtype_str} vs {dtype_str}")

        for r_meta in rollout_list:  # already sorted by global_rank
            r_box = r_meta.ranges
            # Collect all (overlap_range, t_meta) pairs.
            candidates_by_overlap: dict[tuple[GlobalRange, ...], list[ParameterShardMeta]] = defaultdict(list)
            for t_meta in train_list:
                ovr = intersect(r_box, t_meta.ranges)
                if ovr is None:
                    continue
                candidates_by_overlap[ovr].append(t_meta)

            for overlap in sorted(candidates_by_overlap):
                candidates = candidates_by_overlap[overlap]
                # Greedy: pick min send_bytes (tie-break: smaller global_rank).
                winner = min(candidates, key=lambda t: (send_bytes[t.global_rank], t.global_rank))
                src_slice = global_box_to_local_slice(overlap, winner.ranges)
                dst_slice = global_box_to_local_slice(overlap, r_meta.ranges)
                shape = tuple(end - start for start, end in overlap)

                proto_edge = TransferEdge(
                    param_name=param_name,
                    src_global_rank=winner.global_rank,
                    dst_global_rank=r_meta.global_rank,
                    src_local_slice_encoded=src_slice,
                    dst_local_slice_encoded=dst_slice,
                    shape=shape,
                    dtype_str=dtype_str,
                    target_param_name=None,
                    shard_id=None,
                    expert_id=None,
                )
                final_edge = enrich_edge(proto_edge) if enrich_edge is not None else proto_edge
                edges.append(final_edge)
                send_bytes[winner.global_rank] += _bytes_for_box(overlap, dtype_str)

    # Sort edges deterministically: by (param_name, src_rank, dst_rank, src_slice).
    # Keeps plan_hash stable irrespective of dict-iteration nuances on older
    # Python versions and makes diffs human-readable.
    edges.sort(
        key=lambda e: (
            e.param_name,
            e.src_global_rank,
            e.dst_global_rank,
            e.src_local_slice_encoded,
        )
    )
    return TransferPlan(edges=tuple(edges))


def sanity_check_cross_rank(plan_hash: str, all_hashes: Iterable[str]) -> None:
    """Verify every rank arrived at the same plan_hash. Raises on mismatch.

    Use as cheap insurance against accidental nondeterminism::

        local_hash = plan.plan_hash
        all_hashes = ray.get([w.get_plan_hash.remote() for w in workers])
        sanity_check_cross_rank(local_hash, all_hashes)
    """
    distinct = set(all_hashes)
    distinct.add(plan_hash)
    if len(distinct) != 1:
        raise RuntimeError(f"Cross-rank TransferPlan disagreement: {len(distinct)} distinct hashes: {distinct}")


# Re-export so callers don't have to import both modules.
__all__ = [
    "TransferPlan",
    "TransferEdge",
    "ParameterShardMeta",
    "build_transfer_plan",
    "intersect",
    "global_box_to_local_slice",
    "sanity_check_cross_rank",
]


# Silence the unused-import warning while keeping types accessible
# (we re-export above).
_ = (TransferPlan, TransferEdge, ParameterShardMeta, Any)
