# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Hash-based prefix detection for the unified prefix-tree pipeline.

Counterpart to ``prefix_tree_dynamic.build_tree_dynamic`` — both produce the
same ``(TreeNode, leaf_to_sample)`` contract consumed by
:func:`verl.utils.prefix_tree_utils.build_layout_from_tree_node`.

Two-stage detection:
  1. Root prefix length — from per-turn hashes (``prefix_segments_batch``)
     when available, else from a token-level LCP scan.
  2. Multi-level (depth-3) — when ``prefix_segments_batch`` is provided AND
     the batch has ≥2 groups of ≥2 samples sharing a second-turn hash,
     produce a 3-level tree; otherwise fall back to single-level (root +
     per-sample leaves).

Supports depth-2 and depth-3 only. For arbitrary-depth trees, use
``build_tree_dynamic``.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

from torch import Tensor

from verl.utils.prefix_tree_utils import TreeNode, longest_common_prefix_length


def build_tree_hash_based(
    samples: list[Tensor],
    prefix_segments_batch: Optional[list[list[tuple[int, int]]]] = None,
) -> Optional[tuple[TreeNode, list[int]]]:
    """Hash-based prefix detection. Returns ``(TreeNode, leaf_to_sample)`` or None.

    ``leaf_to_sample[i]`` gives the original sample index for the i-th leaf in
    DFS pre-order. Returns ``None`` when no shared prefix exists.
    """
    import os as _os

    n = len(samples)
    if n == 0:
        return None

    if prefix_segments_batch is not None and len(prefix_segments_batch) == n:
        prefix_len = _resolve_prefix_len_from_segments(prefix_segments_batch)
        if _os.environ.get("DEBUG_PREFIX_LEN") == "1":
            scan_len = longest_common_prefix_length(samples)
            T = samples[0].shape[0] if samples else 0
            print(
                f"[PREFIX_LEN] seg_len={prefix_len} scan_len={scan_len} T={T} "
                f"n_segs={[len(s) for s in prefix_segments_batch]}",
                flush=True,
            )
    else:
        prefix_len = longest_common_prefix_length(samples)

    if prefix_len == 0:
        return None

    if prefix_segments_batch is not None:
        actual_root_len = longest_common_prefix_length(samples)
        if actual_root_len > 0:
            multilevel = _resolve_multilevel_tree(samples, prefix_segments_batch, actual_root_len)
            if multilevel is not None:
                root_len, children_info = multilevel  # children_info: [(idxs, group_TreeNode), ...]
                root = TreeNode(segment_len=root_len, children=[g for _, g in children_info])
                leaf_to_sample = [int(idx) for idxs, _ in children_info for idx in idxs]
                return root, leaf_to_sample

    # Single-level fallback: root + per-sample leaves (one leaf per sample).
    leaves = [TreeNode(segment_len=int(t.shape[0]) - prefix_len) for t in samples]
    root = TreeNode(segment_len=prefix_len, children=leaves)
    return root, list(range(n))


def _resolve_prefix_len_from_segments(
    prefix_segments_batch: list[list[tuple[int, int]]],
) -> int:
    """Return the longest shared-prefix length derivable from per-sample segment lists.

    Each element of *prefix_segments_batch* is a list of ``(hash, cumulative_len)``
    pairs produced by the dataset at load time. Two samples share turn k when all
    samples have the same per-turn hash at position k.

    Algorithm: walk turn-by-turn; stop at the first turn where hashes diverge.
    Return the cumulative_len from sample 0 at the last shared turn (0 if none).

    Compares only hashes (not cum_len) because tokenization boundary effects can
    shift cum_len slightly between samples even for identical turns.
    """
    n = len(prefix_segments_batch)
    if n == 0:
        return 0

    min_turns = min(len(segs) for segs in prefix_segments_batch)
    if min_turns == 0:
        return 0

    best = 0
    for turn_idx in range(min_turns):
        h0 = prefix_segments_batch[0][turn_idx][0]
        if all(prefix_segments_batch[i][turn_idx][0] == h0 for i in range(1, n)):
            best = prefix_segments_batch[0][turn_idx][1]
        else:
            break
    return best


def _resolve_multilevel_tree(
    tokens_by_sample: list,
    prefix_segments_batch: list[list[tuple[int, int]]],
    root_prefix_len: int,
) -> Optional[tuple[int, list[tuple[list[int], TreeNode]]]]:
    """Detect a 2-level tree from prefix_segments. Returns ``(root_len, children_info)`` or None.

    Groups samples by their first post-root segment hash. If ≥2 groups each have
    ≥2 samples, a 2-level tree exists and we return ``(root_len, children_info)``
    where ``children_info = [(sample_idxs, group_TreeNode), ...]``.
    """
    n = len(tokens_by_sample)
    if prefix_segments_batch is None or n < 4:
        return None

    # Group samples by the hash of their first post-root turn (O(batch×turns)).
    groups: dict[int, list[int]] = defaultdict(list)
    for i, segs in enumerate(prefix_segments_batch):
        next_seg = next((s for s in segs if s[1] > root_prefix_len), None)
        if next_seg is None:
            return None
        groups[next_seg[0]].append(i)

    # Need ≥2 groups each with ≥2 samples for multi-level to be worthwhile.
    useful = [(h, idxs) for h, idxs in groups.items() if len(idxs) >= 2]
    if len(useful) < 2:
        return None

    # Use token scan (not segment hashes) to get exact turn2 shared prefix length per group;
    # segment hashes can mis-align due to chat-template boundary effects.
    children = []
    for _h, idxs in useful:
        group_tokens = [tokens_by_sample[i] for i in idxs]
        suffixes = [t[root_prefix_len:] for t in group_tokens]
        shared_suffix_len = longest_common_prefix_length(suffixes)
        if shared_suffix_len <= 0:
            return None
        group_turn2_len = shared_suffix_len
        leaves = [TreeNode(int(tokens_by_sample[i].shape[0]) - root_prefix_len - group_turn2_len, []) for i in idxs]
        if any(leaf.segment_len <= 0 for leaf in leaves):
            return None
        children.append((idxs, TreeNode(group_turn2_len, leaves)))

    return (root_prefix_len, children)
