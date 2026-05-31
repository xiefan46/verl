# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Benchmark-only thin wrapper around hash-based's build_prefix_tree_micro_batch.
# Reimplements the data-path with the same helpers from verl/utils/prefix_tree_*
# but instruments 3-layer timing and skips GPU/MAGI key construction.

"""Hash-based static-detection wrapper matching the same API contract."""

from __future__ import annotations

import importlib.util
import os
import sys
import time
from typing import Optional

import torch
from torch import Tensor


def _load_verl_module(rel_path: str, mod_name: str):
    here = os.path.dirname(os.path.abspath(__file__))
    full = os.path.normpath(os.path.join(here, "../../..", rel_path))
    spec = importlib.util.spec_from_file_location(mod_name, full)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


# Ensure ordering: params → utils → magi (magi needs both; we skip magi here since it needs torch.nested internals)
_ptp_mod = _load_verl_module("verl/utils/prefix_tree_params.py", "verl.utils.prefix_tree_params")
_ptu_mod = _load_verl_module("verl/utils/prefix_tree_utils.py", "verl.utils.prefix_tree_utils")

PrefixTreeParams = _ptp_mod.PrefixTreeParams
TreeNode = _ptu_mod.TreeNode
build_prefix_tree_params = _ptu_mod.build_prefix_tree_params
build_multilevel_flex_spec = _ptu_mod.build_multilevel_flex_spec
longest_common_prefix_length = _ptu_mod.longest_common_prefix_length


# Reuse our dynamic-trie wrapper's PrefixTreeMagiBatch since both wrappers should produce same type
from dynamic_trie_wrapper import PrefixTreeMagiBatch  # noqa: E402

# ============================================================================
# Helpers cloned/adapted from prefix_tree_magi.py (so we don't need to import magi)
# ============================================================================


def _resolve_prefix_len_from_segments(prefix_segments_batch: list[list[tuple[int, int]]]) -> int:
    """Find longest shared prefix via per-turn hash match (O(batch × turns))."""
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
):
    """Detect depth-3 tree from prefix_segments. Returns (root_len, [(sample_idxs, TreeNode)]) or None."""
    from collections import defaultdict

    n = len(tokens_by_sample)
    if prefix_segments_batch is None or n < 4:
        return None

    groups: dict[int, list[int]] = defaultdict(list)
    for i, segs in enumerate(prefix_segments_batch):
        next_seg = next((s for s in segs if s[1] > root_prefix_len), None)
        if next_seg is None:
            return None
        groups[next_seg[0]].append(i)

    useful = [(h, idxs) for h, idxs in groups.items() if len(idxs) >= 2]
    if len(useful) < 2:
        return None

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


# ============================================================================
# Multi-level params builder (port from prefix_tree_magi._build_multilevel_prefix_tree_params)
# Only needs torch — skip the magi-side dependencies
# ============================================================================


def _build_multilevel_prefix_tree_params(
    tokens_by_sample: list[Tensor],
    root_len: int,
    children_info: list,  # [(sample_idxs, child_TreeNode), ...]
    loss_masks_by_sample: Optional[list[Tensor]] = None,
    position_ids_by_sample: Optional[list[Tensor]] = None,
) -> PrefixTreeParams:
    """Port of hash-based's depth-3 multi-level params builder.

    Tree structure: virtual_root → root_segment (shared by all, root_len tokens) →
    children_info: list of (group_sample_idxs, group_TreeNode). Each group_TreeNode has
    segment_len = turn2_shared_len, children = [leaf TreeNode per sample in group].

    Produces: flat_tokens = [root_tokens, group1_mid, group1_leaf0, group1_leaf1, ...,
                            group2_mid, group2_leaf0, ...].
    """
    # Build the TreeNode root for build_multilevel_flex_spec
    # Root segment_len = root_len, children = list of group nodes
    group_nodes: list[TreeNode] = []
    for _idxs, child_node in children_info:
        group_nodes.append(child_node)
    root_tn = TreeNode(segment_len=root_len, children=group_nodes)

    q_ranges, k_ranges, mask_types = build_multilevel_flex_spec(root_tn)

    # Now emit flat tensors via DFS pre-order
    # Layout: root_tokens (from sample 0) | for each group: group_mid_tokens (from group's sample 0) |
    #         for each leaf in group: leaf_tokens (from that leaf's sample)
    device = tokens_by_sample[0].device
    flat_pieces: list[Tensor] = []
    flat_lm_pieces: Optional[list[Tensor]] = [] if loss_masks_by_sample is not None else None
    flat_pid_pieces: Optional[list[Tensor]] = [] if position_ids_by_sample is not None else None
    default_pid_pieces: list[Tensor] = []
    leaf_ranges: list[tuple[int, int]] = []
    leaf_to_sample: list[int] = []

    # Root segment from sample 0 [0:root_len]
    flat_pieces.append(tokens_by_sample[0][:root_len])
    if flat_lm_pieces is not None:
        flat_lm_pieces.append(loss_masks_by_sample[0][:root_len])
    if flat_pid_pieces is not None:
        flat_pid_pieces.append(position_ids_by_sample[0][:root_len])
    else:
        default_pid_pieces.append(torch.arange(0, root_len, device=device, dtype=torch.long))

    cursor = root_len

    for grp_idx, (idxs, child_node) in enumerate(children_info):
        # Group mid segment from idxs[0]
        owner = idxs[0]
        mid_len = child_node.segment_len
        mid_start = root_len
        mid_end = root_len + mid_len
        flat_pieces.append(tokens_by_sample[owner][mid_start:mid_end])
        if flat_lm_pieces is not None:
            flat_lm_pieces.append(loss_masks_by_sample[owner][mid_start:mid_end])
        if flat_pid_pieces is not None:
            flat_pid_pieces.append(position_ids_by_sample[owner][mid_start:mid_end])
        else:
            default_pid_pieces.append(torch.arange(mid_start, mid_end, device=device, dtype=torch.long))
        cursor += mid_len

        # Each leaf
        for leaf_idx, sample_idx in enumerate(idxs):
            leaf_node = child_node.children[leaf_idx]
            leaf_len = leaf_node.segment_len
            leaf_token_start = mid_end  # in owner's sample, leaf starts after root+mid
            leaf_token_end = leaf_token_start + leaf_len
            flat_pieces.append(tokens_by_sample[sample_idx][leaf_token_start:leaf_token_end])
            if flat_lm_pieces is not None:
                flat_lm_pieces.append(loss_masks_by_sample[sample_idx][leaf_token_start:leaf_token_end])
            if flat_pid_pieces is not None:
                flat_pid_pieces.append(position_ids_by_sample[sample_idx][leaf_token_start:leaf_token_end])
            else:
                default_pid_pieces.append(
                    torch.arange(leaf_token_start, leaf_token_end, device=device, dtype=torch.long)
                )

            leaf_ranges.append((cursor, cursor + leaf_len))
            leaf_to_sample.append(sample_idx)
            cursor += leaf_len

    flat_tokens = torch.cat(flat_pieces)
    flat_loss_mask = torch.cat(flat_lm_pieces) if flat_lm_pieces is not None else None
    flat_position_ids = torch.cat(flat_pid_pieces) if flat_pid_pieces is not None else torch.cat(default_pid_pieces)

    prefix_range = (0, root_len)
    sample_to_leaf_range = {s: r for s, r in zip(leaf_to_sample, leaf_ranges, strict=False)}

    return PrefixTreeParams(
        prefix_range=prefix_range,
        prefix_segments=[prefix_range],
        leaf_ranges=leaf_ranges,
        leaf_segments=list(leaf_ranges),
        leaf_to_sample=list(leaf_to_sample),
        sample_to_leaf_range=sample_to_leaf_range,
        q_ranges=q_ranges,
        k_ranges=k_ranges,
        mask_types=mask_types,
        total_seqlen_q=flat_tokens.numel(),
        total_seqlen_k=flat_tokens.numel(),
        flat_tokens=flat_tokens,
        flat_labels=None,
        flat_loss_mask=flat_loss_mask,
        flat_position_ids=flat_position_ids,
        multilevel=True,
    )


# ============================================================================
# Helpers
# ============================================================================


def _unpack(x):
    if x is None:
        return None
    if hasattr(x, "offsets"):
        offsets = x.offsets()
        lengths = offsets.diff().tolist()
        flat_vals = x.values()
        out = []
        pos = 0
        for length in lengths:
            out.append(flat_vals[pos : pos + int(length)])
            pos += int(length)
        return out
    return list(x)


# ============================================================================
# Public API: build_prefix_tree_micro_batch_hash_based
# ============================================================================


def build_prefix_tree_micro_batch_hash_based(
    model,
    input_ids,
    loss_mask=None,
    position_ids=None,
    prefix_segments_batch: Optional[list[list[tuple[int, int]]]] = None,
    attention_type: str = "flex",
    tp_size: int = 1,
    cp_size: int = 1,
) -> Optional[PrefixTreeMagiBatch]:
    """Hash-based static wrapper matching dynamic-trie wrapper's API.

    Mirrors verl/utils/prefix_tree_magi.py:build_prefix_tree_micro_batch.
    Returns None when no shared prefix found.
    """
    tokens_by_sample = _unpack(input_ids)
    if not tokens_by_sample:
        return None
    loss_masks_by_sample = _unpack(loss_mask)
    position_ids_by_sample = _unpack(position_ids)

    # Fast path: hash-based detection if prefix_segments_batch provided
    if prefix_segments_batch is not None and len(prefix_segments_batch) == len(tokens_by_sample):
        prefix_len = _resolve_prefix_len_from_segments(prefix_segments_batch)
    else:
        prefix_len = longest_common_prefix_length(tokens_by_sample)

    if prefix_len == 0:
        return None

    multilevel_result = None
    if prefix_segments_batch is not None:
        actual_root_len = longest_common_prefix_length(tokens_by_sample)
        if actual_root_len > 0:
            multilevel_result = _resolve_multilevel_tree(tokens_by_sample, prefix_segments_batch, actual_root_len)

    if multilevel_result is not None:
        root_len, children_info = multilevel_result
        params = _build_multilevel_prefix_tree_params(
            tokens_by_sample,
            root_len,
            children_info,
            loss_masks_by_sample=loss_masks_by_sample,
            position_ids_by_sample=position_ids_by_sample,
        )
    else:
        params = build_prefix_tree_params(
            tokens_by_sample,
            prefix_len=prefix_len,
            loss_masks_by_sample=loss_masks_by_sample,
            position_ids_by_sample=position_ids_by_sample,
        )

    # TP/CP padding
    real_tokens = params.flat_tokens.shape[0]
    if tp_size > 1:
        align_size = (tp_size * cp_size * 2) if cp_size > 1 else tp_size
        pad_len = (align_size - real_tokens % align_size) % align_size
        if pad_len > 0:
            params.flat_tokens = torch.cat([params.flat_tokens, params.flat_tokens.new_zeros(pad_len)])
            params.flat_position_ids = torch.cat(
                [params.flat_position_ids, params.flat_position_ids.new_zeros(pad_len)]
            )
            if params.flat_loss_mask is not None:
                params.flat_loss_mask = torch.cat([params.flat_loss_mask, params.flat_loss_mask.new_zeros(pad_len)])
            params.total_seqlen_q += pad_len
            params.total_seqlen_k += pad_len

    return PrefixTreeMagiBatch(
        flat_input_ids=params.flat_tokens,
        flat_position_ids=params.flat_position_ids,
        flat_loss_mask=params.flat_loss_mask,
        magi_key=None,
        flex_key=None,
        leaf_to_sample=params.leaf_to_sample,
        leaf_ranges=params.leaf_ranges,
        prefix_range=params.prefix_range,
        original_batch_size=len(tokens_by_sample),
        real_tokens=real_tokens,
        leaf_ancestor_ranges=None,
    )


# ============================================================================
# Variant with full output + 3-layer timing
# ============================================================================


def build_prefix_tree_micro_batch_hash_based_full(
    model,
    input_ids,
    loss_mask=None,
    position_ids=None,
    prefix_segments_batch: Optional[list[list[tuple[int, int]]]] = None,
    attention_type: str = "flex",
    tp_size: int = 1,
    cp_size: int = 1,
) -> tuple[Optional[PrefixTreeMagiBatch], Optional[PrefixTreeParams], dict]:
    """Variant returning (batch, params, timings) for sanity check + benchmarking.

    Timings dict keys (mirrors dynamic-trie wrapper):
      - unpack_ms
      - tree_detect_ms (hash-based detection: _resolve_prefix_len + _resolve_multilevel_tree)
      - pack_ms (build_..._prefix_tree_params + flat tensor packing)
      - total_ms
    """
    timings: dict[str, float] = {}
    t_total = time.perf_counter()

    t0 = time.perf_counter()
    tokens_by_sample = _unpack(input_ids)
    loss_masks_by_sample = _unpack(loss_mask)
    position_ids_by_sample = _unpack(position_ids)
    timings["unpack_ms"] = (time.perf_counter() - t0) * 1000

    if not tokens_by_sample:
        timings["total_ms"] = (time.perf_counter() - t_total) * 1000
        return None, None, timings

    # Tree detect
    t0 = time.perf_counter()
    if prefix_segments_batch is not None and len(prefix_segments_batch) == len(tokens_by_sample):
        prefix_len = _resolve_prefix_len_from_segments(prefix_segments_batch)
    else:
        prefix_len = longest_common_prefix_length(tokens_by_sample)

    if prefix_len == 0:
        timings["tree_detect_ms"] = (time.perf_counter() - t0) * 1000
        timings["total_ms"] = (time.perf_counter() - t_total) * 1000
        return None, None, timings

    multilevel_result = None
    if prefix_segments_batch is not None:
        actual_root_len = longest_common_prefix_length(tokens_by_sample)
        if actual_root_len > 0:
            multilevel_result = _resolve_multilevel_tree(tokens_by_sample, prefix_segments_batch, actual_root_len)
    timings["tree_detect_ms"] = (time.perf_counter() - t0) * 1000

    # Pack
    t0 = time.perf_counter()
    if multilevel_result is not None:
        root_len, children_info = multilevel_result
        params = _build_multilevel_prefix_tree_params(
            tokens_by_sample,
            root_len,
            children_info,
            loss_masks_by_sample=loss_masks_by_sample,
            position_ids_by_sample=position_ids_by_sample,
        )
    else:
        params = build_prefix_tree_params(
            tokens_by_sample,
            prefix_len=prefix_len,
            loss_masks_by_sample=loss_masks_by_sample,
            position_ids_by_sample=position_ids_by_sample,
        )
    timings["pack_ms"] = (time.perf_counter() - t0) * 1000

    real_tokens = params.flat_tokens.shape[0]
    pt_batch = PrefixTreeMagiBatch(
        flat_input_ids=params.flat_tokens,
        flat_position_ids=params.flat_position_ids,
        flat_loss_mask=params.flat_loss_mask,
        magi_key=None,
        flex_key=None,
        leaf_to_sample=params.leaf_to_sample,
        leaf_ranges=params.leaf_ranges,
        prefix_range=params.prefix_range,
        original_batch_size=len(tokens_by_sample),
        real_tokens=real_tokens,
        leaf_ancestor_ranges=None,
    )
    timings["total_ms"] = (time.perf_counter() - t_total) * 1000
    return pt_batch, params, timings
