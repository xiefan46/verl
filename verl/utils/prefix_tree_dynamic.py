# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Copyright 2025-2026 The AReaL Authors (Ant Group, Tsinghua University, HKUST)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Dynamic-trie prefix-tree builder for FSDP2 + Magi model integration.

Token-by-token trie insertion that supports **arbitrary tree depth** —
detects the shared-prefix tree directly from the input tokens, no
rollout-side metadata required. Invoked by the unified
:func:`verl.utils.prefix_tree_magi.build_prefix_tree_micro_batch` entry
point when ``dynamic_trie=True``.

REQUIREMENTS (enforced by ``FSDPEngine._build_module`` assertion):
  * ``engine.strategy='fsdp2'`` — FSDP1 produces EVAL/TRAIN forward divergence
    when combined with Magi attention (ppo_kl ≈ 3-4 + intermittent NaN grad).
    MagiAttention's torch_native reference example uses FSDP2; that is the
    only validated combination.
  * ``prefix_tree_attention='magi'`` — the flex_attention path was retired
    due to the AReaL 8× entropy bug.
  * ``ulysses_sequence_parallel_size=1`` — Magi CP and Ulysses SP are
    mutually exclusive seq-parallel schemes.

Trade-offs vs the hash-based static path (from benchmark in
``tests/benchmark/tree_training/``):
  - 10–300× slower at trie *detection* (Python token-by-token insert).
  - For typical RL workloads (<1M total tokens), absolute overhead is
    <250 ms (~2–3% of a 10 s step).
  - Arbitrary depth — required for MCTS-based RL (rStar-Math, DeepSearch)
    where the hash path's depth-2 cap loses key sharing structure.

Algorithm originally derived from AReaL
(https://github.com/inclusionAI/AReaL).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import torch
from torch import Tensor
from torch.nested._internal.nested_tensor import NestedTensor

from verl.utils.prefix_tree_magi import PrefixTreeMagiBatch, _build_magi_key
from verl.utils.prefix_tree_params import PrefixTreeParams
from verl.utils.prefix_tree_utils import TreeNode, build_multilevel_flex_spec

__all__ = [
    "build_prefix_tree_micro_batch_dynamic",
    "prefix_tree_dynamic_forward",
    # Lower-level helpers exposed for testing / benchmarking
    "TrieNode",
    "greedy_build_tries",
    "convert_trie_to_tree_node",
    "build_arbitrary_depth_params",
    "unpack_nested_to_list",
]


# ============================================================================
# Trie construction (token-by-token insertion)
# ============================================================================


@dataclass
class TrieNode:
    """Compressed-trie node (after `_compress` pass).

    Each non-root node represents a contiguous run of tokens shared by the same
    set of sequences. Root has ``start_idx == end_idx == -1`` and stores no
    tokens — children are accessed via ``.children: dict[first_token, TrieNode]``.
    """

    tree_id: int
    start_idx: int = -1
    end_idx: int = -1
    tokens: list[int] = field(default_factory=list)
    sequence_ids: list[int] = field(default_factory=list)
    children: dict[int, TrieNode] = field(default_factory=dict)
    ancestors: list[TrieNode] = field(default_factory=list)
    nodes: list[TrieNode] = field(default_factory=list)

    @property
    def is_root(self) -> bool:
        return self.start_idx == -1 and self.end_idx == -1


class _BuildNode:
    """Internal — temporary uncompressed node used during insertion."""

    __slots__ = ("tree_id", "token_id", "node_id", "children", "is_end", "sequence_ids")

    def __init__(self, tree_id: int, token_id: int, node_id: int):
        self.tree_id = tree_id
        self.token_id = token_id
        self.node_id = node_id
        self.children: dict[int, _BuildNode] = {}
        self.is_end = False
        self.sequence_ids: list[int] = []


def _count_additional_nodes(root: _BuildNode, sequence: list[int]) -> int:
    current = root
    for idx, token in enumerate(sequence):
        child = current.children.get(token)
        if child is None:
            return len(sequence) - idx
        current = child
    return 0


def _insert_sequence(
    root: _BuildNode,
    all_nodes: list[_BuildNode],
    sequence: list[int],
    tree_id: int,
    sequence_id: int,
) -> None:
    current = root
    for token in sequence:
        if token not in current.children:
            node_id = len(all_nodes)
            current.children[token] = _BuildNode(tree_id, token, node_id)
            all_nodes.append(current.children[token])
        current.children[token].sequence_ids.append(sequence_id)
        current = current.children[token]
    current.is_end = True


def _compress_trie(root: _BuildNode) -> TrieNode:
    trie_root = TrieNode(tree_id=root.tree_id)

    def _compress_chain(node: _BuildNode, ancestors: list[TrieNode]) -> TrieNode:
        tokens: list[int] = []
        current = node
        start_id = node.node_id
        while True:
            tokens.append(current.token_id)
            if len(current.children) != 1 or current.is_end:
                break
            next_child = next(iter(current.children.values()))
            if current.sequence_ids != next_child.sequence_ids:
                raise ValueError("Sequence IDs mismatch along chain")
            if next_child.node_id != current.node_id + 1:
                raise ValueError("Node IDs not consecutive along chain")
            current = next_child

        trie_node = TrieNode(
            tree_id=root.tree_id,
            start_idx=start_id,
            end_idx=current.node_id,
            tokens=tokens,
            sequence_ids=current.sequence_ids.copy(),
            ancestors=ancestors.copy(),
        )
        trie_root.nodes.append(trie_node)
        if current.children:
            for token, child in sorted(current.children.items()):
                trie_node.children[token] = _compress_chain(child, ancestors + [trie_node])
        return trie_node

    if root.children:
        for token, child in sorted(root.children.items()):
            trie_root.children[token] = _compress_chain(child, [])
    return trie_root


def greedy_build_tries(
    sequences: list[list[int]],
    max_tokens_per_tree: int,
) -> tuple[list[TrieNode], list[int]]:
    """Token-by-token greedy trie packing across samples.

    Args:
        sequences: per-sample token lists.
        max_tokens_per_tree: upper bound on uncompressed nodes per tree (set to
            a huge value when you want a single forest).

    Returns:
        (tries, num_tokens_list) — list of compressed TrieNode roots + total
        uncompressed nodes per tree.
    """
    forests: list[dict[str, Any]] = []
    for seq_id, seq in enumerate(sequences):
        inserted = False
        for tree_id, tree in enumerate(forests):
            additional = _count_additional_nodes(tree["root"], seq)
            if tree["nodes"] + additional <= max_tokens_per_tree:
                _insert_sequence(tree["root"], tree["all_nodes"], seq, tree_id, seq_id)
                tree["nodes"] += additional
                inserted = True
                break
        if inserted:
            continue
        if len(seq) > max_tokens_per_tree:
            raise ValueError(f"Sequence length {len(seq)} exceeds max_tokens_per_tree {max_tokens_per_tree}")
        new_tree_id = len(forests)
        new_root = _BuildNode(new_tree_id, -1, -1)
        all_nodes: list[_BuildNode] = []
        _insert_sequence(new_root, all_nodes, seq, new_tree_id, seq_id)
        forests.append({"root": new_root, "all_nodes": all_nodes, "nodes": len(seq)})

    tries = [_compress_trie(f["root"]) for f in forests]
    num_tokens_list = [f["nodes"] for f in forests]
    return tries, num_tokens_list


# ============================================================================
# Trie → TreeNode conversion (arbitrary depth preserved)
# ============================================================================


def convert_trie_to_tree_node(
    trie: TrieNode,
) -> Optional[tuple[TreeNode, dict[int, tuple[int, int, int]], list[TreeNode]]]:
    """Convert a compressed trie to a ``TreeNode`` consumed by
    ``build_multilevel_flex_spec``.

    The trie root is a virtual placeholder with no tokens. We promote the
    trie's only child as the TreeNode root so the downstream flex-spec
    builder sees a non-zero root segment.

    Returns ``None`` when there's no real sharing (single sample, no children,
    or multi-forest case).

    Returns ``(root, node_info, leaves_in_dfs)`` where:
      - ``root``: ``TreeNode`` root for downstream packing
      - ``node_info[id(node)] = (owner_sample_idx, range_start, range_end)``:
        for each non-root node, an owning sample + token range in that sample's
        original sequence. Needed for ``build_arbitrary_depth_params`` to emit
        flat tokens from the correct sample.
      - ``leaves_in_dfs``: leaf TreeNodes in DFS pre-order, matching the order
        ``build_multilevel_flex_spec`` walks the tree.
    """
    if not trie.children:
        return None
    if len(trie.children) > 1:
        # Multi-forest — no single shared root prefix
        return None

    node_info: dict[int, tuple[int, int, int]] = {}
    leaves_in_dfs: list[TreeNode] = []

    def _convert(trie_node: TrieNode, offset_in_owner: int) -> TreeNode:
        segment_len = len(trie_node.tokens)
        end_in_owner = offset_in_owner + segment_len

        children: list[TreeNode] = []
        for _tok, child in sorted(trie_node.children.items()):
            children.append(_convert(child, end_in_owner))

        node = TreeNode(segment_len=segment_len, children=children)

        if not children:
            assert len(trie_node.sequence_ids) == 1, (
                f"Trie leaf should belong to exactly 1 sample, got {trie_node.sequence_ids}"
            )
            owner = trie_node.sequence_ids[0]
            node_info[id(node)] = (owner, offset_in_owner, end_in_owner)
            leaves_in_dfs.append(node)
        else:
            first_child_owner = node_info[id(children[0])][0]
            node_info[id(node)] = (first_child_owner, offset_in_owner, end_in_owner)
        return node

    only_child = next(iter(trie.children.values()))
    root = _convert(only_child, 0)
    if not root.children:
        return None
    return root, node_info, leaves_in_dfs


# ============================================================================
# Arbitrary-depth params builder (generalises the hash-path depth-2 limit)
# ============================================================================


def build_arbitrary_depth_params(
    tokens_by_sample: list[Tensor],
    tree_root: TreeNode,
    node_info: dict[int, tuple[int, int, int]],
    leaves_in_dfs: list[TreeNode],
    loss_masks_by_sample: Optional[list[Tensor]] = None,
    position_ids_by_sample: Optional[list[Tensor]] = None,
) -> PrefixTreeParams:
    """Build PrefixTreeParams for arbitrary-depth tree.

    Uses ``build_multilevel_flex_spec`` for q/k_ranges (already supports any
    depth) and walks DFS pre-order to emit flat tokens from each node's owning
    sample.

    Side effect: sets ``params._leaf_ancestor_ranges`` so ``restore_flat_to_nested``
    can reconstruct each sample by concatenating its ancestor segments + leaf.
    """
    q_ranges, k_ranges, mask_types = build_multilevel_flex_spec(tree_root)

    device = tokens_by_sample[0].device
    flat_pieces: list[Tensor] = []
    flat_lm_pieces: Optional[list[Tensor]] = [] if loss_masks_by_sample is not None else None
    flat_pid_pieces: Optional[list[Tensor]] = [] if position_ids_by_sample is not None else None
    default_pid_pieces: list[Tensor] = []  # used when position_ids_by_sample is None

    # Track per-leaf ancestor chain (root → ... → parent) for restore.
    parent_of: dict[int, TreeNode] = {}  # id(child) → parent_node

    def _emit(node: TreeNode):
        if node.segment_len > 0:
            owner_idx, range_s, range_e = node_info[id(node)]
            flat_pieces.append(tokens_by_sample[owner_idx][range_s:range_e])
            if flat_lm_pieces is not None:
                flat_lm_pieces.append(loss_masks_by_sample[owner_idx][range_s:range_e])
            if flat_pid_pieces is not None:
                flat_pid_pieces.append(position_ids_by_sample[owner_idx][range_s:range_e])
            else:
                default_pid_pieces.append(torch.arange(range_s, range_e, device=device, dtype=torch.long))
        for child in node.children:
            parent_of[id(child)] = node
            _emit(child)

    _emit(tree_root)

    flat_tokens = (
        torch.cat(flat_pieces) if flat_pieces else torch.empty(0, dtype=tokens_by_sample[0].dtype, device=device)
    )
    flat_loss_mask = torch.cat(flat_lm_pieces) if flat_lm_pieces is not None else None
    if flat_pid_pieces is not None:
        flat_position_ids = torch.cat(flat_pid_pieces)
    else:
        flat_position_ids = (
            torch.cat(default_pid_pieces) if default_pid_pieces else torch.empty(0, dtype=torch.long, device=device)
        )

    # leaf_ranges from side-effect assignment by build_multilevel_flex_spec
    leaf_ranges = [(leaf._flat_start, leaf._flat_end) for leaf in leaves_in_dfs]  # type: ignore[attr-defined]
    leaf_to_sample = [node_info[id(leaf)][0] for leaf in leaves_in_dfs]

    prefix_range = (tree_root._flat_start, tree_root._flat_end)  # type: ignore[attr-defined]
    sample_to_leaf_range = {s: r for s, r in zip(leaf_to_sample, leaf_ranges, strict=False)}

    # Per-leaf ancestor flat ranges (root → parent), needed by restore_flat_to_nested
    leaf_ancestor_ranges: list[list[tuple[int, int]]] = []
    for leaf in leaves_in_dfs:
        chain: list[tuple[int, int]] = []
        cur = parent_of.get(id(leaf))
        while cur is not None:
            chain.append((cur._flat_start, cur._flat_end))  # type: ignore[attr-defined]
            cur = parent_of.get(id(cur))
        chain.reverse()  # root first
        leaf_ancestor_ranges.append(chain)

    params = PrefixTreeParams(
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
    params._leaf_ancestor_ranges = leaf_ancestor_ranges  # type: ignore[attr-defined]
    return params


# ============================================================================
# NestedTensor / list-of-Tensors unpacking
# ============================================================================


def unpack_nested_to_list(x) -> Optional[list[Tensor]]:
    """Unpack NestedTensor (or pass-through list of Tensors) → list of 1-D tensors.

    Accepts the same input shapes as ``prefix_tree_magi.build_prefix_tree_micro_batch``.
    Returns ``None`` when ``x is None``.
    """
    if x is None:
        return None
    if isinstance(x, NestedTensor) or hasattr(x, "offsets"):
        offsets = x.offsets()
        lengths = offsets.diff().tolist()
        flat_vals = x.values()
        out: list[Tensor] = []
        pos = 0
        for length in lengths:
            out.append(flat_vals[pos : pos + int(length)])
            pos += int(length)
        return out
    return list(x)


# ============================================================================
# Public entry: build_prefix_tree_micro_batch_dynamic
# ============================================================================


def build_prefix_tree_micro_batch_dynamic(
    model,
    input_ids,
    loss_mask=None,
    position_ids=None,
    prefix_segments_batch: Optional[list[list[tuple[int, int]]]] = None,
    attention_type: str = "magi",
    tp_size: int = 1,
    cp_size: int = 1,
    cp_group=None,
) -> Optional[PrefixTreeMagiBatch]:
    """Dynamic-trie implementation of ``build_prefix_tree_micro_batch``.

    Invoked by :func:`verl.utils.prefix_tree_magi.build_prefix_tree_micro_batch`
    when ``dynamic_trie=True``. Detects the shared-prefix tree by token-by-token
    trie insertion, supporting arbitrary depth.

    Args / returns: same contract as the hash-based path. ``prefix_segments_batch``
    is accepted for signature parity but **ignored** — the trie path infers the
    tree structure directly from the token sequences.

    Only ``attention_type="magi"`` is supported. The flex backend was retired
    due to the AReaL 8× entropy bug. Pass ``cp_group`` explicitly for the FSDP
    path so Magi's CP dispatch operates on the correct subgroup.

    Returns ``None`` when there's no shared prefix (single sample, multi-forest
    case, or empty input).
    """
    if attention_type != "magi":
        raise ValueError(
            f"attention_type={attention_type!r} is not supported. Only 'magi' is "
            "supported on FSDP (flex was retired due to the AReaL 8× entropy bug)."
        )
    tokens_by_sample = unpack_nested_to_list(input_ids)
    if not tokens_by_sample:
        return None
    loss_masks_by_sample = unpack_nested_to_list(loss_mask)
    position_ids_by_sample = unpack_nested_to_list(position_ids)

    # Trie insertion expects per-sample int lists (use tolist for the algorithm — Python int dict lookups)
    sequences = [t.tolist() for t in tokens_by_sample]
    max_tokens_per_tree = sum(len(s) for s in sequences) * 10  # one forest
    tries, _ = greedy_build_tries(sequences, max_tokens_per_tree=max_tokens_per_tree)
    if not tries or len(tries) > 1:
        return None

    converted = convert_trie_to_tree_node(tries[0])
    if converted is None:
        return None
    tree_root, node_info, leaves_in_dfs = converted

    params = build_arbitrary_depth_params(
        tokens_by_sample,
        tree_root,
        node_info,
        leaves_in_dfs,
        loss_masks_by_sample=loss_masks_by_sample,
        position_ids_by_sample=position_ids_by_sample,
    )

    # TP/CP padding (mirror of prefix_tree_magi.build_prefix_tree_micro_batch)
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

    # Build Magi attention key. magi_attn_flex_key returns the key (and registers
    # the corresponding runtime mgr inside Magi's per-cp_group LRU). Callers thread
    # this key explicitly to the attention func via the ``magi_attention_key``
    # kwarg on ``model(...)``; the LRU is only an internal Magi cache, never
    # observed from the verl side.
    #
    # When ``model is None`` (CPU-only algorithm tests / benchmarks), skip key
    # construction and return a PrefixTreeMagiBatch whose magi_key is None.
    # Production callers always pass a real model.
    if model is None:
        magi_key = None
    else:
        magi_key = _build_magi_key(model, params, cp_group=cp_group)
    flex_key = None

    local_flat_tokens = params.flat_tokens
    local_flat_position_ids = params.flat_position_ids
    local_flat_loss_mask = params.flat_loss_mask

    return PrefixTreeMagiBatch(
        flat_input_ids=params.flat_tokens,
        flat_position_ids=params.flat_position_ids,
        flat_loss_mask=params.flat_loss_mask,
        magi_key=magi_key,
        flex_key=flex_key,
        leaf_to_sample=params.leaf_to_sample,
        leaf_ranges=params.leaf_ranges,
        prefix_range=params.prefix_range,
        original_batch_size=len(tokens_by_sample),
        real_tokens=real_tokens,
        leaf_ancestor_ranges=getattr(params, "_leaf_ancestor_ranges", None),
        local_flat_input_ids=local_flat_tokens,
        local_flat_position_ids=local_flat_position_ids,
        local_flat_loss_mask=local_flat_loss_mask,
    )


# ============================================================================
# FSDP / HF model forward entry
# ============================================================================


def prefix_tree_dynamic_forward(
    model,
    input_ids,
    loss_mask=None,
    position_ids=None,
    tp_size: int = 1,
    cp_size: int = 1,
    cp_group=None,
    **model_kwargs,
):
    """End-to-end forward through an HF model using dynamic-trie + Magi attention.

    Pipeline:
      1. Build PrefixTreeMagiBatch (also registers the Magi runtime key in
         Magi's per-cp_group cache via ``magi_attn_flex_key``).
      2. Call ``set_magi_attention_key(model, pt_batch.magi_key)`` so the
         ``Magi_Attention`` backend (``_magi_prefix_tree_attention_forward``)
         can read the key off each attention module; then call
         ``model(input_ids=flat, position_ids=flat, attention_mask=None)``.
      3. Caller restores per-sample tensors via ``restore_flat_to_nested``.

    Returns ``(model_output, pt_batch)`` on success or ``(None, None)`` when
    there's no shared prefix (caller should fall back to dense forward).
    """
    from verl.models.transformers.monkey_patch import set_magi_attention_key

    pt_batch = build_prefix_tree_micro_batch_dynamic(
        model,
        input_ids,
        loss_mask=loss_mask,
        position_ids=position_ids,
        attention_type="magi",
        tp_size=tp_size,
        cp_size=cp_size,
        cp_group=cp_group,
    )
    if pt_batch is None:
        return None, None

    flat_input_ids = pt_batch.local_flat_input_ids.unsqueeze(0)
    flat_position_ids = pt_batch.local_flat_position_ids.unsqueeze(0)

    set_magi_attention_key(model, pt_batch.magi_key)
    output = model(
        input_ids=flat_input_ids,
        position_ids=flat_position_ids,
        attention_mask=None,
        **model_kwargs,
    )
    return output, pt_batch
