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
:func:`verl.utils.prefix_tree.build_prefix_tree_micro_batch` entry
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

from torch import Tensor

from verl.utils.prefix_tree import PrefixTreeMagiBatch
from verl.utils.prefix_tree_utils import TreeNode

__all__ = [
    "build_tree_dynamic",
    "build_prefix_tree_micro_batch_dynamic",
    "prefix_tree_dynamic_forward",
    # Lower-level helpers exposed for testing / benchmarking
    "TrieNode",
    "greedy_build_tries",
    "convert_trie_to_tree_node",
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
) -> Optional[tuple[TreeNode, list[int]]]:
    """Convert a compressed trie to ``(TreeNode, leaf_to_sample)`` consumed by
    :func:`verl.utils.prefix_tree_utils.build_layout_from_tree_node`.

    The trie root is a virtual placeholder with no tokens. We promote the
    trie's only child as the TreeNode root so the downstream flex-spec
    builder sees a non-zero root segment.

    Returns ``None`` when there's no real sharing (single sample, no children,
    or multi-forest case).

    Returns ``(root, leaf_to_sample)`` where ``leaf_to_sample[i]`` is the
    original sample index for the i-th leaf in DFS pre-order — matches the
    contract expected by ``build_layout_from_tree_node``.
    """
    if not trie.children:
        return None
    if len(trie.children) > 1:
        return None

    leaf_to_sample: list[int] = []

    def _convert(trie_node: TrieNode) -> TreeNode:
        segment_len = len(trie_node.tokens)
        children: list[TreeNode] = [_convert(child) for _tok, child in sorted(trie_node.children.items())]
        node = TreeNode(segment_len=segment_len, children=children)
        if not children:
            assert len(trie_node.sequence_ids) == 1, (
                f"Trie leaf should belong to exactly 1 sample, got {trie_node.sequence_ids}"
            )
            leaf_to_sample.append(trie_node.sequence_ids[0])
        return node

    only_child = next(iter(trie.children.values()))
    root = _convert(only_child)
    if not root.children:
        return None
    return root, leaf_to_sample


# ============================================================================
# Detection entry: build_tree_dynamic
# ============================================================================


def build_tree_dynamic(samples: list[Tensor]) -> Optional[tuple[TreeNode, list[int]]]:
    """Token-by-token trie detection. Returns ``(TreeNode, leaf_to_sample)`` or None.

    Builds a compressed trie of the input samples, then converts it to the
    canonical ``TreeNode`` representation consumed by
    :func:`verl.utils.prefix_tree_utils.build_layout_from_tree_node`.

    ``leaf_to_sample[i]`` gives the original sample index for the i-th leaf in
    DFS pre-order. Returns ``None`` when there's no shared prefix (empty input,
    single sample, or multi-forest case).
    """
    if not samples:
        return None
    sequences = [t.tolist() for t in samples]
    max_tokens_per_tree = sum(len(s) for s in sequences) * 10  # one forest
    tries, _ = greedy_build_tries(sequences, max_tokens_per_tree=max_tokens_per_tree)
    if not tries or len(tries) > 1:
        return None
    return convert_trie_to_tree_node(tries[0])


# ============================================================================
# Backwards-compat wrapper — delegates to the unified entry with dynamic_trie=True
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
    """Dynamic-only entry — thin wrapper around the unified
    :func:`verl.utils.prefix_tree.build_prefix_tree_micro_batch` with
    ``dynamic_trie=True``.

    Kept as a stable handle for existing callers (FSDP engine + sanity tests +
    benchmark) that pre-date the unified dispatcher. New code should call
    ``build_prefix_tree_micro_batch(..., dynamic_trie=True)`` instead.
    """
    if attention_type != "magi":
        raise ValueError(
            f"attention_type={attention_type!r} is not supported. Only 'magi' is "
            "supported on FSDP (flex was retired due to the AReaL 8× entropy bug)."
        )
    from verl.utils.prefix_tree import build_prefix_tree_micro_batch

    return build_prefix_tree_micro_batch(
        model,
        input_ids,
        loss_mask=loss_mask,
        position_ids=position_ids,
        prefix_segments_batch=prefix_segments_batch,  # ignored when dynamic_trie=True
        attention_type=attention_type,
        tp_size=tp_size,
        cp_size=cp_size,
        dynamic_trie=True,
        cp_group=cp_group,
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
