# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
"""Prefix-tree + MAGI utilities for verl SFT / RL training.

Detects a shared-prefix tree across a micro-batch via **token-by-token trie
insertion** (no hash-based annotation needed), then builds a MAGI attention
key for the resulting packed sequence. Supports **arbitrary tree depth** —
the structure is whatever the input tokens reveal.

Pipeline:

    pt_batch = build_prefix_tree_micro_batch(model, input_ids, loss_mask, position_ids)
    if pt_batch is not None:
        output = model(
            input_ids=pt_batch.flat_input_ids,
            attention_mask=None,
            position_ids=pt_batch.flat_position_ids,
            magi_attention_key=pt_batch.magi_key,
        )
        output = restore_flat_to_nested(output, pt_batch)

Algorithm: per-sample sequences are inserted into a single greedy trie
(``_insert_sequence``), then path-compressed so each non-root node owns a
contiguous run of tokens shared by the same set of samples. The compressed
trie is converted to a ``TreeNode`` (the existing flex-spec data structure)
and walked DFS pre-order to produce ``(flat_tokens, q_ranges, k_ranges,
mask_types)`` consumed by ``magi_attn_flex_key``.

The trie naturally captures any depth — depth-1 (single shared prefix + per-
sample leaves), depth-2 (root + branch + leaves), and arbitrarily deeper
structures (root + ... + branch + ... + leaves) all fall out without special
casing.

Multi-forest support: when the batch contains samples that share no prefix
at all (i.e. greedy packing produces > 1 tries), the function returns ``None``
so the caller falls back to a dense path. Production batches in GRPO / SFT
training share at least the prompt prefix, so this fallback rarely triggers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import torch
from torch import Tensor
from torch.nested._internal.nested_tensor import NestedTensor

from verl.utils.prefix_tree_params import PrefixTreeParams
from verl.utils.prefix_tree_utils import TreeNode, build_multilevel_flex_spec


@dataclass
class PrefixTreeMagiBatch:
    """Holds the flat layout and MAGI key for one prefix-tree micro-batch."""

    # flat input tensors ready to pass to model(...)
    flat_input_ids: Tensor  # (total_tokens,)
    flat_position_ids: Tensor  # (total_tokens,)
    flat_loss_mask: Optional[Tensor]  # (total_tokens,) or None

    # Attention key — MAGI flex key (None when model is None, e.g. CPU-only tests)
    magi_key: object

    # mapping needed for output restoration
    # leaf_to_sample[i] = original sample index for leaf i
    leaf_to_sample: list[int]
    # leaf_ranges[i] = (start, end) token offset in flat layout for leaf i
    leaf_ranges: list[tuple[int, int]]
    # prefix_range = (start, end) of the root segment in the flat layout
    prefix_range: tuple[int, int]

    # original batch size (= number of leaves; leaves correspond 1:1 to samples)
    original_batch_size: int

    # number of real (non-padding) tokens; may be < flat_input_ids.shape[0]
    # when tp_size > 1 padding was added for sequence-parallel divisibility
    real_tokens: int = 0

    # leaf_ancestor_ranges[i] = list of (start,end) flat ranges that precede leaf i,
    # ordered root -> ... -> parent. Always populated by the trie path.
    leaf_ancestor_ranges: Optional[list[list[tuple[int, int]]]] = None

    # CP-local tensors: after magi dispatch, each CP rank only processes its
    # assigned tokens. When CP=1, these equal flat_input_ids/flat_position_ids/
    # flat_loss_mask. Shape: (local_tokens, ...).
    local_flat_input_ids: Optional[Tensor] = None
    local_flat_position_ids: Optional[Tensor] = None
    local_flat_loss_mask: Optional[Tensor] = None

    def __post_init__(self):
        if self.real_tokens == 0:
            self.real_tokens = int(self.flat_input_ids.shape[0])
        if self.local_flat_input_ids is None:
            self.local_flat_input_ids = self.flat_input_ids
        if self.local_flat_position_ids is None:
            self.local_flat_position_ids = self.flat_position_ids
        if self.local_flat_loss_mask is None:
            self.local_flat_loss_mask = self.flat_loss_mask


# ============================================================================
# Trie data structures
# ============================================================================


@dataclass
class _TrieNode:
    """Compressed-trie node (after `_compress` pass).

    Each non-root node represents a contiguous run of tokens shared by the same
    set of samples (its ``sequence_ids``). The root has ``tokens == []`` and
    children keyed by their first token id.
    """

    tokens: list[int] = field(default_factory=list)
    sequence_ids: list[int] = field(default_factory=list)
    children: dict[int, "_TrieNode"] = field(default_factory=dict)


class _BuildNode:
    """Internal — temporary uncompressed node used during insertion (one node
    per token edge). Compressed away by ``_compress_trie``.
    """

    __slots__ = ("token_id", "children", "is_end", "sequence_ids")

    def __init__(self, token_id: int):
        self.token_id = token_id
        self.children: dict[int, "_BuildNode"] = {}
        self.is_end = False
        self.sequence_ids: list[int] = []


# ============================================================================
# Trie construction (token-by-token greedy insertion)
# ============================================================================


def _count_additional_nodes(root: _BuildNode, sequence: list[int]) -> int:
    """How many new trie nodes would be added if we inserted ``sequence`` into
    ``root``? Used by greedy packing to balance tries.
    """
    current = root
    for idx, token in enumerate(sequence):
        child = current.children.get(token)
        if child is None:
            return len(sequence) - idx
        current = child
    return 0


def _insert_sequence(
    root: _BuildNode,
    sequence: list[int],
    sequence_id: int,
) -> None:
    """Insert ``sequence`` (a sample's token list) into the trie rooted at ``root``."""
    current = root
    for token in sequence:
        if token not in current.children:
            current.children[token] = _BuildNode(token)
        current.children[token].sequence_ids.append(sequence_id)
        current = current.children[token]
    current.is_end = True


def _compress_trie(root: _BuildNode) -> _TrieNode:
    """Path-compress the per-token trie so single-child chains collapse into
    one node owning a run of tokens.
    """
    compressed_root = _TrieNode()

    def _compress_chain(node: _BuildNode) -> _TrieNode:
        tokens: list[int] = []
        current = node
        while True:
            tokens.append(current.token_id)
            # Stop compressing when we hit a branch point or a terminal node.
            if len(current.children) != 1 or current.is_end:
                break
            current = next(iter(current.children.values()))
        compressed = _TrieNode(tokens=tokens, sequence_ids=current.sequence_ids.copy())
        for token, child in sorted(current.children.items()):
            compressed.children[token] = _compress_chain(child)
        return compressed

    for token, child in sorted(root.children.items()):
        compressed_root.children[token] = _compress_chain(child)
    return compressed_root


def _build_tries(
    sequences: list[list[int]],
    max_tokens_per_tree: int,
) -> list[_TrieNode]:
    """Greedy multi-forest packing: each sample joins the existing trie that
    adds the fewest new nodes; if no trie fits within budget, a new trie
    opens.

    For typical RL/SFT batches (all samples share at least a prompt) the
    output has length 1. ``max_tokens_per_tree`` is set huge by the caller so
    everything packs into one trie.
    """
    forests: list[dict[str, Any]] = []
    for seq_id, seq in enumerate(sequences):
        inserted = False
        for tree in forests:
            additional = _count_additional_nodes(tree["root"], seq)
            if tree["nodes"] + additional <= max_tokens_per_tree:
                _insert_sequence(tree["root"], seq, seq_id)
                tree["nodes"] += additional
                inserted = True
                break
        if inserted:
            continue
        if len(seq) > max_tokens_per_tree:
            raise ValueError(f"Sequence length {len(seq)} exceeds max_tokens_per_tree {max_tokens_per_tree}")
        new_root = _BuildNode(-1)
        _insert_sequence(new_root, seq, seq_id)
        forests.append({"root": new_root, "nodes": len(seq)})

    return [_compress_trie(f["root"]) for f in forests]


# ============================================================================
# Trie -> TreeNode (the data structure consumed by build_multilevel_flex_spec)
# ============================================================================


def _trie_to_tree_node(
    trie: _TrieNode,
) -> Optional[tuple[TreeNode, dict[int, tuple[int, int, int]], list[TreeNode]]]:
    """Convert a compressed trie into a ``TreeNode`` + bookkeeping.

    The trie root is a virtual placeholder (no tokens). We promote its sole
    child as the ``TreeNode`` root so the root carries the shared prefix.
    Returns ``None`` when there is no shared prefix (root has 0 or >1 children
    — i.e. no sharing or multi-forest).

    Bookkeeping outputs:
      - ``node_info[id(node)] = (owner_sample_idx, start, end)``: for every
        non-virtual node, an owning sample index + token range inside that
        sample's original sequence. Used to emit flat tokens.
      - ``leaves_in_dfs``: leaf TreeNodes in DFS pre-order (matches the order
        ``build_multilevel_flex_spec`` walks the tree).
    """
    if len(trie.children) != 1:
        # 0 children: no shared prefix. >1 children: multi-forest, no
        # single shared root.
        return None

    node_info: dict[int, tuple[int, int, int]] = {}
    leaves_in_dfs: list[TreeNode] = []

    def _convert(trie_node: _TrieNode, offset_in_owner: int) -> TreeNode:
        segment_len = len(trie_node.tokens)
        end_in_owner = offset_in_owner + segment_len

        children: list[TreeNode] = []
        for _tok, child in sorted(trie_node.children.items()):
            children.append(_convert(child, end_in_owner))

        node = TreeNode(segment_len=segment_len, children=children)

        if not children:
            assert len(trie_node.sequence_ids) == 1, (
                f"trie leaf should belong to exactly 1 sample, got {trie_node.sequence_ids}"
            )
            owner = trie_node.sequence_ids[0]
            node_info[id(node)] = (owner, offset_in_owner, end_in_owner)
            leaves_in_dfs.append(node)
        else:
            # Internal nodes inherit ownership from their first child for
            # token sourcing (the tokens are shared, so any descendant
            # sample's slice gives the right content).
            first_child_owner = node_info[id(children[0])][0]
            node_info[id(node)] = (first_child_owner, offset_in_owner, end_in_owner)
        return node

    only_child = next(iter(trie.children.values()))
    root = _convert(only_child, 0)
    if not root.children:
        return None
    return root, node_info, leaves_in_dfs


# ============================================================================
# Flat layout + flex spec for arbitrary-depth tree
# ============================================================================


def _build_params_arbitrary_depth(
    tokens_by_sample: list[Tensor],
    tree_root: TreeNode,
    node_info: dict[int, tuple[int, int, int]],
    leaves_in_dfs: list[TreeNode],
    loss_masks_by_sample: Optional[list[Tensor]] = None,
    position_ids_by_sample: Optional[list[Tensor]] = None,
) -> PrefixTreeParams:
    """Build :class:`PrefixTreeParams` for an arbitrary-depth tree.

    Walks the tree DFS pre-order, emitting each node's tokens (sourced from
    that node's owning sample). Builds ``flat_tokens / flat_loss_mask /
    flat_position_ids`` plus the per-leaf ancestor chain consumed by
    :func:`restore_flat_to_nested`. ``q_ranges / k_ranges / mask_types`` come
    from ``build_multilevel_flex_spec`` which is already arbitrary-depth.
    """
    q_ranges, k_ranges, mask_types = build_multilevel_flex_spec(tree_root)

    device = tokens_by_sample[0].device
    flat_pieces: list[Tensor] = []
    flat_lm_pieces: Optional[list[Tensor]] = [] if loss_masks_by_sample is not None else None
    flat_pid_pieces: Optional[list[Tensor]] = [] if position_ids_by_sample is not None else None
    default_pid_pieces: list[Tensor] = []  # used when position_ids_by_sample is None

    # Track parent for each TreeNode so we can produce the ancestor chain
    # consumed by restore_flat_to_nested.
    parent_of: dict[int, TreeNode] = {}

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

    # Leaf ranges are populated as side-effect by build_multilevel_flex_spec.
    leaf_ranges = [(leaf._flat_start, leaf._flat_end) for leaf in leaves_in_dfs]  # type: ignore[attr-defined]
    leaf_to_sample = [node_info[id(leaf)][0] for leaf in leaves_in_dfs]

    prefix_range = (tree_root._flat_start, tree_root._flat_end)  # type: ignore[attr-defined]
    sample_to_leaf_range = {s: r for s, r in zip(leaf_to_sample, leaf_ranges)}

    # Per-leaf ancestor flat ranges (root → parent), needed by restore_flat_to_nested.
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
# NestedTensor unpack
# ============================================================================


def _unpack_to_list(x) -> Optional[list[Tensor]]:
    """Unpack a NestedTensor (or pass-through list of Tensors) into a list of
    1-D tensors. Returns ``None`` when ``x is None``.
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
# Public entry: build_prefix_tree_micro_batch
# ============================================================================


def build_prefix_tree_micro_batch(
    model,
    input_ids,
    loss_mask=None,
    position_ids=None,
    attention_type: str = "magi",
    tp_size: int = 1,
    cp_size: int = 1,
) -> Optional[PrefixTreeMagiBatch]:
    """Build a :class:`PrefixTreeMagiBatch` from a micro-batch of variable-
    length token sequences.

    Detects the shared-prefix tree via token-by-token trie construction;
    supports arbitrary depth.

    Args:
        model: Megatron / HF model. Used to read num_heads / head_dim from its
            config when constructing the MAGI flex key. Pass ``None`` for
            CPU-only algorithm tests — the returned batch then has
            ``magi_key=None``.
        input_ids: ``NestedTensor`` of shape ``(batch_size, variable_seqlen)``,
            or a list of 1-D tensors.
        loss_mask: Optional, same shape as ``input_ids``.
        position_ids: Optional, same shape as ``input_ids``. When ``None``,
            default RoPE-compatible position IDs are generated (per-sample
            starts at 0).
        attention_type: Only ``"magi"`` is supported; the flex path was retired
            due to the AReaL 8x entropy bug.
        tp_size: Tensor-parallel world size, for SP-divisibility padding.
        cp_size: Context-parallel world size, for SP+CP padding.

    Returns:
        A :class:`PrefixTreeMagiBatch`, or ``None`` when the batch has no
        shared prefix (e.g. samples diverge at token 0, or greedy packing
        produces multiple disjoint tries). Callers should fall back to a
        dense forward path in that case.
    """
    if attention_type != "magi":
        raise ValueError(
            f"attention_type={attention_type!r} is not supported. Only 'magi' is supported "
            "(flex was retired due to the AReaL 8x entropy bug)."
        )

    tokens_by_sample = _unpack_to_list(input_ids)
    if not tokens_by_sample:
        return None

    loss_masks_by_sample = _unpack_to_list(loss_mask)
    position_ids_by_sample = _unpack_to_list(position_ids)

    # Run the trie on Python int lists — dict lookups are O(1) but Python ints
    # are massively faster than tensor indexing in the inner loop.
    sequences = [t.tolist() for t in tokens_by_sample]
    # Set max_tokens_per_tree to sum-of-lengths so all samples greedy-pack
    # into a single trie when they share any prefix. The multi-forest case
    # (returned as len(tries) > 1) signals "no shared root" → caller falls
    # back to dense.
    max_tokens_per_tree = sum(len(s) for s in sequences) + 1
    tries = _build_tries(sequences, max_tokens_per_tree=max_tokens_per_tree)
    if len(tries) != 1:
        return None

    converted = _trie_to_tree_node(tries[0])
    if converted is None:
        return None
    tree_root, node_info, leaves_in_dfs = converted

    params = _build_params_arbitrary_depth(
        tokens_by_sample,
        tree_root,
        node_info,
        leaves_in_dfs,
        loss_masks_by_sample=loss_masks_by_sample,
        position_ids_by_sample=position_ids_by_sample,
    )

    # SP/CP alignment padding.
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

    magi_key = _build_magi_key(model, params) if model is not None else None

    return PrefixTreeMagiBatch(
        flat_input_ids=params.flat_tokens,
        flat_position_ids=params.flat_position_ids,
        flat_loss_mask=params.flat_loss_mask,
        magi_key=magi_key,
        leaf_to_sample=params.leaf_to_sample,
        leaf_ranges=params.leaf_ranges,
        prefix_range=params.prefix_range,
        original_batch_size=len(tokens_by_sample),
        real_tokens=real_tokens,
        leaf_ancestor_ranges=getattr(params, "_leaf_ancestor_ranges", None),
        local_flat_input_ids=params.flat_tokens,
        local_flat_position_ids=params.flat_position_ids,
        local_flat_loss_mask=params.flat_loss_mask,
    )


# ============================================================================
# Restore flat output -> per-sample NestedTensor
# ============================================================================


def restore_flat_to_nested(
    flat_tensor: Tensor,
    pt_batch: PrefixTreeMagiBatch,
) -> NestedTensor:
    """Restore a flat ``(total_tokens, ...)`` tensor to a per-sample
    ``NestedTensor`` of shape ``(batch_size, variable_seqlen, ...)``.

    Each sample's view is the concatenation of its ancestor segments (root
    → ... → parent) followed by its leaf segment. Order matches the original
    per-sample sequence.
    """
    n = pt_batch.original_batch_size
    sample_tensors: list[Optional[Tensor]] = [None] * n

    for leaf_idx, sample_idx in enumerate(pt_batch.leaf_to_sample):
        leaf_start, leaf_end = pt_batch.leaf_ranges[leaf_idx]
        leaf_slice = flat_tensor[leaf_start:leaf_end]
        if pt_batch.leaf_ancestor_ranges is not None:
            parts = [flat_tensor[s:e] for s, e in pt_batch.leaf_ancestor_ranges[leaf_idx]]
            parts.append(leaf_slice)
            sample_tensors[sample_idx] = torch.cat(parts, dim=0)
        else:
            # Fall back to prefix_range when ancestor chain wasn't recorded.
            prefix_start, prefix_end = pt_batch.prefix_range
            sample_tensors[sample_idx] = torch.cat([flat_tensor[prefix_start:prefix_end], leaf_slice], dim=0)

    assert all(t is not None for t in sample_tensors), (
        "restore_flat_to_nested: some sample indices were not covered by leaf_to_sample"
    )
    return torch.nested.as_nested_tensor(sample_tensors, layout=torch.jagged)


# ============================================================================
# Token-hash utility (used by ssm_prefix_cache + multiturn_sft_dataset)
# ============================================================================


def _hash_prefix(token_ids_flat: Tensor) -> int:
    """128-bit hash of a 1-D token-id tensor.

    Uses xxhash when available (faster); falls back to hashlib.md5. Kept
    here as a shared utility for downstream cache / annotation code even
    though the prefix-tree builder no longer relies on hash-based detection.
    """
    raw = token_ids_flat.numpy().tobytes()
    try:
        import xxhash  # type: ignore[import]

        return xxhash.xxh128_intdigest(raw)
    except ImportError:
        import hashlib

        return int.from_bytes(hashlib.md5(raw).digest(), "little")


# ============================================================================
# MAGI flex key construction
# ============================================================================


def _build_magi_key(model, params):
    """Construct a MAGI ``magi_attn_flex_key`` from a model + params bundle."""
    import torch.distributed as dist
    from magi_attention.api import DistAttnConfig, magi_attn_flex_key
    from magi_attention.common import AttnRanges
    from magi_attention.common.enum import AttnMaskType
    from magi_attention.meta.solver.dispatch_solver import DispatchConfig

    from verl.utils.megatron_utils import unwrap_model

    cfg = unwrap_model(model).config
    num_heads_q = cfg.num_attention_heads
    num_heads_kv = getattr(cfg, "num_query_groups", num_heads_q) or num_heads_q
    head_dim = cfg.kv_channels

    try:
        from megatron.core import parallel_state as mpu

        cp_group = mpu.get_context_parallel_group()
    except Exception:
        cp_group = dist.group.WORLD

    return magi_attn_flex_key(
        q_ranges=AttnRanges.from_ranges(params.q_ranges),
        k_ranges=AttnRanges.from_ranges(params.k_ranges),
        attn_mask_type=[AttnMaskType(m) for m in params.mask_types],
        total_seqlen_q=params.total_seqlen_q,
        total_seqlen_k=params.total_seqlen_k,
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        head_dim=head_dim,
        pad_size=0,
        cp_group_or_mesh=cp_group,
        dist_attn_config=DistAttnConfig(dispatch_config=DispatchConfig(uneven_shard=True)),
    )


def _build_magi_key_sp_scaled(original_key, model, tp_size: int):
    """Rebuild a MAGI key scaled for the SP-scattered token domain (T/TP seqlen).

    With sequence_parallel=True, embedding scatter gives T/TP tokens per TP
    rank. The MAGI key must use T/TP as ``total_seqlen`` so dispatch/undispatch
    inside ``_magi_attn_forward`` operate on T/TP → T/(TP*CP) → T/TP.
    """
    import torch.distributed as dist
    from magi_attention.api import DistAttnConfig, magi_attn_flex_key
    from magi_attention.common import AttnRanges
    from magi_attention.meta.solver.dispatch_solver import DispatchConfig

    try:
        from megatron.core import parallel_state as mpu

        cp_group = mpu.get_context_parallel_group()
    except Exception:
        cp_group = dist.group.WORLD

    q_ranges_sp = [(r.start // tp_size, r.end // tp_size) for r in original_key.q_ranges]
    k_ranges_sp = [(r.start // tp_size, r.end // tp_size) for r in original_key.k_ranges]

    return magi_attn_flex_key(
        q_ranges=AttnRanges.from_ranges(q_ranges_sp),
        k_ranges=AttnRanges.from_ranges(k_ranges_sp),
        attn_mask_type=list(original_key.attn_mask_type),
        total_seqlen_q=original_key.total_seqlen_q // tp_size,
        total_seqlen_k=original_key.total_seqlen_k // tp_size,
        num_heads_q=original_key.num_heads_q,
        num_heads_kv=original_key.num_heads_kv,
        head_dim=original_key.head_dim,
        pad_size=0,
        cp_group_or_mesh=cp_group,
        dist_attn_config=DistAttnConfig(dispatch_config=DispatchConfig(uneven_shard=True)),
    )
