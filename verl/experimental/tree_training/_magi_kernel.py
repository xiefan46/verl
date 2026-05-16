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

"""Trie -> MagiAttention AttnRanges conversion + helpers.

Owns Stage 1b (mask construction) of the tree-training pipeline. The output
``(q_ranges, k_ranges, attn_type_map)`` is the tile-based representation
``flex_flash_attn_func`` / ``magi_attn_flex_key`` consume directly.

This module is import-safe without ``magi_attention`` installed (the int
encoding for AttnMaskType is mirrored locally), so it can be unit-tested on
CPU-only machines without the kernel installed.
"""

from __future__ import annotations

from typing import Sequence

import torch

from verl.experimental.tree_training.tree import TrieNode

# Mirror MagiAttention's int encoding for ``AttnMaskType``. The Magi enum lives
# at ``magi_attention/common/enum.py``; copying the int constants keeps this
# module CPU-testable without pulling in ``magi_attention`` (which requires
# CUDA). Constants checked against Magi v1.1.0.
ATTN_TYPE_FULL: int = 0
ATTN_TYPE_CAUSAL: int = 1
ATTN_TYPE_INVCAUSAL: int = 2
ATTN_TYPE_BICAUSAL: int = 3


def build_attn_ranges_from_trie(
    trie_roots: Sequence[TrieNode],
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[int]]:
    """Convert a sequence of trie roots to MagiAttention tile representation.

    For each non-root ``TrieNode`` ``N`` at packed positions ``[start, end_inclusive]``:

    * Emit one self-causal tile ``(q=N, k=N, CAUSAL)`` — tokens inside ``N``
      attend to earlier tokens of the same node.
    * For each non-root ancestor ``A`` of ``N``, emit one full-attention tile
      ``(q=N, k=A, FULL)`` — ``N`` fully attends to its ancestor path.

    Tokens outside any emitted ``q_range`` are not written by the kernel; for
    verl's packed layout, that means pad positions are skipped (downstream
    loss masking handles their contribution).

    Notes
    -----
    * ``TrieNode.end_idx`` is **inclusive**; Magi q/k_ranges use half-open
      ``[start, end)``. Conversion: ``end = end_idx + 1``.
    * The emitted ``q_ranges`` are typically overlapped — the same leaf
      ``q_range`` appears once per ancestor. Callers using
      ``flex_flash_attn_func`` must keep ``disable_fwd_atomic_reduction`` and
      ``disable_bwd_dkv_atomic_reduction`` at their defaults (False).

    Parameters
    ----------
    trie_roots
        One or more **root** ``TrieNode`` instances (``is_root=True``). Each
        root's ``nodes`` list is traversed in pre-order.

    Returns
    -------
    tuple[list, list, list]
        ``(q_ranges, k_ranges, attn_type_map)`` as Python lists. Each entry of
        ``q_ranges`` / ``k_ranges`` is a half-open ``(start, end)`` tuple;
        ``attn_type_map`` entries are integer codes (FULL=0, CAUSAL=1, ...).
        Returns three empty lists for fully-empty input (caller handles dummy
        via :func:`build_attn_ranges_tensors`).

    Raises
    ------
    ValueError
        If any non-root ``TrieNode`` is passed at the top level.
    """
    q_ranges: list[tuple[int, int]] = []
    k_ranges: list[tuple[int, int]] = []
    attn_type_map: list[int] = []

    for root in trie_roots:
        if not root.is_root:
            raise ValueError(
                f"build_attn_ranges_from_trie expects root TrieNodes; got tree_indices={root.tree_indices}"
            )

        for node in root.nodes:
            if node.is_root:
                continue

            q_start, q_end_inclusive = node.tree_indices
            q_end = q_end_inclusive + 1  # exclusive

            # 1) Self-causal: tokens within the node attend to earlier tokens
            #    of the same node (standard autoregressive causal).
            q_ranges.append((q_start, q_end))
            k_ranges.append((q_start, q_end))
            attn_type_map.append(ATTN_TYPE_CAUSAL)

            # 2) Each non-root ancestor: the leaf node fully attends to it.
            for ancestor in node.ancestors:
                if ancestor.is_root:
                    continue
                a_start, a_end_inclusive = ancestor.tree_indices
                a_end = a_end_inclusive + 1
                q_ranges.append((q_start, q_end))
                k_ranges.append((a_start, a_end))
                attn_type_map.append(ATTN_TYPE_FULL)

    return q_ranges, k_ranges, attn_type_map


def build_attn_ranges_tensors(
    trie_roots: Sequence[TrieNode],
    *,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tensor wrapper around :func:`build_attn_ranges_from_trie`.

    Returns ``(q_ranges, k_ranges, attn_type_map)`` as ``int32`` tensors
    ready for ``flex_flash_attn_func``.

    If the input trie is empty (no nodes — happens when a DP rank receives a
    "dummy" microbatch for load-balancing), emits a single zero-size sentinel
    tile ``(0, 0, 0, 0, FULL)`` so the downstream kernel call does not error
    on an empty range list.
    """
    q_list, k_list, type_list = build_attn_ranges_from_trie(trie_roots)

    if not q_list:
        q_t = torch.zeros((1, 2), dtype=torch.int32, device=device)
        k_t = torch.zeros((1, 2), dtype=torch.int32, device=device)
        type_t = torch.tensor([ATTN_TYPE_FULL], dtype=torch.int32, device=device)
        return q_t, k_t, type_t

    q_t = torch.tensor(q_list, dtype=torch.int32, device=device)
    k_t = torch.tensor(k_list, dtype=torch.int32, device=device)
    type_t = torch.tensor(type_list, dtype=torch.int32, device=device)
    return q_t, k_t, type_t


def materialize_dense_mask(
    total_seqlen: int,
    q_ranges: Sequence[tuple[int, int]],
    k_ranges: Sequence[tuple[int, int]],
    attn_type_map: Sequence[int],
) -> torch.Tensor:
    """Reconstruct a dense ``[T, T]`` boolean attention mask from tile lists.

    Test / debug oracle: verifies the tile decomposition emitted by
    :func:`build_attn_ranges_from_trie` represents the intended mask, without
    invoking the actual MagiAttention kernel. Not used at runtime.

    INVCAUSAL and BICAUSAL are implemented for completeness, although tree
    training only emits FULL and CAUSAL tiles.
    """
    mask = torch.zeros((total_seqlen, total_seqlen), dtype=torch.bool)
    for (qs, qe), (ks, ke), t in zip(q_ranges, k_ranges, attn_type_map, strict=False):
        tile_q = qe - qs
        tile_k = ke - ks
        if t == ATTN_TYPE_FULL:
            mask[qs:qe, ks:ke] = True
        elif t == ATTN_TYPE_CAUSAL:
            mask[qs:qe, ks:ke] |= torch.tril(torch.ones((tile_q, tile_k), dtype=torch.bool))
        elif t == ATTN_TYPE_INVCAUSAL:
            mask[qs:qe, ks:ke] |= torch.triu(torch.ones((tile_q, tile_k), dtype=torch.bool))
        elif t == ATTN_TYPE_BICAUSAL:
            mask[qs:qe, ks:ke] = True
        else:
            raise ValueError(f"unknown attn_type {t}")
    return mask
