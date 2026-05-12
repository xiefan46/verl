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

"""Tree training adapter between verl's TensorDict-based engine and the
vendored AReaL tree algorithm.

Design (see ``research/2026-05-12-tree-training-phase2-design.md``):

  - AReaL types (``MicroBatchSpec`` / ``MicroBatchList`` / ``MicroBatchItem``) are
    *confined to this module + the vendored algorithm files*. The engine sees
    ``list[dict]`` symmetric to ``prepare_micro_batches``'s return value.
  - The engine layer remains agnostic of ``TreeTrainingConfig`` field details:
    only the resolved ``max_tokens_per_mb: int`` flows in.
  - The ``tree_token_ratio`` metric is computed here and returned alongside the
    micro-batch list so ``forward_step`` can add it to its ``metrics`` dict,
    which flows up to wandb via the standard engine → trainer path.

Public API (all functions / no class exports):

  - :func:`build_tree_mb_list`
  - :func:`build_tree_model_inputs`
  - :func:`unpack_tree_logprobs`
  - :func:`align_packed_extras_to_labels`
"""

from __future__ import annotations

from typing import Any

import torch

from verl.experimental.tree_training._areal_data import MicroBatchSpec
from verl.experimental.tree_training.functional import gather_packed_tree_logprobs
from verl.experimental.tree_training.module import build_tree_attn_kwargs
from verl.experimental.tree_training.tree import TrieNode, build_packed_tree_batch

__all__ = [
    "build_tree_mb_list",
    "build_tree_model_inputs",
    "unpack_tree_logprobs",
    "align_packed_extras_to_labels",
]


def _seq_lens_from_nested(nested: torch.Tensor) -> torch.Tensor:
    """Extract per-sequence lengths from a jagged nested tensor (in seq order)."""
    return nested.offsets().diff()


def _build_attention_mask_from_lens(seq_lens: torch.Tensor, max_seq_len: int) -> torch.Tensor:
    """Construct ``[N, max_seq_len]`` bool/int attention mask from per-seq lengths.

    Layout convention matches verl's left-pad-then-right-content convention:
    valid tokens are at positions ``[max_seq_len - seq_len, max_seq_len)``,
    padding at the left. ``left_right_2_no_padding`` removes padding via
    ``attention_mask`` so positions are full-seq from left to right by the
    time the adapter sees them — we reconstruct the mask as right-aligned
    valid tokens too. Actually, both conventions produce identical attention
    masks for ``build_packed_tree_batch`` which only uses ``mask.bool()`` to
    extract per-seq token lists. So we use the simpler left-aligned layout:
    valid tokens at ``[0, seq_len)`` then padding at the right.
    """
    device = seq_lens.device
    n = seq_lens.size(0)
    arange = torch.arange(max_seq_len, device=device).unsqueeze(0).expand(n, -1)
    return (arange < seq_lens.unsqueeze(1)).to(torch.long)


def _expand_response_only_to_full_seq(
    response_only: torch.Tensor,
    seq_lens: torch.Tensor,
    response_lens: torch.Tensor,
    max_seq_len: int,
) -> torch.Tensor:
    """Expand a ``[N, max_response_len]`` response-only tensor to ``[N, max_seq_len]``.

    Places each row's response values at positions ``[prompt_len, prompt_len + response_len)``
    of the corresponding row in the output; prompt + padding positions are zero.

    Args:
        response_only: ``[N, max_response_len]`` tensor (advantages, response_mask, etc.)
        seq_lens: ``[N]`` total length (prompt + response) per sequence.
        response_lens: ``[N]`` response length per sequence.
        max_seq_len: padded full-sequence length.

    Returns:
        ``[N, max_seq_len]`` tensor, response values placed at the right offset.
    """
    n = response_only.size(0)
    full = torch.zeros(n, max_seq_len, dtype=response_only.dtype, device=response_only.device)
    prompt_lens = seq_lens - response_lens
    for i in range(n):
        rl = int(response_lens[i].item())
        pl = int(prompt_lens[i].item())
        if rl > 0:
            full[i, pl : pl + rl] = response_only[i, :rl]
    return full


def _nested_to_padded(nested: torch.Tensor, *, padding: int | float, max_seq_len: int) -> torch.Tensor:
    """Wrapper around ``torch.nested.to_padded_tensor`` with explicit output size."""
    n = nested.offsets().numel() - 1
    return torch.nested.to_padded_tensor(nested, padding=padding, output_size=(n, max_seq_len))


def build_tree_mb_list(
    td: Any,  # tensordict.TensorDict, but importing the type here triggers heavy deps
    max_tokens_per_mb: int,
    *,
    pad_token_id: int = 0,
    pad_to_maximum: bool = True,
    dp_group: Any = None,
    parallel_size: int = 1,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Convert engine-side TensorDict into a list of tree-packed micro-batch dicts.

    Re-pads nested input_ids/position_ids back to ``[N, max_seq_len]``, expands
    response-only extras (advantages / old_log_probs / response_mask) to
    full-sequence layout, then invokes the vendored ``build_packed_tree_batch``
    algorithm. The returned ``list[dict]`` is symmetric to verl's existing
    ``prepare_micro_batches`` return — each dict is a ready-to-forward
    micro-batch.

    Args:
        td: TensorDict at the engine boundary, post-``left_right_2_no_padding``.
            Expected fields:
              - ``input_ids``: nested tensor (jagged, full prompt+response)
              - ``position_ids``: nested tensor
              - ``response_mask``: ``[N, max_response_len]`` bool/long
              - ``advantages``, ``old_log_probs`` (optional): ``[N, max_response_len]``
              - non-tensor: ``max_seq_len``, ``max_response_len``, ``indices``
        max_tokens_per_mb: maximum packed tokens per tree (must be multiple of 128).
        pad_token_id: token id used to pad ``input_ids``. Default 0; for real
            models pass the model's pad_token_id via the engine config.
        pad_to_maximum: forwarded to ``build_packed_tree_batch``.
        dp_group: data-parallel process group (None = world group).
        parallel_size: product of TP/SP dims requiring BLOCK_SIZE alignment.

    Returns:
        Tuple of:
          - ``list[dict]``: each dict is a packed micro-batch with keys
            ``input_ids``, ``position_ids``, ``trie_node``, plus packed extras
            (1-D, length ``sum(seq_lens)`` in ``trie.all_sequence_ids`` order).
          - ``dict[str, float]``: tree training metrics, currently just
            ``{"tree_token_ratio": <float>}``.
    """
    input_ids_nested = td["input_ids"]
    assert input_ids_nested.is_nested, "input_ids must be nested at engine boundary"

    seq_lens = _seq_lens_from_nested(input_ids_nested)
    max_seq_len = int(seq_lens.max().item())

    # 1. Re-pad input_ids: nested → [N, max_seq_len]
    input_ids_padded = _nested_to_padded(input_ids_nested, padding=pad_token_id, max_seq_len=max_seq_len)
    attention_mask_padded = _build_attention_mask_from_lens(seq_lens, max_seq_len)

    data: dict[str, Any] = {
        "input_ids": input_ids_padded,
        "attention_mask": attention_mask_padded,
    }

    # 2. Re-pad position_ids if 1-D layout. Multi-modal (3-D) skipped in MVP.
    if "position_ids" in td.keys():
        position_ids_nested = td["position_ids"]
        if position_ids_nested.is_nested:
            if position_ids_nested.values().dim() == 1:
                # 1-D position ids: nested with offsets, pad to [N, max_seq_len]
                data["position_ids"] = _nested_to_padded(position_ids_nested, padding=0, max_seq_len=max_seq_len)
            # else: 3-D position ids (Qwen-VL); intentionally not added to data,
            # so _pack_extra_data won't try to pack them. MVP scope: 1-D only.

    # 3. Expand response-only fields to full-sequence layout if present.
    response_mask_2d = td.get("response_mask")
    if response_mask_2d is None and "loss_mask" in td.keys():
        response_mask_2d = td["loss_mask"]
    if response_mask_2d is not None:
        response_lens = response_mask_2d.sum(dim=1).to(torch.int64)
        data["response_mask"] = _expand_response_only_to_full_seq(
            response_mask_2d.to(torch.long), seq_lens, response_lens, max_seq_len
        )
        for extra_key in ("advantages", "old_log_probs", "ref_log_prob"):
            if extra_key in td.keys():
                extra_2d = td[extra_key]
                # Skip nested tensors and non-2D shapes; only handle [B, max_response_len]
                if not extra_2d.is_nested and extra_2d.dim() == 2:
                    data[extra_key] = _expand_response_only_to_full_seq(extra_2d, seq_lens, response_lens, max_seq_len)

    # 4. Build packed tree batch via vendored algorithm.
    mb_spec = MicroBatchSpec(max_tokens_per_mb=max_tokens_per_mb)
    mb_list = build_packed_tree_batch(
        data,
        mb_spec,
        pad_to_maximum=pad_to_maximum,
        dp_group=dp_group,
        parallel_size=parallel_size,
    )

    # 5. Compute tree_token_ratio = unique trie tokens / total seq tokens.
    # mb_list.group_lens[i] is the unique-token count for trie i;
    # total seq tokens = sum of all seq_lens in input.
    unique_tokens = sum(mb_list.group_lens)
    total_tokens = int(seq_lens.sum().item())
    tree_token_ratio = unique_tokens / total_tokens if total_tokens > 0 else 1.0

    metrics = {"tree_token_ratio": float(tree_token_ratio)}

    # 6. Unwrap MicroBatchList into list[dict]; AReaL types stay behind the adapter.
    return list(mb_list.padded_mbs), metrics


def build_tree_model_inputs(
    mb: dict[str, Any],
    device: torch.device | str,
    *,
    extra_inputs: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the kwargs dict for ``model(**model_inputs)`` from a packed mb.

    Returns ``(model_inputs, output_args)`` where:
      - ``model_inputs``: kwargs passed to ``model.forward`` — includes
        ``input_ids``, ``position_ids``, ``attention_mask=None`` (the patched
        flash_attention reads ``tree_block_mask`` from kwargs instead), and
        ``tree_block_mask`` (or ``tree_triton_data`` if Triton is enabled,
        not in MVP).
      - ``output_args``: opaque dict carrying ``trie_node`` and
        ``packed_input_ids`` so ``unpack_tree_logprobs`` can be called after
        the forward, plus anything from ``extra_inputs`` (e.g. ``temperature``).

    Args:
        mb: one dict from the list returned by :func:`build_tree_mb_list`.
        device: device for the on-the-fly BlockMask construction.
        extra_inputs: optional metadata (e.g. ``{"temperature": 0.7}``) to be
            stashed in ``output_args`` for later use by the loss / unpack layer.
    """
    trie: TrieNode = mb["trie_node"]
    packed_input_ids: torch.Tensor = mb["input_ids"]
    padded_size = packed_input_ids.size(-1)
    dev = torch.device(device) if not isinstance(device, torch.device) else device

    tree_attn_kwargs = build_tree_attn_kwargs(trie, padded_size, device=dev)

    model_inputs: dict[str, Any] = {
        "input_ids": packed_input_ids,
        "position_ids": mb.get("position_ids"),
        "attention_mask": None,  # tree_block_mask supersedes this for flash_attention
        **tree_attn_kwargs,
    }
    # Drop position_ids entry if it wasn't in mb (defensive; tree.py always sets it).
    if model_inputs["position_ids"] is None:
        del model_inputs["position_ids"]

    output_args: dict[str, Any] = {
        "trie": trie,
        "packed_input_ids": packed_input_ids,
    }
    if extra_inputs:
        output_args.update(extra_inputs)

    return model_inputs, output_args


def unpack_tree_logprobs(
    logits: torch.Tensor,
    trie: TrieNode,
    packed_input_ids: torch.Tensor,
    *,
    temperature: float = 1.0,
    chunk_size: int = 1024,
    tp_group: Any = None,
) -> torch.Tensor:
    """Walk the trie and return a flat 1-D tensor of per-seq next-token logprobs.

    Output is in ``trie.all_sequence_ids`` order; length = ``sum(seq_lens - 1)``
    (the algorithm-layer patch in ``functional.py`` ensures one logprob per
    next-token prediction, matching the docstring contract).

    Args:
        logits: model output of shape ``[T_padded, V]`` (already squeezed of
            the leading batch=1 dim).
        trie: the ``TrieNode`` carried from ``build_tree_mb_list`` via
            ``output_args["trie"]``.
        packed_input_ids: the packed input id tensor of shape ``[1, T_padded]``
            (or ``[T_padded]``).
        temperature: softmax temperature; default 1.0.
        chunk_size: memory-efficient chunking budget (passed through).
        tp_group: tensor-parallel process group; MVP enforces ``None``.

    Returns:
        Flat 1-D float tensor, length ``sum(seq_lens - 1)``.
    """
    return gather_packed_tree_logprobs(
        logits,
        trie,
        packed_input_ids,
        temperature=temperature,
        chunk_size=chunk_size,
        tp_group=tp_group,
    )


def align_packed_extras_to_labels(packed: torch.Tensor, segment_lens: list[int]) -> torch.Tensor:
    """Drop position 0 of each segment to align packed extras with next-token logprobs.

    The packed extras (advantages / old_log_probs / response_mask) emerge from
    ``_pack_extra_data`` (tree.py:674) as a flat tensor of length
    ``sum(seq_lens)`` in ``trie.all_sequence_ids`` order, with each sequence's
    portion being the original per-token values. The logprob output from
    :func:`unpack_tree_logprobs` is shorter by one entry per sequence: it
    contains ``log P(token_{i+1} | tokens_{0..i})`` for ``i ∈ [0, L-2]``
    — i.e., one prediction per "next" token. For elementwise loss math the
    extras must be sliced to the same alignment.

    Convention: keep entries ``[1, L)`` of each segment (drop position 0 of
    each seq). Rationale: extras like ``advantages[t]`` are stored at the
    *position the action was taken* in verl's GRPO convention; ``log_probs[t-1]``
    predicts ``token[t]``. So pairing ``log_probs[t-1]`` with ``advantages[t]``
    for ``t ∈ [1, L-1]`` is what GRPO needs. Equivalent to dropping the
    first entry of each segment.

    Args:
        packed: flat 1-D tensor, length ``sum(segment_lens)``.
        segment_lens: per-segment length (in trie order).

    Returns:
        Flat 1-D tensor, length ``sum(L - 1 for L in segment_lens)``.
    """
    parts = []
    offset = 0
    for length in segment_lens:
        if length > 1:
            parts.append(packed[offset + 1 : offset + length])
        offset += length
    if not parts:
        return packed.new_empty(0)
    return torch.cat(parts, dim=0)
