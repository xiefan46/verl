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
from verl.experimental.tree_training.functional import (
    _gather_packed_tree_logprobs,
    _gather_packed_tree_logprobs_entropy,
    gather_packed_tree_logprobs,
)
from verl.experimental.tree_training.module import build_tree_attn_kwargs
from verl.experimental.tree_training.tree import TrieNode, build_packed_tree_batch

__all__ = [
    "build_tree_mb_list",
    "build_tree_model_inputs",
    "unpack_tree_logprobs",
    "unpack_tree_logprobs_per_seq",
    "assemble_tree_per_seq_to_nested",
    "align_packed_extras_to_labels",
    "segment_lens_from_trie",
]


def segment_lens_from_trie(trie: TrieNode) -> list[int]:
    """Per-sequence packed-segment length in ``trie.all_sequence_ids`` order.

    For each sequence id in the trie, sum the token counts of the nodes it
    traverses — i.e., the original (full prompt + response) length. Matches
    the per-segment slicing convention used inside :func:`build_packed_tree_batch`
    (``_pack_extra_data``).
    """
    return [
        sum(end - start + 1 for (start, end) in trie.get_sequence_tree_indices(seq_id))
        for seq_id in trie.all_sequence_ids
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
    # NOTE: position_ids is intentionally NOT inserted into ``data``. The vendored
    # build_packed_tree_batch computes its own trie-aligned position_ids via
    # get_packed_tree_position_ids (length = padded_size, matching input_ids).
    # If we passed the user-side position_ids through ``data`` instead,
    # _pack_extra_data would pick them up as a packable extra (shape matches
    # input_template) and pack them in seq-id-flat layout (length = sum(seq_lens)),
    # then ``**extra_data`` in tree.py would override the trie-aligned ones.
    # That shape mismatch surfaced as a RuntimeError in HF Llama RotaryEmbedding
    # during Task 2.6 e2e smoke (q seq=padded_size, cos/sin seq=sum(seq_lens)).

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

    # 6. Propagate engine-side metadata (set by forward_backward_batch via
    # tu.assign_non_tensor before calling this adapter) into each mb dict.
    # ppo_loss and friends read these directly from `data` — for the dense
    # path they live as TensorDict non_tensor entries; for the tree path
    # we copy them into each plain-dict mb so the loss layer sees the same
    # contract regardless of dispatch.
    propagated_keys = (
        "dp_size",
        "batch_num_tokens",
        "global_batch_size",
        "temperature",
        "pad_mode",
        "use_remove_padding",
        "use_fused_kernels",
        "calculate_entropy",
        "calculate_sum_pi_squared",
    )
    propagated: dict = {}
    for key in propagated_keys:
        try:
            val = td.get(key, None)
        except Exception:
            val = None
        if val is None:
            continue
        # Unwrap NonTensorData if present
        unwrapped = getattr(val, "data", val) if hasattr(val, "data") and not torch.is_tensor(val) else val
        propagated[key] = unwrapped

    mbs = list(mb_list.padded_mbs)
    for mb in mbs:
        for k, v in propagated.items():
            mb.setdefault(k, v)

    # 7. Unwrap MicroBatchList into list[dict]; AReaL types stay behind the adapter.
    return mbs, metrics


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

    # Ensure position_ids has shape [1, T] for HF model compatibility. tree.py's
    # get_packed_tree_position_ids returns 1-D [T]; HF RotaryEmbedding indexes
    # position_ids[:, None, :] which fails on 1-D input.
    position_ids = mb.get("position_ids")
    if position_ids is not None and position_ids.dim() == 1:
        position_ids = position_ids.unsqueeze(0)

    model_inputs: dict[str, Any] = {
        "input_ids": packed_input_ids,
        "position_ids": position_ids,
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


def unpack_tree_logprobs_per_seq(
    logits: torch.Tensor,
    trie: TrieNode,
    packed_input_ids: torch.Tensor,
    *,
    temperature: float = 1.0,
    chunk_size: int = 1024,
    tp_group: Any = None,
    with_entropy: bool = False,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor] | None]:
    """Per-seq variant of :func:`unpack_tree_logprobs`.

    Same forward computation as :func:`unpack_tree_logprobs`, but skips the
    trailing ``torch.cat`` so we keep per-sequence tensors keyed by ``seq_id``.
    Used by ``compute_log_prob`` path (forward-only) to assemble a nested
    tensor aligned with the engine-side TensorDict ordering; the loss-time path
    keeps using the flat-cat variant where trie-order is what the loss kernel
    expects.

    Returns ``(log_probs_per_seq, entropy_per_seq_or_None)``. Empty dummy
    tries (``all_sequence_ids == []``) return ``({}, {} | None)``.
    """
    if not trie.all_sequence_ids:
        empty: dict[int, torch.Tensor] = {}
        return empty, ({} if with_entropy else None)

    if with_entropy:
        logprob_results, entropy_results = _gather_packed_tree_logprobs_entropy(
            logits, trie, packed_input_ids, temperature, chunk_size, tp_group
        )
        return logprob_results, entropy_results
    logprob_results = _gather_packed_tree_logprobs(logits, trie, packed_input_ids, temperature, chunk_size, tp_group)
    return logprob_results, None


def assemble_tree_per_seq_to_nested(
    per_mb_per_seq: list[dict[int, torch.Tensor]],
    *,
    offsets: torch.Tensor,
    sentinel: float = 0.0,
) -> torch.Tensor:
    """Assemble per-mb per-seq logprobs into a nested tensor matching ``offsets``.

    Each input dict maps ``seq_id`` (original input row index) to a 1-D tensor
    of length ``seq_lens[seq_id] - 1`` — one logprob per next-token prediction
    for that sequence: ``transitions[t] = log P(input_ids[t+1] | input_ids[0..t])``.

    Layout: **convention B** (matches verl's production dense path). Each row
    has length ``seq_lens[seq_id]``; row position ``k`` stores
    ``transitions[k] = log P(input_ids[k+1])`` for ``k in [0, L-1)``, with a
    sentinel APPENDED at the last position ``L-1`` (analogous to verl's dense
    path where the rolled-label wraparound puts garbage at the last position).

    The trainer's ``no_padding_2_padding`` slice
    ``values[seq_offset - resp_len - 1 : seq_offset - 1]`` (= within-row
    positions ``[prompt_len-1, prompt_len+resp_len-1)``) is calibrated for
    this convention: it extracts ``log P(input_ids[prompt_len]), ...,
    log P(input_ids[prompt_len+resp_len-1])`` — exactly the response token
    log_probs. The sentinel at position L-1 is the slice's exclusive endpoint
    and never read.

      1. For each ``seq_id`` in ``[0, N)``, look it up across all mb dicts;
         exactly one dict must own it (rank-local; trie partition is disjoint).
      2. Append a ``sentinel`` value so each row has length ``seq_lens[seq_id]``.
      3. Concat in row order and wrap with ``nested_tensor_from_jagged``.

    Dummy tries (mb dicts with no entries) contribute nothing.

    Args:
        per_mb_per_seq: one dict per micro-batch.
        offsets: ``data["input_ids"].offsets()`` from the engine-side
            TensorDict. ``offsets.diff()`` gives the per-row full seq_len.
        sentinel: value to append at position L-1 of each row.

    Returns:
        Nested tensor with the given offsets; ``.values()`` has length
        ``offsets[-1].item() == sum(seq_lens)``.
    """
    row_lens = offsets.diff().tolist()
    num_rows = len(row_lens)

    by_row: dict[int, torch.Tensor] = {}
    for mb_dict in per_mb_per_seq:
        for seq_id, tensor in mb_dict.items():
            if seq_id in by_row:
                raise ValueError(
                    f"seq_id={seq_id} appears in multiple micro-batch dicts; trie partitioning should be disjoint."
                )
            by_row[seq_id] = tensor

    if len(by_row) != num_rows:
        missing = sorted(set(range(num_rows)) - by_row.keys())
        raise ValueError(
            f"Tree per-seq unpack missing seq_ids: {missing[:10]}{'...' if len(missing) > 10 else ''} "
            f"(got {len(by_row)} / {num_rows} rows)"
        )

    # Pick reference device/dtype from the first non-empty tensor; fall back to float32/cpu
    # if every row is length-1 (no transitions anywhere — degenerate but legal).
    ref: torch.Tensor | None = next((t for t in by_row.values() if t.numel() > 0), None)
    if ref is None:
        device, dtype = torch.device("cpu"), torch.float32
    else:
        device, dtype = ref.device, ref.dtype

    parts: list[torch.Tensor] = []
    for i in range(num_rows):
        full_len = row_lens[i]
        seg = by_row[i]
        expected_transitions = full_len - 1
        if seg.numel() != expected_transitions:
            raise ValueError(
                f"seq_id={i}: expected {expected_transitions} transitions (full_len={full_len}), got {seg.numel()}."
            )
        sentinel_val = torch.full((1,), sentinel, device=device, dtype=dtype)
        # Convention B: append sentinel at position L-1 (matches verl dense path's
        # rolled-label wraparound). seg may be empty (full_len == 1); cat handles it.
        parts.append(torch.cat([seg.to(device=device, dtype=dtype), sentinel_val], dim=0))

    values = torch.cat(parts, dim=0) if parts else torch.zeros((0,), device=device, dtype=dtype)
    return torch.nested.nested_tensor_from_jagged(values, offsets=offsets.to(values.device))


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
