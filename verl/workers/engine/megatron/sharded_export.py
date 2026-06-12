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
"""Local Megatron → HF shard conversion for sharded-aware weight refit.

The existing path (``MegatronEngine.get_per_tensor_param`` →
``bridge.export_hf_weights``) performs an EP/TP allgather inside mbridge to
reconstruct the full HF tensor before yielding it. That defeats the purpose
of sharded refit, where the whole point is that each trainer rank sends
only its own slice.

This module provides ``get_local_shards`` which yields the LOCAL HF-format
shards each rank already holds, without any allgather:

    for hf_name, local_tensor, global_ranges in get_local_shards(
        module, hf_config, tp_rank=..., tp_size=..., ep_rank=..., ep_size=...,
    ):
        # local_tensor: this rank's shard in HF layout (e.g. gate_proj, not
        #               the fused linear_fc1)
        # global_ranges: per-dim (start, end) box in the FULL HF tensor —
        #               directly consumable by ``ParameterShardMeta.ranges``
        ...

Layout transformations (e.g. Megatron's fused ``linear_qkv.weight`` →
HF's separate ``q_proj`` / ``k_proj`` / ``v_proj``) are pure local view +
split + cat, mirroring the design of vime PR #161
(``vime/backends/megatron_utils/megatron_to_hf/qwen2.py``). No collective.

The module is parameterized by HF config (head counts, intermediate size,
etc.) so it works across Qwen2 / Qwen3 / similar MoE architectures
without hard-coding model-specific constants.

MVP scope: dense attention (group-query attention OK), dense MLP, MoE with
``moe_grouped_gemm=True`` and ETP=1, shared experts. Mamba2/SSM (e.g.
Qwen3.5-VL) is OUT of scope for MVP — falls through to the 1:1 pass-through
transform with a warning.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Callable

import torch

logger = logging.getLogger(__name__)

# Type alias: a per-dim (start, end) tuple in HF full-tensor coordinates.
GlobalRange = tuple[int, int]
GlobalBox = tuple[GlobalRange, ...]

# A transform takes the Megatron name + param + context (hf_config + parallel
# state) and yields one or more (hf_name, local_tensor, global_box) triples.
TransformFn = Callable[
    ["MegatronToHFContext", str, torch.Tensor],
    Iterator[tuple[str, torch.Tensor, GlobalBox]],
]


@dataclass
class MegatronToHFContext:
    """Context needed to perform local Megatron → HF conversions.

    Args:
        hf_config: HF ``PretrainedConfig`` instance. Used for shape constants
            (``hidden_size``, ``num_attention_heads``, ``num_key_value_heads``,
            ``intermediate_size``, ``num_experts``, ``moe_intermediate_size``,
            ``head_dim`` etc.). VL configs may nest these under
            ``hf_config.text_config`` — the context resolves that.
        tp_rank: This rank's TP rank inside ``[0, tp_size)``.
        tp_size: TP world size.
        ep_rank: Expert-parallel rank for MoE experts. 0 for non-MoE.
        ep_size: Expert-parallel world size. 1 for non-MoE.
        etp_rank: Expert tensor-parallel rank (typically 0 with ``ETP=1``).
        etp_size: Expert TP world size. MVP supports ETP=1 only.
    """

    hf_config: Any
    tp_rank: int
    tp_size: int
    ep_rank: int = 0
    ep_size: int = 1
    etp_rank: int = 0
    etp_size: int = 1

    # ------------------------------------------------------------------
    # HF config accessors. VL configs nest the language-model config
    # under ``hf_config.text_config``; flatten that here so callers can
    # query a single set of attributes.
    # ------------------------------------------------------------------

    def _text_config(self) -> Any:
        return getattr(self.hf_config, "text_config", self.hf_config)

    def hidden_size(self) -> int:
        return int(self._text_config().hidden_size)

    def num_heads(self) -> int:
        return int(self._text_config().num_attention_heads)

    def num_kv_heads(self) -> int:
        cfg = self._text_config()
        # Qwen2/3 fall back to num_attention_heads if absent (MHA).
        return int(getattr(cfg, "num_key_value_heads", cfg.num_attention_heads))

    def head_dim(self) -> int:
        cfg = self._text_config()
        explicit = getattr(cfg, "head_dim", None)
        if explicit:
            return int(explicit)
        return self.hidden_size() // self.num_heads()

    def intermediate_size(self) -> int:
        return int(self._text_config().intermediate_size)

    def vocab_size(self) -> int:
        return int(self._text_config().vocab_size)

    def num_experts(self) -> int | None:
        cfg = self._text_config()
        return getattr(cfg, "num_experts", None) or getattr(cfg, "num_local_experts", None)

    def moe_intermediate_size(self) -> int | None:
        cfg = self._text_config()
        return getattr(cfg, "moe_intermediate_size", None)

    def shared_expert_intermediate_size(self) -> int | None:
        cfg = self._text_config()
        return getattr(cfg, "shared_expert_intermediate_size", None)

    # ------------------------------------------------------------------
    # Local-expert-id ↔ global mapping (MVP: round-robin)
    # ------------------------------------------------------------------

    def expert_local_to_global(self, local_idx: int) -> int:
        """Map a local expert index (within this EP rank) to the global index.

        Convention: EP=4 over 256 experts → rank 0 holds globals 0..63,
        rank 1 holds 64..127, etc. (block partition).
        """
        num_experts = self.num_experts()
        if num_experts is None:
            raise ValueError("expert_local_to_global called but hf_config has no num_experts")
        num_local_experts = num_experts // self.ep_size
        return self.ep_rank * num_local_experts + local_idx


# ----------------------------------------------------------------------
# Transform helpers
# ----------------------------------------------------------------------


def _full_box(shape: tuple[int, ...]) -> GlobalBox:
    """Return a global box that covers the entire tensor (this rank holds full)."""
    return tuple((0, s) for s in shape)


def _tp_split_box(full_shape: tuple[int, ...], split_dim: int, tp_rank: int, tp_size: int) -> GlobalBox:
    """Compute the global box this rank's TP slice covers on ``split_dim``.

    Other dims are full ranges.
    """
    local_size = full_shape[split_dim] // tp_size
    start = tp_rank * local_size
    end = start + local_size
    return tuple((start, end) if d == split_dim else (0, s) for d, s in enumerate(full_shape))


# ----------------------------------------------------------------------
# Concrete transforms
# ----------------------------------------------------------------------


def _copy_1to1(
    ctx: MegatronToHFContext,
    hf_name: str,
    param: torch.Tensor,
) -> Iterator[tuple[str, torch.Tensor, GlobalBox]]:
    """Pass-through for 1:1 mappings (norms, biases, anything not sharded)."""
    yield hf_name, param, _full_box(tuple(param.shape))


def _copy_tp_split(
    ctx: MegatronToHFContext,
    hf_name: str,
    param: torch.Tensor,
    split_dim: int,
    full_size_on_dim: int,
) -> Iterator[tuple[str, torch.Tensor, GlobalBox]]:
    """1:1 with TP split on ``split_dim``.

    Useful for ColumnParallelLinear-style weights where the local tensor
    already represents one of N TP slices.
    """
    full_shape = tuple(s if d != split_dim else full_size_on_dim for d, s in enumerate(param.shape))
    yield hf_name, param, _tp_split_box(full_shape, split_dim, ctx.tp_rank, ctx.tp_size)


def convert_qkv_to_q_k_v(
    ctx: MegatronToHFContext,
    qkv_param: torch.Tensor,
    layer_idx: int,
) -> Iterator[tuple[str, torch.Tensor, GlobalBox]]:
    """Megatron ``linear_qkv.weight`` → HF ``q_proj`` / ``k_proj`` / ``v_proj``.

    Megatron interleaves Q / K / V per query-group along the row dimension
    (output dim) so that a single linear layer materializes all three at
    once. With group-query attention, the row layout per TP rank is::

        [ Q_g0_h0..h{Gq}, K_g0, V_g0,
          Q_g1_h0..h{Gq}, K_g1, V_g1,
          ...                          ]

    where ``Gq = num_heads / num_query_groups`` is the queries-per-group ratio
    and each KV head is the V/K for that group. The transformation reshapes
    to expose this group structure, splits Q/K/V along the per-group axis,
    and concatenates each into the HF-style row-major matrix.

    Borrowed from vime PR #161 (``convert_qwen2_to_hf_shard``); kept agnostic
    to specific architectures by parameterizing through ``ctx``.

    Yields (q, k, v) in HF names:
        ``model.layers.{layer_idx}.self_attn.q_proj.weight``
        ``model.layers.{layer_idx}.self_attn.k_proj.weight``
        ``model.layers.{layer_idx}.self_attn.v_proj.weight``
    """
    num_q_heads = ctx.num_heads()
    num_kv_heads = ctx.num_kv_heads()
    head_dim = ctx.head_dim()
    hidden = ctx.hidden_size()

    num_query_groups = num_kv_heads  # one KV head per group, by convention
    if num_q_heads % num_query_groups != 0:
        raise ValueError(
            f"num_attention_heads ({num_q_heads}) is not divisible by num_query_groups ({num_query_groups})"
        )
    queries_per_group = num_q_heads // num_query_groups

    if num_query_groups % ctx.tp_size != 0:
        raise ValueError(
            f"num_query_groups ({num_query_groups}) is not divisible by tp_size ({ctx.tp_size}); "
            "MVP only supports clean TP partitioning of KV heads"
        )
    groups_per_rank = num_query_groups // ctx.tp_size

    # qkv_param local shape: ((Gq + 1 + 1) * head_dim * groups_per_rank, hidden)
    expected_rows = (queries_per_group + 2) * head_dim * groups_per_rank
    if qkv_param.shape != (expected_rows, hidden):
        raise ValueError(
            f"linear_qkv.weight shape {tuple(qkv_param.shape)} does not match expected "
            f"({expected_rows}, {hidden}) for num_heads={num_q_heads}, num_kv_heads={num_kv_heads}, "
            f"head_dim={head_dim}, hidden={hidden}, tp={ctx.tp_size}"
        )

    # Reshape to (groups_per_rank, queries_per_group + 2, head_dim, hidden)
    reshaped = qkv_param.view(groups_per_rank, queries_per_group + 2, head_dim, hidden)
    # Split: q rows (queries_per_group) + k row (1) + v row (1)
    q, k, v = torch.split(reshaped, [queries_per_group, 1, 1], dim=1)

    # Collapse the group + per-group dims back into flat rows.
    q_flat = q.contiguous().view(-1, hidden)
    k_flat = k.contiguous().view(-1, hidden)
    v_flat = v.contiguous().view(-1, hidden)

    # HF global shapes (un-TP-sharded):
    q_full_rows = num_q_heads * head_dim
    kv_full_rows = num_kv_heads * head_dim

    q_box = _tp_split_box((q_full_rows, hidden), split_dim=0, tp_rank=ctx.tp_rank, tp_size=ctx.tp_size)
    k_box = _tp_split_box((kv_full_rows, hidden), split_dim=0, tp_rank=ctx.tp_rank, tp_size=ctx.tp_size)
    v_box = _tp_split_box((kv_full_rows, hidden), split_dim=0, tp_rank=ctx.tp_rank, tp_size=ctx.tp_size)

    base = f"model.layers.{layer_idx}.self_attn"
    yield f"{base}.q_proj.weight", q_flat, q_box
    yield f"{base}.k_proj.weight", k_flat, k_box
    yield f"{base}.v_proj.weight", v_flat, v_box


def convert_fc1_to_gate_up(
    ctx: MegatronToHFContext,
    fc1_param: torch.Tensor,
    hf_gate_name: str,
    hf_up_name: str,
    intermediate_size: int,
) -> Iterator[tuple[str, torch.Tensor, GlobalBox]]:
    """Megatron ``linear_fc1.weight`` (fused gate+up, dim 0 trunk_concat)
    → HF ``gate_proj.weight`` + ``up_proj.weight``.

    Megatron stores ``[gate_first_half_rows, up_first_half_rows]`` per TP
    shard (trunk_concat). Splitting the local tensor along dim 0 in halves
    yields this rank's gate and up shards directly — no collective needed.

    Args:
        intermediate_size: Full (un-TP-sharded) intermediate size. For dense
            MLP this is ``hf_config.intermediate_size``; for MoE experts it
            is ``moe_intermediate_size``; for shared experts
            ``shared_expert_intermediate_size``.
    """
    hidden = ctx.hidden_size()
    if intermediate_size % ctx.tp_size != 0:
        raise ValueError(f"intermediate_size ({intermediate_size}) is not divisible by tp_size ({ctx.tp_size})")
    local_intermediate = intermediate_size // ctx.tp_size
    expected_rows = 2 * local_intermediate
    if fc1_param.shape != (expected_rows, hidden):
        raise ValueError(
            f"linear_fc1.weight shape {tuple(fc1_param.shape)} != expected ({expected_rows}, {hidden}) "
            f"for intermediate={intermediate_size} tp={ctx.tp_size}"
        )

    gate = fc1_param[:local_intermediate].contiguous()
    up = fc1_param[local_intermediate:].contiguous()
    gate_box = _tp_split_box((intermediate_size, hidden), split_dim=0, tp_rank=ctx.tp_rank, tp_size=ctx.tp_size)
    up_box = _tp_split_box((intermediate_size, hidden), split_dim=0, tp_rank=ctx.tp_rank, tp_size=ctx.tp_size)
    yield hf_gate_name, gate, gate_box
    yield hf_up_name, up, up_box


def convert_fc2_to_down(
    ctx: MegatronToHFContext,
    fc2_param: torch.Tensor,
    hf_down_name: str,
    intermediate_size: int,
) -> Iterator[tuple[str, torch.Tensor, GlobalBox]]:
    """Megatron ``linear_fc2.weight`` (RowParallel, dim 1 TP split)
    → HF ``down_proj.weight``.

    HF shape: ``(hidden, intermediate)``. This rank's local shape:
    ``(hidden, intermediate / tp_size)``. The local tensor already IS the
    HF-format shard; just relabel and report the TP box on dim 1.
    """
    hidden = ctx.hidden_size()
    if intermediate_size % ctx.tp_size != 0:
        raise ValueError("intermediate_size not divisible by tp_size")
    local_intermediate = intermediate_size // ctx.tp_size
    if fc2_param.shape != (hidden, local_intermediate):
        raise ValueError(
            f"linear_fc2.weight shape {tuple(fc2_param.shape)} != expected ({hidden}, {local_intermediate})"
        )
    box = _tp_split_box((hidden, intermediate_size), split_dim=1, tp_rank=ctx.tp_rank, tp_size=ctx.tp_size)
    yield hf_down_name, fc2_param, box


# ----------------------------------------------------------------------
# Dispatcher
# ----------------------------------------------------------------------


# Regex on the param name (with wrapper prefixes stripped, see _strip_prefix below)
# → callable that emits ``(hf_name, local_tensor, global_box)`` triples.
_LAYER_RE = re.compile(r"^decoder\.layers\.(\d+)\.")
_MOE_EXPERT_FC1_RE = re.compile(r"^decoder\.layers\.(\d+)\.mlp\.experts\.linear_fc1\.weight(\d+)$")
_MOE_EXPERT_FC2_RE = re.compile(r"^decoder\.layers\.(\d+)\.mlp\.experts\.linear_fc2\.weight(\d+)$")


def _strip_module_prefix(name: str) -> str:
    """Strip ``module.`` and ``module.language_model.`` wrapper prefixes
    so name patterns match the canonical Megatron path.

    Wrapper chain examples (from M3 verify hook):
      - ``module.decoder.layers.0...`` (Float16Module)
      - ``module.language_model.decoder.layers.0...`` (Qwen3-VL: extra
        language_model intermediate)
    """
    if name.startswith("module.language_model."):
        return name[len("module.language_model.") :]
    if name.startswith("module."):
        return name[len("module.") :]
    return name


def get_local_shards(
    module: torch.nn.Module,
    ctx: MegatronToHFContext,
) -> Iterator[tuple[str, torch.Tensor, GlobalBox]]:
    """Yield this rank's HF-format shards from a Megatron module.

    Iterates ``module.named_parameters()`` and dispatches per param to the
    appropriate local transform. Yields ``(hf_name, local_tensor, global_box)``.

    Caller wraps each triple into a ``ParameterShardMeta`` (constructing
    ``ranges`` from ``global_box``) and feeds the list to
    ``build_transfer_plan``.

    Unrecognized parameters (e.g. Mamba SSM weights on Qwen3.5-VL) currently
    fall through to a 1:1 pass-through with a WARN log. MVP scope is
    Qwen2/Qwen3-style attention; SSM support is future work.
    """
    for raw_name, param in module.named_parameters():
        name = _strip_module_prefix(raw_name)
        yield from _dispatch_one(ctx, name, param)


def _dispatch_one(
    ctx: MegatronToHFContext,
    name: str,
    param: torch.Tensor,
) -> Iterator[tuple[str, torch.Tensor, GlobalBox]]:
    """Route one Megatron parameter to its transform."""

    # ----- Embedding / output -----
    if name == "embedding.word_embeddings.weight":
        # Megatron VocabParallelEmbedding shards vocab on dim 0.
        full_vocab = ctx.vocab_size()
        # Note: tp_size split on dim 0; account for vocab-padded sizes too,
        # but MVP assumes vocab is divisible by tp_size.
        full_shape = (full_vocab, ctx.hidden_size())
        yield (
            "model.embed_tokens.weight",
            param,
            _tp_split_box(full_shape, split_dim=0, tp_rank=ctx.tp_rank, tp_size=ctx.tp_size),
        )
        return

    if name == "output_layer.weight":
        full_vocab = ctx.vocab_size()
        full_shape = (full_vocab, ctx.hidden_size())
        yield (
            "lm_head.weight",
            param,
            _tp_split_box(full_shape, split_dim=0, tp_rank=ctx.tp_rank, tp_size=ctx.tp_size),
        )
        return

    if name == "decoder.final_layernorm.weight":
        yield "model.norm.weight", param, _full_box(tuple(param.shape))
        return

    # ----- Layer-scoped weights -----
    m = _LAYER_RE.match(name)
    if m is None:
        # Unrecognized top-level; pass through 1:1 so caller still sees it.
        logger.warning("sharded_export: unhandled top-level param %r, passing through 1:1", name)
        yield name, param, _full_box(tuple(param.shape))
        return

    layer_idx = int(m.group(1))
    rest = name[m.end() :]

    yield from _dispatch_layer(ctx, layer_idx, rest, param)


def _dispatch_layer(
    ctx: MegatronToHFContext,
    layer_idx: int,
    rest: str,
    param: torch.Tensor,
) -> Iterator[tuple[str, torch.Tensor, GlobalBox]]:
    base = f"model.layers.{layer_idx}"

    # ----- Norms (1:1) -----
    if rest == "input_layernorm.weight":
        yield f"{base}.input_layernorm.weight", param, _full_box(tuple(param.shape))
        return
    if rest == "pre_mlp_layernorm.weight":
        yield f"{base}.post_attention_layernorm.weight", param, _full_box(tuple(param.shape))
        return

    # ----- Attention -----
    if rest == "self_attention.linear_qkv.weight":
        yield from convert_qkv_to_q_k_v(ctx, param, layer_idx)
        return
    if rest == "self_attention.linear_qkv.layer_norm_weight":
        # Often colocated with linear_qkv but really an input_layernorm.
        yield f"{base}.input_layernorm.weight", param, _full_box(tuple(param.shape))
        return
    if rest == "self_attention.linear_proj.weight":
        # RowParallel, dim 1 TP split. HF: o_proj.weight (hidden, hidden).
        hidden = ctx.hidden_size()
        local_in = hidden // ctx.tp_size
        if param.shape != (hidden, local_in):
            raise ValueError(f"linear_proj shape {tuple(param.shape)} != expected ({hidden}, {local_in})")
        box = _tp_split_box((hidden, hidden), split_dim=1, tp_rank=ctx.tp_rank, tp_size=ctx.tp_size)
        yield f"{base}.self_attn.o_proj.weight", param, box
        return

    # ----- Dense MLP -----
    if rest == "mlp.linear_fc1.weight":
        yield from convert_fc1_to_gate_up(
            ctx,
            param,
            f"{base}.mlp.gate_proj.weight",
            f"{base}.mlp.up_proj.weight",
            intermediate_size=ctx.intermediate_size(),
        )
        return
    if rest == "mlp.linear_fc2.weight":
        yield from convert_fc2_to_down(
            ctx,
            param,
            f"{base}.mlp.down_proj.weight",
            intermediate_size=ctx.intermediate_size(),
        )
        return

    # ----- MoE router (no TP) -----
    if rest == "mlp.router.weight":
        yield f"{base}.mlp.gate.weight", param, _full_box(tuple(param.shape))
        return

    # ----- MoE shared expert -----
    if rest == "mlp.shared_experts.linear_fc1.weight" or rest == "mlp.shared_expert.linear_fc1.weight":
        size = ctx.shared_expert_intermediate_size()
        if size is None:
            raise ValueError("shared_expert weight encountered but hf_config has no shared_expert_intermediate_size")
        yield from convert_fc1_to_gate_up(
            ctx,
            param,
            f"{base}.mlp.shared_expert.gate_proj.weight",
            f"{base}.mlp.shared_expert.up_proj.weight",
            intermediate_size=size,
        )
        return
    if rest == "mlp.shared_experts.linear_fc2.weight" or rest == "mlp.shared_expert.linear_fc2.weight":
        size = ctx.shared_expert_intermediate_size()
        if size is None:
            raise ValueError("shared_expert weight encountered but hf_config has no shared_expert_intermediate_size")
        yield from convert_fc2_to_down(
            ctx,
            param,
            f"{base}.mlp.shared_expert.down_proj.weight",
            intermediate_size=size,
        )
        return
    if rest == "mlp.shared_experts.gate_weight" or rest == "mlp.shared_expert.gate_weight":
        # Qwen3 shared expert gating scalar (1, hidden).
        yield f"{base}.mlp.shared_expert_gate.weight", param, _full_box(tuple(param.shape))
        return

    # ----- MoE routed experts (grouped list-of-Parameters layout) -----
    m_fc1 = _MOE_EXPERT_FC1_RE.match(f"decoder.layers.{layer_idx}.{rest}")
    if m_fc1:
        local_e = int(m_fc1.group(2))
        global_e = ctx.expert_local_to_global(local_e)
        size = ctx.moe_intermediate_size()
        if size is None:
            raise ValueError("MoE expert encountered but hf_config has no moe_intermediate_size")
        # Routed experts: ETP=1 in MVP scope, so each expert is held FULLY
        # by this rank along TP. We split gate vs up on dim 0 directly.
        # NOTE: when ETP=1, intermediate is not TP-sharded, so call the
        # converter as if tp_size==1 for shape math.
        etp1_ctx = MegatronToHFContext(
            hf_config=ctx.hf_config,
            tp_rank=0,
            tp_size=1,
            ep_rank=ctx.ep_rank,
            ep_size=ctx.ep_size,
            etp_rank=0,
            etp_size=1,
        )
        yield from convert_fc1_to_gate_up(
            etp1_ctx,
            param,
            f"{base}.mlp.experts.{global_e}.gate_proj.weight",
            f"{base}.mlp.experts.{global_e}.up_proj.weight",
            intermediate_size=size,
        )
        return
    m_fc2 = _MOE_EXPERT_FC2_RE.match(f"decoder.layers.{layer_idx}.{rest}")
    if m_fc2:
        local_e = int(m_fc2.group(2))
        global_e = ctx.expert_local_to_global(local_e)
        size = ctx.moe_intermediate_size()
        if size is None:
            raise ValueError("MoE expert encountered but hf_config has no moe_intermediate_size")
        etp1_ctx = MegatronToHFContext(
            hf_config=ctx.hf_config,
            tp_rank=0,
            tp_size=1,
            ep_rank=ctx.ep_rank,
            ep_size=ctx.ep_size,
            etp_rank=0,
            etp_size=1,
        )
        yield from convert_fc2_to_down(
            etp1_ctx,
            param,
            f"{base}.mlp.experts.{global_e}.down_proj.weight",
            intermediate_size=size,
        )
        return

    # Fallback: 1:1 pass-through with a warning. Mamba SSM weights land here
    # on Qwen3.5-VL and we don't support them in MVP scope.
    logger.warning("sharded_export: unhandled layer-scoped param %r, passing through 1:1", rest)
    yield f"{base}.{rest}", param, _full_box(tuple(param.shape))


__all__ = [
    "MegatronToHFContext",
    "get_local_shards",
    "convert_qkv_to_q_k_v",
    "convert_fc1_to_gate_up",
    "convert_fc2_to_down",
]
