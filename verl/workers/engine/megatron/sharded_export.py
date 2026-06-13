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
shards each rank already holds, without any allgather. Each yield is a
4-tuple: ``(hf_name, local_tensor, global_box, full_shape)``:

* ``hf_name``: HF-canonical parameter name (post-unfuse, with global expert id).
* ``local_tensor``: this rank's tensor in HF layout (e.g. ``gate_proj``,
  not the fused ``linear_fc1``).
* ``global_box``: per-dim ``(start, end)`` box in the FULL HF tensor —
  directly consumable by ``ParameterShardMeta.ranges``.
* ``full_shape``: the FULL HF tensor's shape (un-sharded, un-fused) —
  directly consumable by ``ParameterShardMeta.full_shape``.

Layout transformations (e.g. Megatron's fused ``linear_qkv.weight`` →
HF's separate ``q_proj`` / ``k_proj`` / ``v_proj``) are pure local view +
split + cat, mirroring the design of vime PR #161
(``vime/backends/megatron_utils/megatron_to_hf/qwen2.py``). No collective.

MVP scope: dense attention (group-query attention OK), dense MLP, MoE with
``moe_grouped_gemm=True`` and ETP=1, shared experts. Mamba2/SSM (e.g.
Qwen3.5-VL) falls through to a 1:1 pass-through with a WARN log.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Callable

import torch

logger = logging.getLogger(__name__)

GlobalRange = tuple[int, int]
GlobalBox = tuple[GlobalRange, ...]
LocalShardTuple = tuple[str, torch.Tensor, GlobalBox, tuple[int, ...]]

TransformFn = Callable[
    ["MegatronToHFContext", str, torch.Tensor],
    Iterator[LocalShardTuple],
]


@dataclass
class MegatronToHFContext:
    """Context for local Megatron → HF conversions.

    Args:
        hf_config: HF ``PretrainedConfig`` instance. VL configs nest the
            language-model config under ``hf_config.text_config``; the
            accessors below auto-resolve that.
        tp_rank / tp_size: TP rank space.
        ep_rank / ep_size: Expert-parallel rank space (1/1 for non-MoE).
        etp_rank / etp_size: Expert TP rank space (MVP supports ETP=1 only).
    """

    hf_config: Any
    tp_rank: int
    tp_size: int
    ep_rank: int = 0
    ep_size: int = 1
    etp_rank: int = 0
    etp_size: int = 1

    def _text_config(self) -> Any:
        return getattr(self.hf_config, "text_config", self.hf_config)

    def hidden_size(self) -> int:
        return int(self._text_config().hidden_size)

    def num_heads(self) -> int:
        return int(self._text_config().num_attention_heads)

    def num_kv_heads(self) -> int:
        cfg = self._text_config()
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
        return getattr(self._text_config(), "moe_intermediate_size", None)

    def shared_expert_intermediate_size(self) -> int | None:
        return getattr(self._text_config(), "shared_expert_intermediate_size", None)

    def expert_local_to_global(self, local_idx: int) -> int:
        """Map local expert idx → global. Block partition: rank k holds
        ``[k*N_local, (k+1)*N_local)`` where ``N_local = N_total / ep_size``.
        """
        n = self.num_experts()
        if n is None:
            raise ValueError("expert_local_to_global called but hf_config has no num_experts")
        n_local = n // self.ep_size
        return self.ep_rank * n_local + local_idx


# ----------------------------------------------------------------------
# Box helpers
# ----------------------------------------------------------------------


def _full_box(shape: tuple[int, ...]) -> GlobalBox:
    return tuple((0, s) for s in shape)


def _tp_split_box(full_shape: tuple[int, ...], split_dim: int, tp_rank: int, tp_size: int) -> GlobalBox:
    local_size = full_shape[split_dim] // tp_size
    start = tp_rank * local_size
    end = start + local_size
    return tuple((start, end) if d == split_dim else (0, s) for d, s in enumerate(full_shape))


# ----------------------------------------------------------------------
# Concrete transforms
# ----------------------------------------------------------------------


def convert_qkv_to_q_k_v(
    ctx: MegatronToHFContext,
    qkv_param: torch.Tensor,
    layer_idx: int,
) -> Iterator[LocalShardTuple]:
    """Megatron ``linear_qkv.weight`` → HF ``q_proj`` / ``k_proj`` / ``v_proj``.

    Megatron interleaves Q/K/V per query-group along the row dim. With
    group-query attention, the row layout per TP rank is::

        [ Q_g0_h0..h{Gq}, K_g0, V_g0,
          Q_g1_h0..h{Gq}, K_g1, V_g1,
          ...                          ]

    where ``Gq = num_heads / num_query_groups`` and each KV head is the K/V
    for that group. Reshape exposes the group structure, split Q/K/V along
    the per-group axis, then flatten back to HF row-major matrices.
    """
    nq = ctx.num_heads()
    nkv = ctx.num_kv_heads()
    hd = ctx.head_dim()
    hidden = ctx.hidden_size()

    num_query_groups = nkv
    if nq % num_query_groups != 0:
        raise ValueError(f"num_heads ({nq}) not divisible by num_query_groups ({num_query_groups})")
    qpg = nq // num_query_groups
    if num_query_groups % ctx.tp_size != 0:
        raise ValueError(
            f"num_query_groups ({num_query_groups}) not divisible by tp_size ({ctx.tp_size}); "
            "MVP only supports clean TP partitioning of KV heads"
        )
    groups_per_rank = num_query_groups // ctx.tp_size

    expected_rows = (qpg + 2) * hd * groups_per_rank
    if qkv_param.shape != (expected_rows, hidden):
        raise ValueError(f"linear_qkv.weight shape {tuple(qkv_param.shape)} != expected ({expected_rows}, {hidden})")

    reshaped = qkv_param.view(groups_per_rank, qpg + 2, hd, hidden)
    q, k, v = torch.split(reshaped, [qpg, 1, 1], dim=1)
    q_flat = q.contiguous().view(-1, hidden)
    k_flat = k.contiguous().view(-1, hidden)
    v_flat = v.contiguous().view(-1, hidden)

    q_full = (nq * hd, hidden)
    kv_full = (nkv * hd, hidden)

    base = f"model.layers.{layer_idx}.self_attn"
    yield f"{base}.q_proj.weight", q_flat, _tp_split_box(q_full, 0, ctx.tp_rank, ctx.tp_size), q_full
    yield f"{base}.k_proj.weight", k_flat, _tp_split_box(kv_full, 0, ctx.tp_rank, ctx.tp_size), kv_full
    yield f"{base}.v_proj.weight", v_flat, _tp_split_box(kv_full, 0, ctx.tp_rank, ctx.tp_size), kv_full


def convert_qkv_bias_to_q_k_v(
    ctx: MegatronToHFContext,
    qkv_bias: torch.Tensor,
    layer_idx: int,
) -> Iterator[LocalShardTuple]:
    """Megatron ``linear_qkv.bias`` → HF ``q_proj.bias`` / ``k_proj.bias`` / ``v_proj.bias``.

    Same per-group interleaved layout as the weight, just 1-D.
    """
    nq = ctx.num_heads()
    nkv = ctx.num_kv_heads()
    hd = ctx.head_dim()
    num_query_groups = nkv
    if nq % num_query_groups != 0:
        raise ValueError(f"num_heads ({nq}) not divisible by num_query_groups ({num_query_groups})")
    qpg = nq // num_query_groups
    if num_query_groups % ctx.tp_size != 0:
        raise ValueError(
            f"num_query_groups ({num_query_groups}) not divisible by tp_size ({ctx.tp_size}); "
            "MVP only supports clean TP partitioning of KV heads"
        )
    groups_per_rank = num_query_groups // ctx.tp_size
    expected_rows = (qpg + 2) * hd * groups_per_rank
    if qkv_bias.shape != (expected_rows,):
        raise ValueError(f"linear_qkv.bias shape {tuple(qkv_bias.shape)} != expected ({expected_rows},)")

    reshaped = qkv_bias.view(groups_per_rank, qpg + 2, hd)
    q, k, v = torch.split(reshaped, [qpg, 1, 1], dim=1)
    q_flat = q.contiguous().view(-1)
    k_flat = k.contiguous().view(-1)
    v_flat = v.contiguous().view(-1)

    q_full = (nq * hd,)
    kv_full = (nkv * hd,)
    base = f"model.layers.{layer_idx}.self_attn"
    yield f"{base}.q_proj.bias", q_flat, _tp_split_box(q_full, 0, ctx.tp_rank, ctx.tp_size), q_full
    yield f"{base}.k_proj.bias", k_flat, _tp_split_box(kv_full, 0, ctx.tp_rank, ctx.tp_size), kv_full
    yield f"{base}.v_proj.bias", v_flat, _tp_split_box(kv_full, 0, ctx.tp_rank, ctx.tp_size), kv_full


def convert_fc1_to_gate_up(
    ctx: MegatronToHFContext,
    fc1_param: torch.Tensor,
    hf_gate_name: str,
    hf_up_name: str,
    intermediate_size: int,
) -> Iterator[LocalShardTuple]:
    """``linear_fc1.weight`` (fused gate+up, dim 0 trunk_concat) → ``gate_proj`` + ``up_proj``."""
    hidden = ctx.hidden_size()
    if intermediate_size % ctx.tp_size != 0:
        raise ValueError(f"intermediate_size ({intermediate_size}) not divisible by tp_size ({ctx.tp_size})")
    local_interm = intermediate_size // ctx.tp_size
    expected_rows = 2 * local_interm
    if fc1_param.shape != (expected_rows, hidden):
        raise ValueError(
            f"linear_fc1.weight shape {tuple(fc1_param.shape)} != expected ({expected_rows}, {hidden}) "
            f"for intermediate={intermediate_size} tp={ctx.tp_size}"
        )
    gate = fc1_param[:local_interm].contiguous()
    up = fc1_param[local_interm:].contiguous()
    full = (intermediate_size, hidden)
    box = _tp_split_box(full, 0, ctx.tp_rank, ctx.tp_size)
    yield hf_gate_name, gate, box, full
    yield hf_up_name, up, box, full


def convert_fc2_to_down(
    ctx: MegatronToHFContext,
    fc2_param: torch.Tensor,
    hf_down_name: str,
    intermediate_size: int,
) -> Iterator[LocalShardTuple]:
    """``linear_fc2.weight`` (RowParallel, dim 1 TP split) → ``down_proj.weight``."""
    hidden = ctx.hidden_size()
    if intermediate_size % ctx.tp_size != 0:
        raise ValueError("intermediate_size not divisible by tp_size")
    local_interm = intermediate_size // ctx.tp_size
    if fc2_param.shape != (hidden, local_interm):
        raise ValueError(f"linear_fc2.weight shape {tuple(fc2_param.shape)} != expected ({hidden}, {local_interm})")
    full = (hidden, intermediate_size)
    yield hf_down_name, fc2_param, _tp_split_box(full, 1, ctx.tp_rank, ctx.tp_size), full


# ----------------------------------------------------------------------
# Dispatcher
# ----------------------------------------------------------------------


_LAYER_RE = re.compile(r"^decoder\.layers\.(\d+)\.")
_MOE_EXPERT_FC1_RE = re.compile(r"^decoder\.layers\.(\d+)\.mlp\.experts\.linear_fc1\.weight(\d+)$")
_MOE_EXPERT_FC2_RE = re.compile(r"^decoder\.layers\.(\d+)\.mlp\.experts\.linear_fc2\.weight(\d+)$")


def _strip_module_prefix(name: str) -> str:
    """Strip DDP / Float16Module / language_model wrapper prefixes.

    Megatron with DistributedDataParallel + Float16Module produces names
    like ``module.module.decoder.layers.0...`` — both wrappers add a
    ``module.`` prefix, so a single-pass strip leaves one in place and
    breaks the ``_LAYER_RE`` match. Strip iteratively, then handle the
    VL-only ``language_model.`` namespace.
    """
    while name.startswith("module."):
        name = name[len("module.") :]
    if name.startswith("language_model."):
        name = name[len("language_model.") :]
    return name


def get_local_shards(
    module: torch.nn.Module,
    ctx: MegatronToHFContext,
) -> Iterator[LocalShardTuple]:
    """Yield ``(hf_name, local_tensor, global_box, full_shape)`` per HF param."""
    for raw_name, param in module.named_parameters():
        name = _strip_module_prefix(raw_name)
        yield from _dispatch_one(ctx, name, param)


def _emit_1to1(hf_name: str, param: torch.Tensor) -> LocalShardTuple:
    """Yield helper for a fully-local (replicated) 1:1 mapping."""
    shape = tuple(param.shape)
    return hf_name, param, _full_box(shape), shape


def _dispatch_one(ctx: MegatronToHFContext, name: str, param: torch.Tensor) -> Iterator[LocalShardTuple]:
    if name == "embedding.word_embeddings.weight":
        full = (ctx.vocab_size(), ctx.hidden_size())
        yield (
            "model.embed_tokens.weight",
            param,
            _tp_split_box(full, 0, ctx.tp_rank, ctx.tp_size),
            full,
        )
        return

    if name == "output_layer.weight":
        full = (ctx.vocab_size(), ctx.hidden_size())
        yield "lm_head.weight", param, _tp_split_box(full, 0, ctx.tp_rank, ctx.tp_size), full
        return

    if name == "decoder.final_layernorm.weight":
        yield _emit_1to1("model.norm.weight", param)
        return

    m = _LAYER_RE.match(name)
    if m is None:
        logger.warning("sharded_export: unhandled top-level param %r, passing through 1:1", name)
        yield _emit_1to1(name, param)
        return

    layer_idx = int(m.group(1))
    rest = name[m.end() :]
    yield from _dispatch_layer(ctx, layer_idx, rest, param)


def _dispatch_layer(
    ctx: MegatronToHFContext,
    layer_idx: int,
    rest: str,
    param: torch.Tensor,
) -> Iterator[LocalShardTuple]:
    base = f"model.layers.{layer_idx}"

    if rest == "input_layernorm.weight":
        yield _emit_1to1(f"{base}.input_layernorm.weight", param)
        return
    if rest == "pre_mlp_layernorm.weight":
        yield _emit_1to1(f"{base}.post_attention_layernorm.weight", param)
        return

    if rest == "self_attention.linear_qkv.weight":
        yield from convert_qkv_to_q_k_v(ctx, param, layer_idx)
        return
    if rest == "self_attention.linear_qkv.bias":
        yield from convert_qkv_bias_to_q_k_v(ctx, param, layer_idx)
        return
    if rest == "self_attention.linear_qkv.layer_norm_weight":
        yield _emit_1to1(f"{base}.input_layernorm.weight", param)
        return
    if rest == "self_attention.linear_proj.weight":
        hidden = ctx.hidden_size()
        local_in = hidden // ctx.tp_size
        if param.shape != (hidden, local_in):
            raise ValueError(f"linear_proj shape {tuple(param.shape)} != expected ({hidden}, {local_in})")
        full = (hidden, hidden)
        yield f"{base}.self_attn.o_proj.weight", param, _tp_split_box(full, 1, ctx.tp_rank, ctx.tp_size), full
        return

    if rest == "mlp.linear_fc1.weight":
        yield from convert_fc1_to_gate_up(
            ctx,
            param,
            f"{base}.mlp.gate_proj.weight",
            f"{base}.mlp.up_proj.weight",
            intermediate_size=ctx.intermediate_size(),
        )
        return
    if rest == "mlp.linear_fc1.layer_norm_weight":
        # Megatron fuses post_attention_layernorm into the linear_fc1 module.
        yield _emit_1to1(f"{base}.post_attention_layernorm.weight", param)
        return
    if rest == "mlp.linear_fc2.weight":
        yield from convert_fc2_to_down(
            ctx, param, f"{base}.mlp.down_proj.weight", intermediate_size=ctx.intermediate_size()
        )
        return

    if rest == "mlp.router.weight":
        yield _emit_1to1(f"{base}.mlp.gate.weight", param)
        return

    if rest in ("mlp.shared_experts.linear_fc1.weight", "mlp.shared_expert.linear_fc1.weight"):
        size = ctx.shared_expert_intermediate_size()
        if size is None:
            raise ValueError("shared_expert weight but no shared_expert_intermediate_size in hf_config")
        yield from convert_fc1_to_gate_up(
            ctx,
            param,
            f"{base}.mlp.shared_expert.gate_proj.weight",
            f"{base}.mlp.shared_expert.up_proj.weight",
            intermediate_size=size,
        )
        return
    if rest in ("mlp.shared_experts.linear_fc2.weight", "mlp.shared_expert.linear_fc2.weight"):
        size = ctx.shared_expert_intermediate_size()
        if size is None:
            raise ValueError("shared_expert weight but no shared_expert_intermediate_size in hf_config")
        yield from convert_fc2_to_down(ctx, param, f"{base}.mlp.shared_expert.down_proj.weight", intermediate_size=size)
        return
    if rest in ("mlp.shared_experts.gate_weight", "mlp.shared_expert.gate_weight"):
        yield _emit_1to1(f"{base}.mlp.shared_expert_gate.weight", param)
        return

    fq = _MOE_EXPERT_FC1_RE.match(f"decoder.layers.{layer_idx}.{rest}")
    if fq:
        local_e = int(fq.group(2))
        global_e = ctx.expert_local_to_global(local_e)
        size = ctx.moe_intermediate_size()
        if size is None:
            raise ValueError("MoE expert but no moe_intermediate_size in hf_config")
        # ETP=1: each expert is FULLY held locally; call converter with tp_size=1.
        etp1_ctx = MegatronToHFContext(
            hf_config=ctx.hf_config,
            tp_rank=0,
            tp_size=1,
            ep_rank=ctx.ep_rank,
            ep_size=ctx.ep_size,
        )
        yield from convert_fc1_to_gate_up(
            etp1_ctx,
            param,
            f"{base}.mlp.experts.{global_e}.gate_proj.weight",
            f"{base}.mlp.experts.{global_e}.up_proj.weight",
            intermediate_size=size,
        )
        return
    fq2 = _MOE_EXPERT_FC2_RE.match(f"decoder.layers.{layer_idx}.{rest}")
    if fq2:
        local_e = int(fq2.group(2))
        global_e = ctx.expert_local_to_global(local_e)
        size = ctx.moe_intermediate_size()
        if size is None:
            raise ValueError("MoE expert but no moe_intermediate_size in hf_config")
        etp1_ctx = MegatronToHFContext(
            hf_config=ctx.hf_config,
            tp_rank=0,
            tp_size=1,
            ep_rank=ctx.ep_rank,
            ep_size=ctx.ep_size,
        )
        yield from convert_fc2_to_down(
            etp1_ctx,
            param,
            f"{base}.mlp.experts.{global_e}.down_proj.weight",
            intermediate_size=size,
        )
        return

    logger.warning("sharded_export: unhandled layer param %r, passing through 1:1", rest)
    yield _emit_1to1(f"{base}.{rest}", param)


__all__ = [
    "MegatronToHFContext",
    "get_local_shards",
    "convert_qkv_to_q_k_v",
    "convert_fc1_to_gate_up",
    "convert_fc2_to_down",
    "LocalShardTuple",
]
