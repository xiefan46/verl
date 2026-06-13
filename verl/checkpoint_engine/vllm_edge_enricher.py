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
"""Fill vLLM-side metadata on routing-plan edges.

``build_transfer_plan`` produces edges with ``target_param_name``,
``shard_id``, ``expert_id`` all set to ``None`` — the routing algorithm is
deliberately decoupled from vLLM's fused-tensor packing.

This module bridges that gap. Given an HF-canonical ``param_name`` (e.g.
``"model.layers.0.mlp.experts.3.gate_proj.weight"``), ``vllm_enrich_edge``
returns a new ``TransferEdge`` with the fields vLLM's ``weight_loader``
needs::

    model.experts.weight_loader(
        param=lookup(target_param_name),
        loaded_weight=recv_buf,
        weight_name=param_name,
        shard_id=shard_id,        # "w1"/"w2"/"w3" for MoE, "q"/"k"/"v" for QKV
        expert_id=expert_id,
    )

Mapping rules (derived from the vLLM ``RoutedExperts`` + standard
attention conventions verified in M2 RunPod dump):

================================ ================ ==================== ===========
HF-canonical name                Target           shard_id             expert_id
================================ ================ ==================== ===========
...experts.{i}.gate_proj.weight  ...experts.w13   "w1"                 i
...experts.{i}.up_proj.weight    ...experts.w13   "w3"                 i
...experts.{i}.down_proj.weight  ...experts.w2    "w2"                 i
...self_attn.q_proj.weight       ...qkv_proj      "q"                  None
...self_attn.k_proj.weight       ...qkv_proj      "k"                  None
...self_attn.v_proj.weight       ...qkv_proj      "v"                  None
everything else                  (same)           None                 None
================================ ================ ==================== ===========

Everything not matched falls through 1:1 (same target_param_name as
param_name, no shard_id, no expert_id) — appropriate for embeddings,
o_proj, down_proj for dense MLP, layer norms, router gate, etc.
"""

from __future__ import annotations

import re

from verl.checkpoint_engine.parallel_meta import TransferEdge

# Routed expert patterns. Captures (prefix, expert_idx, last_segment).
_MOE_GATE_RE = re.compile(r"^(.*\.experts)\.(\d+)\.gate_proj\.weight$")
_MOE_UP_RE = re.compile(r"^(.*\.experts)\.(\d+)\.up_proj\.weight$")
_MOE_DOWN_RE = re.compile(r"^(.*\.experts)\.(\d+)\.down_proj\.weight$")

# Standard attention QKV patterns — capture (prefix, tail in {"weight","bias"}).
_ATTN_Q_RE = re.compile(r"^(.*\.self_attn)\.q_proj\.(weight|bias)$")
_ATTN_K_RE = re.compile(r"^(.*\.self_attn)\.k_proj\.(weight|bias)$")
_ATTN_V_RE = re.compile(r"^(.*\.self_attn)\.v_proj\.(weight|bias)$")

# Dense MLP gate / up — vLLM fuses these into ``gate_up_proj`` via
# ``MergedColumnParallelLinear`` with int shard_ids 0/1. These patterns must
# fire AFTER the MoE patterns above (which require ``experts.{i}.`` prefix);
# anything matching here therefore can't be a routed-expert weight.
_MLP_GATE_RE = re.compile(r"^(.+)\.gate_proj\.weight$")
_MLP_UP_RE = re.compile(r"^(.+)\.up_proj\.weight$")


def _replace_edge(
    edge: TransferEdge,
    *,
    target_param_name: str | None,
    shard_id: str | None,
    expert_id: int | None,
) -> TransferEdge:
    """Return a new TransferEdge with the vLLM-side fields filled."""
    return TransferEdge(
        param_name=edge.param_name,
        src_global_rank=edge.src_global_rank,
        dst_global_rank=edge.dst_global_rank,
        src_local_slice_encoded=edge.src_local_slice_encoded,
        dst_local_slice_encoded=edge.dst_local_slice_encoded,
        shape=edge.shape,
        dtype_str=edge.dtype_str,
        target_param_name=target_param_name,
        shard_id=shard_id,
        expert_id=expert_id,
    )


def vllm_enrich_edge(edge: TransferEdge) -> TransferEdge:
    """Resolve vLLM-side ``target_param_name`` / ``shard_id`` / ``expert_id``
    from the HF-canonical ``edge.param_name``.

    Pass as the ``enrich_edge`` callback to
    ``build_transfer_plan(train_metas, rollout_metas, enrich_edge=...)``.
    """
    name = edge.param_name

    # ----- MoE routed experts -----
    m = _MOE_GATE_RE.match(name)
    if m is not None:
        prefix, expert_idx = m.group(1), int(m.group(2))
        return _replace_edge(
            edge,
            target_param_name=f"{prefix}.w13_weight",
            shard_id="w1",
            expert_id=expert_idx,
        )

    m = _MOE_UP_RE.match(name)
    if m is not None:
        prefix, expert_idx = m.group(1), int(m.group(2))
        return _replace_edge(
            edge,
            target_param_name=f"{prefix}.w13_weight",
            shard_id="w3",
            expert_id=expert_idx,
        )

    m = _MOE_DOWN_RE.match(name)
    if m is not None:
        prefix, expert_idx = m.group(1), int(m.group(2))
        return _replace_edge(
            edge,
            target_param_name=f"{prefix}.w2_weight",
            shard_id="w2",
            expert_id=expert_idx,
        )

    # ----- Standard attention QKV (fused into vLLM's qkv_proj). Both .weight
    # and .bias share the qkv_proj fusion, with the same string shard_id. -----
    m = _ATTN_Q_RE.match(name)
    if m is not None:
        return _replace_edge(
            edge,
            target_param_name=f"{m.group(1)}.qkv_proj.{m.group(2)}",
            shard_id="q",
            expert_id=None,
        )
    m = _ATTN_K_RE.match(name)
    if m is not None:
        return _replace_edge(
            edge,
            target_param_name=f"{m.group(1)}.qkv_proj.{m.group(2)}",
            shard_id="k",
            expert_id=None,
        )
    m = _ATTN_V_RE.match(name)
    if m is not None:
        return _replace_edge(
            edge,
            target_param_name=f"{m.group(1)}.qkv_proj.{m.group(2)}",
            shard_id="v",
            expert_id=None,
        )

    # ----- Dense MLP gate / up → MergedColumnParallelLinear gate_up_proj
    # (int shard_ids 0 = gate, 1 = up). The MoE expert regex above guarantees
    # routed experts already returned, so a match here means dense MLP (or a
    # gated shared_expert sub-module that also uses MergedColumn). -----
    m = _MLP_GATE_RE.match(name)
    if m is not None:
        return _replace_edge(
            edge,
            target_param_name=f"{m.group(1)}.gate_up_proj.weight",
            shard_id=0,
            expert_id=None,
        )
    m = _MLP_UP_RE.match(name)
    if m is not None:
        return _replace_edge(
            edge,
            target_param_name=f"{m.group(1)}.gate_up_proj.weight",
            shard_id=1,
            expert_id=None,
        )

    # ----- Everything else: 1:1 -----
    return _replace_edge(
        edge,
        target_param_name=name,
        shard_id=None,
        expert_id=None,
    )


__all__ = ["vllm_enrich_edge"]
