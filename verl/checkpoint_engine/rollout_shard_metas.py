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
"""Rollout-side ``ParameterShardMeta`` enumeration from ``hf_config``.

The trainer side computes its ``ParameterShardMeta`` by walking the live
Megatron module via :mod:`verl.workers.engine.megatron.sharded_export`.
The rollout side has no Megatron module to walk — it lives in a separate
Ray actor that only talks to a vLLM subprocess. We therefore enumerate the
HF-canonical params + shapes purely from ``hf_config`` and emit metas with
ranges set to whatever slice of each param this rollout rank holds.

MVP scope: dense Qwen2-family models with attention `q_proj.bias` /
`k_proj.bias` / `v_proj.bias` (``add_qkv_bias``). Single-rank rollout
(``tp_size=1`` ⇒ ranges = full box). Multi-rank TP support requires
mirroring vLLM's column-/row-parallel partitioning here, future work.
"""

from __future__ import annotations

from typing import Any

from verl.checkpoint_engine.parallel_meta import ParameterShardMeta


def _text_config(hf_config: Any) -> Any:
    """VL configs nest the LM under ``text_config``; auto-resolve."""
    return getattr(hf_config, "text_config", hf_config)


def _has_qkv_bias(text_cfg: Any) -> bool:
    """Heuristic: Qwen2 family always has qkv bias; some others do too."""
    # Explicit flags.
    if getattr(text_cfg, "qkv_bias", None) is True:
        return True
    if getattr(text_cfg, "attention_bias", None) is True:
        return True
    # Family fallback: Qwen2 / Qwen2.5 hardcode it.
    arch_name = type(text_cfg).__name__
    return "Qwen2" in arch_name or "Qwen3" in arch_name


def build_rollout_shard_metas(
    hf_config: Any,
    *,
    tp_rank: int = 0,
    tp_size: int = 1,
    ep_rank: int = 0,
    ep_size: int = 1,
    dtype_str: str = "bfloat16",
) -> list[ParameterShardMeta]:
    """Enumerate HF-canonical param shapes and emit per-param shard metas.

    For MVP we only support ``tp_size = 1`` / ``ep_size = 1`` (vLLM running
    on a single GPU): every meta covers the full tensor (``ranges = full_box``).
    The signature still takes the rank info so this stays the place to add
    real TP/EP partitioning later without changing call sites.
    """
    if tp_size != 1 or ep_size != 1:
        raise NotImplementedError(
            "build_rollout_shard_metas MVP only handles tp_size=ep_size=1 "
            f"(got tp_size={tp_size}, ep_size={ep_size}). "
            "Multi-rank rollout requires mirroring vLLM's column-/row-parallel "
            "split logic here."
        )
    del tp_rank, ep_rank  # placeholders for future multi-rank work

    text_cfg = _text_config(hf_config)
    vocab = int(text_cfg.vocab_size)
    hidden = int(text_cfg.hidden_size)
    num_layers = int(text_cfg.num_hidden_layers)
    num_heads = int(text_cfg.num_attention_heads)
    num_kv_heads = int(getattr(text_cfg, "num_key_value_heads", num_heads))
    head_dim = int(getattr(text_cfg, "head_dim", None) or (hidden // num_heads))
    intermediate = int(text_cfg.intermediate_size)
    tie_embed = bool(getattr(text_cfg, "tie_word_embeddings", False))
    qkv_bias = _has_qkv_bias(text_cfg)

    metas: list[ParameterShardMeta] = []

    def _add(name: str, shape: tuple[int, ...]) -> None:
        metas.append(
            ParameterShardMeta(
                param_name=name,
                full_shape=shape,
                dtype_str=dtype_str,
                ranges=tuple((0, s) for s in shape),
                global_rank=0,  # overridden by build_topology
                role="rollout",
            )
        )

    # 1. Embedding.
    _add("model.embed_tokens.weight", (vocab, hidden))

    # 2. Per-layer block.
    q_rows = num_heads * head_dim
    kv_rows = num_kv_heads * head_dim
    for layer_idx in range(num_layers):
        base = f"model.layers.{layer_idx}"
        _add(f"{base}.input_layernorm.weight", (hidden,))
        _add(f"{base}.self_attn.q_proj.weight", (q_rows, hidden))
        _add(f"{base}.self_attn.k_proj.weight", (kv_rows, hidden))
        _add(f"{base}.self_attn.v_proj.weight", (kv_rows, hidden))
        if qkv_bias:
            _add(f"{base}.self_attn.q_proj.bias", (q_rows,))
            _add(f"{base}.self_attn.k_proj.bias", (kv_rows,))
            _add(f"{base}.self_attn.v_proj.bias", (kv_rows,))
        _add(f"{base}.self_attn.o_proj.weight", (hidden, q_rows))
        _add(f"{base}.post_attention_layernorm.weight", (hidden,))
        _add(f"{base}.mlp.gate_proj.weight", (intermediate, hidden))
        _add(f"{base}.mlp.up_proj.weight", (intermediate, hidden))
        _add(f"{base}.mlp.down_proj.weight", (hidden, intermediate))

    # 3. Final norm + lm_head (lm_head absent under tie_word_embeddings).
    _add("model.norm.weight", (hidden,))
    if not tie_embed:
        _add("lm_head.weight", (vocab, hidden))

    return metas


__all__ = ["build_rollout_shard_metas"]
