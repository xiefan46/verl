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

Scope:

* Dense Qwen2 / Qwen3 family (attention bias auto-detected from
  ``attention_bias`` flag; Qwen3 has it disabled).
* MoE family with ``num_experts`` + per-expert ``gate_proj`` /
  ``up_proj`` / ``down_proj`` (Qwen3-30B-A3B layout). ETP > 1 (expert TP
  splitting each expert's intermediate dim) is NOT supported in MVP — we
  assume each EP rank holds each of its local experts in full.
* Single-stage rollout: no PP support here (PP > 1 would require slicing
  the layer-index range per rank).

Multi-rank partitioning we DO handle:

* ``tp_size``: TP split on attention q/k/v rows, o_proj cols, dense MLP
  gate_up rows + down_proj cols, embed_tokens / lm_head rows.
* ``ep_size``: EP split on routed experts — rank k holds globals
  ``[k*N_local, (k+1)*N_local)``. Router gate is replicated (full per rank).
"""

from __future__ import annotations

from typing import Any

from verl.checkpoint_engine.parallel_meta import ParameterShardMeta


def _text_config(hf_config: Any) -> Any:
    """VL configs nest the LM under ``text_config``; auto-resolve."""
    return getattr(hf_config, "text_config", hf_config)


def _has_qkv_bias(text_cfg: Any) -> bool:
    """Decide whether the family adds attention q_proj/k_proj/v_proj biases.

    Order of preference:
    1. Explicit ``attention_bias`` flag (Qwen3 sets this to False).
    2. Explicit ``qkv_bias`` flag (sometimes present).
    3. Family fallback (Qwen2 hardcodes True; Qwen3 hardcodes False).
    """
    if getattr(text_cfg, "attention_bias", None) is not None:
        return bool(text_cfg.attention_bias)
    if getattr(text_cfg, "qkv_bias", None) is not None:
        return bool(text_cfg.qkv_bias)
    arch_name = type(text_cfg).__name__
    if "Qwen3" in arch_name:
        return False
    if "Qwen2" in arch_name:
        return True
    return False


def _detect_moe(text_cfg: Any) -> tuple[int | None, int | None]:
    """Return ``(num_experts, moe_intermediate_size)`` or ``(None, None)``."""
    num_experts = getattr(text_cfg, "num_experts", None) or getattr(text_cfg, "num_local_experts", None)
    moe_interm = getattr(text_cfg, "moe_intermediate_size", None)
    return num_experts, moe_interm


def _has_qk_norm(text_cfg: Any) -> bool:
    """Detect whether the attention block applies RMSNorm on Q/K before rope.

    Qwen3 hardcodes True (its Megatron config exports ``qk_layernorm=True``).
    """
    for flag in ("use_qk_norm", "qk_norm", "qk_layernorm"):
        v = getattr(text_cfg, flag, None)
        if v is not None:
            return bool(v)
    arch_name = type(text_cfg).__name__
    return "Qwen3" in arch_name


def build_rollout_shard_metas(
    hf_config: Any,
    *,
    tp_rank: int = 0,
    tp_size: int = 1,
    ep_rank: int = 0,
    ep_size: int = 1,
    dtype_str: str = "bfloat16",
) -> list[ParameterShardMeta]:
    """Enumerate this rollout rank's HF-canonical param shape + slice.

    See module docstring for scope. ``global_rank`` on each meta is set to
    0 here and overridden by :meth:`ShardedNCCLCheckpointEngine.build_topology`
    when the NCCL group is wired.
    """
    if tp_size <= 0 or ep_size <= 0:
        raise ValueError(f"tp_size and ep_size must be positive (got {tp_size}, {ep_size})")
    if not (0 <= tp_rank < tp_size):
        raise ValueError(f"tp_rank {tp_rank} out of range [0, {tp_size})")
    if not (0 <= ep_rank < ep_size):
        raise ValueError(f"ep_rank {ep_rank} out of range [0, {ep_size})")

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
    qk_norm = _has_qk_norm(text_cfg)
    num_experts, moe_interm = _detect_moe(text_cfg)
    is_moe = bool(num_experts) and num_experts > 1 and moe_interm is not None
    mlp_only_layers = set(getattr(text_cfg, "mlp_only_layers", None) or [])

    # ---- Divisibility checks (raise rather than silently truncate) ----
    q_rows_full = num_heads * head_dim
    kv_rows_full = num_kv_heads * head_dim
    if q_rows_full % tp_size != 0:
        raise ValueError(f"q rows {q_rows_full} not divisible by tp_size {tp_size}")
    if num_kv_heads % tp_size != 0:
        raise NotImplementedError(
            f"num_kv_heads {num_kv_heads} not divisible by tp_size {tp_size}; "
            "MVP doesn't replicate KV heads across TP yet."
        )
    if intermediate % tp_size != 0:
        raise ValueError(f"intermediate_size {intermediate} not divisible by tp_size {tp_size}")
    if vocab % tp_size != 0:
        # vLLM pads vocab to multiple of TP; handle this case later
        raise NotImplementedError(
            f"vocab_size {vocab} not divisible by tp_size {tp_size}; vLLM padded-vocab case not modeled in MVP."
        )
    if is_moe and num_experts % ep_size != 0:
        raise ValueError(f"num_experts {num_experts} not divisible by ep_size {ep_size}")

    # ---- Per-rank local sizes ----
    q_local = q_rows_full // tp_size
    kv_local = kv_rows_full // tp_size
    interm_local = intermediate // tp_size
    vocab_local = vocab // tp_size
    n_local_experts = (num_experts // ep_size) if is_moe else 0

    # Helper to build a TP-split range tuple on a single dim.
    def _tp_range(full: int, local: int, dim_count: int, split_dim: int) -> tuple[tuple[int, int], ...]:
        ranges = []
        for d in range(dim_count):
            if d == split_dim:
                ranges.append((tp_rank * local, (tp_rank + 1) * local))
            else:
                # Placeholder; caller patches via `dims`.
                ranges.append((0, full))
        return tuple(ranges)

    metas: list[ParameterShardMeta] = []

    def _add(name: str, full_shape: tuple[int, ...], ranges: tuple[tuple[int, int], ...]) -> None:
        metas.append(
            ParameterShardMeta(
                param_name=name,
                full_shape=full_shape,
                dtype_str=dtype_str,
                ranges=ranges,
                global_rank=0,  # overridden by build_topology at NCCL setup
                role="rollout",
            )
        )

    def _add_full(name: str, full_shape: tuple[int, ...]) -> None:
        _add(name, full_shape, tuple((0, s) for s in full_shape))

    # --- Helpers for canonical Megatron→HF slice patterns ---

    def _col_split(name: str, full_shape: tuple[int, int], split_dim_size: int, split_local: int) -> None:
        """A 2-D weight TP-split on dim 0 (rows). Used for q/k/v/gate/up/embed/lm_head."""
        assert full_shape[0] == split_dim_size
        _add(
            name,
            full_shape,
            ((tp_rank * split_local, (tp_rank + 1) * split_local), (0, full_shape[1])),
        )

    def _row_split(name: str, full_shape: tuple[int, int], split_dim_size: int, split_local: int) -> None:
        """A 2-D weight TP-split on dim 1 (cols). Used for o_proj / down_proj."""
        assert full_shape[1] == split_dim_size
        _add(
            name,
            full_shape,
            ((0, full_shape[0]), (tp_rank * split_local, (tp_rank + 1) * split_local)),
        )

    def _bias_split(name: str, full: int, local: int) -> None:
        _add(name, (full,), ((tp_rank * local, (tp_rank + 1) * local),))

    # 1. Embedding (row TP-split).
    _col_split("model.embed_tokens.weight", (vocab, hidden), vocab, vocab_local)

    # 2. Per-layer.
    for layer_idx in range(num_layers):
        base = f"model.layers.{layer_idx}"
        _add_full(f"{base}.input_layernorm.weight", (hidden,))

        _col_split(f"{base}.self_attn.q_proj.weight", (q_rows_full, hidden), q_rows_full, q_local)
        _col_split(f"{base}.self_attn.k_proj.weight", (kv_rows_full, hidden), kv_rows_full, kv_local)
        _col_split(f"{base}.self_attn.v_proj.weight", (kv_rows_full, hidden), kv_rows_full, kv_local)
        if qkv_bias:
            _bias_split(f"{base}.self_attn.q_proj.bias", q_rows_full, q_local)
            _bias_split(f"{base}.self_attn.k_proj.bias", kv_rows_full, kv_local)
            _bias_split(f"{base}.self_attn.v_proj.bias", kv_rows_full, kv_local)
        if qk_norm:
            # Per-head RMSNorm on Q/K (Qwen3). head_dim-shaped, replicated.
            _add_full(f"{base}.self_attn.q_norm.weight", (head_dim,))
            _add_full(f"{base}.self_attn.k_norm.weight", (head_dim,))
        _row_split(f"{base}.self_attn.o_proj.weight", (hidden, q_rows_full), q_rows_full, q_local)

        _add_full(f"{base}.post_attention_layernorm.weight", (hidden,))

        layer_is_moe = is_moe and layer_idx not in mlp_only_layers
        if layer_is_moe:
            # Router weight: replicated (full per rank).
            _add_full(f"{base}.mlp.gate.weight", (num_experts, hidden))
            # Routed experts: this EP rank holds n_local_experts contiguous globals.
            # Each expert is held FULLY on this rank (no ETP split).
            for local_e in range(n_local_experts):
                global_e = ep_rank * n_local_experts + local_e
                _add_full(f"{base}.mlp.experts.{global_e}.gate_proj.weight", (moe_interm, hidden))
                _add_full(f"{base}.mlp.experts.{global_e}.up_proj.weight", (moe_interm, hidden))
                _add_full(f"{base}.mlp.experts.{global_e}.down_proj.weight", (hidden, moe_interm))
        else:
            # Dense MLP, TP-split.
            _col_split(f"{base}.mlp.gate_proj.weight", (intermediate, hidden), intermediate, interm_local)
            _col_split(f"{base}.mlp.up_proj.weight", (intermediate, hidden), intermediate, interm_local)
            _row_split(f"{base}.mlp.down_proj.weight", (hidden, intermediate), intermediate, interm_local)

    # 3. Final norm + lm_head.
    _add_full("model.norm.weight", (hidden,))
    if not tie_embed:
        _col_split("lm_head.weight", (vocab, hidden), vocab, vocab_local)

    return metas


__all__ = ["build_rollout_shard_metas"]
