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

"""MagiAttention-backed tree training: HF + FSDP2 integration layer.

Stage 2 of the tree training pipeline (see
``research/2026-05-16-magi-integration-plan-v2.md`` §1). Provides:

* :class:`TreeCPContext` — manages the context-parallel process group and
  propagates ``cp_group`` onto attention sub-modules so the registered Magi
  attention forward can find it. V1 supports ``cp_size=1`` (single-rank cp_group
  or None); V3 will enable ``cp_size>1`` by extending verl FSDP wrap to a 2D
  ``(dp, cp)`` mesh.

* :func:`tree_attn_scope` — per-microbatch context manager that builds Magi's
  ``magi_attn_flex_key`` runtime key (which auto-registers globally keyed by
  ``cp_group``) and yields it for the caller's ``dispatch`` / ``undispatch``.

* :func:`_magi_tree_attention_forward` — the function registered into HF's
  ``ALL_ATTENTION_FUNCTIONS`` under the name ``"Magi_Tree_Attention"``. When a
  registered key is available for ``module.cp_group``, it routes through
  ``calc_attn`` (CP-aware); otherwise it falls back to ``flash_attention_2``.

* :func:`register_tree_attention` — idempotent registration.

This module imports ``magi_attention`` lazily (inside function bodies) so it can
be loaded on CPU-only machines without MagiAttention installed. Live forward
calls obviously require ``magi_attention`` on the GPU host.
"""

from __future__ import annotations

import contextlib
from typing import Any, Iterator, Sequence

import torch
import torch.distributed as dist

from verl.utils.device import get_device_name

__all__ = [
    "TreeCPContext",
    "tree_attn_scope",
    "register_tree_attention",
]

# Module-level flag — set once ``register_tree_attention`` succeeds. The HF
# ``ALL_ATTENTION_FUNCTIONS`` registry itself is also a global; this flag is
# the local idempotency guard so we don't redundantly call ``.register`` (HF
# may or may not be tolerant; cheap to guard).
_REGISTERED: bool = False


class TreeCPContext:
    """Manages the context-parallel process group for tree training.

    The Magi-registered attention forward reads its runtime key from
    ``module.cp_group`` (a Magi-side convention from
    ``examples/transformers/magi_attention_func.py``). At engine init time we
    build the cp_group once and propagate it to every attention sub-module
    before the model is FSDP-wrapped (FSDP2 ``fully_shard`` does not strip
    attribute assignments).

    V1: ``cp_size=1`` (single-rank cp_group); ``dispatch``/``undispatch`` are
    effectively pad-only no-ops at cp=1.
    V3: ``cp_size>1`` — real CP group on a ``(dp, cp)`` 2D mesh, with
    sequence sharding via ``dispatch``.

    Notes
    -----
    In a single-process (e.g. unit test) environment where
    ``torch.distributed`` is not initialized, ``self.cp_group`` is ``None``.
    Downstream code paths must handle that gracefully (the registered Magi
    forward falls back to ``flash_attention_2``).
    """

    def __init__(self, cp_size: int = 1, device_type: str | None = None):
        """
        Parameters
        ----------
        cp_size
            Number of CP ranks. V1 supports 1.
        device_type
            Device type for the DeviceMesh. If ``None`` (default), resolves at
            runtime via ``verl.utils.device.get_device_name()`` (returns the
            active accelerator name: cuda / npu / cpu).
        """
        if cp_size < 1:
            raise ValueError(f"cp_size must be >= 1, got {cp_size}")
        self.cp_size = cp_size
        self.device_type = device_type if device_type is not None else get_device_name()
        self.cp_group = self._build_cp_group()

    def _build_cp_group(self):
        """Build the cp_group. Returns None when torch.distributed is not init."""
        if not dist.is_initialized():
            # Single-process mode — typical for unit tests on CPU.
            return None

        world_size = dist.get_world_size()
        if world_size % self.cp_size != 0:
            raise ValueError(f"world_size={world_size} not divisible by tree_cp_size={self.cp_size}")

        # 2D mesh (dp, cp). FSDP wrap topology is verl's responsibility; we
        # only need the cp dim for Magi's mask sharding.
        from torch.distributed.device_mesh import DeviceMesh

        dp_size = world_size // self.cp_size
        mesh = DeviceMesh(
            device_type=self.device_type,
            mesh=torch.arange(world_size).reshape(dp_size, self.cp_size),
            mesh_dim_names=("dp", "cp"),
        )
        return mesh.get_group("cp")

    def setup_model(self, model: torch.nn.Module) -> None:
        """Walk ``model.modules()`` and attach ``cp_group`` to attention layers.

        Also flips ``model.config._attn_implementation`` to ``"Magi_Tree_Attention"``
        so HuggingFace dispatches each attention layer's forward through the
        registered Magi function.

        Idempotent: safe to call multiple times. The cp_group attribute assignment
        survives FSDP2 ``fully_shard`` because FSDP2 preserves user attributes on
        sharded modules.
        """
        for module in model.modules():
            if "Attention" in type(module).__name__:
                module.cp_group = self.cp_group

        if hasattr(model, "config"):
            # HF reads ``model.config._attn_implementation`` to pick the dispatch
            # function from ``ALL_ATTENTION_FUNCTIONS``. The attribute is normally
            # set at ``from_pretrained`` time; mutating it post-load works as
            # well because HF resolves the function at each forward call.
            model.config._attn_implementation = "Magi_Tree_Attention"


@contextlib.contextmanager
def tree_attn_scope(
    q_ranges_naive: Sequence[tuple[int, int]],
    k_ranges_naive: Sequence[tuple[int, int]],
    attn_type_map_list: Sequence[int],
    *,
    total_seqlen: int,
    num_heads_q: int,
    num_heads_kv: int,
    head_dim: int,
    cp_group: Any,
    chunk_size: int = 512,
) -> Iterator[Any]:
    """Per-microbatch: build the Magi runtime key and yield it for dispatch use.

    Side effect: registers the runtime key into Magi's internal
    ``dist_attn_runtime_dict_mgr`` keyed by ``cp_group``. The registered HF
    forward retrieves it via ``get_most_recent_key(module.cp_group)``.

    Parameters
    ----------
    q_ranges_naive, k_ranges_naive
        Output of :func:`_magi_kernel.build_attn_ranges_from_trie` —
        ``list[(start, end)]`` half-open ranges.
    attn_type_map_list
        Integer codes for each tile (``ATTN_TYPE_FULL=0`` / ``ATTN_TYPE_CAUSAL=1``).
    total_seqlen
        Packed sequence length (before Magi's chunk-size padding).
    num_heads_q, num_heads_kv, head_dim
        HF model attention shape. ``num_heads_kv`` may differ from
        ``num_heads_q`` for GQA.
    cp_group
        From :class:`TreeCPContext`. May be ``None`` in single-process tests
        — the runtime key construction still works at cp_size=1.
    chunk_size
        Magi's chunk-size alignment unit. 512 matches Magi's torch_native
        example default.

    Yields
    ------
    The Magi runtime key. Caller uses it for ``dispatch(input, key)`` and
    ``undispatch(output, key)``; the registered Magi forward retrieves the
    same key via ``get_most_recent_key(cp_group)`` and does not require the
    key to be threaded through HF model forward kwargs.
    """
    # Lazy import: keep module CPU-importable without magi_attention installed.
    from magi_attention.api import (
        AttnMaskType,
        AttnRanges,
        DistAttnConfig,
        compute_pad_size,
        magi_attn_flex_key,
    )

    q_ranges = AttnRanges.from_ranges(list(q_ranges_naive))
    k_ranges = AttnRanges.from_ranges(list(k_ranges_naive))
    attn_mask_type = [AttnMaskType.from_int_type(int(t)) for t in attn_type_map_list]

    if cp_group is not None and dist.is_initialized():
        cp_size = dist.get_world_size(cp_group)
    else:
        cp_size = 1
    pad_size = compute_pad_size(total_seqlen, cp_size, chunk_size)

    key = magi_attn_flex_key(
        q_ranges=q_ranges,
        k_ranges=k_ranges,
        attn_mask_type=attn_mask_type,
        total_seqlen_q=total_seqlen,
        total_seqlen_k=total_seqlen,
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        head_dim=head_dim,
        pad_size=pad_size,
        chunk_size=chunk_size,
        cp_group_or_mesh=cp_group,
        dist_attn_config=DistAttnConfig(),
    )

    try:
        yield key
    finally:
        # Magi's runtime key registry uses "most recent" semantics and is
        # auto-evicted when the next key for the same cp_group is registered.
        # No explicit cleanup needed at scope exit.
        pass


def _magi_tree_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Any,
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Any,
):
    """HF-registered attention forward for tree training.

    Routes to MagiAttention's ``calc_attn`` when a runtime key is available
    for ``module.cp_group``; otherwise delegates to ``flash_attention_2`` so
    non-tree-training forwards on the same model (e.g. evaluation) keep working.

    Signature follows HF's ``ALL_ATTENTION_FUNCTIONS`` contract.

    HF Layout in: ``(batch=1, num_heads, seq_len, head_dim)``.
    Magi layout: ``(num_tokens, num_heads, head_dim)``. We permute on entry
    and exit.
    """
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    def _fa2_fallback():
        # HF's flash_attention_forward reads ``attn_implementation`` from
        # ``module.config._attn_implementation`` and passes it to a lazy kernel
        # loader. Our config is set to "Magi_Tree_Attention" so the loader will
        # crash (no such kernel on the hub). Temporarily flip config to
        # "flash_attention_2" for the duration of the call. Single-threaded
        # forward so no race window matters in practice.
        #
        # Pass scaling/dropout as keywords — HF's flash_attention_forward
        # signature is (module, q, k, v, attention_mask, scaling, softmax_scale,
        # sliding_window, softcap, is_causal, dropout, **kwargs). Positional
        # 7th would land in softmax_scale and zero out attention -> NaN.
        config_obj = getattr(module, "config", None)
        orig_impl = getattr(config_obj, "_attn_implementation", None) if config_obj else None
        if config_obj is not None:
            config_obj._attn_implementation = "flash_attention_2"
        try:
            return ALL_ATTENTION_FUNCTIONS["flash_attention_2"](
                module,
                query,
                key,
                value,
                attention_mask,
                scaling=scaling,
                dropout=dropout,
                **kwargs,
            )
        finally:
            if config_obj is not None and orig_impl is not None:
                config_obj._attn_implementation = orig_impl

    cp_group = getattr(module, "cp_group", None)
    if cp_group is None:
        # Module was not set up for tree training. Fall through.
        return _fa2_fallback()

    # Lazy import: avoid making this module require magi_attention at load time.
    from einops import rearrange
    from magi_attention.api import calc_attn, get_most_recent_key

    magi_attn_key = get_most_recent_key(cp_group)
    if magi_attn_key is None:
        # cp_group is set but no key registered for this forward — non-tree call.
        return _fa2_fallback()

    orig_dtype = query.dtype

    # HF (1, H, S, D) -> Magi (S, H, D). FFA currently accepts fp16/bf16 only.
    q = rearrange(query, "1 h s d -> s h d").contiguous().to(torch.bfloat16)
    k = rearrange(key, "1 h s d -> s h d").contiguous().to(torch.bfloat16)
    v = rearrange(value, "1 h s d -> s h d").contiguous().to(torch.bfloat16)

    out = calc_attn(q, k, v, magi_attn_key)[0]

    # Magi (S, H, D) -> HF expected (1, S, H*D).
    out = rearrange(out, "s h d -> 1 s (h d)").to(orig_dtype)

    # HF attention forwards return (output, attn_weights). We do not produce
    # attn_weights from FFA; downstream code that requested
    # ``output_attentions=True`` will get None and should handle it.
    return out, None


def register_tree_attention() -> None:
    """Idempotent registration of ``"Magi_Tree_Attention"``.

    Safe to call multiple times across multiple engine inits in the same
    process. Lazy imports ``transformers.modeling_utils`` so the module is
    CPU-loadable without the heavy HF stack being importable at module load.
    """
    global _REGISTERED
    if _REGISTERED:
        return

    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    ALL_ATTENTION_FUNCTIONS.register("Magi_Tree_Attention", _magi_tree_attention_forward)
    _REGISTERED = True
