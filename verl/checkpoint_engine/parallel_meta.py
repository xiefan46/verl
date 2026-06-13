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
"""Data classes for sharded-aware weight refit routing.

A ``ParameterShardMeta`` describes ONE shard of ONE parameter on ONE rank,
using a per-dim global-range box. MoE experts are modeled as independent
parameters (one ``ParameterShardMeta`` per expert), so the routing algorithm
does not need any special-case logic for the expert dimension.

A ``TransferEdge`` describes one P2P send/recv that the route table planner
emits. Edges carry the metadata the vLLM-side ``weight_loader`` needs:
``shard_id`` (``"w1"/"w2"/"w3"`` for MoE, ``"q"/"k"/"v"`` for fused QKV),
``expert_id`` (global expert index), and ``target_param_name`` (the vLLM
internal fused-tensor name).

All classes are JSON-serializable (via ``to_dict`` / ``from_dict``) so the
rank-0-computed routing plan can be broadcast to every worker via Ray RPC.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

# Role tags. "train" = trainer-side shard producer; "rollout" = inference-side
# shard receiver (the verl rollout-side CheckpointEngineWorker, NOT the vLLM
# worker subprocess — see design doc §5.5 two-hop architecture).
Role = Literal["train", "rollout"]


# A 1D inclusive-exclusive range [start, end) in the full-tensor coordinate
# system. Per dim, paired with full_shape, it captures which slab of the
# global tensor this rank holds.
GlobalRange = tuple[int, int]


def _slice_to_tuple(s: slice) -> tuple[int | None, int | None, int | None]:
    """Encode a slice as a JSON-friendly tuple."""
    return (s.start, s.stop, s.step)


def _tuple_to_slice(t: tuple[int | None, int | None, int | None]) -> slice:
    return slice(t[0], t[1], t[2])


@dataclass(frozen=True)
class ParameterShardMeta:
    """One shard of one parameter on one rank.

    For MoE, each expert is treated as an independent parameter. The
    ``param_name`` carries the global expert index, e.g.
    ``"model.layers.0.mlp.experts.42.gate_proj.weight"``. This means the
    routing-plan algorithm (`transfer_plan.py`) treats expert/TP/EP uniformly
    as orthogonal dimensions of an N-D box, and the expert-dim split is
    naturally encoded by which ranks emit which per-expert metas.

    Args:
        param_name: HF-canonical parameter name (unfused, one expert per name
            for MoE). Examples:
              - dense:  ``"model.layers.0.self_attn.q_proj.weight"``
              - MoE:    ``"model.layers.0.mlp.experts.3.gate_proj.weight"``
        full_shape: Logical shape of the FULL parameter in HF layout
            (un-sharded, un-fused), e.g. ``(num_heads * head_dim, hidden)``
            for an attention projection or ``(intermediate, hidden)`` for a
            per-expert ``gate_proj``.
        dtype_str: PyTorch dtype name (e.g. ``"bfloat16"``). String for
            JSON-serializability.
        ranges: Per-dim ``(start, end)`` in the FULL_shape coordinate. Tuple
            of length ``len(full_shape)``. ``end - start`` gives this rank's
            local size on that dim.
        global_rank: Rank in the joint trainer+rollout NCCL group (the same
            group used by the existing broadcast backend, see
            ``nccl_checkpoint_engine.py``).
        role: ``"train"`` or ``"rollout"``.
    """

    param_name: str
    full_shape: tuple[int, ...]
    dtype_str: str
    ranges: tuple[GlobalRange, ...]
    global_rank: int
    role: Role

    def __post_init__(self) -> None:
        if len(self.ranges) != len(self.full_shape):
            raise ValueError(
                f"ranges length {len(self.ranges)} must match full_shape length "
                f"{len(self.full_shape)} (param={self.param_name})"
            )
        for dim, ((start, end), size) in enumerate(zip(self.ranges, self.full_shape, strict=True)):
            if not (0 <= start < end <= size):
                raise ValueError(
                    f"invalid range on dim {dim} for {self.param_name}: ({start}, {end}) not in [0, {size}]"
                )

    def local_shape(self) -> tuple[int, ...]:
        """Shape of this rank's local tensor for this param."""
        return tuple(end - start for start, end in self.ranges)

    def to_dict(self) -> dict[str, Any]:
        return {
            "param_name": self.param_name,
            "full_shape": list(self.full_shape),
            "dtype_str": self.dtype_str,
            "ranges": [list(r) for r in self.ranges],
            "global_rank": self.global_rank,
            "role": self.role,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ParameterShardMeta:
        return cls(
            param_name=d["param_name"],
            full_shape=tuple(d["full_shape"]),
            dtype_str=d["dtype_str"],
            ranges=tuple(tuple(r) for r in d["ranges"]),  # type: ignore[arg-type]
            global_rank=int(d["global_rank"]),
            role=d["role"],
        )


@dataclass(frozen=True)
class TransferEdge:
    """One P2P send/recv operation in the routing plan.

    Each edge represents: rank ``src_global_rank`` sends the slice
    ``src_local_slice`` of its local tensor for ``param_name`` to rank
    ``dst_global_rank``, which stores it at ``dst_local_slice`` of its
    local receive buffer (which itself maps into a vLLM fused tensor
    via ``target_param_name`` / ``shard_id`` / ``expert_id``).

    The vLLM-side caller will eventually invoke::

        model.experts.weight_loader(
            param=lookup(target_param_name),
            loaded_weight=recv_buffer,
            weight_name=param_name,       # HF canonical
            shard_id=shard_id,             # "w1"/"w2"/"w3" or "q"/"k"/"v"
            expert_id=expert_id,           # int for MoE, None for dense
        )

    Args:
        param_name: HF-canonical source param name (same as the meta).
        src_global_rank: Sender's rank in the joint NCCL group.
        dst_global_rank: Receiver's rank in the joint NCCL group.
        src_local_slice: Encoded as ``(start, stop, step)`` per dim (because
            ``slice`` itself is not JSON-serializable). Apply via
            ``tensor[edge.src_local_slice()]`` after decoding.
        dst_local_slice: Same encoding for the receiver's buffer.
        shape: Shape of the actual data being transferred (post-slice).
        dtype_str: dtype name.
        target_param_name: vLLM-side fused parameter name where this shard
            ultimately lands (e.g. ``"model.layers.0.mlp.experts.w13_weight"``).
            ``None`` for dense models with no special packing.
        shard_id: ``"w1"`` (MoE gate), ``"w2"`` (MoE down), ``"w3"`` (MoE up),
            ``"q"/"k"/"v"`` (fused QKV), or ``None`` for 1:1 mappings.
        expert_id: Global expert index for MoE (matches the integer parsed
            from the HF name ``...experts.{i}...``), or ``None`` for dense.
    """

    param_name: str
    src_global_rank: int
    dst_global_rank: int
    src_local_slice_encoded: tuple[tuple[int | None, int | None, int | None], ...]
    dst_local_slice_encoded: tuple[tuple[int | None, int | None, int | None], ...]
    shape: tuple[int, ...]
    dtype_str: str
    target_param_name: str | None = None
    # ``shard_id`` accepts both strings ("q"/"k"/"v" for QKV, "w1"/"w2"/"w3"
    # for MoE) and ints (0/1 for ``MergedColumnParallelLinear``'s
    # gate_proj/up_proj fusion). JSON round-trips both cleanly.
    shard_id: str | int | None = None
    expert_id: int | None = None

    def src_local_slice(self) -> tuple[slice, ...]:
        return tuple(_tuple_to_slice(t) for t in self.src_local_slice_encoded)

    def dst_local_slice(self) -> tuple[slice, ...]:
        return tuple(_tuple_to_slice(t) for t in self.dst_local_slice_encoded)

    def num_bytes(self) -> int:
        """Bytes this edge transfers. Used for load-balancing send-bytes."""
        elem = _DTYPE_SIZE_BYTES.get(self.dtype_str)
        if elem is None:
            raise ValueError(f"unknown dtype_str: {self.dtype_str}")
        n = 1
        for d in self.shape:
            n *= d
        return n * elem

    def to_dict(self) -> dict[str, Any]:
        return {
            "param_name": self.param_name,
            "src_global_rank": self.src_global_rank,
            "dst_global_rank": self.dst_global_rank,
            "src_local_slice": [list(t) for t in self.src_local_slice_encoded],
            "dst_local_slice": [list(t) for t in self.dst_local_slice_encoded],
            "shape": list(self.shape),
            "dtype_str": self.dtype_str,
            "target_param_name": self.target_param_name,
            "shard_id": self.shard_id,
            "expert_id": self.expert_id,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TransferEdge:
        return cls(
            param_name=d["param_name"],
            src_global_rank=int(d["src_global_rank"]),
            dst_global_rank=int(d["dst_global_rank"]),
            src_local_slice_encoded=tuple(tuple(t) for t in d["src_local_slice"]),  # type: ignore[arg-type]
            dst_local_slice_encoded=tuple(tuple(t) for t in d["dst_local_slice"]),  # type: ignore[arg-type]
            shape=tuple(d["shape"]),
            dtype_str=d["dtype_str"],
            target_param_name=d.get("target_param_name"),
            shard_id=d.get("shard_id"),
            expert_id=d.get("expert_id"),
        )


@dataclass(frozen=True)
class TransferPlan:
    """Result of ``build_transfer_plan``: the list of edges + a content hash.

    Stored as a single broadcastable object so every worker can hash the plan
    and ``all_reduce`` the hashes as a sanity check (cheap insurance against
    accidental nondeterminism in plan construction).
    """

    edges: tuple[TransferEdge, ...]
    plan_hash: str = field(default="")  # populated by __post_init__

    def __post_init__(self) -> None:
        if not self.plan_hash:
            # frozen dataclass workaround
            object.__setattr__(self, "plan_hash", self._compute_hash())

    def _compute_hash(self) -> str:
        import hashlib

        # Edges are already in a deterministic order from build_transfer_plan;
        # we hash the canonical JSON for stability across Python versions.
        canonical = json.dumps([e.to_dict() for e in self.edges], sort_keys=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def edges_for_src(self, rank: int) -> list[TransferEdge]:
        return [e for e in self.edges if e.src_global_rank == rank]

    def edges_for_dst(self, rank: int) -> list[TransferEdge]:
        return [e for e in self.edges if e.dst_global_rank == rank]

    def to_json(self) -> str:
        return json.dumps(
            {
                "edges": [e.to_dict() for e in self.edges],
                "plan_hash": self.plan_hash,
            },
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, payload: str) -> TransferPlan:
        d = json.loads(payload)
        edges = tuple(TransferEdge.from_dict(e) for e in d["edges"])
        # Pass plan_hash explicitly so __post_init__ trusts it; we still
        # could re-validate by recomputing.
        return cls(edges=edges, plan_hash=d.get("plan_hash", ""))


# dtype name → bytes/element. Kept narrow on purpose; expand as needed.
_DTYPE_SIZE_BYTES: dict[str, int] = {
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
    "uint8": 1,
    "int8": 1,
    "float8_e4m3fn": 1,
    "float8_e5m2": 1,
}
