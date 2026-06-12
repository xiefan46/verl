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
"""Sharded-aware NCCL checkpoint engine.

This backend implements the two-hop design from the sharded-aware weight refit
proposal (see ``research/2026-06-11-sharded-aware-weight-refit-design.md``):

1. ``ShardedNCCLCheckpointEngine`` runs on **both** trainer and rollout-side
   verl Ray actors (``CheckpointEngineWorker``). They join one shared
   ``ray.util.collective`` NCCL group (same group as the existing broadcast
   backend, see ``nccl_checkpoint_engine.py``).
2. Instead of ``collective.broadcast`` (rank 0 → all), each trainer rank
   does ``collective.send`` per outgoing ``TransferEdge``, and each
   rollout-side rank does ``collective.recv`` per incoming edge.
3. The routing plan is computed once at ``build_topology`` time and
   distributed to every worker via ``init_process_group`` kwargs.

The "second hop" (verl rollout-side actor → vLLM worker subprocess) reuses
``BucketedWeightSender / BucketedWeightReceiver`` over ZMQ + CUDA IPC; that
piece lives in ``vllm_rollout/utils.py`` and is wired up by M3/M4.

MVP simplifications (will be revisited in later milestones):
- No bucket packing on the NCCL hop — each TransferEdge is one send/recv.
  Bucket optimization (analogous to ``BroadcastOperation``) can land later
  without API changes.
- ``send_weights`` collects the full ``{name: tensor}`` map before sending.
  Acceptable for MVP because the trainer already materializes per-tensor
  weights via ``engine.get_local_shards()``; streaming send is a future opt.
- Shard metadata is injected via ``set_shard_metas`` (an MVP method outside
  the ABC). Future work: extend the ABC with a clean shard-meta channel.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import AsyncGenerator, Generator
from dataclasses import dataclass
from typing import Any

import torch

# ``ray.util.collective`` is imported lazily inside the methods that actually
# need it. This keeps CPU-only unit tests (which exercise prepare /
# build_topology / plan logic) free of the ray dependency.
from verl.checkpoint_engine.base import CheckpointEngine, CheckpointEngineRegistry
from verl.checkpoint_engine.parallel_meta import ParameterShardMeta, TransferEdge, TransferPlan
from verl.checkpoint_engine.transfer_plan import build_transfer_plan, sanity_check_cross_rank

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@dataclass
class ShardedNCCLPrepareInfo:
    """Per-worker payload returned by ``prepare()``.

    Aggregated across all workers by ``build_topology`` to build the routing plan.

    The classmethod ``build_topology`` expects the input metadata list to be
    ordered as ``[trainer_0, trainer_1, ..., rollout_0, rollout_1, ...]``,
    which is how ``CheckpointEngineManager.build_process_group`` already
    arranges things (trainer.execute_checkpoint_engine + rollout.execute_*).
    """

    shard_metas: list[dict[str, Any]]  # list of ParameterShardMeta.to_dict()
    role: str  # "train" or "rollout"


@CheckpointEngineRegistry.register("sharded_nccl")
class ShardedNCCLCheckpointEngine(CheckpointEngine):
    """NCCL-based sharded-aware checkpoint engine.

    Unlike ``NCCLCheckpointEngine`` which uses ``collective.broadcast``
    (rank-0 sends, all rollout ranks receive identical data), this backend
    routes weight shards P2P based on a routing plan computed at startup.

    Args:
        group_name: NCCL collective group name; shared with the existing
            broadcast backend lifecycle if reused.
        rebuild_group: Whether ``finalize`` should destroy the group (mirrors
            the same flag on ``NCCLCheckpointEngine`` for symmetry).
        rollout_dtype: Expected dtype on the rollout side. Used for receive
            buffer allocation when src dtype differs (rare; FP8 future work).
    """

    def __init__(
        self,
        group_name: str = "sharded_nccl",
        rebuild_group: bool = False,
        rollout_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.group_name = group_name
        self.rebuild_group = rebuild_group
        self.rollout_dtype = rollout_dtype

        # Injected by ``set_shard_metas`` before ``prepare``.
        self._shard_metas: list[ParameterShardMeta] = []
        self._role: str | None = None

        # Filled in ``init_process_group``.
        self.rank: int | None = None
        self.world_size: int | None = None
        self._plan: TransferPlan | None = None

        # Cached view of edges relevant to this rank — built lazily on first use.
        self._outgoing_edges: list[TransferEdge] | None = None
        self._incoming_edges: list[TransferEdge] | None = None

    # ------------------------------------------------------------------
    # MVP-only API: shard meta injection. Outside the ABC.
    # ------------------------------------------------------------------

    def set_shard_metas(self, metas: list[ParameterShardMeta], role: str) -> None:
        """Inject this rank's shard metadata BEFORE ``prepare()``.

        Args:
            metas: This rank's list of per-parameter shard descriptors.
            role: ``"train"`` or ``"rollout"`` — must match every meta's
                ``role`` field.
        """
        for m in metas:
            if m.role != role:
                raise ValueError(f"meta role {m.role!r} does not match engine role {role!r}: {m}")
        self._shard_metas = list(metas)
        self._role = role

    # ------------------------------------------------------------------
    # CheckpointEngine ABC implementation
    # ------------------------------------------------------------------

    def prepare(self) -> dict[str, Any]:
        if self._role is None:
            raise RuntimeError(
                "ShardedNCCLCheckpointEngine.prepare() called before set_shard_metas(). "
                "Call set_shard_metas(metas, role) first."
            )
        return ShardedNCCLPrepareInfo(
            shard_metas=[m.to_dict() for m in self._shard_metas],
            role=self._role,
        ).__dict__

    @classmethod
    def build_topology(
        cls,
        trainer_world_size: int,
        rollout_world_size: int,
        metadata: list[dict],
    ) -> tuple[dict[str, list[Any]], dict[str, list[Any]]]:
        """Aggregate per-worker shard metas and compute the routing plan.

        Layout assumed for ``metadata``:
            ``[trainer_0, trainer_1, ..., rollout_0, rollout_1, ...]``

        Assigns NCCL group ranks: trainers get ``[0, trainer_ws)`` and
        rollout-side actors get ``[trainer_ws, trainer_ws + rollout_ws)``.

        Returns ``(trainer_kwargs, rollout_kwargs)`` where each kwarg dict
        is a list-of-lists keyed by ``rank`` / ``world_size`` / ``plan_json``,
        dispatched element-wise to each worker's ``init_process_group``.
        """
        if len(metadata) != trainer_world_size + rollout_world_size:
            raise ValueError(
                f"metadata length {len(metadata)} != trainer_world_size + rollout_world_size "
                f"= {trainer_world_size + rollout_world_size}"
            )

        train_metas: list[ParameterShardMeta] = []
        rollout_metas: list[ParameterShardMeta] = []
        for i, info_dict in enumerate(metadata):
            assigned_rank = i  # trainers 0..T-1, rollout T..T+R-1
            role_expected = "train" if i < trainer_world_size else "rollout"
            info_role = info_dict.get("role")
            if info_role != role_expected:
                raise ValueError(f"metadata[{i}] role {info_role!r} != expected {role_expected!r}")
            target_list = train_metas if role_expected == "train" else rollout_metas
            for m_dict in info_dict.get("shard_metas", []):
                m = ParameterShardMeta.from_dict(m_dict)
                # Override global_rank with the freshly-assigned group rank,
                # so downstream routing speaks the NCCL group's rank space.
                m = ParameterShardMeta(
                    param_name=m.param_name,
                    full_shape=m.full_shape,
                    dtype_str=m.dtype_str,
                    ranges=m.ranges,
                    global_rank=assigned_rank,
                    role=m.role,
                )
                target_list.append(m)

        plan = build_transfer_plan(train_metas, rollout_metas)
        plan_json = plan.to_json()

        total_ws = trainer_world_size + rollout_world_size
        trainer_kwargs = {
            "rank": list(range(trainer_world_size)),
            "world_size": [total_ws] * trainer_world_size,
            "transfer_plan_json": [plan_json] * trainer_world_size,
        }
        rollout_kwargs = {
            "rank": list(range(trainer_world_size, total_ws)),
            "world_size": [total_ws] * rollout_world_size,
            "transfer_plan_json": [plan_json] * rollout_world_size,
        }
        return trainer_kwargs, rollout_kwargs

    def init_process_group(
        self,
        rank: int,
        world_size: int,
        transfer_plan_json: str,
    ) -> None:
        self.rank = rank
        self.world_size = world_size
        self._plan = TransferPlan.from_json(transfer_plan_json)
        self._outgoing_edges = self._plan.edges_for_src(rank)
        self._incoming_edges = self._plan.edges_for_dst(rank)

        import ray.util.collective as collective

        if self.rebuild_group or not collective.is_group_initialized(self.group_name):
            collective.init_collective_group(world_size, rank, "nccl", self.group_name)
        collective.barrier(self.group_name)
        logger.info(
            f"sharded_nccl init rank={rank} world_size={world_size} "
            f"out_edges={len(self._outgoing_edges)} in_edges={len(self._incoming_edges)} "
            f"plan_hash={self._plan.plan_hash[:16]}"
        )

    def finalize(self) -> None:
        if self.rebuild_group and self.rank is not None and self.rank >= 0:
            import ray.util.collective as collective

            collective.destroy_collective_group(self.group_name)
        self.rank = None
        self.world_size = None
        self._plan = None
        self._outgoing_edges = None
        self._incoming_edges = None
        torch.cuda.empty_cache()

    @torch.no_grad()
    async def send_weights(
        self,
        weights: Generator[tuple[str, torch.Tensor], None, None],
        global_steps: int | None = None,
    ) -> None:
        """Send shards P2P according to the routing plan.

        ``weights`` yields ``(param_name, tensor)`` where ``tensor`` is this
        rank's LOCAL shard (not the full tensor — this is the contract
        between trainer engine's ``get_local_shards`` and the sharded backend).
        For MVP we collect into a dict before sending; streaming send is
        future work.
        """
        if self._outgoing_edges is None:
            raise RuntimeError("send_weights called before init_process_group")

        # Collect: {param_name: local_tensor}. We may need each tensor for
        # multiple outgoing edges (e.g. same shard going to multiple infer ranks).
        by_name: dict[str, torch.Tensor] = {}
        async for item in _to_async(weights):
            name, tensor = item
            by_name[name] = tensor

        import ray.util.collective as collective

        start_time = time.time()
        for edge in self._outgoing_edges:
            tensor = by_name.get(edge.param_name)
            if tensor is None:
                # This is a real error: routing plan said we'd send this but
                # weights generator didn't include it. Likely a `get_local_shards`
                # bug.
                raise KeyError(
                    f"sharded send: edge requires param {edge.param_name!r} but "
                    f"weights generator did not yield it (rank={self.rank})"
                )
            shard = tensor[edge.src_local_slice()].contiguous()
            collective.send(shard, edge.dst_global_rank, self.group_name)

        logger.info(
            f"sharded send rank={self.rank} edges={len(self._outgoing_edges)} time={time.time() - start_time:.2f}s"
        )

    @torch.no_grad()
    async def receive_weights(
        self,
        global_steps: int | None = None,
    ) -> AsyncGenerator[tuple[str, torch.Tensor], None]:
        """Receive shards P2P and yield ``(encoded_name, tensor)`` pairs.

        ``encoded_name`` encodes the edge metadata as a JSON string so the
        downstream consumer (the vLLM-side bridge in
        ``VllmWorkerExtension.update_weights_from_sharded_ipc``) can recover
        ``target_param_name`` / ``shard_id`` / ``expert_id`` and call vLLM's
        ``weight_loader`` correctly. Format::

            "<param_name>|<edge_index>"

        (the receiver looks up edge metadata by ``edge_index`` from a separate
        ``get_incoming_edges()`` call). This keeps the ABC signature intact
        while threading sharded metadata through.
        """
        if self._incoming_edges is None:
            raise RuntimeError("receive_weights called before init_process_group")

        import ray.util.collective as collective

        start_time = time.time()
        for idx, edge in enumerate(self._incoming_edges):
            buf = torch.empty(edge.shape, dtype=_str_to_dtype(edge.dtype_str), device="cuda")
            collective.recv(buf, edge.src_global_rank, self.group_name)
            encoded_name = f"{edge.param_name}|{idx}"
            yield encoded_name, buf

        logger.info(
            f"sharded recv rank={self.rank} edges={len(self._incoming_edges)} time={time.time() - start_time:.2f}s"
        )

    # ------------------------------------------------------------------
    # MVP-only helpers
    # ------------------------------------------------------------------

    def get_incoming_edges(self) -> list[TransferEdge]:
        """Return this rank's incoming edges in the same order as ``receive_weights`` yields."""
        if self._incoming_edges is None:
            raise RuntimeError("get_incoming_edges called before init_process_group")
        return list(self._incoming_edges)

    def plan_hash(self) -> str:
        """Expose plan hash for cross-rank sanity checking."""
        if self._plan is None:
            raise RuntimeError("plan_hash called before init_process_group")
        return self._plan.plan_hash

    def verify_cross_rank_plan(self, all_hashes_from_workers: list[str]) -> None:
        """Sanity-check that every worker computed the same plan_hash."""
        if self._plan is None:
            raise RuntimeError("verify_cross_rank_plan called before init_process_group")
        sanity_check_cross_rank(self._plan.plan_hash, all_hashes_from_workers)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


_DTYPE_STR_MAP: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "uint8": torch.uint8,
    "int8": torch.int8,
}


def _str_to_dtype(s: str) -> torch.dtype:
    dt = _DTYPE_STR_MAP.get(s)
    if dt is None:
        raise ValueError(f"unknown dtype_str: {s!r}")
    return dt


async def _to_async(gen):
    """Adapt a sync generator to async without blocking the event loop semantics.

    The current weight-export path is sync (``engine.get_local_shards``); we
    yield items lazily so ``async for`` works naturally.
    """
    for item in gen:
        yield item
