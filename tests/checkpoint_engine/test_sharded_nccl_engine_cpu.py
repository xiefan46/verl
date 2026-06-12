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
"""Pure-CPU unit tests for ``ShardedNCCLCheckpointEngine``.

Exercises everything that does NOT need a live NCCL group:
- prepare()'s metadata payload
- build_topology()'s rank assignment + plan computation
- Cross-rank plan_hash consistency (every "worker"'s plan_json matches)
- Error paths (role mismatch, metadata length mismatch, missing set_shard_metas)

The GPU echo test that exercises real ``collective.send/recv`` lives in
``test_sharded_nccl_gpu_smoke.py`` and runs on RunPod (4×H100).
"""

from __future__ import annotations

import pytest

from verl.checkpoint_engine.parallel_meta import ParameterShardMeta, TransferPlan
from verl.checkpoint_engine.sharded_nccl_checkpoint_engine import ShardedNCCLCheckpointEngine


def _meta(
    name: str, full: tuple[int, ...], rng: tuple[tuple[int, int], ...], rank: int, role: str
) -> ParameterShardMeta:
    return ParameterShardMeta(
        param_name=name,
        full_shape=full,
        dtype_str="bfloat16",
        ranges=rng,
        global_rank=rank,
        role=role,  # type: ignore[arg-type]
    )


class TestPrepare:
    def test_prepare_returns_metas(self):
        eng = ShardedNCCLCheckpointEngine()
        metas = [_meta("w", (8, 4), ((0, 4), (0, 4)), 0, "train")]
        eng.set_shard_metas(metas, role="train")
        info = eng.prepare()
        assert info["role"] == "train"
        assert len(info["shard_metas"]) == 1
        assert info["shard_metas"][0]["param_name"] == "w"

    def test_prepare_without_metas_raises(self):
        eng = ShardedNCCLCheckpointEngine()
        with pytest.raises(RuntimeError, match="before set_shard_metas"):
            eng.prepare()

    def test_role_mismatch_raises(self):
        eng = ShardedNCCLCheckpointEngine()
        wrong_role = [_meta("w", (8, 4), ((0, 4), (0, 4)), 0, "train")]
        with pytest.raises(ValueError, match="meta role"):
            eng.set_shard_metas(wrong_role, role="rollout")


class TestBuildTopology:
    """Exercise the classmethod that aggregates per-worker metadata and
    builds the routing plan + per-worker kwargs.
    """

    def _trainer_info(self, metas: list[ParameterShardMeta]) -> dict:
        return {"shard_metas": [m.to_dict() for m in metas], "role": "train"}

    def _rollout_info(self, metas: list[ParameterShardMeta]) -> dict:
        return {"shard_metas": [m.to_dict() for m in metas], "role": "rollout"}

    def test_topology_rank_assignment_2t_2r(self):
        """trainer_ws=2, rollout_ws=2 → trainers get [0,1], rollouts get [2,3]."""
        train0 = [_meta("w", (8, 4), ((0, 4), (0, 4)), 0, "train")]
        train1 = [_meta("w", (8, 4), ((4, 8), (0, 4)), 0, "train")]
        roll0 = [_meta("w", (8, 4), ((0, 4), (0, 4)), 0, "rollout")]
        roll1 = [_meta("w", (8, 4), ((4, 8), (0, 4)), 0, "rollout")]

        metadata = [
            self._trainer_info(train0),
            self._trainer_info(train1),
            self._rollout_info(roll0),
            self._rollout_info(roll1),
        ]
        tk, rk = ShardedNCCLCheckpointEngine.build_topology(2, 2, metadata)
        assert tk["rank"] == [0, 1]
        assert rk["rank"] == [2, 3]
        assert tk["world_size"] == [4, 4]
        assert rk["world_size"] == [4, 4]
        # plan is identical for every worker
        assert tk["transfer_plan_json"][0] == tk["transfer_plan_json"][1]
        assert tk["transfer_plan_json"][0] == rk["transfer_plan_json"][0]

    def test_plan_routes_to_correct_global_ranks(self):
        """train rank 0 (group rank 0) should send to rollout rank 0 (group rank 2), etc."""
        train0 = [_meta("w", (8, 4), ((0, 4), (0, 4)), 0, "train")]
        train1 = [_meta("w", (8, 4), ((4, 8), (0, 4)), 0, "train")]
        roll0 = [_meta("w", (8, 4), ((0, 4), (0, 4)), 0, "rollout")]
        roll1 = [_meta("w", (8, 4), ((4, 8), (0, 4)), 0, "rollout")]

        metadata = [
            self._trainer_info(train0),
            self._trainer_info(train1),
            self._rollout_info(roll0),
            self._rollout_info(roll1),
        ]
        tk, _ = ShardedNCCLCheckpointEngine.build_topology(2, 2, metadata)
        plan = TransferPlan.from_json(tk["transfer_plan_json"][0])
        # Edges: 0→2, 1→3
        assert len(plan.edges) == 2
        pairs = {(e.src_global_rank, e.dst_global_rank) for e in plan.edges}
        assert pairs == {(0, 2), (1, 3)}

    def test_metadata_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="metadata length"):
            ShardedNCCLCheckpointEngine.build_topology(2, 2, [self._trainer_info([])])  # only 1 entry

    def test_metadata_role_mismatch_raises(self):
        """trainer_ws=1, but the only metadata entry has role='rollout'."""
        bad = [self._rollout_info([])]
        with pytest.raises(ValueError, match="role"):
            ShardedNCCLCheckpointEngine.build_topology(1, 0, bad)

    def test_moe_topology_ep2_to_ep1(self):
        """Sanity: MoE EP=2 trainer, EP=1 rollout. Confirms per-expert routing.

        Trainer rank 0 holds experts 0, 1; rank 1 holds 2, 3.
        Rollout rank 0 (group rank 2) holds all 4.
        """
        full = (8, 4)
        ranges = ((0, 8), (0, 4))
        t0 = [_meta(f"experts.{e}.gp", full, ranges, 0, "train") for e in (0, 1)]
        t1 = [_meta(f"experts.{e}.gp", full, ranges, 0, "train") for e in (2, 3)]
        r0 = [_meta(f"experts.{e}.gp", full, ranges, 0, "rollout") for e in range(4)]

        metadata = [self._trainer_info(t0), self._trainer_info(t1), self._rollout_info(r0)]
        tk, _ = ShardedNCCLCheckpointEngine.build_topology(2, 1, metadata)
        plan = TransferPlan.from_json(tk["transfer_plan_json"][0])

        # 4 edges total: train 0 sends experts 0/1 to rollout (rank 2);
        # train 1 sends experts 2/3 to rollout (rank 2).
        assert len(plan.edges) == 4
        for e in plan.edges:
            assert e.dst_global_rank == 2  # the only rollout rank
            assert e.src_global_rank in (0, 1)


class TestRegistry:
    def test_registered_as_sharded_nccl(self):
        from verl.checkpoint_engine.base import CheckpointEngineRegistry

        cls = CheckpointEngineRegistry.get("sharded_nccl")
        assert cls is ShardedNCCLCheckpointEngine

    def test_new_via_registry(self):
        from verl.checkpoint_engine.base import CheckpointEngineRegistry

        eng = CheckpointEngineRegistry.new("sharded_nccl", group_name="test")
        assert isinstance(eng, ShardedNCCLCheckpointEngine)
        assert eng.group_name == "test"


class TestM4MetasProvider:
    """M4: lazy metas_provider closure resolution at prepare()."""

    def test_lazy_provider_fires_once_at_prepare(self):
        called = []

        def provider():
            called.append(1)
            return [_meta("w", (8, 4), ((0, 4), (0, 4)), 0, "train")]

        # Engine accepts is_master/bucket_size via **_unused_kwargs (so
        # CheckpointEngineRegistry.new can pass them generically).
        eng = ShardedNCCLCheckpointEngine(
            is_master=True,
            bucket_size=1024,
            metas_provider=provider,
            role="train",
        )
        assert not called  # not invoked yet
        info = eng.prepare()
        assert called == [1]
        assert info["role"] == "train"
        assert info["shard_metas"][0]["param_name"] == "w"

    def test_explicit_set_shard_metas_short_circuits_provider(self):
        """If set_shard_metas was called explicitly, provider should NOT fire."""
        called = []

        def provider():
            called.append(1)
            return []

        eng = ShardedNCCLCheckpointEngine(metas_provider=provider, role="train")
        eng.set_shard_metas(
            [_meta("w", (8, 4), ((0, 4), (0, 4)), 0, "train")],
            role="train",
        )
        eng.prepare()
        assert not called


class TestM4VllmEnrichInTopology:
    """M4: build_topology must run edges through vllm_enrich_edge."""

    def test_moe_expert_edge_gets_vllm_metadata(self):
        full = (16, 8)
        ranges = ((0, 16), (0, 8))
        train_info = {
            "shard_metas": [
                _meta(
                    "model.layers.0.mlp.experts.3.gate_proj.weight",
                    full,
                    ranges,
                    0,
                    "train",
                ).to_dict()
            ],
            "role": "train",
        }
        rollout_info = {
            "shard_metas": [
                _meta(
                    "model.layers.0.mlp.experts.3.gate_proj.weight",
                    full,
                    ranges,
                    0,
                    "rollout",
                ).to_dict()
            ],
            "role": "rollout",
        }
        tkw, _ = ShardedNCCLCheckpointEngine.build_topology(1, 1, [train_info, rollout_info])
        plan = TransferPlan.from_json(tkw["transfer_plan_json"][0])
        assert len(plan.edges) == 1
        e = plan.edges[0]
        assert e.target_param_name == "model.layers.0.mlp.experts.w13_weight"
        assert e.shard_id == "w1"
        assert e.expert_id == 3


class TestM4IncomingEdgesJsonExport:
    """M4: get_incoming_edges_json gives the JSON the vLLM bridge consumes."""

    def test_roundtrip_via_transfer_edge_from_dict(self):
        import json

        from verl.checkpoint_engine.parallel_meta import TransferEdge

        full = (4, 4)
        ranges = ((0, 4), (0, 4))
        train_info = {
            "shard_metas": [_meta("a", full, ranges, 0, "train").to_dict()],
            "role": "train",
        }
        rollout_info = {
            "shard_metas": [_meta("a", full, ranges, 0, "rollout").to_dict()],
            "role": "rollout",
        }
        _, rkw = ShardedNCCLCheckpointEngine.build_topology(1, 1, [train_info, rollout_info])
        plan = TransferPlan.from_json(rkw["transfer_plan_json"][0])

        eng = ShardedNCCLCheckpointEngine()
        # Bypass NCCL init by setting the rank-filtered edges directly.
        eng._incoming_edges = plan.edges_for_dst(rkw["rank"][0])

        as_json = eng.get_incoming_edges_json()
        decoded = json.loads(as_json)
        assert len(decoded) >= 1
        # Each entry must roundtrip back through TransferEdge.
        edge = TransferEdge.from_dict(decoded[0])
        assert edge.param_name == "a"
