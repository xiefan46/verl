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
"""GPU smoke test for ``ShardedNCCLCheckpointEngine``.

End-to-end exercises the full ``prepare → build_topology →
init_process_group → send_weights / receive_weights`` flow on real GPUs
with a real ``ray.util.collective`` NCCL group. Validates that:

1. The routing plan correctly maps trainer shards to rollout-side shards.
2. ``collective.send`` / ``collective.recv`` between Ray actors actually
   transfers the right bytes.
3. ``send_weights`` and ``receive_weights`` can run concurrently across
   trainer/rollout actors without deadlock.

Topology: 2 trainer actors + 2 rollout-side actors, each on 1 GPU.

Cases:
- ``test_dense_1to1_routing``: single dense param sharded TP=2 on both
  sides → 2 edges (rank 0→2, rank 1→3). Trainer ranks fill their shards
  with a distinctive pattern; rollouts assert they received the matching
  pattern.
- ``test_moe_per_expert_routing``: MoE with EP=2 on both sides, 4 experts
  total. Each trainer holds 2 experts; each rollout receives the matching
  experts → 4 edges. Validates per-expert routing using the param-name
  encoding convention (``experts.{i}.gate_proj.weight``).

Run on RunPod 4×H100::

    pytest -s tests/checkpoint_engine/test_sharded_nccl_gpu_smoke.py -v
"""

from __future__ import annotations

import asyncio

import pytest
import torch

ray = pytest.importorskip("ray")
pytest.importorskip("ray.util.collective")

# Skip the entire module if we are on a machine with fewer than 4 GPUs.
if not torch.cuda.is_available() or torch.cuda.device_count() < 4:
    pytest.skip("Sharded NCCL GPU smoke test needs >=4 CUDA devices.", allow_module_level=True)


# ----------------------------------------------------------------------
# Ray worker for trainer / rollout sides
# ----------------------------------------------------------------------


@ray.remote(num_gpus=1)
class ShardedTestWorker:
    """Generic Ray actor that wraps a ShardedNCCLCheckpointEngine for both
    trainer and rollout-side roles.
    """

    def __init__(self, role: str, shard_metas_dict: list[dict]) -> None:
        # Lazy imports so the module-level import in this test file is light.
        import torch as _torch  # noqa: F401

        from verl.checkpoint_engine.parallel_meta import ParameterShardMeta
        from verl.checkpoint_engine.sharded_nccl_checkpoint_engine import (
            ShardedNCCLCheckpointEngine,
        )

        self.role = role
        self.engine = ShardedNCCLCheckpointEngine(group_name="sharded_nccl_smoke")
        self.metas = [ParameterShardMeta.from_dict(d) for d in shard_metas_dict]
        self.engine.set_shard_metas(self.metas, role=role)

    def prepare(self) -> dict:
        return self.engine.prepare()

    def init_pg(self, rank: int, world_size: int, transfer_plan_json: str) -> dict:
        self.engine.init_process_group(
            rank=rank,
            world_size=world_size,
            transfer_plan_json=transfer_plan_json,
        )
        return {
            "rank": rank,
            "plan_hash": self.engine.plan_hash(),
            "outgoing": len(self.engine._outgoing_edges or []),  # type: ignore[union-attr]
            "incoming": len(self.engine._incoming_edges or []),  # type: ignore[union-attr]
        }

    # ----- trainer side -----
    def send_pattern_tensors(self, pattern: float) -> int:
        """Trainer-only: produce a local tensor for each meta filled with
        the rank-specific ``pattern`` value, then send via the routing plan.
        Returns the number of edges fired.
        """

        async def _send() -> int:
            weights = []
            for m in self.metas:
                t = torch.full(
                    m.local_shape(),
                    fill_value=pattern,
                    dtype=torch.bfloat16,
                    device="cuda",
                )
                weights.append((m.param_name, t))

            def gen():
                yield from weights

            await self.engine.send_weights(gen())
            return len(weights)

        return asyncio.run(_send())

    # ----- rollout side -----
    def receive_and_collect(self) -> list[tuple[str, float, float, tuple[int, ...]]]:
        """Rollout-only: drain receive_weights and return per-buffer summary
        ``(encoded_name, min_value, max_value, shape)`` so the driver can
        verify the trainer's pattern survived end-to-end.
        """

        async def _recv() -> list[tuple[str, float, float, tuple[int, ...]]]:
            collected: list[tuple[str, float, float, tuple[int, ...]]] = []
            async for encoded_name, buf in self.engine.receive_weights():
                collected.append(
                    (
                        encoded_name,
                        float(buf.float().min().item()),
                        float(buf.float().max().item()),
                        tuple(buf.shape),
                    )
                )
            return collected

        return asyncio.run(_recv())

    def get_incoming_edges_summary(self) -> list[dict]:
        """Mirror of ``get_incoming_edges`` returning JSON-friendly dicts."""
        return [e.to_dict() for e in self.engine.get_incoming_edges()]


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _meta_dict(
    name: str,
    full_shape: tuple[int, ...],
    ranges: tuple[tuple[int, int], ...],
    role: str,
) -> dict:
    """Build a ParameterShardMeta.to_dict() payload.

    ``global_rank`` is a placeholder — build_topology will overwrite with
    the freshly assigned NCCL group rank.
    """
    return {
        "param_name": name,
        "full_shape": list(full_shape),
        "dtype_str": "bfloat16",
        "ranges": [list(r) for r in ranges],
        "global_rank": 0,
        "role": role,
    }


def _run_smoke(
    train_metas_per_rank: list[list[dict]],
    rollout_metas_per_rank: list[list[dict]],
    trainer_patterns: list[float],
) -> dict:
    """Spin up the actors, run the full sharded refit cycle, return the
    rollout-side observations for assertion.
    """
    from verl.checkpoint_engine.sharded_nccl_checkpoint_engine import (
        ShardedNCCLCheckpointEngine,
    )

    if not ray.is_initialized():
        ray.init(num_gpus=4, log_to_driver=True)

    trainer_actors = [ShardedTestWorker.remote("train", metas) for metas in train_metas_per_rank]
    rollout_actors = [ShardedTestWorker.remote("rollout", metas) for metas in rollout_metas_per_rank]

    # 1. prepare (collect per-rank metadata payloads)
    prepare_metadata = ray.get(
        [w.prepare.remote() for w in trainer_actors] + [w.prepare.remote() for w in rollout_actors]
    )

    # 2. build_topology (driver side, not on workers)
    tk, rk = ShardedNCCLCheckpointEngine.build_topology(len(trainer_actors), len(rollout_actors), prepare_metadata)

    # 3. init_process_group on every worker concurrently
    init_futures = []
    for i, w in enumerate(trainer_actors):
        init_futures.append(
            w.init_pg.remote(
                rank=tk["rank"][i],
                world_size=tk["world_size"][i],
                transfer_plan_json=tk["transfer_plan_json"][i],
            )
        )
    for i, w in enumerate(rollout_actors):
        init_futures.append(
            w.init_pg.remote(
                rank=rk["rank"][i],
                world_size=rk["world_size"][i],
                transfer_plan_json=rk["transfer_plan_json"][i],
            )
        )
    init_results = ray.get(init_futures)

    # 4. run send + recv concurrently
    send_futures = [w.send_pattern_tensors.remote(pat) for w, pat in zip(trainer_actors, trainer_patterns, strict=True)]
    recv_futures = [w.receive_and_collect.remote() for w in rollout_actors]
    sent_counts = ray.get(send_futures)
    received = ray.get(recv_futures)
    edges_summaries = ray.get([w.get_incoming_edges_summary.remote() for w in rollout_actors])

    # Cleanup
    for w in trainer_actors + rollout_actors:
        ray.kill(w)

    return {
        "init_results": init_results,
        "sent_counts": sent_counts,
        "received": received,
        "edges_summaries": edges_summaries,
    }


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------


class TestDenseRouting:
    def test_dense_1to1_routing(self):
        """Single dense param, TP=2 on both sides → 2 edges (0→2, 1→3).

        Trainer rank 0 fills its half with 1.0, trainer rank 1 with 2.0.
        Rollout rank 0 must receive 1.0, rollout rank 1 must receive 2.0.
        """
        full = (8, 4)
        train_metas_per_rank = [
            [_meta_dict("w", full, ((0, 4), (0, 4)), "train")],  # rank 0
            [_meta_dict("w", full, ((4, 8), (0, 4)), "train")],  # rank 1
        ]
        rollout_metas_per_rank = [
            [_meta_dict("w", full, ((0, 4), (0, 4)), "rollout")],  # rank 0
            [_meta_dict("w", full, ((4, 8), (0, 4)), "rollout")],  # rank 1
        ]
        result = _run_smoke(train_metas_per_rank, rollout_metas_per_rank, trainer_patterns=[1.0, 2.0])

        # Every actor sees the same plan_hash
        hashes = {r["plan_hash"] for r in result["init_results"]}
        assert len(hashes) == 1, f"plan_hash disagreement: {hashes}"

        # 2 sends and 2 receives total
        assert result["sent_counts"] == [1, 1]
        assert all(len(r) == 1 for r in result["received"])

        # Rollout 0 sees 1.0, rollout 1 sees 2.0
        rollout0 = result["received"][0][0]
        rollout1 = result["received"][1][0]
        assert rollout0[0].startswith("w|")
        assert rollout0[1] == pytest.approx(1.0)
        assert rollout0[2] == pytest.approx(1.0)
        assert rollout0[3] == (4, 4)
        assert rollout1[1] == pytest.approx(2.0)
        assert rollout1[2] == pytest.approx(2.0)
        assert rollout1[3] == (4, 4)


class TestMoERouting:
    def test_moe_per_expert_routing(self):
        """MoE EP=2 → EP=2 with 4 experts.

        Trainer rank 0 holds experts 0, 1 (full local) and sends to rollout 0.
        Trainer rank 1 holds experts 2, 3 (full local) and sends to rollout 1.
        → 4 edges (1 per expert).
        Patterns: trainer 0 fills 10.0, trainer 1 fills 20.0.
        """
        full = (8, 4)
        ranges = ((0, 8), (0, 4))  # each expert is held fully (no TP on experts)
        train_metas_per_rank = [
            [_meta_dict(f"experts.{e}.gate_proj.weight", full, ranges, "train") for e in (0, 1)],
            [_meta_dict(f"experts.{e}.gate_proj.weight", full, ranges, "train") for e in (2, 3)],
        ]
        rollout_metas_per_rank = [
            [_meta_dict(f"experts.{e}.gate_proj.weight", full, ranges, "rollout") for e in (0, 1)],
            [_meta_dict(f"experts.{e}.gate_proj.weight", full, ranges, "rollout") for e in (2, 3)],
        ]
        result = _run_smoke(train_metas_per_rank, rollout_metas_per_rank, trainer_patterns=[10.0, 20.0])

        # 2 experts sent per trainer
        assert result["sent_counts"] == [2, 2]
        # 2 experts received per rollout
        assert all(len(r) == 2 for r in result["received"])

        # Rollout 0 received experts 0, 1 → pattern 10.0; rollout 1 → experts 2, 3 → 20.0
        for entry in result["received"][0]:
            assert entry[1] == pytest.approx(10.0)
            assert entry[2] == pytest.approx(10.0)
            assert entry[3] == (8, 4)
        for entry in result["received"][1]:
            assert entry[1] == pytest.approx(20.0)
            assert entry[2] == pytest.approx(20.0)
            assert entry[3] == (8, 4)

        # The encoded_name for each rollout's first edge should reference experts.0/1 or 2/3 respectively
        rollout0_names = {entry[0].split("|")[0] for entry in result["received"][0]}
        rollout1_names = {entry[0].split("|")[0] for entry in result["received"][1]}
        assert rollout0_names == {"experts.0.gate_proj.weight", "experts.1.gate_proj.weight"}
        assert rollout1_names == {"experts.2.gate_proj.weight", "experts.3.gate_proj.weight"}
