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
"""Pure-CPU unit tests for the sharded-aware weight refit routing plan.

Covers:
- ParameterShardMeta / TransferEdge / TransferPlan data classes
- intersect / global_box_to_local_slice utilities
- build_transfer_plan over the parallelism configurations the MVP must
  handle (same TP, cross TP, DP replicas, MoE expert dim, 3D boxes,
  partial overlap, missing-param edge cases)
- Determinism (100 runs → identical hash; same inputs across "ranks"
  produce identical plans)
- enrich_edge callback for vLLM-side metadata injection
- Serialization round-trips

Run::
    pytest -s tests/checkpoint_engine/test_transfer_plan.py -v
"""

from __future__ import annotations

import pytest

from verl.checkpoint_engine.parallel_meta import (
    ParameterShardMeta,
    TransferEdge,
    TransferPlan,
)
from verl.checkpoint_engine.transfer_plan import (
    build_transfer_plan,
    global_box_to_local_slice,
    intersect,
    sanity_check_cross_rank,
)

# -------- helpers --------


def _meta(
    name: str,
    full_shape: tuple[int, ...],
    ranges: tuple[tuple[int, int], ...],
    rank: int,
    role: str,
    dtype: str = "bfloat16",
) -> ParameterShardMeta:
    return ParameterShardMeta(
        param_name=name,
        full_shape=full_shape,
        dtype_str=dtype,
        ranges=ranges,
        global_rank=rank,
        role=role,  # type: ignore[arg-type]
    )


# ============================================================
# Data-class basics
# ============================================================


class TestParameterShardMeta:
    def test_local_shape(self):
        m = _meta("w", (768, 2048), ((0, 384), (0, 2048)), 0, "train")
        assert m.local_shape() == (384, 2048)

    def test_to_dict_round_trip(self):
        m = _meta("w", (768, 2048), ((0, 384), (0, 2048)), 0, "train")
        m2 = ParameterShardMeta.from_dict(m.to_dict())
        assert m == m2

    def test_invalid_ranges_length_raises(self):
        with pytest.raises(ValueError, match="ranges length"):
            ParameterShardMeta(
                param_name="w",
                full_shape=(768, 2048),
                dtype_str="bfloat16",
                ranges=((0, 384),),  # only 1 dim, but full_shape is 2D
                global_rank=0,
                role="train",
            )

    @pytest.mark.parametrize(
        ("ranges",),
        [
            (((100, 50), (0, 2048)),),  # start > end
            (((0, 0), (0, 2048)),),  # empty range
            (((-1, 100), (0, 2048)),),  # negative start
            (((0, 1000), (0, 2048)),),  # end > full_shape on dim 0 (768)
        ],
    )
    def test_invalid_range_values_raise(self, ranges):
        with pytest.raises(ValueError, match="invalid range"):
            ParameterShardMeta(
                param_name="w",
                full_shape=(768, 2048),
                dtype_str="bfloat16",
                ranges=ranges,
                global_rank=0,
                role="train",
            )


class TestTransferEdge:
    def test_num_bytes(self):
        e = TransferEdge(
            param_name="w",
            src_global_rank=0,
            dst_global_rank=4,
            src_local_slice_encoded=((None, None, None),),
            dst_local_slice_encoded=((None, None, None),),
            shape=(384, 2048),
            dtype_str="bfloat16",
        )
        assert e.num_bytes() == 384 * 2048 * 2

    def test_slice_decode(self):
        e = TransferEdge(
            param_name="w",
            src_global_rank=0,
            dst_global_rank=4,
            src_local_slice_encoded=((0, 384, None), (0, 2048, None)),
            dst_local_slice_encoded=((128, 512, None), (0, 2048, None)),
            shape=(384, 2048),
            dtype_str="bfloat16",
        )
        ss = e.src_local_slice()
        assert ss == (slice(0, 384, None), slice(0, 2048, None))
        ds = e.dst_local_slice()
        assert ds == (slice(128, 512, None), slice(0, 2048, None))

    def test_to_dict_round_trip(self):
        e = TransferEdge(
            param_name="w",
            src_global_rank=0,
            dst_global_rank=4,
            src_local_slice_encoded=((0, 384, None), (None, None, None)),
            dst_local_slice_encoded=((0, 384, None), (None, None, None)),
            shape=(384, 2048),
            dtype_str="bfloat16",
            target_param_name="w13_weight",
            shard_id="w1",
            expert_id=3,
        )
        assert TransferEdge.from_dict(e.to_dict()) == e


# ============================================================
# Geometry primitives
# ============================================================


class TestIntersect:
    def test_full_overlap(self):
        assert intersect(((0, 10), (0, 20)), ((0, 10), (0, 20))) == ((0, 10), (0, 20))

    def test_partial_overlap(self):
        assert intersect(((0, 10), (0, 20)), ((5, 15), (10, 30))) == ((5, 10), (10, 20))

    def test_no_overlap_on_one_dim(self):
        assert intersect(((0, 10), (0, 20)), ((10, 20), (0, 20))) is None

    def test_no_overlap_returns_none_not_empty(self):
        assert intersect(((0, 5),), ((5, 10),)) is None

    def test_different_ndim_raises(self):
        with pytest.raises(ValueError, match="different ndim"):
            intersect(((0, 10),), ((0, 10), (0, 20)))


class TestGlobalBoxToLocalSlice:
    def test_basic_offset_subtract(self):
        # meta holds [(100, 200), (0, 2048)]; overlap is [(100, 150), (0, 2048)]
        # local slice should be [(0, 50), (0, 2048)]
        s = global_box_to_local_slice(((100, 150), (0, 2048)), ((100, 200), (0, 2048)))
        assert s == ((0, 50, None), (0, 2048, None))

    def test_overlap_at_end_of_shard(self):
        s = global_box_to_local_slice(((150, 200), (0, 2048)), ((100, 200), (0, 2048)))
        assert s == ((50, 100, None), (0, 2048, None))


# ============================================================
# Routing plan: dense parallelism configurations
# ============================================================


class TestRoutingDense:
    def test_same_tp_same_ep_1to1(self):
        """TP=2 → TP=2: each train rank sends its full shard to matching rollout rank."""
        train = [
            _meta("gate_proj", (768, 2048), ((0, 384), (0, 2048)), 0, "train"),
            _meta("gate_proj", (768, 2048), ((384, 768), (0, 2048)), 1, "train"),
        ]
        rollout = [
            _meta("gate_proj", (768, 2048), ((0, 384), (0, 2048)), 2, "rollout"),
            _meta("gate_proj", (768, 2048), ((384, 768), (0, 2048)), 3, "rollout"),
        ]
        plan = build_transfer_plan(train, rollout)
        assert len(plan.edges) == 2
        assert (plan.edges[0].src_global_rank, plan.edges[0].dst_global_rank) == (0, 2)
        assert (plan.edges[1].src_global_rank, plan.edges[1].dst_global_rank) == (1, 3)
        # Each edge sends the FULL local tensor.
        assert plan.edges[0].shape == (384, 2048)
        assert plan.edges[0].src_local_slice_encoded == ((0, 384, None), (0, 2048, None))
        assert plan.edges[0].dst_local_slice_encoded == ((0, 384, None), (0, 2048, None))

    def test_cross_tp_4to2(self):
        """train TP=4 → infer TP=2: each train rank's shard fits inside a target rank's slab."""
        train = [
            _meta("gate_proj", (768, 2048), ((0, 192), (0, 2048)), 0, "train"),
            _meta("gate_proj", (768, 2048), ((192, 384), (0, 2048)), 1, "train"),
            _meta("gate_proj", (768, 2048), ((384, 576), (0, 2048)), 2, "train"),
            _meta("gate_proj", (768, 2048), ((576, 768), (0, 2048)), 3, "train"),
        ]
        rollout = [
            _meta("gate_proj", (768, 2048), ((0, 384), (0, 2048)), 4, "rollout"),
            _meta("gate_proj", (768, 2048), ((384, 768), (0, 2048)), 5, "rollout"),
        ]
        plan = build_transfer_plan(train, rollout)
        assert len(plan.edges) == 4
        # Train ranks 0/1 → infer 4 (first half); 2/3 → infer 5 (second half)
        e_by_pair = {(e.src_global_rank, e.dst_global_rank): e for e in plan.edges}
        assert (0, 4) in e_by_pair
        assert (1, 4) in e_by_pair
        assert (2, 5) in e_by_pair
        assert (3, 5) in e_by_pair
        # Train rank 0 fills dst[0:192], rank 1 fills dst[192:384]
        assert e_by_pair[(0, 4)].dst_local_slice_encoded == ((0, 192, None), (0, 2048, None))
        assert e_by_pair[(1, 4)].dst_local_slice_encoded == ((192, 384, None), (0, 2048, None))

    def test_cross_tp_2to4(self):
        """train TP=2 → infer TP=4: each train rank's shard is fragmented across 2 rollout ranks."""
        train = [
            _meta("gate_proj", (768, 2048), ((0, 384), (0, 2048)), 0, "train"),
            _meta("gate_proj", (768, 2048), ((384, 768), (0, 2048)), 1, "train"),
        ]
        rollout = [
            _meta("gate_proj", (768, 2048), ((0, 192), (0, 2048)), 2, "rollout"),
            _meta("gate_proj", (768, 2048), ((192, 384), (0, 2048)), 3, "rollout"),
            _meta("gate_proj", (768, 2048), ((384, 576), (0, 2048)), 4, "rollout"),
            _meta("gate_proj", (768, 2048), ((576, 768), (0, 2048)), 5, "rollout"),
        ]
        plan = build_transfer_plan(train, rollout)
        assert len(plan.edges) == 4
        # Train 0 → infer 2 + infer 3; train 1 → infer 4 + infer 5
        pair_to_shape = {(e.src_global_rank, e.dst_global_rank): e.shape for e in plan.edges}
        assert pair_to_shape[(0, 2)] == (192, 2048)
        assert pair_to_shape[(0, 3)] == (192, 2048)
        assert pair_to_shape[(1, 4)] == (192, 2048)
        assert pair_to_shape[(1, 5)] == (192, 2048)

    def test_unaligned_tp_3to2(self):
        """train TP=3 → infer TP=2 (256 doesn't divide cleanly). Verify slices are still correct."""
        # full_shape (768, 8) so TP=3 → 256 per shard, TP=2 → 384 per shard
        train = [
            _meta("w", (768, 8), ((0, 256), (0, 8)), 0, "train"),
            _meta("w", (768, 8), ((256, 512), (0, 8)), 1, "train"),
            _meta("w", (768, 8), ((512, 768), (0, 8)), 2, "train"),
        ]
        rollout = [
            _meta("w", (768, 8), ((0, 384), (0, 8)), 3, "rollout"),
            _meta("w", (768, 8), ((384, 768), (0, 8)), 4, "rollout"),
        ]
        plan = build_transfer_plan(train, rollout)
        # train 0 → rollout 3 (full 256 fits in dst [0:384])
        # train 1 → rollout 3 [256:384] (128 rows) and rollout 4 [0:128] (128 rows)
        # train 2 → rollout 4 [128:384] (256 rows)
        # Total: 4 edges
        assert len(plan.edges) == 4
        # Verify total bytes transferred == total bytes needed
        total = sum(e.num_bytes() for e in plan.edges)
        expected = 768 * 8 * 2  # full param, BF16
        assert total == expected, f"sharded transfer must cover the full param exactly: {total} vs {expected}"

    def test_replicated_dim(self):
        """Replicated dimensions: train and rollout both hold full range on dim 1.

        Translates to a single edge per matching rank pair.
        """
        # All ranks hold full dim 1 (Replicate); split only on dim 0 (TP)
        train = [
            _meta("w", (1024, 2048), ((0, 512), (0, 2048)), 0, "train"),
            _meta("w", (1024, 2048), ((512, 1024), (0, 2048)), 1, "train"),
        ]
        rollout = [
            _meta("w", (1024, 2048), ((0, 512), (0, 2048)), 2, "rollout"),
            _meta("w", (1024, 2048), ((512, 1024), (0, 2048)), 3, "rollout"),
        ]
        plan = build_transfer_plan(train, rollout)
        assert len(plan.edges) == 2


class TestRoutingDP:
    def test_dp_replica_load_balanced(self):
        """Two DP replicas of the same param → greedy picks both to balance send_bytes."""
        train = [
            _meta("q_proj", (1024, 2048), ((0, 1024), (0, 2048)), 0, "train"),
            _meta("q_proj", (1024, 2048), ((0, 1024), (0, 2048)), 1, "train"),
        ]
        rollout = [
            _meta("q_proj", (1024, 2048), ((0, 512), (0, 2048)), 2, "rollout"),
            _meta("q_proj", (1024, 2048), ((512, 1024), (0, 2048)), 3, "rollout"),
        ]
        plan = build_transfer_plan(train, rollout)
        assert len(plan.edges) == 2
        sources = {e.src_global_rank for e in plan.edges}
        assert sources == {0, 1}, "greedy must use both DP replicas to load-balance"


# ============================================================
# Routing plan: MoE / expert dimension
# ============================================================


class TestRoutingMoE:
    """MoE experts are modeled as independent params: one ParameterShardMeta
    per (rank, expert). Routing algo treats them like any other param —
    expert dim is just another partitioning axis.
    """

    def test_moe_ep2_to_ep1(self):
        """train EP=2 (each rank holds half the experts) → infer EP=1 (one rank holds all experts).

        Concretely 4 experts: train rank 0 holds experts 0, 1; train rank 1 holds 2, 3.
        Rollout rank 2 needs all 4 → 4 P2P edges.
        """
        train = []
        rollout = []
        full = (768, 2048)
        ranges = ((0, 768), (0, 2048))
        # train rank 0 holds expert 0, 1
        for e in (0, 1):
            train.append(_meta(f"experts.{e}.gate_proj", full, ranges, 0, "train"))
        # train rank 1 holds expert 2, 3
        for e in (2, 3):
            train.append(_meta(f"experts.{e}.gate_proj", full, ranges, 1, "train"))
        # rollout rank 2 holds all 4 experts
        for e in range(4):
            rollout.append(_meta(f"experts.{e}.gate_proj", full, ranges, 2, "rollout"))

        plan = build_transfer_plan(train, rollout)
        assert len(plan.edges) == 4
        # Train rank 0 sends 2 edges (experts 0, 1); train rank 1 sends 2 (experts 2, 3)
        src_counts = {0: 0, 1: 0}
        for e in plan.edges:
            src_counts[e.src_global_rank] += 1
        assert src_counts == {0: 2, 1: 2}

    def test_moe_ep4_to_ep2(self):
        """train EP=4 → infer EP=2. Each infer rank receives experts from 2 train ranks."""
        train = []
        rollout = []
        full = (768, 2048)
        ranges = ((0, 768), (0, 2048))
        # train: 8 experts, EP=4 → 2 experts per rank
        for rank, (e1, e2) in enumerate([(0, 1), (2, 3), (4, 5), (6, 7)]):
            train.append(_meta(f"experts.{e1}.gate_proj", full, ranges, rank, "train"))
            train.append(_meta(f"experts.{e2}.gate_proj", full, ranges, rank, "train"))
        # rollout: 8 experts, EP=2 → 4 experts per rank
        for rank, experts in enumerate([(0, 1, 2, 3), (4, 5, 6, 7)], start=4):
            for e in experts:
                rollout.append(_meta(f"experts.{e}.gate_proj", full, ranges, rank, "rollout"))

        plan = build_transfer_plan(train, rollout)
        assert len(plan.edges) == 8
        # Expert 0-3 → rollout rank 4; experts 4-7 → rollout rank 5
        for e in plan.edges:
            expert_id = int(e.param_name.split(".")[1])
            expected_dst = 4 if expert_id < 4 else 5
            assert e.dst_global_rank == expected_dst, f"expert {expert_id} → wrong dst {e.dst_global_rank}"


class TestRouting3D:
    def test_3d_box_intersection(self):
        """MoE w13_weight as a 3D logical box ``[num_experts, 2*I, H]``.

        train EP=2 holds half the experts, no TP; rollout EP=1 TP=2 splits the
        fused dim. → cross-partition routing.
        """
        # 4 experts, 2*I=1024, H=2048
        full = (4, 1024, 2048)
        # train rank 0: experts 0, 1 (no TP)
        # train rank 1: experts 2, 3 (no TP)
        train = [
            _meta("w13", full, ((0, 2), (0, 1024), (0, 2048)), 0, "train"),
            _meta("w13", full, ((2, 4), (0, 1024), (0, 2048)), 1, "train"),
        ]
        # rollout rank 2: all experts, dim 1 [0, 512] (TP rank 0 of TP=2)
        # rollout rank 3: all experts, dim 1 [512, 1024] (TP rank 1)
        rollout = [
            _meta("w13", full, ((0, 4), (0, 512), (0, 2048)), 2, "rollout"),
            _meta("w13", full, ((0, 4), (512, 1024), (0, 2048)), 3, "rollout"),
        ]
        plan = build_transfer_plan(train, rollout)
        # 4 intersections: (train 0 ∩ rollout 2), (train 0 ∩ rollout 3),
        # (train 1 ∩ rollout 2), (train 1 ∩ rollout 3)
        assert len(plan.edges) == 4
        # Each edge transfers half the experts × half the fused dim × hidden
        for e in plan.edges:
            assert e.shape == (2, 512, 2048)


# ============================================================
# Determinism & cross-rank consistency
# ============================================================


class TestDeterminism:
    def test_100_runs_same_hash(self):
        train = [
            _meta("w", (768, 2048), ((0, 192), (0, 2048)), 0, "train"),
            _meta("w", (768, 2048), ((192, 384), (0, 2048)), 1, "train"),
            _meta("w", (768, 2048), ((384, 576), (0, 2048)), 2, "train"),
            _meta("w", (768, 2048), ((576, 768), (0, 2048)), 3, "train"),
        ]
        rollout = [
            _meta("w", (768, 2048), ((0, 384), (0, 2048)), 4, "rollout"),
            _meta("w", (768, 2048), ((384, 768), (0, 2048)), 5, "rollout"),
        ]
        hashes = {build_transfer_plan(train, rollout).plan_hash for _ in range(100)}
        assert len(hashes) == 1

    def test_input_reorder_same_plan(self):
        """Passing metas in different input orders → identical plan_hash.

        This is the property that lets multiple ranks compute the plan
        independently without coordinating order.
        """
        train_a = [
            _meta("w", (768, 2048), ((0, 192), (0, 2048)), 0, "train"),
            _meta("w", (768, 2048), ((192, 384), (0, 2048)), 1, "train"),
        ]
        train_b = list(reversed(train_a))
        rollout = [
            _meta("w", (768, 2048), ((0, 384), (0, 2048)), 4, "rollout"),
        ]
        h_a = build_transfer_plan(train_a, rollout).plan_hash
        h_b = build_transfer_plan(train_b, rollout).plan_hash
        assert h_a == h_b

    def test_sanity_check_cross_rank_pass(self):
        sanity_check_cross_rank("abc123", ["abc123", "abc123", "abc123"])  # no raise

    def test_sanity_check_cross_rank_fail(self):
        with pytest.raises(RuntimeError, match="disagreement"):
            sanity_check_cross_rank("abc123", ["abc123", "DIFFERENT", "abc123"])


# ============================================================
# Edge cases / error handling
# ============================================================


class TestEdgeCases:
    def test_missing_param_in_trainer_skipped(self):
        """Rollout wants a param trainer doesn't have → skipped, no error.

        (E.g. lm_head when tied embedding; sanity check is the caller's job.)
        """
        train = [
            _meta("a", (10, 10), ((0, 10), (0, 10)), 0, "train"),
        ]
        rollout = [
            _meta("a", (10, 10), ((0, 10), (0, 10)), 1, "rollout"),
            _meta("b", (10, 10), ((0, 10), (0, 10)), 1, "rollout"),  # not in trainer
        ]
        plan = build_transfer_plan(train, rollout)
        assert len(plan.edges) == 1
        assert plan.edges[0].param_name == "a"

    def test_shape_mismatch_raises(self):
        train = [_meta("w", (768, 2048), ((0, 768), (0, 2048)), 0, "train")]
        rollout = [_meta("w", (768, 4096), ((0, 768), (0, 4096)), 1, "rollout")]
        with pytest.raises(ValueError, match="full_shape mismatch"):
            build_transfer_plan(train, rollout)

    def test_dtype_mismatch_raises(self):
        train = [_meta("w", (768, 2048), ((0, 768), (0, 2048)), 0, "train", dtype="bfloat16")]
        rollout = [_meta("w", (768, 2048), ((0, 768), (0, 2048)), 1, "rollout", dtype="float16")]
        with pytest.raises(ValueError, match="dtype_str mismatch"):
            build_transfer_plan(train, rollout)

    def test_role_validation(self):
        # rollout list contains a train-tagged meta → should reject
        train = [_meta("w", (10, 10), ((0, 10), (0, 10)), 0, "train")]
        rollout = [_meta("w", (10, 10), ((0, 10), (0, 10)), 1, "train")]  # wrong role
        with pytest.raises(ValueError, match="non-rollout"):
            build_transfer_plan(train, rollout)


# ============================================================
# enrich_edge callback
# ============================================================


class TestEnrichEdge:
    def test_enrich_adds_vllm_metadata(self):
        train = [_meta("model.layers.0.mlp.experts.3.gate_proj.weight", (768, 2048), ((0, 768), (0, 2048)), 0, "train")]
        rollout = [
            _meta("model.layers.0.mlp.experts.3.gate_proj.weight", (768, 2048), ((0, 768), (0, 2048)), 1, "rollout")
        ]

        def enrich(proto: TransferEdge) -> TransferEdge:
            # Pretend we parse the param_name and figure out vLLM mapping
            import re

            m = re.search(r"experts\.(\d+)\.gate_proj", proto.param_name)
            expert_id = int(m.group(1)) if m else None
            return (
                TransferEdge(
                    **{**proto.to_dict(), "shape": proto.shape, "dtype_str": proto.dtype_str},
                    # to_dict / from_dict round trips lose the encoded tuples ─ use direct field replace:
                )
                if False
                else TransferEdge(
                    param_name=proto.param_name,
                    src_global_rank=proto.src_global_rank,
                    dst_global_rank=proto.dst_global_rank,
                    src_local_slice_encoded=proto.src_local_slice_encoded,
                    dst_local_slice_encoded=proto.dst_local_slice_encoded,
                    shape=proto.shape,
                    dtype_str=proto.dtype_str,
                    target_param_name="model.layers.0.mlp.experts.w13_weight",
                    shard_id="w1",
                    expert_id=expert_id,
                )
            )

        plan = build_transfer_plan(train, rollout, enrich_edge=enrich)
        assert len(plan.edges) == 1
        e = plan.edges[0]
        assert e.target_param_name == "model.layers.0.mlp.experts.w13_weight"
        assert e.shard_id == "w1"
        assert e.expert_id == 3


# ============================================================
# Serialization round-trip
# ============================================================


class TestSerialization:
    def test_plan_to_json_round_trip(self):
        train = [
            _meta("w", (768, 2048), ((0, 192), (0, 2048)), 0, "train"),
            _meta("w", (768, 2048), ((192, 384), (0, 2048)), 1, "train"),
        ]
        rollout = [
            _meta("w", (768, 2048), ((0, 384), (0, 2048)), 4, "rollout"),
        ]
        plan = build_transfer_plan(train, rollout)
        plan2 = TransferPlan.from_json(plan.to_json())
        assert plan.plan_hash == plan2.plan_hash
        assert plan.edges == plan2.edges
