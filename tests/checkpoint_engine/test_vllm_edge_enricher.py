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
"""Pure-CPU unit tests for ``vllm_enrich_edge``.

Confirms the regex-based pattern matching populates the right
``target_param_name`` / ``shard_id`` / ``expert_id`` for every kind of
HF-canonical name the M3 ``get_local_shards`` emits.
"""

from __future__ import annotations

from verl.checkpoint_engine.parallel_meta import ParameterShardMeta, TransferEdge
from verl.checkpoint_engine.transfer_plan import build_transfer_plan
from verl.checkpoint_engine.vllm_edge_enricher import vllm_enrich_edge


def _proto_edge(param_name: str) -> TransferEdge:
    """Build a TransferEdge with placeholder geometry. Only param_name matters
    for the enricher; the rest is irrelevant to the test.
    """
    return TransferEdge(
        param_name=param_name,
        src_global_rank=0,
        dst_global_rank=1,
        src_local_slice_encoded=((None, None, None),),
        dst_local_slice_encoded=((None, None, None),),
        shape=(1,),
        dtype_str="bfloat16",
    )


class TestMoEExperts:
    def test_gate_proj_w1(self):
        edge = vllm_enrich_edge(_proto_edge("model.layers.7.mlp.experts.42.gate_proj.weight"))
        assert edge.target_param_name == "model.layers.7.mlp.experts.w13_weight"
        assert edge.shard_id == "w1"
        assert edge.expert_id == 42

    def test_up_proj_w3(self):
        edge = vllm_enrich_edge(_proto_edge("model.layers.7.mlp.experts.42.up_proj.weight"))
        assert edge.target_param_name == "model.layers.7.mlp.experts.w13_weight"
        assert edge.shard_id == "w3"
        assert edge.expert_id == 42

    def test_down_proj_w2(self):
        edge = vllm_enrich_edge(_proto_edge("model.layers.7.mlp.experts.42.down_proj.weight"))
        assert edge.target_param_name == "model.layers.7.mlp.experts.w2_weight"
        assert edge.shard_id == "w2"
        assert edge.expert_id == 42

    def test_expert_idx_zero(self):
        edge = vllm_enrich_edge(_proto_edge("model.layers.0.mlp.experts.0.gate_proj.weight"))
        assert edge.expert_id == 0


class TestAttentionQKV:
    def test_q_proj(self):
        edge = vllm_enrich_edge(_proto_edge("model.layers.3.self_attn.q_proj.weight"))
        assert edge.target_param_name == "model.layers.3.self_attn.qkv_proj.weight"
        assert edge.shard_id == "q"
        assert edge.expert_id is None

    def test_k_proj(self):
        edge = vllm_enrich_edge(_proto_edge("model.layers.3.self_attn.k_proj.weight"))
        assert edge.shard_id == "k"

    def test_v_proj(self):
        edge = vllm_enrich_edge(_proto_edge("model.layers.3.self_attn.v_proj.weight"))
        assert edge.shard_id == "v"

    def test_q_proj_bias(self):
        # Qwen2 has q_proj.bias — vLLM fuses bias too (qkv_proj.bias).
        edge = vllm_enrich_edge(_proto_edge("model.layers.3.self_attn.q_proj.bias"))
        assert edge.target_param_name == "model.layers.3.self_attn.qkv_proj.bias"
        assert edge.shard_id == "q"

    def test_k_proj_bias(self):
        edge = vllm_enrich_edge(_proto_edge("model.layers.3.self_attn.k_proj.bias"))
        assert edge.target_param_name == "model.layers.3.self_attn.qkv_proj.bias"
        assert edge.shard_id == "k"


class TestFallthrough:
    def test_o_proj_1to1(self):
        edge = vllm_enrich_edge(_proto_edge("model.layers.3.self_attn.o_proj.weight"))
        assert edge.target_param_name == "model.layers.3.self_attn.o_proj.weight"
        assert edge.shard_id is None
        assert edge.expert_id is None

    def test_dense_mlp_gate_proj_merged(self):
        # Dense MLP: vLLM fuses gate+up into gate_up_proj (MergedColumnParallelLinear).
        edge = vllm_enrich_edge(_proto_edge("model.layers.3.mlp.gate_proj.weight"))
        assert edge.target_param_name == "model.layers.3.mlp.gate_up_proj.weight"
        assert edge.shard_id == 0
        assert edge.expert_id is None

    def test_dense_mlp_up_proj_merged(self):
        edge = vllm_enrich_edge(_proto_edge("model.layers.3.mlp.up_proj.weight"))
        assert edge.target_param_name == "model.layers.3.mlp.gate_up_proj.weight"
        assert edge.shard_id == 1

    def test_dense_mlp_down_proj_1to1(self):
        # down_proj is RowParallelLinear (single-tensor), so 1:1.
        edge = vllm_enrich_edge(_proto_edge("model.layers.3.mlp.down_proj.weight"))
        assert edge.target_param_name == "model.layers.3.mlp.down_proj.weight"
        assert edge.shard_id is None

    def test_layernorm_1to1(self):
        edge = vllm_enrich_edge(_proto_edge("model.layers.3.input_layernorm.weight"))
        assert edge.target_param_name == "model.layers.3.input_layernorm.weight"

    def test_router_1to1(self):
        edge = vllm_enrich_edge(_proto_edge("model.layers.3.mlp.gate.weight"))
        assert edge.target_param_name == "model.layers.3.mlp.gate.weight"

    def test_embedding_1to1(self):
        edge = vllm_enrich_edge(_proto_edge("model.embed_tokens.weight"))
        assert edge.target_param_name == "model.embed_tokens.weight"

    def test_shared_expert_gate_proj_merged(self):
        # shared_expert is its own MergedColumnParallelLinear in Qwen MoE —
        # not a routed expert, but still fused gate+up.
        edge = vllm_enrich_edge(_proto_edge("model.layers.3.mlp.shared_expert.gate_proj.weight"))
        assert edge.target_param_name == "model.layers.3.mlp.shared_expert.gate_up_proj.weight"
        assert edge.shard_id == 0


class TestIntegrationWithBuildTransferPlan:
    """build_transfer_plan(enrich_edge=vllm_enrich_edge) should produce
    plans where every edge is enriched in place."""

    def test_moe_plan_with_enricher(self):
        full = (16, 8)
        ranges = ((0, 16), (0, 8))
        train = [
            ParameterShardMeta(
                param_name=f"model.layers.0.mlp.experts.{e}.gate_proj.weight",
                full_shape=full,
                dtype_str="bfloat16",
                ranges=ranges,
                global_rank=0,
                role="train",
            )
            for e in range(2)
        ]
        rollout = [
            ParameterShardMeta(
                param_name=f"model.layers.0.mlp.experts.{e}.gate_proj.weight",
                full_shape=full,
                dtype_str="bfloat16",
                ranges=ranges,
                global_rank=1,
                role="rollout",
            )
            for e in range(2)
        ]
        plan = build_transfer_plan(train, rollout, enrich_edge=vllm_enrich_edge)
        assert len(plan.edges) == 2
        for e in plan.edges:
            assert e.shard_id == "w1"
            assert e.target_param_name == "model.layers.0.mlp.experts.w13_weight"
            assert e.expert_id in (0, 1)
