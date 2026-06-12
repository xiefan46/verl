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
"""Pure-CPU unit tests for the Megatron → HF local-shard transforms.

Each transform is checked by:
1. Constructing a tiny synthetic Megatron parameter with values that
   uniquely identify each row / column.
2. Running the local transform on that parameter.
3. Asserting that the emitted HF-format shards have the expected
   shapes, global boxes, AND values (so that re-stitching all TP
   ranks would reproduce the original Megatron tensor faithfully).

The "re-stitching" check is the bit-for-bit-equivalence guarantee
between sharded refit and broadcast refit (design doc §5.4 invariant).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from verl.workers.engine.megatron.sharded_export import (
    MegatronToHFContext,
    convert_fc1_to_gate_up,
    convert_fc2_to_down,
    convert_qkv_to_q_k_v,
    get_local_shards,
)

# ----------------------------------------------------------------------
# Synthetic HF configs
# ----------------------------------------------------------------------


def _dense_cfg(
    hidden: int = 16,
    intermediate: int = 32,
    num_heads: int = 4,
    num_kv_heads: int = 2,
    head_dim: int | None = None,
    vocab: int = 64,
) -> SimpleNamespace:
    return SimpleNamespace(
        hidden_size=hidden,
        intermediate_size=intermediate,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim,
        vocab_size=vocab,
        num_experts=None,
        moe_intermediate_size=None,
        shared_expert_intermediate_size=None,
    )


def _moe_cfg(
    hidden: int = 16,
    moe_intermediate: int = 8,
    num_heads: int = 4,
    num_kv_heads: int = 2,
    num_experts: int = 8,
    shared_expert_intermediate: int = 8,
    vocab: int = 64,
) -> SimpleNamespace:
    return SimpleNamespace(
        hidden_size=hidden,
        intermediate_size=moe_intermediate,  # fallback dense (unused)
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=None,
        vocab_size=vocab,
        num_experts=num_experts,
        moe_intermediate_size=moe_intermediate,
        shared_expert_intermediate_size=shared_expert_intermediate,
    )


# ----------------------------------------------------------------------
# convert_qkv_to_q_k_v
# ----------------------------------------------------------------------


class TestQKVConvert:
    def test_qkv_split_tp2_gqa(self):
        """num_heads=4, num_kv_heads=2, head_dim=8, hidden=16, tp=2.

        Per rank: groups_per_rank=1, queries_per_group=2.
        Megatron rank-0 shape: ((2 + 2) * 8 * 1, 16) = (32, 16).
        Splits to Q (16, 16), K (8, 16), V (8, 16).
        """
        hf = _dense_cfg(hidden=16, num_heads=4, num_kv_heads=2, head_dim=8)
        ctx = MegatronToHFContext(hf_config=hf, tp_rank=0, tp_size=2)

        # Build a Megatron qkv where each (group, slot, head_dim) cell
        # holds a unique value: 1000*group + 100*slot + head_dim_idx.
        # That makes it trivial to verify which rows ended up in q/k/v.
        groups_per_rank = 1
        qpg = 2  # queries_per_group
        head_dim = 8
        rows = (qpg + 2) * head_dim * groups_per_rank  # 32
        param = torch.zeros(rows, 16, dtype=torch.float32)
        for g in range(groups_per_rank):
            for slot in range(qpg + 2):
                for d in range(head_dim):
                    row = (g * (qpg + 2) + slot) * head_dim + d
                    param[row, :] = float(1000 * g + 100 * slot + d)

        outs = list(convert_qkv_to_q_k_v(ctx, param, layer_idx=0))
        names = [o[0] for o in outs]
        assert names == [
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.k_proj.weight",
            "model.layers.0.self_attn.v_proj.weight",
        ]

        q, q_box = outs[0][1], outs[0][2]
        k, k_box = outs[1][1], outs[1][2]
        v, v_box = outs[2][1], outs[2][2]

        # Shapes
        assert tuple(q.shape) == (16, 16)
        assert tuple(k.shape) == (8, 16)
        assert tuple(v.shape) == (8, 16)

        # Q rows should be slots 0..1 (queries_per_group=2) per group,
        # interleaved at the same head_dim positions.
        for q_idx in range(qpg):  # slot 0..1
            for d in range(head_dim):
                # The output Q is contiguous: groups_per_rank x qpg x head_dim flat.
                # Since groups_per_rank=1, row = q_idx*head_dim + d.
                row = q_idx * head_dim + d
                assert q[row, 0].item() == pytest.approx(100 * q_idx + d)

        # K rows: slot=2, head_dim 0..7
        for d in range(head_dim):
            assert k[d, 0].item() == pytest.approx(200 + d)
        # V rows: slot=3
        for d in range(head_dim):
            assert v[d, 0].item() == pytest.approx(300 + d)

        # Global boxes: full HF q/k/v shapes are (num_heads*head_dim, hidden)
        # = (32, 16) for q, (16, 16) for k,v. TP=2, rank=0 → first half.
        assert q_box == ((0, 16), (0, 16))  # first 16 rows of 32
        assert k_box == ((0, 8), (0, 16))
        assert v_box == ((0, 8), (0, 16))

    def test_qkv_tp_rank1_global_box(self):
        hf = _dense_cfg(hidden=16, num_heads=4, num_kv_heads=2, head_dim=8)
        ctx = MegatronToHFContext(hf_config=hf, tp_rank=1, tp_size=2)
        param = torch.zeros(32, 16)
        outs = list(convert_qkv_to_q_k_v(ctx, param, layer_idx=0))
        _, _, q_box = outs[0]
        _, _, k_box = outs[1]
        _, _, v_box = outs[2]
        # rank=1 → second half
        assert q_box == ((16, 32), (0, 16))
        assert k_box == ((8, 16), (0, 16))
        assert v_box == ((8, 16), (0, 16))


# ----------------------------------------------------------------------
# convert_fc1_to_gate_up
# ----------------------------------------------------------------------


class TestFC1GateUp:
    def test_fc1_dense_tp2(self):
        """intermediate=32, tp=2 → local_intermediate=16, fc1 shape (32, 16)."""
        hf = _dense_cfg(hidden=16, intermediate=32)
        ctx = MegatronToHFContext(hf_config=hf, tp_rank=0, tp_size=2)
        # Gate rows 0..15 = value 1..16, up rows 16..31 = value 100..115.
        param = torch.zeros(32, 16)
        for r in range(16):
            param[r, :] = float(r + 1)
        for r in range(16, 32):
            param[r, :] = float(100 + (r - 16))

        outs = list(
            convert_fc1_to_gate_up(
                ctx,
                param,
                "gate.weight",
                "up.weight",
                intermediate_size=32,
            )
        )
        assert outs[0][0] == "gate.weight"
        assert outs[1][0] == "up.weight"
        gate, gate_box = outs[0][1], outs[0][2]
        up, up_box = outs[1][1], outs[1][2]
        assert tuple(gate.shape) == (16, 16)
        assert tuple(up.shape) == (16, 16)
        # Gate values 1..16
        for r in range(16):
            assert gate[r, 0].item() == pytest.approx(r + 1)
        # Up values 100..115
        for r in range(16):
            assert up[r, 0].item() == pytest.approx(100 + r)
        # Boxes: rank=0, first half of (32, 16)
        assert gate_box == ((0, 16), (0, 16))
        assert up_box == ((0, 16), (0, 16))

    def test_fc1_wrong_shape_raises(self):
        hf = _dense_cfg(hidden=16, intermediate=32)
        ctx = MegatronToHFContext(hf_config=hf, tp_rank=0, tp_size=2)
        with pytest.raises(ValueError, match="linear_fc1.weight shape"):
            list(
                convert_fc1_to_gate_up(
                    ctx,
                    torch.zeros(33, 16),  # wrong row count
                    "g",
                    "u",
                    intermediate_size=32,
                )
            )


# ----------------------------------------------------------------------
# convert_fc2_to_down
# ----------------------------------------------------------------------


class TestFC2Down:
    def test_fc2_dense_tp2(self):
        """intermediate=32, tp=2 → fc2 shape (16, 16) = (hidden, intermediate/tp)."""
        hf = _dense_cfg(hidden=16, intermediate=32)
        ctx = MegatronToHFContext(hf_config=hf, tp_rank=1, tp_size=2)
        param = torch.ones(16, 16)
        outs = list(convert_fc2_to_down(ctx, param, "down.weight", intermediate_size=32))
        assert outs[0][0] == "down.weight"
        down, box = outs[0][1], outs[0][2]
        assert tuple(down.shape) == (16, 16)
        # rank=1 → dim 1 second half
        assert box == ((0, 16), (16, 32))


# ----------------------------------------------------------------------
# get_local_shards integration
# ----------------------------------------------------------------------


def _make_module(named_params: list[tuple[str, torch.Tensor]]) -> torch.nn.Module:
    """Build a stub Module exposing the given named_parameters().

    We can't subclass nn.Module easily for arbitrary names with dots,
    so just override ``named_parameters`` to return our list directly.
    """

    class _Stub(torch.nn.Module):
        def named_parameters(self, *args, **kw):  # type: ignore[override]
            return iter(named_params)

    return _Stub()


class TestGetLocalShardsIntegration:
    def test_dense_layer_full_dispatch(self):
        """Mini dense layer: input_layernorm + linear_qkv + linear_proj + linear_fc1 + linear_fc2."""
        hf = _dense_cfg(hidden=16, intermediate=32, num_heads=4, num_kv_heads=2, head_dim=8)
        ctx = MegatronToHFContext(hf_config=hf, tp_rank=0, tp_size=2)

        params = [
            ("module.decoder.layers.0.input_layernorm.weight", torch.ones(16)),
            ("module.decoder.layers.0.self_attention.linear_qkv.weight", torch.zeros(32, 16)),
            ("module.decoder.layers.0.self_attention.linear_proj.weight", torch.zeros(16, 8)),
            ("module.decoder.layers.0.pre_mlp_layernorm.weight", torch.ones(16)),
            ("module.decoder.layers.0.mlp.linear_fc1.weight", torch.zeros(32, 16)),
            ("module.decoder.layers.0.mlp.linear_fc2.weight", torch.zeros(16, 16)),
        ]
        module = _make_module(params)

        results = list(get_local_shards(module, ctx))
        names = [r[0] for r in results]
        assert "model.layers.0.input_layernorm.weight" in names
        assert "model.layers.0.self_attn.q_proj.weight" in names
        assert "model.layers.0.self_attn.k_proj.weight" in names
        assert "model.layers.0.self_attn.v_proj.weight" in names
        assert "model.layers.0.self_attn.o_proj.weight" in names
        assert "model.layers.0.post_attention_layernorm.weight" in names
        assert "model.layers.0.mlp.gate_proj.weight" in names
        assert "model.layers.0.mlp.up_proj.weight" in names
        assert "model.layers.0.mlp.down_proj.weight" in names

    def test_moe_layer_per_expert(self):
        """MoE layer with 2 experts per rank (EP=2 over 4 total experts)."""
        hf = _moe_cfg(hidden=16, moe_intermediate=8, num_experts=4, shared_expert_intermediate=8)
        ctx = MegatronToHFContext(hf_config=hf, tp_rank=0, tp_size=1, ep_rank=0, ep_size=2)

        params = [
            ("module.decoder.layers.0.mlp.router.weight", torch.zeros(4, 16)),
            # Local expert 0 (global 0) and 1 (global 1) — EP=2, rank 0 holds 0,1
            ("module.decoder.layers.0.mlp.experts.linear_fc1.weight0", torch.zeros(16, 16)),
            ("module.decoder.layers.0.mlp.experts.linear_fc1.weight1", torch.zeros(16, 16)),
            ("module.decoder.layers.0.mlp.experts.linear_fc2.weight0", torch.zeros(16, 8)),
            ("module.decoder.layers.0.mlp.experts.linear_fc2.weight1", torch.zeros(16, 8)),
            ("module.decoder.layers.0.mlp.shared_experts.linear_fc1.weight", torch.zeros(16, 16)),
            ("module.decoder.layers.0.mlp.shared_experts.linear_fc2.weight", torch.zeros(16, 8)),
        ]
        module = _make_module(params)
        results = list(get_local_shards(module, ctx))
        names = [r[0] for r in results]

        # Router: ungated, no TP split
        assert "model.layers.0.mlp.gate.weight" in names
        # Routed experts: 0/1 (global)
        assert "model.layers.0.mlp.experts.0.gate_proj.weight" in names
        assert "model.layers.0.mlp.experts.0.up_proj.weight" in names
        assert "model.layers.0.mlp.experts.0.down_proj.weight" in names
        assert "model.layers.0.mlp.experts.1.gate_proj.weight" in names
        # Shared expert
        assert "model.layers.0.mlp.shared_expert.gate_proj.weight" in names
        assert "model.layers.0.mlp.shared_expert.down_proj.weight" in names

    def test_moe_ep_rank1_expert_id_offset(self):
        """EP=2, rank 1 → local expert 0 is global expert 2 (4 experts / 2 ranks)."""
        hf = _moe_cfg(num_experts=4, moe_intermediate=8)
        ctx = MegatronToHFContext(hf_config=hf, tp_rank=0, tp_size=1, ep_rank=1, ep_size=2)
        params = [
            ("module.decoder.layers.0.mlp.experts.linear_fc1.weight0", torch.zeros(16, 16)),
            ("module.decoder.layers.0.mlp.experts.linear_fc1.weight1", torch.zeros(16, 16)),
        ]
        module = _make_module(params)
        results = list(get_local_shards(module, ctx))
        names = [r[0] for r in results]
        # local 0 → global 2, local 1 → global 3
        assert "model.layers.0.mlp.experts.2.gate_proj.weight" in names
        assert "model.layers.0.mlp.experts.3.gate_proj.weight" in names
        assert "model.layers.0.mlp.experts.0.gate_proj.weight" not in names
        assert "model.layers.0.mlp.experts.1.gate_proj.weight" not in names

    def test_strip_vl_prefix(self):
        """`module.language_model.decoder.layers.0.` should strip cleanly (Qwen3-VL)."""
        hf = _dense_cfg()
        ctx = MegatronToHFContext(hf_config=hf, tp_rank=0, tp_size=1)
        params = [
            ("module.language_model.decoder.layers.0.input_layernorm.weight", torch.ones(16)),
        ]
        module = _make_module(params)
        results = list(get_local_shards(module, ctx))
        assert results[0][0] == "model.layers.0.input_layernorm.weight"

    def test_embedding_dispatch(self):
        hf = _dense_cfg(hidden=16, vocab=64)
        ctx = MegatronToHFContext(hf_config=hf, tp_rank=0, tp_size=2)
        # vocab=64 / tp=2 → local vocab_per_rank=32
        params = [
            ("module.embedding.word_embeddings.weight", torch.zeros(32, 16)),
        ]
        module = _make_module(params)
        results = list(get_local_shards(module, ctx))
        assert results[0][0] == "model.embed_tokens.weight"
        # Global box on dim 0: rank 0 → first 32 of 64
        assert results[0][2] == ((0, 32), (0, 16))

    def test_unhandled_falls_through_with_warning(self, caplog):
        hf = _dense_cfg()
        ctx = MegatronToHFContext(hf_config=hf, tp_rank=0, tp_size=1)
        params = [
            ("module.decoder.layers.0.self_attention.dt_bias", torch.zeros(16)),  # Mamba SSM
        ]
        module = _make_module(params)
        with caplog.at_level("WARNING"):
            results = list(get_local_shards(module, ctx))
        assert results[0][0] == "model.layers.0.self_attention.dt_bias"
        assert any("unhandled" in r.message for r in caplog.records)
