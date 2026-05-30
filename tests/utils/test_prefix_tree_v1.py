# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""CPU-only unit tests for verl/utils/prefix_tree_v1.py.

Test surface:
  * TestV1BuildBasic         — V1 build correctness on depth-3 inputs
                               (parallels test_prefix_tree_magi.TestBuildPrefixTreeLayout)
  * TestV1ArbitraryDepth     — V1 reaches depth 4-5 where Meituan path caps at 3
  * TestV1RestoreRoundTrip   — restore_flat_to_nested round-trip via _leaf_ancestor_ranges
  * TestApplyV1Patch         — monkey-patch idempotency + kwarg handling (needs transformers)
  * TestPrefixTreeV1Forward  — mini Qwen2 forward through V1 path == dense baseline
                               (needs transformers + flex_attention)

Reuse pattern from test_prefix_tree_magi.py: imports inside test bodies so pytest
collection works in environments missing optional deps.
"""

from __future__ import annotations

import pytest
import torch

# All tests below need the full verl env (transformers triggers verl.__init__).
# Skip the entire module if missing — keeps `pytest --collect-only` clean on
# minimal CPU envs (e.g. the trie_construction benchmark machine).
pytest.importorskip("transformers")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_nested(token_lists: list[list[int]], device: str = "cpu") -> torch.Tensor:
    tensors = [torch.tensor(t, dtype=torch.long, device=device) for t in token_lists]
    return torch.nested.nested_tensor(tensors, layout=torch.jagged)


def _make_nested_default(token_lists: list[list[int]]) -> torch.Tensor:
    return _make_nested(token_lists)


def _reconstruct_per_sample(pt_batch, params) -> dict[int, torch.Tensor]:
    """Per-sample reconstruction from pt_batch + params (uses q/k_ranges)."""
    flat = pt_batch.flat_input_ids
    out: dict[int, torch.Tensor] = {}
    for leaf_idx, sample_idx in enumerate(pt_batch.leaf_to_sample):
        leaf_s, leaf_e = pt_batch.leaf_ranges[leaf_idx]
        ancestor_ranges = []
        for (qs, qe), (ks, ke), mtype in zip(params.q_ranges, params.k_ranges, params.mask_types, strict=False):
            if (qs, qe) == (leaf_s, leaf_e) and mtype == "full":
                ancestor_ranges.append((ks, ke))
        all_ranges = sorted(ancestor_ranges) + [(leaf_s, leaf_e)]
        out[sample_idx] = torch.cat([flat[s:e] for s, e in all_ranges])
    return out


# ---------------------------------------------------------------------------
# Core build correctness
# ---------------------------------------------------------------------------


class TestV1BuildBasic:
    """Verify V1 produces correct PrefixTreeMagiBatch on simple depth-3 inputs."""

    def test_depth3_two_groups_per_sample_reconstruction(self):
        from verl.utils.prefix_tree_v1 import (
            build_arbitrary_depth_params,
            convert_v1_trie_to_meituan,
            v1_greedy_build_tries,
        )

        # 4 samples in 2 groups of 2: [1,2,3] root → [100,200] mid → [101,102]/[201,202] leaves
        # and another group: [300,400] mid → [301,302]/[401,402] leaves
        samples = [
            [1, 2, 3, 100, 200, 101, 102],
            [1, 2, 3, 100, 200, 201, 202],
            [1, 2, 3, 300, 400, 301, 302],
            [1, 2, 3, 300, 400, 401, 402],
        ]
        sample_tensors = [torch.tensor(s, dtype=torch.long) for s in samples]
        tries, _ = v1_greedy_build_tries(samples, max_tokens_per_tree=10_000)
        assert len(tries) == 1, "All 4 samples share root [1,2,3], expect 1 forest"

        converted = convert_v1_trie_to_meituan(tries[0])
        assert converted is not None
        root_tn, node_info, leaves = converted
        assert len(leaves) == 4

        params = build_arbitrary_depth_params(sample_tensors, root_tn, node_info, leaves)
        # flat layout: root + group1_mid + leaf1 + leaf2 + group2_mid + leaf3 + leaf4
        expected_flat = torch.tensor([1, 2, 3, 100, 200, 101, 102, 201, 202, 300, 400, 301, 302, 401, 402])
        assert torch.equal(params.flat_tokens, expected_flat)
        assert params.prefix_range == (0, 3)
        assert len(params.leaf_ranges) == 4

        # _leaf_ancestor_ranges should be populated (depth-3 → 2 ancestors per leaf)
        anc = params._leaf_ancestor_ranges
        assert anc is not None
        assert len(anc) == 4
        for a in anc:
            assert len(a) == 2, "depth-3 leaf has 2 ancestors (root + mid)"

    def test_full_pipeline_via_micro_batch_v1(self):
        """End-to-end with build_prefix_tree_micro_batch_v1 on NestedTensor (no model needed for tree build)."""
        pytest.importorskip("transformers")  # build_prefix_tree_micro_batch_v1 transitively touches HF imports
        from verl.utils.prefix_tree_v1 import build_prefix_tree_micro_batch_v1

        samples = [
            [10, 20, 30, 41, 42, 43],
            [10, 20, 30, 51, 52, 53],
            [10, 20, 30, 61, 62, 63],
        ]
        nested = _make_nested(samples)
        # attention_type="flex" attempts BlockMask build — on Mac CPU it works but is slow.
        # Use a tiny example so this is fast.
        pytest.importorskip("magi_attention")
        pt_batch = build_prefix_tree_micro_batch_v1(model=None, input_ids=nested, attention_type="magi")
        assert pt_batch is not None
        # flat layout: [10,20,30] + 3 leaves of [4i,4i+1,4i+2]
        assert pt_batch.flat_input_ids.tolist() == [10, 20, 30, 41, 42, 43, 51, 52, 53, 61, 62, 63]
        assert pt_batch.prefix_range == (0, 3)
        assert sorted(pt_batch.leaf_to_sample) == [0, 1, 2]
        assert pt_batch.original_batch_size == 3

    def test_no_shared_prefix_returns_none(self):
        """No shared root → V1 packs samples as siblings under root; convert detects this and returns None."""
        from verl.utils.prefix_tree_v1 import (
            convert_v1_trie_to_meituan,
            v1_greedy_build_tries,
        )

        samples = [[1, 2, 3], [4, 5, 6]]  # no shared prefix
        tries, _ = v1_greedy_build_tries(samples, max_tokens_per_tree=10_000)
        # V1's greedy may pack both into one tree (as siblings) or two forests.
        # Either way, no single shared prefix exists.
        if len(tries) == 1:
            assert len(tries[0].children) >= 2, "no-share case should have ≥2 root children"
            assert convert_v1_trie_to_meituan(tries[0]) is None
        else:
            assert len(tries) == 2

    def test_single_sample_no_tree(self):
        from verl.utils.prefix_tree_v1 import (
            convert_v1_trie_to_meituan,
            v1_greedy_build_tries,
        )

        samples = [[1, 2, 3, 4, 5]]
        tries, _ = v1_greedy_build_tries(samples, max_tokens_per_tree=10_000)
        # Single sample → single leaf chain, no real sharing
        assert len(tries) == 1
        converted = convert_v1_trie_to_meituan(tries[0])
        # convert returns None when root has only 1 child that is itself a leaf
        assert converted is None


# ---------------------------------------------------------------------------
# V1 arbitrary-depth coverage (Meituan can't go past depth-3)
# ---------------------------------------------------------------------------


class TestV1ArbitraryDepth:
    """V1's headline capability: trees of arbitrary depth."""

    def test_depth4_balanced(self):
        """8 samples, real depth-4 (branch_factor=2 binary tree)."""
        from verl.utils.prefix_tree_v1 import (
            build_arbitrary_depth_params,
            convert_v1_trie_to_meituan,
            v1_greedy_build_tries,
        )

        # Build 8 samples that form a depth-4 binary tree:
        # root [1,2,3] → 2 chains → 2 sub-chains → 8 leaves
        prefix = [1, 2, 3]
        samples = []
        for g1 in range(2):  # 2 first-level branches
            for g2 in range(2):  # 2 second-level branches
                for g3 in range(2):  # 2 third-level branches (= leaves)
                    seq = (
                        prefix
                        + [100 + g1 * 10, 200 + g1 * 10]  # depth-2 seg
                        + [300 + g1 * 10 + g2, 400 + g1 * 10 + g2]  # depth-3 seg
                        + [500 + g1 * 100 + g2 * 10 + g3, 600 + g1 * 100 + g2 * 10 + g3]
                    )  # leaf
                    samples.append(seq)
        sample_tensors = [torch.tensor(s, dtype=torch.long) for s in samples]
        tries, _ = v1_greedy_build_tries(samples, max_tokens_per_tree=10_000)
        converted = convert_v1_trie_to_meituan(tries[0])
        assert converted is not None
        root_tn, node_info, leaves = converted

        def _max_depth(n):
            return 1 if not n.children else 1 + max(_max_depth(c) for c in n.children)

        assert _max_depth(root_tn) == 4, f"expected real depth-4, got {_max_depth(root_tn)}"
        assert len(leaves) == 8

        params = build_arbitrary_depth_params(sample_tensors, root_tn, node_info, leaves)
        # Each leaf should have 3 ancestors (root, depth-2 seg, depth-3 seg)
        for a in params._leaf_ancestor_ranges:
            assert len(a) == 3

        # Per-sample reconstruction must equal input
        recon = _reconstruct_per_sample(_FakePTBatch(params), params)
        for i, sample in enumerate(samples):
            assert torch.equal(recon[i], torch.tensor(sample, dtype=torch.long)), f"sample {i} reconstruction failed"


# ---------------------------------------------------------------------------
# restore_flat_to_nested round-trip
# ---------------------------------------------------------------------------


class TestV1RestoreRoundTrip:
    """V1 build → restore → original NestedTensor."""

    def test_restore_depth3(self):
        pytest.importorskip("transformers")
        from verl.utils.prefix_tree_magi import restore_flat_to_nested
        from verl.utils.prefix_tree_v1 import build_prefix_tree_micro_batch_v1

        samples = [
            [10, 20, 30, 41, 42, 43],
            [10, 20, 30, 51, 52, 53],
            [10, 20, 30, 61, 62, 63],
        ]
        nested = _make_nested(samples)
        pytest.importorskip("magi_attention")
        pt_batch = build_prefix_tree_micro_batch_v1(model=None, input_ids=nested, attention_type="magi")
        assert pt_batch is not None

        # Restore the flat_input_ids tensor itself (acts as identity test)
        restored = restore_flat_to_nested(pt_batch.flat_input_ids, pt_batch)
        for i, sample in enumerate(samples):
            assert restored[i].tolist() == sample, f"sample {i} restore mismatch"


# ---------------------------------------------------------------------------
# Magi backend registration (requires transformers; magi_attention optional)
# ---------------------------------------------------------------------------


class TestApplyMagiBackend:
    """apply_magi_prefix_tree_v1_backend should idempotently register Magi_Attention."""

    def test_idempotent_register(self):
        pytest.importorskip("transformers")
        pytest.importorskip("magi_attention")
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        from verl.models.transformers.monkey_patch import apply_magi_prefix_tree_v1_backend

        apply_magi_prefix_tree_v1_backend()
        assert "Magi_Attention" in ALL_ATTENTION_FUNCTIONS
        first_ref = ALL_ATTENTION_FUNCTIONS["Magi_Attention"]
        apply_magi_prefix_tree_v1_backend()  # idempotent
        assert ALL_ATTENTION_FUNCTIONS["Magi_Attention"] is first_ref


# ---------------------------------------------------------------------------
# Forward integration with mini HF model + Magi
# ---------------------------------------------------------------------------


class TestPrefixTreeV1Forward:
    """Mini Qwen2 forward via V1+Magi path must match dense baseline at sample positions."""

    def test_v1_magi_vs_dense_baseline(self):
        pytest.importorskip("transformers")
        pytest.importorskip("magi_attention")
        if not torch.cuda.is_available():
            pytest.skip("Magi FFA requires CUDA")

        # Initialize torch.distributed if not yet — torchrun sets env vars
        # (RANK/WORLD_SIZE/MASTER_ADDR/MASTER_PORT) but does not call
        # init_process_group itself. Magi's cp_group needs a real PG.
        import os

        if not torch.distributed.is_initialized():
            if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
                pytest.skip("Run via torchrun so RANK/WORLD_SIZE env vars are set")
            torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
            torch.distributed.init_process_group(backend="nccl")

        from transformers import Qwen2Config, Qwen2ForCausalLM

        from verl.models.transformers.monkey_patch import apply_magi_prefix_tree_v1_backend
        from verl.utils.prefix_tree_magi import restore_flat_to_nested
        from verl.utils.prefix_tree_v1 import prefix_tree_v1_forward

        # Register Magi_Attention BEFORE constructing the model — HF's
        # Qwen2ForCausalLM.__init__ validates _attn_implementation against
        # ALL_ATTENTION_FUNCTIONS, so we need the backend registered first.
        apply_magi_prefix_tree_v1_backend()

        torch.manual_seed(0)
        # head_dim must be 64 or 128 to hit Magi's AOT-prebuilt FFA kernels —
        # other head_dims trigger 1-3 min runtime JIT per shape, which slows
        # the test enough to look hung. hidden_size=256 + num_heads=4 → head_dim=64.
        cfg = Qwen2Config(
            vocab_size=256,
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=64,
            _attn_implementation="Magi_Attention",
        )
        model = Qwen2ForCausalLM(cfg).eval().cuda()

        # Attach cp_group to each attention layer (single-rank → WORLD)
        cp_group = torch.distributed.group.WORLD
        for name, mod in model.named_modules():
            if mod.__class__.__name__.lower().endswith(("attention", "self_attn", "selfattention")):
                mod.cp_group = cp_group

        samples = [
            [1, 2, 3, 4, 10, 11, 12, 13],
            [1, 2, 3, 4, 20, 21, 22, 23],
            [1, 2, 3, 4, 30, 31, 32, 33],
        ]
        nested = _make_nested(samples, device="cuda")

        with torch.no_grad():
            output, pt_batch = prefix_tree_v1_forward(model, nested, cp_group=cp_group)
        assert output is not None and pt_batch is not None
        flat_logits = output.logits[0]
        nested_logits = restore_flat_to_nested(flat_logits, pt_batch)

        # Dense baseline: switch model to vanilla SDPA attention temporarily and
        # forward each sample independently. Magi vs SDPA dense should agree
        # within bf16 tolerance.
        for cfg_ in (model.config,):
            cfg_._attn_implementation = "sdpa"
        for i, sample in enumerate(samples):
            inp = torch.tensor([sample], dtype=torch.long, device="cuda")
            with torch.no_grad():
                dense_out = model(input_ids=inp, attention_mask=None)
            dense_logits = dense_out.logits[0]  # (seq_len, vocab)

            tree_logits = nested_logits[i]  # (seq_len, vocab)
            max_diff = (tree_logits - dense_logits).abs().max().item()
            # bf16 tolerance — Magi FFA uses bf16 compute internally
            assert max_diff < 5e-2, f"sample {i} logits diverged: max abs diff = {max_diff}"


# ---------------------------------------------------------------------------
# Tiny shim — PrefixTreeMagiBatch-like for _reconstruct_per_sample helper
# ---------------------------------------------------------------------------


class _FakePTBatch:
    """Minimal duck-type for the reconstruct helper (only flat_input_ids/leaf_*/prefix_range used)."""

    def __init__(self, params):
        self.flat_input_ids = params.flat_tokens
        self.leaf_to_sample = params.leaf_to_sample
        self.leaf_ranges = params.leaf_ranges
        self.prefix_range = params.prefix_range
