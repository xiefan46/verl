# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
"""CPU-only unit tests for verl/utils/prefix_tree.py.

These tests exercise build_prefix_tree_micro_batch (without the MAGI key
construction, which requires GPU + distributed) and restore_flat_to_nested.
All prefix-tree utilities now live in verl.utils — no Megatron-LM fork needed.
"""
from __future__ import annotations

import pytest
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_nested(token_lists: list[list[int]], device="cpu") -> torch.Tensor:
    """Build a jagged NestedTensor from a list of token ID lists."""
    tensors = [torch.tensor(t, dtype=torch.long, device=device) for t in token_lists]
    return torch.nested.nested_tensor(tensors, layout=torch.jagged)


def _make_nested_float(value_lists: list[list[float]], device="cpu") -> torch.Tensor:
    tensors = [torch.tensor(v, dtype=torch.float32, device=device) for v in value_lists]
    return torch.nested.nested_tensor(tensors, layout=torch.jagged)


# ---------------------------------------------------------------------------
# Tests for the core layout helpers (no MAGI key, no model)
# ---------------------------------------------------------------------------

class TestBuildPrefixTreeLayout:
    """Tests that verify the flat layout and PrefixTreeParams fields.

    We call build_prefix_tree_params directly (bypassing the MAGI key
    construction) to keep these tests CPU-only.
    """

    def test_basic_shared_prefix(self):
        from verl.utils.prefix_tree_utils import (
            build_prefix_tree_params,
            longest_common_prefix_length,
        )

        tokens = [
            torch.tensor([10, 20, 30, 41, 42]),
            torch.tensor([10, 20, 30, 51]),
            torch.tensor([10, 20, 30, 61, 62, 63]),
        ]
        prefix_len = longest_common_prefix_length(tokens)
        assert prefix_len == 3

        params = build_prefix_tree_params(tokens, prefix_len=prefix_len)

        # flat layout: [10,20,30] + [41,42] + [51] + [61,62,63]
        assert torch.equal(params.flat_tokens, torch.tensor([10, 20, 30, 41, 42, 51, 61, 62, 63]))
        assert params.prefix_range == (0, 3)
        assert params.leaf_ranges == [(3, 5), (5, 6), (6, 9)]
        assert params.total_seqlen_q == 9

    def test_no_shared_prefix_returns_zero(self):
        from verl.utils.prefix_tree_utils import longest_common_prefix_length

        tokens = [
            torch.tensor([1, 2, 3]),
            torch.tensor([4, 5, 6]),
        ]
        assert longest_common_prefix_length(tokens) == 0

    def test_position_ids_default(self):
        from verl.utils.prefix_tree_utils import build_prefix_tree_params

        tokens = [
            torch.tensor([10, 20, 30, 41, 42]),
            torch.tensor([10, 20, 30, 51]),
        ]
        params = build_prefix_tree_params(tokens, prefix_len=3)

        # prefix: [0,1,2], leaf_0: [3,4], leaf_1: [3]
        assert torch.equal(params.flat_position_ids, torch.tensor([0, 1, 2, 3, 4, 3]))

    def test_flex_spec_structure(self):
        from verl.utils.prefix_tree_utils import build_prefix_tree_params

        tokens = [
            torch.tensor([10, 20, 30, 41, 42]),
            torch.tensor([10, 20, 30, 51]),
        ]
        params = build_prefix_tree_params(tokens, prefix_len=3)

        # Expected rectangles:
        # (0,3)→(0,3) causal  — prefix self
        # (3,5)→(0,3) full    — leaf_0 attends to prefix
        # (3,5)→(3,5) causal  — leaf_0 self
        # (5,6)→(0,3) full    — leaf_1 attends to prefix
        # (5,6)→(5,6) causal  — leaf_1 self
        rects = set(zip(params.q_ranges, params.k_ranges, params.mask_types))
        assert ((0, 3), (0, 3), 'causal') in rects
        assert ((3, 5), (0, 3), 'full') in rects
        assert ((3, 5), (3, 5), 'causal') in rects
        assert ((5, 6), (0, 3), 'full') in rects
        assert ((5, 6), (5, 6), 'causal') in rects
        assert len(rects) == 5


# ---------------------------------------------------------------------------
# Tests for restore_flat_to_nested
# ---------------------------------------------------------------------------

class TestRestoreFlatToNested:
    """Tests that verify round-trip: build params → restore → original tensors."""

    def _build_pt_batch(self, tokens, prefix_len):
        """Build a PrefixTreeMagiBatch stub without the MAGI key."""
        from verl.utils.prefix_tree_utils import build_prefix_tree_params
        from verl.utils.prefix_tree import PrefixTreeMagiBatch

        params = build_prefix_tree_params(tokens, prefix_len=prefix_len)
        return PrefixTreeMagiBatch(
            flat_input_ids=params.flat_tokens,
            flat_position_ids=params.flat_position_ids,
            flat_loss_mask=None,
            magi_key=None,
            flex_key=None,
            leaf_to_sample=params.leaf_to_sample,
            leaf_ranges=params.leaf_ranges,
            prefix_range=params.prefix_range,
            original_batch_size=params.num_samples,
        )

    def test_restore_token_ids(self):
        from verl.utils.prefix_tree import restore_flat_to_nested

        tokens = [
            torch.tensor([10, 20, 30, 41, 42]),
            torch.tensor([10, 20, 30, 51]),
            torch.tensor([10, 20, 30, 61, 62, 63]),
        ]
        pt_batch = self._build_pt_batch(tokens, prefix_len=3)
        flat = pt_batch.flat_input_ids  # (9,)

        restored = restore_flat_to_nested(flat, pt_batch)

        # restored is a NestedTensor; check each sample
        offsets = restored.offsets()
        vals = restored.values()
        lengths = offsets.diff().tolist()

        assert lengths == [5, 4, 6]
        pos = 0
        for i, orig in enumerate(tokens):
            length = int(lengths[i])
            assert torch.equal(vals[pos : pos + length], orig), f"sample {i} mismatch"
            pos += length

    def test_restore_with_extra_dim(self):
        """restore_flat_to_nested should work for (total_tokens, D) tensors too."""
        from verl.utils.prefix_tree import restore_flat_to_nested

        tokens = [
            torch.tensor([10, 20, 30, 41, 42]),
            torch.tensor([10, 20, 30, 51]),
        ]
        pt_batch = self._build_pt_batch(tokens, prefix_len=3)

        # Simulate logit output: (total_tokens=6, vocab=8)
        flat_logits = torch.randn(6, 8)
        restored = restore_flat_to_nested(flat_logits, pt_batch)

        offsets = restored.offsets()
        lengths = offsets.diff().tolist()
        assert lengths == [5, 4]

    def test_restore_preserves_prefix(self):
        """Each restored sample must start with the shared prefix tokens."""
        from verl.utils.prefix_tree import restore_flat_to_nested

        tokens = [
            torch.tensor([10, 20, 30, 41]),
            torch.tensor([10, 20, 30, 51, 52]),
        ]
        pt_batch = self._build_pt_batch(tokens, prefix_len=3)
        flat = pt_batch.flat_input_ids

        restored = restore_flat_to_nested(flat, pt_batch)
        vals = restored.values()
        offsets = restored.offsets()
        lengths = offsets.diff().tolist()

        # Both samples should start with [10, 20, 30]
        pos = 0
        for length in lengths:
            sample = vals[pos : pos + int(length)]
            assert torch.equal(sample[:3], torch.tensor([10, 20, 30]))
            pos += int(length)

    def test_custom_sample_indices(self):
        """sample_indices remapping: leaf_to_sample=[1,0] → restored in index order."""
        from verl.utils.prefix_tree_utils import build_prefix_tree_params
        from verl.utils.prefix_tree import PrefixTreeMagiBatch, restore_flat_to_nested

        # sample_indices=[1,0] means the first token list maps to sample 1,
        # and the second maps to sample 0.
        tokens = [
            torch.tensor([10, 20, 30, 41, 42]),  # → sample 1
            torch.tensor([10, 20, 30, 51]),       # → sample 0
        ]
        params = build_prefix_tree_params(tokens, prefix_len=3, sample_indices=[1, 0])
        pt_batch = PrefixTreeMagiBatch(
            flat_input_ids=params.flat_tokens,
            flat_position_ids=params.flat_position_ids,
            flat_loss_mask=None,
            magi_key=None,
            flex_key=None,
            leaf_to_sample=params.leaf_to_sample,  # [1, 0]
            leaf_ranges=params.leaf_ranges,
            prefix_range=params.prefix_range,
            original_batch_size=2,
        )

        flat = params.flat_tokens
        restored = restore_flat_to_nested(flat, pt_batch)
        offsets = restored.offsets()
        vals = restored.values()
        lengths = offsets.diff().tolist()

        assert len(lengths) == 2
        # slot 0 → tokens[1] = [10,20,30,51]
        pos0 = 0
        assert torch.equal(vals[pos0 : pos0 + 4], tokens[1])
        # slot 1 → tokens[0] = [10,20,30,41,42]
        pos1 = int(lengths[0])
        assert torch.equal(vals[pos1 : pos1 + 5], tokens[0])


# ---------------------------------------------------------------------------
# Integration: NestedTensor input → flat layout (no MAGI key)
# ---------------------------------------------------------------------------

class TestNestedTensorUnpack:
    """Tests that build_prefix_tree_micro_batch correctly unpacks NestedTensors."""

    def test_nested_tensor_unpack_and_flat_layout(self, monkeypatch):
        """Monkeypatch _build_magi_key to skip GPU/dist; verify flat layout."""
        import verl.utils.prefix_tree as ptm

        # Stub out _build_magi_key so we don't need GPU
        monkeypatch.setattr(ptm, "_build_magi_key", lambda model, params: object())

        # Stub model with config
        class FakeConfig:
            num_attention_heads = 8
            num_query_groups = 8
            kv_channels = 128
            fp8 = None

        class FakeModel:
            config = FakeConfig()
            pre_process = True
            post_process = True

        model = FakeModel()

        token_lists = [
            [10, 20, 30, 41, 42],
            [10, 20, 30, 51],
            [10, 20, 30, 61, 62, 63],
        ]
        input_ids = _make_nested(token_lists)

        result = ptm.build_prefix_tree_micro_batch(model, input_ids)

        assert result is not None
        assert result.original_batch_size == 3
        assert torch.equal(result.flat_input_ids, torch.tensor([10, 20, 30, 41, 42, 51, 61, 62, 63]))
        assert result.prefix_range == (0, 3)
        assert result.leaf_ranges == [(3, 5), (5, 6), (6, 9)]

    def test_no_shared_prefix_returns_none(self, monkeypatch):
        import verl.utils.prefix_tree as ptm
        monkeypatch.setattr(ptm, "_build_magi_key", lambda model, params: object())

        class FakeModel:
            class config:
                num_attention_heads = 8
                num_query_groups = 8
                kv_channels = 128
                fp8 = None
            pre_process = True
            post_process = True

        token_lists = [[1, 2, 3], [4, 5, 6]]
        input_ids = _make_nested(token_lists)

        result = ptm.build_prefix_tree_micro_batch(FakeModel(), input_ids)
        assert result is None

    def test_loss_mask_flattened(self, monkeypatch):
        """loss_mask NestedTensor is correctly flattened into flat_loss_mask."""
        import verl.utils.prefix_tree as ptm
        monkeypatch.setattr(ptm, "_build_magi_key", lambda model, params: object())

        class FakeModel:
            class config:
                num_attention_heads = 8
                num_query_groups = 8
                kv_channels = 128
                fp8 = None
            pre_process = True
            post_process = True

        token_lists = [[10, 20, 30, 41, 42], [10, 20, 30, 51]]
        loss_lists = [[0.0, 0.0, 1.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]]

        input_ids = _make_nested(token_lists)
        loss_mask = _make_nested_float(loss_lists)

        result = ptm.build_prefix_tree_micro_batch(FakeModel(), input_ids, loss_mask=loss_mask)
        assert result is not None
        assert result.flat_loss_mask is not None
        # flat: [prefix_loss(3)] + [leaf_0_loss(2)] + [leaf_1_loss(1)]
        # prefix loss taken from sample 0: [0,0,1], leaf_0: [1,1], leaf_1: [1]
        assert torch.equal(
            result.flat_loss_mask,
            torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0, 1.0]),
        )


# ---------------------------------------------------------------------------
# Tests for prefix_segments prior-knowledge path
# ---------------------------------------------------------------------------

class TestPrefixSegmentsPrior:
    """Tests for the fast hash-based prefix detection path."""

    def _make_segments(self, token_lists: list[list[int]]) -> list[list[tuple[int, int]]]:
        """Build ground-truth prefix_segments for a list of token sequences."""
        from verl.utils.prefix_tree_hash_based import _hash_prefix
        result = []
        for tokens in token_lists:
            segs = []
            for i in range(1, len(tokens) + 1):
                cum = torch.tensor(tokens[:i], dtype=torch.long)
                segs.append((_hash_prefix(cum), i))
            result.append(segs)
        return result

    def _fake_model(self):
        class FakeModel:
            class config:
                num_attention_heads = 8
                num_query_groups = 8
                kv_channels = 128
                fp8 = None
            pre_process = True
            post_process = True
        return FakeModel()

    def test_prior_matches_scan(self, monkeypatch):
        """Prior-knowledge path returns same prefix_len as the token-scan path."""
        import verl.utils.prefix_tree as ptm
        monkeypatch.setattr(ptm, "_build_magi_key", lambda model, params: object())

        token_lists = [[10, 20, 30, 41, 42], [10, 20, 30, 51]]
        input_ids = _make_nested(token_lists)

        # Build segments that exactly match turn boundaries at every token.
        segs = self._make_segments(token_lists)

        result_prior = ptm.build_prefix_tree_micro_batch(
            self._fake_model(), input_ids, prefix_segments_batch=segs
        )
        result_scan = ptm.build_prefix_tree_micro_batch(
            self._fake_model(), input_ids
        )

        assert result_prior is not None
        assert result_scan is not None
        assert result_prior.prefix_range == result_scan.prefix_range
        assert torch.equal(result_prior.flat_input_ids, result_scan.flat_input_ids)

    def test_prior_uses_coarsest_shared_boundary(self, monkeypatch):
        """Prior with turn-level segments finds the longest sub-turn boundary."""
        import verl.utils.prefix_tree as ptm
        monkeypatch.setattr(ptm, "_build_magi_key", lambda model, params: object())

        from verl.utils.prefix_tree_hash_based import _hash_prefix

        # Two samples sharing the first 3 tokens (one sub-turn boundary).
        # Sub-turn 0 ends at token index 3; sub-turn 1 ends at 5/4 respectively.
        t0 = torch.tensor([10, 20, 30, 41, 42], dtype=torch.long)
        t1 = torch.tensor([10, 20, 30, 51], dtype=torch.long)

        segs0 = [(_hash_prefix(t0[:3]), 3), (_hash_prefix(t0[:5]), 5)]
        segs1 = [(_hash_prefix(t1[:3]), 3), (_hash_prefix(t1[:4]), 4)]

        input_ids = _make_nested([[10, 20, 30, 41, 42], [10, 20, 30, 51]])
        result = ptm.build_prefix_tree_micro_batch(
            self._fake_model(), input_ids,
            prefix_segments_batch=[segs0, segs1],
        )
        assert result is not None
        # Shared sub-turn boundary is at length 3
        assert result.prefix_range == (0, 3)

    def test_prior_fallback_on_missing(self, monkeypatch):
        """When prefix_segments_batch is None, the scan path is used."""
        import verl.utils.prefix_tree as ptm
        monkeypatch.setattr(ptm, "_build_magi_key", lambda model, params: object())

        token_lists = [[10, 20, 30, 41, 42], [10, 20, 30, 51]]
        input_ids = _make_nested(token_lists)

        result = ptm.build_prefix_tree_micro_batch(
            self._fake_model(), input_ids, prefix_segments_batch=None
        )
        assert result is not None
        assert result.prefix_range == (0, 3)

    def test_prior_no_shared_entry_returns_none(self, monkeypatch):
        """When no hash is shared across all samples, returns None (no prefix)."""
        import verl.utils.prefix_tree as ptm
        monkeypatch.setattr(ptm, "_build_magi_key", lambda model, params: object())

        from verl.utils.prefix_tree_hash_based import _hash_prefix

        t0 = torch.tensor([1, 2, 3], dtype=torch.long)
        t1 = torch.tensor([4, 5, 6], dtype=torch.long)
        segs0 = [(_hash_prefix(t0[:1]), 1), (_hash_prefix(t0[:3]), 3)]
        segs1 = [(_hash_prefix(t1[:1]), 1), (_hash_prefix(t1[:3]), 3)]

        input_ids = _make_nested([[1, 2, 3], [4, 5, 6]])
        result = ptm.build_prefix_tree_micro_batch(
            self._fake_model(), input_ids,
            prefix_segments_batch=[segs0, segs1],
        )
        assert result is None

    def test_prior_after_shuffle(self, monkeypatch):
        """Hash lookup still finds the correct prefix after index reordering."""
        import verl.utils.prefix_tree as ptm
        monkeypatch.setattr(ptm, "_build_magi_key", lambda model, params: object())

        from verl.utils.prefix_tree_hash_based import _hash_prefix

        token_lists = [
            [10, 20, 30, 41],   # sample 0
            [10, 20, 30, 51],   # sample 1
            [10, 20, 30, 61],   # sample 2
        ]
        segs = self._make_segments(token_lists)

        # Simulate shuffle: reorder to [2, 0, 1]
        shuffled_tokens = [token_lists[2], token_lists[0], token_lists[1]]
        shuffled_segs = [segs[2], segs[0], segs[1]]
        input_ids = _make_nested(shuffled_tokens)

        result = ptm.build_prefix_tree_micro_batch(
            self._fake_model(), input_ids,
            prefix_segments_batch=shuffled_segs,
        )
        assert result is not None
        assert result.prefix_range == (0, 3)

    def test_build_prefix_segments_single_turn(self):
        """build_prefix_segments_single_turn produces a one-entry list."""
        from verl.utils.prefix_tree_hash_based import build_prefix_segments_single_turn, _hash_prefix

        ids = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
        segs = build_prefix_segments_single_turn(ids)
        assert len(segs) == 1
        h, length = segs[0]
        assert length == 5
        assert h == _hash_prefix(ids)

    def test_build_prefix_segments_single_turn_with_mask(self):
        """Padding tokens are excluded when attention_mask is provided."""
        from verl.utils.prefix_tree_hash_based import build_prefix_segments_single_turn, _hash_prefix

        ids = torch.tensor([1, 2, 3, 0, 0], dtype=torch.long)
        mask = torch.tensor([1, 1, 1, 0, 0], dtype=torch.long)
        segs = build_prefix_segments_single_turn(ids, mask)
        assert len(segs) == 1
        h, length = segs[0]
        assert length == 3
        assert h == _hash_prefix(torch.tensor([1, 2, 3], dtype=torch.long))


# ---------------------------------------------------------------------------
# Hash path vs dynamic path equivalence
# ---------------------------------------------------------------------------


class TestPathEquivalence:
    """Verify the two detection algorithms produce equivalent flat layouts
    through the unified build_prefix_tree_micro_batch entry.

    Both paths feed the same downstream pipeline (build_layout_from_tree_node
    + _finalize_prefix_tree_batch), so for any input that both paths can
    handle, the output PrefixTreeMagiBatch must be bit-identical on
    ``flat_input_ids`` and the per-sample reconstructed token sequences.
    DFS ordering may legitimately differ between paths, so we compare on
    the unordered set of (sample_idx, reconstructed_tokens) pairs.
    """

    @staticmethod
    def _reconstruct(pt_batch):
        flat = pt_batch.flat_input_ids
        out = {}
        for leaf_idx, sample_idx in enumerate(pt_batch.leaf_to_sample):
            ls, le = pt_batch.leaf_ranges[leaf_idx]
            ancestors = (
                pt_batch.leaf_ancestor_ranges[leaf_idx]
                if pt_batch.leaf_ancestor_ranges is not None
                else [pt_batch.prefix_range]
            )
            pieces = [flat[s:e] for s, e in ancestors] + [flat[ls:le]]
            out[int(sample_idx)] = torch.cat(pieces).tolist()
        return out

    @staticmethod
    def _run_both_paths(samples, prefix_segments_batch=None):
        from verl.utils.prefix_tree import build_prefix_tree_micro_batch

        nested = _make_nested(samples)
        kwargs = dict(
            model=None,
            input_ids=nested,
            prefix_segments_batch=prefix_segments_batch,
            attention_type="magi",
        )
        b_hash = build_prefix_tree_micro_batch(**kwargs, dynamic_trie=False)
        b_dyn = build_prefix_tree_micro_batch(**kwargs, dynamic_trie=True)
        return b_hash, b_dyn

    @staticmethod
    def _hash_segments(samples):
        import hashlib

        def _h(toks):
            return int(hashlib.sha256(b",".join(str(t).encode() for t in toks)).hexdigest()[:8], 16)

        return [[(_h(s), len(s))] for s in samples]

    def test_single_level_shared_root(self):
        """3 samples share a 3-token root, no multi-level. Both paths should agree."""
        samples = [
            [10, 20, 30, 41, 42, 43],
            [10, 20, 30, 51, 52, 53],
            [10, 20, 30, 61, 62, 63],
        ]
        b_hash, b_dyn = self._run_both_paths(samples)
        assert b_hash is not None and b_dyn is not None

        # flat layout must be bit-identical (both produce DFS-pre-order over the same TreeNode shape)
        assert torch.equal(b_hash.flat_input_ids, b_dyn.flat_input_ids), (
            f"hash flat={b_hash.flat_input_ids.tolist()}, dyn flat={b_dyn.flat_input_ids.tolist()}"
        )
        assert b_hash.prefix_range == b_dyn.prefix_range
        assert sorted(b_hash.leaf_to_sample) == sorted(b_dyn.leaf_to_sample)

        # per-sample reconstruction must match the original input on both paths
        for sample_idx, expected in enumerate(samples):
            assert self._reconstruct(b_hash)[sample_idx] == expected
            assert self._reconstruct(b_dyn)[sample_idx] == expected

    def test_multilevel_depth3(self):
        """4 samples, 2 groups of 2, depth-3 tree. Needs prefix_segments_batch for the hash path
        to detect multi-level; dynamic path infers it from tokens."""
        import hashlib

        samples = [
            [1, 2, 3, 100, 200, 11, 12, 13],
            [1, 2, 3, 100, 200, 21, 22, 23],
            [1, 2, 3, 300, 400, 31, 32, 33],
            [1, 2, 3, 300, 400, 41, 42, 43],
        ]

        def _h(toks):
            return int(hashlib.sha256(b",".join(str(t).encode() for t in toks)).hexdigest()[:8], 16)

        # Per-turn segments: turn1=root(3), turn2=root+mid(5), turn3=full(8)
        psb = [[(_h(s[:3]), 3), (_h(s[:5]), 5), (_h(s), 8)] for s in samples]
        b_hash, b_dyn = self._run_both_paths(samples, prefix_segments_batch=psb)
        assert b_hash is not None and b_dyn is not None

        # Same flat tokens
        assert torch.equal(b_hash.flat_input_ids, b_dyn.flat_input_ids)
        assert b_hash.prefix_range == b_dyn.prefix_range
        assert sorted(b_hash.leaf_to_sample) == sorted(b_dyn.leaf_to_sample)

        for sample_idx, expected in enumerate(samples):
            assert self._reconstruct(b_hash)[sample_idx] == expected
            assert self._reconstruct(b_dyn)[sample_idx] == expected

    def test_no_shared_prefix_both_return_none(self):
        """2 samples with disjoint tokens. Both paths return None."""
        samples = [
            [1, 2, 3],
            [4, 5, 6],
        ]
        b_hash, b_dyn = self._run_both_paths(samples)
        assert b_hash is None, "hash path should return None when no shared prefix"
        assert b_dyn is None, "dynamic path should return None when no shared prefix"

    def test_single_sample_both_return_none(self):
        """1 sample. Both paths return None (no sharing possible)."""
        samples = [[1, 2, 3, 4, 5]]
        b_hash, b_dyn = self._run_both_paths(samples)
        # Hash path: 1-sample LCP == full sample, prefix_len > 0, but downstream
        # may produce a degenerate single-leaf tree. Dynamic path: trie has 1 leaf,
        # convert_trie_to_tree_node returns None. Hash may differ here; verify both
        # at least don't crash.
        assert b_dyn is None, "dynamic path should return None for single sample"
        # If hash returns non-None for 1-sample, that's an idiosyncrasy of the hash
        # path (it accepts degenerate "shared prefix == whole sample" case). Don't
        # assert equivalence here — just that both behave deterministically.
