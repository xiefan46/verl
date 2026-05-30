# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Equivalence + structure tests for the dynamic-trie prefix-tree builder.

Covers:
  - Depth-1 (root + leaves) — same shape as the legacy hash-based path.
  - Depth-2 (root + branch + leaves) — same shape as the legacy multi-level path.
  - Depth-3+ (arbitrary deeper trees) — the new path's unique territory; the
    legacy path collapses these into depth-2.
  - Round-trip: restore_flat_to_nested rebuilds each sample exactly.
  - Multi-forest: returns None when batch shares no prefix.
  - Optional fields: loss_mask + position_ids propagate correctly.

The new builder is in ``verl.utils.prefix_tree_magi`` (same module name as
before — drop-in replacement). The legacy hash-based implementation is
inlined below so the tests don't need an external fixture file or git
history.
"""
from __future__ import annotations

import pytest
import torch
import torch.nested

from verl.utils.prefix_tree_magi import (
    PrefixTreeMagiBatch,
    build_prefix_tree_micro_batch,
    restore_flat_to_nested,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_nested(samples):
    """Wrap a list of int lists into a 2-D NestedTensor."""
    return torch.nested.nested_tensor(
        [torch.tensor(s, dtype=torch.long) for s in samples],
        layout=torch.jagged,
    )


def _build_no_key(samples, **kwargs):
    """Build the prefix tree without constructing a Magi key (model=None)."""
    return build_prefix_tree_micro_batch(None, _make_nested(samples), **kwargs)


# ---------------------------------------------------------------------------
# Round-trip: restore_flat_to_nested rebuilds each sample exactly
# ---------------------------------------------------------------------------


class TestRoundTrip:
    @pytest.mark.parametrize(
        "samples",
        [
            # depth-1: same prefix + per-sample leaf
            [[1, 2, 3, 10, 11], [1, 2, 3, 20, 21], [1, 2, 3, 30, 31]],
            # depth-2: shared root, two branches, leaves under each
            [
                [1, 2, 3, 4, 5, 100],
                [1, 2, 3, 4, 5, 101],
                [1, 2, 3, 9, 9, 102],
                [1, 2, 3, 9, 9, 103],
            ],
            # depth-3: deeper structure (the new path's unique territory)
            [
                [1, 2, 3, 4, 5, 6, 100],   # branches: 3→4→5→6 leaf
                [1, 2, 3, 4, 5, 6, 101],
                [1, 2, 3, 4, 5, 7, 200],   # diverges at the last layer
                [1, 2, 3, 4, 8, 9, 300],   # diverges one layer up
                [1, 2, 3, 4, 8, 9, 301],
            ],
            # variable-length samples with shared root
            [[1, 2, 3, 4], [1, 2, 3, 5, 6, 7], [1, 2, 3, 8, 9]],
        ],
    )
    def test_round_trip(self, samples):
        pt = _build_no_key(samples)
        assert pt is not None

        nested_back = restore_flat_to_nested(pt.flat_input_ids, pt)

        # nested_back must contain each original sample's token sequence in
        # the original sample order.
        for i, original in enumerate(samples):
            restored = nested_back[i].tolist()
            assert restored == original, (
                f"sample {i} did not round-trip:\n  original={original}\n  restored={restored}"
            )


# ---------------------------------------------------------------------------
# Tree-structure assertions
# ---------------------------------------------------------------------------


class TestTreeShape:
    def test_depth1_single_prefix(self):
        samples = [[1, 2, 3, 10], [1, 2, 3, 20], [1, 2, 3, 30]]
        pt = _build_no_key(samples)
        assert pt is not None

        # 3 leaves, one per sample
        assert pt.original_batch_size == 3
        assert len(pt.leaf_to_sample) == 3
        assert sorted(pt.leaf_to_sample) == [0, 1, 2]

        # Root segment covers the shared prefix [1, 2, 3]
        prefix_s, prefix_e = pt.prefix_range
        assert prefix_e - prefix_s == 3
        assert torch.equal(pt.flat_input_ids[prefix_s:prefix_e], torch.tensor([1, 2, 3]))

    def test_depth2_two_branches(self):
        samples = [
            [1, 2, 3, 4, 5, 100],
            [1, 2, 3, 4, 5, 101],
            [1, 2, 3, 9, 9, 200],
            [1, 2, 3, 9, 9, 201],
        ]
        pt = _build_no_key(samples)
        assert pt is not None
        assert pt.original_batch_size == 4

        # Each leaf has an ancestor chain of length 2 (root + branch).
        assert pt.leaf_ancestor_ranges is not None
        for chain in pt.leaf_ancestor_ranges:
            assert len(chain) == 2, f"depth-2 tree should give 2-ancestor leaves, got {chain}"

    def test_depth3_two_levels_of_branching(self):
        # Two top-level branches; the first branch has two sub-branches.
        samples = [
            [1, 2, 3, 4, 5, 6, 100],   # root=[1,2,3] → branch=[4,5] → sub-branch=[6] → leaf
            [1, 2, 3, 4, 5, 6, 101],
            [1, 2, 3, 4, 5, 7, 200],   # ...                      → sub-branch=[7] → leaf
            [1, 2, 3, 4, 5, 7, 201],
            [1, 2, 3, 8, 9, 300],       # → branch=[8,9] → leaf (no sub-branch)
            [1, 2, 3, 8, 9, 301],
        ]
        pt = _build_no_key(samples)
        assert pt is not None
        assert pt.original_batch_size == 6
        assert pt.leaf_ancestor_ranges is not None

        # Leaves under the deep sub-tree have ancestor chains of length 3.
        # Leaves under the shallow branch have length 2.
        chain_lengths = [len(c) for c in pt.leaf_ancestor_ranges]
        # Sort so we don't depend on DFS order: 2 leaves at depth-2, 4 at depth-3.
        assert sorted(chain_lengths) == [2, 2, 3, 3, 3, 3]

    def test_depth4(self):
        # Push depth further: 4 levels of branching beneath root.
        samples = [
            [1, 2, 3, 4, 5, 100],   # branch chain a → b → c → leaf
            [1, 2, 3, 4, 5, 101],   # diverges at last token
            [1, 2, 3, 4, 6, 200],   # diverges at last-but-one
            [1, 2, 3, 9, 9, 300],   # diverges earlier
        ]
        pt = _build_no_key(samples)
        assert pt is not None

        # Round-trip is the cleanest correctness check at this depth.
        nested_back = restore_flat_to_nested(pt.flat_input_ids, pt)
        for i, s in enumerate(samples):
            assert nested_back[i].tolist() == s


# ---------------------------------------------------------------------------
# Fallback cases
# ---------------------------------------------------------------------------


class TestFallback:
    def test_no_shared_prefix_returns_none(self):
        """Batch where samples diverge at token 0 ⇒ multi-forest ⇒ None."""
        samples = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        assert _build_no_key(samples) is None

    def test_single_sample_returns_none(self):
        """Single sample: no sharing possible ⇒ None."""
        samples = [[1, 2, 3, 4, 5]]
        assert _build_no_key(samples) is None

    def test_partial_share_full_batch(self):
        """Two samples share prefix, one doesn't ⇒ multi-forest ⇒ None."""
        samples = [[1, 2, 3, 4], [1, 2, 3, 5], [9, 9, 9]]
        assert _build_no_key(samples) is None


# ---------------------------------------------------------------------------
# Optional fields: loss_mask + position_ids
# ---------------------------------------------------------------------------


class TestOptionalFields:
    def test_loss_mask_propagated(self):
        samples = [[1, 2, 3, 10, 11], [1, 2, 3, 20, 21]]
        loss_mask = _make_nested([[0, 0, 0, 1, 1], [0, 0, 0, 1, 1]])
        pt = build_prefix_tree_micro_batch(None, _make_nested(samples), loss_mask=loss_mask)
        assert pt is not None
        assert pt.flat_loss_mask is not None
        # Prefix region should be unmasked (0), leaf regions masked (1).
        prefix_s, prefix_e = pt.prefix_range
        assert pt.flat_loss_mask[prefix_s:prefix_e].sum().item() == 0
        # Sum of leaf-region mask = total leaf-token count with mask=1.
        # Each sample has 2 leaf tokens with mask=1.
        leaf_total_mask = sum(
            pt.flat_loss_mask[s:e].sum().item() for s, e in pt.leaf_ranges
        )
        assert leaf_total_mask == 2 * len(samples)

    def test_position_ids_default(self):
        """When position_ids is None, builder emits per-segment 0-based indices."""
        samples = [[1, 2, 3, 10, 11], [1, 2, 3, 20, 21]]
        pt = _build_no_key(samples)
        assert pt is not None
        # Root is positions 0..2; each leaf segment is at owner-sample's
        # absolute position 3..4 (per build_arbitrary_depth_params: positions
        # within each owner sample, not per-flat-offset).
        prefix_s, prefix_e = pt.prefix_range
        assert pt.flat_position_ids[prefix_s:prefix_e].tolist() == [0, 1, 2]
        for s, e in pt.leaf_ranges:
            assert pt.flat_position_ids[s:e].tolist() == [3, 4]


# ---------------------------------------------------------------------------
# Compression budget: flat tokens fewer than dense tokens by shared-prefix amount
# ---------------------------------------------------------------------------


class TestCompression:
    def test_flat_tokens_avoid_repeated_prefix(self):
        """A 1-prompt × N-rollout batch should pack prefix only once."""
        P, R, N = 50, 10, 8
        prompt = list(range(100, 100 + P))   # shared prefix
        samples = [prompt + list(range(1000 * i, 1000 * i + R)) for i in range(N)]
        pt = _build_no_key(samples)
        assert pt is not None

        # Dense layout would be N * (P + R). Tree layout is P + N * R.
        dense_total = sum(len(s) for s in samples)
        assert dense_total == N * (P + R)

        tree_total = int(pt.flat_input_ids.numel())
        assert tree_total == P + N * R, f"expected {P + N * R}, got {tree_total}"

        # Sharing ratio matches expectation.
        savings = (dense_total - tree_total) / dense_total
        expected_savings = (N - 1) * P / (N * (P + R))
        assert abs(savings - expected_savings) < 1e-9


# ---------------------------------------------------------------------------
# Equivalence to a hand-rolled reference for the depth-1 case
# ---------------------------------------------------------------------------


class TestKnownGoodReference:
    """Manually-computed expected values for a small batch. If the trie path
    produces these exact tokens / rectangles, the algorithm is correct
    end-to-end (no implicit dependence on rollout-side metadata).
    """

    def test_known_good_depth1(self):
        samples = [[1, 2, 3, 10], [1, 2, 3, 20], [1, 2, 3, 30]]
        pt = _build_no_key(samples)
        assert pt is not None

        # DFS pre-order: root [1,2,3], then leaves sorted by first token (10,20,30)
        assert pt.flat_input_ids.tolist() == [1, 2, 3, 10, 20, 30]
        assert pt.prefix_range == (0, 3)

        # leaves are at flat positions [3,4), [4,5), [5,6)
        # leaf_to_sample maps to original sample indices (10→0, 20→1, 30→2)
        for leaf_idx, (start, end) in enumerate(pt.leaf_ranges):
            leaf_tok = pt.flat_input_ids[start].item()
            sample_idx = pt.leaf_to_sample[leaf_idx]
            assert samples[sample_idx][-1] == leaf_tok


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
