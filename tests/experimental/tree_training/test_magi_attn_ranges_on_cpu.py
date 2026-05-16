# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""CPU-only unit tests for ``_magi_kernel.build_attn_ranges_from_trie``.

Validates the trie-to-AttnRanges-tile transform against hand-computed dense
boolean masks, without invoking MagiAttention or any GPU code. These tests are
the local-only correctness gate before running ``test_magi_forward_equivalence``
on a GPU.

Hand-built ``TrieNode`` instances are used (rather than calling
``build_packed_tree_batch``) so the tests exercise the conversion in
isolation, decoupled from the upstream packing pipeline.
"""

from __future__ import annotations

import unittest

import torch

from verl.experimental.tree_training._magi_kernel import (
    ATTN_TYPE_CAUSAL,
    ATTN_TYPE_FULL,
    build_attn_ranges_from_trie,
    build_attn_ranges_tensors,
    materialize_dense_mask,
)
from verl.experimental.tree_training.tree import TrieNode


def _make_node(
    tree_id: int,
    start: int,
    end_inclusive: int,
    *,
    sequence_ids: list[int] | None = None,
    ancestors: list[TrieNode] | None = None,
) -> TrieNode:
    """Construct a non-root ``TrieNode`` at a fixed packed range.

    Token contents are filled with zeros — they're irrelevant for mask
    construction tests.
    """
    node = TrieNode(
        tree_id=tree_id,
        start_idx=start,
        end_idx=end_inclusive,
        tokens=[0] * (end_inclusive - start + 1),
        sequence_ids=list(sequence_ids) if sequence_ids else [],
    )
    node.ancestors = list(ancestors) if ancestors else []
    return node


def _make_root(tree_id: int = 0) -> TrieNode:
    """Construct a root ``TrieNode`` (start_idx=end_idx=-1)."""
    return TrieNode(tree_id=tree_id)


class TestBasicShapes(unittest.TestCase):
    """D.T1-D.T8 — basic trie topologies."""

    def test_d_t1_single_prefix_two_leaves(self) -> None:
        """Prefix [0-2] -> leaf A [3-4], leaf B [5-6].

        Expected mask (T=7), 1 = attend:
            row 0 prefix:  [1 0 0 0 0 0 0]
            row 1 prefix:  [1 1 0 0 0 0 0]
            row 2 prefix:  [1 1 1 0 0 0 0]
            row 3 leaf_a:  [1 1 1 1 0 0 0]
            row 4 leaf_a:  [1 1 1 1 1 0 0]
            row 5 leaf_b:  [1 1 1 0 0 1 0]   leaf_b does NOT attend leaf_a
            row 6 leaf_b:  [1 1 1 0 0 1 1]
        """
        prefix = _make_node(tree_id=0, start=0, end_inclusive=2, sequence_ids=[0, 1])
        leaf_a = _make_node(tree_id=0, start=3, end_inclusive=4, sequence_ids=[0], ancestors=[prefix])
        leaf_b = _make_node(tree_id=0, start=5, end_inclusive=6, sequence_ids=[1], ancestors=[prefix])
        root = _make_root()
        root.nodes = [prefix, leaf_a, leaf_b]

        q, k, types = build_attn_ranges_from_trie([root])

        # 3 self-causal (prefix, leaf_a, leaf_b) + 2 full (leaf_a->prefix, leaf_b->prefix)
        self.assertEqual(len(q), 5)
        self.assertEqual(types.count(ATTN_TYPE_CAUSAL), 3)
        self.assertEqual(types.count(ATTN_TYPE_FULL), 2)

        dense = materialize_dense_mask(7, q, k, types)
        expected = torch.tensor(
            [
                [1, 0, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0, 0],
                [1, 1, 1, 0, 0, 0, 0],
                [1, 1, 1, 1, 0, 0, 0],
                [1, 1, 1, 1, 1, 0, 0],
                [1, 1, 1, 0, 0, 1, 0],
                [1, 1, 1, 0, 0, 1, 1],
            ],
            dtype=torch.bool,
        )
        self.assertTrue(
            torch.equal(dense, expected),
            f"\nGot:\n{dense.int()}\nExpected:\n{expected.int()}",
        )

    def test_d_t2_depth_three_chain(self) -> None:
        """Chain A[0-1] -> B[2-3] -> C[4-5]; expected mask is full tril."""
        a = _make_node(tree_id=0, start=0, end_inclusive=1)
        b = _make_node(tree_id=0, start=2, end_inclusive=3, ancestors=[a])
        c = _make_node(tree_id=0, start=4, end_inclusive=5, ancestors=[a, b])
        root = _make_root()
        root.nodes = [a, b, c]

        q, k, types = build_attn_ranges_from_trie([root])
        dense = materialize_dense_mask(6, q, k, types)
        expected = torch.tril(torch.ones(6, 6, dtype=torch.bool))
        self.assertTrue(torch.equal(dense, expected))

    def test_d_t3_two_independent_trees(self) -> None:
        """Two trees packed contiguously must not cross-attend."""
        p0 = _make_node(tree_id=0, start=0, end_inclusive=1)
        l0 = _make_node(tree_id=0, start=2, end_inclusive=3, ancestors=[p0])
        r0 = _make_root(tree_id=0)
        r0.nodes = [p0, l0]

        p1 = _make_node(tree_id=1, start=4, end_inclusive=5)
        l1 = _make_node(tree_id=1, start=6, end_inclusive=7, ancestors=[p1])
        r1 = _make_root(tree_id=1)
        r1.nodes = [p1, l1]

        q, k, types = build_attn_ranges_from_trie([r0, r1])
        dense = materialize_dense_mask(8, q, k, types)

        tri4 = torch.tril(torch.ones(4, 4, dtype=torch.bool))
        self.assertTrue(torch.equal(dense[0:4, 0:4], tri4))
        self.assertTrue(torch.equal(dense[4:8, 4:8], tri4))
        self.assertFalse(dense[0:4, 4:8].any().item())
        self.assertFalse(dense[4:8, 0:4].any().item())

    def test_d_t4_n4_rollouts(self) -> None:
        """N=4 rollouts shape -> 1 + 2N = 9 tiles."""
        prefix_len, rollout_len, n = 6, 4, 4
        prefix = _make_node(tree_id=0, start=0, end_inclusive=prefix_len - 1)
        leaves = [
            _make_node(
                tree_id=0,
                start=prefix_len + i * rollout_len,
                end_inclusive=prefix_len + (i + 1) * rollout_len - 1,
                ancestors=[prefix],
            )
            for i in range(n)
        ]
        root = _make_root()
        root.nodes = [prefix, *leaves]

        q, k, types = build_attn_ranges_from_trie([root])
        self.assertEqual(len(q), 1 + 2 * n)

        T = prefix_len + n * rollout_len
        dense = materialize_dense_mask(T, q, k, types)
        # diagonal everywhere true (every position attends to itself)
        self.assertTrue(dense.diagonal().all().item())
        # leaves attend prefix fully
        for i in range(n):
            ls = prefix_len + i * rollout_len
            le = prefix_len + (i + 1) * rollout_len
            self.assertTrue(dense[ls:le, 0:prefix_len].all().item())
        # leaves do not cross-attend
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                qs, qe = prefix_len + i * rollout_len, prefix_len + (i + 1) * rollout_len
                ks, ke = prefix_len + j * rollout_len, prefix_len + (j + 1) * rollout_len
                self.assertFalse(
                    dense[qs:qe, ks:ke].any().item(),
                    f"leaf {i} unexpectedly attends leaf {j}",
                )

    def test_d_t5_n64_rollouts_stress(self) -> None:
        """N=64 rollouts -> 129 tiles, no crash."""
        prefix_len, rollout_len, n = 8, 2, 64
        prefix = _make_node(tree_id=0, start=0, end_inclusive=prefix_len - 1)
        leaves = [
            _make_node(
                tree_id=0,
                start=prefix_len + i * rollout_len,
                end_inclusive=prefix_len + (i + 1) * rollout_len - 1,
                ancestors=[prefix],
            )
            for i in range(n)
        ]
        root = _make_root()
        root.nodes = [prefix, *leaves]

        q, k, types = build_attn_ranges_from_trie([root])
        self.assertEqual(len(q), 1 + 2 * n)
        self.assertEqual(len(types), 129)

    def test_d_t6_multi_level_three_deep(self) -> None:
        """Root -> A -> B (depth 3): B attends self + A + (transitively recorded
        through ancestors list)."""
        a = _make_node(tree_id=0, start=0, end_inclusive=1)
        b = _make_node(tree_id=0, start=2, end_inclusive=3, ancestors=[a])
        root = _make_root()
        root.nodes = [a, b]

        q, k, types = build_attn_ranges_from_trie([root])
        # a: 1 self-causal; b: 1 self-causal + 1 full to a = 3 tiles
        self.assertEqual(len(q), 3)

        dense = materialize_dense_mask(4, q, k, types)
        expected = torch.tril(torch.ones(4, 4, dtype=torch.bool))
        self.assertTrue(torch.equal(dense, expected))

    def test_d_t7_asymmetric_branching(self) -> None:
        """Prefix[0-2] -> leaf_a[3-3] (1 token), leaf_b[4-7] (4 tokens)."""
        prefix = _make_node(tree_id=0, start=0, end_inclusive=2)
        leaf_a = _make_node(tree_id=0, start=3, end_inclusive=3, ancestors=[prefix])
        leaf_b = _make_node(tree_id=0, start=4, end_inclusive=7, ancestors=[prefix])
        root = _make_root()
        root.nodes = [prefix, leaf_a, leaf_b]

        q, k, types = build_attn_ranges_from_trie([root])
        self.assertEqual(len(q), 5)  # 3 self-causal + 2 full

        dense = materialize_dense_mask(8, q, k, types)
        # leaf_a (row 3) attends prefix (cols 0-2) + self (col 3); not leaf_b
        self.assertTrue(dense[3, 0:4].all().item())
        self.assertFalse(dense[3, 4:8].any().item())
        # leaf_b (rows 4-7) attends prefix + self-causal among itself; not leaf_a
        self.assertTrue(dense[4:8, 0:3].all().item())
        self.assertFalse(dense[4:8, 3:4].any().item())

    def test_d_t8_single_leaf_depth_one(self) -> None:
        """Single node, no ancestors -> exactly 1 self-causal tile."""
        n = _make_node(tree_id=0, start=0, end_inclusive=3)
        root = _make_root()
        root.nodes = [n]

        q, k, types = build_attn_ranges_from_trie([root])
        self.assertEqual(len(q), 1)
        self.assertEqual(types[0], ATTN_TYPE_CAUSAL)

        dense = materialize_dense_mask(4, q, k, types)
        expected = torch.tril(torch.ones(4, 4, dtype=torch.bool))
        self.assertTrue(torch.equal(dense, expected))


class TestBoundariesAndDataContracts(unittest.TestCase):
    """D.T9-D.T13 — edge cases and data contracts."""

    def test_d_t9_one_token_node_inclusive_to_exclusive(self) -> None:
        """1-token node [5, 5] (inclusive) -> q_range [5, 6) (exclusive)."""
        n = _make_node(tree_id=0, start=5, end_inclusive=5)
        root = _make_root()
        root.nodes = [n]

        q, k, types = build_attn_ranges_from_trie([root])
        self.assertEqual(q, [(5, 6)])
        self.assertEqual(k, [(5, 6)])

    def test_d_t10_prefix_length_one(self) -> None:
        """Prefix of length 1, leaf of length 2. Verify boundary handling."""
        prefix = _make_node(tree_id=0, start=0, end_inclusive=0)
        leaf = _make_node(tree_id=0, start=1, end_inclusive=2, ancestors=[prefix])
        root = _make_root()
        root.nodes = [prefix, leaf]

        q, k, types = build_attn_ranges_from_trie([root])
        self.assertEqual(len(q), 3)
        dense = materialize_dense_mask(3, q, k, types)
        expected = torch.tril(torch.ones(3, 3, dtype=torch.bool))
        self.assertTrue(torch.equal(dense, expected))

    def test_d_t11_tensor_dtype_int32(self) -> None:
        """Tensor builder must emit int32 tensors (FFA kernel requirement)."""
        n = _make_node(tree_id=0, start=0, end_inclusive=2)
        root = _make_root()
        root.nodes = [n]

        q_t, k_t, type_t = build_attn_ranges_tensors([root])
        self.assertEqual(q_t.dtype, torch.int32)
        self.assertEqual(k_t.dtype, torch.int32)
        self.assertEqual(type_t.dtype, torch.int32)
        self.assertEqual(q_t.shape[1], 2)  # (N, 2) shape

    def test_d_t12_dense_oracle_matches_tree_build_attention_mask(self) -> None:
        """The dense mask reconstructed from tiles should match the canonical
        ``_build_attention_mask`` output from tree.py, bit-for-bit.

        This is the regression sentinel against tree.py's existing dense mask
        construction (line ~579) for a simple topology.
        """
        # Import locally to avoid triggering tree.py imports at module load.
        from verl.experimental.tree_training.tree import _build_attention_mask  # noqa: E402

        prefix = _make_node(tree_id=0, start=0, end_inclusive=2, sequence_ids=[0, 1])
        leaf_a = _make_node(tree_id=0, start=3, end_inclusive=4, sequence_ids=[0], ancestors=[prefix])
        leaf_b = _make_node(tree_id=0, start=5, end_inclusive=6, sequence_ids=[1], ancestors=[prefix])
        root = _make_root()
        root.nodes = [prefix, leaf_a, leaf_b]

        # Reconstruct via Magi tiles
        q, k, types = build_attn_ranges_from_trie([root])
        magi_dense = materialize_dense_mask(7, q, k, types)

        # Reconstruct via tree.py canonical builder
        # Note: _build_attention_mask expects ``trie`` as root + max_tokens.
        canonical = _build_attention_mask(root, max_tokens=7, device=torch.device("cpu"))
        # canonical is shape [1, 1, T, T] or [T, T] depending on version; squeeze
        if canonical.dim() > 2:
            canonical = canonical.squeeze()
        self.assertTrue(
            torch.equal(magi_dense, canonical.bool()),
            f"\nMagi tile mask:\n{magi_dense.int()}\nCanonical:\n{canonical.int()}",
        )

    def test_d_t13_large_tile_count(self) -> None:
        """64 leaves should produce 129 tiles without error."""
        prefix = _make_node(tree_id=0, start=0, end_inclusive=7)
        leaves = [
            _make_node(
                tree_id=0,
                start=8 + i * 2,
                end_inclusive=9 + i * 2,
                ancestors=[prefix],
            )
            for i in range(64)
        ]
        root = _make_root()
        root.nodes = [prefix, *leaves]

        q, k, types = build_attn_ranges_from_trie([root])
        self.assertEqual(len(q), 129)


class TestDummyAndState(unittest.TestCase):
    """D.T14-D.T16 — dummy trie and state isolation."""

    def test_d_t14_empty_trie_returns_no_tiles(self) -> None:
        root = _make_root()
        root.nodes = []
        q, k, types = build_attn_ranges_from_trie([root])
        self.assertEqual(q, [])
        self.assertEqual(k, [])
        self.assertEqual(types, [])

    def test_d_t15_empty_trie_tensor_sentinel(self) -> None:
        """Tensor builder pads empty input with a (0, 0, 0, 0, FULL) sentinel."""
        root = _make_root()
        root.nodes = []
        q_t, k_t, type_t = build_attn_ranges_tensors([root])
        self.assertEqual(q_t.shape, (1, 2))
        self.assertEqual(k_t.shape, (1, 2))
        self.assertEqual(type_t.shape, (1,))
        self.assertEqual(int(q_t.sum().item()), 0)
        self.assertEqual(int(k_t.sum().item()), 0)
        self.assertEqual(int(type_t.item()), ATTN_TYPE_FULL)

    def test_d_t16_no_state_leakage_across_calls(self) -> None:
        """Repeated calls must not mutate or share state."""
        root1 = _make_root()
        root1.nodes = [_make_node(tree_id=0, start=0, end_inclusive=2)]

        empty_root = _make_root()
        empty_root.nodes = []

        # call 1: real ranges
        q1, _, _ = build_attn_ranges_from_trie([root1])
        # call 2: empty
        q2, _, _ = build_attn_ranges_from_trie([empty_root])
        # call 3: same as call 1
        q3, _, _ = build_attn_ranges_from_trie([root1])

        self.assertEqual(q1, [(0, 3)])
        self.assertEqual(q2, [])
        self.assertEqual(q3, [(0, 3)])


class TestRobustness(unittest.TestCase):
    """D.T17-D.T18 — robustness against malformed input."""

    def test_d_t17_rejects_non_root_at_top(self) -> None:
        non_root = _make_node(tree_id=0, start=0, end_inclusive=2)
        with self.assertRaises(ValueError):
            build_attn_ranges_from_trie([non_root])

    def test_d_t18_ancestor_order_does_not_change_mask(self) -> None:
        """Order of nodes in TrieNode.ancestors should not affect the resulting
        dense mask (mask is the union of tiles, which is order-independent)."""
        a = _make_node(tree_id=0, start=0, end_inclusive=1)
        b = _make_node(tree_id=0, start=2, end_inclusive=3, ancestors=[a])
        # construct two C variants with different ancestor orderings
        c_v1 = _make_node(tree_id=0, start=4, end_inclusive=5, ancestors=[a, b])
        c_v2 = _make_node(tree_id=0, start=4, end_inclusive=5, ancestors=[b, a])

        r1 = _make_root()
        r1.nodes = [a, b, c_v1]
        r2 = _make_root()
        r2.nodes = [a, b, c_v2]

        q1, k1, t1 = build_attn_ranges_from_trie([r1])
        q2, k2, t2 = build_attn_ranges_from_trie([r2])

        dense1 = materialize_dense_mask(6, q1, k1, t1)
        dense2 = materialize_dense_mask(6, q2, k2, t2)
        self.assertTrue(torch.equal(dense1, dense2))


if __name__ == "__main__":
    unittest.main()
