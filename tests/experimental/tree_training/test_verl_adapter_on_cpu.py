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

"""Unit tests for the tree training DataProto adapter (`_verl_adapter.py`).

Focuses on shape correctness and trie-order alignment of returned tensors.
Numerical correctness of the underlying algorithm is covered by Phase 1
equivalence tests; this file only exercises the verl-side glue (CPU-only).
"""

from __future__ import annotations

import unittest

import torch

from verl.experimental.tree_training._verl_adapter import (
    _build_attention_mask_from_lens,
    _expand_response_only_to_full_seq,
    _nested_to_padded,
    _seq_lens_from_nested,
    align_packed_extras_to_labels,
    build_tree_mb_list,
    build_tree_model_inputs,
)


def _make_nested(rows: list[list[int]]) -> torch.Tensor:
    """Build a jagged nested int64 tensor from a list of 1-D lists."""
    tensors = [torch.tensor(row, dtype=torch.long) for row in rows]
    return torch.nested.as_nested_tensor(tensors, layout=torch.jagged)


def _make_td(
    seqs: list[list[int]],
    response_lens: list[int] | None = None,
    *,
    max_response_len: int | None = None,
    include_advantages: bool = True,
) -> dict:
    """Mimic the structure of a TensorDict post-`left_right_2_no_padding`.

    A plain dict is used (adapter only relies on dict-like access). Constructs
    nested input_ids / position_ids and 2-D response-only extras matching the
    layout the engine sees.
    """
    n = len(seqs)
    seq_lens = [len(s) for s in seqs]
    if response_lens is None:
        # Default: half the sequence is response.
        response_lens = [max(1, L // 2) for L in seq_lens]
    assert len(response_lens) == n
    assert all(rl <= sl for rl, sl in zip(response_lens, seq_lens, strict=True))

    max_rl = max_response_len if max_response_len is not None else max(response_lens)
    input_ids_nested = _make_nested(seqs)
    position_ids_nested = _make_nested([list(range(L)) for L in seq_lens])

    response_mask = torch.zeros(n, max_rl, dtype=torch.long)
    for i, rl in enumerate(response_lens):
        response_mask[i, :rl] = 1

    td: dict = {
        "input_ids": input_ids_nested,
        "position_ids": position_ids_nested,
        "response_mask": response_mask,
        "loss_mask": response_mask,  # alias as in left_right_2_no_padding
    }
    if include_advantages:
        # Plausible advantage values: distinguishable per row to verify alignment.
        adv = torch.zeros(n, max_rl, dtype=torch.float32)
        for i, rl in enumerate(response_lens):
            adv[i, :rl] = torch.arange(1, rl + 1, dtype=torch.float32) * (10**i)
        td["advantages"] = adv
        td["old_log_probs"] = -adv  # negative for distinguishability
    return td


# =============================================================================
# Internal helpers
# =============================================================================


class TestInternalHelpers(unittest.TestCase):
    def test_seq_lens_from_nested(self):
        nested = _make_nested([[1, 2, 3, 4, 5], [10, 20, 30], [100, 200]])
        lens = _seq_lens_from_nested(nested)
        self.assertEqual(lens.tolist(), [5, 3, 2])

    def test_build_attention_mask_from_lens(self):
        seq_lens = torch.tensor([5, 3, 2])
        mask = _build_attention_mask_from_lens(seq_lens, max_seq_len=6)
        self.assertEqual(tuple(mask.shape), (3, 6))
        self.assertEqual(mask.tolist(), [[1, 1, 1, 1, 1, 0], [1, 1, 1, 0, 0, 0], [1, 1, 0, 0, 0, 0]])

    def test_nested_to_padded(self):
        nested = _make_nested([[1, 2, 3, 4, 5], [10, 20]])
        padded = _nested_to_padded(nested, padding=0, max_seq_len=5)
        self.assertEqual(tuple(padded.shape), (2, 5))
        self.assertEqual(padded[0].tolist(), [1, 2, 3, 4, 5])
        self.assertEqual(padded[1].tolist(), [10, 20, 0, 0, 0])

    def test_expand_response_only_to_full_seq(self):
        # 2 sequences. Seq 0: prompt_len=3, response_len=2. Seq 1: prompt_len=2, response_len=3.
        response_only = torch.tensor([[1.0, 2.0, 0.0], [10.0, 20.0, 30.0]])  # [B=2, max_response_len=3]
        seq_lens = torch.tensor([5, 5])
        response_lens = torch.tensor([2, 3])
        full = _expand_response_only_to_full_seq(response_only, seq_lens, response_lens, max_seq_len=6)
        self.assertEqual(tuple(full.shape), (2, 6))
        # seq 0: prompt[0..3) = 0, response[3..5) = [1,2], padding[5..6) = 0
        self.assertEqual(full[0].tolist(), [0, 0, 0, 1.0, 2.0, 0])
        # seq 1: prompt[0..2) = 0, response[2..5) = [10,20,30], padding[5..6) = 0
        self.assertEqual(full[1].tolist(), [0, 0, 10.0, 20.0, 30.0, 0])


# =============================================================================
# align_packed_extras_to_labels
# =============================================================================


class TestAlignPackedExtras(unittest.TestCase):
    def test_drops_position_zero_per_segment(self):
        # 3 segments of lengths 4, 2, 3 → total 9
        packed = torch.tensor([10, 11, 12, 13, 20, 21, 30, 31, 32], dtype=torch.float32)
        aligned = align_packed_extras_to_labels(packed, [4, 2, 3])
        # Drop index 0 of each segment:
        # seg 0 [10,11,12,13] -> [11,12,13]
        # seg 1 [20,21]       -> [21]
        # seg 2 [30,31,32]    -> [31,32]
        self.assertEqual(aligned.tolist(), [11, 12, 13, 21, 31, 32])

    def test_empty_segments_skipped(self):
        # Length-1 segment has no entries after dropping pos 0; should produce empty contribution.
        packed = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
        aligned = align_packed_extras_to_labels(packed, [1, 3])
        # seg 0 [1]    -> drop -> empty
        # seg 1 [2,3,4] -> [3, 4]
        self.assertEqual(aligned.tolist(), [3, 4])

    def test_all_empty(self):
        packed = torch.tensor([1, 2], dtype=torch.float32)
        aligned = align_packed_extras_to_labels(packed, [1, 1])
        self.assertEqual(aligned.numel(), 0)


# =============================================================================
# build_tree_mb_list end-to-end
# =============================================================================


class TestBuildTreeMbList(unittest.TestCase):
    def test_basic_shape_contracts(self):
        # 4 sequences sharing a prompt prefix [1, 2, 3], diverging tails.
        seqs = [
            [1, 2, 3, 100, 101, 102],
            [1, 2, 3, 200, 201, 202],
            [1, 2, 3, 300, 301, 302],
            [1, 2, 3, 400, 401, 402],
        ]
        td = _make_td(seqs, response_lens=[3, 3, 3, 3])

        mbs, metrics = build_tree_mb_list(td, max_tokens_per_mb=128, pad_token_id=0)

        # Single trie fits in one mb (15 unique tokens ≪ 128).
        self.assertEqual(len(mbs), 1)
        mb = mbs[0]
        self.assertIn("input_ids", mb)
        self.assertIn("position_ids", mb)
        self.assertIn("trie_node", mb)
        self.assertEqual(mb["input_ids"].shape[0], 1)  # batch dim
        self.assertEqual(mb["input_ids"].shape[-1] % 128, 0)  # BLOCK_SIZE aligned
        # Packed extras must be 1-D, length sum(seq_lens) = 4*6 = 24
        self.assertEqual(mb["response_mask"].dim(), 1)
        self.assertEqual(mb["response_mask"].numel(), 24)
        self.assertEqual(mb["advantages"].dim(), 1)
        self.assertEqual(mb["advantages"].numel(), 24)

        # tree_token_ratio: total seq tokens 4*6=24, unique trie tokens =
        # 3 shared prompt + 4*3 unique tails = 15. So ratio = 15/24 = 0.625.
        self.assertAlmostEqual(metrics["tree_token_ratio"], 15 / 24, places=4)

    def test_no_extras_path(self):
        """Adapter handles a TensorDict missing advantages / old_log_probs (e.g. infer path)."""
        seqs = [[1, 2, 3, 4], [1, 2, 3, 5]]
        td = _make_td(seqs, response_lens=[2, 2], include_advantages=False)
        mbs, metrics = build_tree_mb_list(td, max_tokens_per_mb=128)
        self.assertEqual(len(mbs), 1)
        mb = mbs[0]
        # response_mask still present
        self.assertIn("response_mask", mb)
        # but advantages / old_log_probs absent
        self.assertNotIn("advantages", mb)
        self.assertNotIn("old_log_probs", mb)
        self.assertGreater(metrics["tree_token_ratio"], 0.0)
        self.assertLessEqual(metrics["tree_token_ratio"], 1.0)

    def test_metric_no_sharing(self):
        """tree_token_ratio == 1.0 when there's no prefix sharing (all sequences distinct)."""
        seqs = [[1, 2, 3, 4], [10, 20, 30, 40], [100, 200, 300, 400]]
        td = _make_td(seqs, response_lens=[2, 2, 2])
        _, metrics = build_tree_mb_list(td, max_tokens_per_mb=128)
        self.assertAlmostEqual(metrics["tree_token_ratio"], 1.0, places=4)

    def test_extras_alignment_with_response_mask(self):
        """After packing, response_mask + advantages live at the right (response) positions."""
        # 2 seqs, prompt [1,2], response [50,51]. 4 tokens each, 2 are prompt.
        seqs = [[1, 2, 50, 51], [1, 2, 60, 61]]
        td = _make_td(seqs, response_lens=[2, 2])
        # response_mask = [[1,1], [1,1]] (full response), advantages distinguish rows
        mbs, _ = build_tree_mb_list(td, max_tokens_per_mb=128)
        mb = mbs[0]
        # Packed length = sum(seq_lens) = 8
        self.assertEqual(mb["response_mask"].numel(), 8)
        # Packed in trie.all_sequence_ids order. Each seq's first 2 entries
        # should be 0 (prompt mask), last 2 should be 1 (response mask).
        packed_rm = mb["response_mask"]
        # Validate that we have 2*2=4 prompt zeros and 2*2=4 response ones, total.
        self.assertEqual(int((packed_rm == 0).sum()), 4)
        self.assertEqual(int((packed_rm == 1).sum()), 4)


# =============================================================================
# build_tree_model_inputs (structure only; full forward needs GPU + HF model)
# =============================================================================


class TestBuildTreeModelInputs(unittest.TestCase):
    def test_keys_and_kwargs_shape(self):
        """build_tree_model_inputs returns (model_inputs, output_args, scope_args).

        Magi path: ``model_inputs`` no longer carries the attention mask; the
        mask flows via ``_magi_backend.tree_attn_scope`` instead. ``scope_args``
        carries the AttnRanges tile representation (q/k_ranges + attn_type_map +
        total_seqlen) ready to feed the scope.

        CPU-runnable: build_tree_model_inputs is now pure Python data shaping
        (no kernel calls), so no CUDA skip needed.
        """
        seqs = [[1, 2, 3, 4, 5], [1, 2, 3, 6, 7]]
        td = _make_td(seqs, response_lens=[2, 2])
        mbs, _ = build_tree_mb_list(td, max_tokens_per_mb=128)
        mb = mbs[0]
        device = torch.device("cpu")
        model_inputs, output_args, scope_args = build_tree_model_inputs(mb, device)

        # model_inputs: just input_ids + position_ids (no attention_mask, no kwargs).
        self.assertIn("input_ids", model_inputs)
        self.assertIn("position_ids", model_inputs)
        self.assertNotIn("attention_mask", model_inputs)
        self.assertNotIn("tree_block_mask", model_inputs)

        # output_args: trie + packed_input_ids (unchanged contract).
        self.assertIn("trie", output_args)
        self.assertIn("packed_input_ids", output_args)

        # scope_args: q/k_ranges + attn_type_map + total_seqlen.
        self.assertIn("q_ranges_naive", scope_args)
        self.assertIn("k_ranges_naive", scope_args)
        self.assertIn("attn_type_map_list", scope_args)
        self.assertIn("total_seqlen", scope_args)
        # Shape contract: each entry is list[tuple[int, int]] or list[int].
        self.assertIsInstance(scope_args["q_ranges_naive"], list)
        self.assertIsInstance(scope_args["k_ranges_naive"], list)
        self.assertIsInstance(scope_args["attn_type_map_list"], list)
        self.assertEqual(
            len(scope_args["q_ranges_naive"]),
            len(scope_args["k_ranges_naive"]),
        )
        self.assertEqual(
            len(scope_args["q_ranges_naive"]),
            len(scope_args["attn_type_map_list"]),
        )
        self.assertEqual(scope_args["total_seqlen"], model_inputs["input_ids"].size(-1))


if __name__ == "__main__":
    unittest.main()
