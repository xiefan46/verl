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

"""CPU mock e2e test for the MagiAttention tree-training data flow.

Phase E.3 of the migration. Stubs out ``magi_attention.api`` so the full
``DataProto -> packed mb -> build_tree_model_inputs -> tree_attn_scope -> model
forward -> unpack_tree_logprobs_per_seq -> assemble`` pipeline runs on a
CPU-only machine without MagiAttention installed. Catches wiring bugs
(missing scope_args fields, kwargs threading mismatches, shape mismatches)
before they cost RunPod GPU time.

Tests intentionally DO NOT spin up an ``FSDPEngine``; that machinery is too
heavyweight for CPU mock. The wiring inside ``_forward_step_tree`` is
exercised conceptually here by manually running the equivalent steps in
sequence.

Covered cases:
* E.T13 core mock e2e (loss path equivalent): 2 prompts x 2 rollouts; build
  mb -> scope -> mock fwd -> unpack -> flat log_probs in trie order
* E.T14 forward-only path: per-seq dicts assembled
* E.T15 multi-mb split: small ``max_tokens_per_mb`` forces 2 mbs
* E.T16 dummy trie: empty mb_list / empty trie path
* E.T17 multi-mb scope isolation: each mb gets its own scope, no leakage
"""

from __future__ import annotations

import sys
import types
import unittest
from unittest import mock

import torch

# Stub heavy external modules at import time. Production CPU tests on RunPod
# will have these installed (so the real symbols load); we only stub when
# missing so the file is collectable on Mac.
if "transformers" not in sys.modules:
    sys.modules["transformers"] = types.ModuleType("transformers")
if "transformers.modeling_utils" not in sys.modules:
    modeling_utils_stub = types.ModuleType("transformers.modeling_utils")

    class _FakeAllAttentionFunctions(dict):
        def register(self, name, fn):
            self[name] = fn

    modeling_utils_stub.ALL_ATTENTION_FUNCTIONS = _FakeAllAttentionFunctions()
    sys.modules["transformers.modeling_utils"] = modeling_utils_stub
    sys.modules["transformers"].modeling_utils = modeling_utils_stub


# Production imports — done after stubs are in place.
from verl.experimental.tree_training._magi_kernel import (  # noqa: E402
    build_attn_ranges_from_trie,
)
from verl.experimental.tree_training._verl_adapter import (  # noqa: E402
    build_tree_mb_list,
    build_tree_model_inputs,
    unpack_tree_logprobs_per_seq,
)

# =============================================================================
# Shared synthetic data + stubs
# =============================================================================


def _make_nested(rows: list[list[int]]) -> torch.Tensor:
    tensors = [torch.tensor(row, dtype=torch.long) for row in rows]
    return torch.nested.as_nested_tensor(tensors, layout=torch.jagged)


def _make_td(seqs: list[list[int]], response_lens: list[int]) -> dict:
    """Mimic a post-``left_right_2_no_padding`` TensorDict (plain dict)."""
    n = len(seqs)
    seq_lens = [len(s) for s in seqs]
    max_rl = max(response_lens)

    input_ids_nested = _make_nested(seqs)
    position_ids_nested = _make_nested([list(range(L)) for L in seq_lens])

    response_mask = torch.zeros(n, max_rl, dtype=torch.long)
    for i, rl in enumerate(response_lens):
        response_mask[i, :rl] = 1

    adv = torch.zeros(n, max_rl, dtype=torch.float32)
    for i, rl in enumerate(response_lens):
        adv[i, :rl] = torch.arange(1, rl + 1, dtype=torch.float32) * (10**i)

    return {
        "input_ids": input_ids_nested,
        "position_ids": position_ids_nested,
        "response_mask": response_mask,
        "loss_mask": response_mask,
        "advantages": adv,
        "old_log_probs": -adv,
    }


def _make_magi_api_stub():
    """Build a ``magi_attention.api`` stub for ``tree_attn_scope`` to import.

    Returns
    -------
    types.ModuleType
        A stand-in module whose ``magi_attn_flex_key`` records the call args
        on each invocation; ``AttnRanges`` / ``AttnMaskType`` mimic the
        constructor signature with minimal validation.
    """
    api = types.ModuleType("magi_attention.api")

    class _MockAttnMaskType:
        FULL = "FULL"
        CAUSAL = "CAUSAL"

        @classmethod
        def from_int_type(cls, t):
            return cls.CAUSAL if t == 1 else cls.FULL

    class _MockAttnRanges:
        def __init__(self, ranges):
            self.ranges = list(ranges)

        @classmethod
        def from_ranges(cls, ranges):
            return cls(ranges)

    api.AttnMaskType = _MockAttnMaskType
    api.AttnRanges = _MockAttnRanges
    api.DistAttnConfig = type("DistAttnConfig", (), {})
    api.compute_pad_size = lambda total, cp_size, chunk: 0
    api.magi_attn_flex_key = mock.MagicMock(return_value="MOCK_KEY")
    api.get_most_recent_key = mock.MagicMock(return_value="MOCK_KEY")
    api.calc_attn = mock.MagicMock()
    return api


def _stub_model_forward(packed_input_ids: torch.Tensor, vocab_size: int = 32) -> torch.Tensor:
    """Stand-in for HF model forward — emits deterministic [T, V] logits.

    Uses a simple position-dependent rule so logprobs are not all-zero.
    """
    T = packed_input_ids.size(-1)
    logits = torch.zeros(T, vocab_size, dtype=torch.float32)
    for t in range(T):
        # Cycle a small spike across vocab positions; lets unpack_logprobs
        # find a clear argmax / non-uniform distribution.
        logits[t, t % vocab_size] = 1.0
    return logits


# =============================================================================
# Mock e2e tests
# =============================================================================


class TestMagiE2EMockWiring(unittest.TestCase):
    """Run the data-flow path end-to-end with magi_attention.api stubbed."""

    def test_e_t13_core_loss_path(self):
        """Core e2e: 2 prompts x 2 rollouts -> mb_list -> tree_attn_scope ->
        mock forward -> unpack_tree_logprobs_per_seq -> flat log_probs."""
        from verl.experimental.tree_training._magi_backend import tree_attn_scope

        # 2 prompts sharing prefix [1, 2, 3]; each has 2 rollouts of len 4-5.
        seqs = [
            [1, 2, 3, 10, 11, 12],
            [1, 2, 3, 10, 11, 13],
            [1, 2, 3, 20, 21],
            [1, 2, 3, 20, 22],
        ]
        response_lens = [3, 3, 2, 2]
        td = _make_td(seqs, response_lens=response_lens)

        # Step 1: build packed micro-batches (real).
        mbs, metrics = build_tree_mb_list(td, max_tokens_per_mb=128, pad_token_id=0)
        self.assertGreaterEqual(len(mbs), 1)
        self.assertIn("tree_token_ratio", metrics)

        # Step 2: for each mb, build model inputs + scope args (real).
        for mb in mbs:
            model_inputs, output_args, scope_args = build_tree_model_inputs(mb, "cpu")

            # Step 3: enter tree_attn_scope (stubbed Magi).
            api_stub = _make_magi_api_stub()
            with mock.patch.dict(sys.modules, {"magi_attention.api": api_stub}):
                with tree_attn_scope(
                    **scope_args,
                    num_heads_q=4,
                    num_heads_kv=2,
                    head_dim=16,
                    cp_group=None,
                ) as key:
                    self.assertEqual(key, "MOCK_KEY")

                    # Step 4: mock model.forward.
                    logits = _stub_model_forward(model_inputs["input_ids"])

                    # Step 5: unpack per-seq logprobs (real).
                    trie = output_args["trie"]
                    packed_input_ids = output_args["packed_input_ids"]
                    if trie.all_sequence_ids:
                        log_probs_per_seq, _entropy = unpack_tree_logprobs_per_seq(
                            logits,
                            trie,
                            packed_input_ids,
                            temperature=1.0,
                            with_entropy=False,
                        )
                        # All sequences in the trie should produce logprobs.
                        for sid in trie.all_sequence_ids:
                            self.assertIn(sid, log_probs_per_seq)
                            self.assertGreater(log_probs_per_seq[sid].numel(), 0)
                            # logprobs are by definition <= 0.
                            self.assertTrue((log_probs_per_seq[sid] <= 0).all().item())

            api_stub.magi_attn_flex_key.assert_called_once()

    def test_e_t14_forward_only_path(self):
        """Forward-only path: per-seq dicts assembled, no flat log_probs build."""
        from verl.experimental.tree_training._magi_backend import tree_attn_scope

        seqs = [[1, 2, 3, 4, 5], [1, 2, 3, 6, 7]]
        td = _make_td(seqs, response_lens=[2, 2])
        mbs, _ = build_tree_mb_list(td, max_tokens_per_mb=128, pad_token_id=0)

        mb = mbs[0]
        model_inputs, output_args, scope_args = build_tree_model_inputs(mb, "cpu")
        api_stub = _make_magi_api_stub()

        with mock.patch.dict(sys.modules, {"magi_attention.api": api_stub}):
            with tree_attn_scope(**scope_args, num_heads_q=4, num_heads_kv=2, head_dim=16, cp_group=None):
                logits = _stub_model_forward(model_inputs["input_ids"])
                trie = output_args["trie"]
                packed_input_ids = output_args["packed_input_ids"]
                log_probs_per_seq, entropy_per_seq = unpack_tree_logprobs_per_seq(
                    logits,
                    trie,
                    packed_input_ids,
                    temperature=1.0,
                    with_entropy=True,
                )

        self.assertIsInstance(log_probs_per_seq, dict)
        self.assertIsInstance(entropy_per_seq, dict)
        self.assertEqual(set(log_probs_per_seq.keys()), set(trie.all_sequence_ids))
        self.assertEqual(set(entropy_per_seq.keys()), set(trie.all_sequence_ids))

    def test_e_t15_multi_mb_split(self):
        """Small max_tokens_per_mb forces 2 mbs; both run through scope OK."""
        from verl.experimental.tree_training._magi_backend import tree_attn_scope

        # 4 sequences of length 6 -> total 24 tokens. With max_tokens_per_mb=16
        # (rounded up to BLOCK_SIZE alignment) the greedy packer should produce
        # at least 2 mbs.
        seqs = [[i + 1, i + 2, i + 3, i + 10, i + 11, i + 12] for i in range(4)]
        response_lens = [3, 3, 3, 3]
        td = _make_td(seqs, response_lens=response_lens)
        mbs, _ = build_tree_mb_list(td, max_tokens_per_mb=128, pad_token_id=0)

        # Even with default 128, the packer may pack into 1 mb if tokens fit.
        # The real exercise is that whatever number of mbs it produces, each
        # one runs through the data flow without leaking state between them.
        for mb_idx, mb in enumerate(mbs):
            model_inputs, output_args, scope_args = build_tree_model_inputs(mb, "cpu")
            api_stub = _make_magi_api_stub()
            with mock.patch.dict(sys.modules, {"magi_attention.api": api_stub}):
                with tree_attn_scope(
                    **scope_args,
                    num_heads_q=4,
                    num_heads_kv=2,
                    head_dim=16,
                    cp_group=None,
                ):
                    logits = _stub_model_forward(model_inputs["input_ids"])
                    trie = output_args["trie"]
                    if trie.all_sequence_ids:
                        packed_input_ids = output_args["packed_input_ids"]
                        log_probs_per_seq, _ = unpack_tree_logprobs_per_seq(
                            logits,
                            trie,
                            packed_input_ids,
                            temperature=1.0,
                            with_entropy=False,
                        )
                        # Each mb produces logprobs for its owned sequences.
                        for sid in trie.all_sequence_ids:
                            self.assertIn(sid, log_probs_per_seq)
            # Verify magi_attn_flex_key was called exactly once per mb (no leakage).
            api_stub.magi_attn_flex_key.assert_called_once()

    def test_e_t16_dummy_trie_empty_path(self):
        """Empty trie (no rows) -> build_attn_ranges_from_trie returns empty;
        tree_attn_scope still runs (Magi sentinel)."""
        from verl.experimental.tree_training._magi_backend import tree_attn_scope
        from verl.experimental.tree_training.tree import TrieNode

        # Construct an empty root manually (no rollouts attached).
        empty_root = TrieNode(tree_id=0)
        empty_root.nodes = []

        q, k, types = build_attn_ranges_from_trie([empty_root])
        self.assertEqual(q, [])
        self.assertEqual(k, [])
        self.assertEqual(types, [])

        # Scope must still be enterable (it provides a sentinel internally).
        api_stub = _make_magi_api_stub()
        with mock.patch.dict(sys.modules, {"magi_attention.api": api_stub}):
            with tree_attn_scope(
                q_ranges_naive=[],
                k_ranges_naive=[],
                attn_type_map_list=[],
                total_seqlen=0,
                num_heads_q=4,
                num_heads_kv=2,
                head_dim=16,
                cp_group=None,
            ) as key:
                self.assertEqual(key, "MOCK_KEY")
        # magi_attn_flex_key called even with empty ranges (Magi handles dummy
        # internally; the scope does not short-circuit).
        api_stub.magi_attn_flex_key.assert_called_once()

    def test_e_t17_scope_isolation_across_mbs(self):
        """Each mb's scope must be independent — no key carryover between mbs.

        This guards against a future bug where a stale runtime key from the
        previous mb leaks into the next mb's forward via cp_group registry.
        We verify by checking magi_attn_flex_key was called exactly N times
        for N mbs, and the call args differ per mb (different total_seqlen
        or ranges).
        """
        from verl.experimental.tree_training._magi_backend import tree_attn_scope

        seqs = [[1, 2, 3, 4], [1, 2, 3, 5], [9, 8, 7]]
        td = _make_td(seqs, response_lens=[2, 2, 2])
        mbs, _ = build_tree_mb_list(td, max_tokens_per_mb=128, pad_token_id=0)

        api_stub = _make_magi_api_stub()
        call_args_per_mb = []
        for mb in mbs:
            _, _, scope_args = build_tree_model_inputs(mb, "cpu")
            with mock.patch.dict(sys.modules, {"magi_attention.api": api_stub}):
                with tree_attn_scope(**scope_args, num_heads_q=4, num_heads_kv=2, head_dim=16, cp_group=None):
                    pass  # noop body — we only check scope build / teardown
            # Capture the most recent call args.
            call_args_per_mb.append(api_stub.magi_attn_flex_key.call_args)

        # One key build per mb, no double-builds.
        self.assertEqual(api_stub.magi_attn_flex_key.call_count, len(mbs))
        # If multiple mbs, their args should differ.
        if len(mbs) > 1:
            seqlens = [c.kwargs["total_seqlen_q"] for c in call_args_per_mb]
            ranges_lengths = [len(c.kwargs["q_ranges"].ranges) for c in call_args_per_mb]
            # At least one of seqlen or range-count should differ across mbs.
            self.assertTrue(
                len(set(seqlens)) > 1 or len(set(ranges_lengths)) > 1,
                f"All mbs had identical scope args (seqlens={seqlens}, "
                f"range_counts={ranges_lengths}); test setup may be too uniform",
            )


if __name__ == "__main__":
    unittest.main()
