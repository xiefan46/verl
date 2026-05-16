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

"""CPU-only wiring tests for ``_magi_backend.py``.

Validates the plumbing of :class:`TreeCPContext`, :func:`tree_attn_scope`,
``register_tree_attention``, and ``_magi_tree_attention_forward`` without
invoking the actual MagiAttention CUDA kernel. All Magi API entry points are
stubbed at the ``magi_attention.api`` module level via :mod:`unittest.mock`.

These tests exercise wiring contracts:
* ``TreeCPContext`` constructs at ``cp_size=1`` without ``torch.distributed``
* ``setup_model`` propagates ``cp_group`` to attention sub-modules
* ``register_tree_attention`` is idempotent and routes through HF's registry
* ``tree_attn_scope`` builds + yields runtime key cleanly, including on
  exceptions and nested calls
* ``_magi_tree_attention_forward`` falls back to ``flash_attention_2`` when
  no cp_group / no key, and calls ``calc_attn`` with the right shapes when
  both are present

Total: 12 cases.
"""

from __future__ import annotations

import sys
import types
import unittest
from unittest import mock

import torch

# Stub heavy modules at import time so the test file is collectable on
# Mac without ``magi_attention`` / ``transformers`` installed. Production
# CPU testing (RunPod) will have these installed, so the real symbols load;
# we only stub when missing.

if "transformers" not in sys.modules:
    transformers_stub = types.ModuleType("transformers")
    sys.modules["transformers"] = transformers_stub

if "transformers.modeling_utils" not in sys.modules:
    modeling_utils_stub = types.ModuleType("transformers.modeling_utils")

    class _FakeAllAttentionFunctions(dict):
        """Drop-in replacement for HF's ``ALL_ATTENTION_FUNCTIONS`` registry."""

        def register(self, name, fn):
            self[name] = fn

    modeling_utils_stub.ALL_ATTENTION_FUNCTIONS = _FakeAllAttentionFunctions()
    sys.modules["transformers.modeling_utils"] = modeling_utils_stub
    sys.modules["transformers"].modeling_utils = modeling_utils_stub


from verl.experimental.tree_training._magi_backend import (
    TreeCPContext,
    _magi_tree_attention_forward,
    register_tree_attention,
    tree_attn_scope,
)


class _MockAttentionLayer(torch.nn.Module):
    """A minimal stand-in for an HF attention layer (class name contains 'Attention')."""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)


class _MockModelWithAttention(torch.nn.Module):
    """A toy model with 3 attention sub-modules + a config object."""

    def __init__(self):
        super().__init__()
        self.attn_0 = _MockAttentionLayer()
        self.attn_1 = _MockAttentionLayer()
        self.attn_2 = _MockAttentionLayer()
        self.config = types.SimpleNamespace(_attn_implementation="flash_attention_2")


class TestTreeCPContext(unittest.TestCase):
    """E.T1-E.T3 — TreeCPContext construction and model setup."""

    def test_e_t1_cp_size_one_single_process_constructs(self):
        """Single-process (no torch.distributed) cp_size=1 must construct cleanly."""
        ctx = TreeCPContext(cp_size=1)
        self.assertEqual(ctx.cp_size, 1)
        # cp_group is None because torch.distributed is not initialized in test env.
        self.assertIsNone(ctx.cp_group)

    def test_e_t1b_cp_size_zero_raises(self):
        """cp_size < 1 must raise (defense against config bugs)."""
        with self.assertRaises(ValueError):
            TreeCPContext(cp_size=0)

    def test_e_t2_setup_model_propagates_cp_group(self):
        """setup_model walks modules, sets cp_group on each Attention-named one."""
        ctx = TreeCPContext(cp_size=1)
        model = _MockModelWithAttention()

        # Before: no cp_group attribute
        for module in [model.attn_0, model.attn_1, model.attn_2]:
            self.assertFalse(hasattr(module, "cp_group"))

        ctx.setup_model(model)

        # After: all 3 attention modules have cp_group set (to None at cp_size=1)
        for module in [model.attn_0, model.attn_1, model.attn_2]:
            self.assertTrue(hasattr(module, "cp_group"))
            self.assertIs(module.cp_group, ctx.cp_group)

    def test_e_t3_setup_model_sets_attn_implementation(self):
        """setup_model flips config._attn_implementation to Magi_Tree_Attention."""
        ctx = TreeCPContext(cp_size=1)
        model = _MockModelWithAttention()
        self.assertEqual(model.config._attn_implementation, "flash_attention_2")

        ctx.setup_model(model)
        self.assertEqual(model.config._attn_implementation, "Magi_Tree_Attention")


class TestRegisterTreeAttention(unittest.TestCase):
    """E.T4-E.T6 — registration into HF ALL_ATTENTION_FUNCTIONS."""

    def setUp(self):
        # Reset module-level _REGISTERED flag and registry between tests.
        from verl.experimental.tree_training import _magi_backend

        _magi_backend._REGISTERED = False
        # Also clear the stub registry if present (we mutate global state).
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        ALL_ATTENTION_FUNCTIONS.pop("Magi_Tree_Attention", None)

    def test_e_t4_register_idempotent(self):
        """Calling register_tree_attention twice should be a no-op."""
        register_tree_attention()
        register_tree_attention()  # no error, no double-register
        # Verified by no exception. _REGISTERED state is module-level.
        from verl.experimental.tree_training import _magi_backend

        self.assertTrue(_magi_backend._REGISTERED)

    def test_e_t5_register_under_correct_name(self):
        """After registration, ALL_ATTENTION_FUNCTIONS['Magi_Tree_Attention'] exists."""
        register_tree_attention()
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        self.assertIn("Magi_Tree_Attention", ALL_ATTENTION_FUNCTIONS)
        self.assertIs(
            ALL_ATTENTION_FUNCTIONS["Magi_Tree_Attention"],
            _magi_tree_attention_forward,
        )

    def test_e_t6_other_attention_implementations_coexist(self):
        """Registering Magi_Tree_Attention should not displace flash_attention_2."""
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        # Seed the registry with a fake flash_attention_2 entry.
        fake_fa2 = lambda *a, **kw: ("fa2_out", None)  # noqa: E731
        ALL_ATTENTION_FUNCTIONS["flash_attention_2"] = fake_fa2

        register_tree_attention()

        self.assertIs(ALL_ATTENTION_FUNCTIONS["flash_attention_2"], fake_fa2)
        self.assertIn("Magi_Tree_Attention", ALL_ATTENTION_FUNCTIONS)


class TestTreeAttnScope(unittest.TestCase):
    """E.T7-E.T9 — tree_attn_scope context manager."""

    def _patch_magi_api(self):
        """Build a mock magi_attention.api with the symbols tree_attn_scope reads."""
        mock_api = types.ModuleType("magi_attention.api")

        class MockAttnMaskType:
            FULL = "FULL"
            CAUSAL = "CAUSAL"

            @classmethod
            def from_int_type(cls, t):
                return cls.CAUSAL if t == 1 else cls.FULL

        class MockAttnRanges:
            def __init__(self, ranges):
                self.ranges = list(ranges)

            @classmethod
            def from_ranges(cls, ranges):
                return cls(ranges)

        class MockDistAttnConfig:
            pass

        mock_api.AttnMaskType = MockAttnMaskType
        mock_api.AttnRanges = MockAttnRanges
        mock_api.DistAttnConfig = MockDistAttnConfig

        # compute_pad_size: just return 0 (no actual padding).
        mock_api.compute_pad_size = lambda total, cp_size, chunk: 0

        # magi_attn_flex_key: return an opaque sentinel object capturing inputs.
        mock_api.magi_attn_flex_key = mock.MagicMock(return_value="MOCK_KEY")

        # calc_attn and get_most_recent_key are referenced by the forward but
        # not by tree_attn_scope itself.
        mock_api.calc_attn = mock.MagicMock()
        mock_api.get_most_recent_key = mock.MagicMock()

        return mock_api

    def test_e_t7_normal_enter_exit(self):
        """Scope yields the runtime key; magi_attn_flex_key is called with right args."""
        mock_api = self._patch_magi_api()
        with mock.patch.dict(sys.modules, {"magi_attention.api": mock_api}):
            with tree_attn_scope(
                q_ranges_naive=[(0, 4)],
                k_ranges_naive=[(0, 4)],
                attn_type_map_list=[1],  # CAUSAL
                total_seqlen=4,
                num_heads_q=4,
                num_heads_kv=2,
                head_dim=64,
                cp_group=None,
                chunk_size=512,
            ) as key:
                self.assertEqual(key, "MOCK_KEY")

            # Verify magi_attn_flex_key was called with our parameters.
            mock_api.magi_attn_flex_key.assert_called_once()
            call_kwargs = mock_api.magi_attn_flex_key.call_args.kwargs
            self.assertEqual(call_kwargs["total_seqlen_q"], 4)
            self.assertEqual(call_kwargs["total_seqlen_k"], 4)
            self.assertEqual(call_kwargs["num_heads_q"], 4)
            self.assertEqual(call_kwargs["num_heads_kv"], 2)
            self.assertEqual(call_kwargs["head_dim"], 64)
            self.assertEqual(call_kwargs["chunk_size"], 512)
            self.assertIsNone(call_kwargs["cp_group_or_mesh"])

    def test_e_t8_exception_path_does_not_leak(self):
        """Exception inside scope should propagate cleanly; no state left over."""
        mock_api = self._patch_magi_api()
        with mock.patch.dict(sys.modules, {"magi_attention.api": mock_api}):
            with self.assertRaises(RuntimeError):
                with tree_attn_scope(
                    q_ranges_naive=[(0, 4)],
                    k_ranges_naive=[(0, 4)],
                    attn_type_map_list=[1],
                    total_seqlen=4,
                    num_heads_q=4,
                    num_heads_kv=2,
                    head_dim=64,
                    cp_group=None,
                ):
                    raise RuntimeError("inside scope")

    def test_e_t9_nested_scopes(self):
        """Nested scopes: inner key is most-recent; outer not corrupted on exit.

        Magi's runtime registry has "most recent" semantics — both keys exist
        in the registry, but ``get_most_recent_key`` returns the most recently
        registered. We just verify magi_attn_flex_key is called twice with
        the expected args.
        """
        mock_api = self._patch_magi_api()
        with mock.patch.dict(sys.modules, {"magi_attention.api": mock_api}):
            with tree_attn_scope(
                q_ranges_naive=[(0, 8)],
                k_ranges_naive=[(0, 8)],
                attn_type_map_list=[1],
                total_seqlen=8,
                num_heads_q=4,
                num_heads_kv=2,
                head_dim=64,
                cp_group=None,
            ) as outer_key:
                self.assertEqual(outer_key, "MOCK_KEY")
                with tree_attn_scope(
                    q_ranges_naive=[(0, 4)],
                    k_ranges_naive=[(0, 4)],
                    attn_type_map_list=[1],
                    total_seqlen=4,
                    num_heads_q=4,
                    num_heads_kv=2,
                    head_dim=64,
                    cp_group=None,
                ) as inner_key:
                    self.assertEqual(inner_key, "MOCK_KEY")

            self.assertEqual(mock_api.magi_attn_flex_key.call_count, 2)


class TestMagiTreeAttentionForward(unittest.TestCase):
    """E.T10-E.T12 — the HF-registered forward function."""

    def _make_fake_qkv(self, batch=1, heads=4, seqlen=8, head_dim=16):
        q = torch.randn(batch, heads, seqlen, head_dim, dtype=torch.float32)
        k = torch.randn(batch, heads, seqlen, head_dim, dtype=torch.float32)
        v = torch.randn(batch, heads, seqlen, head_dim, dtype=torch.float32)
        return q, k, v

    def test_e_t10_falls_back_when_no_cp_group(self):
        """Module without cp_group attribute -> flash_attention_2 path."""
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        fake_fa2 = mock.MagicMock(return_value=("FA2_OUT", None))
        ALL_ATTENTION_FUNCTIONS["flash_attention_2"] = fake_fa2

        module = _MockAttentionLayer()  # no cp_group set
        q, k, v = self._make_fake_qkv()

        out, attn_w = _magi_tree_attention_forward(module, q, k, v, attention_mask=None, scaling=1.0)
        self.assertEqual(out, "FA2_OUT")
        fake_fa2.assert_called_once()

    def test_e_t11_falls_back_when_no_key(self):
        """Module with cp_group but no registered key -> flash_attention_2 path."""
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        fake_fa2 = mock.MagicMock(return_value=("FA2_OUT", None))
        ALL_ATTENTION_FUNCTIONS["flash_attention_2"] = fake_fa2

        module = _MockAttentionLayer()
        module.cp_group = "FAKE_GROUP"

        # Stub get_most_recent_key to return None (no key for this group).
        mock_api = types.ModuleType("magi_attention.api")
        mock_api.get_most_recent_key = mock.MagicMock(return_value=None)
        mock_api.calc_attn = mock.MagicMock()

        with mock.patch.dict(sys.modules, {"magi_attention.api": mock_api}):
            q, k, v = self._make_fake_qkv()
            out, _ = _magi_tree_attention_forward(module, q, k, v, attention_mask=None, scaling=1.0)
        self.assertEqual(out, "FA2_OUT")
        # calc_attn should not have been called (no key path).
        mock_api.calc_attn.assert_not_called()

    def test_e_t12_calls_calc_attn_with_correct_shapes(self):
        """When cp_group + key present, calc_attn called with (S, H, D) shaped tensors."""
        # einops is a runtime dep of _magi_backend's hot path; skip gracefully if missing.
        try:
            import einops  # noqa: F401
        except ImportError:
            self.skipTest("einops not installed in test env")

        module = _MockAttentionLayer()
        module.cp_group = "FAKE_GROUP"

        # calc_attn returns (out_tensor, ?). Pre-compute the expected (S, H, D) shape.
        batch, heads, seqlen, head_dim = 1, 4, 8, 16
        expected_out = torch.randn(seqlen, heads, head_dim, dtype=torch.bfloat16)

        mock_api = types.ModuleType("magi_attention.api")
        mock_api.get_most_recent_key = mock.MagicMock(return_value="MOCK_KEY")
        mock_api.calc_attn = mock.MagicMock(return_value=(expected_out, None))

        with mock.patch.dict(sys.modules, {"magi_attention.api": mock_api}):
            q, k, v = self._make_fake_qkv(batch, heads, seqlen, head_dim)
            out, attn_w = _magi_tree_attention_forward(module, q, k, v, attention_mask=None, scaling=0.5)

        # Verify calc_attn was called.
        mock_api.calc_attn.assert_called_once()
        call_args = mock_api.calc_attn.call_args.args
        q_arg, k_arg, v_arg, key_arg = call_args
        # Shape should be (S, H, D) — squashed batch dim, bf16 dtype.
        self.assertEqual(q_arg.shape, (seqlen, heads, head_dim))
        self.assertEqual(k_arg.shape, (seqlen, heads, head_dim))
        self.assertEqual(v_arg.shape, (seqlen, heads, head_dim))
        self.assertEqual(q_arg.dtype, torch.bfloat16)
        self.assertEqual(key_arg, "MOCK_KEY")

        # Output should be back in HF layout (1, S, H*D) and original dtype.
        self.assertEqual(out.shape, (1, seqlen, heads * head_dim))
        self.assertEqual(out.dtype, q.dtype)  # back to float32


if __name__ == "__main__":
    unittest.main()
