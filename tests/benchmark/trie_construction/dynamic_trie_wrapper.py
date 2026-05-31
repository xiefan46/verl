# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Benchmark-only thin wrapper around verl/utils/prefix_tree_dynamic.py.
# Adds 3-layer timing instrumentation; skips attention-key construction
# (which needs GPU + MAGI install) so the benchmark stays CPU-only.

"""Benchmark wrapper around the dynamic-trie production module with timing."""

from __future__ import annotations

import importlib.util
import os
import sys
import time
from typing import Optional


def _load_verl_module(rel_path: str, mod_name: str):
    """Load a verl util module without triggering verl/__init__.py heavy deps."""
    here = os.path.dirname(os.path.abspath(__file__))
    full = os.path.normpath(os.path.join(here, "../../..", rel_path))
    spec = importlib.util.spec_from_file_location(mod_name, full)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


# Load production modules in dependency order
_ptp = _load_verl_module("verl/utils/prefix_tree_params.py", "verl.utils.prefix_tree_params")
_ptu = _load_verl_module("verl/utils/prefix_tree_utils.py", "verl.utils.prefix_tree_utils")
_ptm = _load_verl_module("verl/utils/prefix_tree_magi.py", "verl.utils.prefix_tree_magi")
_dynamic = _load_verl_module("verl/utils/prefix_tree_dynamic.py", "verl.utils.prefix_tree_dynamic")

# Re-export production-API symbols for compatibility with the benchmark harness
PrefixTreeMagiBatch = _ptm.PrefixTreeMagiBatch
PrefixTreeParams = _ptp.PrefixTreeParams
TreeNode = _ptu.TreeNode
build_multilevel_flex_spec = _ptu.build_multilevel_flex_spec

# Production entry — re-export
build_prefix_tree_micro_batch_dynamic = _dynamic.build_prefix_tree_micro_batch_dynamic

# Lower-level production helpers (used by benchmark `_full` variant for timing)
_unpack = _dynamic.unpack_nested_to_list
_greedy_build_tries = _dynamic.greedy_build_tries
_convert_trie_to_tree_node = _dynamic.convert_trie_to_tree_node
_build_arbitrary_depth_params = _dynamic.build_arbitrary_depth_params


def tree_max_depth(node: TreeNode) -> int:
    """Max depth of a TreeNode tree."""
    if not node.children:
        return 1
    return 1 + max(tree_max_depth(c) for c in node.children)


# ============================================================================
# Benchmark variant: returns (batch, params, timings) and SKIPS key construction
# ============================================================================


def build_prefix_tree_micro_batch_dynamic_full(
    model,
    input_ids,
    loss_mask=None,
    position_ids=None,
    prefix_segments_batch=None,  # dynamic path ignores this
    attention_type: str = "flex",
    tp_size: int = 1,
    cp_size: int = 1,
) -> tuple[Optional[PrefixTreeMagiBatch], Optional[PrefixTreeParams], dict]:
    """Benchmark variant of build_prefix_tree_micro_batch_dynamic.

    Same algorithm as the production entry, but:
      - Returns (batch, params, timings) instead of just batch
      - SKIPS magi_key / flex_key construction (CPU benchmark)
      - 3-layer timing decomposition:
          unpack_ms / tree_detect_ms / pack_ms / total_ms

    Calls the same production-side helpers (`_greedy_build_tries`,
    `_convert_trie_to_tree_node`, `_build_arbitrary_depth_params`) so any
    correctness drift in production code is reflected here.
    """
    timings: dict[str, float] = {}
    t_total = time.perf_counter()

    t0 = time.perf_counter()
    tokens_by_sample = _unpack(input_ids)
    loss_masks_by_sample = _unpack(loss_mask)
    position_ids_by_sample = _unpack(position_ids)
    timings["unpack_ms"] = (time.perf_counter() - t0) * 1000

    if not tokens_by_sample:
        timings["total_ms"] = (time.perf_counter() - t_total) * 1000
        return None, None, timings

    # Tree detect = trie build + convert
    t0 = time.perf_counter()
    sequences = [t.tolist() for t in tokens_by_sample]
    max_tokens_per_tree = sum(len(s) for s in sequences) * 10
    tries, _ = _greedy_build_tries(sequences, max_tokens_per_tree=max_tokens_per_tree)
    if not tries or len(tries) > 1:
        timings["tree_detect_ms"] = (time.perf_counter() - t0) * 1000
        timings["total_ms"] = (time.perf_counter() - t_total) * 1000
        return None, None, timings
    converted = _convert_trie_to_tree_node(tries[0])
    if converted is None:
        timings["tree_detect_ms"] = (time.perf_counter() - t0) * 1000
        timings["total_ms"] = (time.perf_counter() - t_total) * 1000
        return None, None, timings
    tree_root, node_info, leaves_in_dfs = converted
    timings["tree_detect_ms"] = (time.perf_counter() - t0) * 1000

    # Pack
    t0 = time.perf_counter()
    params = _build_arbitrary_depth_params(
        tokens_by_sample,
        tree_root,
        node_info,
        leaves_in_dfs,
        loss_masks_by_sample=loss_masks_by_sample,
        position_ids_by_sample=position_ids_by_sample,
    )
    timings["pack_ms"] = (time.perf_counter() - t0) * 1000

    real_tokens = params.flat_tokens.shape[0]
    # Construct PrefixTreeMagiBatch WITHOUT magi_key/flex_key (benchmark skip)
    pt_batch = PrefixTreeMagiBatch(
        flat_input_ids=params.flat_tokens,
        flat_position_ids=params.flat_position_ids,
        flat_loss_mask=params.flat_loss_mask,
        magi_key=None,
        flex_key=None,
        leaf_to_sample=params.leaf_to_sample,
        leaf_ranges=params.leaf_ranges,
        prefix_range=params.prefix_range,
        original_batch_size=len(tokens_by_sample),
        real_tokens=real_tokens,
        leaf_ancestor_ranges=None,
    )
    timings["total_ms"] = (time.perf_counter() - t_total) * 1000
    return pt_batch, params, timings
