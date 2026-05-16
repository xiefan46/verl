# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Legacy AReaL/flex_attention tree-training tests — reference only.

These tests were written against the flex_attention path that has been
replaced by MagiAttention (see 2026-05-16-magi-integration-plan-v2.md). They
import symbols that no longer exist (``build_tree_attn_kwargs``,
``patch_fsdp_for_tree_training``, etc.) and are not run by the live test suite.

After Phase J e2e validation on RunPod (Sunday), the most useful ones
(forward_equivalence, e2e_tree_training_step) will be rewritten against the
Magi path and moved back out of this directory.
"""
