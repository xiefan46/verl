# Copyright 2025-2026 The AReaL Authors (Ant Group, Tsinghua University, HKUST)
# Copyright 2026 Bytedance Ltd. and/or its affiliates (verl integration & modifications)
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
#
# This file is copied (with modifications) from AReaL:
#   https://github.com/inclusionAI/AReaL
# Original location: areal/models/tree_attn/module.py
#
# Based on the AReaL-DTA paper (arXiv:2602.00482):
#   "AReaL-DTA: Dynamic Tree Attention for Efficient Reinforcement Learning
#    of Large Language Models"
#   Jiarui Zhang, Yuchen Yang, Ran Yan, Zhiyu Mei, Liyuan Zhang, Daifeng Li,
#   Wei Fu, Jiaxuan Gao, Shusheng Xu, Yi Wu, Binhang Yuan
#   https://arxiv.org/abs/2602.00482

from verl.experimental.tree_training.constants import USE_TRITON_TREE_ATTN
from verl.experimental.tree_training.module_fsdp import (
    create_block_mask_from_dense,
    patch_fsdp_for_tree_training,
    restore_patch_fsdp_for_tree_training,
)
from verl.experimental.tree_training.tree import (
    build_attention_mask_from_trie,
    build_block_mask_from_trie,
    build_tree_attn_kwargs,
    build_triton_attn_data_from_trie,
)

# Conditionally import Triton functionality
try:
    from verl.experimental.tree_training.triton_kernel import (
        TRITON_AVAILABLE,
        TreeAttentionData,
        tree_attention,
    )
except ImportError:
    TRITON_AVAILABLE = False
    TreeAttentionData = None
    tree_attention = None

# Megatron / Archon backends from AReaL were intentionally not vendored:
#   - module_megatron.py: verl uses its own Megatron path; integration deferred (Phase V2).
#   - module_archon.py: verl has no Archon engine.

__all__ = [
    # Shared constants
    "USE_TRITON_TREE_ATTN",
    # FSDP/common exports
    "create_block_mask_from_dense",
    "patch_fsdp_for_tree_training",
    "restore_patch_fsdp_for_tree_training",
    "build_attention_mask_from_trie",
    "build_block_mask_from_trie",
    "build_tree_attn_kwargs",
    "build_triton_attn_data_from_trie",
    # Triton exports (may be None if Triton not installed)
    "TRITON_AVAILABLE",
    "TreeAttentionData",
    "tree_attention",
]
