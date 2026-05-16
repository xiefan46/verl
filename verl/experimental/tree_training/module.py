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

"""Public re-export hub for the tree training package.

MagiAttention-backed entry points (V1 FSDP2-only). The previous flex_attention
re-exports (``create_block_mask_from_dense``, ``patch_fsdp_for_tree_training``,
``build_block_mask_from_trie``, ``build_tree_attn_kwargs``, etc.) are gone — see
``_areal_legacy/`` for the archived flex_attention implementation and
``research/2026-05-16-magi-integration-plan-v2.md`` for the migration design.

Megatron / Archon backends from AReaL were intentionally not vendored:
  - module_megatron.py: verl has its own Megatron path; integration is V4 scope.
  - module_archon.py: verl has no Archon engine.
"""

from verl.experimental.tree_training._magi_backend import (
    TreeCPContext,
    register_tree_attention,
    tree_attn_scope,
)
from verl.experimental.tree_training._magi_kernel import (
    build_attn_ranges_from_trie,
    build_attn_ranges_tensors,
)

__all__ = [
    # Magi mask construction
    "build_attn_ranges_from_trie",
    "build_attn_ranges_tensors",
    # Magi HF + CP integration
    "TreeCPContext",
    "register_tree_attention",
    "tree_attn_scope",
]
