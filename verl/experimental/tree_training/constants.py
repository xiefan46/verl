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

"""Shared constants for tree attention functionality.

V2 (MagiAttention) migration: the original ``AREAL_FLEX_ATTENTION_BLOCK_SIZE``
and ``AREAL_USE_TRITON_TREE_ATTN`` env vars (flex_attention-specific) are
replaced by Magi-side alignment constants. The greedy packer keeps a 128-token
block alignment for downstream stability; Magi handles chunk-size alignment
internally via ``compute_pad_size``.
"""

import os

# Greedy tree packer block alignment. Historically tied to the flex_attention
# BLOCK_SIZE default (128); kept at 128 for the Magi path since the packer
# math (``math.lcm(BLOCK_SIZE, parallel_size)`` at tree.py:_greedy_build_tries)
# works the same. Override via env if necessary.
BLOCK_SIZE = int(os.environ.get("VERL_TREE_PACKER_BLOCK_SIZE", "128"))

# MagiAttention chunk-size alignment passed to ``compute_pad_size`` /
# ``magi_attn_flex_key`` (see ``_magi_backend.tree_attn_scope``). Matches the
# default used in MagiAttention's torch_native example (examples/torch_native).
MAGI_CHUNK_SIZE = int(os.environ.get("VERL_MAGI_CHUNK_SIZE", "512"))
