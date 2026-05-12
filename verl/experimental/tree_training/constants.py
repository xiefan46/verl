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
# Original location: areal/models/tree_attn/constants.py
#
# Based on the AReaL-DTA paper (arXiv:2602.00482):
#   "AReaL-DTA: Dynamic Tree Attention for Efficient Reinforcement Learning
#    of Large Language Models"
#   Jiarui Zhang, Yuchen Yang, Ran Yan, Zhiyu Mei, Liyuan Zhang, Daifeng Li,
#   Wei Fu, Jiaxuan Gao, Shusheng Xu, Yi Wu, Binhang Yuan
#   https://arxiv.org/abs/2602.00482

"""Shared constants for tree attention functionality."""

import logging
import os

logger = logging.getLogger("TreeAttentionConstants")

BLOCK_SIZE = int(os.environ.get("AREAL_FLEX_ATTENTION_BLOCK_SIZE", "128"))
USE_TRITON_TREE_ATTN = int(os.environ.get("AREAL_USE_TRITON_TREE_ATTN", "0")) == 1

if USE_TRITON_TREE_ATTN:
    logger.warning(
        "Triton tree attention kernel is only an experimental feature "
        "that requires further practical RL experiment testing."
    )
