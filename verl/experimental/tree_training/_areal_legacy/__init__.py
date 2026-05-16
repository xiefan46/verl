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

"""Legacy AReaL flex_attention path — reference only, NOT used at runtime.

Files in this directory were moved here from ``verl/experimental/tree_training/``
during the MagiAttention migration (2026-05-16). They preserve the original
flex_attention-based tree training implementation for historical reference and
numerical comparison, but are not imported by the live training pipeline.

The current backend lives at:
- ``verl/experimental/tree_training/_magi_kernel.py``  — trie -> Magi AttnRanges
- ``verl/experimental/tree_training/_magi_backend.py`` — HF + MagiAttention wiring

Do NOT add new code under this directory. To inspect the original
flex_attention implementation, refer to git history before the migration commit
or to the files moved here:
- ``module_fsdp.py`` (moved in Phase E)
- ``triton_kernel.py`` (moved in Phase D)
"""
