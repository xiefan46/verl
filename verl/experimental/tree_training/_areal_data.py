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
# Minimal vendored versions of dataclasses from AReaL:
#   - MicroBatchSpec (areal/api/cli_args.py)
#   - MicroBatchItem, MicroBatchList (areal/utils/data.py)
#
# Vendored to keep the tree training algorithm self-contained inside verl
# without pulling in the full AReaL config / data utility tree. Phase 2 of the
# verl tree-training MVP will wire DataProto into MicroBatchList via an adapter;
# at that point these stubs may be replaced or moved.
#
# Notes on the modifications relative to upstream:
#   - PACKING_ALGORITHMS validation dropped (verl ships only the tree packer).
#   - MicroBatchList.to() dropped (not used by tree algorithm; verl moves
#     tensors via DataProto.to upstream of micro-batching).

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, NamedTuple

import torch


@dataclass
class MicroBatchSpec:
    """Specification for splitting micro-batches during training."""

    n_mbs: int | None = field(
        default=1,
        metadata={
            "help": (
                "Number of micro-batches (or minimum number if max_tokens_per_mb is set). "
                "Used when max_tokens_per_mb is None or as minimum count."
            ),
        },
    )
    granularity: int = field(
        default=1,
        metadata={
            "help": (
                "Granularity of each micro-batch. Adjacent sequences are grouped by "
                "this size when dividing microbatches."
            ),
        },
    )
    max_tokens_per_mb: int | None = field(
        default=None,
        metadata={
            "help": (
                "Maximum tokens per micro-batch for each forward pass. When set, "
                "n_mbs becomes the minimum number of micro-batches."
            ),
        },
    )
    n_mbs_divisor: int = field(
        default=1,
        metadata={
            "help": (
                "Divisor for the number of micro-batches. The final number of micro-batches "
                "will be adjusted to be divisible by this value."
            ),
        },
    )

    @classmethod
    def new(cls, mb_spec: "MicroBatchSpec", **kwargs: Any) -> "MicroBatchSpec":
        """Create new spec with updated fields while maintaining Omegaconf compatibility."""
        fields = dict(
            n_mbs=mb_spec.n_mbs,
            granularity=mb_spec.granularity,
            max_tokens_per_mb=mb_spec.max_tokens_per_mb,
            n_mbs_divisor=mb_spec.n_mbs_divisor,
        )
        fields.update(kwargs)
        return cls(**fields)


class MicroBatchItem(NamedTuple):
    """A single micro-batch item from MicroBatchList iteration."""

    orig_mb: dict[str, Any]
    padded_mb: dict[str, Any]
    padding_length: int
    old_cu_seqlens: torch.Tensor | None
    padded_to_length: int | None = None


@dataclass
class MicroBatchList:
    data: dict[str, Any]
    mb_spec: MicroBatchSpec
    mbs: list[dict[str, Any]]
    group_lens: list[int]
    forward_indices: list[int] | None = None
    backward_indices: list[int] | None = None
    padded_mbs: list[dict[str, Any]] | None = None
    _max_seqlen: int | None = None
    # Batch-level padding information
    padding_lengths: list[int] | None = None
    padded_to_lengths: list[int] | None = None
    # sequence-level padding information
    align_to_lengths: list[int] | None = None
    old_cu_seqlens_list: list[torch.Tensor] | None = None

    @property
    def max_seqlen(self) -> int:
        """Return the maximum sequence length across all padded micro-batches."""
        if self.padded_mbs is None:
            raise ValueError("padded_mbs is None. Call pad_mb_list first.")
        if self._max_seqlen is None:
            assert all("cu_seqlens" in m for m in self.padded_mbs), "cu_seqlens not found in some padded micro-batches."
            self._max_seqlen = max(m["cu_seqlens"][-1].item() for m in self.padded_mbs)
        return self._max_seqlen

    def __len__(self) -> int:
        return len(self.mbs)

    def __iter__(self) -> Iterator[MicroBatchItem]:
        """Iterate over micro-batches, yielding MicroBatchItem named tuples."""
        if self.padded_mbs is None:
            raise ValueError("padded_mbs is None. Call pad_mb_list first.")
        for i in range(len(self.mbs)):
            old_cu_seqlens = self.old_cu_seqlens_list[i] if self.old_cu_seqlens_list else None
            padded_to_length = self.padded_to_lengths[i] if self.padded_to_lengths else None
            yield MicroBatchItem(
                orig_mb=self.mbs[i],
                padded_mb=self.padded_mbs[i],
                padding_length=self.padding_lengths[i] if self.padding_lengths else 0,
                old_cu_seqlens=old_cu_seqlens,
                padded_to_length=padded_to_length,
            )
