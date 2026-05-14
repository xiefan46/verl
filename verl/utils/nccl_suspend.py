# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""NCCL communicator suspend/resume utilities for colocated mode.

Uses the NCCL 2.29.7+ native ``ncclCommSuspend`` / ``ncclCommResume`` API
(loaded via ctypes from ``libnccl.so.2``) to release the GPU memory held by
idle communicators during sleep/wake transitions. On older NCCL the public
entry points gracefully no-op so callers can enable the feature unconditionally.

This module implements **Method A**: a reflective scan of
``megatron.core.parallel_state``'s named group globals. Other training
backends are expected to land their own enumeration strategy when they
expose an engine-native suspend/resume API.

References:
  * RFC: https://github.com/verl-project/verl/issues/6266
  * NCCL 2.29.7 release notes: https://github.com/NVIDIA/nccl/releases/tag/v2.29.7-1
  * Suspend/Resume API: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html
"""

from __future__ import annotations

import ctypes
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import torch

logger = logging.getLogger(__name__)
# verl/__init__.py pins the root logger at WARNING via set_basic_config, so
# module loggers do not inherit INFO by default. Mirror the pattern used in
# verl/utils/memory_utils.py: opt this module into VERL_LOGGING_LEVEL so the
# suspend/resume telemetry surfaces when users (or e2e scripts) request INFO.
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Bitmask passed to ncclCommSuspend: release dynamic GPU memory allocations
# while preserving topology / connection state.
NCCL_SUSPEND_MEM = 0x01
NCCL_SUCCESS = 0

_nccl_lib: Optional[ctypes.CDLL] = None
_lib_load_attempted: bool = False


@dataclass
class CommStat:
    """Per-communicator suspend or resume statistics."""

    name: str
    handle: int
    duration_ms: float
    # Memory delta in MB attributed to this communicator. Only populated when
    # ``measure_per_comm=True`` is passed to the public entry point. Default 0.
    delta_mb: float = 0.0
    success: bool = True


@dataclass
class SuspendResult:
    """Aggregate result of a suspend call."""

    success: bool = False
    skipped_reason: Optional[str] = None
    freed_mb: float = 0.0
    total_ms: float = 0.0
    comms: list[CommStat] = field(default_factory=list)


@dataclass
class ResumeResult:
    """Aggregate result of a resume call."""

    success: bool = False
    skipped_reason: Optional[str] = None
    reclaimed_mb: float = 0.0
    total_ms: float = 0.0
    comms: list[CommStat] = field(default_factory=list)


def _get_nccl_lib() -> Optional[ctypes.CDLL]:
    """Lazily load ``libnccl.so.2`` and resolve ``ncclCommSuspend`` / ``ncclCommResume``.

    Returns ``None`` (cached) if the library can't be loaded or the suspend/resume
    symbols aren't present (NCCL < 2.29.7).
    """
    global _nccl_lib, _lib_load_attempted
    if _lib_load_attempted:
        return _nccl_lib
    _lib_load_attempted = True

    try:
        lib = ctypes.CDLL("libnccl.so.2")
    except OSError:
        logger.warning("Failed to load libnccl.so.2; NCCL suspend/resume disabled.")
        return None

    if not hasattr(lib, "ncclCommSuspend") or not hasattr(lib, "ncclCommResume"):
        logger.warning(
            "libnccl.so.2 does not export ncclCommSuspend/ncclCommResume; "
            "NCCL suspend/resume disabled. Requires NCCL >= 2.29.7."
        )
        return None

    lib.ncclCommSuspend.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.ncclCommSuspend.restype = ctypes.c_int
    lib.ncclCommResume.argtypes = [ctypes.c_void_p]
    lib.ncclCommResume.restype = ctypes.c_int

    _nccl_lib = lib
    return lib


def is_supported() -> bool:
    """Whether the loaded NCCL library supports ncclCommSuspend/Resume (>= 2.29.7)."""
    return _get_nccl_lib() is not None


def _gpu_used_mb() -> float:
    """Driver-level GPU memory used (MB).

    Uses ``cuMemGetInfo`` via ``torch.cuda.mem_get_info`` so we capture NCCL's
    ``cudaMalloc`` allocations, which the CUDA caching allocator does not see.
    """
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    return (total - free) / 1024**2


def _suspend_one(handle: int) -> bool:
    lib = _get_nccl_lib()
    if lib is None or handle == 0:
        return False
    rc = lib.ncclCommSuspend(ctypes.c_void_p(handle), NCCL_SUSPEND_MEM)
    if rc != NCCL_SUCCESS:
        logger.warning("ncclCommSuspend failed: handle=0x%x rc=%d", handle, rc)
        return False
    return True


def _resume_one(handle: int) -> bool:
    lib = _get_nccl_lib()
    if lib is None or handle == 0:
        return False
    rc = lib.ncclCommResume(ctypes.c_void_p(handle))
    if rc != NCCL_SUCCESS:
        logger.warning("ncclCommResume failed: handle=0x%x rc=%d", handle, rc)
        return False
    return True


def _suspend_batch(handles: list[tuple[str, int]], *, measure_per_comm: bool) -> SuspendResult:
    """Suspend a batch of ``(name, handle)`` communicators.

    When ``measure_per_comm`` is True, inserts ``empty_cache + synchronize``
    between each suspend so the freed memory can be attributed to individual
    comms. This adds roughly 5-10 ms per comm and is intended for tests; the
    production hot path leaves it off.
    """
    if not handles:
        return SuspendResult(success=False, skipped_reason="no_warm_comms")

    total_before = _gpu_used_mb()
    total_start = time.perf_counter()
    comms_stats: list[CommStat] = []

    for name, handle in handles:
        before_mb = _gpu_used_mb() if measure_per_comm else 0.0
        t0 = time.perf_counter()
        ok = _suspend_one(handle)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        delta_mb = 0.0
        if measure_per_comm:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            delta_mb = before_mb - _gpu_used_mb()

        comms_stats.append(CommStat(name=name, handle=handle, duration_ms=elapsed_ms, delta_mb=delta_mb, success=ok))

    total_ms = (time.perf_counter() - total_start) * 1000
    if not measure_per_comm:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    freed_mb = total_before - _gpu_used_mb()

    n_ok = sum(1 for c in comms_stats if c.success)
    logger.info("NCCL suspend: %d/%d comms in %.0f ms, freed %.0f MB", n_ok, len(comms_stats), total_ms, freed_mb)
    return SuspendResult(success=n_ok > 0, freed_mb=freed_mb, total_ms=total_ms, comms=comms_stats)


def _resume_batch(handles: list[tuple[str, int]], *, measure_per_comm: bool) -> ResumeResult:
    """Resume a batch of ``(name, handle)`` communicators."""
    if not handles:
        return ResumeResult(success=False, skipped_reason="no_warm_comms")

    total_before = _gpu_used_mb()
    total_start = time.perf_counter()
    comms_stats: list[CommStat] = []

    for name, handle in handles:
        before_mb = _gpu_used_mb() if measure_per_comm else 0.0
        t0 = time.perf_counter()
        ok = _resume_one(handle)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        delta_mb = 0.0
        if measure_per_comm:
            torch.cuda.synchronize()
            delta_mb = _gpu_used_mb() - before_mb

        comms_stats.append(CommStat(name=name, handle=handle, duration_ms=elapsed_ms, delta_mb=delta_mb, success=ok))

    total_ms = (time.perf_counter() - total_start) * 1000
    torch.cuda.synchronize()
    reclaimed_mb = _gpu_used_mb() - total_before

    n_ok = sum(1 for c in comms_stats if c.success)
    logger.info(
        "NCCL resume: %d/%d comms in %.0f ms, reclaimed %.0f MB",
        n_ok,
        len(comms_stats),
        total_ms,
        reclaimed_mb,
    )
    return ResumeResult(success=n_ok > 0, reclaimed_mb=reclaimed_mb, total_ms=total_ms, comms=comms_stats)


# Module state for idempotent suspend/resume of Megatron communicators.
_megatron_suspended: bool = False
_megatron_suspended_handles: list[tuple[str, int]] = []


def _collect_megatron_comms() -> list[tuple[str, int]]:
    """Reflect over ``megatron.core.parallel_state``'s named globals to collect
    every warm NCCL ``ncclComm_t`` handle.

    Walks module-level attributes matching ``_*GROUP*`` and skips ``_*GLOO*``
    (CPU-only). Handles three container shapes Megatron uses for its group
    globals:

      * Singleton ``ProcessGroup`` (most groups, e.g. ``_TENSOR_MODEL_PARALLEL_GROUP``)
      * List of groups (``_HIERARCHICAL_CONTEXT_PARALLEL_GROUPS``)
      * Dict of groups (``_HYBRID_DP_CP_GROUPS``)

    Deduplicates by ``ncclComm_t`` handle: PyTorch may share an underlying
    communicator across multiple ``ProcessGroup`` objects with the same rank
    set, and ``ncclCommSuspend`` errors when called twice on the same handle.

    Returns ``[(display_name, handle_int), ...]``. Empty list if Megatron is
    unavailable or model parallel is not yet initialized.
    """
    try:
        from megatron.core import parallel_state as ps
    except ImportError:
        logger.debug("megatron.core.parallel_state not importable; skipping comm collection.")
        return []

    try:
        if not ps.model_parallel_is_initialized():
            logger.debug("Megatron model parallel not initialized; skipping comm collection.")
            return []
    except Exception as e:
        logger.debug("Megatron model_parallel_is_initialized check failed: %s", e)
        return []

    handles: list[tuple[str, int]] = []
    seen_ptrs: set[int] = set()

    for attr_name in sorted(dir(ps)):
        if not attr_name.startswith("_") or "GROUP" not in attr_name or "GLOO" in attr_name:
            continue
        attr = getattr(ps, attr_name, None)
        if attr is None:
            continue

        if isinstance(attr, dict):
            items = [(f"{attr_name.lstrip('_')}[{k}]", v) for k, v in attr.items()]
        elif isinstance(attr, list | tuple):
            items = [(f"{attr_name.lstrip('_')}[{i}]", v) for i, v in enumerate(attr)]
        else:
            items = [(attr_name.lstrip("_"), attr)]

        for label, pg in items:
            if pg is None:
                continue
            try:
                backend = pg._get_backend(torch.device("cuda"))
            except Exception:
                continue
            if not hasattr(backend, "_comm_ptr"):
                continue
            try:
                ptr = backend._comm_ptr()
            except Exception:
                continue
            if ptr == 0 or ptr in seen_ptrs:
                continue
            seen_ptrs.add(ptr)
            handles.append((label, int(ptr)))

    logger.info(
        "Method A discovered %d warm Megatron NCCL comm(s): %s",
        len(handles),
        [name for name, _ in handles],
    )
    return handles


def suspend_via_parallel_state(*, measure_per_comm: bool = False) -> SuspendResult:
    """Suspend all warm NCCL comms reachable via ``megatron.core.parallel_state``.

    Idempotent: if already suspended, returns a no-op result.

    Args:
        measure_per_comm: When True, attribute freed memory per communicator.
            Adds ~5-10 ms per comm; intended for tests, off in production.
    """
    global _megatron_suspended, _megatron_suspended_handles

    if not is_supported():
        return SuspendResult(success=False, skipped_reason="nccl_too_old")

    if _megatron_suspended:
        logger.debug("Megatron NCCL comms already suspended; no-op.")
        return SuspendResult(success=False, skipped_reason="already_suspended")

    handles = _collect_megatron_comms()
    if not handles:
        return SuspendResult(success=False, skipped_reason="no_warm_comms")

    result = _suspend_batch(handles, measure_per_comm=measure_per_comm)
    if result.success:
        _megatron_suspended = True
        _megatron_suspended_handles = handles
    return result


def resume_via_parallel_state(*, measure_per_comm: bool = False) -> ResumeResult:
    """Resume the Megatron NCCL comms suspended by the last
    :func:`suspend_via_parallel_state` call.

    Idempotent: if not suspended, returns a no-op result.
    """
    global _megatron_suspended

    if not is_supported():
        return ResumeResult(success=False, skipped_reason="nccl_too_old")

    if not _megatron_suspended:
        logger.debug("Megatron NCCL comms not suspended; no-op.")
        return ResumeResult(success=False, skipped_reason="not_suspended")

    handles = _megatron_suspended_handles
    if not handles:
        return ResumeResult(success=False, skipped_reason="no_warm_comms")

    result = _resume_batch(handles, measure_per_comm=measure_per_comm)
    if result.success:
        _megatron_suspended = False
    return result
