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
"""
NCCL Communicator Suspend/Resume utilities.

Uses NCCL 2.29.7+ native ncclCommSuspend/ncclCommResume API to release
GPU memory held by idle NCCL communicators during colocated training/inference.

Two independent comm sets are managed:
  - Training comms: torch.distributed ProcessGroups (FSDP/Megatron TP/DP/EP/CP/PP)
  - Rollout comms: vLLM pynccl communicators (TP group, PP group)

Each set is cached on first extraction and supports idempotent suspend/resume.
"""

import ctypes
import logging
import time

import torch

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(levelname)s:%(asctime)s:%(message)s"))
    logger.addHandler(_handler)

NCCL_SUSPEND_MEM = 0x01  # Release dynamic GPU memory allocations
NCCL_SUCCESS = 0

_nccl_lib = None

# ---------------------------------------------------------------------------
# NCCL library loading
# ---------------------------------------------------------------------------


def _get_nccl_lib():
    """Lazily load libnccl and define function signatures."""
    global _nccl_lib
    if _nccl_lib is not None:
        return _nccl_lib

    try:
        lib = ctypes.CDLL("libnccl.so.2")
    except OSError:
        logger.warning("[NCCLSuspend] Failed to load libnccl.so.2. Suspend/resume disabled.")
        return None

    if not hasattr(lib, "ncclCommSuspend"):
        logger.warning("[NCCLSuspend] ncclCommSuspend not found. Requires NCCL >= 2.29.7.")
        return None

    lib.ncclCommSuspend.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.ncclCommSuspend.restype = ctypes.c_int
    lib.ncclCommResume.argtypes = [ctypes.c_void_p]
    lib.ncclCommResume.restype = ctypes.c_int

    _nccl_lib = lib
    return lib


def _gpu_used_mb():
    """Driver-level GPU memory used (MB). Captures NCCL's cudaMalloc allocations."""
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    return (total - free) / 1024**2


def _normalize_comm_handle(comm):
    """Normalize comm handle to int for ctypes usage."""
    if isinstance(comm, int):
        return comm
    if hasattr(comm, "value"):  # ctypes c_void_p
        return comm.value
    return comm


# ---------------------------------------------------------------------------
# Low-level: suspend/resume a single comm
# ---------------------------------------------------------------------------


def suspend_nccl_comm(comm_handle) -> bool:
    """Suspend a single NCCL communicator to release its GPU memory."""
    lib = _get_nccl_lib()
    if lib is None or comm_handle is None:
        return False

    if isinstance(comm_handle, int):
        comm_handle = ctypes.c_void_p(comm_handle)

    result = lib.ncclCommSuspend(comm_handle, NCCL_SUSPEND_MEM)
    if result != NCCL_SUCCESS:
        logger.warning(f"[NCCLSuspend] ncclCommSuspend failed: handle={comm_handle}, error={result}")
        return False
    return True


def resume_nccl_comm(comm_handle) -> bool:
    """Resume a previously suspended NCCL communicator."""
    lib = _get_nccl_lib()
    if lib is None or comm_handle is None:
        return False

    if isinstance(comm_handle, int):
        comm_handle = ctypes.c_void_p(comm_handle)

    result = lib.ncclCommResume(comm_handle)
    if result != NCCL_SUCCESS:
        logger.warning(f"[NCCLSuspend] ncclCommResume failed: handle={comm_handle}, error={result}")
        return False
    return True


# ---------------------------------------------------------------------------
# Batch suspend/resume with logging and measurement
# ---------------------------------------------------------------------------


def _suspend_comms(handles: list[tuple[str, int]], label: str) -> tuple[bool, float]:
    """Suspend a list of (name, handle) comms. Returns (any_suspended, freed_mb)."""
    if not handles:
        logger.debug(f"[NCCLSuspend] {label}: no comms to suspend.")
        return False, 0.0

    mem_before = _gpu_used_mb()
    total_start = time.perf_counter()
    succeeded = []
    failed = []

    for name, handle in handles:
        t0 = time.perf_counter()
        ok = suspend_nccl_comm(handle)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        if ok:
            succeeded.append(name)
            logger.debug(f"[NCCLSuspend] {label}: suspend '{name}' (0x{handle:x}) OK ({elapsed_ms:.0f} ms)")
        else:
            failed.append(name)
            logger.warning(f"[NCCLSuspend] {label}: suspend '{name}' (0x{handle:x}) FAILED ({elapsed_ms:.0f} ms)")

    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    total_ms = (time.perf_counter() - total_start) * 1000
    mem_after = _gpu_used_mb()
    freed = mem_before - mem_after

    if succeeded:
        logger.info(
            f"[NCCLSuspend] {label}: suspended {len(succeeded)}/{len(handles)} comms "
            f"in {total_ms:.0f} ms, freed {freed:.0f} MB "
            f"(gpu: {mem_before:.0f} → {mem_after:.0f} MB)"
        )
    if failed:
        logger.warning(f"[NCCLSuspend] {label}: {len(failed)} comms failed: {failed}")

    return len(succeeded) > 0, freed


def _resume_comms(handles: list[tuple[str, int]], label: str) -> tuple[bool, float]:
    """Resume a list of (name, handle) comms. Returns (any_resumed, reclaimed_mb)."""
    if not handles:
        logger.debug(f"[NCCLSuspend] {label}: no comms to resume.")
        return False, 0.0

    mem_before = _gpu_used_mb()
    total_start = time.perf_counter()
    succeeded = []
    failed = []

    for name, handle in handles:
        t0 = time.perf_counter()
        ok = resume_nccl_comm(handle)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        if ok:
            succeeded.append(name)
            logger.debug(f"[NCCLSuspend] {label}: resume '{name}' (0x{handle:x}) OK ({elapsed_ms:.0f} ms)")
        else:
            failed.append(name)
            logger.warning(f"[NCCLSuspend] {label}: resume '{name}' (0x{handle:x}) FAILED ({elapsed_ms:.0f} ms)")

    torch.cuda.synchronize()
    total_ms = (time.perf_counter() - total_start) * 1000
    mem_after = _gpu_used_mb()
    reclaimed = mem_after - mem_before

    if succeeded:
        logger.info(
            f"[NCCLSuspend] {label}: resumed {len(succeeded)}/{len(handles)} comms "
            f"in {total_ms:.0f} ms, reclaimed {reclaimed:.0f} MB "
            f"(gpu: {mem_before:.0f} → {mem_after:.0f} MB)"
        )
    if failed:
        logger.warning(f"[NCCLSuspend] {label}: {len(failed)} comms failed: {failed}")

    return len(succeeded) > 0, reclaimed


# ===========================================================================
# Training side: torch.distributed ProcessGroup comms
# ===========================================================================

_training_comm_handles: list[tuple[str, int]] | None = None  # cached
_training_suspended = False


def _extract_training_comm_handles() -> list[tuple[str, int]]:
    """Extract ncclComm_t from all torch.distributed ProcessGroups in this process.

    Uses ProcessGroupNCCL._comm_ptr() which returns the ncclComm_t as int.
    Handles are cached after first successful extraction.

    Returns list of (group_name, comm_ptr_int).
    """
    global _training_comm_handles
    if _training_comm_handles is not None:
        return _training_comm_handles

    import torch.distributed as dist

    if not dist.is_initialized():
        logger.debug("[NCCLSuspend] Training: torch.distributed not initialized.")
        return []

    handles = []

    # Extract from default group
    try:
        default_pg = dist.distributed_c10d._get_default_group()
        backend = default_pg._get_backend(torch.device("cuda"))
        if hasattr(backend, "_comm_ptr"):
            ptr = backend._comm_ptr()
            if ptr != 0:
                handles.append(("default", ptr))
                logger.debug(f"[NCCLSuspend] Training: default group _comm_ptr() = 0x{ptr:x}")
            else:
                logger.debug("[NCCLSuspend] Training: default group _comm_ptr() = 0 (not initialized)")
    except Exception as e:
        logger.debug(f"[NCCLSuspend] Training: failed to extract default group: {e}")

    # Extract from all sub-groups via internal registry
    try:
        pg_map = dist.distributed_c10d._world.pg_map
        seen_ptrs = {h[1] for h in handles}  # avoid duplicates
        for pg, _ in pg_map.items():
            if pg == default_pg:
                continue
            try:
                backend = pg._get_backend(torch.device("cuda"))
                if hasattr(backend, "_comm_ptr"):
                    ptr = backend._comm_ptr()
                    if ptr != 0 and ptr not in seen_ptrs:
                        # Try to get a name for the group
                        pg_name = dist.distributed_c10d._world.pg_names.get(pg, f"pg_{len(handles)}")
                        handles.append((pg_name, ptr))
                        seen_ptrs.add(ptr)
                        logger.debug(f"[NCCLSuspend] Training: '{pg_name}' _comm_ptr() = 0x{ptr:x}")
            except Exception:
                pass
    except Exception as e:
        logger.debug(f"[NCCLSuspend] Training: failed to enumerate sub-groups: {e}")

    if handles:
        logger.info(f"[NCCLSuspend] Training: extracted {len(handles)} comm handles "
                     f"({[name for name, _ in handles]})")
        _training_comm_handles = handles
    else:
        logger.info("[NCCLSuspend] Training: no comm handles found (comms may not be initialized yet)")
        # Don't cache empty — retry next time (comms may get initialized later)

    return handles


def suspend_training_comms() -> bool:
    """Suspend all training-side NCCL comms (torch ProcessGroups) in this process.

    Idempotent: if already suspended, this is a no-op.
    Must be called from the training worker process (not the driver).

    Returns True if any comm was suspended.
    """
    global _training_suspended
    if _training_suspended:
        logger.debug("[NCCLSuspend] Training: already suspended, skipping.")
        return False

    handles = _extract_training_comm_handles()
    if not handles:
        return False

    ok, freed = _suspend_comms(handles, "Training")
    if ok:
        _training_suspended = True
    return ok


def resume_training_comms() -> bool:
    """Resume all training-side NCCL comms in this process.

    Idempotent: if not suspended, this is a no-op.

    Returns True if any comm was resumed.
    """
    global _training_suspended
    if not _training_suspended:
        logger.debug("[NCCLSuspend] Training: not suspended, skipping resume.")
        return False

    handles = _extract_training_comm_handles()
    if not handles:
        return False

    ok, reclaimed = _resume_comms(handles, "Training")
    if ok:
        _training_suspended = False
    return ok


# ===========================================================================
# Rollout side: vLLM pynccl comms
# ===========================================================================

_rollout_comm_handles: list[tuple[str, int]] | None = None  # cached
_rollout_suspended = False


def _extract_rollout_comm_handles() -> list[tuple[str, int]]:
    """Extract ncclComm_t from vLLM's pynccl communicators.

    Path: group.device_communicator.pynccl_comm.comm

    Returns list of (group_name, comm_ptr_int).
    """
    global _rollout_comm_handles
    if _rollout_comm_handles is not None:
        return _rollout_comm_handles

    try:
        from vllm.distributed import parallel_state as ps
    except ImportError:
        logger.debug("[NCCLSuspend] Rollout: vLLM not available.")
        return []

    handles = []
    group_accessors = [("tp", "get_tp_group"), ("pp", "get_pp_group")]

    for name, accessor_name in group_accessors:
        accessor = getattr(ps, accessor_name, None)
        if accessor is None:
            continue
        try:
            group = accessor()
            group_ws = getattr(group, "world_size", None)
            if group_ws is not None and group_ws <= 1:
                continue

            device_comm = getattr(group, "device_communicator", None)
            if device_comm is None:
                continue
            pynccl_comm = getattr(device_comm, "pynccl_comm", None)
            if pynccl_comm is None:
                continue
            comm = getattr(pynccl_comm, "comm", None)
            if comm is None:
                continue

            ptr = _normalize_comm_handle(comm)
            if ptr and ptr != 0:
                handles.append((name, ptr))
                logger.debug(f"[NCCLSuspend] Rollout: '{name}' pynccl_comm = 0x{ptr:x}")
        except Exception as e:
            logger.debug(f"[NCCLSuspend] Rollout: failed to get '{name}' comm: {e}")

    if handles:
        logger.info(f"[NCCLSuspend] Rollout: extracted {len(handles)} comm handles "
                     f"({[name for name, _ in handles]})")
        _rollout_comm_handles = handles
    else:
        logger.info("[NCCLSuspend] Rollout: no comm handles found")

    return handles


def _gloo_barrier():
    """CPU-based barrier (gloo) for synchronizing before/after suspend.

    NCCL barrier would deadlock after suspend, so we use gloo.
    """
    try:
        import torch.distributed as dist

        if not dist.is_initialized():
            return

        from vllm.distributed.parallel_state import get_tp_group

        tp_group = get_tp_group()
        cpu_group = getattr(tp_group, "cpu_group", None)
        if cpu_group is not None:
            dist.barrier(group=cpu_group)
        else:
            torch.cuda.synchronize()
    except Exception:
        torch.cuda.synchronize()


def suspend_rollout_comms() -> bool:
    """Suspend all rollout-side NCCL comms (vLLM pynccl) in this process.

    Idempotent: if already suspended, this is a no-op.
    Must be called from the vLLM rollout server process.

    Returns True if any comm was suspended.
    """
    global _rollout_suspended
    if _rollout_suspended:
        logger.debug("[NCCLSuspend] Rollout: already suspended, skipping.")
        return False

    handles = _extract_rollout_comm_handles()
    if not handles:
        return False

    _gloo_barrier()
    ok, freed = _suspend_comms(handles, "Rollout")
    _gloo_barrier()

    if ok:
        _rollout_suspended = True
    return ok


def resume_rollout_comms() -> bool:
    """Resume all rollout-side NCCL comms (vLLM pynccl) in this process.

    Idempotent: if not suspended, this is a no-op.

    Returns True if any comm was resumed.
    """
    global _rollout_suspended
    if not _rollout_suspended:
        logger.debug("[NCCLSuspend] Rollout: not suspended, skipping resume.")
        return False

    handles = _extract_rollout_comm_handles()
    if not handles:
        return False

    _gloo_barrier()
    ok, reclaimed = _resume_comms(handles, "Rollout")
    _gloo_barrier()

    if ok:
        _rollout_suspended = False
    return ok


# Backward-compatible aliases
suspend_vllm_comms = suspend_rollout_comms
resume_vllm_comms = resume_rollout_comms
