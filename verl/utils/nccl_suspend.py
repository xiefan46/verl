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
        print("[NCCLSuspend] WARNING: Failed to load libnccl.so.2. Suspend/resume disabled.", flush=True)
        return None

    if not hasattr(lib, "ncclCommSuspend"):
        print("[NCCLSuspend] WARNING: ncclCommSuspend not found. Requires NCCL >= 2.29.7.", flush=True)
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
    """Suspend a list of (name, handle) comms. Returns (any_suspended, freed_mb).

    Per-comm GPU memory delta is measured by calling empty_cache + synchronize
    between each suspend call. This adds ~1-10 ms overhead per comm but lets us
    verify each comm has its own independent NCCL channel buffer.
    """
    if not handles:
        print(f"[NCCLSuspend] {label}: no comms to suspend.", flush=True)
        return False, 0.0

    mem_total_before = _gpu_used_mb()
    total_start = time.perf_counter()
    succeeded = []
    failed = []

    for name, handle in handles:
        mem_before = _gpu_used_mb()
        t0 = time.perf_counter()
        ok = suspend_nccl_comm(handle)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        # Force per-comm reclaim so the delta isolates this comm's buffer.
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        mem_after = _gpu_used_mb()
        per_comm_freed = mem_before - mem_after

        if ok:
            succeeded.append(name)
            print(
                f"[NCCLSuspend] {label}: suspend '{name}' (0x{handle:x}) OK ({elapsed_ms:.0f} ms), "
                f"freed {per_comm_freed:.0f} MB ({mem_before:.0f} → {mem_after:.0f} MB)",
                flush=True,
            )
        else:
            failed.append(name)
            print(
                f"[NCCLSuspend] {label}: suspend '{name}' (0x{handle:x}) FAILED ({elapsed_ms:.0f} ms)",
                flush=True,
            )

    total_ms = (time.perf_counter() - total_start) * 1000
    mem_total_after = _gpu_used_mb()
    freed = mem_total_before - mem_total_after

    print(
        f"[NCCLSuspend] {label}: suspended {len(succeeded)}/{len(handles)} comms "
        f"in {total_ms:.0f} ms, freed {freed:.0f} MB total "
        f"(gpu: {mem_total_before:.0f} → {mem_total_after:.0f} MB)",
        flush=True,
    )
    if failed:
        print(f"[NCCLSuspend] {label}: {len(failed)} comms failed: {failed}", flush=True)

    return len(succeeded) > 0, freed


def _resume_comms(handles: list[tuple[str, int]], label: str) -> tuple[bool, float]:
    """Resume a list of (name, handle) comms. Returns (any_resumed, reclaimed_mb).

    Per-comm GPU memory delta is measured by calling synchronize between each
    resume call. NCCL allocates channel buffers on the next collective, but
    ncclCommResume itself reclaims most of the buffer state.
    """
    if not handles:
        print(f"[NCCLSuspend] {label}: no comms to resume.", flush=True)
        return False, 0.0

    mem_total_before = _gpu_used_mb()
    total_start = time.perf_counter()
    succeeded = []
    failed = []

    for name, handle in handles:
        mem_before = _gpu_used_mb()
        t0 = time.perf_counter()
        ok = resume_nccl_comm(handle)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        torch.cuda.synchronize()
        mem_after = _gpu_used_mb()
        per_comm_reclaimed = mem_after - mem_before

        if ok:
            succeeded.append(name)
            print(
                f"[NCCLSuspend] {label}: resume '{name}' (0x{handle:x}) OK ({elapsed_ms:.0f} ms), "
                f"reclaimed {per_comm_reclaimed:.0f} MB ({mem_before:.0f} → {mem_after:.0f} MB)",
                flush=True,
            )
        else:
            failed.append(name)
            print(
                f"[NCCLSuspend] {label}: resume '{name}' (0x{handle:x}) FAILED ({elapsed_ms:.0f} ms)",
                flush=True,
            )

    total_ms = (time.perf_counter() - total_start) * 1000
    mem_total_after = _gpu_used_mb()
    reclaimed = mem_total_after - mem_total_before

    print(
        f"[NCCLSuspend] {label}: resumed {len(succeeded)}/{len(handles)} comms "
        f"in {total_ms:.0f} ms, reclaimed {reclaimed:.0f} MB total "
        f"(gpu: {mem_total_before:.0f} → {mem_total_after:.0f} MB)",
        flush=True,
    )
    if failed:
        print(f"[NCCLSuspend] {label}: {len(failed)} comms failed: {failed}", flush=True)

    return len(succeeded) > 0, reclaimed


# ===========================================================================
# Training side: torch.distributed ProcessGroup comms
# ===========================================================================

_training_suspended_handles: list[tuple[str, int]] = []  # handles currently suspended
_training_suspended = False


def _scan_warm_training_comms() -> list[tuple[str, int]]:
    """Scan pg_map and return all PGs whose NCCL comm has been warmed up.

    Re-scans every call — Megatron creates many PGs (TP/DP/PP/embedding/PP P2P
    pairs/dist-optimizer/etc.) but `_comm_ptr()` is lazy and returns 0 until the
    first collective runs on that PG. Late-warming PGs would be missed by a
    one-shot cache, so we rescan before each suspend.
    """
    import torch.distributed as dist

    if not dist.is_initialized():
        return []

    handles: list[tuple[str, int]] = []
    seen_ptrs: set[int] = set()
    default_pg = None

    try:
        default_pg = dist.distributed_c10d._get_default_group()
        backend = default_pg._get_backend(torch.device("cuda"))
        if hasattr(backend, "_comm_ptr"):
            ptr = backend._comm_ptr()
            if ptr != 0:
                handles.append(("default", ptr))
                seen_ptrs.add(ptr)
    except Exception as e:
        logger.debug(f"[NCCLSuspend] Training: failed to extract default group: {e}")

    pg_total = 0
    try:
        pg_map = dist.distributed_c10d._world.pg_map
        pg_total = len(pg_map)
        for pg, _ in pg_map.items():
            if pg is default_pg:
                continue
            try:
                backend = pg._get_backend(torch.device("cuda"))
                if not hasattr(backend, "_comm_ptr"):
                    continue
                ptr = backend._comm_ptr()
                if ptr == 0 or ptr in seen_ptrs:
                    continue
                pg_name = dist.distributed_c10d._world.pg_names.get(pg, f"pg_{len(handles)}")
                handles.append((pg_name, ptr))
                seen_ptrs.add(ptr)
            except Exception:
                pass
    except Exception as e:
        logger.debug(f"[NCCLSuspend] Training: failed to enumerate sub-groups: {e}")

    print(
        f"[NCCLSuspend] Training: pg_map size={pg_total}, warm comms={len(handles)} ({[name for name, _ in handles]})",
        flush=True,
    )
    return handles


def suspend_training_comms() -> bool:
    """Suspend all currently-warm training-side NCCL comms in this process.

    Re-scans pg_map each call so newly-warmed PGs get included.
    Idempotent: if already suspended, this is a no-op.

    Returns True if any comm was suspended.
    """
    global _training_suspended, _training_suspended_handles
    if _training_suspended:
        print("[NCCLSuspend] Training: already suspended, skipping.", flush=True)
        return False

    handles = _scan_warm_training_comms()
    if not handles:
        print("[NCCLSuspend] Training suspend: no warm comms found", flush=True)
        return False

    ok, freed = _suspend_comms(handles, "Training")
    if ok:
        _training_suspended = True
        _training_suspended_handles = handles
    return ok


def resume_training_comms() -> bool:
    """Resume the comms that were suspended by the most recent suspend call.

    Idempotent: if not suspended, this is a no-op.
    """
    global _training_suspended
    if not _training_suspended:
        print("[NCCLSuspend] Training: not suspended, skipping resume.", flush=True)
        return False

    handles = _training_suspended_handles
    if not handles:
        print("[NCCLSuspend] Training resume: no handles available", flush=True)
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
    """Extract ncclComm_t handles from all NCCL comms in the vLLM worker process.

    vLLM workers hold MULTIPLE independent NCCL comms:
      1. pynccl (vLLM's own ctypes-based NCCL): group.device_communicator.pynccl_comm.comm
      2. torch.distributed ProcessGroupNCCL: dist.group.WORLD._get_backend()._comm_ptr()
      3. Any sub-groups created via dist.new_group() for TP/PP/etc.

    pynccl warm-up only does a 1-element all_reduce (small channel buffer).
    The torch PG default comm is what carries actual barrier/broadcast traffic
    and typically has the larger NCCL channel buffer.

    Returns list of (group_name, comm_ptr_int).
    """
    global _rollout_comm_handles
    if _rollout_comm_handles is not None:
        return _rollout_comm_handles

    handles = []

    # --- 1) vLLM pynccl comms ---
    try:
        from vllm.distributed import parallel_state as ps

        group_accessors = [("vllm_tp_pynccl", "get_tp_group"), ("vllm_pp_pynccl", "get_pp_group")]
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
            except Exception as e:
                print(f"[NCCLSuspend] Rollout: failed to get '{name}': {e}", flush=True)
    except ImportError:
        pass

    # --- 2) torch.distributed default group NCCL comm ---
    try:
        import torch.distributed as dist

        if dist.is_initialized():
            default_pg = dist.group.WORLD
            try:
                backend = default_pg._get_backend(torch.device("cuda"))
                ptr = backend._comm_ptr()
                if ptr and ptr != 0:
                    handles.append(("torch_pg_default", ptr))
            except Exception as e:
                print(f"[NCCLSuspend] Rollout: torch PG default _comm_ptr() failed: {e}", flush=True)

            # --- 3) named sub-groups (vllm typically registers TP/PP via init_model_parallel_group) ---
            try:
                for pg in dist.distributed_c10d._world.pg_map.keys():
                    if pg is default_pg:
                        continue
                    try:
                        backend = pg._get_backend(torch.device("cuda"))
                        ptr = backend._comm_ptr()
                        if ptr and ptr != 0 and not any(p == ptr for _, p in handles):
                            pg_name = getattr(pg, "group_name", None) or "torch_pg_sub"
                            handles.append((f"torch_pg:{pg_name}", ptr))
                    except Exception:
                        pass
            except Exception:
                pass
    except Exception as e:
        print(f"[NCCLSuspend] Rollout: torch.distributed introspection failed: {e}", flush=True)

    if handles:
        print(
            f"[NCCLSuspend] Rollout: extracted {len(handles)} comm handles ({[name for name, _ in handles]})",
            flush=True,
        )
        _rollout_comm_handles = handles
    else:
        print("[NCCLSuspend] Rollout: no comm handles found", flush=True)

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
        print("[NCCLSuspend] Rollout: already suspended, skipping.", flush=True)
        return False

    handles = _extract_rollout_comm_handles()
    if not handles:
        print("[NCCLSuspend] Rollout suspend: no handles available", flush=True)
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
        _extract_rollout_comm_handles()  # cache handles for next suspend
        print("[NCCLSuspend] Rollout: not suspended, skipping resume.", flush=True)
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
