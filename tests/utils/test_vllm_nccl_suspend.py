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
Test vLLM NCCL communicator suspend/resume integration.

Verifies suspend/resume on TWO separate communication paths in vLLM:
  A. vLLM pynccl comm — the one actually used by vLLM inference (CUDA graph compatible)
  B. torch ProcessGroup comm — the underlying torch PG for the same TP group

Each path gets its OWN warmup and verification using its native API:
  - pynccl: warmup/verify via pynccl_comm.all_reduce()
  - torch PG: warmup/verify via dist.all_reduce(group=device_group)

This avoids false positives where suspending one comm but verifying
through the other would always pass regardless.

NOTE: This test intentionally does NOT issue collectives while suspended.
Doing so would likely hang or crash, which is the expected behavior of
ncclCommSuspend.

Usage:
    torchrun --nproc_per_node=2 tests/utils/test_vllm_nccl_suspend.py
"""

import gc
import os
import sys
import time

import torch
import torch.distributed as dist

RANK = None
LOCAL_RANK = None
WORLD_SIZE = None

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def log(msg=""):
    if RANK == 0:
        print(msg, flush=True)


def gpu_used_mb():
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    return (total - free) / 1024**2


def clean_and_measure():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return gpu_used_mb()


def gather_stats(x: float, group=None):
    t = torch.tensor([x], device="cuda", dtype=torch.float64)
    ws = dist.get_world_size(group=group)
    vals = [torch.zeros_like(t) for _ in range(ws)]
    dist.all_gather(vals, t, group=group)
    vals = torch.stack(vals).flatten()
    return vals.min().item(), vals.mean().item(), vals.max().item()


def fmt_stats(mn, avg, mx):
    if abs(mx - mn) < 1.0:
        return f"{avg:.0f}"
    return f"{mn:.0f}/{avg:.0f}/{mx:.0f}"


def normalize_comm_handle(comm):
    """Normalize comm handle to int (pointer value) for ctypes suspend/resume."""
    if isinstance(comm, int):
        return comm
    if hasattr(comm, "value"):  # ctypes c_void_p
        return comm.value
    if hasattr(comm, "ptr"):
        return comm.ptr
    log(f"  WARNING: unknown comm type {type(comm)}, using as-is")
    return comm


# ---------------------------------------------------------------------------
# vLLM comm handle extraction
# ---------------------------------------------------------------------------

def extract_vllm_pynccl_comm(group, name=""):
    """Extract ncclComm_t AND pynccl_comm object from vLLM GroupCoordinator.

    Returns (raw_handle, pynccl_comm_object, description).
    raw_handle: int pointer for suspend/resume
    pynccl_comm_object: PyNcclCommunicator for calling all_reduce directly
    """
    group_ws = getattr(group, "world_size", None)
    if group_ws is not None and group_ws <= 1:
        return None, None, f"world_size={group_ws} (no NCCL comm for TP=1)"

    device_comm = getattr(group, "device_communicator", None)
    if device_comm is None:
        return None, None, "no device_communicator attribute"

    pynccl_comm = getattr(device_comm, "pynccl_comm", None)
    if pynccl_comm is None:
        return None, None, "no pynccl_comm attribute"

    comm = getattr(pynccl_comm, "comm", None)
    if comm is None:
        return None, None, "no comm attribute on pynccl_comm"

    raw = normalize_comm_handle(comm)
    return raw, pynccl_comm, f"pynccl_comm.comm = {raw} (type={type(comm).__name__})"


def extract_torch_pg_comm(group, name=""):
    """Extract ncclComm_t from vLLM's underlying torch ProcessGroup via _comm_ptr()."""
    device_group = getattr(group, "device_group", None)
    if device_group is None:
        return None, None, "no device_group attribute"

    try:
        backend = device_group._get_backend(torch.device("cuda"))
        if hasattr(backend, "_comm_ptr"):
            ptr = backend._comm_ptr()
            return ptr, device_group, f"_comm_ptr() = {hex(ptr)}"
        return None, None, "_comm_ptr not available"
    except Exception as e:
        return None, None, f"failed: {e}"


# ---------------------------------------------------------------------------
# Warmup / verify helpers for each comm path
# ---------------------------------------------------------------------------

def warmup_pynccl(pynccl_comm_obj, rounds=10, size=4096):
    """Warmup vLLM pynccl comm using its native all_reduce."""
    for _ in range(rounds):
        buf = torch.randn(size, size, device="cuda")
        pynccl_comm_obj.all_reduce(buf)
        del buf
    torch.cuda.synchronize()


def verify_pynccl(pynccl_comm_obj, tp_ranks):
    """Verify pynccl comm works by doing all_reduce and checking result."""
    x = torch.ones(1024, 1024, device="cuda") * (RANK + 1)
    pynccl_comm_obj.all_reduce(x)
    torch.cuda.synchronize()
    expected = sum(r + 1 for r in tp_ranks)
    assert torch.allclose(x, torch.full_like(x, float(expected))), \
        f"pynccl allreduce wrong: got {x[0,0].item()}, expected {expected}"
    del x
    return expected


def warmup_torch_pg(device_group, rounds=10, size=4096):
    """Warmup torch PG comm using dist.all_reduce."""
    for _ in range(rounds):
        buf = torch.randn(size, size, device="cuda")
        dist.all_reduce(buf, group=device_group)
        del buf
    torch.cuda.synchronize()


def verify_torch_pg(device_group, tp_ranks):
    """Verify torch PG comm works by doing all_reduce and checking result."""
    x = torch.ones(1024, 1024, device="cuda") * (RANK + 1)
    dist.all_reduce(x, group=device_group)
    expected = sum(r + 1 for r in tp_ranks)
    assert torch.allclose(x, torch.full_like(x, float(expected))), \
        f"torch PG allreduce wrong: got {x[0,0].item()}, expected {expected}"
    del x
    return expected


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_extract_handles(tp_group):
    """Test 1: Verify we can extract ncclComm_t from both paths."""
    log("\n" + "=" * 70)
    log("Test 1: Extract ncclComm_t handles from vLLM")
    log("=" * 70)

    pynccl_handle, pynccl_obj, pynccl_msg = extract_vllm_pynccl_comm(tp_group)
    log(f"\n  pynccl path: {pynccl_msg}")

    torch_handle, torch_pg, torch_msg = extract_torch_pg_comm(tp_group)
    log(f"  torch PG path: {torch_msg}")

    if pynccl_handle is not None and torch_handle is not None:
        same = (pynccl_handle == torch_handle)
        log(f"\n  Same ncclComm_t: {same}")
        if same:
            log("  → vLLM pynccl reuses torch PG's ncclComm_t. Suspending either is equivalent.")
        else:
            log("  → DIFFERENT comms! In colocated mode, BOTH need to be suspended.")
    elif pynccl_handle is None and torch_handle is None:
        log("\n  FAIL: Cannot extract ncclComm_t from either path.")
        # Debug info
        log(f"  tp_group type = {type(tp_group)}")
        attrs = [a for a in dir(tp_group) if not a.startswith('_')]
        log(f"  tp_group attrs = {attrs[:20]}")
        dc = getattr(tp_group, "device_communicator", None)
        if dc:
            log(f"  device_communicator type = {type(dc)}")
            dc_attrs = [a for a in dir(dc) if not a.startswith('_')]
            log(f"  device_communicator attrs = {dc_attrs[:20]}")
        return None, None, None, None

    log("  PASS")
    return pynccl_handle, pynccl_obj, torch_handle, torch_pg


def test_pynccl_suspend_resume(pynccl_handle, pynccl_obj, tp_ranks, suspend_fn, resume_fn):
    """Test 2A: Suspend/resume vLLM pynccl comm with pynccl-native warmup/verify."""
    log("\n" + "=" * 70)
    log("Test 2A: pynccl comm — suspend/resume (pynccl-native warmup/verify)")
    log("=" * 70)

    # Warmup via pynccl all_reduce
    log("\n  Warming up pynccl comm (pynccl_comm.all_reduce)...")
    warmup_pynccl(pynccl_obj)
    baseline = clean_and_measure()
    log(f"  Post-warmup gpu_used: {baseline:.0f} MB")

    # Suspend
    dist.barrier()
    torch.cuda.synchronize()
    pre = gpu_used_mb()
    dist.barrier()

    t0 = time.perf_counter()
    suspend_fn(pynccl_handle)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    t2 = time.perf_counter()

    post = gpu_used_mb()
    freed = pre - post
    freed_stats = gather_stats(freed)

    log(f"\n  Before: {pre:.0f} MB → After: {post:.0f} MB")
    log(f"  Freed (min/avg/max): {fmt_stats(*freed_stats)} MB")
    log(f"  Suspend API: {(t1 - t0) * 1000:.0f} ms, reclaim: {(t2 - t0) * 1000:.0f} ms")

    # Resume
    t3 = time.perf_counter()
    resume_fn(pynccl_handle)
    torch.cuda.synchronize()
    t4 = time.perf_counter()
    log(f"  Resume: {(t4 - t3) * 1000:.0f} ms")

    # Verify via pynccl all_reduce (NOT torch PG!)
    log("  Verifying via pynccl_comm.all_reduce...")
    expected = verify_pynccl(pynccl_obj, tp_ranks)
    log(f"  pynccl allreduce OK (result={expected})")
    log("  PASS")
    return freed_stats[1]


def test_torch_pg_suspend_resume(torch_handle, torch_pg, tp_ranks, suspend_fn, resume_fn):
    """Test 2B: Suspend/resume torch PG comm with torch-native warmup/verify."""
    log("\n" + "=" * 70)
    log("Test 2B: torch PG comm — suspend/resume (torch-native warmup/verify)")
    log("=" * 70)

    # Warmup via dist.all_reduce
    log("\n  Warming up torch PG comm (dist.all_reduce)...")
    warmup_torch_pg(torch_pg)
    baseline = clean_and_measure()
    log(f"  Post-warmup gpu_used: {baseline:.0f} MB")

    # Suspend
    dist.barrier()
    torch.cuda.synchronize()
    pre = gpu_used_mb()
    dist.barrier()

    t0 = time.perf_counter()
    suspend_fn(torch_handle)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    t2 = time.perf_counter()

    post = gpu_used_mb()
    freed = pre - post

    log(f"\n  Before: {pre:.0f} MB → After: {post:.0f} MB, freed: {freed:.0f} MB")
    log(f"  Suspend API: {(t1 - t0) * 1000:.0f} ms")

    # Resume
    resume_fn(torch_handle)
    torch.cuda.synchronize()

    # Verify via dist.all_reduce (NOT pynccl!)
    log("  Verifying via dist.all_reduce...")
    expected = verify_torch_pg(torch_pg, tp_ranks)
    log(f"  torch PG allreduce OK (result={expected})")
    log("  PASS")
    return freed


def test_multicycle(handle, warmup_fn, verify_fn, label, suspend_fn, resume_fn, tp_ranks):
    """Test 3: Multi-cycle stability using the correct native API."""
    log(f"\n" + "=" * 70)
    log(f"Test 3: Multi-cycle stability — {label} (5 cycles)")
    log("=" * 70)

    for cycle in range(5):
        dist.barrier()
        torch.cuda.synchronize()
        pre = gpu_used_mb()
        dist.barrier()

        suspend_fn(handle)
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        freed = pre - gpu_used_mb()

        resume_fn(handle)
        torch.cuda.synchronize()

        verify_fn(tp_ranks)
        log(f"  Cycle {cycle + 1}: freed={freed:.0f} MB, verify OK")

    log("  PASS")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global RANK, LOCAL_RANK, WORLD_SIZE

    # Let vLLM handle torch.distributed init if possible
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
    RANK = int(os.environ.get("RANK", "0"))
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(LOCAL_RANK)

    log(f"=== vLLM NCCL Suspend/Resume Test (world_size={WORLD_SIZE}) ===")

    if WORLD_SIZE < 2:
        log("SKIP: Need 2+ GPUs (TP=1 has no NCCL comm)")
        return

    # Check vLLM
    try:
        import vllm
        log(f"vLLM version: {vllm.__version__}")
    except ImportError:
        log("SKIP: vLLM not installed")
        return

    # Init distributed + vLLM parallel state
    # vLLM >= 0.18 requires set_current_vllm_config() context for initialize_model_parallel()
    log("Initializing vLLM parallel state...")
    try:
        from vllm.config import VllmConfig, set_current_vllm_config
        from vllm.distributed.parallel_state import (
            get_tp_group,
            init_distributed_environment,
            initialize_model_parallel,
        )

        vllm_config = VllmConfig()
        _vllm_config_ctx = set_current_vllm_config(vllm_config)
        _vllm_config_ctx.__enter__()

        init_distributed_environment(
            world_size=WORLD_SIZE, rank=RANK, local_rank=LOCAL_RANK,
        )
        initialize_model_parallel(tensor_model_parallel_size=WORLD_SIZE)

        # Re-read rank from torch.distributed (vLLM may have initialized it)
        RANK = dist.get_rank()
        WORLD_SIZE = dist.get_world_size()

        tp_group = get_tp_group()
        log(f"  TP group initialized (world_size={tp_group.world_size})")
    except Exception as e:
        log(f"SKIP: vLLM parallel init failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Get TP group ranks for expected value calculation
    tp_device_group = getattr(tp_group, "device_group", None)
    if tp_device_group is not None:
        tp_ranks = dist.get_process_group_ranks(tp_device_group)
    else:
        tp_ranks = list(range(WORLD_SIZE))
    log(f"  TP ranks: {tp_ranks}")

    # Load suspend/resume
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from verl.utils.nccl_suspend import _get_nccl_lib, resume_nccl_comm, suspend_nccl_comm

    if _get_nccl_lib() is None:
        log("SKIP: NCCL >= 2.29.7 required")
        dist.destroy_process_group()
        return

    # Test 1: Extract handles
    pynccl_handle, pynccl_obj, torch_handle, torch_pg = test_extract_handles(tp_group)
    if pynccl_handle is None and torch_handle is None:
        dist.destroy_process_group()
        return

    # Test 2A: pynccl path (if available)
    pynccl_freed = None
    if pynccl_handle is not None and pynccl_obj is not None:
        pynccl_freed = test_pynccl_suspend_resume(
            pynccl_handle, pynccl_obj, tp_ranks, suspend_nccl_comm, resume_nccl_comm)
    else:
        log("\n  SKIP Test 2A: pynccl comm not available")

    # Test 2B: torch PG path (if available)
    torch_freed = None
    if torch_handle is not None and torch_pg is not None:
        torch_freed = test_torch_pg_suspend_resume(
            torch_handle, torch_pg, tp_ranks, suspend_nccl_comm, resume_nccl_comm)
    else:
        log("\n  SKIP Test 2B: torch PG comm not available")

    # Test 3: Multi-cycle on whichever path is available
    if pynccl_handle is not None and pynccl_obj is not None:
        test_multicycle(
            pynccl_handle,
            lambda: warmup_pynccl(pynccl_obj, rounds=2, size=2048),
            lambda tp_r: verify_pynccl(pynccl_obj, tp_r),
            "pynccl", suspend_nccl_comm, resume_nccl_comm, tp_ranks)
    elif torch_handle is not None and torch_pg is not None:
        test_multicycle(
            torch_handle,
            lambda: warmup_torch_pg(torch_pg, rounds=2, size=2048),
            lambda tp_r: verify_torch_pg(torch_pg, tp_r),
            "torch_pg", suspend_nccl_comm, resume_nccl_comm, tp_ranks)

    # Summary
    log(f"\n{'=' * 70}")
    log("ALL vLLM TESTS PASSED")
    if pynccl_freed is not None:
        log(f"  pynccl comm freed: {pynccl_freed:.0f} MB")
    if torch_freed is not None:
        log(f"  torch PG comm freed: {torch_freed:.0f} MB")
    if pynccl_handle is not None and torch_handle is not None:
        same = (pynccl_handle == torch_handle)
        log(f"  Same ncclComm_t: {same}")
    log(f"{'=' * 70}")

    # Cleanup
    try:
        from vllm.distributed.parallel_state import destroy_model_parallel
        destroy_model_parallel()
    except Exception:
        pass
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
