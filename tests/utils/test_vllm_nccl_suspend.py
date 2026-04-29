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

Verifies that suspend/resume works on vLLM's internal pynccl communicators
(TP group, PP group) — the ones used in colocated RL training's rollout side.

Tests:
  1. Extract ncclComm_t from vLLM's pynccl wrapper
  2. Suspend → measure freed GPU memory (driver-level)
  3. Resume → verify allreduce still works
  4. Multi-cycle stability
  5. Compare: vLLM pynccl comm vs torch ProcessGroup comm on same group

Usage:
    torchrun --nproc_per_node=2 tests/utils/test_vllm_nccl_suspend.py

Requires:
    - 2+ GPUs
    - vLLM installed
    - NCCL >= 2.29.7
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
# Utilities (same conventions as profile_nccl_memory.py)
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


def gather_stats(x: float):
    t = torch.tensor([x], device="cuda", dtype=torch.float64)
    vals = [torch.zeros_like(t) for _ in range(WORLD_SIZE)]
    dist.all_gather(vals, t)
    vals = torch.stack(vals).flatten()
    return vals.min().item(), vals.mean().item(), vals.max().item()


def fmt_stats(mn, avg, mx):
    if abs(mx - mn) < 1.0:
        return f"{avg:.0f}"
    return f"{mn:.0f}/{avg:.0f}/{mx:.0f}"


# ---------------------------------------------------------------------------
# vLLM comm handle extraction
# ---------------------------------------------------------------------------

def extract_vllm_pynccl_comm(group, name=""):
    """Extract ncclComm_t from vLLM GroupCoordinator → device_communicator → pynccl_comm → comm.

    Returns (comm_handle, path_description) or (None, error_msg).
    """
    if group is None:
        return None, "group is None"
    if group.world_size <= 1:
        return None, f"world_size={group.world_size} (no NCCL comm for TP=1)"

    # Path: group.device_communicator.pynccl_comm.comm
    device_comm = getattr(group, "device_communicator", None)
    if device_comm is None:
        return None, "no device_communicator attribute"

    pynccl_comm = getattr(device_comm, "pynccl_comm", None)
    if pynccl_comm is None:
        return None, "no pynccl_comm attribute"

    comm = getattr(pynccl_comm, "comm", None)
    if comm is None:
        return None, "no comm attribute on pynccl_comm"

    return comm, f"group.device_communicator.pynccl_comm.comm = {comm}"


def extract_torch_pg_comm(group, name=""):
    """Extract ncclComm_t from vLLM's underlying torch ProcessGroup via _comm_ptr()."""
    device_group = getattr(group, "device_group", None)
    if device_group is None:
        return None, "no device_group attribute"

    try:
        backend = device_group._get_backend(torch.device("cuda"))
        if hasattr(backend, "_comm_ptr"):
            ptr = backend._comm_ptr()
            return ptr, f"device_group._get_backend()._comm_ptr() = {hex(ptr)}"
        return None, "_comm_ptr not available"
    except Exception as e:
        return None, f"failed: {e}"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_extract_handles():
    """Test 1: Verify we can extract ncclComm_t from vLLM's parallel state."""
    log("\n" + "=" * 70)
    log("Test 1: Extract ncclComm_t handles from vLLM")
    log("=" * 70)

    from vllm.distributed.parallel_state import get_tp_group

    tp_group = get_tp_group()
    log(f"\n  TP group: world_size={tp_group.world_size}")

    # Try pynccl path
    pynccl_comm, pynccl_msg = extract_vllm_pynccl_comm(tp_group, "tp")
    log(f"  pynccl path: {pynccl_msg}")

    # Try torch PG path
    torch_comm, torch_msg = extract_torch_pg_comm(tp_group, "tp")
    log(f"  torch PG path: {torch_msg}")

    # Also check PP group if it exists
    try:
        from vllm.distributed.parallel_state import get_pp_group
        pp_group = get_pp_group()
        if pp_group and pp_group.world_size > 1:
            pp_comm, pp_msg = extract_vllm_pynccl_comm(pp_group, "pp")
            log(f"  PP pynccl path: {pp_msg}")
        else:
            log(f"  PP group: world_size={pp_group.world_size if pp_group else 'N/A'} (no comm)")
    except Exception as e:
        log(f"  PP group: {e}")

    # Determine which handle to use
    comm = pynccl_comm or torch_comm
    comm_source = "pynccl" if pynccl_comm else "torch_pg" if torch_comm else None

    if comm is None:
        log("\n  FAIL: Cannot extract ncclComm_t from vLLM.")
        log("  Check vLLM version and pynccl wrapper structure.")
        # Log the full attribute chain for debugging
        log(f"\n  Debug: tp_group type = {type(tp_group)}")
        log(f"  Debug: tp_group attrs = {[a for a in dir(tp_group) if not a.startswith('_')][:20]}")
        dc = getattr(tp_group, "device_communicator", None)
        if dc:
            log(f"  Debug: device_communicator type = {type(dc)}")
            log(f"  Debug: device_communicator attrs = {[a for a in dir(dc) if not a.startswith('_')][:20]}")
        return None, None

    log(f"\n  Using {comm_source} handle: {comm}")
    log("  PASS")
    return comm, comm_source


def test_suspend_resume(comm, comm_source, suspend_fn, resume_fn):
    """Test 2: Suspend vLLM NCCL comm, measure freed memory, resume, verify."""
    log("\n" + "=" * 70)
    log("Test 2: Suspend/resume vLLM NCCL comm + memory measurement")
    log("=" * 70)

    from vllm.distributed.parallel_state import get_tp_group
    tp_group = get_tp_group()

    # Warmup: allreduce on the TP group to inflate NCCL buffers
    log("\n  Warming up vLLM TP group (allreduce)...")
    for _ in range(10):
        buf = torch.randn(4096, 4096, device="cuda")
        dist.all_reduce(buf, group=tp_group.device_group)
        del buf

    baseline = clean_and_measure()
    log(f"  Post-warmup gpu_used: {baseline:.0f} MB")

    # Suspend
    dist.barrier()
    torch.cuda.synchronize()
    pre = gpu_used_mb()
    dist.barrier()

    t0 = time.perf_counter()
    suspend_fn(comm)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    t2 = time.perf_counter()

    post = gpu_used_mb()
    freed = pre - post
    freed_stats = gather_stats(freed)

    log(f"\n  Before suspend: {pre:.0f} MB")
    log(f"  After suspend:  {post:.0f} MB")
    log(f"  Freed (min/avg/max): {fmt_stats(*freed_stats)} MB")
    log(f"  Suspend API: {(t1 - t0) * 1000:.0f} ms, reclaim total: {(t2 - t0) * 1000:.0f} ms")

    # Resume
    t3 = time.perf_counter()
    resume_fn(comm)
    torch.cuda.synchronize()
    t4 = time.perf_counter()
    log(f"  Resume: {(t4 - t3) * 1000:.0f} ms")

    # Verify allreduce works
    log("\n  Verifying allreduce after resume...")
    x = torch.ones(1024, 1024, device="cuda") * (RANK + 1)
    dist.all_reduce(x, group=tp_group.device_group)
    expected = sum(range(1, WORLD_SIZE + 1))
    assert torch.allclose(x, torch.full_like(x, expected)), "allreduce failed after resume!"
    del x
    log(f"  allreduce OK (result={expected})")
    log("  PASS")

    return freed_stats[1]  # avg freed


def test_multicycle(comm, suspend_fn, resume_fn):
    """Test 3: Multiple suspend/resume cycles with vLLM comm."""
    log("\n" + "=" * 70)
    log("Test 3: Multi-cycle stability (5 cycles)")
    log("=" * 70)

    from vllm.distributed.parallel_state import get_tp_group
    tp_group = get_tp_group()
    expected = sum(range(1, WORLD_SIZE + 1))

    for cycle in range(5):
        # Suspend
        dist.barrier()
        torch.cuda.synchronize()
        pre = gpu_used_mb()
        dist.barrier()
        suspend_fn(comm)
        gc.collect()
        torch.cuda.empty_cache()
        freed = pre - gpu_used_mb()

        # Resume
        resume_fn(comm)
        torch.cuda.synchronize()

        # Verify with increasing tensor sizes
        size = 1024 * (cycle + 1)
        x = torch.ones(size, size, device="cuda") * (RANK + 1)
        dist.all_reduce(x, group=tp_group.device_group)
        assert torch.allclose(x, torch.full_like(x, expected)), f"cycle {cycle + 1} failed"
        del x
        log(f"  Cycle {cycle + 1}: freed={freed:.0f} MB, allreduce({size}x{size}) OK")

    log("  PASS")


def test_compare_pynccl_vs_torch(suspend_fn, resume_fn):
    """Test 4: Compare suspend on vLLM pynccl comm vs torch ProcessGroup comm.

    vLLM maintains both a pynccl comm and a torch ProcessGroup for the same
    TP group. This test checks if suspending one vs the other gives different
    results, and whether they point to the same underlying ncclComm_t.
    """
    log("\n" + "=" * 70)
    log("Test 4: Compare pynccl vs torch PG comm handles")
    log("=" * 70)

    from vllm.distributed.parallel_state import get_tp_group
    tp_group = get_tp_group()

    pynccl_comm, _ = extract_vllm_pynccl_comm(tp_group)
    torch_comm, _ = extract_torch_pg_comm(tp_group)

    if pynccl_comm is None or torch_comm is None:
        log("  SKIP: need both pynccl and torch PG handles")
        return

    same = (pynccl_comm == torch_comm) if isinstance(pynccl_comm, int) and isinstance(torch_comm, int) else "N/A"
    log(f"\n  pynccl comm: {pynccl_comm}")
    log(f"  torch PG comm: {hex(torch_comm) if isinstance(torch_comm, int) else torch_comm}")
    log(f"  Same handle: {same}")

    if same:
        log("  They share the same ncclComm_t — suspending either one is equivalent.")
    else:
        log("  Different ncclComm_t — vLLM uses a separate pynccl comm from torch PG.")
        log("  In colocated mode, BOTH may need to be suspended.")

        # Measure each independently
        for name, comm in [("pynccl", pynccl_comm), ("torch_pg", torch_comm)]:
            # Warmup
            for _ in range(5):
                buf = torch.randn(2048, 2048, device="cuda")
                dist.all_reduce(buf, group=tp_group.device_group)
                del buf
            clean_and_measure()

            dist.barrier()
            torch.cuda.synchronize()
            pre = gpu_used_mb()
            dist.barrier()
            suspend_fn(comm)
            gc.collect()
            torch.cuda.empty_cache()
            freed = pre - gpu_used_mb()
            resume_fn(comm)
            torch.cuda.synchronize()

            log(f"  {name}: freed={freed:.0f} MB")

    log("  PASS")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global RANK, LOCAL_RANK, WORLD_SIZE

    dist.init_process_group(backend="nccl")
    RANK = dist.get_rank()
    WORLD_SIZE = dist.get_world_size()
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", RANK % torch.cuda.device_count()))
    torch.cuda.set_device(LOCAL_RANK)

    log(f"=== vLLM NCCL Suspend/Resume Test (world_size={WORLD_SIZE}) ===")

    if WORLD_SIZE < 2:
        log("SKIP: Need 2+ GPUs (TP=1 has no NCCL comm)")
        dist.destroy_process_group()
        return

    # Check vLLM
    try:
        import vllm
        log(f"vLLM version: {vllm.__version__}")
    except ImportError:
        log("SKIP: vLLM not installed")
        dist.destroy_process_group()
        return

    # Init vLLM parallel state
    log("Initializing vLLM parallel state...")
    try:
        from vllm.distributed.parallel_state import (
            get_tp_group,
            init_distributed_environment,
            initialize_model_parallel,
        )
        init_distributed_environment(world_size=WORLD_SIZE, rank=RANK, local_rank=LOCAL_RANK)
        initialize_model_parallel(tensor_model_parallel_size=WORLD_SIZE)
        log(f"  TP group initialized (world_size={get_tp_group().world_size})")
    except Exception as e:
        log(f"SKIP: vLLM parallel init failed: {e}")
        dist.destroy_process_group()
        return

    # Load suspend/resume
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from verl.utils.nccl_suspend import _get_nccl_lib, resume_nccl_comm, suspend_nccl_comm

    if _get_nccl_lib() is None:
        log("SKIP: NCCL >= 2.29.7 required")
        dist.destroy_process_group()
        return

    # Run tests
    comm, comm_source = test_extract_handles()
    if comm is None:
        dist.destroy_process_group()
        return

    avg_freed = test_suspend_resume(comm, comm_source, suspend_nccl_comm, resume_nccl_comm)
    test_multicycle(comm, suspend_nccl_comm, resume_nccl_comm)
    test_compare_pynccl_vs_torch(suspend_nccl_comm, resume_nccl_comm)

    log(f"\n{'=' * 70}")
    log("ALL vLLM TESTS PASSED")
    log(f"  comm source: {comm_source}")
    log(f"  avg freed per suspend: {avg_freed:.0f} MB")
    log(f"{'=' * 70}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
