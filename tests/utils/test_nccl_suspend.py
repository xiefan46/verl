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
Standalone test for NCCL communicator suspend/resume.

Tests:
  1. Basic suspend/resume with multiple process groups + allreduce warmup
  2. P2P safety: send/recv between ranks, suspend, resume, verify P2P still works
  3. All-to-all memory: measure NCCL memory with all_to_all (larger internal buffers)

Usage:
    torchrun --nproc_per_node=2 tests/utils/test_nccl_suspend.py
    torchrun --nproc_per_node=8 tests/utils/test_nccl_suspend.py

Requires:
    - 2+ GPUs
    - NCCL >= 2.29.7 (for ncclCommSuspend/ncclCommResume)
    - PyTorch with NCCL backend
"""

import os
import sys

import torch
import torch.distributed as dist

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_memory_mb():
    """Get GPU memory usage in MB.

    Uses torch.cuda.mem_get_info() to query driver-level free/total memory,
    which captures NCCL's internal cudaMalloc allocations that are invisible
    to PyTorch's caching allocator (memory_allocated/reserved).
    """
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    return {
        "allocated": torch.cuda.memory_allocated() / 1024**2,
        "reserved": torch.cuda.memory_reserved() / 1024**2,
        "gpu_used": (total - free) / 1024**2,
        "gpu_free": free / 1024**2,
        "gpu_total": total / 1024**2,
    }


RANK = None
WORLD_SIZE = None


def log(msg):
    if RANK == 0:
        print(msg, flush=True)


def log_mem(label, mem):
    log(f"  {label}")
    log(f"    pytorch: allocated={mem['allocated']:.1f} MB, reserved={mem['reserved']:.1f} MB")
    log(f"    driver:  gpu_used={mem['gpu_used']:.1f} MB, gpu_free={mem['gpu_free']:.1f} MB")


def extract_comm(pg, name):
    """Extract ncclComm_t int pointer from a ProcessGroup."""
    try:
        backend = pg._get_backend(torch.device("cuda"))
        if hasattr(backend, "_comm_ptr"):
            return backend._comm_ptr()
    except Exception as e:
        log(f"  WARNING: failed to extract comm for '{name}': {e}")
    return None


def suspend_all(comm_handles, suspend_fn):
    """Suspend all comms. Returns True if all succeeded."""
    ok = True
    for name, comm in comm_handles:
        success = suspend_fn(comm)
        log(f"    Suspend '{name}': {'OK' if success else 'FAILED'}")
        if not success:
            ok = False
    torch.cuda.empty_cache()
    return ok


def resume_all(comm_handles, resume_fn):
    """Resume all comms. Returns True if all succeeded."""
    ok = True
    for name, comm in comm_handles:
        success = resume_fn(comm)
        log(f"    Resume '{name}': {'OK' if success else 'FAILED'}")
        if not success:
            ok = False
    torch.cuda.synchronize()
    return ok


def measure_suspend_resume(comm_handles, suspend_fn, resume_fn, label=""):
    """Suspend all comms, measure freed memory, resume, return freed_mb."""
    dist.barrier()
    torch.cuda.synchronize()
    mem_before = get_memory_mb()

    dist.barrier()
    suspend_all(comm_handles, suspend_fn)
    mem_after = get_memory_mb()
    freed = mem_before["gpu_used"] - mem_after["gpu_used"]

    log(f"  [{label}] before={mem_before['gpu_used']:.1f} MB, after={mem_after['gpu_used']:.1f} MB, freed={freed:.1f} MB")

    resume_all(comm_handles, resume_fn)
    return freed


# ---------------------------------------------------------------------------
# Test 1: Basic multi-group allreduce
# ---------------------------------------------------------------------------

def test_basic_allreduce(suspend_fn, resume_fn):
    """Original test: multiple process groups, allreduce warmup, suspend/resume."""
    log("\n" + "=" * 70)
    log("TEST 1: Basic multi-group allreduce suspend/resume")
    log("=" * 70)

    expected = sum(range(1, WORLD_SIZE + 1))
    all_ranks = list(range(WORLD_SIZE))

    # Create groups
    groups = {}
    group_names = ["tp", "dp", "ep", "cp", "pp", "tp_dp"]
    for name in group_names:
        groups[name] = dist.new_group(ranks=all_ranks)
    log(f"\n  Created {len(groups)} process groups: {group_names}")

    all_groups = [("default", None)] + [(n, pg) for n, pg in groups.items()]

    # Heavy warmup
    log("  Warming up (large allreduce on all groups)...")
    warmup_sizes = [(8192, 8192), (4096, 4096), (16384, 4096)]
    for gname, pg in all_groups:
        for size in warmup_sizes:
            for _ in range(3):
                buf = torch.randn(*size, device="cuda")
                if pg is None:
                    dist.all_reduce(buf)
                else:
                    dist.all_reduce(buf, group=pg)
                del buf

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    log_mem("After warmup:", get_memory_mb())

    # Extract comms
    comm_handles = []
    groups_to_extract = [("default", dist.distributed_c10d._get_default_group())]
    groups_to_extract += [(n, pg) for n, pg in groups.items()]
    for name, pg in groups_to_extract:
        ptr = extract_comm(pg, name)
        if ptr is not None:
            comm_handles.append((name, ptr))
    log(f"  Extracted {len(comm_handles)} comm handles")

    # Suspend / resume / measure
    freed = measure_suspend_resume(comm_handles, suspend_fn, resume_fn, label="allreduce")

    # Verify all groups still work
    for gname, pg in all_groups:
        z = torch.ones(1024, 1024, device="cuda") * (RANK + 1)
        if pg is None:
            dist.all_reduce(z)
        else:
            dist.all_reduce(z, group=pg)
        assert torch.allclose(z, torch.full_like(z, expected)), f"allreduce failed ({gname})"
        del z
    log(f"  Post-resume verify: all {len(all_groups)} groups OK")

    return comm_handles, groups, freed


# ---------------------------------------------------------------------------
# Test 2: P2P safety (send/recv)
# ---------------------------------------------------------------------------

def test_p2p_safety(suspend_fn, resume_fn):
    """Test that P2P send/recv works correctly after suspend/resume.

    P2P operations create cross-rank memory mappings (IPC handles via NVLink).
    NCCL suspend must properly unmap peer buffers and re-exchange handles on resume.
    """
    log("\n" + "=" * 70)
    log("TEST 2: P2P send/recv safety across suspend/resume")
    log("=" * 70)

    pg = dist.distributed_c10d._get_default_group()

    # Step 1: Warm up P2P paths — every rank sends to every other rank
    log("\n  Warming up P2P (all-pairs send/recv)...")
    tensor_size = (4096, 4096)  # 64 MB float32
    for round_idx in range(3):
        for src in range(WORLD_SIZE):
            dst = (src + 1) % WORLD_SIZE
            if RANK == src:
                buf = torch.ones(*tensor_size, device="cuda") * (src + 1)
                dist.send(buf, dst)
                del buf
            elif RANK == dst:
                buf = torch.zeros(*tensor_size, device="cuda")
                dist.recv(buf, src)
                assert buf[0, 0].item() == src + 1, f"P2P warmup recv wrong: got {buf[0, 0].item()}"
                del buf
            dist.barrier()

    # Also do ring pattern send/recv (simultaneous bidirectional P2P)
    log("  Warming up ring P2P (simultaneous send+recv)...")
    for _ in range(3):
        send_buf = torch.ones(*tensor_size, device="cuda") * (RANK + 1)
        recv_buf = torch.zeros(*tensor_size, device="cuda")
        send_dst = (RANK + 1) % WORLD_SIZE
        recv_src = (RANK - 1) % WORLD_SIZE
        send_op = dist.isend(send_buf, send_dst)
        recv_op = dist.irecv(recv_buf, recv_src)
        send_op.wait()
        recv_op.wait()
        assert recv_buf[0, 0].item() == recv_src + 1
        del send_buf, recv_buf

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    log_mem("After P2P warmup:", get_memory_mb())

    # Step 2: Suspend
    comm_ptr = extract_comm(pg, "default")
    assert comm_ptr is not None
    comm_handles = [("default", comm_ptr)]

    freed = measure_suspend_resume(comm_handles, suspend_fn, resume_fn, label="p2p")

    # Step 3: Verify P2P still works after resume
    log("  Verifying P2P after resume...")

    # Pairwise send/recv
    for src in range(WORLD_SIZE):
        dst = (src + 1) % WORLD_SIZE
        if RANK == src:
            buf = torch.ones(2048, 2048, device="cuda") * (src * 10 + 7)
            dist.send(buf, dst)
            del buf
        elif RANK == dst:
            buf = torch.zeros(2048, 2048, device="cuda")
            dist.recv(buf, src)
            assert buf[0, 0].item() == src * 10 + 7, f"P2P post-resume recv wrong: {buf[0, 0].item()}"
            del buf
        dist.barrier()

    # Ring pattern
    send_buf = torch.ones(2048, 2048, device="cuda") * (RANK * 100 + 42)
    recv_buf = torch.zeros(2048, 2048, device="cuda")
    send_dst = (RANK + 1) % WORLD_SIZE
    recv_src = (RANK - 1) % WORLD_SIZE
    send_op = dist.isend(send_buf, send_dst)
    recv_op = dist.irecv(recv_buf, recv_src)
    send_op.wait()
    recv_op.wait()
    expected_val = recv_src * 100 + 42
    assert recv_buf[0, 0].item() == expected_val, f"Ring P2P post-resume wrong: {recv_buf[0, 0].item()} != {expected_val}"
    del send_buf, recv_buf

    # Step 4: Multiple suspend/resume cycles with P2P
    log("  Testing 3 suspend/resume cycles with P2P...")
    for cycle in range(3):
        dist.barrier()
        dist.barrier()
        suspend_all(comm_handles, suspend_fn)
        resume_all(comm_handles, resume_fn)

        # Verify ring P2P
        send_buf = torch.ones(1024, 1024, device="cuda") * (RANK + cycle + 1)
        recv_buf = torch.zeros(1024, 1024, device="cuda")
        send_op = dist.isend(send_buf, (RANK + 1) % WORLD_SIZE)
        recv_op = dist.irecv(recv_buf, (RANK - 1) % WORLD_SIZE)
        send_op.wait()
        recv_op.wait()
        expected_val = (RANK - 1) % WORLD_SIZE + cycle + 1
        assert recv_buf[0, 0].item() == expected_val
        del send_buf, recv_buf
        log(f"    Cycle {cycle + 1}: P2P OK")

    log(f"  P2P safety test PASSED (freed={freed:.1f} MB)")
    return freed


# ---------------------------------------------------------------------------
# Test 3: All-to-all memory measurement
# ---------------------------------------------------------------------------

def test_alltoall_memory(suspend_fn, resume_fn):
    """Measure NCCL memory with all_to_all operations.

    AMem notes that all-to-all allocates more internal NCCL buffers because
    it creates O(n^2) peer connections (every rank sends to every other rank),
    unlike allreduce which uses ring/tree topology.
    """
    log("\n" + "=" * 70)
    log("TEST 3: All-to-all NCCL memory measurement")
    log("=" * 70)

    # Baseline: memory before any all_to_all
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    mem_baseline = get_memory_mb()
    log_mem("\n  Baseline (before all_to_all):", mem_baseline)

    # Create a fresh group for all_to_all testing
    a2a_group = dist.new_group(ranks=list(range(WORLD_SIZE)))

    # Warm up all_to_all with increasing sizes
    log("  Warming up all_to_all...")
    chunk_sizes = [
        1024 * 1024,      # 4 MB per chunk × world_size
        4 * 1024 * 1024,  # 16 MB per chunk × world_size
        16 * 1024 * 1024, # 64 MB per chunk × world_size
    ]
    for cs in chunk_sizes:
        for _ in range(3):
            input_tensor = torch.randn(cs * WORLD_SIZE, device="cuda")
            output_tensor = torch.empty_like(input_tensor)
            dist.all_to_all_single(output_tensor, input_tensor, group=a2a_group)
            del input_tensor, output_tensor

        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        mem_now = get_memory_mb()
        log(f"    chunk={cs // 1024 // 1024}MB: gpu_used={mem_now['gpu_used']:.1f} MB")

    # Also test list-based all_to_all (different shapes per rank)
    log("  Warming up all_to_all with per-rank tensors...")
    for _ in range(3):
        input_list = [torch.randn(2048, 2048, device="cuda") for _ in range(WORLD_SIZE)]
        output_list = [torch.empty(2048, 2048, device="cuda") for _ in range(WORLD_SIZE)]
        dist.all_to_all(output_list, input_list, group=a2a_group)
        del input_list, output_list

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    mem_after_a2a = get_memory_mb()
    a2a_nccl_mem = mem_after_a2a["gpu_used"] - mem_baseline["gpu_used"]
    log_mem("  After all_to_all warmup:", mem_after_a2a)
    log(f"  >>> NCCL memory from all_to_all group: {a2a_nccl_mem:.1f} MB <<<")

    # Suspend / resume the all_to_all group
    a2a_comm = extract_comm(a2a_group, "a2a")
    assert a2a_comm is not None
    comm_handles = [("a2a", a2a_comm)]

    freed = measure_suspend_resume(comm_handles, suspend_fn, resume_fn, label="all_to_all")

    # Verify all_to_all still works
    input_tensor = torch.ones(1024 * WORLD_SIZE, device="cuda") * (RANK + 1)
    output_tensor = torch.empty_like(input_tensor)
    dist.all_to_all_single(output_tensor, input_tensor, group=a2a_group)
    # Each chunk should contain (src_rank + 1)
    for r in range(WORLD_SIZE):
        chunk = output_tensor[r * 1024 : (r + 1) * 1024]
        assert torch.allclose(chunk, torch.full_like(chunk, r + 1)), f"all_to_all wrong for rank {r}"
    del input_tensor, output_tensor
    log("  Post-resume all_to_all verify: OK")

    log(f"  All-to-all memory test PASSED (nccl_mem={a2a_nccl_mem:.1f} MB, freed={freed:.1f} MB)")
    return a2a_nccl_mem, freed


# ---------------------------------------------------------------------------
# Test 4: Comparison — allreduce-only vs all_to_all NCCL memory per comm
# ---------------------------------------------------------------------------

def test_memory_comparison(suspend_fn, resume_fn):
    """Compare NCCL memory usage: allreduce-only group vs all_to_all group.

    AMem claims all_to_all uses significantly more memory due to O(n^2) peer
    connections. This test measures both on the same world_size for comparison.
    """
    log("\n" + "=" * 70)
    log("TEST 4: Memory comparison — allreduce-only vs all_to_all per comm")
    log("=" * 70)

    all_ranks = list(range(WORLD_SIZE))

    results = {}
    for op_name, warmup_fn in [("allreduce", None), ("all_to_all", None)]:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        mem_before_group = get_memory_mb()

        pg = dist.new_group(ranks=all_ranks)

        # Warmup
        if op_name == "allreduce":
            for _ in range(5):
                buf = torch.randn(8192, 8192, device="cuda")
                dist.all_reduce(buf, group=pg)
                del buf
        else:
            for _ in range(5):
                buf_in = torch.randn(4096 * WORLD_SIZE, 1024, device="cuda")
                buf_out = torch.empty_like(buf_in)
                dist.all_to_all_single(buf_out, buf_in, group=pg)
                del buf_in, buf_out

        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        mem_after_warmup = get_memory_mb()
        nccl_mem = mem_after_warmup["gpu_used"] - mem_before_group["gpu_used"]

        # Suspend to measure freeable portion
        comm_ptr = extract_comm(pg, op_name)
        assert comm_ptr is not None
        dist.barrier()
        torch.cuda.synchronize()
        mem_pre = get_memory_mb()
        dist.barrier()
        suspend_all([(op_name, comm_ptr)], suspend_fn)
        mem_post = get_memory_mb()
        freed = mem_pre["gpu_used"] - mem_post["gpu_used"]
        resume_all([(op_name, comm_ptr)], resume_fn)

        results[op_name] = {"nccl_mem": nccl_mem, "freed": freed}
        log(f"\n  {op_name}: nccl_mem={nccl_mem:.1f} MB, freed_by_suspend={freed:.1f} MB")

    ar = results["allreduce"]
    a2a = results["all_to_all"]
    ratio = a2a["nccl_mem"] / ar["nccl_mem"] if ar["nccl_mem"] > 0 else float("inf")
    log(f"\n  >>> all_to_all / allreduce memory ratio: {ratio:.2f}x <<<")
    log(f"  >>> allreduce freed: {ar['freed']:.1f} MB, all_to_all freed: {a2a['freed']:.1f} MB <<<")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global RANK, WORLD_SIZE

    dist.init_process_group(backend="nccl")
    RANK = dist.get_rank()
    WORLD_SIZE = dist.get_world_size()
    torch.cuda.set_device(RANK)

    log(f"=== NCCL Suspend/Resume Test Suite (world_size={WORLD_SIZE}) ===")

    # Load suspend/resume
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from verl.utils.nccl_suspend import _get_nccl_lib, resume_nccl_comm, suspend_nccl_comm

    nccl_lib = _get_nccl_lib()
    if nccl_lib is None:
        log("SKIP: NCCL library does not support suspend/resume (need >= 2.29.7)")
        dist.destroy_process_group()
        return

    log("ncclCommSuspend available!\n")

    # Run tests
    _, _, freed_basic = test_basic_allreduce(suspend_nccl_comm, resume_nccl_comm)
    freed_p2p = test_p2p_safety(suspend_nccl_comm, resume_nccl_comm)
    a2a_nccl_mem, freed_a2a = test_alltoall_memory(suspend_nccl_comm, resume_nccl_comm)
    comparison = test_memory_comparison(suspend_nccl_comm, resume_nccl_comm)

    # Final summary
    log(f"\n{'=' * 70}")
    log("FINAL SUMMARY")
    log(f"{'=' * 70}")
    log(f"  world_size:              {WORLD_SIZE}")
    log(f"  Test 1 (basic allreduce):  freed={freed_basic:.1f} MB  PASS")
    log(f"  Test 2 (P2P safety):       freed={freed_p2p:.1f} MB  PASS")
    log(f"  Test 3 (all_to_all):       nccl_mem={a2a_nccl_mem:.1f} MB, freed={freed_a2a:.1f} MB  PASS")
    log(f"  Test 4 (comparison):")
    for op, r in comparison.items():
        log(f"    {op:15s}: nccl_mem={r['nccl_mem']:.1f} MB, freed={r['freed']:.1f} MB")
    if comparison["allreduce"]["nccl_mem"] > 0:
        ratio = comparison["all_to_all"]["nccl_mem"] / comparison["allreduce"]["nccl_mem"]
        log(f"    all_to_all / allreduce ratio: {ratio:.2f}x")
    log(f"{'=' * 70}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
