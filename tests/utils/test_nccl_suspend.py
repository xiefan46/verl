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

Usage:
    torchrun --nproc_per_node=2 tests/utils/test_nccl_suspend.py

Requires:
    - 2+ GPUs
    - NCCL >= 2.29.7 (for ncclCommSuspend/ncclCommResume)
    - PyTorch with NCCL backend
"""

import os
import sys

import torch
import torch.distributed as dist


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


def log(rank, msg):
    if rank == 0:
        print(msg, flush=True)


def test_nccl_suspend_resume():
    # --- Init distributed ---
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)

    log(rank, f"=== NCCL Suspend/Resume Test (world_size={world_size}) ===\n")

    # --- Step 1: Create multiple process groups (simulate Megatron TP/DP/EP/CP/PP) ---
    log(rank, "[Step 1] Creating multiple process groups to simulate real training...")
    expected = sum(range(1, world_size + 1))
    all_ranks = list(range(world_size))

    groups = {}
    group_names = ["tp", "dp", "ep", "cp", "pp", "tp_dp"]
    for name in group_names:
        groups[name] = dist.new_group(ranks=all_ranks)
    log(rank, f"  Created {len(groups)} process groups: {group_names}")

    # Verify default group works
    x = torch.ones(1024, 1024, device="cuda") * (rank + 1)
    dist.all_reduce(x)
    assert torch.allclose(x, torch.full_like(x, expected)), "allreduce failed"
    log(rank, f"  default group allreduce OK (result={x[0, 0].item()})")
    del x

    # --- Step 2: Heavy warmup to force NCCL internal buffer allocation ---
    log(rank, "\n[Step 2] Heavy warmup (large allreduce on all groups)...")
    warmup_sizes = [
        (8192, 8192),   # 256 MB per tensor (float32)
        (4096, 4096),   # 64 MB
        (16384, 4096),  # 256 MB
    ]
    all_groups = [("default", None)] + [(name, pg) for name, pg in groups.items()]

    for gname, pg in all_groups:
        for size in warmup_sizes:
            for _ in range(3):
                buf = torch.randn(*size, device="cuda")
                if pg is None:
                    dist.all_reduce(buf)
                else:
                    dist.all_reduce(buf, group=pg)
                del buf
        log(rank, f"  warmed up '{gname}'")

    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    mem_before = get_memory_mb()
    log(rank, f"\n  Memory after warmup + empty_cache:")
    log(rank, f"  pytorch: allocated={mem_before['allocated']:.1f} MB, reserved={mem_before['reserved']:.1f} MB")
    log(rank, f"  driver:  gpu_used={mem_before['gpu_used']:.1f} MB, gpu_free={mem_before['gpu_free']:.1f} MB")

    # --- Step 3: Load nccl_suspend and check availability ---
    log(rank, "\n[Step 3] Loading NCCL suspend API...")
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from verl.utils.nccl_suspend import _get_nccl_lib, resume_nccl_comm, suspend_nccl_comm

    nccl_lib = _get_nccl_lib()
    if nccl_lib is None:
        log(rank, "  SKIP: NCCL library does not support suspend/resume (need >= 2.29.7)")
        dist.destroy_process_group()
        return

    log(rank, "  ncclCommSuspend available!")

    # --- Step 4: Extract ncclComm_t from ProcessGroupNCCL ---
    log(rank, "\n[Step 4] Extracting ncclComm_t handles...")

    comm_handles = []

    # Use ProcessGroupNCCL._comm_ptr() to get ncclComm_t as int pointer.
    # This is a public (but unsafe) PyTorch API available in recent versions.
    groups_to_extract = [("default", dist.distributed_c10d._get_default_group())]
    groups_to_extract += [(name, pg) for name, pg in groups.items()]

    for name, pg in groups_to_extract:
        try:
            pg_nccl = pg._get_backend(torch.device("cuda"))
            if hasattr(pg_nccl, "_comm_ptr"):
                comm_ptr = pg_nccl._comm_ptr()
                log(rank, f"  {name}: _comm_ptr() = {hex(comm_ptr)}")
                comm_handles.append((name, comm_ptr))
            else:
                log(rank, f"  {name}: _comm_ptr not available")
        except Exception as e:
            log(rank, f"  {name}: failed to extract comm: {e}")

    if not comm_handles:
        log(rank, "\n  Cannot extract ncclComm_t. ProcessGroupNCCL._comm_ptr() not available.")
        log(rank, "  Requires PyTorch >= 2.x with NCCL backend.")
        dist.destroy_process_group()
        return

    # --- Step 5: Suspend and measure ---
    dist.barrier()  # sync all ranks before suspend
    torch.cuda.synchronize()

    mem_before_suspend = get_memory_mb()
    log(rank, "\n[Step 5] Memory before suspend:")
    log(rank, f"  pytorch: allocated={mem_before_suspend['allocated']:.1f} MB, reserved={mem_before_suspend['reserved']:.1f} MB")
    log(rank, f"  driver:  gpu_used={mem_before_suspend['gpu_used']:.1f} MB, gpu_free={mem_before_suspend['gpu_free']:.1f} MB")

    log(rank, "\n  Calling ncclCommSuspend on all comms...")
    dist.barrier()  # ensure all ranks suspend together

    for name, comm in comm_handles:
        success = suspend_nccl_comm(comm)
        log(rank, f"  Suspend '{name}': {'OK' if success else 'FAILED'}")

    torch.cuda.empty_cache()

    mem_after_suspend = get_memory_mb()
    freed_gpu = mem_before_suspend["gpu_used"] - mem_after_suspend["gpu_used"]
    log(rank, "\n  Memory after suspend + empty_cache:")
    log(rank, f"  pytorch: allocated={mem_after_suspend['allocated']:.1f} MB, reserved={mem_after_suspend['reserved']:.1f} MB")
    log(rank, f"  driver:  gpu_used={mem_after_suspend['gpu_used']:.1f} MB, gpu_free={mem_after_suspend['gpu_free']:.1f} MB")
    log(rank, f"  >>> GPU memory freed by suspend: {freed_gpu:.1f} MB <<<")

    # --- Step 6: Resume and verify ---
    # NOTE: cannot use dist.barrier() here — NCCL comms are suspended!
    # All ranks are already synchronized from the barrier before suspend.
    log(rank, "\n[Step 6] Calling ncclCommResume...")

    for name, comm in comm_handles:
        success = resume_nccl_comm(comm)
        log(rank, f"  Resume '{name}': {'OK' if success else 'FAILED'}")

    torch.cuda.synchronize()

    mem_after_resume = get_memory_mb()
    log(rank, "\n  Memory after resume:")
    log(rank, f"  pytorch: allocated={mem_after_resume['allocated']:.1f} MB, reserved={mem_after_resume['reserved']:.1f} MB")
    log(rank, f"  driver:  gpu_used={mem_after_resume['gpu_used']:.1f} MB, gpu_free={mem_after_resume['gpu_free']:.1f} MB")

    # --- Step 7: Verify NCCL still works after resume ---
    log(rank, "\n[Step 7] Verify NCCL allreduce works after resume...")

    for gname, pg in all_groups:
        z = torch.ones(1024, 1024, device="cuda") * (rank + 1)
        if pg is None:
            dist.all_reduce(z)
        else:
            dist.all_reduce(z, group=pg)
        assert torch.allclose(z, torch.full_like(z, expected)), f"allreduce failed after resume ({gname})"
        del z
    log(rank, f"  All {len(all_groups)} groups verified OK")

    # --- Summary ---
    log(rank, f"\n{'=' * 60}")
    log(rank, f"SUMMARY (rank {rank}):")
    log(rank, f"  NCCL comms tested:  {[name for name, _ in comm_handles]}")
    log(rank, f"  GPU memory freed:   {freed_gpu:.1f} MB")
    log(rank, "  Post-resume verify: PASS")
    log(rank, f"{'=' * 60}")

    dist.destroy_process_group()


if __name__ == "__main__":
    test_nccl_suspend_resume()
