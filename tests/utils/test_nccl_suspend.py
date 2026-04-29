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
    """Get current GPU memory usage in MB."""
    return {
        "allocated": torch.cuda.memory_allocated() / 1024**2,
        "reserved": torch.cuda.memory_reserved() / 1024**2,
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

    # --- Step 1: Verify NCCL works ---
    log(rank, "[Step 1] Verify NCCL allreduce works...")
    x = torch.ones(1024, 1024, device="cuda") * (rank + 1)
    dist.all_reduce(x)
    expected = sum(range(1, world_size + 1))
    assert torch.allclose(x, torch.full_like(x, expected)), "allreduce failed"
    log(rank, f"  allreduce OK (result={x[0, 0].item()})")

    # Also create a second process group to test multi-group suspend
    sub_group = dist.new_group(ranks=list(range(world_size)))
    y = torch.ones(512, 512, device="cuda") * (rank + 1)
    dist.all_reduce(y, group=sub_group)
    assert torch.allclose(y, torch.full_like(y, expected)), "sub_group allreduce failed"
    log(rank, "  sub_group allreduce OK")

    torch.cuda.synchronize()
    del x, y
    torch.cuda.empty_cache()

    # --- Step 2: Measure baseline memory ---
    mem_before = get_memory_mb()
    log(rank, "\n[Step 2] Memory before suspend:")
    log(rank, f"  allocated={mem_before['allocated']:.1f} MB, reserved={mem_before['reserved']:.1f} MB")

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
    groups_to_extract = [
        ("default", dist.distributed_c10d._get_default_group()),
        ("sub_group", sub_group),
    ]

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
    log(rank, f"  allocated={mem_before_suspend['allocated']:.1f} MB, reserved={mem_before_suspend['reserved']:.1f} MB")

    log(rank, "\n  Calling ncclCommSuspend on all comms...")
    dist.barrier()  # ensure all ranks suspend together

    for name, comm in comm_handles:
        success = suspend_nccl_comm(comm)
        log(rank, f"  Suspend '{name}': {'OK' if success else 'FAILED'}")

    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    mem_after_suspend = get_memory_mb()
    freed_reserved = mem_before_suspend["reserved"] - mem_after_suspend["reserved"]
    freed_allocated = mem_before_suspend["allocated"] - mem_after_suspend["allocated"]
    log(rank, "\n  Memory after suspend + empty_cache:")
    log(rank, f"  allocated={mem_after_suspend['allocated']:.1f} MB, reserved={mem_after_suspend['reserved']:.1f} MB")
    log(rank, f"  Freed: {freed_reserved:.1f} MB reserved, {freed_allocated:.1f} MB allocated")

    # --- Step 6: Resume and verify ---
    log(rank, "\n[Step 6] Calling ncclCommResume...")
    dist.barrier()

    for name, comm in comm_handles:
        success = resume_nccl_comm(comm)
        log(rank, f"  Resume '{name}': {'OK' if success else 'FAILED'}")

    torch.cuda.synchronize()

    mem_after_resume = get_memory_mb()
    log(rank, "\n  Memory after resume:")
    log(rank, f"  allocated={mem_after_resume['allocated']:.1f} MB, reserved={mem_after_resume['reserved']:.1f} MB")

    # --- Step 7: Verify NCCL still works after resume ---
    log(rank, "\n[Step 7] Verify NCCL allreduce works after resume...")

    # Test default process group
    z = torch.ones(1024, 1024, device="cuda") * (rank + 1)
    dist.all_reduce(z)
    assert torch.allclose(z, torch.full_like(z, expected)), "allreduce after resume failed (default group)"
    log(rank, f"  default group allreduce OK (result={z[0, 0].item()})")

    # Test sub_group
    w = torch.ones(512, 512, device="cuda") * (rank + 1)
    dist.all_reduce(w, group=sub_group)
    assert torch.allclose(w, torch.full_like(w, expected)), "allreduce after resume failed (sub_group)"
    log(rank, "  sub_group allreduce OK")

    # --- Summary ---
    log(rank, f"\n{'=' * 60}")
    log(rank, f"SUMMARY (rank {rank}):")
    log(rank, f"  NCCL comms tested:  {[name for name, _ in comm_handles]}")
    log(rank, f"  Memory freed:       {freed_reserved:.1f} MB reserved, {freed_allocated:.1f} MB allocated")
    log(rank, "  Post-resume verify: PASS")
    log(rank, f"{'=' * 60}")

    dist.destroy_process_group()


if __name__ == "__main__":
    test_nccl_suspend_resume()
