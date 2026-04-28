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

    # Try to get comm handles from the default process group
    default_pg = dist.group.WORLD
    backend = dist.get_backend(default_pg)
    log(rank, f"  Default PG backend: {backend}")

    # Method: use torch's internal API to get the NCCL backend object
    try:
        pg_nccl = dist.distributed_c10d._get_default_group()._get_backend(torch.device("cuda"))
        log(rank, f"  Got backend object: {type(pg_nccl).__name__}")

        # Try to access comm via bound_device_id or other attributes
        # This is exploratory - log what attributes are available
        attrs = [a for a in dir(pg_nccl) if not a.startswith("__")]
        log(rank, f"  Available attributes: {attrs[:20]}...")

        # Check for _get_backend_name or similar
        if hasattr(pg_nccl, "comm"):
            log(rank, "  Found pg_nccl.comm")
            comm_handles.append(("default", pg_nccl.comm))
        if hasattr(pg_nccl, "_get_communicator"):
            log(rank, "  Found pg_nccl._get_communicator()")
        if hasattr(pg_nccl, "nccl_comm"):
            log(rank, "  Found pg_nccl.nccl_comm")
            comm_handles.append(("default", pg_nccl.nccl_comm))

    except Exception as e:
        log(rank, f"  Failed to get NCCL backend: {e}")

    if not comm_handles:
        log(rank, "\n  Cannot extract ncclComm_t from PyTorch ProcessGroupNCCL.")
        log(rank, "  This is expected - training-side suspend needs C++ extension (Step 3 in plan).")
        log(rank, "  Testing with ctypes direct NCCL comm creation instead...\n")

        # Fallback: create our own NCCL comm via ctypes to test suspend/resume works
        import ctypes

        nccl = _get_nccl_lib()

        # Create a unique ID on rank 0 and broadcast
        unique_id = (ctypes.c_byte * 128)()
        if rank == 0:
            nccl.ncclGetUniqueId(ctypes.byref(unique_id))

        # Broadcast unique_id via gloo
        id_tensor = torch.frombuffer(unique_id, dtype=torch.uint8).clone().cuda()
        dist.broadcast(id_tensor, src=0)
        ctypes.memmove(unique_id, id_tensor.cpu().numpy().ctypes.data, 128)
        del id_tensor

        # Init comm
        comm = ctypes.c_void_p()
        result = nccl.ncclCommInitRank(ctypes.byref(comm), world_size, unique_id, rank)
        if result != 0:
            log(rank, f"  ncclCommInitRank failed with {result}")
            dist.destroy_process_group()
            return
        log(rank, f"  Created standalone NCCL comm: {comm.value}")
        comm_handles.append(("standalone", comm))

        # Do an allreduce to warm up the comm and allocate buffers
        sendbuf = torch.ones(4096, 4096, device="cuda") * (rank + 1)
        recvbuf = torch.zeros_like(sendbuf)

        nccl.ncclAllReduce.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        nccl.ncclAllReduce.restype = ctypes.c_int

        stream = torch.cuda.current_stream().cuda_stream
        # ncclFloat=7, ncclSum=0
        result = nccl.ncclAllReduce(
            sendbuf.data_ptr(),
            recvbuf.data_ptr(),
            sendbuf.numel(),
            7,
            0,
            comm,
            stream,
        )
        torch.cuda.synchronize()
        assert result == 0, f"ncclAllReduce failed with {result}"
        log(rank, f"  Standalone allreduce OK (result={recvbuf[0, 0].item()})")

        del sendbuf, recvbuf
        torch.cuda.empty_cache()

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

    if comm_handles[0][0] == "standalone":
        # Test standalone comm
        name, comm = comm_handles[0]
        sendbuf = torch.ones(1024, 1024, device="cuda") * (rank + 1)
        recvbuf = torch.zeros_like(sendbuf)
        stream = torch.cuda.current_stream().cuda_stream
        result = nccl.ncclAllReduce(
            sendbuf.data_ptr(),
            recvbuf.data_ptr(),
            sendbuf.numel(),
            7,
            0,
            comm,
            stream,
        )
        torch.cuda.synchronize()
        assert result == 0, f"ncclAllReduce after resume failed with {result}"
        assert torch.allclose(recvbuf, torch.full_like(recvbuf, expected)), "allreduce result wrong after resume"
        log(rank, f"  Standalone allreduce after resume OK (result={recvbuf[0, 0].item()})")

        # Cleanup standalone comm
        nccl.ncclCommDestroy(comm)
    else:
        # Test PyTorch process group
        z = torch.ones(1024, 1024, device="cuda") * (rank + 1)
        dist.all_reduce(z)
        assert torch.allclose(z, torch.full_like(z, expected)), "allreduce after resume failed"
        log(rank, f"  allreduce after resume OK (result={z[0, 0].item()})")

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
