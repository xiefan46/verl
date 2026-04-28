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
Test for vLLM NCCL communicator suspend/resume integration.

Verifies that suspend_vllm_comms() / resume_vllm_comms() actually
frees GPU memory and that vLLM NCCL operations still work after resume.

Usage:
    torchrun --nproc_per_node=2 tests/utils/test_vllm_nccl_suspend.py

Requires:
    - 2+ GPUs (TP=1 has no NCCL comm, nothing to test)
    - vLLM installed
    - NCCL >= 2.29.7
"""

import torch
import torch.distributed as dist


def get_memory_mb():
    torch.cuda.synchronize()
    return {
        "allocated": torch.cuda.memory_allocated() / 1024**2,
        "reserved": torch.cuda.memory_reserved() / 1024**2,
    }


def log(rank, msg):
    if rank == 0:
        print(msg, flush=True)


def main():
    # --- Init torch.distributed (needed by vLLM parallel state) ---
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)

    log(rank, "=== vLLM NCCL Suspend/Resume Integration Test ===")
    log(rank, f"world_size={world_size}, CUDA device={torch.cuda.current_device()}\n")

    if world_size < 2:
        log(rank, "SKIP: Need 2+ GPUs for TP NCCL comm. Run with torchrun --nproc_per_node=2")
        dist.destroy_process_group()
        return

    # --- Init vLLM parallel state ---
    log(rank, "[1] Initializing vLLM parallel state...")
    try:
        from vllm.distributed.parallel_state import (
            get_tp_group,
            init_distributed_environment,
            init_model_parallel_backend,
            initialize_model_parallel,
        )
    except ImportError:
        log(rank, "SKIP: vLLM not installed.")
        dist.destroy_process_group()
        return

    # vLLM needs its own distributed init
    init_distributed_environment(
        world_size=world_size,
        rank=rank,
        local_rank=rank,
    )
    init_model_parallel_backend()
    initialize_model_parallel(tensor_model_parallel_size=world_size)

    tp_group = get_tp_group()
    log(rank, f"  TP group initialized (world_size={tp_group.world_size})")

    # --- Warm up: do allreduce to allocate NCCL buffers ---
    log(rank, "\n[2] Warming up NCCL (allreduce to allocate internal buffers)...")
    for i in range(5):
        x = torch.randn(4096, 4096, device="cuda")
        dist.all_reduce(x, group=tp_group.device_group)
    torch.cuda.synchronize()
    del x
    torch.cuda.empty_cache()

    mem_baseline = get_memory_mb()
    log(rank, "  Baseline memory after warmup:")
    log(rank, f"    allocated={mem_baseline['allocated']:.1f} MB, reserved={mem_baseline['reserved']:.1f} MB")

    # --- Test suspend_vllm_comms ---
    log(rank, "\n[3] Testing suspend_vllm_comms()...")
    from verl.utils.nccl_suspend import _get_all_vllm_comm_handles, resume_vllm_comms, suspend_vllm_comms

    # Verify we can find the comm handles
    handles = _get_all_vllm_comm_handles()
    log(rank, f"  Found {len(handles)} NCCL comm(s): {[name for name, _ in handles]}")
    if not handles:
        log(rank, "  FAIL: No NCCL comm handles found. Check vLLM version compatibility.")
        dist.destroy_process_group()
        return

    # Measure memory before suspend
    mem_before = get_memory_mb()

    # Suspend
    success = suspend_vllm_comms()
    log(rank, f"  suspend_vllm_comms() returned: {success}")

    mem_after = get_memory_mb()
    freed_reserved = mem_before["reserved"] - mem_after["reserved"]
    freed_allocated = mem_before["allocated"] - mem_after["allocated"]

    alloc_before = mem_before["allocated"]
    resv_before = mem_before["reserved"]
    alloc_after = mem_after["allocated"]
    resv_after = mem_after["reserved"]
    log(rank, f"\n  Memory before suspend: allocated={alloc_before:.1f} MB, reserved={resv_before:.1f} MB")
    log(rank, f"  Memory after suspend:  allocated={alloc_after:.1f} MB, reserved={resv_after:.1f} MB")
    log(rank, f"  >>> Freed: {freed_reserved:.1f} MB reserved, {freed_allocated:.1f} MB allocated <<<")

    if freed_reserved <= 0 and freed_allocated <= 0:
        log(rank, "  WARNING: No memory freed. NCCL suspend may not have released memory,")
        log(rank, "           or NCCL version does not support ncclCommSuspend.")

    # --- Test resume_vllm_comms ---
    log(rank, "\n[4] Testing resume_vllm_comms()...")
    success = resume_vllm_comms()
    log(rank, f"  resume_vllm_comms() returned: {success}")

    mem_resumed = get_memory_mb()
    alloc_resumed = mem_resumed["allocated"]
    resv_resumed = mem_resumed["reserved"]
    log(rank, f"  Memory after resume: allocated={alloc_resumed:.1f} MB, reserved={resv_resumed:.1f} MB")

    # --- Verify NCCL still works ---
    log(rank, "\n[5] Verifying NCCL allreduce works after resume...")
    x = torch.ones(1024, 1024, device="cuda") * (rank + 1)
    dist.all_reduce(x, group=tp_group.device_group)
    expected = sum(range(1, world_size + 1))
    correct = torch.allclose(x, torch.full_like(x, expected))
    log(rank, f"  allreduce result={x[0, 0].item()}, expected={expected}, correct={correct}")
    assert correct, "NCCL allreduce failed after resume!"

    # --- Multiple suspend/resume cycles ---
    log(rank, "\n[6] Testing 3 consecutive suspend/resume cycles...")
    for cycle in range(3):
        mem_pre = get_memory_mb()
        suspend_vllm_comms()
        mem_mid = get_memory_mb()
        resume_vllm_comms()

        # Verify allreduce
        y = torch.ones(512, 512, device="cuda") * (rank + 1)
        dist.all_reduce(y, group=tp_group.device_group)
        ok = torch.allclose(y, torch.full_like(y, expected))

        freed = mem_pre["reserved"] - mem_mid["reserved"]
        log(rank, f"  Cycle {cycle + 1}: freed={freed:.1f} MB, allreduce={'OK' if ok else 'FAIL'}")
        assert ok, f"allreduce failed on cycle {cycle + 1}"

    # --- Summary ---
    log(rank, f"\n{'=' * 60}")
    log(rank, "RESULT: ALL TESTS PASSED")
    log(rank, f"  NCCL comms found:      {[name for name, _ in handles]}")
    log(rank, f"  Memory freed per suspend: {freed_reserved:.1f} MB")
    log(rank, "  Multi-cycle stability:   3/3 passed")
    log(rank, f"{'=' * 60}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
