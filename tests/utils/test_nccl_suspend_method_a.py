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
Test Method A (megatron.core.parallel_state reflection) for NCCL pause/resume.

Compares Method A against Method B (pg_map scan) on a real Megatron parallel
state, validates that both find the same set of NCCL communicators (modulo
weight-transfer false positives that only Method B is susceptible to), and
verifies the suspend/resume cycle releases and restores GPU memory cleanly.

Requirements:
  * NCCL >= 2.29.7 (libnccl.so.2 must export ncclCommSuspend / ncclCommResume).
    The test gracefully skips if the API is unavailable.
  * megatron-core installed.
  * 8 GPUs with NVLink (single-node).

Usage:
    torchrun --nproc_per_node=8 tests/utils/test_nccl_suspend_method_a.py

Optional env vars:
    TP_SIZE, PP_SIZE, DP_SIZE, CP_SIZE  parallel dims (defaults: 2/2/2/1)
    NCCL_NVLS_ENABLE=0                  recommended in CI without Fabric Manager
"""

import os
import sys
import time

import torch
import torch.distributed as dist

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def log(msg: str) -> None:
    rank = int(os.environ.get("RANK", "0"))
    if rank == 0:
        print(f"[Test rank0] {msg}", flush=True)


def gpu_used_mb() -> float:
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    return (total - free) / (1024**2)


def warm_up_collectives_per_group(groups: dict) -> None:
    """Issue one allreduce per named group so each ProcessGroup's underlying
    NCCL communicator gets lazily initialized. Without this, ``_comm_ptr()``
    returns 0 and neither Method A nor Method B can extract handles.
    """
    for name, group in groups.items():
        if group is None:
            continue
        try:
            world = dist.get_world_size(group=group)
        except Exception:
            continue
        if world <= 1:
            # NCCL doesn't allocate a comm for size-1 groups.
            continue
        # Use a 1-MB tensor — small enough that warmup overhead is negligible
        # but large enough to force NCCL to actually pick a real algorithm.
        x = torch.zeros(256 * 1024, dtype=torch.float32, device="cuda")
        dist.all_reduce(x, group=group)
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    tp_size = int(os.environ.get("TP_SIZE", "2"))
    pp_size = int(os.environ.get("PP_SIZE", "2"))
    cp_size = int(os.environ.get("CP_SIZE", "1"))
    expected_dp = world_size // (tp_size * pp_size * cp_size)

    log(f"world_size={world_size}, TP={tp_size}, PP={pp_size}, CP={cp_size}, DP={expected_dp}")

    # 1. Sanity: NCCL has the suspend/resume API.
    from verl.utils.nccl_suspend import _get_nccl_lib  # noqa: PLC0415

    if _get_nccl_lib() is None:
        log("SKIP: libnccl.so.2 missing ncclCommSuspend (need NCCL >= 2.29.7)")
        dist.destroy_process_group()
        return 0

    # 2. Initialize Megatron parallel_state. We import lazily so the test file
    #    can be parsed without megatron-core installed.
    try:
        from megatron.core import parallel_state as ps  # noqa: PLC0415
    except ImportError as e:
        log(f"SKIP: megatron.core not importable: {e}")
        dist.destroy_process_group()
        return 0

    ps.initialize_model_parallel(
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=pp_size,
        context_parallel_size=cp_size,
    )
    log("Megatron parallel_state initialized")

    # 3. Warm up every named group so _comm_ptr() returns non-zero.
    candidate_groups = {
        "tp": ps.get_tensor_model_parallel_group(),
        "dp": ps.get_data_parallel_group(),
        "pp": ps.get_pipeline_model_parallel_group(),
        "model_parallel": ps.get_model_parallel_group(),
        "tp_dp": ps.get_tensor_and_data_parallel_group(),
    }
    warm_up_collectives_per_group(candidate_groups)
    log("All candidate groups warmed up")

    # ---------------------------------------------------------------- Method A
    from verl.utils.nccl_suspend import (  # noqa: PLC0415
        _collect_megatron_comms,
        resume_training_comms_megatron,
        suspend_training_comms_megatron,
    )

    handles_a = _collect_megatron_comms()
    log(f"Method A discovered {len(handles_a)} comms: {[n for n, _ in handles_a]}")
    assert len(handles_a) > 0, "Method A must find at least one warm comm"

    # 4. Suspend via Method A and check we release real memory.
    mem_before = gpu_used_mb()
    t0 = time.perf_counter()
    ok = suspend_training_comms_megatron()
    suspend_ms = (time.perf_counter() - t0) * 1000.0
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    mem_after = gpu_used_mb()
    freed = mem_before - mem_after
    log(
        f"Method A suspend: ok={ok}, freed={freed:.0f} MB ({mem_before:.0f} -> {mem_after:.0f} MB), {suspend_ms:.0f} ms"
    )
    assert ok, "Method A suspend reported failure"
    assert freed > 100.0, f"Method A freed only {freed:.0f} MB (expected >>100)"

    # Idempotency: a second suspend should be a no-op.
    ok_again = suspend_training_comms_megatron()
    assert not ok_again, "Method A should be idempotent (second suspend = no-op)"

    # 5. Resume via Method A.
    mem_before_resume = gpu_used_mb()
    t0 = time.perf_counter()
    ok = resume_training_comms_megatron()
    resume_ms = (time.perf_counter() - t0) * 1000.0
    torch.cuda.synchronize()
    mem_after_resume = gpu_used_mb()
    reclaimed = mem_after_resume - mem_before_resume
    log(
        f"Method A resume: ok={ok}, reclaimed={reclaimed:.0f} MB "
        f"({mem_before_resume:.0f} -> {mem_after_resume:.0f} MB), {resume_ms:.0f} ms"
    )
    assert ok, "Method A resume reported failure"

    # 6. Post-resume sanity: collectives must still work.
    x = torch.ones(1024, dtype=torch.float32, device="cuda")
    dist.all_reduce(x, group=ps.get_tensor_model_parallel_group())
    torch.cuda.synchronize()
    log("Post-resume allreduce on TP group succeeded")

    # ---------------------------------------------------------------- Method B
    from verl.utils.nccl_suspend import (  # noqa: PLC0415
        _scan_warm_training_comms,
        resume_training_comms,
        suspend_training_comms,
    )

    handles_b = _scan_warm_training_comms()
    log(f"Method B discovered {len(handles_b)} comms: {[n for n, _ in handles_b]}")

    handles_a_set = {h for _, h in handles_a}
    handles_b_set = {h for _, h in handles_b}
    only_a = handles_a_set - handles_b_set
    only_b = handles_b_set - handles_a_set
    common = handles_a_set & handles_b_set
    log(
        f"Comm handle overlap: |A|={len(handles_a_set)}, |B|={len(handles_b_set)}, "
        f"|A∩B|={len(common)}, |A\\B|={len(only_a)}, |B\\A|={len(only_b)}"
    )
    # Method A is a subset of what Method B sees: every parallel_state group is
    # also a torch.distributed PG and therefore in pg_map. Method B may pick up
    # additional comms (e.g. the default world group) that A correctly skips.
    assert handles_a_set.issubset(handles_b_set), (
        f"Expected Method A handles to be a subset of Method B's; unique-to-A: {only_a}"
    )

    # 7. Run Method B for completeness and confirm it can suspend/resume
    #    independently of Method A's earlier cycle.
    ok = suspend_training_comms()
    assert ok, "Method B suspend reported failure"
    ok = resume_training_comms()
    assert ok, "Method B resume reported failure"
    log("Method B suspend/resume cycle completed")

    # 8. Final sanity: more collectives after both A and B cycles.
    for _ in range(3):
        x = torch.ones(1024, dtype=torch.float32, device="cuda")
        dist.all_reduce(x, group=ps.get_data_parallel_group())
    torch.cuda.synchronize()
    log("Post-cycle DP allreduces succeeded")

    log("PASS")
    ps.destroy_model_parallel()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
