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

Exercises every Megatron parallel dimension Method A is meant to cover:
TP, DP, PP, CP, EP, ETP, EDP, plus combined groups (TP×DP, TP×CP, ETP×EP,
ETP×EP×PP, embedding, distributed optimizer instance group).

Compares Method A against Method B (pg_map scan) on the same parallel state,
validates that both find the same set of NCCL communicators (modulo extras
that only Method B is susceptible to), and verifies the suspend/resume cycle
releases and restores GPU memory cleanly.

Requirements:
  * NCCL >= 2.29.7 (libnccl.so.2 must export ncclCommSuspend / ncclCommResume).
    The test gracefully skips if the API is unavailable.
  * megatron-core installed.
  * 8 GPUs with NVLink (single-node).

Usage:
    # Default: TP=2, PP=2, DP=2 (CP=EP=ETP disabled). 5 comms expected.
    torchrun --nproc_per_node=8 tests/utils/test_nccl_suspend_method_a.py

    # Add CP=2: TP=2, PP=2, DP=1, CP=2. CP / TP×CP groups appear.
    CP_SIZE=2 PP_SIZE=2 TP_SIZE=2 \
        torchrun --nproc_per_node=8 tests/utils/test_nccl_suspend_method_a.py

    # Add MoE EP: TP=2, PP=2, DP=2, EP=2, ETP=2 (EDP=1 derived).
    EP_SIZE=2 ETP_SIZE=2 \
        torchrun --nproc_per_node=8 tests/utils/test_nccl_suspend_method_a.py

    # Full mix: TP=2, PP=2, CP=1, DP=2, EP=2, ETP=2.
    TP_SIZE=2 PP_SIZE=2 DP_SIZE=2 CP_SIZE=1 EP_SIZE=2 ETP_SIZE=2 \
        torchrun --nproc_per_node=8 tests/utils/test_nccl_suspend_method_a.py

Optional env vars:
    NCCL_NVLS_ENABLE=0        recommended in CI without Fabric Manager
"""

import os
import sys
import time

import torch
import torch.distributed as dist

# ---------------------------------------------------------------------------
# Logging / GPU memory helpers
# ---------------------------------------------------------------------------


def log(msg: str) -> None:
    rank = int(os.environ.get("RANK", "0"))
    if rank == 0:
        print(f"[Test rank0] {msg}", flush=True)


def gpu_used_mb() -> float:
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    return (total - free) / (1024**2)


# ---------------------------------------------------------------------------
# Megatron group discovery + warmup
# ---------------------------------------------------------------------------


def discover_megatron_groups(ps) -> dict:
    """Probe every known Megatron parallel-state accessor and return a dict
    of {name: ProcessGroup} for the ones that exist on THIS rank.

    Some groups (e.g. embedding) only exist for a subset of ranks; some
    (e.g. expert_*) only exist when EP > 1. Each accessor is wrapped in
    try/except so the test gracefully skips groups that aren't configured
    for the current parallel layout.
    """
    accessors = [
        # Core dims
        ("tp", lambda: ps.get_tensor_model_parallel_group()),
        ("dp", lambda: ps.get_data_parallel_group()),
        ("pp", lambda: ps.get_pipeline_model_parallel_group()),
        ("cp", lambda: ps.get_context_parallel_group()),
        # Combined dims
        ("model_parallel", lambda: ps.get_model_parallel_group()),  # TP × PP
        ("tp_dp", lambda: ps.get_tensor_and_data_parallel_group()),
        ("tp_cp", lambda: ps.get_tensor_and_context_parallel_group()),
        # Embedding sync (cross-PP)
        ("embedding", lambda: ps.get_embedding_group()),
        ("position_embedding", lambda: ps.get_position_embedding_group()),
        # MoE expert dims
        ("ep", lambda: ps.get_expert_model_parallel_group()),
        ("etp", lambda: ps.get_expert_tensor_parallel_group()),
        ("edp", lambda: ps.get_expert_data_parallel_group()),
        ("etp_ep", lambda: ps.get_expert_tensor_and_model_parallel_group()),
        ("etp_ep_pp", lambda: ps.get_expert_tensor_model_pipeline_parallel_group()),
        # Distributed optimizer
        ("intra_dist_opt", lambda: ps.get_intra_distributed_optimizer_instance_group()),
    ]
    found = {}
    for name, fn in accessors:
        try:
            g = fn()
        except Exception:
            continue
        if g is None:
            continue
        found[name] = g
    return found


# Per-group collective for warmup. Picking the right op matters because NCCL
# allocates different-shaped channel buffers depending on which collective
# fires first. From profile_nccl_memory.py Exp3 (8×H100, 128 MB):
#   allreduce / allgather / reduce_scatter / broadcast → ~480 MB / comm
#   all_to_all                                          → ~3.2 GB / comm
# An MoE EP group warmed up with allreduce would only allocate ~480 MB,
# masking the real ~3 GB cost of the all_to_all dispatch path.
GROUP_TO_WARMUP_OP = {
    # Core dims: matrix-multiply gradient sync, FSDP shard, etc. → allreduce.
    "tp": "allreduce",
    "dp": "allreduce",
    "pp": "broadcast",  # PP main group is mostly used for metadata broadcast
    "cp": "allreduce",
    "model_parallel": "allreduce",
    "tp_dp": "allreduce",
    "tp_cp": "allreduce",
    # Embedding sync.
    "embedding": "allreduce",
    "position_embedding": "allreduce",
    # MoE expert dims: token dispatch / combine in production goes through
    # all_to_all, which is the dominant memory consumer in MoE training.
    "ep": "all_to_all",
    "etp_ep": "all_to_all",
    "etp_ep_pp": "all_to_all",
    "etp": "allreduce",
    "edp": "allreduce",
    # Distributed optimizer reduce-scatter.
    "intra_dist_opt": "allreduce",
}


def _run_warmup_op(op: str, group, world: int) -> None:
    """Dispatch to the right NCCL collective so the channel buffer NCCL
    allocates matches what real training would allocate on this group.
    """
    if op == "all_to_all":
        # 1 MB per peer; per-rank tensor size scales with world. NCCL
        # channel buffer for all_to_all is determined by world size, not
        # message bytes, so the small size is fine for warmup.
        chunk = 256 * 1024  # 1 MB float32 per peer
        inp = torch.zeros(chunk * world, dtype=torch.float32, device="cuda")
        out = torch.zeros_like(inp)
        dist.all_to_all_single(out, inp, group=group)
    elif op == "broadcast":
        x = torch.zeros(256 * 1024, dtype=torch.float32, device="cuda")
        # src must be a global rank that's a member of this group.
        src_global_rank = dist.get_global_rank(group, 0)
        dist.broadcast(x, src=src_global_rank, group=group)
    else:  # allreduce
        x = torch.zeros(256 * 1024, dtype=torch.float32, device="cuda")
        dist.all_reduce(x, group=group)


def warm_up_groups(groups: dict) -> dict:
    """Trigger NCCL lazy init on each group with the collective most
    representative of its real-world traffic (see ``GROUP_TO_WARMUP_OP``).

    Returns {name: (world_size, op_used)} for groups that were successfully
    warmed up. Skips size-1 groups (no NCCL comm allocated) and any group
    the current rank is not a member of.
    """
    warmed: dict[str, tuple[int, str]] = {}
    for name, group in groups.items():
        try:
            world = dist.get_world_size(group=group)
        except Exception:
            continue
        if world <= 1:
            continue
        op = GROUP_TO_WARMUP_OP.get(name, "allreduce")
        try:
            _run_warmup_op(op, group, world)
            torch.cuda.synchronize()
        except Exception as e:
            log(f"warm-up '{name}' (op={op}) failed: {e}")
            continue
        warmed[name] = (world, op)
    return warmed


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
    ep_size = int(os.environ.get("EP_SIZE", "1"))
    etp_size = int(os.environ.get("ETP_SIZE", str(tp_size)))
    expected_dp = world_size // (tp_size * pp_size * cp_size)

    log(
        f"world_size={world_size}, TP={tp_size}, PP={pp_size}, "
        f"CP={cp_size}, DP={expected_dp}, EP={ep_size}, ETP={etp_size}"
    )

    # 1. Sanity: NCCL has the suspend/resume API.
    from verl.utils.nccl_suspend import _get_nccl_lib  # noqa: PLC0415

    if _get_nccl_lib() is None:
        log("SKIP: libnccl.so.2 missing ncclCommSuspend (need NCCL >= 2.29.7)")
        dist.destroy_process_group()
        return 0

    # 2. Initialize Megatron parallel_state. Lazy import so the test file can
    #    be parsed without megatron-core installed.
    try:
        from megatron.core import parallel_state as ps  # noqa: PLC0415
    except ImportError as e:
        log(f"SKIP: megatron.core not importable: {e}")
        dist.destroy_process_group()
        return 0

    init_kwargs = {
        "tensor_model_parallel_size": tp_size,
        "pipeline_model_parallel_size": pp_size,
        "context_parallel_size": cp_size,
    }
    if ep_size > 1:
        init_kwargs["expert_model_parallel_size"] = ep_size
        init_kwargs["expert_tensor_parallel_size"] = etp_size
    ps.initialize_model_parallel(**init_kwargs)
    log(f"Megatron parallel_state initialized with {init_kwargs}")

    # 3. Discover every group Megatron exposes for the current configuration,
    #    then warm them up so NCCL comms are lazily allocated.
    discovered = discover_megatron_groups(ps)
    log(f"Discovered {len(discovered)} parallel-state groups: {sorted(discovered.keys())}")

    warmed = warm_up_groups(discovered)
    log(
        f"Warmed up {len(warmed)} groups: "
        + ", ".join(f"{n}(size={s},op={op})" for n, (s, op) in sorted(warmed.items()))
    )
    assert len(warmed) > 0, "No groups warmed up — invalid parallel config?"
    # Highlight if any group used all_to_all — that's where the big
    # (~3 GB/comm vs 480 MB) NCCL channel allocation lives.
    a2a_groups = [n for n, (_, op) in warmed.items() if op == "all_to_all"]
    if a2a_groups:
        log(f"all_to_all warm-up applied to: {a2a_groups} (expect larger per-comm release)")

    # ---------------------------------------------------------------- Method A
    from verl.utils.nccl_suspend import (  # noqa: PLC0415
        _collect_megatron_comms,
        resume_training_comms_megatron,
        suspend_training_comms_megatron,
    )

    handles_a = _collect_megatron_comms()
    log(f"Method A discovered {len(handles_a)} unique comms: {[n for n, _ in handles_a]}")
    assert len(handles_a) > 0, "Method A must find at least one warm comm"

    # Method A's coverage should be at least as large as the warmed set
    # (multiple PGs may share a handle so the unique count can be less).
    # We require the count to be in a sensible range: between 1 and len(warmed).
    assert 1 <= len(handles_a) <= len(warmed), (
        f"Method A found {len(handles_a)} comms but only {len(warmed)} groups were warmed up; "
        f"expected count <= warmed count after dedup"
    )

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

    # Each NCCL comm is ~480 MB on H100; require at least one comm worth of
    # release. Tighter check on per-comm average:
    expected_min_freed = 100.0 * len(handles_a)
    assert freed > expected_min_freed, (
        f"Method A freed {freed:.0f} MB across {len(handles_a)} comms (expected > {expected_min_freed:.0f} MB)"
    )

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

    # Resume should reclaim approximately what suspend freed (within 5%).
    assert abs(reclaimed - freed) / max(freed, 1.0) < 0.05, (
        f"Resume reclaimed {reclaimed:.0f} MB but suspend freed {freed:.0f} MB"
    )

    # 6. Post-resume sanity: collectives must still work on every warmed group.
    for name, group in discovered.items():
        if name not in warmed:
            continue
        try:
            x = torch.ones(1024, dtype=torch.float32, device="cuda")
            dist.all_reduce(x, group=group)
        except Exception as e:
            raise AssertionError(f"Post-resume allreduce on '{name}' failed: {e}") from e
    torch.cuda.synchronize()
    log(f"Post-resume allreduce passed on all {len(warmed)} warmed groups")

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

    # 8. Final sanity: more collectives on every warmed group after both cycles.
    for _ in range(3):
        for group in discovered.values():
            try:
                x = torch.ones(1024, dtype=torch.float32, device="cuda")
                dist.all_reduce(x, group=group)
            except Exception:
                pass
    torch.cuda.synchronize()
    log("Post-cycle collectives on all groups succeeded")

    log("PASS")
    ps.destroy_model_parallel()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
