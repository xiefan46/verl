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
Profile NCCL communicator memory usage under different conditions.

Systematically measures how NCCL internal memory is affected by:
  Exp 1: Group size (2/4/6/8 GPUs) — fixed allreduce, 256MB msg
  Exp 2: Message size (4KB → 1GB) — fixed 8 GPU allreduce
  Exp 3: Collective type (allreduce/allgather/reduce_scatter/all_to_all/broadcast/p2p)
  Exp 4: Number of groups (1/2/4/8/16) — fixed 8 GPU allreduce
  Exp 5: Warmup rounds (1/3/10/30) — does NCCL allocate more over time?
  Exp 6: NCCL_BUFFSIZE effect — env var controls channel buffer size

Each experiment measures:
  - NCCL memory per comm (gpu_used delta)
  - Suspend freed amount
  - Suspend/resume latency

Usage:
    torchrun --nproc_per_node=8 tests/utils/profile_nccl_memory.py
    torchrun --nproc_per_node=2 tests/utils/profile_nccl_memory.py  # subset of experiments

Results are printed as tables for easy copy-paste into research notes.
"""

import os
import sys
import time

import torch
import torch.distributed as dist

RANK = None
WORLD_SIZE = None

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def log(msg=""):
    if RANK == 0:
        print(msg, flush=True)


def gpu_used_mb():
    """Driver-level GPU memory used (includes NCCL's cudaMalloc)."""
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    return (total - free) / 1024**2


def extract_comm(pg):
    try:
        backend = pg._get_backend(torch.device("cuda"))
        if hasattr(backend, "_comm_ptr"):
            return backend._comm_ptr()
    except Exception:
        pass
    return None


def do_warmup(pg, op, msg_elements, rounds, rank, world_size):
    """Run warmup operations on a process group.

    Args:
        pg: ProcessGroup (None = default group)
        op: one of "allreduce", "allgather", "reduce_scatter", "all_to_all",
            "broadcast", "p2p"
        msg_elements: number of float32 elements per tensor
        rounds: number of warmup iterations
        rank: current rank
        world_size: group world size
    """
    for _ in range(rounds):
        if op == "allreduce":
            buf = torch.randn(msg_elements, device="cuda")
            dist.all_reduce(buf, group=pg)
            del buf

        elif op == "allgather":
            chunk = msg_elements // world_size
            inp = torch.randn(chunk, device="cuda")
            out_list = [torch.empty(chunk, device="cuda") for _ in range(world_size)]
            dist.all_gather(out_list, inp, group=pg)
            del inp, out_list

        elif op == "reduce_scatter":
            chunk = msg_elements // world_size
            inp = torch.randn(chunk * world_size, device="cuda")
            out = torch.empty(chunk, device="cuda")
            dist.reduce_scatter_tensor(out, inp, group=pg)
            del inp, out

        elif op == "all_to_all":
            per_rank = msg_elements // world_size
            inp = torch.randn(per_rank * world_size, device="cuda")
            out = torch.empty_like(inp)
            dist.all_to_all_single(out, inp, group=pg)
            del inp, out

        elif op == "broadcast":
            buf = torch.randn(msg_elements, device="cuda")
            dist.broadcast(buf, src=0, group=pg)
            del buf

        elif op == "p2p":
            send_buf = torch.randn(msg_elements, device="cuda")
            recv_buf = torch.empty_like(send_buf)
            # Get group ranks
            group_ranks = dist.get_process_group_ranks(pg) if pg is not None else list(range(world_size))
            my_idx = group_ranks.index(rank) if rank in group_ranks else -1
            if my_idx >= 0:
                dst_idx = (my_idx + 1) % len(group_ranks)
                src_idx = (my_idx - 1) % len(group_ranks)
                dst_rank = group_ranks[dst_idx]
                src_rank = group_ranks[src_idx]
                s = dist.isend(send_buf, dst_rank, group=pg)
                r = dist.irecv(recv_buf, src_rank, group=pg)
                s.wait()
                r.wait()
            del send_buf, recv_buf

        else:
            raise ValueError(f"Unknown op: {op}")

    torch.cuda.synchronize()


def measure_group(pg, op, msg_elements, warmup_rounds, group_size, suspend_fn, resume_fn):
    """Create, warmup, and measure a single group's NCCL memory.

    Returns dict with nccl_mem_mb, freed_mb, suspend_ms, resume_ms.
    Returns None if this rank is not in the group.
    """
    comm = extract_comm(pg)
    if comm is None:
        return None

    group_ranks = dist.get_process_group_ranks(pg)
    if RANK not in group_ranks:
        return None

    # Warmup
    torch.cuda.empty_cache()
    before = gpu_used_mb()
    do_warmup(pg, op, msg_elements, warmup_rounds, RANK, group_size)
    torch.cuda.empty_cache()
    after = gpu_used_mb()
    nccl_mem = after - before

    # Suspend
    dist.barrier(group=pg)
    torch.cuda.synchronize()
    pre_suspend = gpu_used_mb()
    dist.barrier(group=pg)

    t0 = time.perf_counter()
    suspend_fn(comm)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    post_suspend = gpu_used_mb()
    freed = pre_suspend - post_suspend
    suspend_ms = (t1 - t0) * 1000

    # Resume
    t2 = time.perf_counter()
    resume_fn(comm)
    torch.cuda.synchronize()
    t3 = time.perf_counter()
    resume_ms = (t3 - t2) * 1000

    return {
        "nccl_mem_mb": nccl_mem,
        "freed_mb": freed,
        "suspend_ms": suspend_ms,
        "resume_ms": resume_ms,
    }


def print_table(headers, rows, title=""):
    """Print a formatted table."""
    if title:
        log(f"\n{'=' * 70}")
        log(title)
        log(f"{'=' * 70}")

    # Calculate column widths
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    # Header
    header_line = " | ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers))
    log(header_line)
    log("-+-".join("-" * w for w in widths))

    # Rows
    for row in rows:
        line = " | ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row))
        log(line)
    log()


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

def exp_group_size(suspend_fn, resume_fn):
    """Exp 1: How does group size affect NCCL memory per comm?"""
    sizes = [s for s in [2, 4, 6, 8] if s <= WORLD_SIZE]
    msg_elements = 64 * 1024 * 1024  # 256 MB float32
    warmup = 5

    rows = []
    for gs in sizes:
        group_ranks = list(range(gs))
        pg = dist.new_group(ranks=group_ranks)

        if RANK in group_ranks:
            # Need baseline before warmup for this group
            torch.cuda.empty_cache()
            baseline = gpu_used_mb()
            do_warmup(pg, "allreduce", msg_elements, warmup, RANK, gs)
            torch.cuda.empty_cache()
            after_warmup = gpu_used_mb()
            nccl_mem = after_warmup - baseline

            comm = extract_comm(pg)
            # Suspend/resume
            dist.barrier(group=pg)
            torch.cuda.synchronize()
            pre = gpu_used_mb()
            dist.barrier(group=pg)
            t0 = time.perf_counter()
            suspend_fn(comm)
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            post = gpu_used_mb()
            freed = pre - post
            suspend_ms = (t1 - t0) * 1000
            t2 = time.perf_counter()
            resume_fn(comm)
            torch.cuda.synchronize()
            t3 = time.perf_counter()
            resume_ms = (t3 - t2) * 1000

            if RANK == 0:
                rows.append([
                    gs,
                    f"{nccl_mem:.1f}",
                    f"{freed:.1f}",
                    f"{nccl_mem - freed:.1f}" if nccl_mem > 0 else "N/A",
                    f"{suspend_ms:.1f}",
                    f"{resume_ms:.1f}",
                ])
        dist.barrier()  # sync all ranks

    print_table(
        ["group_size", "nccl_mem(MB)", "freed(MB)", "persist(MB)", "suspend(ms)", "resume(ms)"],
        rows,
        "Exp 1: NCCL memory vs group size (allreduce, 256MB msg)",
    )


def exp_message_size(suspend_fn, resume_fn):
    """Exp 2: How does message size affect NCCL memory?"""
    # Sizes in float32 elements: 1K(4KB) → 256M(1GB)
    sizes = [
        (1024, "4KB"),
        (64 * 1024, "256KB"),
        (1024 * 1024, "4MB"),
        (16 * 1024 * 1024, "64MB"),
        (64 * 1024 * 1024, "256MB"),
        (256 * 1024 * 1024, "1GB"),
    ]
    warmup = 5

    rows = []
    for elements, label in sizes:
        pg = dist.new_group(ranks=list(range(WORLD_SIZE)))
        torch.cuda.empty_cache()
        baseline = gpu_used_mb()
        do_warmup(pg, "allreduce", elements, warmup, RANK, WORLD_SIZE)
        torch.cuda.empty_cache()
        after = gpu_used_mb()
        nccl_mem = after - baseline

        comm = extract_comm(pg)
        dist.barrier()
        torch.cuda.synchronize()
        pre = gpu_used_mb()
        dist.barrier()
        t0 = time.perf_counter()
        suspend_fn(comm)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        freed = pre - gpu_used_mb()
        resume_fn(comm)
        torch.cuda.synchronize()
        resume_ms = (time.perf_counter() - t1) * 1000

        if RANK == 0:
            rows.append([
                label,
                f"{elements}",
                f"{nccl_mem:.1f}",
                f"{freed:.1f}",
                f"{(t1 - t0) * 1000:.1f}",
                f"{resume_ms:.1f}",
            ])
        dist.barrier()

    print_table(
        ["msg_size", "elements", "nccl_mem(MB)", "freed(MB)", "suspend(ms)", "resume(ms)"],
        rows,
        f"Exp 2: NCCL memory vs message size (allreduce, {WORLD_SIZE} GPUs)",
    )


def exp_collective_type(suspend_fn, resume_fn):
    """Exp 3: How does collective type affect NCCL memory?"""
    ops = ["allreduce", "allgather", "reduce_scatter", "all_to_all", "broadcast", "p2p"]
    msg_elements = 64 * 1024 * 1024  # 256 MB
    warmup = 5

    rows = []
    for op in ops:
        pg = dist.new_group(ranks=list(range(WORLD_SIZE)))
        torch.cuda.empty_cache()
        baseline = gpu_used_mb()
        do_warmup(pg, op, msg_elements, warmup, RANK, WORLD_SIZE)
        torch.cuda.empty_cache()
        after = gpu_used_mb()
        nccl_mem = after - baseline

        comm = extract_comm(pg)
        dist.barrier()
        torch.cuda.synchronize()
        pre = gpu_used_mb()
        dist.barrier()
        t0 = time.perf_counter()
        suspend_fn(comm)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        freed = pre - gpu_used_mb()
        t2 = time.perf_counter()
        resume_fn(comm)
        torch.cuda.synchronize()
        t3 = time.perf_counter()

        if RANK == 0:
            rows.append([
                op,
                f"{nccl_mem:.1f}",
                f"{freed:.1f}",
                f"{nccl_mem - freed:.1f}" if nccl_mem > 0 else "N/A",
                f"{(t1 - t0) * 1000:.1f}",
                f"{(t3 - t2) * 1000:.1f}",
            ])
        dist.barrier()

    print_table(
        ["collective", "nccl_mem(MB)", "freed(MB)", "persist(MB)", "suspend(ms)", "resume(ms)"],
        rows,
        f"Exp 3: NCCL memory vs collective type ({WORLD_SIZE} GPUs, 256MB msg)",
    )


def exp_num_groups(suspend_fn, resume_fn):
    """Exp 4: How does number of groups affect total NCCL memory?"""
    counts = [1, 2, 4, 8, 16]
    msg_elements = 64 * 1024 * 1024  # 256 MB
    warmup = 3

    rows = []
    for n in counts:
        torch.cuda.empty_cache()
        baseline = gpu_used_mb()

        pgs = []
        comms = []
        for i in range(n):
            pg = dist.new_group(ranks=list(range(WORLD_SIZE)))
            pgs.append(pg)
            do_warmup(pg, "allreduce", msg_elements, warmup, RANK, WORLD_SIZE)
            comm = extract_comm(pg)
            if comm is not None:
                comms.append((f"g{i}", comm))

        torch.cuda.empty_cache()
        after = gpu_used_mb()
        total_nccl = after - baseline
        per_group = total_nccl / n if n > 0 else 0

        # Suspend all
        dist.barrier()
        torch.cuda.synchronize()
        pre = gpu_used_mb()
        dist.barrier()
        t0 = time.perf_counter()
        for _, c in comms:
            suspend_fn(c)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        freed = pre - gpu_used_mb()
        t2 = time.perf_counter()
        for _, c in comms:
            resume_fn(c)
        torch.cuda.synchronize()
        t3 = time.perf_counter()

        if RANK == 0:
            rows.append([
                n,
                f"{total_nccl:.1f}",
                f"{per_group:.1f}",
                f"{freed:.1f}",
                f"{(t1 - t0) * 1000:.1f}",
                f"{(t3 - t2) * 1000:.1f}",
            ])
        dist.barrier()

    print_table(
        ["num_groups", "total_nccl(MB)", "per_group(MB)", "freed(MB)", "suspend(ms)", "resume(ms)"],
        rows,
        f"Exp 4: NCCL memory vs number of groups ({WORLD_SIZE} GPUs, allreduce 256MB)",
    )


def exp_warmup_rounds(suspend_fn, resume_fn):
    """Exp 5: Does NCCL allocate more memory with more warmup rounds?"""
    rounds_list = [1, 3, 10, 30, 100]
    msg_elements = 64 * 1024 * 1024  # 256 MB

    rows = []
    for rounds in rounds_list:
        pg = dist.new_group(ranks=list(range(WORLD_SIZE)))
        torch.cuda.empty_cache()
        baseline = gpu_used_mb()
        do_warmup(pg, "allreduce", msg_elements, rounds, RANK, WORLD_SIZE)
        torch.cuda.empty_cache()
        after = gpu_used_mb()
        nccl_mem = after - baseline

        comm = extract_comm(pg)
        dist.barrier()
        torch.cuda.synchronize()
        pre = gpu_used_mb()
        dist.barrier()
        suspend_fn(comm)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        freed = pre - gpu_used_mb()
        resume_fn(comm)
        torch.cuda.synchronize()

        if RANK == 0:
            rows.append([rounds, f"{nccl_mem:.1f}", f"{freed:.1f}"])
        dist.barrier()

    print_table(
        ["warmup_rounds", "nccl_mem(MB)", "freed(MB)"],
        rows,
        f"Exp 5: NCCL memory vs warmup rounds ({WORLD_SIZE} GPUs, allreduce 256MB)",
    )


def exp_mixed_ops(suspend_fn, resume_fn):
    """Exp 6: Memory when a single group is used for MULTIPLE collective types.

    In real training, the same process group may be used for allreduce (gradient sync),
    allgather (FSDP unshard), reduce_scatter (FSDP shard), etc. This tests whether
    using more op types on a single comm increases its memory footprint.
    """
    msg_elements = 64 * 1024 * 1024  # 256 MB
    warmup = 3

    op_combos = [
        (["allreduce"], "allreduce only"),
        (["allreduce", "broadcast"], "allreduce+broadcast"),
        (["allreduce", "allgather", "reduce_scatter"], "allreduce+allgather+reduce_scatter (FSDP-like)"),
        (["allreduce", "allgather", "reduce_scatter", "all_to_all"], "+all_to_all (EP-like)"),
        (["allreduce", "allgather", "reduce_scatter", "all_to_all", "p2p"], "+p2p (full mix)"),
    ]

    rows = []
    for ops, label in op_combos:
        pg = dist.new_group(ranks=list(range(WORLD_SIZE)))
        torch.cuda.empty_cache()
        baseline = gpu_used_mb()
        for op in ops:
            do_warmup(pg, op, msg_elements, warmup, RANK, WORLD_SIZE)
        torch.cuda.empty_cache()
        after = gpu_used_mb()
        nccl_mem = after - baseline

        comm = extract_comm(pg)
        dist.barrier()
        torch.cuda.synchronize()
        pre = gpu_used_mb()
        dist.barrier()
        t0 = time.perf_counter()
        suspend_fn(comm)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        freed = pre - gpu_used_mb()
        resume_fn(comm)
        torch.cuda.synchronize()

        if RANK == 0:
            rows.append([
                label,
                f"{nccl_mem:.1f}",
                f"{freed:.1f}",
                f"{nccl_mem - freed:.1f}" if nccl_mem > 0 else "N/A",
                f"{(t1 - t0) * 1000:.1f}",
            ])
        dist.barrier()

    print_table(
        ["op_mix", "nccl_mem(MB)", "freed(MB)", "persist(MB)", "suspend(ms)"],
        rows,
        f"Exp 6: NCCL memory vs collective mix on SAME group ({WORLD_SIZE} GPUs, 256MB)",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global RANK, WORLD_SIZE

    dist.init_process_group(backend="nccl")
    RANK = dist.get_rank()
    WORLD_SIZE = dist.get_world_size()
    torch.cuda.set_device(RANK)

    log(f"=== NCCL Memory Profiling (world_size={WORLD_SIZE}) ===")
    log(f"GPU: {torch.cuda.get_device_name(0)}")
    log(f"Baseline gpu_used: {gpu_used_mb():.0f} MB")

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from verl.utils.nccl_suspend import _get_nccl_lib, resume_nccl_comm, suspend_nccl_comm

    if _get_nccl_lib() is None:
        log("SKIP: NCCL >= 2.29.7 required")
        dist.destroy_process_group()
        return

    exp_group_size(suspend_nccl_comm, resume_nccl_comm)
    exp_message_size(suspend_nccl_comm, resume_nccl_comm)
    exp_collective_type(suspend_nccl_comm, resume_nccl_comm)
    exp_num_groups(suspend_nccl_comm, resume_nccl_comm)
    exp_warmup_rounds(suspend_nccl_comm, resume_nccl_comm)
    exp_mixed_ops(suspend_nccl_comm, resume_nccl_comm)

    log(f"\n{'=' * 70}")
    log("ALL PROFILING EXPERIMENTS COMPLETE")
    log(f"{'=' * 70}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
