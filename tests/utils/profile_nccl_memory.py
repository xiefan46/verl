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
Profile NCCL communicator memory usage with production-realistic parameters.

Parameters are calibrated to real Megatron/FSDP training workloads:
  - Group sizes match typical TP/EP/DP/PP/CP parallelism dimensions
  - Message sizes match actual activation/gradient/token sizes
  - Collective types match each parallelism dimension's communication pattern
  - Group counts match real Megatron process group configurations

Reference model configs for parameter calibration:
  Qwen2.5-7B:   hidden=3584,  intermediate=18944, num_heads=28,  num_layers=28
  Qwen2.5-32B:  hidden=5120,  intermediate=27648, num_heads=40,  num_layers=64
  Qwen2.5-72B:  hidden=8192,  intermediate=29568, num_heads=64,  num_layers=80
  Qwen3-30B-A3B (MoE): hidden=4096, intermediate=4096, num_experts=128, top_k=8

Typical training micro-batch: batch_size=1-4, seq_len=4096-32768

Experiments:
  Exp 1: Group size — simulates TP/EP/DP group sizes (1/2/4/8)
  Exp 2: Message size — calibrated to real activation/gradient/token sizes
  Exp 3: Collective type — maps to actual parallelism dimensions
  Exp 4: Group count — simulates real Megatron group configurations
  Exp 5: Warmup rounds — how many training steps until NCCL memory stabilizes
  Exp 6: Mixed ops on same group — FSDP allgather+reduce_scatter on same comm

Usage:
    torchrun --nproc_per_node=8 tests/utils/profile_nccl_memory.py
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
    """Driver-level GPU memory used (includes NCCL's cudaMalloc)."""
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    return (total - free) / 1024**2


def clean_baseline():
    """Force GC + empty cache + sync + barrier for accurate baseline measurement."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    dist.barrier()
    return gpu_used_mb()


def clean_measure():
    """Force GC + empty cache + sync for accurate post-warmup measurement."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return gpu_used_mb()


def extract_comm(pg):
    try:
        backend = pg._get_backend(torch.device("cuda"))
        if hasattr(backend, "_comm_ptr"):
            return backend._comm_ptr()
        log(f"  WARNING: backend {type(backend).__name__} has no _comm_ptr")
    except Exception as e:
        log(f"  WARNING: extract_comm failed: {e}")
    return None


def destroy_pg(pg):
    """Destroy a non-default process group and clean up."""
    try:
        dist.destroy_process_group(pg)
    except Exception:
        pass
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def gather_stats(x: float, group=None):
    """Gather a scalar from all ranks in group, return (min, avg, max) on all ranks."""
    t = torch.tensor([x], device="cuda", dtype=torch.float64)
    if group is not None:
        ws = dist.get_world_size(group)
        vals = [torch.zeros_like(t) for _ in range(ws)]
        dist.all_gather(vals, t, group=group)
    else:
        vals = [torch.zeros_like(t) for _ in range(WORLD_SIZE)]
        dist.all_gather(vals, t)
    vals = torch.stack(vals).flatten()
    return vals.min().item(), vals.mean().item(), vals.max().item()


def fmt_stats(mn, avg, mx):
    """Format min/avg/max stats. If spread is tiny, just show avg."""
    if abs(mx - mn) < 1.0:
        return f"{avg:.0f}"
    return f"{mn:.0f}/{avg:.0f}/{mx:.0f}"


def fmt_persist(nccl_avg, freed_avg):
    """Format persist column, marking negative values as noise."""
    p = nccl_avg - freed_avg
    if p < -1:
        return f"{p:.0f}*"
    return f"{max(p, 0):.0f}"


def do_warmup(pg, op, msg_elements, rounds, rank, world_size):
    """Run warmup operations on a process group."""
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
            group_ranks = dist.get_process_group_ranks(pg) if pg is not None else list(range(world_size))
            my_idx = group_ranks.index(rank) if rank in group_ranks else -1
            if my_idx >= 0:
                dst = group_ranks[(my_idx + 1) % len(group_ranks)]
                src = group_ranks[(my_idx - 1) % len(group_ranks)]
                # Use batch_isend_irecv to avoid lazy 2-rank comm creation
                ops = [
                    dist.P2POp(dist.isend, send_buf, dst, group=pg),
                    dist.P2POp(dist.irecv, recv_buf, src, group=pg),
                ]
                reqs = dist.batch_isend_irecv(ops)
                for req in reqs:
                    req.wait()
            del send_buf, recv_buf
        else:
            raise ValueError(f"Unknown op: {op}")
    torch.cuda.synchronize()


def measure_one(pg, suspend_fn, resume_fn):
    """Suspend one comm, measure freed + latency, resume.

    Returns (freed_mb, suspend_api_ms, reclaim_total_ms, resume_ms).
    suspend_api_ms:    pure ncclCommSuspend + cuda sync
    reclaim_total_ms:  suspend + gc + empty_cache + sync (end-to-end reclaim)
    """
    comm = extract_comm(pg)
    if comm is None:
        return 0, 0, 0, 0

    dist.barrier(group=pg)
    torch.cuda.synchronize()
    pre = gpu_used_mb()
    dist.barrier(group=pg)

    # Suspend API latency
    t0 = time.perf_counter()
    suspend_fn(comm)
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    # Reclaim (gc + empty_cache)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    t2 = time.perf_counter()

    freed = pre - gpu_used_mb()

    # Resume
    t3 = time.perf_counter()
    resume_fn(comm)
    torch.cuda.synchronize()
    t4 = time.perf_counter()

    suspend_api_ms = (t1 - t0) * 1000
    reclaim_total_ms = (t2 - t0) * 1000
    resume_ms = (t4 - t3) * 1000
    return freed, suspend_api_ms, reclaim_total_ms, resume_ms


def print_table(headers, rows, title="", footnote=""):
    if title:
        log(f"\n{'=' * 90}")
        log(title)
        log(f"{'=' * 90}")
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    log(" | ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers)))
    log("-+-".join("-" * w for w in widths))
    for row in rows:
        log(" | ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)))
    if footnote:
        log(footnote)
    log()


def cuda_warmup():
    """Dummy warmup to trigger CUDA context / cuRAND / PyTorch lazy init.
    Call once before all experiments to eliminate first-allocation noise."""
    tmp = torch.randn(1024, 1024, device="cuda")
    tmp = tmp @ tmp.T
    del tmp
    # Also trigger a dummy NCCL collective on default group
    buf = torch.ones(256, device="cuda")
    dist.all_reduce(buf)
    del buf
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    dist.barrier()


# ---------------------------------------------------------------------------
# Exp 1: Group size
# ---------------------------------------------------------------------------

def exp_group_size(suspend_fn, resume_fn):
    sizes = [s for s in [2, 4, 8] if s <= WORLD_SIZE]
    msg_elements = 2 * 4096 * 8192  # 256 MB float32
    warmup = 10

    rows = []
    for gs in sizes:
        group_ranks = list(range(gs))
        pg = dist.new_group(ranks=group_ranks)

        # Global barrier must be outside if-block to avoid sync mismatch
        baseline = clean_baseline()

        if RANK in group_ranks:
            do_warmup(pg, "allreduce", msg_elements, warmup, RANK, gs)
            nccl_mem = clean_measure() - baseline

            freed, sus_api, sus_total, res_ms = measure_one(pg, suspend_fn, resume_fn)

            # Gather stats within the subgroup
            nccl_stats = gather_stats(nccl_mem, group=pg)
            freed_stats = gather_stats(freed, group=pg)

            real_world = {2: "TP=2 or DP=2", 4: "TP=4 (Qwen-32B)", 8: "TP=8 (Qwen-72B)"}
            if RANK == 0:
                rows.append([
                    gs, real_world.get(gs, ""),
                    fmt_stats(*nccl_stats), fmt_stats(*freed_stats),
                    fmt_persist(nccl_stats[1], freed_stats[1]),
                    f"{sus_api:.0f}", f"{sus_total:.0f}", f"{res_ms:.0f}",
                ])
        dist.barrier()
        destroy_pg(pg)

    print_table(
        ["grp_sz", "real_world", "nccl(MB)", "freed(MB)", "persist(MB)",
         "sus_api(ms)", "sus_total(ms)", "res(ms)"],
        rows,
        "Exp 1: Group size (allreduce, batch=2 x seq=4096 x hidden=8192 = 256MB)",
        "  nccl/freed: min/avg/max across ranks in group. * = measurement noise.",
    )


# ---------------------------------------------------------------------------
# Exp 2: Message size
# ---------------------------------------------------------------------------

def exp_message_size(suspend_fn, resume_fn):
    sizes = [
        (1024 * 1024,        "4MB",    "FSDP grad bucket"),
        (8 * 1024 * 1024,    "32MB",   "TP (b=1,s=4k,h=4096)"),
        (20 * 1024 * 1024,   "80MB",   "TP Qwen-32B (b=2,s=4k,h=5120)"),
        (32 * 1024 * 1024,   "128MB",  "TP Qwen-72B (b=2,s=4k,h=8192)"),
        (128 * 1024 * 1024,  "512MB",  "TP long-seq / FSDP Qwen-72B/TP4"),
        (256 * 1024 * 1024,  "1GB",    "MoE all_to_all (b=2,s=4k,top8)"),
        (512 * 1024 * 1024,  "2GB",    "FSDP allgather Qwen-72B no TP"),
    ]
    warmup = 10
    log(f"  Testing {len(sizes)} message sizes ...")

    rows = []
    for elements, label, scenario in sizes:
        assert elements % WORLD_SIZE == 0, f"{label}: {elements} not divisible by {WORLD_SIZE}"
        pg = dist.new_group(ranks=list(range(WORLD_SIZE)))

        baseline = clean_baseline()
        do_warmup(pg, "allreduce", elements, warmup, RANK, WORLD_SIZE)
        nccl_mem = clean_measure() - baseline

        freed, sus_api, sus_total, res_ms = measure_one(pg, suspend_fn, resume_fn)
        nccl_stats = gather_stats(nccl_mem)
        freed_stats = gather_stats(freed)

        if RANK == 0:
            rows.append([label, scenario, fmt_stats(*nccl_stats), fmt_stats(*freed_stats),
                         f"{sus_api:.0f}", f"{res_ms:.0f}"])
            log(f"    {label} done")
        dist.barrier()
        destroy_pg(pg)

    print_table(
        ["msg_size", "scenario", "nccl(MB)", "freed(MB)", "sus_api(ms)", "res(ms)"],
        rows,
        f"Exp 2: Message size ({WORLD_SIZE} GPUs, allreduce)",
        "  nccl/freed: min/avg/max across all ranks.",
    )


# ---------------------------------------------------------------------------
# Exp 3: Collective type
# ---------------------------------------------------------------------------

def exp_collective_type(suspend_fn, resume_fn):
    ops = [
        ("allreduce",      "TP grad sync"),
        ("allgather",      "FSDP unshard / SeqParallel"),
        ("reduce_scatter", "FSDP grad shard"),
        ("all_to_all",     "EP token dispatch (MoE)"),
        ("broadcast",      "PP weight broadcast"),
        ("p2p",            "PP activation transfer"),
    ]
    msg_elements = 2 * 4096 * 4096  # 128 MB float32
    assert msg_elements % WORLD_SIZE == 0
    warmup = 10

    rows = []
    for op, scenario in ops:
        pg = dist.new_group(ranks=list(range(WORLD_SIZE)))

        baseline = clean_baseline()
        do_warmup(pg, op, msg_elements, warmup, RANK, WORLD_SIZE)
        nccl_mem = clean_measure() - baseline

        freed, sus_api, sus_total, res_ms = measure_one(pg, suspend_fn, resume_fn)
        nccl_stats = gather_stats(nccl_mem)
        freed_stats = gather_stats(freed)

        if RANK == 0:
            rows.append([op, scenario, fmt_stats(*nccl_stats), fmt_stats(*freed_stats),
                         fmt_persist(nccl_stats[1], freed_stats[1]),
                         f"{sus_api:.0f}", f"{res_ms:.0f}"])
        dist.barrier()
        destroy_pg(pg)

    print_table(
        ["collective", "parallelism", "nccl(MB)", "freed(MB)", "persist(MB)",
         "sus_api(ms)", "res(ms)"],
        rows,
        f"Exp 3: Collective type ({WORLD_SIZE} GPUs, 128MB msg)",
        "  nccl/freed: min/avg/max across all ranks. * = measurement noise.",
    )


# ---------------------------------------------------------------------------
# Exp 4: Number of groups
# ---------------------------------------------------------------------------

def exp_num_groups(suspend_fn, resume_fn):
    counts = [1, 2, 4, 6, 8, 12]
    msg_elements = 2 * 4096 * 4096  # 128MB
    warmup = 5

    real_world = {
        1: "single TP or DP",
        2: "FSDP (shard+replicate)",
        4: "FSDP+TP",
        6: "Megatron (TP+DP+EP+CP+PP+cross)",
        8: "Megatron + aux groups",
        12: "Megatron 3D+EP+CP all combos",
    }

    log(f"  Testing group counts: {counts} ...")
    rows = []
    for n in counts:
        log(f"    n={n} groups ...")

        baseline = clean_baseline()

        pgs = []
        comms = []
        for i in range(n):
            pg = dist.new_group(ranks=list(range(WORLD_SIZE)))
            pgs.append(pg)
            do_warmup(pg, "allreduce", msg_elements, warmup, RANK, WORLD_SIZE)
            comm = extract_comm(pg)
            if comm is not None:
                comms.append((f"g{i}", comm))

        total_nccl = clean_measure() - baseline
        per_group = total_nccl / n if n > 0 else 0

        # Suspend all
        dist.barrier()
        torch.cuda.synchronize()
        pre = gpu_used_mb()
        dist.barrier()
        t0 = time.perf_counter()
        for _, c in comms:
            suspend_fn(c)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        freed = pre - gpu_used_mb()
        t3 = time.perf_counter()
        for _, c in comms:
            resume_fn(c)
        torch.cuda.synchronize()
        t4 = time.perf_counter()

        total_stats = gather_stats(total_nccl)
        freed_stats = gather_stats(freed)

        if RANK == 0:
            rows.append([
                n, real_world.get(n, ""),
                fmt_stats(*total_stats), f"{per_group:.0f}", fmt_stats(*freed_stats),
                f"{(t1 - t0) * 1000:.0f}", f"{(t4 - t3) * 1000:.0f}",
            ])
        dist.barrier()
        for pg in pgs:
            destroy_pg(pg)

    print_table(
        ["n_grps", "scenario", "total(MB)", "per_grp(MB)", "freed(MB)",
         "sus_api(ms)", "res(ms)"],
        rows,
        f"Exp 4: Number of groups ({WORLD_SIZE} GPUs, allreduce 128MB)",
        "  total/freed: min/avg/max across all ranks. per_grp from rank 0.",
    )


# ---------------------------------------------------------------------------
# Exp 5: Warmup rounds
# ---------------------------------------------------------------------------

def exp_warmup_rounds(suspend_fn, resume_fn):
    rounds_list = [1, 5, 20, 100, 500]
    msg_elements = 2 * 4096 * 4096  # 128MB

    log(f"  Testing warmup rounds: {rounds_list} ...")
    rows = []
    for rounds in rounds_list:
        log(f"    {rounds} rounds ...")
        pg = dist.new_group(ranks=list(range(WORLD_SIZE)))

        baseline = clean_baseline()
        do_warmup(pg, "allreduce", msg_elements, rounds, RANK, WORLD_SIZE)
        nccl_mem = clean_measure() - baseline

        freed, sus_api, sus_total, res_ms = measure_one(pg, suspend_fn, resume_fn)
        nccl_stats = gather_stats(nccl_mem)
        freed_stats = gather_stats(freed)

        if RANK == 0:
            rows.append([rounds, fmt_stats(*nccl_stats), fmt_stats(*freed_stats),
                         f"{sus_api:.0f}", f"{res_ms:.0f}"])
        dist.barrier()
        destroy_pg(pg)

    print_table(
        ["rounds", "nccl(MB)", "freed(MB)", "sus_api(ms)", "res(ms)"],
        rows,
        f"Exp 5: Warmup rounds ({WORLD_SIZE} GPUs, allreduce 128MB)",
        "  nccl/freed: min/avg/max across all ranks.",
    )


# ---------------------------------------------------------------------------
# Exp 6: Mixed ops on same group
# ---------------------------------------------------------------------------

def exp_mixed_ops(suspend_fn, resume_fn):
    msg_elements = 2 * 4096 * 4096  # 128MB
    assert msg_elements % WORLD_SIZE == 0
    warmup = 5

    op_combos = [
        (["allreduce"],
         "allreduce only (TP grad sync)"),
        (["allgather", "reduce_scatter"],
         "allgather+reduce_scatter (FSDP)"),
        (["allreduce", "allgather"],
         "allreduce+allgather (Megatron TP+SeqP)"),
        (["allreduce", "allgather", "reduce_scatter"],
         "allreduce+allgather+reduce_scatter (FSDP+TP)"),
        (["all_to_all", "allreduce"],
         "all_to_all+allreduce (MoE EP)"),
        (["allreduce", "allgather", "reduce_scatter", "all_to_all", "p2p"],
         "all ops (full Megatron 3D+EP+PP)"),
    ]

    rows = []
    for ops, label in op_combos:
        pg = dist.new_group(ranks=list(range(WORLD_SIZE)))

        baseline = clean_baseline()
        for op in ops:
            do_warmup(pg, op, msg_elements, warmup, RANK, WORLD_SIZE)
        nccl_mem = clean_measure() - baseline

        freed, sus_api, sus_total, _ = measure_one(pg, suspend_fn, resume_fn)
        nccl_stats = gather_stats(nccl_mem)
        freed_stats = gather_stats(freed)

        if RANK == 0:
            rows.append([
                label, ", ".join(ops),
                fmt_stats(*nccl_stats), fmt_stats(*freed_stats),
                fmt_persist(nccl_stats[1], freed_stats[1]),
                f"{sus_api:.0f}",
            ])
        dist.barrier()
        destroy_pg(pg)

    print_table(
        ["scenario", "ops", "nccl(MB)", "freed(MB)", "persist(MB)", "sus_api(ms)"],
        rows,
        f"Exp 6: Mixed ops on SAME group ({WORLD_SIZE} GPUs, 128MB per op)",
        "  nccl/freed: min/avg/max across all ranks. * = measurement noise.",
    )


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

    log(f"=== NCCL Memory Profiling (world_size={WORLD_SIZE}) ===")
    log(f"GPU: {torch.cuda.get_device_name(0)}")

    # Dummy warmup to trigger CUDA context / cuRAND / NCCL lazy init
    # before any measurement, eliminating first-allocation noise.
    cuda_warmup()
    log(f"Baseline gpu_used (post-init): {gpu_used_mb():.0f} MB")

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from verl.utils.nccl_suspend import _get_nccl_lib, resume_nccl_comm, suspend_nccl_comm

    if _get_nccl_lib() is None:
        log("SKIP: NCCL >= 2.29.7 required")
        dist.destroy_process_group()
        return

    experiments = [
        ("Exp 1/6: Group size",      exp_group_size,     "~15s"),
        ("Exp 2/6: Message size",    exp_message_size,   "~60s"),
        ("Exp 3/6: Collective type", exp_collective_type, "~30s"),
        ("Exp 4/6: Num groups",      exp_num_groups,     "~90s"),
        ("Exp 5/6: Warmup rounds",   exp_warmup_rounds,  "~120s"),
        ("Exp 6/6: Mixed ops",       exp_mixed_ops,      "~60s"),
    ]
    total_start = time.perf_counter()
    for i, (name, fn, est) in enumerate(experiments):
        log(f"\n>>> Starting {name} (est {est}) ...")
        t = time.perf_counter()
        fn(suspend_nccl_comm, resume_nccl_comm)
        elapsed = time.perf_counter() - t
        total_elapsed = time.perf_counter() - total_start
        log(f">>> {name} done in {elapsed:.0f}s (total {total_elapsed:.0f}s)")

    log(f"\n{'=' * 90}")
    log("ALL PROFILING EXPERIMENTS COMPLETE")
    log(f"{'=' * 90}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
