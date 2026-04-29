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
                s = dist.isend(send_buf, dst, group=pg)
                r = dist.irecv(recv_buf, src, group=pg)
                s.wait()
                r.wait()
            del send_buf, recv_buf
        else:
            raise ValueError(f"Unknown op: {op}")
    torch.cuda.synchronize()


def measure_one(pg, suspend_fn, resume_fn, group_ranks):
    """Suspend one comm, measure freed + latency, resume. Returns (freed_mb, suspend_ms, resume_ms)."""
    comm = extract_comm(pg)
    if comm is None:
        return 0, 0, 0

    dist.barrier(group=pg)
    torch.cuda.synchronize()
    pre = gpu_used_mb()
    dist.barrier(group=pg)

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

    return freed, (t1 - t0) * 1000, (t3 - t2) * 1000


def print_table(headers, rows, title=""):
    if title:
        log(f"\n{'=' * 80}")
        log(title)
        log(f"{'=' * 80}")
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    log(" | ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers)))
    log("-+-".join("-" * w for w in widths))
    for row in rows:
        log(" | ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)))
    log()


def elements_to_label(n):
    """Convert float32 element count to human-readable size label."""
    bytes_val = n * 4
    if bytes_val >= 1024**3:
        return f"{bytes_val / 1024**3:.0f}GB"
    elif bytes_val >= 1024**2:
        return f"{bytes_val / 1024**2:.0f}MB"
    elif bytes_val >= 1024:
        return f"{bytes_val / 1024:.0f}KB"
    else:
        return f"{bytes_val}B"


# ---------------------------------------------------------------------------
# Exp 1: Group size
#
# Real-world mapping:
#   group_size=1 → no comm (baseline)
#   group_size=2 → TP=2 (small model) or DP=2 (8 GPU / TP4)
#   group_size=4 → TP=4 (Qwen2.5-32B typical) or EP=4
#   group_size=8 → TP=8 (Qwen2.5-72B) or EP=8 (MoE full-node expert group)
#
# Fixed: allreduce (TP gradient sync), 256MB msg (~activation size for
#   batch=2, seq=4096, hidden=8192, bf16 = 128MB, we use float32 so 256MB)
# ---------------------------------------------------------------------------

def exp_group_size(suspend_fn, resume_fn):
    sizes = [s for s in [2, 4, 8] if s <= WORLD_SIZE]
    # Activation allreduce: batch=2, seq=4096, hidden=8192, float32 → 256MB
    msg_elements = 2 * 4096 * 8192  # 67M elements = 256 MB float32
    warmup = 10

    rows = []
    for gs in sizes:
        group_ranks = list(range(gs))
        pg = dist.new_group(ranks=group_ranks)

        if RANK in group_ranks:
            torch.cuda.empty_cache()
            baseline = gpu_used_mb()
            do_warmup(pg, "allreduce", msg_elements, warmup, RANK, gs)
            torch.cuda.empty_cache()
            nccl_mem = gpu_used_mb() - baseline

            freed, sus_ms, res_ms = measure_one(pg, suspend_fn, resume_fn, group_ranks)

            real_world = {2: "TP=2 or DP=2", 4: "TP=4 (Qwen-32B)", 8: "TP=8 (Qwen-72B)"}
            if RANK == 0:
                rows.append([
                    gs,
                    real_world.get(gs, ""),
                    f"{nccl_mem:.1f}",
                    f"{freed:.1f}",
                    f"{nccl_mem - freed:.1f}",
                    f"{sus_ms:.1f}",
                    f"{res_ms:.1f}",
                ])
        dist.barrier()

    print_table(
        ["group_sz", "real_world", "nccl(MB)", "freed(MB)", "persist(MB)", "sus(ms)", "res(ms)"],
        rows,
        "Exp 1: Group size (allreduce, batch=2 × seq=4096 × hidden=8192 = 256MB)",
    )


# ---------------------------------------------------------------------------
# Exp 2: Message size
#
# Real-world message sizes in training:
#   ~256KB  → Megatron TP allreduce for small hidden layers (batch=1, seq=512, hidden=128)
#   ~4MB    → FSDP gradient bucket (PyTorch default bucket size ~25MB, but per-param can be smaller)
#   ~32MB   → Megatron TP allreduce (batch=1, seq=4096, hidden=4096, bf16)
#   ~128MB  → Megatron TP allreduce (batch=2, seq=4096, hidden=8192, bf16)
#   ~256MB  → Large activation allreduce (batch=4, seq=4096, hidden=8192, float32)
#   ~1GB    → FSDP allgather full layer params (Qwen-72B: ~1B params/layer × 2 bytes)
# ---------------------------------------------------------------------------

def exp_message_size(suspend_fn, resume_fn):
    sizes = [
        (64 * 1024,          "256KB",  "small TP (hidden=128)"),
        (1024 * 1024,        "4MB",    "FSDP grad bucket"),
        (8 * 1024 * 1024,    "32MB",   "TP (b=1,s=4k,h=4096,bf16)"),
        (32 * 1024 * 1024,   "128MB",  "TP (b=2,s=4k,h=8192,bf16)"),
        (64 * 1024 * 1024,   "256MB",  "TP (b=4,s=4k,h=8192,f32)"),
        (256 * 1024 * 1024,  "1GB",    "FSDP allgather full layer"),
    ]
    warmup = 10

    rows = []
    for elements, label, scenario in sizes:
        pg = dist.new_group(ranks=list(range(WORLD_SIZE)))
        torch.cuda.empty_cache()
        baseline = gpu_used_mb()
        do_warmup(pg, "allreduce", elements, warmup, RANK, WORLD_SIZE)
        torch.cuda.empty_cache()
        nccl_mem = gpu_used_mb() - baseline

        freed, sus_ms, res_ms = measure_one(pg, suspend_fn, resume_fn, list(range(WORLD_SIZE)))

        if RANK == 0:
            rows.append([label, scenario, f"{nccl_mem:.1f}", f"{freed:.1f}", f"{sus_ms:.1f}", f"{res_ms:.1f}"])
        dist.barrier()

    print_table(
        ["msg_size", "scenario", "nccl(MB)", "freed(MB)", "sus(ms)", "res(ms)"],
        rows,
        f"Exp 2: Message size ({WORLD_SIZE} GPUs, allreduce)",
    )


# ---------------------------------------------------------------------------
# Exp 3: Collective type
#
# Each collective maps to a parallelism dimension:
#   allreduce       → TP gradient sync (Megatron column/row parallel)
#   allgather       → FSDP unshard / Megatron sequence parallel
#   reduce_scatter  → FSDP gradient shard
#   all_to_all      → Expert Parallelism token dispatch/combine (MoE)
#   broadcast       → PP (pipeline parallel) weight broadcast, or TP parameter broadcast
#   p2p (send/recv) → PP inter-stage activation transfer
#
# Fixed: 8 GPU, message size calibrated per collective:
#   allreduce/allgather/reduce_scatter: 128MB (TP activation, b=2,s=4k,h=8192,bf16)
#   all_to_all: 128MB (MoE token dispatch: batch=2, seq=4k, hidden=4096, top_k=8)
#   broadcast: 128MB
#   p2p: 128MB (PP micro-batch activation)
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
    # 128MB float32 = batch=2, seq=4096, hidden=4096
    msg_elements = 2 * 4096 * 4096  # 33M elements = 128 MB float32
    warmup = 10

    rows = []
    for op, scenario in ops:
        pg = dist.new_group(ranks=list(range(WORLD_SIZE)))
        torch.cuda.empty_cache()
        baseline = gpu_used_mb()
        do_warmup(pg, op, msg_elements, warmup, RANK, WORLD_SIZE)
        torch.cuda.empty_cache()
        nccl_mem = gpu_used_mb() - baseline

        freed, sus_ms, res_ms = measure_one(pg, suspend_fn, resume_fn, list(range(WORLD_SIZE)))

        if RANK == 0:
            rows.append([op, scenario, f"{nccl_mem:.1f}", f"{freed:.1f}",
                         f"{nccl_mem - freed:.1f}" if nccl_mem > 0 else "N/A",
                         f"{sus_ms:.1f}", f"{res_ms:.1f}"])
        dist.barrier()

    print_table(
        ["collective", "parallelism", "nccl(MB)", "freed(MB)", "persist(MB)", "sus(ms)", "res(ms)"],
        rows,
        f"Exp 3: Collective type ({WORLD_SIZE} GPUs, 128MB msg)",
    )


# ---------------------------------------------------------------------------
# Exp 4: Number of groups
#
# Real Megatron group configurations (8 GPU, single node):
#   Minimal FSDP:   2 groups  (DP shard + DP replicate)
#   FSDP + TP:      4 groups  (TP, DP shard, DP replicate, TP-DP)
#   Full Megatron:  6 groups  (TP, DP, EP, CP, PP, TP-DP cross)
#   Megatron + aux: 8 groups  (above + TP-EP, DP-CP)
#   Upper bound:    12 groups (all combined groups for Megatron 3D+EP+CP)
# ---------------------------------------------------------------------------

def exp_num_groups(suspend_fn, resume_fn):
    counts = [1, 2, 4, 6, 8, 12]
    # TP-like allreduce, 128MB
    msg_elements = 2 * 4096 * 4096
    warmup = 5

    real_world = {
        1: "single TP or DP",
        2: "FSDP (shard+replicate)",
        4: "FSDP+TP",
        6: "Megatron (TP+DP+EP+CP+PP+cross)",
        8: "Megatron + aux groups",
        12: "Megatron 3D+EP+CP all combos",
    }

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
        total_nccl = gpu_used_mb() - baseline
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
                real_world.get(n, ""),
                f"{total_nccl:.1f}",
                f"{per_group:.1f}",
                f"{freed:.1f}",
                f"{(t1 - t0) * 1000:.1f}",
                f"{(t3 - t2) * 1000:.1f}",
            ])
        dist.barrier()

    print_table(
        ["n_groups", "scenario", "total(MB)", "per_grp(MB)", "freed(MB)", "sus(ms)", "res(ms)"],
        rows,
        f"Exp 4: Number of groups ({WORLD_SIZE} GPUs, allreduce 128MB)",
    )


# ---------------------------------------------------------------------------
# Exp 5: Warmup rounds
#
# Real training does thousands of steps, each step may do 10-50 collectives
# per group (forward + backward + optimizer). Question: does NCCL allocate
# more internal buffers over time, or is it fixed after first few ops?
#
# We test: 1, 5, 20, 100, 500 rounds to cover first-op → steady-state.
# ---------------------------------------------------------------------------

def exp_warmup_rounds(suspend_fn, resume_fn):
    rounds_list = [1, 5, 20, 100, 500]
    # 128MB activation allreduce
    msg_elements = 2 * 4096 * 4096

    rows = []
    for rounds in rounds_list:
        pg = dist.new_group(ranks=list(range(WORLD_SIZE)))
        torch.cuda.empty_cache()
        baseline = gpu_used_mb()
        do_warmup(pg, "allreduce", msg_elements, rounds, RANK, WORLD_SIZE)
        torch.cuda.empty_cache()
        nccl_mem = gpu_used_mb() - baseline

        freed, sus_ms, res_ms = measure_one(pg, suspend_fn, resume_fn, list(range(WORLD_SIZE)))

        if RANK == 0:
            rows.append([rounds, f"{nccl_mem:.1f}", f"{freed:.1f}", f"{sus_ms:.1f}", f"{res_ms:.1f}"])
        dist.barrier()

    print_table(
        ["rounds", "nccl(MB)", "freed(MB)", "sus(ms)", "res(ms)"],
        rows,
        f"Exp 5: Warmup rounds ({WORLD_SIZE} GPUs, allreduce 128MB)",
    )


# ---------------------------------------------------------------------------
# Exp 6: Mixed ops on same group
#
# In real training, the same ProcessGroup may be used for multiple op types:
#   - FSDP: allgather (unshard) + reduce_scatter (shard) on same group
#   - Megatron TP: allreduce + allgather (sequence parallel) on same group
#   - MoE: all_to_all + allreduce on same EP group
#
# Question: does using more op types on the SAME comm increase memory?
# (Each op type may trigger different NCCL algorithms → different buffers)
# ---------------------------------------------------------------------------

def exp_mixed_ops(suspend_fn, resume_fn):
    msg_elements = 2 * 4096 * 4096  # 128MB
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
        torch.cuda.empty_cache()
        baseline = gpu_used_mb()
        for op in ops:
            do_warmup(pg, op, msg_elements, warmup, RANK, WORLD_SIZE)
        torch.cuda.empty_cache()
        nccl_mem = gpu_used_mb() - baseline

        freed, sus_ms, _ = measure_one(pg, suspend_fn, resume_fn, list(range(WORLD_SIZE)))

        if RANK == 0:
            rows.append([
                label,
                ", ".join(ops),
                f"{nccl_mem:.1f}",
                f"{freed:.1f}",
                f"{nccl_mem - freed:.1f}" if nccl_mem > 0 else "N/A",
                f"{sus_ms:.1f}",
            ])
        dist.barrier()

    print_table(
        ["scenario", "ops", "nccl(MB)", "freed(MB)", "persist(MB)", "sus(ms)"],
        rows,
        f"Exp 6: Mixed ops on SAME group ({WORLD_SIZE} GPUs, 128MB per op)",
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

    log(f"\n{'=' * 80}")
    log("ALL PROFILING EXPERIMENTS COMPLETE")
    log(f"{'=' * 80}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
