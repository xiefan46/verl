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
Functional correctness tests for NCCL communicator suspend/resume.

Tests:
  1. Basic: multi-group allreduce, suspend, resume, verify
  2. P2P safety: send/recv across suspend/resume cycles
  3. All-to-all: suspend/resume with all_to_all warmup
  4. Multi-cycle stability: repeated suspend/resume with mixed ops

Usage:
    torchrun --nproc_per_node=2 tests/utils/test_nccl_suspend.py
    torchrun --nproc_per_node=8 tests/utils/test_nccl_suspend.py
"""

import os
import sys
import time

import torch
import torch.distributed as dist

RANK = None
WORLD_SIZE = None


def log(msg):
    if RANK == 0:
        print(msg, flush=True)


def gpu_used_mb():
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    return (total - free) / 1024**2


def extract_comm(pg, name=""):
    try:
        backend = pg._get_backend(torch.device("cuda"))
        if hasattr(backend, "_comm_ptr"):
            return backend._comm_ptr()
    except Exception as e:
        log(f"  WARNING: extract_comm failed for '{name}': {e}")
    return None


def suspend_resume_cycle(comm_handles, suspend_fn, resume_fn):
    """Suspend all → measure → resume. Returns freed_mb."""
    dist.barrier()
    torch.cuda.synchronize()
    before = gpu_used_mb()
    dist.barrier()
    for _, comm in comm_handles:
        suspend_fn(comm)
    torch.cuda.empty_cache()
    after = gpu_used_mb()
    for _, comm in comm_handles:
        resume_fn(comm)
    torch.cuda.synchronize()
    return before - after


# ---------------------------------------------------------------------------
# Test 1: Basic multi-group allreduce
# ---------------------------------------------------------------------------
def test_basic(suspend_fn, resume_fn):
    log("\n[Test 1] Basic multi-group allreduce suspend/resume")
    expected = sum(range(1, WORLD_SIZE + 1))
    all_ranks = list(range(WORLD_SIZE))

    groups = {f"g{i}": dist.new_group(ranks=all_ranks) for i in range(6)}
    all_pgs = [("default", dist.distributed_c10d._get_default_group())] + list(groups.items())

    # Warmup
    for name, pg in all_pgs:
        for _ in range(3):
            buf = torch.randn(4096, 4096, device="cuda")
            dist.all_reduce(buf, group=pg) if name != "default" else dist.all_reduce(buf)
            del buf
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    # Extract comms
    comm_handles = [(n, extract_comm(pg, n)) for n, pg in all_pgs]
    comm_handles = [(n, c) for n, c in comm_handles if c is not None]

    freed = suspend_resume_cycle(comm_handles, suspend_fn, resume_fn)
    log(f"  {len(comm_handles)} comms, freed={freed:.0f} MB")

    # Verify
    for name, pg in all_pgs:
        z = torch.ones(1024, 1024, device="cuda") * (RANK + 1)
        dist.all_reduce(z, group=pg) if name != "default" else dist.all_reduce(z)
        assert torch.allclose(z, torch.full_like(z, expected)), f"FAIL: {name}"
        del z
    log(f"  PASS: all {len(all_pgs)} groups verified")


# ---------------------------------------------------------------------------
# Test 2: P2P safety (using batch_isend_irecv to avoid lazy 2-rank comm creation)
# ---------------------------------------------------------------------------
def _ring_p2p(pg, size, value_fn):
    """Ring send/recv using batch_isend_irecv. Returns recv tensor."""
    send_buf = torch.ones(size, size, device="cuda") * value_fn(RANK)
    recv_buf = torch.zeros(size, size, device="cuda")
    dst = (RANK + 1) % WORLD_SIZE
    src = (RANK - 1) % WORLD_SIZE
    ops = [
        dist.P2POp(dist.isend, send_buf, dst, group=pg),
        dist.P2POp(dist.irecv, recv_buf, src, group=pg),
    ]
    reqs = dist.batch_isend_irecv(ops)
    for req in reqs:
        req.wait()
    return recv_buf


def test_p2p(suspend_fn, resume_fn):
    log("\n[Test 2] P2P send/recv safety across suspend/resume")
    pg = dist.new_group(ranks=list(range(WORLD_SIZE)))

    # Warmup: ring P2P with large tensors
    for _ in range(5):
        recv = _ring_p2p(pg, 4096, lambda r: r + 1)
        del recv
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    comm = extract_comm(pg, "p2p")
    assert comm is not None

    # 5 suspend/resume cycles, each followed by P2P verify
    for cycle in range(5):
        freed = suspend_resume_cycle([("p2p", comm)], suspend_fn, resume_fn)

        recv = _ring_p2p(pg, 2048, lambda r: r + cycle + 1)
        src = (RANK - 1) % WORLD_SIZE
        expected = src + cycle + 1
        assert recv[0, 0].item() == expected, f"FAIL cycle {cycle}: got {recv[0, 0].item()}"
        del recv
        log(f"  Cycle {cycle + 1}: P2P OK, freed={freed:.0f} MB")

    log("  PASS")


# ---------------------------------------------------------------------------
# Test 3: All-to-all
# ---------------------------------------------------------------------------
def test_alltoall(suspend_fn, resume_fn):
    log("\n[Test 3] All-to-all suspend/resume")
    a2a_group = dist.new_group(ranks=list(range(WORLD_SIZE)))

    # Warmup
    for _ in range(5):
        inp = torch.randn(4096 * WORLD_SIZE, 1024, device="cuda")
        out = torch.empty_like(inp)
        dist.all_to_all_single(out, inp, group=a2a_group)
        del inp, out
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    comm = extract_comm(a2a_group, "a2a")
    assert comm is not None

    freed = suspend_resume_cycle([("a2a", comm)], suspend_fn, resume_fn)

    # Verify
    inp = torch.ones(1024 * WORLD_SIZE, device="cuda") * (RANK + 1)
    out = torch.empty_like(inp)
    dist.all_to_all_single(out, inp, group=a2a_group)
    for r in range(WORLD_SIZE):
        chunk = out[r * 1024 : (r + 1) * 1024]
        assert torch.allclose(chunk, torch.full_like(chunk, r + 1)), f"FAIL rank {r}"
    del inp, out
    log(f"  freed={freed:.0f} MB, verify OK")
    log("  PASS")


# ---------------------------------------------------------------------------
# Test 4: Multi-cycle stability with mixed ops
# ---------------------------------------------------------------------------
def test_multicycle_mixed(suspend_fn, resume_fn):
    log("\n[Test 4] Multi-cycle stability with mixed collective ops")
    pg = dist.new_group(ranks=list(range(WORLD_SIZE)))
    comm = extract_comm(pg, "mixed")
    assert comm is not None

    for cycle in range(10):
        # Mix of collective ops before suspend (no all_to_all — it can put
        # the comm in a state that rejects suspend on some NCCL versions;
        # all_to_all suspend is already tested separately in Test 3)
        buf = torch.randn(2048, 2048, device="cuda")
        dist.all_reduce(buf, group=pg)
        dist.broadcast(buf, src=0, group=pg)
        gathered = [torch.empty_like(buf) for _ in range(WORLD_SIZE)]
        dist.all_gather(gathered, buf, group=pg)
        del gathered

        # reduce_scatter: input=2048*2048*ws, output=2048*2048
        rs_inp = torch.randn(2048 * 2048 * WORLD_SIZE, device="cuda")
        rs_out = torch.empty(2048 * 2048, device="cuda")
        dist.reduce_scatter_tensor(rs_out, rs_inp, group=pg)
        del rs_inp, rs_out, buf

        # Suspend / resume
        freed = suspend_resume_cycle([("mixed", comm)], suspend_fn, resume_fn)

        # Quick verify
        v = torch.ones(256, device="cuda") * (RANK + 1)
        dist.all_reduce(v, group=pg)
        assert v[0].item() == sum(range(1, WORLD_SIZE + 1))
        del v

        if cycle % 3 == 0:
            log(f"  Cycle {cycle + 1}/10: freed={freed:.0f} MB, OK")

    log("  PASS: 10 cycles stable")


# ---------------------------------------------------------------------------
def main():
    global RANK, WORLD_SIZE
    dist.init_process_group(backend="nccl")
    RANK = dist.get_rank()
    WORLD_SIZE = dist.get_world_size()
    torch.cuda.set_device(RANK)

    log(f"=== NCCL Suspend/Resume Functional Tests (world_size={WORLD_SIZE}) ===")

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from verl.utils.nccl_suspend import _get_nccl_lib, resume_nccl_comm, suspend_nccl_comm

    if _get_nccl_lib() is None:
        log("SKIP: NCCL >= 2.29.7 required")
        dist.destroy_process_group()
        return

    test_basic(suspend_nccl_comm, resume_nccl_comm)
    test_p2p(suspend_nccl_comm, resume_nccl_comm)
    test_alltoall(suspend_nccl_comm, resume_nccl_comm)
    test_multicycle_mixed(suspend_nccl_comm, resume_nccl_comm)

    log(f"\n{'=' * 60}")
    log("ALL FUNCTIONAL TESTS PASSED")
    log(f"{'=' * 60}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
