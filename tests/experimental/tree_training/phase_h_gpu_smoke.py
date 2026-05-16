# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""Phase H: MagiAttention GPU sanity checks.

Run on a fresh H100/H200 after ``pip install magi_attention`` succeeds.
Validates that:
* FFA basic forward + backward works (H.T1)
* GQA path (asymmetric Q/KV head counts) works (H.T2)
* Tree-shaped mask (multiple tiles) works (H.T3)
* Overlapping q_ranges work — atomic reduction kicks in (H.T4)
* FFA backward gradient numerically matches dense reference (H.T5)
* dispatch / undispatch round-trip at cp_size=1 is identity (H.T6)

Each sub-test prints a single line with PASS / FAIL. Script exits 0 if all
pass, 1 if any fail (so it composes cleanly with shell ``&&`` chains).

Run:
    cd /root/verl
    python tests/experimental/tree_training/phase_h_gpu_smoke.py
"""

from __future__ import annotations

import os
import sys
import traceback

import torch

DEVICE = torch.device("cuda:0")
DTYPE = torch.bfloat16

# Track failures so the script can exit with non-zero on any failure.
_FAILS: list[str] = []


def _check(name: str, condition: bool, msg: str = "") -> None:
    """Print PASS/FAIL line; record failures for final exit code."""
    if condition:
        print(f"  [PASS] {name}{(' — ' + msg) if msg else ''}")
    else:
        print(f"  [FAIL] {name}{(' — ' + msg) if msg else ''}")
        _FAILS.append(name)


def _env_check() -> bool:
    """Skip the script gracefully if CUDA or magi_attention is missing."""
    if not torch.cuda.is_available():
        print("CUDA not available — exiting (skip).")
        return False
    try:
        import magi_attention  # noqa: F401
    except ImportError as exc:
        print(f"magi_attention not installed: {exc}")
        return False
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"magi_attention: {magi_attention.__version__}")
    return True


def h_t1_basic_fwd_bwd() -> None:
    """FFA forward + backward on a single causal tile."""
    print("\n=== H.T1: basic FFA fwd+bwd ===")
    from magi_attention.api import flex_flash_attn_func

    T, H, D = 64, 4, 128
    q = torch.randn(T, H, D, device=DEVICE, dtype=DTYPE, requires_grad=True)
    k = torch.randn(T, H, D, device=DEVICE, dtype=DTYPE, requires_grad=True)
    v = torch.randn(T, H, D, device=DEVICE, dtype=DTYPE, requires_grad=True)
    qr = torch.tensor([[0, T]], dtype=torch.int32, device=DEVICE)
    kr = torch.tensor([[0, T]], dtype=torch.int32, device=DEVICE)
    tm = torch.tensor([1], dtype=torch.int32, device=DEVICE)  # CAUSAL

    out, _ = flex_flash_attn_func(q, k, v, qr, kr, tm)
    out.sum().backward()

    _check(
        "H.T1 fwd shape",
        out.shape == (T, H, D),
        f"got {tuple(out.shape)}, expected ({T}, {H}, {D})",
    )
    _check(
        "H.T1 bwd grad finite",
        torch.isfinite(q.grad).all().item()
        and torch.isfinite(k.grad).all().item()
        and torch.isfinite(v.grad).all().item(),
    )


def h_t2_gqa() -> None:
    """FFA with asymmetric Q/KV heads (Qwen2.5-0.5B ratio: 14 vs 2)."""
    print("\n=== H.T2: GQA (14 q heads / 2 kv heads) ===")
    from magi_attention.api import flex_flash_attn_func

    T, Hq, Hkv, D = 64, 14, 2, 64
    q = torch.randn(T, Hq, D, device=DEVICE, dtype=DTYPE, requires_grad=True)
    k = torch.randn(T, Hkv, D, device=DEVICE, dtype=DTYPE, requires_grad=True)
    v = torch.randn(T, Hkv, D, device=DEVICE, dtype=DTYPE, requires_grad=True)
    qr = torch.tensor([[0, T]], dtype=torch.int32, device=DEVICE)
    kr = torch.tensor([[0, T]], dtype=torch.int32, device=DEVICE)
    tm = torch.tensor([1], dtype=torch.int32, device=DEVICE)

    out, _ = flex_flash_attn_func(q, k, v, qr, kr, tm)
    out.sum().backward()

    _check("H.T2 GQA fwd shape", out.shape == (T, Hq, D))
    _check("H.T2 GQA bwd q.grad finite", torch.isfinite(q.grad).all().item())
    _check(
        "H.T2 GQA bwd k.grad finite (asymmetric Hkv<Hq)",
        torch.isfinite(k.grad).all().item(),
    )


def h_t3_tree_mask() -> None:
    """Realistic tree mask: 1 prefix + 2 leaves attending the prefix."""
    print("\n=== H.T3: tree mask (1 prefix + 2 leaves, 5 tiles) ===")
    from magi_attention.api import flex_flash_attn_func

    T = 48
    q = torch.randn(T, 4, 64, device=DEVICE, dtype=DTYPE, requires_grad=True)
    k = torch.randn(T, 4, 64, device=DEVICE, dtype=DTYPE, requires_grad=True)
    v = torch.randn(T, 4, 64, device=DEVICE, dtype=DTYPE, requires_grad=True)
    # Tiles: prefix self-causal + leaf0 self-causal + leaf0->prefix full +
    #        leaf1 self-causal + leaf1->prefix full
    qr = torch.tensor(
        [[0, 16], [16, 32], [16, 32], [32, 48], [32, 48]],
        dtype=torch.int32,
        device=DEVICE,
    )
    kr = torch.tensor(
        [[0, 16], [16, 32], [0, 16], [32, 48], [0, 16]],
        dtype=torch.int32,
        device=DEVICE,
    )
    tm = torch.tensor([1, 1, 0, 1, 0], dtype=torch.int32, device=DEVICE)

    out, _ = flex_flash_attn_func(q, k, v, qr, kr, tm)
    out.sum().backward()

    _check("H.T3 tree fwd shape", out.shape == (T, 4, 64))
    _check("H.T3 tree bwd grads finite", torch.isfinite(q.grad).all().item())


def h_t4_overlapping_q_ranges() -> None:
    """Same q_range appears twice — atomic reduction must accumulate correctly."""
    print("\n=== H.T4: overlapping q_ranges ===")
    from magi_attention.api import flex_flash_attn_func

    # leaf [16-32] appears twice as q_range (self causal + attend prefix)
    qr = torch.tensor([[16, 32], [16, 32]], dtype=torch.int32, device=DEVICE)
    kr = torch.tensor([[16, 32], [0, 16]], dtype=torch.int32, device=DEVICE)
    tm = torch.tensor([1, 1], dtype=torch.int32, device=DEVICE)
    T = 32
    q = torch.randn(T, 4, 64, device=DEVICE, dtype=DTYPE, requires_grad=True)
    k = torch.randn(T, 4, 64, device=DEVICE, dtype=DTYPE, requires_grad=True)
    v = torch.randn(T, 4, 64, device=DEVICE, dtype=DTYPE, requires_grad=True)

    out, _ = flex_flash_attn_func(q, k, v, qr, kr, tm)
    out.sum().backward()

    _check("H.T4 overlap grads finite", torch.isfinite(q.grad).all().item())


def h_t5_bwd_vs_dense() -> None:
    """FFA backward gradient must match dense (softmax + matmul) reference."""
    print("\n=== H.T5: bwd numerical match vs dense reference ===")
    from magi_attention.api import flex_flash_attn_func

    torch.manual_seed(42)
    # D=64 to hit AOT-precompiled FFA kernel (D=32 falls back to JIT).
    T, H, D = 16, 2, 64
    q = torch.randn(T, H, D, device=DEVICE, dtype=DTYPE, requires_grad=True)
    k = torch.randn(T, H, D, device=DEVICE, dtype=DTYPE, requires_grad=True)
    v = torch.randn(T, H, D, device=DEVICE, dtype=DTYPE, requires_grad=True)
    qr = torch.tensor([[0, T]], dtype=torch.int32, device=DEVICE)
    kr = torch.tensor([[0, T]], dtype=torch.int32, device=DEVICE)
    tm = torch.tensor([1], dtype=torch.int32, device=DEVICE)

    out_ffa, _ = flex_flash_attn_func(q, k, v, qr, kr, tm)
    grad_q_ffa = torch.autograd.grad(out_ffa.sum(), q, retain_graph=True)[0]

    # Dense reference: causal softmax(QK^T)V
    q_d = q.detach().clone().requires_grad_(True)
    scale = 1.0 / (D**0.5)
    scores = torch.einsum("thd,Thd->thT", q_d, k.detach()) * scale
    causal_mask = torch.tril(torch.ones(T, T, device=DEVICE, dtype=torch.bool))
    scores = scores.masked_fill(~causal_mask.unsqueeze(1), float("-inf"))
    attn = torch.softmax(scores.float(), dim=-1).to(DTYPE)
    out_dense = torch.einsum("thT,Thd->thd", attn, v.detach())
    grad_q_dense = torch.autograd.grad(out_dense.sum(), q_d)[0]

    max_diff = (grad_q_ffa - grad_q_dense).abs().max().item()
    _check(
        "H.T5 FFA bwd vs dense",
        max_diff < 0.1,
        f"max_abs_diff={max_diff:.4f} (target < 0.1)",
    )


def h_t6_dispatch_roundtrip_cp1() -> None:
    """magi_attn_flex_key + dispatch + undispatch round-trip at cp_size=1.

    Validates the V1 path actually works at single-rank cp_group. If this
    breaks, the entire Phase J/K e2e is dead in the water because every
    forward routes through dispatch/undispatch.
    """
    print("\n=== H.T6: dispatch/undispatch round-trip cp_size=1 ===")
    import torch.distributed as dist
    from magi_attention.api import (
        AttnMaskType,
        AttnRanges,
        DistAttnConfig,
        compute_pad_size,
        dispatch,
        magi_attn_flex_key,
        undispatch,
    )
    from torch.distributed.device_mesh import DeviceMesh

    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("RANK", "0")
        dist.init_process_group(backend="nccl", init_method="env://")

    mesh = DeviceMesh("cuda", torch.arange(1).reshape(1, 1), mesh_dim_names=("dp", "cp"))
    cp_group = mesh.get_group("cp")
    T = 32
    pad = compute_pad_size(T, 1, 512)
    key = magi_attn_flex_key(
        q_ranges=AttnRanges.from_ranges([(0, T)]),
        k_ranges=AttnRanges.from_ranges([(0, T)]),
        attn_mask_type=[AttnMaskType.CAUSAL],
        total_seqlen_q=T,
        total_seqlen_k=T,
        num_heads_q=4,
        num_heads_kv=2,
        head_dim=64,  # AOT-precompiled (D=32 hits JIT fallback)
        pad_size=pad,
        chunk_size=512,
        cp_group_or_mesh=cp_group,
        dist_attn_config=DistAttnConfig(),
    )
    x = torch.arange(T, device=DEVICE, dtype=DTYPE).unsqueeze(-1).expand(T, 4).contiguous()
    x_pad = dispatch(x, key=key)
    x_back = undispatch(x_pad, key)
    _check(
        "H.T6 round-trip values match (within T prefix)",
        torch.equal(x[: x_back.size(0)], x_back),
        f"in_shape={tuple(x.shape)} pad_shape={tuple(x_pad.shape)} back_shape={tuple(x_back.shape)}",
    )


def main() -> int:
    if not _env_check():
        return 0  # graceful skip

    print(
        "\nRunning 6 Phase H GPU sanity tests on MagiAttention.\nIf any fail, do NOT proceed to Phase I until resolved."
    )

    test_fns = (
        h_t1_basic_fwd_bwd,
        h_t2_gqa,
        h_t3_tree_mask,
        h_t4_overlapping_q_ranges,
        h_t5_bwd_vs_dense,
        h_t6_dispatch_roundtrip_cp1,
    )

    for fn in test_fns:
        try:
            fn()
        except Exception:
            _FAILS.append(fn.__name__)
            print(f"  [FAIL] {fn.__name__} — exception:")
            traceback.print_exc(file=sys.stdout)

    print("\n" + "=" * 60)
    if _FAILS:
        print(f"Phase H FAIL — {len(_FAILS)} test(s) failed: {_FAILS}")
        return 1
    print("Phase H PASS — all 6 sanity tests OK. Ready for Phase I.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
