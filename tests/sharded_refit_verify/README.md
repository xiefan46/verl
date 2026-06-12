# Sharded-Aware Refit — M3 Layout Verify (Qwen3.5-35B-A3B)

One-shot verify utility for the **sharded-aware weight refit** feature
(see `research/2026-06-11-sharded-aware-weight-refit-design.md`).

**Goal**: confirm the exact storage layout that Megatron and vLLM use for
Qwen3.5 MoE weights, so the M3 implementation of `get_local_shards()` and
the HF→vLLM packed-modules mapping can be written against real, observed
shapes instead of assumed ones.

**Cost**: ~15 minutes of GPU time total. Decides 2-3 days of M3 direction.

---

## What we already know

- verl forces `moe_grouped_gemm=True` for Qwen3.5 MoE
  (`verl/models/mcore/config_converter.py:206` and `:257`).
  → **Megatron uses grouped layout**, not list-of-experts.
- Qwen3.5 = Qwen3 MoE family + Gated Delta Net (GDN) linear attention;
  same `experts` block structure, so dump for Qwen3-30B-A3B is informative
  for Qwen3.5-35B-A3B (used as fallback if Qwen3.5 not yet supported by
  your vLLM build).

## What this verify confirms

1. Exact `linear_fc1.weight` shape and dim ordering on each rank
   (`[E_local, 2*I/tp, H]` vs `[2*I/tp, E_local, H]` vs other).
2. Full `named_parameters()` keys for layer 0 in the wrapped Megatron module
   (so M3 knows the actual key strings to map from).
3. Same for vLLM: how the model exposes MoE params after load
   (`experts.w13_weight`, `experts.0.gate_proj.weight`, etc.) — determines
   the HF→vLLM mapping and `dst_slice` fused offset.
4. Router weight and shared-expert weight names + shapes.
5. `packed_modules_mapping` on the vLLM model class.

---

## Environment prep (RunPod)

Qwen3.5 needs newer packages on top of the standard `verl-env-cache`:

```bash
pip install --upgrade transformers
pip install flash-linear-attention                      # required by Qwen3.5 GDN
pip install -U git+https://github.com/ISEEKYAN/mbridge.git
# Megatron-LM 0.16.0 should already be in the env cache; if not:
#   pip install megatron-core==0.16.0
```

HF cache: download Qwen3.5-35B-A3B (~70 GB BF16) ahead of time so verify
doesn't stall waiting for download.

```bash
hf download Qwen/Qwen3.5-35B-A3B --local-dir ~/models/Qwen3.5-35B-A3B
# Or just set HF_HUB_OFFLINE=0 and let vLLM/HF download on first run
```

Dataset placeholder (hook exits before data is read but Hydra needs paths):

```bash
mkdir -p ~/data/geo3k
# Either download geo3k parquet (used by existing launcher) or symlink any
# valid parquet file to satisfy the path:
touch ~/data/geo3k/train.parquet
touch ~/data/geo3k/test.parquet
```

---

## How to run

### 1) Megatron side (4×H100, ~5 min)

```bash
cd ~/verl
bash tests/sharded_refit_verify/run_verify_megatron.sh 2>&1 | tee /tmp/verify_megatron.log
```

Default config: TP=2 EP=2 GEN_TP=2 on 4 GPUs.

Overrides:

```bash
# Canonical 8 GPU config — exercise EP=8 (the upstream tested setup)
TP=2 EP=8 GEN_TP=8 NDEVICES_PER_NODE=8 \
  bash tests/sharded_refit_verify/run_verify_megatron.sh

# Use local HF cache path
HF_MODEL_PATH=/root/models/Qwen3.5-35B-A3B \
  bash tests/sharded_refit_verify/run_verify_megatron.sh
```

### 2) vLLM side (2×H100, ~5 min)

```bash
cd ~/verl
python tests/sharded_refit_verify/verify_vllm_moe_layout.py \
    --model Qwen/Qwen3.5-35B-A3B --tp 2 \
    2>&1 | tee /tmp/verify_vllm.log
```

If your vLLM build doesn't support Qwen3.5 yet (it's been supported in main
since ~2026-03, but pinned builds may lag), fall back to Qwen3-30B-A3B —
same MoE block structure:

```bash
python tests/sharded_refit_verify/verify_vllm_moe_layout.py \
    --model Qwen/Qwen3-30B-A3B --tp 2
```

---

## What to look for in the output

### Megatron output

1. `.experts type:` — should be `TEGroupedMLP` or `GroupedMLP` (grouped).
2. `.experts.linear_fc1.weight` shape — write down whether expert dim is
   leading (`[E, 2I/tp, H]`) or trailing.
3. Full `named_parameters()` dump for layer 0 — copy verbatim into the M3
   design doc as the reference layout.
4. Wrapper chain (DDP / Float16Module / ...) — determines how to unwrap
   in `get_local_shards()`.

### vLLM output

1. `.experts type:` (likely `FusedMoE`).
2. Whether `experts.w13_weight` is `[E_local, 2*I/tp, H]` fused (grouped)
   or `experts.0.gate_proj.weight` list-style.
3. `packed_modules_mapping` — HF→vLLM fused offset for `qkv_proj` /
   `gate_up_proj` / MoE experts. This is **the** info we need to compute
   `dst_slice` in the route table.

---

## After verify is done

1. Paste both logs into
   `research/2026-06-11-sharded-aware-weight-refit-design.md` under a new
   "M3 Layout Confirmed" section.
2. Update M3 pseudo-code in the design doc with the actual key names + shapes.
3. The hook code stays (gated by `VERIFY_MOE_LAYOUT=1`) — no runtime cost
   when the env var is unset, useful for later debugging.

---

## Files

| File | Purpose |
|---|---|
| `run_verify_megatron.sh` | wraps `run_qwen3_5_35b_megatron.sh` + `VERIFY_MOE_LAYOUT=1` |
| `verify_vllm_moe_layout.py` | standalone vLLM init + named_parameters dump |
| `README.md` | this file |
| `verl/workers/engine/megatron/_verify_moe_layout_hook.py` | print + sys.exit logic |
| `verl/workers/engine/megatron/transformer_impl.py` (4-line edit) | env-var trigger after `make_megatron_module()` |
