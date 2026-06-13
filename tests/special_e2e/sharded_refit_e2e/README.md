# Sharded-Aware NCCL Refit — End-to-End Gate

Validates that the new `sharded_nccl` checkpoint engine backend delivers
**bit-for-bit equivalent vLLM rollout weights** as the existing `nccl`
broadcast backend.

## How it works

`run.sh` invokes `driver.py` twice as separate Python processes (one per
backend). Each invocation:

1. Composes `ppo_megatron_trainer.yaml` with single-GPU Qwen2.5-0.5B
   overrides (all parallel dims = 1, no critic, no reward model).
2. Spins up the verl actor+rollout colocated stack via
   `init_agent_loop_manager` — that helper already calls
   `CheckpointEngineManager.update_weights()` once at the end, which is
   the entire data path under test (trainer Megatron module →
   `get_local_shards_and_metas` → routing plan → NCCL P2P →
   rollout-side bridge → vLLM `update_weights_from_sharded_ipc`).
3. Hits the freshly-loaded vLLM with a deterministic greedy completion
   (`temperature=0.0`, fixed prompt) and dumps the generated tokens +
   per-position chosen-token logprobs to JSON.

`compare.py` then asserts:

- **Gate 1 (hard)**: greedy token sequences are byte-identical. Any
  divergence here means a shard landed in the wrong slot.
- **Gate 2 (soft)**: per-position chosen-token logprobs agree within
  `--atol` (default `5e-3`, tight for BF16). Tolerance is non-zero
  because the two routing topologies accumulate reductions in different
  orders.

## Prerequisites

- 1×H100 (or any single GPU with ≥40 GB)
- `Qwen/Qwen2.5-0.5B-Instruct` weights cached locally (will auto-download
  via `hf download` if missing)
- verl env from `verl-deploy/setup_env.sh`

## Run

### Dense Qwen2.5-0.5B (2×H100, ~3-5 min)

```bash
cd /root/verl
bash tests/special_e2e/sharded_refit_e2e/run.sh
# Distinguishing — zero out trainer pre-update; both backends must
# produce identical garbage output:
DISTINGUISHING=1 bash tests/special_e2e/sharded_refit_e2e/run.sh
```

### MoE Qwen3-30B-A3B (8×H100, ~25-40 min)

```bash
cd /root/verl
bash tests/special_e2e/sharded_refit_e2e/run_moe.sh
DISTINGUISHING=1 bash tests/special_e2e/sharded_refit_e2e/run_moe.sh
```

Trainer Megatron 4 GPU **TP=2 EP=2** (param+grad+optim offload),
rollout vLLM 4 GPU **TP=4 EP=4 DP=1** (``gpu_memory_utilization=0.7``
to leave room for the NCCL bucket buffer + CUDA IPC handles). The
asymmetric configuration is deliberate — it forces cross-TP (2→4) and
cross-EP (2→4) redistribution, which is the actual value-add of
sharded refit over the broadcast baseline. First run downloads
Qwen3-30B-A3B-Instruct-2507 (~60 GB) — subsequent runs reuse the local
copy.

## Tunables (env vars)

| Var | Default | Notes |
|---|---|---|
| `MODEL_ID` | `Qwen/Qwen2.5-0.5B-Instruct` | HF model id |
| `MODEL_PATH` | `~/models/${MODEL_ID}` | Local model dir |
| `OUT_DIR` | `/tmp/sharded_refit_e2e` | JSON dump dir |
| `PROMPT` | `"The capital of France is"` | Greedy completion prompt |
| `ATOL` | `5e-3` | Logprob tolerance (gate 2) |

## Expected output on success

```
[compare] PASS gate 1 (tokens): 8 positions identical
[compare] max chosen-token logprob diff: 0.000XYZ at position N
[compare] PASS gate 2 (logprobs)
[compare] ===== sharded_nccl e2e: PASSED =====
[run] ===== sharded_refit e2e: ALL GATES PASSED =====
```

## When it fails

Gate 1 mismatch → wrong tensor at wrong slot. First suspects:

- `vllm_edge_enricher.py` regex missing a param family (e.g. dense
  `MergedColumn` gate_up_proj isn't covered in MVP).
- `MegatronToHFContext` mis-naming a Megatron layer — compare against
  `verl/workers/engine/megatron/sharded_export.py` against
  `model.named_parameters()` of the loaded Qwen2.5 module.
- `update_weights_from_sharded_ipc` calling `weight_loader` with the
  wrong kwargs for `QKVParallelLinear` / `MergedColumnParallelLinear`.

Gate 2 mismatch only → numerically close but not identical. Suspects:

- BF16 cast happening at a different stage.
- A param somehow loaded as FP32 in one path and BF16 in the other.

Each driver run also dumps `init_seconds` in its JSON, useful for
flagging unusually slow cold-starts.
