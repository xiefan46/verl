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

"""Phase 3 Task 3.1: engine-postprocess equivalence (no full FSDPEngine).

Phase 1's ``test_forward_equivalence`` already validates the *algorithm*
layer: tree-packed forward → ``gather_packed_tree_logprobs`` (flat 1-D in
trie order) matches per-sequence independent forward. The remaining gap
is the *engine postprocess* layer added in commit 03106821:

  - ``unpack_tree_logprobs_per_seq`` (returns ``dict[seq_id, Tensor]``)
  - ``assemble_tree_per_seq_to_nested`` (row-major nested with sentinel
    appended at position L-1 of each row, convention B matching verl's
    production dense path)

This pipeline is what ``compute_log_prob`` returns to the trainer, so a
silent off-by-one or trie-order leak would corrupt PPO ratio computation
while still passing Phase 1's flat-tensor equivalence check.

This test goes through the full forward + per-seq unpack + aggregate path
and compares to a row-major dense baseline using a tiny Llama (CUDA only;
flex_attention has no CPU backend).
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Tree training engine path needs CUDA (flex_attention + flash_attention_2).",
)


def _make_tiny_llama(vocab_size: int, *, dtype: torch.dtype, device: torch.device):
    """Same tiny Llama factory as Phase 1 forward/backward equivalence tests."""
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=2048,
        rope_theta=10000.0,
        attn_implementation="flash_attention_2",
    )
    torch.manual_seed(0)
    model = LlamaForCausalLM(config).to(device=device, dtype=dtype)
    model.eval()
    return model


def _dense_baseline_per_row(
    model,
    batch: dict[str, torch.Tensor],
    *,
    with_entropy: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Forward each sequence independently; return ``[B, L]`` logprobs and entropy.

    Layout: **convention B** (matches verl production dense path and the
    updated ``assemble_tree_per_seq_to_nested``). Positions ``[0, L-1)`` store
    transitions ``log P(input_ids[k+1] | input_ids[0..k])``; position ``L-1``
    is a sentinel (0.0) — analogous to verl dense path's rolled-label
    wraparound garbage, and discarded by ``no_padding_2_padding``'s slice
    exclusive endpoint.
    """
    from verl.experimental.tree_training._vocab_parallel import gather_logprobs

    input_ids = batch["input_ids"]
    attn_mask = batch["attention_mask"].to(torch.long)
    bsz, total_len = input_ids.shape

    logprob_rows = torch.zeros(bsz, total_len, device=input_ids.device, dtype=torch.float32)
    entropy_rows = torch.zeros(bsz, total_len, device=input_ids.device, dtype=torch.float32) if with_entropy else None

    for i in range(bsz):
        with torch.no_grad():
            out = model(input_ids=input_ids[i : i + 1], attention_mask=attn_mask[i : i + 1])
        # Predict input_ids[t+1] from logits[t] for t in [0, L-2]; cast to fp32 for stable comparison.
        shifted_logits = out.logits[0, :-1].float()  # [L-1, V]
        shifted_labels = input_ids[i, 1:]  # [L-1]
        per_seq_logprobs = gather_logprobs(shifted_logits, shifted_labels)  # [L-1]
        # Convention B: place transitions at positions [0, L-1); leave position L-1 as sentinel 0.
        logprob_rows[i, :-1] = per_seq_logprobs.float()

        if with_entropy:
            # H[p] = -sum(p log p); use log_softmax for numerical stability.
            log_probs_full = torch.log_softmax(shifted_logits, dim=-1)  # [L-1, V]
            probs_full = log_probs_full.exp()
            entropy_seq = -(probs_full * log_probs_full).sum(dim=-1)  # [L-1]
            entropy_rows[i, :-1] = entropy_seq.float()

    return logprob_rows, entropy_rows


def _tree_pipeline_nested(
    model,
    batch: dict[str, torch.Tensor],
    *,
    max_tokens_per_mb: int,
    with_entropy: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run the full engine forward-only pipeline and return nested logprobs/entropy.

    Mirrors ``_forward_step_tree`` (forward_only branch) + ``_postprocess_tree_batch``
    for the engine-postprocess layer this test targets: build the trie, run
    the patched flash_attention forward, extract per-seq dicts, assemble
    to nested using the explicit ``offsets`` we'd see at the engine boundary.
    """
    from verl.experimental.tree_training._areal_data import MicroBatchSpec
    from verl.experimental.tree_training._verl_adapter import (
        assemble_tree_per_seq_to_nested,
        unpack_tree_logprobs_per_seq,
    )
    from verl.experimental.tree_training.module import build_tree_attn_kwargs
    from verl.experimental.tree_training.module_fsdp import (
        patch_fsdp_for_tree_training,
        restore_patch_fsdp_for_tree_training,
    )
    from verl.experimental.tree_training.tree import build_packed_tree_batch

    data = {
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"].to(torch.long),
    }
    mb_spec = MicroBatchSpec(max_tokens_per_mb=max_tokens_per_mb)
    mb_list = build_packed_tree_batch(data, mb_spec)

    bsz, total_len = batch["input_ids"].shape
    # All seqs same length L; cu-seqlens-style offsets [0, L, 2L, ..., B*L]
    offsets = torch.arange(0, (bsz + 1) * total_len, total_len, dtype=torch.long, device=batch["input_ids"].device)

    per_mb_logprob_dicts: list[dict[int, torch.Tensor]] = []
    per_mb_entropy_dicts: list[dict[int, torch.Tensor]] = []

    patch_fsdp_for_tree_training(enable=True)
    try:
        for mb in mb_list.padded_mbs:
            trie = mb["trie_node"]
            packed_input_ids = mb["input_ids"]
            position_ids = mb["position_ids"]
            padded_size = packed_input_ids.size(-1)

            tree_attn_kwargs = build_tree_attn_kwargs(trie, padded_size, device=packed_input_ids.device)
            with torch.no_grad():
                out = model(input_ids=packed_input_ids, position_ids=position_ids, **tree_attn_kwargs)
            logits = out.logits.squeeze(0).float()  # [T_padded, V]

            logprob_dict, entropy_dict = unpack_tree_logprobs_per_seq(
                logits,
                trie,
                packed_input_ids,
                temperature=1.0,
                with_entropy=with_entropy,
            )
            per_mb_logprob_dicts.append(logprob_dict)
            if with_entropy:
                per_mb_entropy_dicts.append(entropy_dict)
    finally:
        restore_patch_fsdp_for_tree_training()

    log_probs_nested = assemble_tree_per_seq_to_nested(per_mb_logprob_dicts, offsets=offsets, sentinel=0.0)
    entropy_nested = None
    if with_entropy:
        entropy_nested = assemble_tree_per_seq_to_nested(per_mb_entropy_dicts, offsets=offsets, sentinel=0.0)
    return log_probs_nested, entropy_nested


# build_packed_tree_batch requires max_tokens_per_mb to be a multiple of BLOCK_SIZE=128.
@pytest.mark.parametrize(
    "prompt_len,response_len,max_tokens_per_mb",
    [
        # POR ≈ 0.25 — short prompt, long response (minimal sharing)
        (32, 96, 1024),
        # POR = 0.5 — balanced
        (64, 64, 1024),
        # POR ≈ 0.75 — long prompt, short response (heavy sharing)
        (96, 32, 1024),
    ],
)
def test_engine_postprocess_logprob_equivalence(prompt_len: int, response_len: int, max_tokens_per_mb: int) -> None:
    """Per-row nested logprobs from tree pipeline should match dense baseline."""
    from tests.experimental.tree_training.synthetic import make_prompt_sharing_batch

    vocab_size = 1024
    device = torch.device("cuda")
    dtype = torch.bfloat16

    batch = make_prompt_sharing_batch(
        num_prompts=2,
        rollouts_per_prompt=4,
        prompt_len=prompt_len,
        response_len=response_len,
        vocab_size=vocab_size,
        device=device,
    )

    model = _make_tiny_llama(vocab_size, dtype=dtype, device=device)

    baseline_logprobs_2d, _ = _dense_baseline_per_row(model, batch, with_entropy=False)
    tree_log_probs_nested, _ = _tree_pipeline_nested(
        model, batch, max_tokens_per_mb=max_tokens_per_mb, with_entropy=False
    )

    # Verify offset layout: nested.offsets() should match what we passed
    bsz, total_len = batch["input_ids"].shape
    expected_offsets = torch.arange(
        0, (bsz + 1) * total_len, total_len, dtype=torch.long, device=tree_log_probs_nested.values().device
    )
    assert torch.equal(tree_log_probs_nested.offsets(), expected_offsets), (
        f"offsets mismatch: got {tree_log_probs_nested.offsets().tolist()}, expected {expected_offsets.tolist()}"
    )

    # Reshape tree values back to [B, L] for direct per-row comparison.
    tree_values_2d = tree_log_probs_nested.values().view(bsz, total_len)

    # Sentinel at position 0 must be exact (it's a fixed constant, not a forward output).
    assert torch.all(tree_values_2d[:, -1] == 0.0), (
        f"sentinel at last position should be exactly 0.0, got {tree_values_2d[:, -1].tolist()}"
    )

    # Convention B layout: row[t] is logprob predicting input_ids[t+1] for t in [0, L-1);
    # row[L-1] is sentinel. Baseline and tree both follow this convention so rows align
    # position-by-position; the trainer's no_padding_2_padding slice
    # [prompt_len-1, prompt_len+resp_len-1) then extracts response-token logprobs.
    diff = (tree_values_2d.float() - baseline_logprobs_2d.float()).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    print(f"\n[logprob diff] max_abs={max_abs:.6f}, mean_abs={mean_abs:.6f}")

    # Response-window diff specifically (the part PPO ratio reads):
    resp_diff = diff[:, prompt_len:]
    resp_max = resp_diff.max().item()
    resp_mean = resp_diff.mean().item()
    print(f"[response-window diff] max_abs={resp_max:.6f}, mean_abs={resp_mean:.6f}")

    # atol/rtol consistent with Phase 1 forward equivalence
    torch.testing.assert_close(
        tree_values_2d.float(),
        baseline_logprobs_2d.float(),
        atol=0.01,
        rtol=0.01,
    )


@pytest.mark.parametrize(
    "prompt_len,response_len,max_tokens_per_mb",
    [
        (64, 64, 1024),
    ],
)
def test_engine_postprocess_entropy_equivalence(prompt_len: int, response_len: int, max_tokens_per_mb: int) -> None:
    """Per-row nested entropy from tree pipeline should match dense baseline.

    Entropy is computed in the tree path only when ``calculate_entropy=True``
    (which the trainer sets in ``_compute_old_log_prob``). This exercises the
    ``with_entropy=True`` branch added in commit 03106821.
    """
    from tests.experimental.tree_training.synthetic import make_prompt_sharing_batch

    vocab_size = 1024
    device = torch.device("cuda")
    dtype = torch.bfloat16

    batch = make_prompt_sharing_batch(
        num_prompts=2,
        rollouts_per_prompt=4,
        prompt_len=prompt_len,
        response_len=response_len,
        vocab_size=vocab_size,
        device=device,
    )

    model = _make_tiny_llama(vocab_size, dtype=dtype, device=device)

    _, baseline_entropy_2d = _dense_baseline_per_row(model, batch, with_entropy=True)
    _, tree_entropy_nested = _tree_pipeline_nested(model, batch, max_tokens_per_mb=max_tokens_per_mb, with_entropy=True)

    bsz, total_len = batch["input_ids"].shape
    tree_values_2d = tree_entropy_nested.values().view(bsz, total_len)

    assert torch.all(tree_values_2d[:, -1] == 0.0), "entropy sentinel at last position should be 0.0"

    diff = (tree_values_2d.float() - baseline_entropy_2d.float()).abs()
    print(f"\n[entropy diff] max_abs={diff.max().item():.6f}, mean_abs={diff.mean().item():.6f}")

    torch.testing.assert_close(
        tree_values_2d.float(),
        baseline_entropy_2d.float(),
        atol=0.01,
        rtol=0.01,
    )


# =============================================================================
# Phase 3 Task 3.2: edge cases
# =============================================================================


def test_engine_postprocess_degenerate_trie() -> None:
    """Single-sequence batch (1 prompt × 1 rollout): trie has 1 segment, no sharing.

    This is the degenerate case where the tree advantage vanishes (POR = 0) but
    the pipeline must still produce correct row-major nested output. Catches
    failures where ``assemble_tree_per_seq_to_nested`` accidentally requires
    ``len(by_row) > 1`` or where ``unpack_tree_logprobs_per_seq`` mishandles
    a single-seq trie.
    """
    from tests.experimental.tree_training.synthetic import make_prompt_sharing_batch

    vocab_size = 1024
    device = torch.device("cuda")
    dtype = torch.bfloat16

    batch = make_prompt_sharing_batch(
        num_prompts=1,
        rollouts_per_prompt=1,
        prompt_len=64,
        response_len=64,
        vocab_size=vocab_size,
        device=device,
    )
    assert batch["input_ids"].shape[0] == 1, "degenerate test requires batch_size=1"

    model = _make_tiny_llama(vocab_size, dtype=dtype, device=device)

    baseline_logprobs_2d, _ = _dense_baseline_per_row(model, batch, with_entropy=False)
    tree_log_probs_nested, _ = _tree_pipeline_nested(model, batch, max_tokens_per_mb=256, with_entropy=False)

    bsz, total_len = batch["input_ids"].shape
    tree_values_2d = tree_log_probs_nested.values().view(bsz, total_len)

    diff = (tree_values_2d.float() - baseline_logprobs_2d.float()).abs()
    print(f"\n[degenerate trie] max_abs={diff.max().item():.6f}, mean_abs={diff.mean().item():.6f}")

    assert torch.all(tree_values_2d[:, -1] == 0.0), "sentinel at last position should be 0.0"
    torch.testing.assert_close(
        tree_values_2d.float(),
        baseline_logprobs_2d.float(),
        atol=0.01,
        rtol=0.01,
    )


def test_engine_postprocess_multi_mb() -> None:
    """Large batch forced into multiple trie partitions (mb).

    Ensures ``assemble_tree_per_seq_to_nested`` correctly stitches per-seq
    dicts across multiple micro-batches in row-major order, not just within
    a single mb. Catches failures where seq_ids from different mbs collide
    or where the cross-mb merge accidentally reorders rows.
    """
    from tests.experimental.tree_training.synthetic import make_prompt_sharing_batch
    from verl.experimental.tree_training._areal_data import MicroBatchSpec
    from verl.experimental.tree_training.tree import build_packed_tree_batch

    vocab_size = 1024
    device = torch.device("cuda")
    dtype = torch.bfloat16

    # 4 distinct prompts × 4 rollouts = 16 seqs × 128 tokens = 2048 raw tokens.
    # Each prompt group's trie can share prompt tokens within group, so unique
    # trie tokens ≈ 4 × 64 + 16 × 64 = 1280. max_tokens_per_mb=256 (BLOCK_SIZE=128
    # multiple) forces split into multiple mbs.
    num_prompts = 4
    rollouts_per_prompt = 4
    prompt_len = 64
    response_len = 64
    max_tokens_per_mb = 256

    batch = make_prompt_sharing_batch(
        num_prompts=num_prompts,
        rollouts_per_prompt=rollouts_per_prompt,
        prompt_len=prompt_len,
        response_len=response_len,
        vocab_size=vocab_size,
        device=device,
    )

    # Verify we actually trigger a multi-mb split (otherwise the test is moot).
    data = {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"].to(torch.long)}
    probe_mb_list = build_packed_tree_batch(data, MicroBatchSpec(max_tokens_per_mb=max_tokens_per_mb))
    num_mbs = len(probe_mb_list.padded_mbs)
    print(f"\n[multi-mb] split into {num_mbs} micro-batches")
    assert num_mbs > 1, f"expected multi-mb split, got {num_mbs}; raise batch size or lower max_tokens_per_mb"

    model = _make_tiny_llama(vocab_size, dtype=dtype, device=device)

    baseline_logprobs_2d, _ = _dense_baseline_per_row(model, batch, with_entropy=False)
    tree_log_probs_nested, _ = _tree_pipeline_nested(
        model, batch, max_tokens_per_mb=max_tokens_per_mb, with_entropy=False
    )

    bsz, total_len = batch["input_ids"].shape
    tree_values_2d = tree_log_probs_nested.values().view(bsz, total_len)

    diff = (tree_values_2d.float() - baseline_logprobs_2d.float()).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    print(f"[multi-mb diff] max_abs={max_abs:.6f}, mean_abs={mean_abs:.6f}")

    # Per-row diff to catch cross-mb stitch errors that would corrupt some rows
    # while leaving others intact (a within-row max alone could miss this).
    per_row_max = diff.max(dim=1).values
    print(f"[multi-mb per-row max] {per_row_max.tolist()}")
    assert per_row_max.max().item() < 0.01, f"per-row max exceeds tolerance: {per_row_max.tolist()}"

    torch.testing.assert_close(
        tree_values_2d.float(),
        baseline_logprobs_2d.float(),
        atol=0.01,
        rtol=0.01,
    )
