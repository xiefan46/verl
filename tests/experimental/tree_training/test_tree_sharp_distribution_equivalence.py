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

"""Sharp-distribution probe for tree vs dense entropy / KL equivalence.

Phase 4 production showed actor/kl_loss=5.4 and actor/entropy=1.21 for tree
path vs 0.001 and 0.156 for dense at step 1 (same weights). Convention-A
fix (commit f98889a2) fixed ppo_kl (-0.017 → 0) but did NOT change kl_loss
or entropy. Phase 1 / Phase 3.1 / production-reproducer tests all use
RANDOM tiny Llama, which produces uniform-entropy (~ln(vocab)) distributions
that mask any per-position entropy mismatches. Real instruct models produce
SHARP distributions where small logit perturbations can give very different
entropy values — that regime is what Phase 4 hit.

This test pre-trains the tiny Llama for a few steps on a fixed batch to
make distributions sharper, then verifies tree-path entropy and log_probs
match the dense baseline at each response position. If tree produces
materially different values than dense for sharp distributions, the bug
is in the algorithm layer (``_gather_packed_tree_logprobs_entropy`` or the
trie's flex_attention path), not in postprocess layout.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Tree training pipeline requires CUDA (flex_attention).",
)


def _make_sharper_tiny_llama(vocab_size: int, *, dtype: torch.dtype, device: torch.device, sharpen_steps: int = 30):
    """Build a tiny Llama and pre-train briefly so distributions get sharper.

    Random Llama outputs ~uniform softmax (entropy ≈ ln(vocab_size)), which
    masks per-position entropy mismatches. A few SGD steps on a fixed batch
    drives the distribution toward delta functions (very low entropy on
    seen tokens), reproducing the sharp regime that real instruct models
    operate in.
    """
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

    # Brief CE training on a tiny fixed batch to sharpen output distribution.
    train_input_ids = torch.randint(
        0, vocab_size, (2, 64), device=device, generator=torch.Generator(device=device).manual_seed(1)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    for _ in range(sharpen_steps):
        optimizer.zero_grad()
        out = model(input_ids=train_input_ids, labels=train_input_ids)
        out.loss.backward()
        optimizer.step()
    model.eval()
    return model


def _verl_dense_logprob_and_entropy_per_row(model, batch):
    """Convention-B per-row layout (matches verl production)."""
    from verl.experimental.tree_training._vocab_parallel import gather_logprobs

    input_ids = batch["input_ids"]
    attn_mask = batch["attention_mask"].to(torch.long)
    bsz, total_len = input_ids.shape

    log_probs_rows = torch.zeros(bsz, total_len, device=input_ids.device, dtype=torch.float32)
    entropy_rows = torch.zeros(bsz, total_len, device=input_ids.device, dtype=torch.float32)

    for i in range(bsz):
        with torch.no_grad():
            out = model(input_ids=input_ids[i : i + 1], attention_mask=attn_mask[i : i + 1])
        labels = torch.roll(input_ids[i], shifts=-1, dims=0)
        logits_full = out.logits[0].float()
        per_seq_logprobs = gather_logprobs(logits_full, labels)
        log_probs_rows[i, :] = per_seq_logprobs
        log_probs_full = torch.log_softmax(logits_full, dim=-1)
        probs_full = log_probs_full.exp()
        entropy_seq = -(probs_full * log_probs_full).sum(dim=-1)
        entropy_rows[i, :] = entropy_seq.float()

    return log_probs_rows, entropy_rows


def _tree_pipeline_nested(model, batch, *, max_tokens_per_mb, with_entropy):
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
    offsets = torch.arange(0, (bsz + 1) * total_len, total_len, dtype=torch.long, device=batch["input_ids"].device)

    per_mb_logprob_dicts = []
    per_mb_entropy_dicts = []

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
            logits = out.logits.squeeze(0).float()
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
    entropy_nested = (
        assemble_tree_per_seq_to_nested(per_mb_entropy_dicts, offsets=offsets, sentinel=0.0) if with_entropy else None
    )
    return log_probs_nested, entropy_nested


def _slice_response_window(nested_values_2d, prompt_len, resp_len):
    """Verl no_padding_2_padding slice convention."""
    return nested_values_2d[:, prompt_len - 1 : prompt_len + resp_len - 1]


def test_tree_dense_sharp_distribution_response_window_match() -> None:
    """With a SHARPENED model, tree vs dense response-window logprob/entropy must match.

    Production at step 1 (actor == ref weights) shows tree entropy 1.21 vs
    dense 0.156 — 8x mismatch. If this test reproduces the gap, the bug is
    in the algorithm layer; if it shows a tight match, the bug is somewhere
    in the trainer-side data flow (build_tree_mb_list expand/pack/align or
    similar).
    """
    from tests.experimental.tree_training.synthetic import make_prompt_sharing_batch

    vocab_size = 256  # small vocab → easier to sharpen via training
    device = torch.device("cuda")
    dtype = torch.bfloat16
    prompt_len, response_len = 32, 32
    max_tokens_per_mb = 256

    batch = make_prompt_sharing_batch(
        num_prompts=2,
        rollouts_per_prompt=4,
        prompt_len=prompt_len,
        response_len=response_len,
        vocab_size=vocab_size,
        device=device,
    )
    bsz, total_len = batch["input_ids"].shape

    model = _make_sharper_tiny_llama(vocab_size, dtype=dtype, device=device, sharpen_steps=30)

    dense_log_probs_2d, dense_entropy_2d = _verl_dense_logprob_and_entropy_per_row(model, batch)
    tree_log_probs_nested, tree_entropy_nested = _tree_pipeline_nested(
        model, batch, max_tokens_per_mb=max_tokens_per_mb, with_entropy=True
    )
    tree_log_probs_2d = tree_log_probs_nested.values().view(bsz, total_len)
    tree_entropy_2d = tree_entropy_nested.values().view(bsz, total_len)

    dense_resp_logprobs = _slice_response_window(dense_log_probs_2d, prompt_len, response_len)
    tree_resp_logprobs = _slice_response_window(tree_log_probs_2d, prompt_len, response_len)
    dense_resp_entropy = _slice_response_window(dense_entropy_2d, prompt_len, response_len)
    tree_resp_entropy = _slice_response_window(tree_entropy_2d, prompt_len, response_len)

    # Diagnostic: how sharp is the distribution? Mean response entropy ~< 1 indicates sharp.
    print(
        f"\n[sharpness check] dense response entropy: "
        f"mean={dense_resp_entropy.mean().item():.4f} max={dense_resp_entropy.max().item():.4f}"
    )
    assert dense_resp_entropy.mean().item() < 2.0, (
        f"Sharpening insufficient (mean entropy {dense_resp_entropy.mean().item():.4f} >= 2.0); "
        "test may not reproduce production-regime bug. Increase sharpen_steps."
    )

    # Element-wise diff
    logprob_diff = (tree_resp_logprobs - dense_resp_logprobs).abs()
    entropy_diff = (tree_resp_entropy - dense_resp_entropy).abs()
    print(
        f"[response-window diff] "
        f"logprob max={logprob_diff.max().item():.4f} mean={logprob_diff.mean().item():.4f} | "
        f"entropy max={entropy_diff.max().item():.4f} mean={entropy_diff.mean().item():.4f}"
    )
    print(
        f"  dense response_entropy_mean={dense_resp_entropy.mean().item():.4f} | "
        f"tree response_entropy_mean={tree_resp_entropy.mean().item():.4f}"
    )

    # Sharp distribution can have larger fp noise on entropy, but logprob+entropy
    # should still match within Phase 1 tolerance (atol=0.01, rtol=0.05 to be generous).
    torch.testing.assert_close(
        tree_resp_logprobs.float(),
        dense_resp_logprobs.float(),
        atol=0.05,
        rtol=0.05,
        msg="Tree response-window log_probs diverge from dense in sharp-distribution regime",
    )
    torch.testing.assert_close(
        tree_resp_entropy.float(),
        dense_resp_entropy.float(),
        atol=0.05,
        rtol=0.05,
        msg="Tree response-window entropy diverges from dense in sharp-distribution regime",
    )
