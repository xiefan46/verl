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

"""Production-scale reproducer: real instruct model + long sequences.

Phase 4 (2026-05-13) revealed actor/kl_loss=5.4 at step 1 even after the
convention-A fix. Algorithm tests at tiny-model + short-seq scale pass.
Debug prints show the bug manifests at shared-response-prefix trie nodes
in production-scale runs: cached transitions are correctly reused across
seqs (identical values at 1024-position offsets), but the cached values
are wrong (log_prob -25 vs ref ~0).

This test loads the real Qwen2.5-0.5B-Instruct model (already on RunPod
at /root/models/), generates 4 rollouts of the same prompt at temp=1,
packs them into a single trie with max_tokens_per_mb=4096 (production
config), and compares tree forward output vs dense per-seq forward
output. If the bug reproduces here, we have a fast iteration loop for
debugging.
"""

from __future__ import annotations

import os

import pytest
import torch

MODEL_PATH = "/root/models/Qwen/Qwen3-1.7B"

pytestmark = [
    pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="Tree training pipeline requires CUDA (flex_attention).",
    ),
    pytest.mark.skipif(
        not os.path.isdir(MODEL_PATH),
        reason=f"Qwen3-1.7B not at {MODEL_PATH}; download via download_models.sh",
    ),
]


def _load_qwen_instruct(device: torch.device, dtype: torch.dtype):
    """Load Qwen3-1.7B as eval-mode HF model on device."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=dtype,
        attn_implementation="flash_attention_2",
    ).to(device=device)
    model.eval()
    return model, tokenizer


def _build_real_batch(model, tokenizer, *, num_rollouts: int, max_new_tokens: int, device: torch.device):
    """Sample ``num_rollouts`` responses from the model for one fixed prompt.

    Returns dict with ``input_ids`` ``[num_rollouts, total_len]`` and
    ``attention_mask`` all-ones (no padding — we pad/truncate so all
    sequences have the same length).
    """
    prompt_text = (
        "Solve the following math problem step by step.\n\n"
        "Janet has 12 apples. She gives half to her brother and then buys 3 more. "
        "How many apples does Janet have now?\n\nLet's think carefully."
    )
    messages = [{"role": "user", "content": prompt_text}]
    encoded = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    # apply_chat_template may return a tensor (older transformers) or BatchEncoding
    # (newer). BatchEncoding doesn't subclass dict cleanly so use tensor check.
    prompt_ids = encoded if isinstance(encoded, torch.Tensor) else encoded["input_ids"]
    prompt_ids = prompt_ids.to(device)
    prompt_len = prompt_ids.shape[1]

    torch.manual_seed(42)
    with torch.no_grad():
        rollouts = model.generate(
            prompt_ids.repeat(num_rollouts, 1),
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=1.0,
            top_p=1.0,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    # Trim/pad to uniform length for synthetic batch (mimics post-rollout shape).
    bsz = rollouts.shape[0]
    target_len = prompt_len + max_new_tokens
    if rollouts.shape[1] < target_len:
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        padded = torch.full((bsz, target_len), pad_id, dtype=rollouts.dtype, device=device)
        padded[:, : rollouts.shape[1]] = rollouts
        rollouts = padded
    else:
        rollouts = rollouts[:, :target_len]

    attention_mask = torch.ones_like(rollouts)
    return {
        "input_ids": rollouts,
        "attention_mask": attention_mask,
        "prompt_len": prompt_len,
        "total_len": target_len,
    }


def _verl_dense_logprob_per_row(model, batch):
    """Convention-B per-row layout from per-seq dense forward."""
    from verl.experimental.tree_training._vocab_parallel import gather_logprobs

    input_ids = batch["input_ids"]
    attn_mask = batch["attention_mask"].to(torch.long)
    bsz, total_len = input_ids.shape

    log_probs_rows = torch.zeros(bsz, total_len, device=input_ids.device, dtype=torch.float32)
    entropy_rows = torch.zeros(bsz, total_len, device=input_ids.device, dtype=torch.float32)
    raw_logits_rows = torch.zeros(bsz, total_len, model.config.vocab_size, device=input_ids.device, dtype=torch.float32)

    for i in range(bsz):
        with torch.no_grad():
            out = model(input_ids=input_ids[i : i + 1], attention_mask=attn_mask[i : i + 1])
        labels = torch.roll(input_ids[i], shifts=-1, dims=0)
        logits_full = out.logits[0].float()
        raw_logits_rows[i] = logits_full
        log_probs_rows[i] = gather_logprobs(logits_full, labels)
        log_probs_full = torch.log_softmax(logits_full, dim=-1)
        probs_full = log_probs_full.exp()
        entropy_rows[i] = -(probs_full * log_probs_full).sum(dim=-1).float()

    return log_probs_rows, entropy_rows, raw_logits_rows


def _tree_pipeline_with_logits(model, batch, *, max_tokens_per_mb):
    """Run tree forward and return per-mb (logits, trie, packed_input_ids)."""
    from verl.experimental.tree_training._areal_data import MicroBatchSpec
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

    mb_results = []
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
            mb_results.append(
                {
                    "trie": trie,
                    "packed_input_ids": packed_input_ids.squeeze(0),
                    "position_ids": position_ids.squeeze(0),
                    "logits": logits,
                }
            )
    finally:
        restore_patch_fsdp_for_tree_training()

    return mb_results, mb_list


@pytest.mark.parametrize(
    "max_tokens_per_mb,num_rollouts,max_new_tokens",
    [
        (2048, 4, 300),  # mid-scale
        (4096, 4, 500),  # production-scale
    ],
)
def test_tree_dense_real_instruct_logits_match(max_tokens_per_mb: int, num_rollouts: int, max_new_tokens: int) -> None:
    """Compare tree-forward logits vs dense-forward logits at each trie position.

    For each trie node containing tokens from sequence i at trie positions
    [s, e], the model's logits at trie position p ∈ [s, e] should match
    dense forward's logits at the corresponding seq position in seq i.
    Mismatches indicate tree attention / position_ids are broken.
    """
    device = torch.device("cuda")
    dtype = torch.bfloat16

    model, tokenizer = _load_qwen_instruct(device, dtype)
    batch = _build_real_batch(
        model,
        tokenizer,
        num_rollouts=num_rollouts,
        max_new_tokens=max_new_tokens,
        device=device,
    )

    print(
        f"\n[batch] bsz={num_rollouts} prompt_len={batch['prompt_len']} "
        f"total_len={batch['total_len']} max_tokens_per_mb={max_tokens_per_mb}"
    )

    _, _, dense_logits = _verl_dense_logprob_per_row(model, batch)
    # dense_logits[i, t] is the logits at seq i's seq position t.

    mb_results, mb_list = _tree_pipeline_with_logits(model, batch, max_tokens_per_mb=max_tokens_per_mb)
    print(f"[tree] num mbs: {len(mb_results)}")

    # For each mb, walk trie nodes and compare tree logits at each trie position
    # to dense logits at the corresponding seq position.
    total_mismatches = 0
    total_compared = 0
    max_logit_diff_overall = 0.0
    max_logit_diff_position = None

    for mb_idx, mb_r in enumerate(mb_results):
        trie = mb_r["trie"]
        tree_logits = mb_r["logits"]
        position_ids = mb_r["position_ids"]

        for trie_node in trie.nodes:
            seq_ids = trie_node.sequence_ids
            start, end = trie_node.tree_indices
            # tokens in this node: positions [start, end] in trie
            # In each owning seq, these tokens occupy positions [seq_pos, seq_pos+num_tokens)
            seq_pos_start = sum(anc.num_tokens for anc in trie_node.ancestors)
            num_tokens = trie_node.num_tokens
            for offset in range(num_tokens):
                trie_pos = start + offset
                seq_pos = seq_pos_start + offset
                tree_logit_vec = tree_logits[trie_pos]
                for seq_id in seq_ids:
                    dense_logit_vec = dense_logits[seq_id, seq_pos]
                    diff = (tree_logit_vec - dense_logit_vec).abs()
                    diff_max = diff.max().item()
                    if diff_max > max_logit_diff_overall:
                        max_logit_diff_overall = diff_max
                        max_logit_diff_position = (mb_idx, seq_id, seq_pos, trie_pos, position_ids[trie_pos].item())
                    if diff_max > 0.5:  # arbitrary loose threshold to count "mismatches"
                        total_mismatches += 1
                    total_compared += 1

    print(
        f"[compare] compared {total_compared} (trie_pos, seq) pairs, "
        f"mismatches (max|diff|>0.5): {total_mismatches} "
        f"({100 * total_mismatches / max(1, total_compared):.2f}%)"
    )
    print(
        f"[max diff] {max_logit_diff_overall:.4f} at "
        f"mb={max_logit_diff_position[0] if max_logit_diff_position else None} "
        f"seq_id={max_logit_diff_position[1] if max_logit_diff_position else None} "
        f"seq_pos={max_logit_diff_position[2] if max_logit_diff_position else None} "
        f"trie_pos={max_logit_diff_position[3] if max_logit_diff_position else None} "
        f"position_id={max_logit_diff_position[4] if max_logit_diff_position else None}"
    )

    # Loose tolerance — we're checking for the production-scale bug which gave
    # ~25 nat divergence. Within ~1.0 logit diff is reasonable noise.
    assert max_logit_diff_overall < 1.0, (
        f"Tree forward logits diverge from dense forward by {max_logit_diff_overall:.2f} — "
        "production-scale bug reproduced. See memory/tree-training-phase4-kl-bug.md."
    )
