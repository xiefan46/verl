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

"""Forward equivalence: tree-packed forward vs independent per-sequence forward.

Validates the foundational correctness claim of AReaL-DTA — packing
prompt-sharing rollouts into a tree and running a single forward through
flex_attention with a tree block mask produces the same logprobs as forwarding
each sequence independently.

GPU is required: flex_attention has no CPU backend, and the patched flash
attention path runs only on CUDA with bf16 compute.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Tree attention (flex_attention / triton) requires a CUDA GPU.",
)


def _make_tiny_llama(vocab_size: int, *, dtype: torch.dtype, device: torch.device):
    """Initialize a tiny Llama-style causal LM with flash_attention_2 enabled."""
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


def _baseline_logprobs(model, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Forward each sequence independently; return concatenated next-token logprobs."""
    from verl.experimental.tree_training._vocab_parallel import gather_logprobs

    input_ids = batch["input_ids"]
    attn_mask = batch["attention_mask"].to(torch.long)
    parts: list[torch.Tensor] = []
    for i in range(input_ids.size(0)):
        with torch.no_grad():
            out = model(input_ids=input_ids[i : i + 1], attention_mask=attn_mask[i : i + 1])
        # Predict input_ids[t+1] from logits[t]; cast to fp32 for stable comparison.
        shifted_logits = out.logits[0, :-1].float()
        shifted_labels = input_ids[i, 1:]
        parts.append(gather_logprobs(shifted_logits, shifted_labels))
    return torch.cat(parts, dim=0).float()


def _tree_logprobs(
    model,
    batch: dict[str, torch.Tensor],
    max_tokens_per_mb: int,
) -> torch.Tensor:
    """Pack into tree, forward once per micro-batch with flex_attention, unpack.

    Drops the trailing "spurious" logprob each sequence carries: the algorithm
    in ``_gather_packed_tree_logprobs`` unconditionally appends one transition
    logprob per node, including for the terminal node where ``next_start``
    falls back to the sentinel value ``0`` — that final entry is a prediction
    of ``input_ids[0]`` of the packed buffer from the last node's end position
    and has no semantic counterpart in independent forward. Returns only the
    first ``seq_len - 1`` logprobs per sequence (matching the docstring's
    advertised shape, which the implementation overshoots by one).
    """
    from verl.experimental.tree_training._areal_data import MicroBatchSpec
    from verl.experimental.tree_training.functional import gather_packed_tree_logprobs
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

    n_seqs, total_len = batch["input_ids"].shape

    patch_fsdp_for_tree_training(enable=True)
    all_logprobs: list[torch.Tensor] = []
    try:
        for mb in mb_list.padded_mbs:
            trie = mb["trie_node"]
            packed_input_ids = mb["input_ids"]
            position_ids = mb["position_ids"]
            padded_size = packed_input_ids.size(-1)

            tree_attn_kwargs = build_tree_attn_kwargs(
                trie,
                padded_size,
                device=packed_input_ids.device,
            )
            with torch.no_grad():
                out = model(
                    input_ids=packed_input_ids,
                    position_ids=position_ids,
                    **tree_attn_kwargs,
                )
            logits = out.logits.squeeze(0).float()
            all_logprobs.append(gather_packed_tree_logprobs(logits, trie, packed_input_ids))
    finally:
        restore_patch_fsdp_for_tree_training()

    flat = torch.cat(all_logprobs, dim=0).float()
    # All synthetic sequences are equal length; the algorithm emits total_len
    # logprobs per sequence with the spurious entry as the last one. Reshape
    # to [N, T] and drop the last column to align with the baseline shape.
    assert flat.numel() == n_seqs * total_len, (
        f"unexpected tree logprob count: got {flat.numel()}, expected {n_seqs * total_len} "
        f"({n_seqs} seqs × {total_len} entries/seq)"
    )
    return flat.view(n_seqs, total_len)[:, :-1].reshape(-1)


# build_packed_tree_batch requires max_tokens_per_mb to be a multiple of BLOCK_SIZE=128
# (and the lcm with any TP/SP parallel_size, which is 1 in this single-GPU test).
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
def test_forward_equivalence(prompt_len: int, response_len: int, max_tokens_per_mb: int) -> None:
    """Tree-packed forward should match independent forward across POR levels."""
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

    baseline = _baseline_logprobs(model, batch)
    tree = _tree_logprobs(model, batch, max_tokens_per_mb=max_tokens_per_mb)

    assert baseline.shape == tree.shape, f"shape mismatch: baseline={tuple(baseline.shape)}, tree={tuple(tree.shape)}"

    # Tolerance calibrated against AReaL upstream tests; flex_attention with
    # custom block masks introduces non-trivial numerical error vs eager attention.
    rtol, atol = 0.2, 0.2
    is_close = torch.isclose(baseline, tree, rtol=rtol, atol=atol)
    if not is_close.all():
        abs_diff = (baseline - tree).abs()
        n_bad = int((~is_close).sum().item())
        pytest.fail(
            f"forward equivalence failed at POR≈{prompt_len / (prompt_len + response_len):.2f}: "
            f"{n_bad}/{is_close.numel()} elements differ "
            f"(max abs diff={abs_diff.max().item():.6f}, "
            f"mean={abs_diff.mean().item():.6f}, "
            f"median={abs_diff.median().item():.6f})"
        )
