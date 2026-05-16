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

"""Reproducer for the Phase 4 production correctness bug.

Phase 4 dense vs tree training runs showed actor/entropy=1.21 (tree) vs 0.156
(dense) and actor/kl_loss=5.4 (tree) vs ~0.001 (dense), all at step 1 where
actor weights == ref weights. Phase 3.1 equivalence test passed atol=0.005,
yet production is off by 8x — proving Phase 3.1's test had a convention bug
matching the tree path's bug (both used the same wrong convention so they
"agreed").

Two indexing conventions for per-token logprobs in a length-L sequence:

  - Convention A: nested[k] = log P(input_ids[k] | input_ids[0..k-1])
    "log_prob of predicting token AT position k" (uses logits at k-1)
    Length L; position 0 is sentinel/undefined.

  - Convention B: nested[k] = log P(input_ids[k+1] | input_ids[0..k])
    "log_prob of predicting NEXT token after position k" (uses logits at k)
    Length L; position L-1 is garbage (label rolled with wraparound).

Verl's production dense path uses convention B (via input_ids_rmpad_rolled
and logprobs_from_logits, then nested_tensor_from_jagged of length L per
row). The trainer's no_padding_2_padding slice
``values[seq_offset - resp_len - 1 : seq_offset - 1]`` is calibrated for
convention B — it extracts row positions [prompt_len-1, prompt_len+resp_len-1)
which under convention B correspond to log_probs of response tokens
[0, resp_len). Position L-1 (the garbage slot) is the exclusive endpoint
and never read.

Our ``assemble_tree_per_seq_to_nested`` (commit 03106821) PREPENDS sentinel
to the L-1 transitions tensor, producing convention A output. When the
trainer applies the same slice, it gets a shifted window: slot 0 = last
prompt token, slots 1..resp_len-1 = first (resp_len-1) response tokens,
MISSING the last response token.

This test:
  1. Builds two parallel nested layouts (convention A and convention B)
     from the SAME per-token logprobs.
  2. Applies the no_padding_2_padding slice to both.
  3. Asserts only convention B matches the expected response window.

It also verifies entropy under both conventions to characterize the
production gap directly.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Tree training pipeline requires CUDA (flex_attention).",
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


def _slice_response_window(
    nested_values_2d: torch.Tensor,
    prompt_len: int,
    resp_len: int,
) -> torch.Tensor:
    """Apply verl's no_padding_2_padding slice convention.

    Slice ``values[seq_offset - resp_len - 1 : seq_offset - 1]`` per sequence,
    which in within-row terms is positions ``[prompt_len-1, prompt_len+resp_len-1)``
    of length ``resp_len``.
    """
    return nested_values_2d[:, prompt_len - 1 : prompt_len + resp_len - 1]


def _verl_dense_logprob_and_entropy_per_row(
    model,
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Emulate verl's production dense path: convention B per-row layout.

    Verl computes log_probs via ``logprobs_from_logits(logits, labels=input_ids_rmpad_rolled)``
    where labels are rolled by -1 (so labels[t] == input_ids[t+1]).
    log_probs[t] = log P(input_ids[t+1] | logits[t]). Then wraps with
    nested_tensor_from_jagged using cu_seqlens of the original input_ids.

    Each per-row tensor has length seq_len (= L). Position k stores
    log_prob predicting the token at position k+1 (convention B).
    Position L-1 is garbage (wraparound on the rolled labels).
    """
    from verl.experimental.tree_training._vocab_parallel import gather_logprobs

    input_ids = batch["input_ids"]
    attn_mask = batch["attention_mask"].to(torch.long)
    bsz, total_len = input_ids.shape

    log_probs_rows = torch.zeros(bsz, total_len, device=input_ids.device, dtype=torch.float32)
    entropy_rows = torch.zeros(bsz, total_len, device=input_ids.device, dtype=torch.float32)

    for i in range(bsz):
        with torch.no_grad():
            out = model(input_ids=input_ids[i : i + 1], attention_mask=attn_mask[i : i + 1])
        # Roll labels by -1 to align label[t] = input_ids[t+1] (with wraparound at last).
        labels = torch.roll(input_ids[i], shifts=-1, dims=0)  # [L]
        # log_probs[t] = log P(labels[t] | logits[t]) = log P(input_ids[t+1] | logits[t]) for t in [0, L-1)
        # log_probs[L-1] = garbage (label is input_ids[0], wraparound).
        logits_full = out.logits[0].float()  # [L, V]
        per_seq_logprobs = gather_logprobs(logits_full, labels)  # [L]
        log_probs_rows[i, :] = per_seq_logprobs

        # Entropy at position k uses logits at position k (entropy of "next token" distribution).
        log_probs_full = torch.log_softmax(logits_full, dim=-1)
        probs_full = log_probs_full.exp()
        entropy_seq = -(probs_full * log_probs_full).sum(dim=-1)  # [L]
        entropy_rows[i, :] = entropy_seq.float()

    return log_probs_rows, entropy_rows


def _tree_pipeline_nested(
    model,
    batch: dict[str, torch.Tensor],
    *,
    max_tokens_per_mb: int,
    with_entropy: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run the actual tree pipeline (forward + assemble_tree_per_seq_to_nested)."""
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
    entropy_nested = None
    if with_entropy:
        entropy_nested = assemble_tree_per_seq_to_nested(per_mb_entropy_dicts, offsets=offsets, sentinel=0.0)
    return log_probs_nested, entropy_nested


@pytest.mark.parametrize(
    "prompt_len,response_len,max_tokens_per_mb",
    [
        (64, 64, 1024),
        (32, 96, 1024),
        (96, 32, 1024),
    ],
)
def test_tree_response_window_matches_dense_production(
    prompt_len: int, response_len: int, max_tokens_per_mb: int
) -> None:
    """The response-window slice of tree-path output must match dense production.

    This is the test Phase 3.1 should have been but wasn't. Phase 3.1 used a
    convention-A baseline that happened to match the tree path's convention-A
    bug. Here we explicitly use verl's production convention-B layout.
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
    bsz, total_len = batch["input_ids"].shape

    model = _make_tiny_llama(vocab_size, dtype=dtype, device=device)

    # Dense production layout (convention B).
    dense_log_probs_2d, dense_entropy_2d = _verl_dense_logprob_and_entropy_per_row(model, batch)

    # Tree pipeline output.
    tree_log_probs_nested, tree_entropy_nested = _tree_pipeline_nested(
        model, batch, max_tokens_per_mb=max_tokens_per_mb, with_entropy=True
    )
    tree_log_probs_2d = tree_log_probs_nested.values().view(bsz, total_len)
    tree_entropy_2d = tree_entropy_nested.values().view(bsz, total_len)

    # Slice both per the trainer's no_padding_2_padding convention.
    dense_resp_logprobs = _slice_response_window(dense_log_probs_2d, prompt_len, response_len)
    tree_resp_logprobs = _slice_response_window(tree_log_probs_2d, prompt_len, response_len)
    dense_resp_entropy = _slice_response_window(dense_entropy_2d, prompt_len, response_len)
    tree_resp_entropy = _slice_response_window(tree_entropy_2d, prompt_len, response_len)

    # Diagnostic prints — show the off-by-one signature clearly when test fails.
    logprob_diff = (tree_resp_logprobs - dense_resp_logprobs).abs()
    entropy_diff = (tree_resp_entropy - dense_resp_entropy).abs()
    print(
        f"\n[response-window diff prompt_len={prompt_len} resp_len={response_len}] "
        f"logprob max={logprob_diff.max().item():.4f} mean={logprob_diff.mean().item():.4f} | "
        f"entropy max={entropy_diff.max().item():.4f} mean={entropy_diff.mean().item():.4f}"
    )
    # Mean over response window (mimics actor/entropy aggregation).
    print(
        f"  dense response_entropy_mean={dense_resp_entropy.mean().item():.4f} | "
        f"tree response_entropy_mean={tree_resp_entropy.mean().item():.4f}"
    )

    # Hard assertion: must match within Phase 1 tolerance.
    torch.testing.assert_close(
        tree_resp_logprobs.float(),
        dense_resp_logprobs.float(),
        atol=0.01,
        rtol=0.01,
        msg="Tree response-window log_probs diverge from dense production layout",
    )
    torch.testing.assert_close(
        tree_resp_entropy.float(),
        dense_resp_entropy.float(),
        atol=0.01,
        rtol=0.01,
        msg="Tree response-window entropy diverges from dense production layout",
    )


def test_assemble_uses_convention_b_layout() -> None:
    """Direct check: ``assemble_tree_per_seq_to_nested`` must produce convention B.

    Construct a 2-row toy example where we know the expected transitions
    exactly, and verify the assembled nested tensor places transitions[k]
    at row position k (NOT k+1).
    """
    from verl.experimental.tree_training._verl_adapter import assemble_tree_per_seq_to_nested

    # Row 0: length 5 → 4 transitions = [11, 12, 13, 14]
    # Row 1: length 3 → 2 transitions = [21, 22]
    transitions_0 = torch.tensor([11.0, 12.0, 13.0, 14.0])
    transitions_1 = torch.tensor([21.0, 22.0])

    offsets = torch.tensor([0, 5, 8], dtype=torch.long)  # rows of length 5, 3
    per_mb = [{0: transitions_0, 1: transitions_1}]

    nested = assemble_tree_per_seq_to_nested(per_mb, offsets=offsets, sentinel=0.0)
    values = nested.values()

    # Convention B expected: row[k] = transitions[k] for k in [0, L-1), garbage at L-1
    # Row 0 expected: [11, 12, 13, 14, *garbage*]
    # Row 1 expected: [21, 22, *garbage*]
    # Total: [11, 12, 13, 14, garbage, 21, 22, garbage]
    expected_non_garbage = torch.tensor([11.0, 12.0, 13.0, 14.0, 21.0, 22.0])
    non_garbage_positions = torch.tensor([0, 1, 2, 3, 5, 6])
    actual_non_garbage = values[non_garbage_positions]
    print(f"\n[convention check] values = {values.tolist()}")
    print(f"  expected non-garbage = {expected_non_garbage.tolist()}")
    print(f"  actual non-garbage = {actual_non_garbage.tolist()}")
    torch.testing.assert_close(actual_non_garbage, expected_non_garbage)
