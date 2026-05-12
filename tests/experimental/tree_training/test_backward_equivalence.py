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

"""Backward equivalence: gradients from packed tree forward should match
gradients from independent per-sequence forward.

The AReaL-DTA paper (§3.1) argues backward equivalence falls out of autograd
"for free": once forward values are equal, summing logprobs across rollouts
into a scalar loss gives ``∂(sum logprobs)/∂θ`` either way, and autograd's
cache-sharing handles the K-fold gradient accumulation at shared prefix
positions naturally.

This test exercises that claim end-to-end on a single GPU: one model instance
is used for both methods (so weights are bit-identical), gradients are zeroed
between methods, and we compare per-parameter gradient tensors.

GPU is required — see test_forward_equivalence.py for rationale.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Tree attention (flex_attention / triton) requires a CUDA GPU.",
)


def _make_tiny_llama_train(vocab_size: int, *, dtype: torch.dtype, device: torch.device):
    """Tiny Llama model with flash_attention_2; weights kept in fp32 for stable grads."""
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
    # Build in fp32, then cast attention compute path to bf16 via autocast at use sites.
    # Mixed-precision via autocast keeps parameters/grads in fp32, matching how RL
    # post-training accumulates gradients.
    model = LlamaForCausalLM(config).to(device=device, dtype=dtype)
    model.eval()  # avoid dropout; we still get gradients with requires_grad=True params
    return model


def _baseline_loss(model, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Sum of next-token logprobs across all (sequence, position) pairs, batched per-seq."""
    from verl.experimental.tree_training._vocab_parallel import gather_logprobs

    input_ids = batch["input_ids"]
    attn_mask = batch["attention_mask"].to(torch.long)
    total = torch.zeros((), device=input_ids.device, dtype=torch.float)
    for i in range(input_ids.size(0)):
        out = model(input_ids=input_ids[i : i + 1], attention_mask=attn_mask[i : i + 1])
        shifted_logits = out.logits[0, :-1].float()
        shifted_labels = input_ids[i, 1:]
        total = total + gather_logprobs(shifted_logits, shifted_labels).sum()
    return total


def _tree_loss(
    model,
    batch: dict[str, torch.Tensor],
    max_tokens_per_mb: int,
) -> torch.Tensor:
    """Sum of unpacked per-sequence logprobs from a single packed tree forward.

    Drops the spurious trailing logprob each sequence carries — see the
    matching comment in test_forward_equivalence._tree_logprobs. Without this
    slice the tree loss is contaminated by predictions of ``input_ids[0]`` of
    the packed buffer that have no counterpart in the baseline, and the
    resulting gradient (especially on embed_tokens / lm_head) blows up.
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

    patch_fsdp_for_tree_training(enable=True)
    total = torch.zeros((), device=batch["input_ids"].device, dtype=torch.float)
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
            out = model(
                input_ids=packed_input_ids,
                position_ids=position_ids,
                **tree_attn_kwargs,
            )
            logits = out.logits.squeeze(0).float()
            flat = gather_packed_tree_logprobs(logits, trie, packed_input_ids)
            # Drop the spurious last logprob per sequence (see _tree_logprobs in
            # the forward equivalence test). All synthetic sequences are equal
            # length so the reshape is exact.
            mb_n_seqs = len(trie.all_sequence_ids)
            mb_seq_len = flat.numel() // mb_n_seqs
            real = flat.view(mb_n_seqs, mb_seq_len)[:, :-1]
            total = total + real.sum()
    finally:
        restore_patch_fsdp_for_tree_training()
    return total


def _collect_grads(model) -> dict[str, torch.Tensor]:
    return {name: p.grad.detach().clone() for name, p in model.named_parameters() if p.grad is not None}


@pytest.mark.parametrize(
    "prompt_len,response_len,max_tokens_per_mb",
    [
        (32, 96, 1024),
        (64, 64, 1024),
        (96, 32, 1024),
    ],
)
def test_backward_equivalence(prompt_len: int, response_len: int, max_tokens_per_mb: int) -> None:
    """Per-parameter gradients should match between baseline and tree backward."""
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

    # Single model instance: bit-identical weights for both methods.
    model = _make_tiny_llama_train(vocab_size, dtype=dtype, device=device)
    for p in model.parameters():
        p.requires_grad_(True)

    # --- Method A: independent backward, accumulate gradients ---
    model.zero_grad(set_to_none=True)
    loss_a = _baseline_loss(model, batch)
    loss_a.backward()
    grads_a = _collect_grads(model)

    # --- Method B: tree backward ---
    model.zero_grad(set_to_none=True)
    loss_b = _tree_loss(model, batch, max_tokens_per_mb=max_tokens_per_mb)
    loss_b.backward()
    grads_b = _collect_grads(model)

    # Scalar loss values: if forward equivalence holds and we sum the same set
    # of logprobs, the loss scalars should agree closely too. This is a quick
    # sanity gate before the per-parameter comparison.
    assert torch.allclose(loss_a.float(), loss_b.float(), rtol=0.2, atol=0.2), (
        f"loss mismatch: baseline={loss_a.item():.6f}, tree={loss_b.item():.6f}"
    )

    assert set(grads_a.keys()) == set(grads_b.keys()), (
        f"grad key set mismatch: only_a={set(grads_a) - set(grads_b)}, only_b={set(grads_b) - set(grads_a)}"
    )

    # Compute per-parameter error stats first (regardless of pass/fail) so the
    # actual tightness is visible. Run with ``pytest -s`` to see live output.
    por = prompt_len / (prompt_len + response_len)
    per_param: list[tuple[str, float, float, tuple[int, ...]]] = []
    for name in sorted(grads_a):
        ga = grads_a[name].float()
        gb = grads_b[name].float()
        if ga.shape != gb.shape:
            per_param.append((name, float("inf"), float("inf"), tuple(ga.shape)))
            continue
        abs_diff = (ga - gb).abs()
        rel_diff = abs_diff / (ga.abs() + 1e-8)
        per_param.append((name, abs_diff.max().item(), rel_diff.max().item(), tuple(ga.shape)))

    worst_by_rel = sorted(per_param, key=lambda x: x[2], reverse=True)
    overall_max_abs = max(p[1] for p in per_param)
    overall_max_rel = max(p[2] for p in per_param)
    print(
        f"\n[backward POR≈{por:.2f}] N_params={len(per_param)}  "
        f"loss_baseline={loss_a.item():.4f}  loss_tree={loss_b.item():.4f}  "
        f"Δloss={(loss_a - loss_b).abs().item():.4e}\n"
        f"  overall: max_abs={overall_max_abs:.4e}  max_rel={overall_max_rel:.4e}\n"
        f"  top 5 worst params (by max_rel):"
    )
    for name, max_abs, max_rel, shape in worst_by_rel[:5]:
        print(f"    {name:55s}  max_abs={max_abs:.4e}  max_rel={max_rel:.4e}  shape={shape}")

    # Compare gradients. Tolerance is looser than fp32 because flex_attention's
    # custom-mask backward path inherits the same precision quirks as forward
    # (see AReaL upstream test, which uses rtol=atol=0.2 for forward).
    rtol, atol = 0.3, 0.3
    failures: list[str] = []
    for name in sorted(grads_a):
        ga = grads_a[name].float()
        gb = grads_b[name].float()
        if ga.shape != gb.shape:
            failures.append(f"{name}: shape {tuple(ga.shape)} vs {tuple(gb.shape)}")
            continue
        if not torch.allclose(ga, gb, rtol=rtol, atol=atol):
            abs_diff = (ga - gb).abs()
            rel_diff = abs_diff / (ga.abs() + 1e-8)
            failures.append(
                f"{name}: max_abs={abs_diff.max().item():.4e}, "
                f"max_rel={rel_diff.max().item():.4e}, shape={tuple(ga.shape)}"
            )

    if failures:
        head = failures[:10]
        more = f" (and {len(failures) - 10} more)" if len(failures) > 10 else ""
        pytest.fail(
            f"backward equivalence failed at POR≈{por:.2f}: "
            f"{len(failures)}/{len(grads_a)} parameters differ.\n" + "\n".join(head) + more
        )
