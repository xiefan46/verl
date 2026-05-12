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

"""End-to-end smoke test for Phase 2 (Task 2.6).

Runs one tree-training training step through the engine-layer integration
points added in Tasks 2.3-2.5, *without* spinning up the full Ray + FSDP
worker stack (that's Phase 3+ territory). Specifically exercises:

  - ``_verl_adapter.build_tree_mb_list`` on a TensorDict that mimics the
    output of ``left_right_2_no_padding`` (nested input_ids/position_ids,
    2-D loss_mask/advantages/old_log_probs/response_mask, non-tensor
    metadata for dp_size / batch_num_tokens / global_batch_size).
  - ``patch_fsdp_for_tree_training`` global monkey-patch + later restore.
  - ``build_tree_model_inputs`` → HF model forward → ``unpack_tree_logprobs``.
  - ``losses.ppo_loss`` dispatch into the tree branch (``_ppo_loss_tree``).
  - ``loss.backward()`` flows gradients through the flex_attention path.

Verifies: loss is finite, backward produces finite grads on every parameter,
and ``tree_token_ratio`` is computable.

Requires a CUDA GPU (flex_attention CPU has no backend; flash_attention_2
is GPU-only). Skipped off-CUDA at module level.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest
import torch
from tensordict import TensorDict

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
    return model


def _build_engine_tensor_dict(
    num_prompts: int,
    rollouts_per_prompt: int,
    prompt_len: int,
    response_len: int,
    vocab_size: int,
    *,
    device: torch.device,
    seed: int = 42,
) -> TensorDict:
    """Construct a TensorDict mimicking the layout post-``left_right_2_no_padding``.

    Each sequence is full length ``prompt_len + response_len``. Sequences sharing
    a prompt group identical prompt tokens; response tokens are distinct. Nested
    representation for input_ids / position_ids; 2-D response-only layout for
    advantages / old_log_probs / loss_mask / response_mask — matching what the
    real engine sees at ``forward_backward_batch`` entry.
    """
    from verl.utils import tensordict_utils as tu

    n_seqs = num_prompts * rollouts_per_prompt
    total_len = prompt_len + response_len
    gen = torch.Generator(device=device).manual_seed(seed)

    # Sample tokens excluding pad_token=0 by drawing from [1, vocab_size).
    def _sample(shape):
        return torch.randint(1, vocab_size, shape, generator=gen, device=device)

    prompts = _sample((num_prompts, prompt_len))
    responses = _sample((n_seqs, response_len))

    # Build nested input_ids: each sequence is full length total_len.
    seq_list = []
    for p in range(num_prompts):
        for r in range(rollouts_per_prompt):
            row = p * rollouts_per_prompt + r
            seq = torch.cat([prompts[p], responses[row]])
            seq_list.append(seq.to(torch.long))
    input_ids_nested = torch.nested.as_nested_tensor(seq_list, layout=torch.jagged)
    position_ids_nested = torch.nested.as_nested_tensor(
        [torch.arange(total_len, device=device, dtype=torch.long) for _ in range(n_seqs)],
        layout=torch.jagged,
    )

    # 2-D response-only fields (matches left_right_2_no_padding output).
    response_mask = torch.ones(n_seqs, response_len, dtype=torch.long, device=device)
    advantages = torch.randn(n_seqs, response_len, generator=gen, device=device, dtype=torch.float32) * 0.1
    old_log_probs = torch.randn(n_seqs, response_len, generator=gen, device=device, dtype=torch.float32) * 0.5

    td = TensorDict(
        {
            "input_ids": input_ids_nested,
            "position_ids": position_ids_nested,
            "response_mask": response_mask,
            "loss_mask": response_mask,  # alias as left_right_2_no_padding sets
            "advantages": advantages,
            "old_log_probs": old_log_probs,
        },
        batch_size=[n_seqs],
    )

    # Non-tensor metadata that the engine sets via tu.assign_non_tensor before
    # calling build_tree_mb_list / ppo_loss. Mimic that here.
    tu.assign_non_tensor_data(td, "dp_size", 1)
    tu.assign_non_tensor_data(td, "batch_num_tokens", int(response_mask.sum().item()))
    tu.assign_non_tensor_data(td, "global_batch_size", n_seqs)
    tu.assign_non_tensor_data(td, "temperature", 1.0)
    tu.assign_non_tensor_data(td, "pad_mode", "no_padding")
    return td


@dataclass
class _PolicyLossCfg:
    loss_mode: str = "vanilla"
    clip_cov_ratio: float = 0.0002
    clip_cov_lb: float = 1.0
    clip_cov_ub: float = 5.0
    kl_cov_ratio: float = 0.0002
    ppo_kl_coef: float = 0.1

    def get(self, key, default=None):
        return getattr(self, key, default)


@dataclass
class _ActorCfgShim:
    """Minimal ActorConfig-shaped object the loss functions read.

    We deliberately avoid instantiating the real ``ActorConfig`` because that
    requires a populated optimizer / engine / model sub-config tree. ``ppo_loss``
    only touches a handful of fields; we provide just those.

    ``BaseConfig`` exposes a Mapping-like ``.get(key, default)``; verl's
    ``compute_policy_loss_vanilla`` reads e.g. ``config.get("clip_ratio_c", 3.0)``,
    so the shim must support the same interface.
    """

    clip_ratio: float = 0.2
    clip_ratio_low: float = 0.2
    clip_ratio_high: float = 0.2
    clip_ratio_c: float = 3.0
    loss_agg_mode: str = "token-mean"
    loss_scale_factor: float | None = None
    use_kl_loss: bool = False
    kl_loss_coef: float = 0.001
    kl_loss_type: str = "low_var_kl"
    entropy_coeff: float = 0.0
    policy_loss: _PolicyLossCfg = field(default_factory=_PolicyLossCfg)
    global_batch_info: dict = field(default_factory=dict)

    def get(self, key, default=None):
        return getattr(self, key, default)


def test_tree_training_e2e_one_step():
    """One tree training forward + backward, end-to-end through the verl glue."""
    from verl.experimental.tree_training._verl_adapter import (
        build_tree_mb_list,
        build_tree_model_inputs,
        unpack_tree_logprobs,
    )
    from verl.experimental.tree_training.module_fsdp import (
        patch_fsdp_for_tree_training,
        restore_patch_fsdp_for_tree_training,
    )
    from verl.workers.utils.losses import ppo_loss

    device = torch.device("cuda")
    dtype = torch.bfloat16
    vocab_size = 1024

    # Build synthetic engine-side TensorDict (4 prompts × 4 rollouts = 16 seqs,
    # each of length 64 prompt + 64 response = 128 tokens; total 2048 tokens,
    # fits in a single trie of max_tokens_per_mb=2048).
    td = _build_engine_tensor_dict(
        num_prompts=2,
        rollouts_per_prompt=4,
        prompt_len=64,
        response_len=64,
        vocab_size=vocab_size,
        device=device,
    )

    micro_batches, tree_metrics = build_tree_mb_list(td, max_tokens_per_mb=2048)
    assert len(micro_batches) >= 1, "expected at least one tree micro-batch"
    assert "tree_token_ratio" in tree_metrics
    assert 0.0 < tree_metrics["tree_token_ratio"] <= 1.0, (
        f"tree_token_ratio outside (0, 1]: {tree_metrics['tree_token_ratio']:.4f}"
    )
    print(f"\n[tree_token_ratio] {tree_metrics['tree_token_ratio']:.4f}")

    # Tiny Llama for the forward; requires_grad on all params so backward populates .grad.
    model = _make_tiny_llama(vocab_size, dtype=dtype, device=device)
    model.train()  # not eval(), so dropout etc. would activate — but tiny Llama has none.

    cfg = _ActorCfgShim()

    patch_fsdp_for_tree_training(enable=True)
    try:
        total_loss_value = 0.0
        for mb in micro_batches:
            # Build the kwargs dict the model forward expects.
            model_inputs, output_args = build_tree_model_inputs(mb, device, extra_inputs={"temperature": 1.0})

            # Forward — exercise the patched flash_attention path.
            with torch.autocast(device_type="cuda", dtype=dtype):
                raw = model(**model_inputs, use_cache=False)
            logits = raw.logits.squeeze(0).float()

            # Unpack logprobs back into trie order.
            log_probs_flat = unpack_tree_logprobs(
                logits,
                output_args["trie"],
                output_args["packed_input_ids"],
                temperature=1.0,
            )

            # Assemble model_output dict matching what _forward_step_tree produces.
            model_output = {
                "log_probs": log_probs_flat,
                "is_tree_packed": True,
                "trie": output_args["trie"],
                "advantages_packed": mb["advantages"],
                "old_log_probs_packed": mb["old_log_probs"],
                "response_mask_packed": mb["response_mask"],
            }

            # Call into the real ppo_loss; it dispatches to _ppo_loss_tree
            # because is_tree_packed=True.
            loss, metrics = ppo_loss(cfg, model_output, mb, dp_group=None)

            assert torch.isfinite(loss).all(), f"loss not finite: {loss}"
            print(f"[mb] loss={loss.item():.6f}  metrics_keys={sorted(metrics.keys())}")

            loss.backward()
            total_loss_value += loss.item()

        # Verify gradients flowed to every parameter that received forward.
        nan_or_zero = []
        for name, p in model.named_parameters():
            if p.grad is None:
                nan_or_zero.append(f"{name}: grad is None")
            elif not torch.isfinite(p.grad).all():
                nan_or_zero.append(f"{name}: grad has non-finite entries")
        assert not nan_or_zero, "non-finite or missing grads:\n" + "\n".join(nan_or_zero[:10])

        # Total loss should also be finite.
        assert torch.isfinite(torch.tensor(total_loss_value)).all()
    finally:
        restore_patch_fsdp_for_tree_training()
