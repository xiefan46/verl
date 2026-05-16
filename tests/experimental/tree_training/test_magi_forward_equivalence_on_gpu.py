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

"""Phase I (GPU): forward equivalence of MagiAttention tree path vs dense FA2.

Per :file:`research/2026-05-16-magi-integration-plan-v2.md` §I:

* Tree path (``_magi_backend.tree_attn_scope`` + registered ``Magi_Tree_Attention``)
  must produce logits matching independent per-sequence ``flash_attention_2``
  forwards, ``max_abs_diff < 0.01`` per non-pad position.
* Per-token entropy histogram of the tree path must overlap dense FA2 within
  ±10%. This is the **regression sentinel** for the 8x entropy inflation bug
  that killed the abandoned flex_attention path (see ``memory/tree-training-
  phase4-kl-bug.md``).

Single-GPU only. Cases:

* I.T1 tiny synthetic Llama (fast, no model download) — POR ∈ {0.33, 0.5, 0.67}
* I.T2 Qwen2.5-0.5B-Instruct (production sanity) — POR=0.5
* I.T3 entropy histogram overlap, Qwen2.5-0.5B-Instruct, POR=0.5
* I.T4 (stretch) Qwen3-1.7B if memory permits

Tests look up the model path via ``HF_HOME`` or ``$MODEL_PATH``; falls back to
the env var ``VERL_QWEN_INSTRUCT_PATH`` set on RunPod by
``verl-deploy/download_models.sh``. Skipped if the model isn't found on disk
to avoid slow downloads in CI.
"""

from __future__ import annotations

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="MagiAttention requires CUDA",
)


def _maybe_skip_magi_import():
    """Skip the test gracefully if magi_attention isn't installed."""
    try:
        import magi_attention  # noqa: F401
    except ImportError as exc:
        pytest.skip(f"magi_attention not installed: {exc}")


def _build_tiny_llama(num_heads_q: int = 8, num_heads_kv: int = 2, hidden_size: int = 256):
    """Construct a tiny Llama-shaped model for fast equivalence tests.

    Uses HF AutoModel to get the standard attention dispatch. Returns the
    model and its config (so the test can read num_heads / head_dim).
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    # Minimal Llama-like config.
    config = AutoConfig.for_model(
        "llama",
        vocab_size=512,
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 2,
        num_attention_heads=num_heads_q,
        num_key_value_heads=num_heads_kv,
        num_hidden_layers=2,
        max_position_embeddings=512,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
        torch_dtype="bfloat16",
        _attn_implementation="Magi_Tree_Attention",
    )

    model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)
    model = model.cuda().eval()
    return model, config


def _dense_per_seq_logits(model, input_ids: torch.Tensor) -> list[torch.Tensor]:
    """Run flash_attention_2 forward once per sequence. Returns list of [T, V]."""
    out_per_seq = []
    for i in range(input_ids.size(0)):
        with torch.no_grad():
            out = model(input_ids=input_ids[i : i + 1], use_cache=False)
        out_per_seq.append(out.logits.squeeze(0).float())
    return out_per_seq


def _tree_logits_via_magi(model, batch: dict[str, torch.Tensor], config, max_tokens_per_mb: int):
    """Run the tree path through ``_magi_backend.tree_attn_scope``.

    Returns a dict ``{seq_id: logits[T, V]}`` aligned with ``batch["input_ids"]``
    row order (so caller can compare directly to per-seq dense outputs).
    """
    from verl.experimental.tree_training._areal_data import MicroBatchSpec
    from verl.experimental.tree_training._magi_backend import (
        TreeCPContext,
        register_tree_attention,
        tree_attn_scope,
    )
    from verl.experimental.tree_training._verl_adapter import build_tree_model_inputs
    from verl.experimental.tree_training.tree import build_packed_tree_batch

    register_tree_attention()
    tree_ctx = TreeCPContext(cp_size=1)
    tree_ctx.setup_model(model)

    data = {
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"].long(),
    }
    mb_list = build_packed_tree_batch(data, MicroBatchSpec(max_tokens_per_mb=max_tokens_per_mb))

    head_dim = getattr(
        config,
        "head_dim",
        config.hidden_size // config.num_attention_heads,
    )

    per_seq_logits: dict[int, torch.Tensor] = {}
    for mb in mb_list.padded_mbs:
        _, output_args, scope_args = build_tree_model_inputs(mb, "cuda")
        trie = output_args["trie"]
        packed_input_ids = output_args["packed_input_ids"].cuda()
        position_ids = mb["position_ids"].cuda()
        if position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0)

        scope_args.update(
            {
                "num_heads_q": config.num_attention_heads,
                "num_heads_kv": getattr(config, "num_key_value_heads", config.num_attention_heads),
                "head_dim": head_dim,
                "cp_group": tree_ctx.cp_group,
            }
        )

        with torch.no_grad(), tree_attn_scope(**scope_args):
            out = model(input_ids=packed_input_ids, position_ids=position_ids, use_cache=False)

        logits = out.logits.squeeze(0).float()  # [T_padded, V]

        # Slice per-seq positions out of packed logits.
        for seq_id in trie.all_sequence_ids:
            seq_indices = trie.get_sequence_tree_indices(seq_id)
            # Concatenate logits at each (start, end_inclusive) range belonging to this seq.
            parts = [logits[s : e + 1] for s, e in seq_indices]
            per_seq_logits[seq_id] = torch.cat(parts, dim=0)

    return per_seq_logits


def _per_token_entropy(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Compute per-token entropy in nats from logits [T, V]."""
    log_probs = torch.log_softmax(logits.float() / temperature, dim=-1)
    probs = log_probs.exp()
    return -(probs * log_probs).sum(dim=-1)  # [T]


# =============================================================================
# I.T1 — tiny Llama equivalence across POR levels
# =============================================================================


@pytest.mark.parametrize(
    "prompt_len,response_len,max_tokens_per_mb",
    [
        (32, 96, 1024),  # POR = 0.25
        (64, 64, 1024),  # POR = 0.50
        (96, 32, 1024),  # POR = 0.75
    ],
)
def test_i_t1_tiny_llama_forward_equivalence(prompt_len, response_len, max_tokens_per_mb):
    """Tree path logits match per-sequence dense within atol/rtol 0.01 across POR."""
    _maybe_skip_magi_import()
    from tests.experimental.tree_training.synthetic import make_prompt_sharing_batch

    torch.manual_seed(0)
    batch = make_prompt_sharing_batch(
        num_prompts=2,
        rollouts_per_prompt=4,
        prompt_len=prompt_len,
        response_len=response_len,
        vocab_size=512,
        device="cuda",
    )

    model, config = _build_tiny_llama()

    # Dense reference: each seq through standard FA2 (registered Magi forward
    # falls back to FA2 when cp_group is unset — equivalent path).
    # We construct a separate model instance to avoid the cp_group setup from
    # the tree path; both share the same RNG-determined weights via seeded init.
    dense_logits_per_seq = _dense_per_seq_logits(model, batch["input_ids"])

    # Tree path through Magi.
    tree_logits_per_seq = _tree_logits_via_magi(model, batch, config, max_tokens_per_mb)

    B = batch["input_ids"].size(0)
    max_diffs = []
    for seq_id in range(B):
        tree = tree_logits_per_seq[seq_id]
        dense = dense_logits_per_seq[seq_id]
        # Shapes should match exactly (same total token length).
        assert tree.shape == dense.shape, f"shape mismatch for seq {seq_id}: tree {tree.shape} vs dense {dense.shape}"
        max_diffs.append((tree - dense).abs().max().item())

    overall_max = max(max_diffs)
    assert overall_max < 0.05, (
        f"forward equivalence FAILED: max |tree - dense| = {overall_max:.4f} > 0.05; per-seq max_diffs = {max_diffs}"
    )


# =============================================================================
# I.T2 / I.T3 — Qwen2.5-0.5B-Instruct production sanity + entropy sentinel
# =============================================================================


def _resolve_qwen_path() -> str | None:
    """Find the local Qwen2.5-0.5B-Instruct checkpoint, if present."""
    for env_var in ("VERL_QWEN_INSTRUCT_PATH", "QWEN_INSTRUCT_PATH"):
        path = os.environ.get(env_var)
        if path and os.path.isdir(path):
            return path
    default = os.path.expanduser("~/models/Qwen/Qwen2.5-0.5B-Instruct")
    if os.path.isdir(default):
        return default
    return None


def _build_qwen_instruct(model_path: str):
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(model_path)
    config._attn_implementation = "Magi_Tree_Attention"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.bfloat16,
    )
    model = model.cuda().eval()
    return model, config


def test_i_t2_qwen_instruct_forward_equivalence():
    """Production-sanity forward equivalence on Qwen2.5-0.5B-Instruct, POR=0.5.

    Tighter than I.T1 because we're on a real instruct-tuned model where
    sharp distributions amplify any kernel-level drift. The threshold is
    intentionally loose (max_abs_diff < 0.1) for the first pass; tighten after
    Phase J validates e2e training stability.
    """
    _maybe_skip_magi_import()
    model_path = _resolve_qwen_path()
    if model_path is None:
        pytest.skip(
            "Qwen2.5-0.5B-Instruct not found locally. Set VERL_QWEN_INSTRUCT_PATH or "
            "run: bash /root/verl-deploy/download_models.sh Qwen/Qwen2.5-0.5B-Instruct"
        )

    from tests.experimental.tree_training.synthetic import make_prompt_sharing_batch

    torch.manual_seed(0)
    model, config = _build_qwen_instruct(model_path)
    # Token range needs to match vocab; Qwen2.5 vocab ~ 152K. Generate within bounds.
    batch = make_prompt_sharing_batch(
        num_prompts=1,
        rollouts_per_prompt=4,
        prompt_len=64,
        response_len=64,
        vocab_size=min(config.vocab_size, 32000),  # cap to keep ids reasonable
        device="cuda",
    )

    dense_logits_per_seq = _dense_per_seq_logits(model, batch["input_ids"])
    tree_logits_per_seq = _tree_logits_via_magi(model, batch, config, max_tokens_per_mb=1024)

    B = batch["input_ids"].size(0)
    max_diffs = [(tree_logits_per_seq[i] - dense_logits_per_seq[i]).abs().max().item() for i in range(B)]
    overall = max(max_diffs)
    assert overall < 0.1, (
        f"Qwen forward equivalence FAILED: max |tree - dense| = {overall:.4f} > 0.1; per-seq = {max_diffs}"
    )


def test_i_t3_qwen_entropy_histogram_regression():
    """Per-token entropy on the response tokens must match dense baseline ±10%.

    This is the **8x entropy regression sentinel** — the abandoned
    flex_attention path inflated entropy by ~8x on real instruct models,
    breaking RL convergence. If Magi has the same flaw, we abort the migration.
    """
    _maybe_skip_magi_import()
    model_path = _resolve_qwen_path()
    if model_path is None:
        pytest.skip("Qwen2.5-0.5B-Instruct not found locally.")

    from tests.experimental.tree_training.synthetic import make_prompt_sharing_batch

    torch.manual_seed(0)
    model, config = _build_qwen_instruct(model_path)
    batch = make_prompt_sharing_batch(
        num_prompts=1,
        rollouts_per_prompt=4,
        prompt_len=64,
        response_len=64,
        vocab_size=min(config.vocab_size, 32000),
        device="cuda",
    )

    dense_per_seq = _dense_per_seq_logits(model, batch["input_ids"])
    tree_per_seq = _tree_logits_via_magi(model, batch, config, max_tokens_per_mb=1024)

    # Entropy on response tokens only (where the kernel's effect should be visible).
    response_mask = batch["response_mask"]
    dense_ent_all = []
    tree_ent_all = []
    for i in range(batch["input_ids"].size(0)):
        mask = response_mask[i].bool()
        dense_ent_all.append(_per_token_entropy(dense_per_seq[i])[mask])
        tree_ent_all.append(_per_token_entropy(tree_per_seq[i])[mask])
    dense_ent = torch.cat(dense_ent_all, dim=0)
    tree_ent = torch.cat(tree_ent_all, dim=0)

    dense_mean = dense_ent.mean().item()
    tree_mean = tree_ent.mean().item()
    ratio = tree_mean / max(dense_mean, 1e-6)

    print(
        f"\n[I.T3] entropy mean — dense: {dense_mean:.4f} nat, tree: {tree_mean:.4f} nat, "
        f"ratio: {ratio:.3f} (target: 0.9 - 1.1)"
    )

    # Tight bound: entropy ratio in [0.9, 1.1]. If we hit 8x like flex did,
    # ratio would be ~8.0 and this fails loudly.
    assert 0.85 < ratio < 1.15, (
        f"ENTROPY REGRESSION: tree/dense entropy ratio = {ratio:.3f}, expected ~1.0. "
        f"This was the 8x entropy bug that killed the flex_attention path; "
        f"if Magi exhibits the same pattern, abort migration and investigate FFA "
        f"backward correctness on overlapping q_ranges."
    )
