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

"""Phase L: multi-DP regression for the MagiAttention tree path.

Designed to be **torchrun-launched** with ``--nproc_per_node=2``:

    cd /root/verl
    NCCL_NVLS_ENABLE=0 torchrun --nproc_per_node=2 \\
        tests/experimental/tree_training/test_magi_multi_dp_on_gpu.py

Each rank runs an FSDP2-wrapped tiny model through the tree path. The test
deliberately gives one rank a real trie and the other a dummy trie (empty
``all_sequence_ids``) so that the dummy-trie loss path
(``loss = logits.sum() * 0.0``, see commit ``e9d24f2e``) is exercised
end-to-end with FSDP gradient all-reduce.

Validates:

* L.T1 Multi-DP fwd+bwd runs without ``RuntimeError: element 0 ... does not
  require grad and does not have a grad_fn`` (Phase 4 bug).
* L.T2 Optimizer step actually changes parameters on the rank that had a real
  trie. The dummy-trie rank's gradient should be exactly zero on the relevant
  parameters.
* L.T3 The tree_token_ratio metric aggregates correctly across ranks
  (skipped in this minimal version; covered by the e2e run).

Single-file standalone — does NOT spin up a full FSDPEngine. Manually wraps
a tiny Llama in FSDP2 and exercises the data flow.
"""

from __future__ import annotations

import os
import sys

# Skip cleanly if not launched via torchrun (i.e. WORLD_SIZE not set or == 1).
_WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "0"))
if _WORLD_SIZE < 2:
    print(
        "test_magi_multi_dp_on_gpu.py requires torchrun --nproc_per_node>=2; got WORLD_SIZE={}. Skipping.".format(
            _WORLD_SIZE
        ),
        file=sys.stderr,
    )
    sys.exit(0)

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
from torch.distributed._composable.fsdp import fully_shard  # noqa: E402
from torch.distributed.device_mesh import DeviceMesh  # noqa: E402

try:
    import magi_attention  # noqa: F401, E402
except ImportError as exc:
    print(f"magi_attention not installed: {exc}", file=sys.stderr)
    sys.exit(0)


def _init_dist() -> tuple[int, int]:
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    return rank, world_size


def _build_tiny_llama_for_fsdp():
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.for_model(
        "llama",
        vocab_size=512,
        hidden_size=128,
        intermediate_size=256,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=2,
        max_position_embeddings=512,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
        torch_dtype="bfloat16",
        _attn_implementation="Magi_Tree_Attention",
    )
    model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16).cuda()
    return model, config


def _apply_fsdp2(model, mesh):
    """Wrap each decoder layer + the root model with fully_shard on the dp_cp mesh."""
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer

    for module in model.modules():
        if isinstance(module, LlamaDecoderLayer):
            fully_shard(module, mesh=mesh)
    fully_shard(model, mesh=mesh)
    return model


def main():
    rank, world_size = _init_dist()
    print(f"[rank {rank}/{world_size}] starting Phase L multi-DP test", flush=True)

    # Lazy imports (need dist to be init first for some).
    from verl.experimental.tree_training._areal_data import MicroBatchSpec
    from verl.experimental.tree_training._magi_backend import (
        TreeCPContext,
        register_tree_attention,
        tree_attn_scope,
    )
    from verl.experimental.tree_training._verl_adapter import build_tree_model_inputs
    from verl.experimental.tree_training.tree import build_packed_tree_batch

    # Build the (dp, cp) mesh; V1 cp_size=1, so cp dim is trivial.
    cp_size = 1
    dp_size = world_size // cp_size
    mesh = DeviceMesh(
        device_type="cuda",
        mesh=torch.arange(world_size).reshape(dp_size, cp_size),
        mesh_dim_names=("dp", "cp"),
    )
    mesh["dp", "cp"]._flatten("dp_cp")

    # Build model + FSDP2-wrap.
    model, config = _build_tiny_llama_for_fsdp()
    model = _apply_fsdp2(model, mesh["dp_cp"])

    # Wire Magi.
    register_tree_attention()
    tree_ctx = TreeCPContext(cp_size=cp_size)
    tree_ctx.setup_model(model)

    # Synthetic batch: rank 0 gets real sequences, rank 1 gets empty (forces
    # dummy trie path on rank 1, exercising the e9d24f2e fix).
    from tests.experimental.tree_training.synthetic import make_prompt_sharing_batch

    if rank == 0:
        batch = make_prompt_sharing_batch(
            num_prompts=1,
            rollouts_per_prompt=4,
            prompt_len=32,
            response_len=32,
            vocab_size=512,
            device="cuda",
        )
        data = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"].long(),
        }
    else:
        # Empty data: zero sequences. build_packed_tree_batch will emit a dummy
        # padded mb to keep DP rank counts in sync (tree.py:_greedy_build_tries
        # + dist.all_gather guard at line ~384-392 in legacy version).
        # For Phase L we manually construct an empty data dict — the packer
        # tolerates this and produces a dummy trie.
        data = {
            "input_ids": torch.zeros(0, 64, dtype=torch.long, device="cuda"),
            "attention_mask": torch.zeros(0, 64, dtype=torch.long, device="cuda"),
        }

    mb_list = build_packed_tree_batch(
        data,
        MicroBatchSpec(max_tokens_per_mb=1024),
        dp_group=dist.group.WORLD,
        parallel_size=world_size,
    )

    # Capture pre-step params on rank 0 (the rank with real data).
    param_snapshot = None
    if rank == 0:
        first_param = next(model.parameters())
        param_snapshot = first_param.detach().clone()

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    optimizer.zero_grad(set_to_none=True)

    head_dim = config.hidden_size // config.num_attention_heads
    total_loss = torch.zeros((), device="cuda", dtype=torch.float32, requires_grad=True)

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
                "num_heads_kv": config.num_key_value_heads,
                "head_dim": head_dim,
                "cp_group": tree_ctx.cp_group,
            }
        )

        with tree_attn_scope(**scope_args):
            out = model(input_ids=packed_input_ids, position_ids=position_ids, use_cache=False)
        logits = out.logits.squeeze(0).float()

        if not trie.all_sequence_ids:
            # Dummy-trie path: keep grad_fn connected so FSDP all-reduce stays
            # in sync with the non-dummy rank. (commit e9d24f2e)
            mb_loss = logits.sum() * 0.0
        else:
            mb_loss = logits.float().mean()
        total_loss = total_loss + mb_loss

    total_loss.backward()

    # L.T1 — backward must complete without grad_fn error.
    print(f"[rank {rank}] backward OK, loss = {total_loss.item():.4f}", flush=True)
    dist.barrier()

    optimizer.step()

    # L.T2 — params on rank 0 should have changed; on rank 1 (dummy) the
    # parameter delta from THIS rank's contribution is zero, but FSDP all-reduce
    # means rank 1 sees the updated shard too. So we only validate rank 0.
    if rank == 0:
        first_param = next(model.parameters())
        delta = (first_param.detach() - param_snapshot).abs().max().item()
        assert delta > 0, f"rank 0 params did not change after optimizer.step (delta={delta})"
        print(f"[rank 0] param delta = {delta:.6f} (>0, PASS)", flush=True)

    dist.barrier()
    if rank == 0:
        print("\n=== Phase L multi-DP PASS ===")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
