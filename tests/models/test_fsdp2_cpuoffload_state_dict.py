# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

"""
Regression test for GitHub issue #5995:
FSDP2 CPUOffloadPolicy + state_dict() crashes with device mismatch during update_weights.

When offload_policy=True, FSDP2 uses CPUOffloadPolicy to manage parameter placement.
However, get_per_tensor_param() calls self.module.state_dict() which crashes because
parameters are on CPU while PyTorch's state_dict hooks expect CUDA tensors.

Usage:
    # Requires at least 2 GPUs
    pytest tests/models/test_fsdp2_cpuoffload_state_dict.py -s -x
"""

import os

os.environ["NCCL_DEBUG"] = "WARN"

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import AutoConfig, AutoModelForCausalLM, Qwen3Config

from verl.trainer.config import CheckpointConfig
from verl.utils.fsdp_utils import offload_fsdp_model_to_cpu
from verl.workers.config import FSDPEngineConfig, FSDPOptimizerConfig, HFModelConfig
from verl.workers.engine import BaseEngine, EngineRegistry


def _create_model(tmp_path, config):
    """Create a small model for testing."""
    model = AutoModelForCausalLM.from_config(config)
    path = os.path.join(tmp_path, "test_model")
    model.save_pretrained(path)
    config.save_pretrained(path)
    return path


def _worker(rank: int, world_size: int, rendezvous_file: str, model_path: str,
            offload_policy: bool):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        init_method=f"file://{rendezvous_file}",
        rank=rank,
        world_size=world_size,
    )

    ref_model_config = AutoConfig.from_pretrained(model_path)
    with torch.device("meta"):
        ref_model = AutoModelForCausalLM.from_config(ref_model_config)

    model_config = HFModelConfig(path=model_path, load_tokenizer=False)
    engine_config = FSDPEngineConfig(
        forward_only=False,
        fsdp_size=world_size,
        strategy="fsdp2",
        offload_policy=offload_policy,
        param_offload=True,
        optimizer_offload=True,
    )
    optimizer_config = FSDPOptimizerConfig()
    checkpoint_config = CheckpointConfig()

    engine: BaseEngine = EngineRegistry.new(
        model_type="language_model",
        backend="fsdp2",
        model_config=model_config,
        engine_config=engine_config,
        optimizer_config=optimizer_config,
        checkpoint_config=checkpoint_config,
    )

    engine.initialize()

    # Simulate post-training state where params have been offloaded to CPU.
    if offload_policy:
        # CPUOffloadPolicy: FSDP2 manages offload automatically after forward/backward.
        # Force params to CPU to simulate this state.
        engine.module.cpu()
        torch.cuda.empty_cache()
    else:
        # Manual offload (param_offload=True): verl calls offload_fsdp_model_to_cpu()
        # after training step. Simulate this.
        offload_fsdp_model_to_cpu(engine.module)

    # This is the call that crashes with issue #5995 (offload_policy=True):
    # get_per_tensor_param() -> self.module.state_dict()
    # RuntimeError: Attempted to set the storage of a tensor on device "cpu"
    # to a storage on different device "cuda:0"
    per_tensor_params, _ = engine.get_per_tensor_param()

    ref_state_dict = ref_model.state_dict()

    for key, value in per_tensor_params:
        assert key in ref_state_dict, f"{key} not in ref_state_dict"
        assert value.shape == ref_state_dict[key].shape, (
            f"{key} shape mismatch: {value.shape} != {ref_state_dict[key].shape}"
        )
        if rank == 0:
            print(f"  {key}: {value.shape}")

    mode = "offload_policy (CPUOffloadPolicy)" if offload_policy else "param_offload (manual)"
    if rank == 0:
        print(f"SUCCESS: get_per_tensor_param() [{mode}] completed without error")

    dist.barrier()
    dist.destroy_process_group()


@pytest.mark.parametrize(
    "offload_policy",
    [False, True],
    ids=["param_offload_manual", "offload_policy_cpuoffloadpolicy"],
)
def test_fsdp2_get_per_tensor_param_with_cpuoffload(tmp_path, offload_policy):
    """Test get_per_tensor_param with both FSDP2 offload modes after params are on CPU.

    param_offload (manual): verl manages offload, load_fsdp_model_to_gpu + state_dict().
    offload_policy (CPUOffloadPolicy): FSDP2 manages offload, needs get_fsdp_full_state_dict (#5995).
    """
    world_size = min(torch.cuda.device_count(), 4)
    if world_size < 2:
        pytest.skip("Need at least 2 GPUs")

    config = Qwen3Config(num_hidden_layers=2)
    model_path = _create_model(str(tmp_path), config)
    rendezvous_file = str(tmp_path / f"rdzv_{offload_policy}")

    mp.spawn(
        fn=_worker,
        args=(world_size, rendezvous_file, model_path, offload_policy),
        nprocs=world_size,
        join=True,
    )
