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

"""Fail-fast validation tests for tree training configuration (Task 2.2).

Covers the 8 incompatible configurations that should raise immediately when
``actor.use_tree_training=True``:

  Local (in ActorConfig / FSDPActorConfig / McoreActorConfig __post_init__):
    1. actor.shuffle=True
    2. actor.use_dynamic_bsz=True
    3. actor.fsdp_config.ulysses_sequence_parallel_size > 1
    4. actor.strategy=megatron (any McoreActorConfig)
    5. tree_training.max_tokens_per_mb not a positive multiple of 128

  Cross-section (in RayPPOTrainer._validate_tree_training_compatibility):
    6. trainer.balance_batch=True
    7. use_critic (gae adv_estimator etc)
    8. actor.use_prefix_grouper=True

CPU-only — pure dataclass validation, no model / GPU / Ray.
"""

import unittest
from types import SimpleNamespace

import pytest
from hydra.errors import InstantiationException
from omegaconf import OmegaConf

from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config import (
    McoreActorConfig,
    TreeTrainingConfig,
)

# Hydra's instantiate wraps any exception raised from __post_init__ in
# InstantiationException. The wrapped exception text is preserved in the
# str of the wrapper, so ``match=`` still works on substring patterns.
# Tests that go through omega_conf_to_dataclass need to catch this wrapper;
# tests that construct dataclasses directly (e.g. TreeTrainingConfig()) get
# the raw exception type.


def _make_fsdp_actor_dict(**overrides):
    """Build a minimal FSDPActorConfig-shaped dict; merge overrides on top."""
    base = {
        "_target_": "verl.workers.config.FSDPActorConfig",
        "strategy": "fsdp",
        "ppo_mini_batch_size": 256,
        "ppo_micro_batch_size_per_gpu": 256,
        "rollout_n": 1,
        "optim": {"_target_": "verl.workers.config.FSDPOptimizerConfig", "lr": 0.1},
    }
    base.update(overrides)
    return base


def _make_mcore_actor_dict(**overrides):
    """Build a minimal McoreActorConfig-shaped dict; merge overrides on top."""
    base = {
        "_target_": "verl.workers.config.McoreActorConfig",
        "strategy": "megatron",
        "ppo_mini_batch_size": 256,
        "ppo_micro_batch_size_per_gpu": 256,
        "rollout_n": 1,
        "optim": {"_target_": "verl.workers.config.McoreOptimizerConfig", "lr": 0.1},
    }
    base.update(overrides)
    return base


class TestTreeTrainingConfigPostInit(unittest.TestCase):
    """TreeTrainingConfig.__post_init__ validates max_tokens_per_mb."""

    def test_default_ok(self):
        cfg = TreeTrainingConfig()
        self.assertEqual(cfg.max_tokens_per_mb, 4096)
        self.assertTrue(cfg.pad_to_maximum)
        self.assertEqual(cfg.chunk_size, 1024)

    def test_multiple_of_128_ok(self):
        TreeTrainingConfig(max_tokens_per_mb=2048)
        TreeTrainingConfig(max_tokens_per_mb=128)
        TreeTrainingConfig(max_tokens_per_mb=16384)

    def test_not_multiple_of_128_raises(self):
        with pytest.raises(ValueError, match="positive multiple of 128"):
            TreeTrainingConfig(max_tokens_per_mb=100)
        with pytest.raises(ValueError, match="positive multiple of 128"):
            TreeTrainingConfig(max_tokens_per_mb=129)

    def test_zero_raises(self):
        with pytest.raises(ValueError, match="positive multiple of 128"):
            TreeTrainingConfig(max_tokens_per_mb=0)

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="positive multiple of 128"):
            TreeTrainingConfig(max_tokens_per_mb=-128)


class TestActorConfigTreeTrainingChecks(unittest.TestCase):
    """ActorConfig.__post_init__ tree training compatibility checks."""

    def test_use_tree_training_disabled_no_check(self):
        """Default use_tree_training=False — no tree-related checks run."""
        cfg = omega_conf_to_dataclass(_make_fsdp_actor_dict(shuffle=True, use_dynamic_bsz=False))
        self.assertFalse(cfg.use_tree_training)

    def test_tree_training_with_shuffle_raises(self):
        with pytest.raises(InstantiationException, match="shuffle=True"):
            omega_conf_to_dataclass(_make_fsdp_actor_dict(use_tree_training=True, shuffle=True))

    def test_tree_training_with_dynamic_bsz_raises(self):
        with pytest.raises(InstantiationException, match="use_dynamic_bsz=True"):
            omega_conf_to_dataclass(
                _make_fsdp_actor_dict(
                    use_tree_training=True,
                    use_dynamic_bsz=True,
                    ppo_micro_batch_size_per_gpu=None,
                )
            )

    def test_tree_training_clean_ok(self):
        """Bare tree training enable with no conflicting fields — should succeed."""
        cfg = omega_conf_to_dataclass(_make_fsdp_actor_dict(use_tree_training=True))
        self.assertTrue(cfg.use_tree_training)
        self.assertEqual(cfg.tree_training.max_tokens_per_mb, 4096)


class TestFSDPActorConfigTreeTrainingChecks(unittest.TestCase):
    """FSDPActorConfig.__post_init__ ulysses_sp check."""

    def test_tree_training_with_ulysses_sp_raises(self):
        with pytest.raises(InstantiationException, match="ulysses_sequence_parallel_size"):
            omega_conf_to_dataclass(
                _make_fsdp_actor_dict(
                    use_tree_training=True,
                    ulysses_sequence_parallel_size=2,
                )
            )

    def test_tree_training_with_ulysses_sp_1_ok(self):
        cfg = omega_conf_to_dataclass(
            _make_fsdp_actor_dict(
                use_tree_training=True,
                ulysses_sequence_parallel_size=1,
            )
        )
        self.assertEqual(cfg.ulysses_sequence_parallel_size, 1)


class TestMcoreActorConfigTreeTrainingChecks(unittest.TestCase):
    """McoreActorConfig.__post_init__ blanket Megatron-not-supported check."""

    def test_tree_training_on_megatron_raises(self):
        with pytest.raises(InstantiationException, match="Megatron"):
            omega_conf_to_dataclass(_make_mcore_actor_dict(use_tree_training=True))

    def test_megatron_without_tree_training_ok(self):
        """Default use_tree_training=False — McoreActor still works."""
        cfg = omega_conf_to_dataclass(_make_mcore_actor_dict())
        self.assertIsInstance(cfg, McoreActorConfig)
        self.assertFalse(cfg.use_tree_training)


class TestRayTrainerCrossSectionValidation(unittest.TestCase):
    """Test RayPPOTrainer._validate_tree_training_compatibility cross-section logic.

    We call the unbound method on a SimpleNamespace shim instead of instantiating
    RayPPOTrainer (heavyweight: needs Ray, dataloaders, etc.). The helper only
    reads three attributes: ``config.trainer.balance_batch``, ``use_critic``,
    ``use_prefix_grouper``.
    """

    @staticmethod
    def _make_trainer(*, balance_batch=False, use_critic=False, use_prefix_grouper=False):
        return SimpleNamespace(
            config=OmegaConf.create({"trainer": {"balance_batch": balance_batch}}),
            use_critic=use_critic,
            use_prefix_grouper=use_prefix_grouper,
        )

    def _validate(self, trainer):
        # Import lazily so this test module doesn't pull in Ray / vLLM at collection time.
        from verl.trainer.ppo.ray_trainer import RayPPOTrainer

        RayPPOTrainer._validate_tree_training_compatibility(trainer)

    def test_all_clean_ok(self):
        self._validate(self._make_trainer())

    def test_balance_batch_raises(self):
        with pytest.raises(ValueError, match="balance_batch"):
            self._validate(self._make_trainer(balance_batch=True))

    def test_critic_raises(self):
        with pytest.raises(ValueError, match="critic"):
            self._validate(self._make_trainer(use_critic=True))

    def test_prefix_grouper_raises(self):
        with pytest.raises(ValueError, match="use_prefix_grouper"):
            self._validate(self._make_trainer(use_prefix_grouper=True))


if __name__ == "__main__":
    unittest.main()
