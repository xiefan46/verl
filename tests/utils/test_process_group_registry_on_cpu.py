# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""
Unit tests for verl/utils/process_group_registry.py.

These tests do NOT require a real distributed environment; we mock
torch.distributed.new_group and the NCCL ctypes wrappers.
"""

import os
from unittest import mock
from unittest.mock import MagicMock

import pytest
import torch.distributed as dist

# Important: import the module under test BEFORE patching its dependencies,
# so we can patch its module-level imports cleanly.
from verl.utils import process_group_registry as pgr


@pytest.fixture(autouse=True)
def _reset_registry():
    """Ensure each test starts and ends with a clean registry / no patch."""
    pgr.ProcessGroupRegistry.clear()
    if getattr(dist, "_verl_pg_registry_patched", False):
        pgr.ProcessGroupRegistry.uninstall()
    yield
    pgr.ProcessGroupRegistry.clear()
    if getattr(dist, "_verl_pg_registry_patched", False):
        pgr.ProcessGroupRegistry.uninstall()


def _make_fake_pg(world_size: int = 4):
    """Create a fake ProcessGroup-like object."""
    pg = MagicMock()
    backend = MagicMock()
    backend._comm_ptr.return_value = 0xDEAD_BEEF  # nonzero handle
    pg._get_backend.return_value = backend
    return pg


def _patch_new_group(world_size: int = 4):
    """Replace dist.new_group + dist.get_world_size with mocks for this test."""
    fake_pg_factory = lambda *a, **kw: _make_fake_pg(world_size)  # noqa: E731

    # Patch BEFORE install() so that install() captures our mock as 'original'
    p1 = mock.patch.object(dist, "new_group", side_effect=fake_pg_factory)
    p2 = mock.patch.object(dist, "get_world_size", return_value=world_size)
    return p1, p2


# ---------------------------------------------------------------------------
# install / uninstall
# ---------------------------------------------------------------------------


def test_install_replaces_new_group():
    p1, p2 = _patch_new_group()
    with p1, p2:
        original_before = dist.new_group
        pgr.ProcessGroupRegistry.install()
        assert dist.new_group is not original_before
        assert dist.original_new_group is original_before


def test_install_is_idempotent():
    p1, p2 = _patch_new_group()
    with p1, p2:
        pgr.ProcessGroupRegistry.install()
        hooked = dist.new_group
        pgr.ProcessGroupRegistry.install()  # second call
        assert dist.new_group is hooked  # unchanged


def test_uninstall_restores():
    p1, p2 = _patch_new_group()
    with p1, p2:
        pgr.ProcessGroupRegistry.install()
        pgr.ProcessGroupRegistry.uninstall()
        # After uninstall, dist.new_group should be back to the (mocked) original
        assert not getattr(dist, "_verl_pg_registry_patched", False)


# ---------------------------------------------------------------------------
# session-based tagging
# ---------------------------------------------------------------------------


def test_group_in_session_is_tracked_with_tag():
    p1, p2 = _patch_new_group(world_size=4)
    with p1, p2:
        pgr.ProcessGroupRegistry.install()

        with pgr.comm_session("init", tag="training_actor"):
            pg = dist.new_group(ranks=[0, 1, 2, 3])

        assert pg is not None
        snapshot = pgr.ProcessGroupRegistry.dump()
        assert len(snapshot) == 1
        assert snapshot[0]["tag"] == "training_actor"
        assert snapshot[0]["session_name"] == "init"
        assert snapshot[0]["suspended"] is False


def test_group_outside_session_is_untracked():
    p1, p2 = _patch_new_group(world_size=4)
    with p1, p2:
        pgr.ProcessGroupRegistry.install()
        dist.new_group(ranks=[0, 1, 2, 3])  # no session

        snapshot = pgr.ProcessGroupRegistry.dump()
        assert len(snapshot) == 1
        assert snapshot[0]["tag"] == pgr.CommTag.UNTRACKED


def test_session_nesting_inner_overrides():
    p1, p2 = _patch_new_group(world_size=4)
    with p1, p2:
        pgr.ProcessGroupRegistry.install()

        with pgr.comm_session("outer", tag="training"):
            dist.new_group(ranks=[0, 1])
            with pgr.comm_session("inner", tag="ref"):
                dist.new_group(ranks=[0, 1])
            dist.new_group(ranks=[0, 1])

        snapshot = pgr.ProcessGroupRegistry.dump()
        tags = [g["tag"] for g in snapshot]
        assert tags == ["training", "ref", "training"]


def test_session_after_exit_restores_outside():
    p1, p2 = _patch_new_group(world_size=4)
    with p1, p2:
        pgr.ProcessGroupRegistry.install()

        with pgr.comm_session("a", tag="training"):
            dist.new_group(ranks=[0, 1])
        # exited session
        dist.new_group(ranks=[0, 1])  # should be untracked

        snapshot = pgr.ProcessGroupRegistry.dump()
        assert snapshot[0]["tag"] == "training"
        assert snapshot[1]["tag"] == pgr.CommTag.UNTRACKED


# ---------------------------------------------------------------------------
# filtering: gloo / world_size=1 should be skipped
# ---------------------------------------------------------------------------


def test_gloo_backend_is_not_tracked():
    p1, p2 = _patch_new_group(world_size=4)
    with p1, p2:
        pgr.ProcessGroupRegistry.install()

        with pgr.comm_session("init", tag="training"):
            dist.new_group(ranks=[0, 1, 2, 3], backend="gloo")

        snapshot = pgr.ProcessGroupRegistry.dump()
        assert len(snapshot) == 0


def test_world_size_one_is_not_tracked():
    p1, p2 = _patch_new_group(world_size=1)
    with p1, p2:
        pgr.ProcessGroupRegistry.install()

        with pgr.comm_session("init", tag="training"):
            dist.new_group(ranks=[0])

        snapshot = pgr.ProcessGroupRegistry.dump()
        assert len(snapshot) == 0


# ---------------------------------------------------------------------------
# suspend / resume by tag
# ---------------------------------------------------------------------------


def test_suspend_by_tag_calls_underlying_api():
    p1, p2 = _patch_new_group(world_size=4)
    with p1, p2, mock.patch.object(pgr, "suspend_nccl_comm", return_value=True) as mock_susp:
        pgr.ProcessGroupRegistry.install()
        with pgr.comm_session("a", tag="training"):
            dist.new_group(ranks=[0, 1])
            dist.new_group(ranks=[0, 1])

        n = pgr.ProcessGroupRegistry.suspend_by_tag("training")
        assert n == 2
        assert mock_susp.call_count == 2

        # All groups should be marked suspended
        snapshot = pgr.ProcessGroupRegistry.dump()
        assert all(g["suspended"] for g in snapshot)


def test_suspend_by_tag_only_targets_matching():
    p1, p2 = _patch_new_group(world_size=4)
    with p1, p2, mock.patch.object(pgr, "suspend_nccl_comm", return_value=True):
        pgr.ProcessGroupRegistry.install()
        with pgr.comm_session("a", tag="training_actor"):
            dist.new_group(ranks=[0, 1])
        with pgr.comm_session("b", tag="training_ref"):
            dist.new_group(ranks=[0, 1])

        pgr.ProcessGroupRegistry.suspend_by_tag("training_actor")
        snapshot = pgr.ProcessGroupRegistry.dump()
        actor_g = next(g for g in snapshot if g["tag"] == "training_actor")
        ref_g = next(g for g in snapshot if g["tag"] == "training_ref")
        assert actor_g["suspended"] is True
        assert ref_g["suspended"] is False


def test_suspend_untracked_is_rejected():
    pgr.ProcessGroupRegistry.install()
    with pytest.raises(ValueError, match="untracked"):
        pgr.ProcessGroupRegistry.suspend_by_tag(pgr.CommTag.UNTRACKED)


def test_resume_only_resumes_suspended_groups():
    p1, p2 = _patch_new_group(world_size=4)
    with (
        p1,
        p2,
        mock.patch.object(pgr, "suspend_nccl_comm", return_value=True),
        mock.patch.object(pgr, "resume_nccl_comm", return_value=True) as mock_resume,
    ):
        pgr.ProcessGroupRegistry.install()

        with pgr.comm_session("a", tag="training"):
            dist.new_group(ranks=[0, 1])
            dist.new_group(ranks=[0, 1])

        pgr.ProcessGroupRegistry.suspend_by_tag("training")
        n_resumed = pgr.ProcessGroupRegistry.resume_by_tag("training")
        assert n_resumed == 2
        assert mock_resume.call_count == 2


def test_resume_skips_groups_that_were_not_suspended():
    """If a group was never suspended, resume_by_tag should not call resume on it."""
    p1, p2 = _patch_new_group(world_size=4)
    with p1, p2, mock.patch.object(pgr, "resume_nccl_comm", return_value=True) as mock_resume:
        pgr.ProcessGroupRegistry.install()
        with pgr.comm_session("a", tag="training"):
            dist.new_group(ranks=[0, 1])

        n = pgr.ProcessGroupRegistry.resume_by_tag("training")
        assert n == 0
        assert mock_resume.call_count == 0


# ---------------------------------------------------------------------------
# partial failure
# ---------------------------------------------------------------------------


def test_partial_suspend_failure_raises_aggregated():
    """If some suspend calls fail, others should still be attempted; finally raise."""
    p1, p2 = _patch_new_group(world_size=4)

    call_count = [0]

    def flaky_suspend(_handle):
        call_count[0] += 1
        if call_count[0] == 2:
            raise RuntimeError("simulated NCCL error")
        return True

    with p1, p2, mock.patch.object(pgr, "suspend_nccl_comm", side_effect=flaky_suspend):
        pgr.ProcessGroupRegistry.install()
        with pgr.comm_session("a", tag="training"):
            for _ in range(3):
                dist.new_group(ranks=[0, 1])

        with pytest.raises(pgr.NcclSuspendError) as excinfo:
            pgr.ProcessGroupRegistry.suspend_by_tag("training")

        assert len(excinfo.value.errors) == 1
        # First and third should have succeeded; second is the failure
        snapshot = pgr.ProcessGroupRegistry.dump()
        suspended_count = sum(1 for g in snapshot if g["suspended"])
        assert suspended_count == 2  # 3 attempted, 2 succeeded


# ---------------------------------------------------------------------------
# escape hatch
# ---------------------------------------------------------------------------


def test_original_new_group_bypasses_hook():
    p1, p2 = _patch_new_group(world_size=4)
    with p1, p2:
        pgr.ProcessGroupRegistry.install()

        with pgr.comm_session("a", tag="training"):
            dist.original_new_group(ranks=[0, 1])  # ← escape hatch

        snapshot = pgr.ProcessGroupRegistry.dump()
        assert len(snapshot) == 0  # not tracked


# ---------------------------------------------------------------------------
# introspection
# ---------------------------------------------------------------------------


def test_list_untracked():
    p1, p2 = _patch_new_group(world_size=4)
    with p1, p2:
        pgr.ProcessGroupRegistry.install()

        with pgr.comm_session("a", tag="training"):
            dist.new_group(ranks=[0, 1])
        # Outside session
        dist.new_group(ranks=[0, 1])
        dist.new_group(ranks=[0, 1])

        untracked = pgr.ProcessGroupRegistry.list_untracked()
        assert len(untracked) == 2


def test_dump_returns_snapshot_dict_list():
    p1, p2 = _patch_new_group(world_size=8)
    with p1, p2:
        pgr.ProcessGroupRegistry.install()
        with pgr.comm_session("init", tag="training"):
            dist.new_group(ranks=list(range(8)))
        snapshot = pgr.ProcessGroupRegistry.dump()
        assert len(snapshot) == 1
        entry = snapshot[0]
        assert {"tag", "session_name", "world_size", "suspended", "created_at"}.issubset(entry.keys())
        assert entry["tag"] == "training"
        assert entry["world_size"] == 8


# ---------------------------------------------------------------------------
# pid isolation
# ---------------------------------------------------------------------------


def test_registry_is_pid_keyed():
    """Verify registry uses os.getpid() as key."""
    p1, p2 = _patch_new_group(world_size=4)
    with p1, p2:
        pgr.ProcessGroupRegistry.install()
        with pgr.comm_session("a", tag="training"):
            dist.new_group(ranks=[0, 1])

        my_pid = os.getpid()
        assert my_pid in pgr.ProcessGroupRegistry._registry
        assert len(pgr.ProcessGroupRegistry._registry[my_pid]) == 1
