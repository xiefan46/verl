# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""
Unit tests for verl/utils/process_group_registry.py (snapshot-diff impl).

These tests do NOT require a real distributed environment; we patch
`_pg_map_snapshot` to inject a controllable {pg -> backend_name} dict and
mock the NCCL ctypes wrappers.
"""

import contextlib
import os
from unittest import mock
from unittest.mock import MagicMock

import pytest

from verl.utils import process_group_registry as pgr


class FakePG:
    """Minimal ProcessGroup stand-in used by tests.

    Identity-hashable (default object hashing) so it works as a dict key.
    Holds a backend mock whose `_comm_ptr()` returns a controllable handle.
    """

    def __init__(self, world_size: int = 4, comm_ptr: int = 0xDEAD_BEEF):
        self._world_size = world_size
        self._comm_ptr_val = comm_ptr
        self._backend = MagicMock()
        self._backend._comm_ptr.return_value = comm_ptr

    def _get_backend(self, _device):
        return self._backend


@contextlib.contextmanager
def fake_pg_map(initial=None):
    """Replace `_pg_map_snapshot` with a closure over a mutable {pg: backend} dict.

    Yields the dict so tests can mutate it inside / outside sessions to
    simulate group creation in real code paths.
    """
    state = dict(initial or {})

    def _snapshot():
        return dict(state)  # snapshot semantics: caller-mutation-safe

    # Also stub get_world_size for dump()'s _safe_world_size helper.
    def _ws(pg):
        return getattr(pg, "_world_size", None)

    with (
        mock.patch.object(pgr, "_pg_map_snapshot", side_effect=_snapshot),
        mock.patch("torch.distributed.get_world_size", side_effect=_ws),
    ):
        yield state


@pytest.fixture(autouse=True)
def _reset_registry():
    """Clean state before/after each test."""
    pgr.ProcessGroupRegistry.clear()
    pgr._session_active = False
    yield
    pgr.ProcessGroupRegistry.clear()
    pgr._session_active = False


# ---------------------------------------------------------------------------
# session capture: snapshot diff fundamentals
# ---------------------------------------------------------------------------


def test_session_captures_groups_added_inside():
    with fake_pg_map() as pg_map:
        pg = FakePG()
        with pgr.comm_session("init", tag="training_actor"):
            pg_map[pg] = "nccl"  # simulate group creation inside session

        snap = pgr.ProcessGroupRegistry.dump()
        assert len(snap) == 1
        assert snap[0]["tag"] == "training_actor"
        assert snap[0]["session_name"] == "init"
        assert snap[0]["suspended"] is False


def test_session_captures_multiple_groups():
    with fake_pg_map() as pg_map:
        with pgr.comm_session("init", tag="training_actor"):
            pg_map[FakePG()] = "nccl"
            pg_map[FakePG()] = "nccl"
            pg_map[FakePG()] = "nccl"

        snap = pgr.ProcessGroupRegistry.dump()
        assert len(snap) == 3
        assert all(g["tag"] == "training_actor" for g in snap)


def test_pre_existing_groups_not_captured():
    """Groups already in pg_map BEFORE session enter are excluded by diff."""
    with fake_pg_map() as pg_map:
        pg_map[FakePG()] = "nccl"  # exists before session
        pg_map[FakePG()] = "nccl"

        with pgr.comm_session("init", tag="training"):
            pass  # nothing new created

        assert pgr.ProcessGroupRegistry.dump() == []


def test_groups_outside_session_not_tracked():
    with fake_pg_map() as pg_map:
        pg_map[FakePG()] = "nccl"  # never inside any session
        assert pgr.ProcessGroupRegistry.dump() == []


def test_two_sequential_sessions_isolated():
    with fake_pg_map() as pg_map:
        with pgr.comm_session("a", tag="training_actor"):
            pg_map[FakePG()] = "nccl"
        with pgr.comm_session("b", tag="training_ref"):
            pg_map[FakePG()] = "nccl"

        snap = pgr.ProcessGroupRegistry.dump()
        tags = sorted(g["tag"] for g in snap)
        assert tags == ["training_actor", "training_ref"]


# ---------------------------------------------------------------------------
# backend filter: only NCCL is recorded
# ---------------------------------------------------------------------------


def test_gloo_backend_skipped():
    with fake_pg_map() as pg_map:
        with pgr.comm_session("init", tag="training"):
            pg_map[FakePG()] = "gloo"
        assert pgr.ProcessGroupRegistry.dump() == []


def test_mixed_backend_only_nccl_recorded():
    with fake_pg_map() as pg_map:
        with pgr.comm_session("init", tag="training"):
            pg_map[FakePG()] = "nccl"
            pg_map[FakePG()] = "gloo"
            pg_map[FakePG()] = "nccl"

        snap = pgr.ProcessGroupRegistry.dump()
        assert len(snap) == 2  # gloo excluded


# ---------------------------------------------------------------------------
# nesting: explicitly forbidden
# ---------------------------------------------------------------------------


def test_nested_session_raises():
    with fake_pg_map():
        with pgr.comm_session("outer", tag="A"):
            with pytest.raises(RuntimeError, match="[Nn]ested"):
                with pgr.comm_session("inner", tag="B"):
                    pass


def test_session_state_recovered_after_exception_in_body():
    """If user code inside a session raises, _session_active must reset."""
    with fake_pg_map():
        with pytest.raises(ValueError):
            with pgr.comm_session("a", tag="t"):
                raise ValueError("boom")
        # subsequent session must work — no leaked active flag
        assert pgr._session_active is False
        with pgr.comm_session("b", tag="t"):
            pass


# ---------------------------------------------------------------------------
# suspend / resume by tag
# ---------------------------------------------------------------------------


def test_suspend_by_tag_calls_underlying_api():
    with fake_pg_map() as pg_map:
        with pgr.comm_session("a", tag="training"):
            pg_map[FakePG()] = "nccl"
            pg_map[FakePG()] = "nccl"

        with mock.patch.object(pgr, "suspend_nccl_comm", return_value=True) as mock_susp:
            n = pgr.ProcessGroupRegistry.suspend_by_tag("training")

        assert n == 2
        assert mock_susp.call_count == 2
        snap = pgr.ProcessGroupRegistry.dump()
        assert all(g["suspended"] for g in snap)


def test_suspend_by_tag_only_targets_matching():
    with fake_pg_map() as pg_map:
        with pgr.comm_session("a", tag="training_actor"):
            pg_map[FakePG()] = "nccl"
        with pgr.comm_session("b", tag="training_ref"):
            pg_map[FakePG()] = "nccl"

        with mock.patch.object(pgr, "suspend_nccl_comm", return_value=True):
            pgr.ProcessGroupRegistry.suspend_by_tag("training_actor")

        snap = pgr.ProcessGroupRegistry.dump()
        actor_g = next(g for g in snap if g["tag"] == "training_actor")
        ref_g = next(g for g in snap if g["tag"] == "training_ref")
        assert actor_g["suspended"] is True
        assert ref_g["suspended"] is False


def test_suspend_untracked_is_rejected():
    with pytest.raises(ValueError, match="untracked"):
        pgr.ProcessGroupRegistry.suspend_by_tag(pgr.CommTag.UNTRACKED)


def test_resume_untracked_is_rejected():
    with pytest.raises(ValueError, match="untracked"):
        pgr.ProcessGroupRegistry.resume_by_tag(pgr.CommTag.UNTRACKED)


def test_resume_resumes_previously_suspended():
    with fake_pg_map() as pg_map:
        with pgr.comm_session("a", tag="training"):
            pg_map[FakePG()] = "nccl"
            pg_map[FakePG()] = "nccl"

        with (
            mock.patch.object(pgr, "suspend_nccl_comm", return_value=True),
            mock.patch.object(pgr, "resume_nccl_comm", return_value=True) as mock_resume,
        ):
            pgr.ProcessGroupRegistry.suspend_by_tag("training")
            n_resumed = pgr.ProcessGroupRegistry.resume_by_tag("training")

        assert n_resumed == 2
        assert mock_resume.call_count == 2


def test_resume_skips_groups_that_were_not_suspended():
    with fake_pg_map() as pg_map:
        with pgr.comm_session("a", tag="training"):
            pg_map[FakePG()] = "nccl"

        with mock.patch.object(pgr, "resume_nccl_comm", return_value=True) as mock_resume:
            n = pgr.ProcessGroupRegistry.resume_by_tag("training")

        assert n == 0
        assert mock_resume.call_count == 0


# ---------------------------------------------------------------------------
# lazy NCCL init: _comm_ptr() == 0
# ---------------------------------------------------------------------------


def test_suspend_skips_lazy_init_groups():
    """Groups whose backend._comm_ptr() returns 0 are skipped (NCCL not yet allocated)."""
    with fake_pg_map() as pg_map:
        with pgr.comm_session("a", tag="training"):
            pg_map[FakePG(comm_ptr=0)] = "nccl"

        with mock.patch.object(pgr, "suspend_nccl_comm", return_value=True) as mock_susp:
            n = pgr.ProcessGroupRegistry.suspend_by_tag("training")

        assert n == 0
        assert mock_susp.call_count == 0


# ---------------------------------------------------------------------------
# partial failure
# ---------------------------------------------------------------------------


def test_partial_suspend_failure_raises_aggregated():
    """If some suspend calls fail, others still attempted; finally raise NcclSuspendError."""
    with fake_pg_map() as pg_map:
        with pgr.comm_session("a", tag="training"):
            for _ in range(3):
                pg_map[FakePG()] = "nccl"

        call_count = [0]

        def flaky_suspend(_handle):
            call_count[0] += 1
            if call_count[0] == 2:
                raise RuntimeError("simulated NCCL error")
            return True

        with mock.patch.object(pgr, "suspend_nccl_comm", side_effect=flaky_suspend):
            with pytest.raises(pgr.NcclSuspendError) as excinfo:
                pgr.ProcessGroupRegistry.suspend_by_tag("training")

        assert len(excinfo.value.errors) == 1
        snap = pgr.ProcessGroupRegistry.dump()
        suspended_count = sum(1 for g in snap if g["suspended"])
        assert suspended_count == 2  # 3 attempted, 2 succeeded


# ---------------------------------------------------------------------------
# introspection
# ---------------------------------------------------------------------------


def test_list_untracked_returns_nccl_pgs_outside_session():
    """list_untracked() = NCCL pgs in pg_map but not in our registry."""
    with fake_pg_map() as pg_map:
        # Two NCCL groups created outside any session
        out1 = FakePG()
        out2 = FakePG()
        pg_map[out1] = "nccl"
        pg_map[out2] = "nccl"
        # And one gloo that should NOT count as untracked
        pg_map[FakePG()] = "gloo"
        # And one inside session (tracked)
        with pgr.comm_session("a", tag="training"):
            pg_map[FakePG()] = "nccl"

        untracked = pgr.ProcessGroupRegistry.list_untracked()
        assert len(untracked) == 2
        assert set(untracked) == {out1, out2}


def test_dump_returns_snapshot_dict_list():
    with fake_pg_map() as pg_map:
        with pgr.comm_session("init", tag="training"):
            pg_map[FakePG(world_size=8)] = "nccl"

        snap = pgr.ProcessGroupRegistry.dump()
        assert len(snap) == 1
        entry = snap[0]
        assert {"tag", "session_name", "world_size", "suspended", "created_at"}.issubset(entry.keys())
        assert entry["tag"] == "training"
        assert entry["world_size"] == 8


# ---------------------------------------------------------------------------
# pid isolation
# ---------------------------------------------------------------------------


def test_registry_is_pid_keyed():
    with fake_pg_map() as pg_map:
        with pgr.comm_session("a", tag="training"):
            pg_map[FakePG()] = "nccl"

        my_pid = os.getpid()
        assert my_pid in pgr.ProcessGroupRegistry._registry
        assert len(pgr.ProcessGroupRegistry._registry[my_pid]) == 1
