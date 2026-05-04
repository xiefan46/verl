# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
Session-tagged Process Group Registry.

Provides selective NCCL suspend/resume for process groups created within
declared "comm sessions". Groups created outside any session are tagged
"untracked" and cannot be suspended (fail-safe default).

Design overview (see research/2026-05-04-session-tagged-pg-registry-design.md):
  1. Monkey-patch torch.distributed.new_group to capture group creation
  2. Use contextvars to associate groups with caller-declared tags
  3. Store metadata in registry-side dict (do NOT wrap ProcessGroup,
     do NOT add attributes to ProcessGroup which is a C++ object)
  4. Selective suspend/resume by tag using NCCL 2.29.7+ native API
"""

import contextvars
import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.distributed as dist

from verl.utils.nccl_suspend import (
    _normalize_comm_handle,
    resume_nccl_comm,
    suspend_nccl_comm,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tag constants (use these to avoid typos)
# ---------------------------------------------------------------------------


class CommTag:
    """Standard tags for verl's NCCL communicator categories."""

    TRAINING_ACTOR = "training_actor"
    TRAINING_REF = "training_ref"
    TRAINING_CRITIC = "training_critic"
    ROLLOUT = "rollout"
    UNTRACKED = "untracked"  # default for groups created outside any session


# ---------------------------------------------------------------------------
# Session: contextvars-based tag declaration
# ---------------------------------------------------------------------------


@dataclass
class SessionInfo:
    name: str  # Human-readable session identifier (for logging/debug)
    tag: str  # Categorization tag (matches CommTag values or custom string)


_current_session: contextvars.ContextVar[Optional[SessionInfo]] = contextvars.ContextVar(
    "verl_comm_session", default=None
)


@contextmanager
def comm_session(name: str, tag: str):
    """Mark all NCCL groups created within this block with the given tag.

    Nested sessions: inner session fully overrides outer session's tag while
    its block is active. After inner session exits, outer session's tag is
    restored automatically (contextvars stack semantics).

    Example:
        with comm_session("training_actor_init", tag=CommTag.TRAINING_ACTOR):
            self.engine = FSDPEngine(...)  # all internal new_group calls tagged
    """
    info = SessionInfo(name=name, tag=tag)
    token = _current_session.set(info)
    try:
        yield info
    finally:
        _current_session.reset(token)


# ---------------------------------------------------------------------------
# GroupInfo: metadata about a tracked process group
# ---------------------------------------------------------------------------


@dataclass
class GroupInfo:
    pg: dist.ProcessGroup  # the real ProcessGroup (not wrapped)
    tag: str
    session_name: str
    suspended: bool = False
    created_at: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class NcclSuspendError(RuntimeError):
    """Raised when one or more NCCL communicators fail to suspend/resume."""

    def __init__(self, message: str, errors: list):
        super().__init__(message)
        self.errors = errors  # list of (GroupInfo, Exception)


# ---------------------------------------------------------------------------
# ProcessGroupRegistry: the core
# ---------------------------------------------------------------------------


class ProcessGroupRegistry:
    """Tracks NCCL ProcessGroups by tag, enables selective suspend/resume.

    Process-local: each process has its own registry. The class-level dict
    is keyed by pid for clarity and to support test environments where
    multiple subprocess registries might leak into the same Python class
    (e.g., fork-based testing).
    """

    # pid -> List[GroupInfo]
    _registry: dict[int, list[GroupInfo]] = {}

    # ----------------------------------------------------------------- install

    @classmethod
    def install(cls):
        """Monkey-patch torch.distributed.new_group. Idempotent.

        Must be called BEFORE any engine (FSDP/Megatron/...) initializes,
        otherwise their already-created groups won't be tracked.
        """
        if getattr(dist, "_verl_pg_registry_patched", False):
            return  # idempotent
        dist._verl_pg_registry_patched = True

        # Save original for escape hatch and internal use
        dist.original_new_group = dist.new_group  # type: ignore[attr-defined]
        dist.new_group = cls._hook_new_group  # type: ignore[assignment]

        # Use print() rather than logger.info() because verl workers default
        # to WARNING log level, which would hide INFO messages.
        print(
            f"[Registry] installed in pid={os.getpid()} (patched torch.distributed.new_group)",
            flush=True,
        )

    @classmethod
    def uninstall(cls):
        """Restore original torch.distributed.new_group. For testing."""
        if not getattr(dist, "_verl_pg_registry_patched", False):
            return
        dist.new_group = dist.original_new_group  # type: ignore[assignment]
        del dist.original_new_group
        dist._verl_pg_registry_patched = False
        logger.info("ProcessGroupRegistry uninstalled")

    # ------------------------------------------------------------------ hook

    @classmethod
    def _should_skip(cls, args: tuple, kwargs: dict) -> bool:
        """Skip wrapping for non-NCCL groups."""
        # Check backend kwarg first
        backend = kwargs.get("backend")
        # Then positional: dist.new_group(ranks, timeout, backend, ...)
        if backend is None and len(args) >= 3:
            backend = args[2]
        if backend == "gloo":
            return True
        return False

    @classmethod
    def _hook_new_group(cls, *args, **kwargs):
        """Replacement for dist.new_group that tracks NCCL groups by session."""
        # Filter 1: skip explicit gloo backend
        if cls._should_skip(args, kwargs):
            return dist.original_new_group(*args, **kwargs)

        # Always create the real group (we never wrap it)
        real_group = dist.original_new_group(*args, **kwargs)

        # Filter 2: skip world_size==1 groups (no NCCL allocation)
        try:
            ws = dist.get_world_size(real_group)
        except Exception:
            ws = None
        if ws is not None and ws <= 1:
            return real_group

        # Determine tag from current session
        sess = _current_session.get()
        if sess is None:
            # Fail-safe default: outside any session → "untracked"
            tag = CommTag.UNTRACKED
            session_name = "<anonymous>"
        else:
            tag = sess.tag
            session_name = sess.name

        info = GroupInfo(pg=real_group, tag=tag, session_name=session_name)
        cls._registry.setdefault(os.getpid(), []).append(info)

        print(
            f"[Registry] tracked group: tag={tag}, session={session_name}, world_size={ws}, pid={os.getpid()}",
            flush=True,
        )

        return real_group

    # ---------------------------------------------------------- suspend/resume

    @classmethod
    def suspend_by_tag(cls, *tags: str) -> int:
        """Suspend all NCCL communicators matching the given tag(s).

        Returns the number of groups actually suspended.
        Raises NcclSuspendError if any individual suspend fails (after
        attempting all groups).
        """
        if CommTag.UNTRACKED in tags:
            raise ValueError(
                f"Cannot operate on '{CommTag.UNTRACKED}' tag directly. "
                f"If you want to manage these groups, wrap their creation in comm_session(...)."
            )

        groups = cls._registry.get(os.getpid(), [])
        print(
            f"[Registry] suspend_by_tag(tags={list(tags)}) called in pid={os.getpid()}, "
            f"total tracked groups={len(groups)}",
            flush=True,
        )
        suspended_count = 0
        errors = []

        for info in groups:
            if info.tag not in tags or info.suspended:
                continue
            try:
                backend = info.pg._get_backend(torch.device("cuda"))
                comm_ptr = _normalize_comm_handle(backend._comm_ptr())
                if not comm_ptr or comm_ptr == 0:
                    print(
                        f"[Registry] skipping {info.session_name} (tag={info.tag}): "
                        f"_comm_ptr() returned 0 (likely lazy-init, no collective yet)",
                        flush=True,
                    )
                    continue
                if suspend_nccl_comm(comm_ptr):
                    info.suspended = True
                    suspended_count += 1
                else:
                    errors.append((info, RuntimeError("ncclCommSuspend returned non-success")))
            except Exception as e:
                print(
                    f"[Registry] failed to suspend {info.session_name} (tag={info.tag}): {e}",
                    flush=True,
                )
                errors.append((info, e))

        total = sum(1 for info in groups if info.tag in tags)
        print(
            f"[Registry] suspended {suspended_count}/{total} groups for tags={list(tags)}",
            flush=True,
        )

        if errors:
            raise NcclSuspendError(
                f"Failed to suspend {len(errors)} group(s) (others succeeded). See errors attribute for details.",
                errors,
            )
        return suspended_count

    @classmethod
    def resume_by_tag(cls, *tags: str) -> int:
        """Resume all NCCL communicators matching the given tag(s)."""
        if CommTag.UNTRACKED in tags:
            raise ValueError(f"Cannot operate on '{CommTag.UNTRACKED}' tag directly.")

        groups = cls._registry.get(os.getpid(), [])
        print(
            f"[Registry] resume_by_tag(tags={list(tags)}) called in pid={os.getpid()}, "
            f"total tracked groups={len(groups)}",
            flush=True,
        )
        resumed_count = 0
        errors = []

        for info in groups:
            if info.tag not in tags or not info.suspended:
                continue
            try:
                backend = info.pg._get_backend(torch.device("cuda"))
                comm_ptr = _normalize_comm_handle(backend._comm_ptr())
                if not comm_ptr or comm_ptr == 0:
                    continue
                if resume_nccl_comm(comm_ptr):
                    info.suspended = False
                    resumed_count += 1
                else:
                    errors.append((info, RuntimeError("ncclCommResume returned non-success")))
            except Exception as e:
                print(
                    f"[Registry] failed to resume {info.session_name} (tag={info.tag}): {e}",
                    flush=True,
                )
                errors.append((info, e))

        print(f"[Registry] resumed {resumed_count} groups for tags={list(tags)}", flush=True)

        if errors:
            raise NcclSuspendError(f"Failed to resume {len(errors)} group(s).", errors)
        return resumed_count

    # ---------------------------------------------------------- introspection

    @classmethod
    def dump(cls) -> list[dict]:
        """Return a snapshot of all tracked groups (for debug/logging)."""
        groups = cls._registry.get(os.getpid(), [])
        return [
            {
                "tag": info.tag,
                "session_name": info.session_name,
                "world_size": _safe_world_size(info.pg),
                "suspended": info.suspended,
                "created_at": info.created_at,
            }
            for info in groups
        ]

    @classmethod
    def list_untracked(cls) -> list[GroupInfo]:
        """Return groups tagged 'untracked' (created outside any session).

        Useful for sanity-checking that all expected sessions wrapped their
        engine init properly. If this list contains unexpected entries,
        some code is creating NCCL groups without declaring intent.
        """
        return [info for info in cls._registry.get(os.getpid(), []) if info.tag == CommTag.UNTRACKED]

    @classmethod
    def clear(cls):
        """Clear the registry (for testing). Does NOT uninstall the patch."""
        cls._registry.pop(os.getpid(), None)


def _safe_world_size(pg) -> Optional[int]:
    try:
        return dist.get_world_size(pg)
    except Exception:
        return None
