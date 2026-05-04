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
Session-tagged Process Group Registry (snapshot-diff implementation).

Provides selective NCCL suspend/resume for process groups created within
declared "comm sessions". Groups created outside any session are not
tracked and cannot be suspended (fail-safe default).

Mechanism: NO monkey-patching. On session enter we snapshot the keys of
torch.distributed.distributed_c10d._world.pg_map; on exit we diff to
identify groups created within the session, filter to NCCL backend, and
record them with the session's tag.

Why not hook torch.distributed APIs:
  - dist.new_group is bypassed by init_device_mesh's split_group path
  - _new_process_group_helper is bypassed by split_group as well
  - _register_process_group covers all three paths but is a private C++ binding
  - Snapshot diff relies only on _world.pg_map which is stable across PyTorch
    creation paths (new_group / split_group / init_process_group all write to it).

Limitations:
  - Nested sessions raise RuntimeError (not needed for verl colocated mode)
  - Lazy NCCL init: groups whose NCCL comm hasn't been allocated yet
    (no collective called) are skipped at suspend time and retried later
"""

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


class CommTag:
    """Standard tags for verl's NCCL communicator categories."""

    TRAINING_ACTOR = "training_actor"
    TRAINING_REF = "training_ref"
    TRAINING_CRITIC = "training_critic"
    ROLLOUT = "rollout"
    UNTRACKED = "untracked"  # reserved sentinel; cannot be suspended/resumed


@dataclass
class GroupInfo:
    pg: dist.ProcessGroup
    tag: str
    session_name: str
    suspended: bool = False
    created_at: float = field(default_factory=time.time)


class NcclSuspendError(RuntimeError):
    """Raised when one or more NCCL communicators fail to suspend/resume."""

    def __init__(self, message: str, errors: list):
        super().__init__(message)
        self.errors = errors  # list of (GroupInfo, Exception)


def _pg_map_snapshot() -> dict:
    """Return {pg -> backend_name} snapshot of torch.distributed _world.pg_map.

    Single point of access for testability — tests patch this helper to
    inject controllable group state without initializing torch.distributed.
    """
    try:
        return {pg: entry[0] for pg, entry in dist.distributed_c10d._world.pg_map.items()}
    except Exception:
        return {}


_session_active: bool = False


@contextmanager
def comm_session(name: str, tag: str):
    """Tag NCCL groups created within this block with the given tag.

    Implementation: snapshots _world.pg_map at enter, diffs at exit, registers
    the difference (filtered to NCCL backend) with the session's tag.

    Nested sessions raise RuntimeError. The verl colocated training pattern
    creates engines flat (one comm_session per engine init), no nesting needed.

    Example:
        with comm_session("training_actor_init", tag=CommTag.TRAINING_ACTOR):
            self.engine = FSDPEngine(...)
        # all NCCL groups created during FSDPEngine init are now tagged
    """
    global _session_active
    if _session_active:
        raise RuntimeError(
            f"comm_session('{name}', tag='{tag}') is nested inside another active session. "
            "Nested sessions are not supported by this registry."
        )

    before_keys = set(_pg_map_snapshot().keys())
    _session_active = True
    try:
        yield
    finally:
        _session_active = False
        after = _pg_map_snapshot()
        new_pgs = [pg for pg in after.keys() if pg not in before_keys]
        captured = 0
        for pg in new_pgs:
            backend_name = after[pg]
            if backend_name != "nccl":
                continue
            ProcessGroupRegistry._append(GroupInfo(pg=pg, tag=tag, session_name=name))
            captured += 1
        print(
            f"[Registry] session={name!r} tag={tag!r} captured {captured} NCCL group(s) (pid={os.getpid()})",
            flush=True,
        )


class ProcessGroupRegistry:
    """Tracks NCCL ProcessGroups by tag and enables selective suspend/resume.

    Process-local: keyed by pid so a stale class-level dict cannot leak
    across forked subprocesses (e.g., test workers).
    """

    _registry: dict[int, list[GroupInfo]] = {}

    @classmethod
    def _append(cls, info: GroupInfo) -> None:
        cls._registry.setdefault(os.getpid(), []).append(info)

    @classmethod
    def _groups(cls) -> list[GroupInfo]:
        return cls._registry.get(os.getpid(), [])

    # ---------------------------------------------------------- suspend/resume

    @classmethod
    def suspend_by_tag(cls, *tags: str) -> int:
        """Suspend all NCCL communicators whose tag matches one of `tags`.

        Returns the number of groups actually suspended (excludes already-
        suspended and lazy-init groups).
        Raises NcclSuspendError if any individual suspend fails (after
        attempting all groups).
        """
        if CommTag.UNTRACKED in tags:
            raise ValueError(
                f"Cannot operate on '{CommTag.UNTRACKED}' tag. "
                f"If you want to manage these groups, wrap their creation in comm_session(...)."
            )

        groups = cls._groups()
        print(
            f"[Registry] suspend_by_tag(tags={list(tags)}) called in pid={os.getpid()}, "
            f"total tracked groups={len(groups)}",
            flush=True,
        )
        suspended_count = 0
        errors: list[tuple[GroupInfo, Exception]] = []

        for info in groups:
            if info.tag not in tags or info.suspended:
                continue
            try:
                backend = info.pg._get_backend(torch.device("cuda"))
                comm_ptr = _normalize_comm_handle(backend._comm_ptr())
                if not comm_ptr:
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
        """Resume all NCCL communicators whose tag matches one of `tags`."""
        if CommTag.UNTRACKED in tags:
            raise ValueError(f"Cannot operate on '{CommTag.UNTRACKED}' tag directly.")

        groups = cls._groups()
        print(
            f"[Registry] resume_by_tag(tags={list(tags)}) called in pid={os.getpid()}, "
            f"total tracked groups={len(groups)}",
            flush=True,
        )
        resumed_count = 0
        errors: list[tuple[GroupInfo, Exception]] = []

        for info in groups:
            if info.tag not in tags or not info.suspended:
                continue
            try:
                backend = info.pg._get_backend(torch.device("cuda"))
                comm_ptr = _normalize_comm_handle(backend._comm_ptr())
                if not comm_ptr:
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
        return [
            {
                "tag": info.tag,
                "session_name": info.session_name,
                "world_size": _safe_world_size(info.pg),
                "suspended": info.suspended,
                "created_at": info.created_at,
            }
            for info in cls._groups()
        ]

    @classmethod
    def list_untracked(cls) -> list[dist.ProcessGroup]:
        """Return NCCL ProcessGroups in this process that we have NOT tracked.

        Computed by reading torch.distributed._world.pg_map and excluding our
        registry. Useful as a fail-safe sanity check: anything in this list
        was created outside any comm_session and would NOT be suspendable.
        """
        tracked = {info.pg for info in cls._groups()}
        return [pg for pg, backend_name in _pg_map_snapshot().items() if backend_name == "nccl" and pg not in tracked]

    @classmethod
    def clear(cls) -> None:
        """Clear the registry (for testing)."""
        cls._registry.pop(os.getpid(), None)


def _safe_world_size(pg) -> Optional[int]:
    try:
        return dist.get_world_size(pg)
    except Exception:
        return None
