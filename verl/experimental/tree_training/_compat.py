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

"""Noop shims for AReaL instrumentation hooks used by the vendored tree training code.

Replaces ``areal.utils.perf_tracer`` and ``areal.utils.stats_tracker`` with
zero-overhead noops so the vendored algorithm stays unchanged. A real verl
integration (profiler / metric logger) can swap these out later.
"""

from contextlib import contextmanager
from typing import Any


def trace_perf(name: str, *, category: Any = None):
    """Noop replacement for areal.utils.perf_tracer.trace_perf decorator."""

    def decorator(func):
        return func

    return decorator


@contextmanager
def trace_scope(name: str, **kwargs: Any):
    """Noop replacement for areal.utils.perf_tracer.trace_scope context manager."""
    yield


class _StatsTrackerShim:
    """Noop replacement for areal.utils.stats_tracker module-level helpers."""

    @staticmethod
    def scalar(**kwargs: Any) -> None:
        return None

    @staticmethod
    def histogram(**kwargs: Any) -> None:
        return None


stats_tracker = _StatsTrackerShim()
