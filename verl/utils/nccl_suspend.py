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
NCCL Communicator Suspend/Resume utilities.

Uses NCCL 2.29.7+ native ncclCommSuspend/ncclCommResume API to release
GPU memory held by idle NCCL communicators during colocated training/inference.
"""

import ctypes
import logging

from verl.utils.device import get_torch_device

logger = logging.getLogger(__name__)

NCCL_SUSPEND_MEM = 0x01  # Release dynamic GPU memory allocations
NCCL_SUCCESS = 0

_nccl_lib = None


def _get_nccl_lib():
    """Lazily load libnccl and define function signatures."""
    global _nccl_lib
    if _nccl_lib is not None:
        return _nccl_lib

    try:
        lib = ctypes.CDLL("libnccl.so.2")
    except OSError:
        logger.warning("Failed to load libnccl.so.2. NCCL suspend/resume will be disabled.")
        return None

    # ncclResult_t ncclCommSuspend(ncclComm_t comm, int flags)
    if not hasattr(lib, "ncclCommSuspend"):
        logger.warning("libnccl.so.2 does not have ncclCommSuspend. Requires NCCL >= 2.29.7.")
        return None

    lib.ncclCommSuspend.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.ncclCommSuspend.restype = ctypes.c_int

    # ncclResult_t ncclCommResume(ncclComm_t comm)
    lib.ncclCommResume.argtypes = [ctypes.c_void_p]
    lib.ncclCommResume.restype = ctypes.c_int

    _nccl_lib = lib
    return lib


def suspend_nccl_comm(comm_handle) -> bool:
    """Suspend a single NCCL communicator to release its GPU memory.

    Args:
        comm_handle: The ncclComm_t handle (ctypes pointer or integer).

    Returns:
        True if successful, False otherwise.
    """
    lib = _get_nccl_lib()
    if lib is None:
        return False

    if comm_handle is None:
        return False

    # Ensure we have a c_void_p
    if isinstance(comm_handle, int):
        comm_handle = ctypes.c_void_p(comm_handle)

    result = lib.ncclCommSuspend(comm_handle, NCCL_SUSPEND_MEM)
    if result != NCCL_SUCCESS:
        logger.warning(f"ncclCommSuspend failed with error code {result}")
        return False
    return True


def resume_nccl_comm(comm_handle) -> bool:
    """Resume a previously suspended NCCL communicator.

    Args:
        comm_handle: The ncclComm_t handle (ctypes pointer or integer).

    Returns:
        True if successful, False otherwise.
    """
    lib = _get_nccl_lib()
    if lib is None:
        return False

    if comm_handle is None:
        return False

    if isinstance(comm_handle, int):
        comm_handle = ctypes.c_void_p(comm_handle)

    result = lib.ncclCommResume(comm_handle)
    if result != NCCL_SUCCESS:
        logger.warning(f"ncclCommResume failed with error code {result}")
        return False
    return True


def _extract_nccl_comm_from_group(group, group_name: str = "unknown"):
    """Extract ncclComm_t from a vLLM GroupCoordinator.

    Path: group.device_communicator.pynccl_comm.comm

    Returns:
        The ncclComm_t handle, or None if unavailable.
    """
    if group is None or group.world_size <= 1:
        return None

    device_comm = getattr(group, "device_communicator", None)
    if device_comm is None:
        return None

    pynccl_comm = getattr(device_comm, "pynccl_comm", None)
    if pynccl_comm is None:
        return None

    comm = getattr(pynccl_comm, "comm", None)
    if comm is None:
        return None

    logger.debug(f"Found NCCL comm for vLLM {group_name} group (world_size={group.world_size})")
    return comm


def _get_all_vllm_comm_handles() -> list[tuple[str, object]]:
    """Get all ncclComm_t handles from vLLM's parallel groups.

    Returns:
        List of (group_name, ncclComm_t) tuples.
    """
    try:
        from vllm.distributed import parallel_state as ps
    except ImportError:
        logger.debug("vLLM not available.")
        return []

    handles = []

    # Collect all known group accessors
    group_accessors = [
        ("tp", "get_tp_group"),
        ("pp", "get_pp_group"),
    ]

    for name, accessor_name in group_accessors:
        accessor = getattr(ps, accessor_name, None)
        if accessor is None:
            continue
        try:
            group = accessor()
            comm = _extract_nccl_comm_from_group(group, name)
            if comm is not None:
                handles.append((name, comm))
        except Exception as e:
            logger.debug(f"Failed to get vLLM {name} comm: {e}")

    return handles


def _gloo_barrier():
    """Execute a barrier using gloo backend (CPU-based, doesn't need NCCL).

    Used to synchronize all TP ranks before/after NCCL suspend/resume
    to avoid P2P dangling references.
    """
    try:
        import torch.distributed as dist

        if not dist.is_initialized():
            return

        # Use the default process group's gloo backend if available,
        # otherwise try to find a gloo group.
        # In vLLM, the CPU group is typically gloo-backed.
        from vllm.distributed.parallel_state import get_tp_group

        tp_group = get_tp_group()
        cpu_group = getattr(tp_group, "cpu_group", None)
        if cpu_group is not None:
            dist.barrier(group=cpu_group)
        else:
            # Fallback: device synchronize as a weaker form of coordination
            get_torch_device().synchronize()
    except Exception as e:
        logger.debug(f"Gloo barrier failed, falling back to device synchronize: {e}")
        get_torch_device().synchronize()


def suspend_vllm_comms() -> bool:
    """Suspend all vLLM NCCL communicators (TP, PP, etc.) to free GPU memory.

    Must be called from within the vLLM rollout server process.
    All ranks must call this simultaneously.

    Returns:
        True if any comm was suspended.
    """
    handles = _get_all_vllm_comm_handles()
    if not handles:
        logger.debug("No vLLM NCCL comms to suspend.")
        return False

    _gloo_barrier()

    suspended = []
    for name, comm in handles:
        if suspend_nccl_comm(comm):
            suspended.append(name)

    if suspended:
        get_torch_device().empty_cache()
        logger.info(f"Suspended vLLM NCCL comms: {suspended}")

    _gloo_barrier()

    return len(suspended) > 0


def resume_vllm_comms() -> bool:
    """Resume all vLLM NCCL communicators.

    Must be called from within the vLLM rollout server process.
    All ranks must call this simultaneously.

    Returns:
        True if any comm was resumed.
    """
    handles = _get_all_vllm_comm_handles()
    if not handles:
        logger.debug("No vLLM NCCL comms to resume.")
        return False

    _gloo_barrier()

    resumed = []
    for name, comm in handles:
        if resume_nccl_comm(comm):
            resumed.append(name)

    if resumed:
        logger.info(f"Resumed vLLM NCCL comms: {resumed}")

    _gloo_barrier()

    return len(resumed) > 0
