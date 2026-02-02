#!/usr/bin/env python3
"""
GPU resource monitoring utilities.
"""

import subprocess
from typing import Dict

from ttop.utils import mib_to_gb


def get_total_gpu_memory() -> int:
    """Get total GPU memory in MiB."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.total',
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=1
        )

        lines = result.stdout.strip().split('\n')
        if lines:
            return int(lines[0].strip())
        return 0
    except Exception:
        return 0


def get_used_gpu_memory() -> int:
    """Get total used GPU memory in MiB."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used',
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=1
        )

        lines = result.stdout.strip().split('\n')
        if lines:
            return int(lines[0].strip())
        return 0
    except Exception:
        return 0


def get_gpu_usage() -> Dict[int, int]:
    """Get GPU usage for all processes using nvidia-smi.

    Returns:
        Dictionary mapping PID to GPU memory usage in MiB
    """
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-compute-apps=pid,used_memory',
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=1
        )

        gpu_usage = {}
        for line in result.stdout.strip().split('\n'):
            if line:
                parts = line.split(',')
                if len(parts) == 2:
                    pid = int(parts[0].strip())
                    mem = int(parts[1].strip())
                    gpu_usage[pid] = mem

        return gpu_usage
    except Exception:
        return {}


def get_gpu_memory_info() -> Dict:
    """Get GPU memory information.

    Returns:
        Dictionary with total and used memory in MiB and GB
    """
    total_mib = get_total_gpu_memory()
    used_mib = get_used_gpu_memory()

    return {
        "total_mib": total_mib,
        "used_mib": used_mib,
        "total_gb": mib_to_gb(total_mib),
        "used_gb": mib_to_gb(used_mib)
    }


def get_gpu_process_usage() -> Dict:
    """Get GPU usage by process.

    Returns:
        Dictionary with process usage information
    """
    return {"process_usage": get_gpu_usage()}


def get_gpu_usage_for_pids(pids: list) -> int:
    """Get total GPU memory usage for given PIDs.

    Args:
        pids: List of process IDs

    Returns:
        Total GPU memory usage in MiB
    """
    gpu_usage = get_gpu_usage()
    return sum(gpu_usage.get(pid, 0) for pid in pids)


def get_gpu_usage_for_pid(pid: int) -> int:
    """Get GPU memory usage for a single PID.

    Args:
        pid: Process ID

    Returns:
        GPU memory usage in MiB for this PID (0 if not using GPU)
    """
    gpu_usage = get_gpu_usage()
    return gpu_usage.get(pid, 0)


def get_gpu_stats_for_pids(pids: list, max_gpu_mem_gb: float = None) -> Dict[str, any]:
    """Get GPU memory usage statistics for given PIDs.

    Args:
        pids: List of process IDs
        max_gpu_mem_gb: Maximum GPU memory in GB allocated (defaults to total GPU memory)

    Returns:
        Dictionary with GPU usage in GB, percentages, and limits
    """
    gpu_usage_map = get_gpu_usage()
    total_gpu_mib = sum(mem for pid, mem in gpu_usage_map.items() if pid in pids)

    gpu_used_gb = mib_to_gb(total_gpu_mib)

    gpu_total_mib = get_total_gpu_memory()
    gpu_total_gb = mib_to_gb(gpu_total_mib)

    effective_max_gpu_gb = max_gpu_mem_gb if max_gpu_mem_gb else gpu_total_gb

    gpu_usage_percent = gpu_used_gb / effective_max_gpu_gb * 100 if effective_max_gpu_gb > 0 else 0

    return {
        "gpu_used_gb": gpu_used_gb,
        "gpu_total_gb": gpu_total_gb,
        "gpu_usage_percent": gpu_usage_percent,
        "max_gpu_gb": effective_max_gpu_gb
    }
