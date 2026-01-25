#!/usr/bin/env python3
"""
CPU resource monitoring utilities.
"""

import psutil
from typing import List, Dict, Optional

from rci.ttop.utils import MemoryInfo


def get_process_tree(pid: int) -> List[int]:
    """Get all PIDs in the process tree (parent + all descendants).

    Args:
        pid: Process ID of the parent process

    Returns:
        List of PIDs including parent and all descendants
    """
    try:
        parent = psutil.Process(pid)
        pids = [pid]
        children = parent.children(recursive=True)
        pids.extend([child.pid for child in children])
        return pids
    except psutil.NoSuchProcess:
        return []


def get_thread_name(pid: int, tid: int) -> str:
    """Get thread name from /proc/[pid]/task/[tid]/comm on Linux.

    Args:
        pid: Process ID
        tid: Thread ID

    Returns:
        Thread name or fallback to "Thread-{tid}" if not readable
    """
    try:
        with open(f"/proc/{pid}/task/{tid}/comm", 'r') as f:
            return f.read().strip()
    except (FileNotFoundError, PermissionError, OSError):
        return f"Thread-{tid}"


def get_thread_info(process: psutil.Process) -> List[Dict]:
    """Get information about all threads in a process.

    Args:
        process: psutil.Process object

    Returns:
        List of dictionaries with thread IDs and names from /proc
    """
    threads = []
    try:
        pid = process.pid
        for thread in process.threads():
            thread_name = get_thread_name(pid, thread.id)
            threads.append({
                "thread_id": thread.id,
                "thread_name": thread_name
            })
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    return threads


def get_process_info(pid: int) -> Optional[Dict]:
    """Get detailed information about a single process.

    Args:
        pid: Process ID

    Returns:
        Dictionary with process information or None if process doesn't exist
    """
    try:
        from rci.ttop.gpu_resources import get_gpu_usage_for_pid
        from rci.ttop.utils import mib_to_gb

        proc = psutil.Process(pid)
        cpu_percent = proc.cpu_percent(interval=None)
        memory_info = proc.memory_info()
        ram_bytes = memory_info.rss
        num_threads = proc.num_threads()
        threads = get_thread_info(proc)

        gpu_mem_mib = get_gpu_usage_for_pid(pid)
        gpu_mem_gb = mib_to_gb(gpu_mem_mib)

        return {
            "pid": pid,
            "name": proc.name(),
            "cpu_percent": cpu_percent,
            "ram_bytes": ram_bytes,
            "ram_mb": MemoryInfo(ram_bytes).mb,
            "ram_gb": MemoryInfo(ram_bytes).gb,
            "num_threads": num_threads,
            "threads": threads,
            "gpu_mem_mib": gpu_mem_mib,
            "gpu_mem_gb": gpu_mem_gb
        }
    except psutil.NoSuchProcess:
        return None


def get_process_tree_resources(pid: int) -> Optional[Dict]:
    """Get detailed resource information for a process tree.

    Args:
        pid: Process ID of the parent process

    Returns:
        Dictionary with process tree resources or None if process doesn't exist
    """
    if not psutil.pid_exists(pid):
        return None

    pids = get_process_tree(pid)
    if not pids:
        return None

    processes = []
    total_cpu = 0.0
    total_ram_bytes = 0
    total_threads = 0

    for process_pid in pids:
        proc_info = get_process_info(process_pid)
        if proc_info:
            processes.append(proc_info)
            total_cpu += proc_info["cpu_percent"]
            total_ram_bytes += proc_info["ram_bytes"]
            total_threads += proc_info["num_threads"]

    return {
        "processes": processes,
        "num_processes": len(processes),
        "total_cpu_percent": total_cpu,
        "total_ram_bytes": total_ram_bytes,
        "total_ram_mb": MemoryInfo(total_ram_bytes).mb,
        "total_ram_gb": MemoryInfo(total_ram_bytes).gb,
        "total_threads": total_threads
    }


def get_system_resources(pid: int) -> Optional[Dict[str, any]]:
    """Get system resources for a PID and its subprocesses.

    Args:
        pid: Process ID

    Returns:
        Dictionary with resource information or None if process doesn't exist
    """
    tree_resources = get_process_tree_resources(pid)
    if not tree_resources:
        return None

    return {
        "processes": tree_resources["num_processes"],
        "threads": tree_resources["total_threads"],
        "cpu_percent": tree_resources["total_cpu_percent"],
        "ram_mb": tree_resources["total_ram_mb"],
    }


def get_user_processes_resources(username: Optional[str] = None) -> Dict[str, any]:
    """Get resource usage for all processes owned by a user.

    Args:
        username: Username to filter by (defaults to current user)

    Returns:
        Dictionary with total CPU, RAM usage, and list of PIDs
    """
    if username is None:
        username = psutil.Process().username()

    total_cpu = 0.0
    total_ram_bytes = 0
    user_pids = set()

    for proc in psutil.process_iter(['pid', 'username']):
        try:
            if proc.info['username'] == username:
                cpu_pct = proc.cpu_percent(interval=None)
                total_cpu += cpu_pct
                total_ram_bytes += proc.memory_info().rss
                user_pids.add(proc.pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    return {
        "total_cpu": total_cpu,
        "total_ram_bytes": total_ram_bytes,
        "user_pids": user_pids
    }


def get_user_cpu_ram_stats(username: Optional[str] = None,
                            max_cpus: Optional[int] = None,
                            max_ram_gb: Optional[float] = None) -> Dict[str, any]:
    """Get CPU and RAM usage statistics for user processes.

    Args:
        username: Username to filter by (defaults to current user)
        max_cpus: Maximum CPU cores allocated (defaults to all available)
        max_ram_gb: Maximum RAM in GB allocated (defaults to total system RAM)

    Returns:
        Dictionary with CPU and RAM usage, percentages, and user PIDs
    """
    cpu_count = psutil.cpu_count()
    ram_total_gb = MemoryInfo(psutil.virtual_memory().total).gb

    user_resources = get_user_processes_resources(username)
    total_cpu = user_resources["total_cpu"]
    total_ram_bytes = user_resources["total_ram_bytes"]
    user_pids = user_resources["user_pids"]

    ram_used_gb = MemoryInfo(total_ram_bytes).gb

    effective_max_cpus = max_cpus if max_cpus else cpu_count
    effective_max_ram_gb = max_ram_gb if max_ram_gb else ram_total_gb

    cpu_usage_percent = total_cpu / effective_max_cpus
    ram_usage_percent = ram_used_gb / effective_max_ram_gb * 100 if effective_max_ram_gb > 0 else 0

    return {
        "cpu_count": cpu_count,
        "cpu_used": total_cpu,
        "cpu_usage_percent": cpu_usage_percent,
        "max_cpus": effective_max_cpus,
        "ram_used_gb": ram_used_gb,
        "ram_total_gb": ram_total_gb,
        "ram_usage_percent": ram_usage_percent,
        "max_ram_gb": effective_max_ram_gb,
        "user_pids": user_pids
    }
