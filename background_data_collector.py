#!/usr/bin/env python3
"""
Background data collection thread for training monitor.
Collects training data and system resources without blocking the UI.
"""

import time
import json
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from rci.ttop.gpu_resources import get_gpu_usage_for_pids, get_gpu_stats_for_pids
from rci.ttop.cpu_resources import get_process_tree, get_user_cpu_ram_stats, get_process_tree_resources
from rci.ttop.csv_writer import MatrixCSVWriter
from rci.ttop.data_formatting import build_data_matrix, calculate_column_widths_from_matrix


class BackgroundDataCollector(threading.Thread):
    """Background thread that collects ALL data without blocking UI."""

    def __init__(self, folder_path: Path, interval: float = 2.0,
                 max_cpus: Optional[int] = None,
                 max_ram_gb: Optional[float] = None,
                 max_gpu_mem_gb: Optional[float] = None,
                 csv_output_interval: Optional[float] = None,
                 csv_output_dir: Optional[Path] = None):
        super().__init__(daemon=True)
        self.folder_path = folder_path
        self.interval = interval
        self.max_cpus = max_cpus
        self.max_ram_gb = max_ram_gb
        self.max_gpu_mem_gb = max_gpu_mem_gb
        self.running = True
        self.lock = threading.Lock()
        self.latest_data = None
        self.cpu_percent_cache = {}
        self.training_data_cache = {}
        self.csv_output_interval = csv_output_interval
        self.csv_writer = None
        self.last_csv_write = 0

        if csv_output_interval and csv_output_dir:
            self.csv_writer = MatrixCSVWriter(csv_output_dir)

    def run(self):
        import psutil
        from rci.ttop.utils import natural_sort_key

        for proc in psutil.process_iter(['pid']):
            try:
                proc.cpu_percent(interval=None)
            except:
                pass

        while self.running:
            start_time = time.time()

            # Look for training folders (each folder contains metadata.json, progress.json, control.json)
            training_folders = sorted([d for d in self.folder_path.iterdir() if d.is_dir()], key=natural_sort_key)
            training_folder_names = {f.name for f in training_folders}

            # Start with cached data, then update with fresh reads
            training_data = dict(self.training_data_cache)

            for training_folder in training_folders:
                progress_file = training_folder / "progress.json"
                control_file = training_folder / "control.json"

                # Use cached data if file doesn't exist (temporary filesystem issue)
                if not progress_file.exists():
                    if training_folder.name in training_data:
                        continue
                    else:
                        continue

                # Read progress (includes metadata and dynamic data)
                progress_data = self.read_json_file(progress_file)
                if not progress_data:
                    if training_folder.name in training_data:
                        continue
                    else:
                        continue

                # Extract metadata from progress.json
                metadata = progress_data.get("metadata", {})

                # Read control (from ttop)
                control_data = self.read_json_file(control_file) if control_file.exists() else {}

                # Merge data for compatibility
                data = {
                    "metadata": metadata,
                    "state": progress_data.get("state", {}),
                    "timing": progress_data.get("timing", {}),
                    "estimates": progress_data.get("estimates", {}),
                    "metrics": progress_data.get("metrics", {}),
                    "status": progress_data.get("status", "UNKNOWN"),
                    "end_time": progress_data.get("end_time"),
                    "error_message": progress_data.get("error_message"),
                    "control": control_data
                }

                pid = metadata.get("pid")
                current_status = progress_data.get("status", "UNKNOWN")

                # Handle invalid or missing PID
                if not pid:
                    # Show as invalid but include all available info
                    resources = {
                        "processes": 0,
                        "threads": 0,
                        "cpu_percent": 0.0,
                        "ram_mb": 0.0,
                        "gpu_mem_mib": 0.0,
                        "detailed_processes": []
                    }
                else:
                    tree_resources = get_process_tree_resources(pid)

                    # Handle case where process no longer exists
                    if tree_resources is None:
                        resources = None
                        # If status is RUNNING but process doesn't exist, update status to KILLED
                        if current_status == "RUNNING":
                            progress_data["status"] = "KILLED"
                            progress_data["end_time"] = time.time()
                            self.write_json_file(progress_file, progress_data)
                    else:
                        pids = get_process_tree(pid)
                        gpu_mem = get_gpu_usage_for_pids(pids)

                        resources = {
                            "processes": tree_resources["num_processes"],
                            "threads": tree_resources["total_threads"],
                            "cpu_percent": tree_resources["total_cpu_percent"],
                            "ram_mb": tree_resources["total_ram_mb"],
                            "gpu_mem_mib": gpu_mem,
                            "detailed_processes": tree_resources["processes"]
                        }

                training_data[training_folder.name] = (training_folder, data, resources)

            # Remove trainings that no longer exist in the folder
            training_data = {name: data for name, data in training_data.items()
                           if name in training_folder_names}

            # Update cache with latest data
            self.training_data_cache = training_data

            overall_resources = self.get_overall_system_resources()

            training_data_list = list(training_data.values())
            column_names, rows = build_data_matrix(training_data_list)
            column_widths = calculate_column_widths_from_matrix(column_names, rows)

            with self.lock:
                self.latest_data = {
                    'json_files': training_folders,  # Now folders instead of JSON files
                    'overall_resources': overall_resources,
                    'training_data': training_data_list,
                    'column_names': column_names,
                    'data_rows': rows,
                    'column_widths': column_widths,
                    'timestamp': time.time()
                }

            if self.csv_output_interval and self.csv_writer:
                current_time = time.time()
                if current_time - self.last_csv_write >= self.csv_output_interval:
                    self.write_csv_snapshot(column_names, rows, overall_resources, current_time)
                    self.last_csv_write = current_time

            elapsed = time.time() - start_time
            sleep_time = max(0, self.interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def get_overall_system_resources(self) -> Dict:
        """Get overall system resource usage for current user.

        Note: This may block briefly (~1s) due to nvidia-smi subprocess calls.
        """
        cpu_ram_stats = get_user_cpu_ram_stats(
            max_cpus=self.max_cpus,
            max_ram_gb=self.max_ram_gb
        )

        gpu_stats = get_gpu_stats_for_pids(
            pids=cpu_ram_stats["user_pids"],
            max_gpu_mem_gb=self.max_gpu_mem_gb
        )

        return {
            "cpu_count": cpu_ram_stats["cpu_count"],
            "cpu_used": cpu_ram_stats["cpu_used"],
            "cpu_usage_percent": cpu_ram_stats["cpu_usage_percent"],
            "max_cpus": cpu_ram_stats["max_cpus"],
            "ram_used_gb": cpu_ram_stats["ram_used_gb"],
            "ram_total_gb": cpu_ram_stats["ram_total_gb"],
            "ram_usage_percent": cpu_ram_stats["ram_usage_percent"],
            "max_ram_gb": cpu_ram_stats["max_ram_gb"],
            "gpu_used_gb": gpu_stats["gpu_used_gb"],
            "gpu_total_gb": gpu_stats["gpu_total_gb"],
            "gpu_usage_percent": gpu_stats["gpu_usage_percent"],
            "max_gpu_gb": gpu_stats["max_gpu_gb"]
        }

    def read_json_file(self, file_path: Path) -> Optional[Dict]:
        """Read and parse a JSON file."""
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return None

    def write_json_file(self, file_path: Path, data: Dict):
        """Write updated data back to JSON file."""
        try:
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def write_csv_snapshot(self, column_names: List[str], rows: List[List[str]],
                           overall_resources: Dict, timestamp: float):
        """Write current data snapshot to CSV file."""
        full_column_names = ['Timestamp'] + column_names + [
            'Overall_CPU_Used', 'Overall_CPU_Max', 'Overall_CPU_Percent',
            'Overall_RAM_Used_GB', 'Overall_RAM_Max_GB', 'Overall_RAM_Percent',
            'Overall_GPU_Used_GB', 'Overall_GPU_Max_GB', 'Overall_GPU_Percent'
        ]

        timestamp_str = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')

        resource_suffix = [
            f"{overall_resources['cpu_used']:.1f}",
            f"{overall_resources['max_cpus']}",
            f"{overall_resources['cpu_usage_percent']:.1f}",
            f"{overall_resources['ram_used_gb']:.2f}",
            f"{overall_resources['max_ram_gb']:.2f}",
            f"{overall_resources['ram_usage_percent']:.1f}",
            f"{overall_resources['gpu_used_gb']:.2f}",
            f"{overall_resources['max_gpu_gb']:.2f}",
            f"{overall_resources['gpu_usage_percent']:.1f}"
        ]

        if len(rows) > 0:
            for row in rows:
                csv_row = [timestamp_str] + row + resource_suffix
                self.csv_writer.write_row(full_column_names, csv_row)
        else:
            csv_row = [timestamp_str] + [''] * len(column_names) + resource_suffix
            self.csv_writer.write_row(full_column_names, csv_row)

    def get_latest_data(self):
        """Get the latest collected data."""
        with self.lock:
            return self.latest_data

    def stop(self):
        """Stop the background thread."""
        self.running = False

        if self.csv_writer:
            self.csv_writer.finalize()
