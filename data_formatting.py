#!/usr/bin/env python3
"""
Data formatting utilities for training monitor.
Handles building data matrices and calculating column widths.
"""

import time
from typing import Dict, List, Tuple

from rci.ttop.time_utils import format_duration, format_short_duration


def build_data_matrix(training_data: List[Tuple]) -> Tuple[List[str], List[List[str]]]:
    """Build a matrix of all column names and data rows.

    Returns:
        Tuple of (column_names, data_rows) where each row is a list of strings
    """
    column_names = [
        "Run Name", "Status", "Phase", "Epoch", "Batch",
        "Run Elapsed", "Run Remaining", "Run Total",
        "Epoch Elapsed", "Epoch Remaining", "Epoch Total",
        "Train Elapsed", "Train Remaining", "Train Total",
        "Val Time", "Val Elapsed", "Val Remaining", "Val Total",
        "CPU", "RAM", "GPU", "Procs", "Threads",
        "PID", "Wandb URL"
    ]

    rows = []

    for json_file, data, resources in training_data:
        metadata = data.get("metadata", {})
        state = data.get("state", {})
        estimates = data.get("estimates", {})
        timing = data.get("timing", {})

        # Check if PID is invalid due to missing metadata
        pid = str(metadata.get("pid", "Invalid"))
        status = str(data.get("status", "Invalid"))
        wandb_run_name = metadata.get("wandb_run_name") or json_file.stem
        wandb_url = metadata.get("wandb_url", "Invalid")
        phase = str(state.get("phase", "Invalid"))
        epoch = state.get("epoch", 0)
        total_epochs = state.get("total_epochs", 0)
        batch = state.get("batch", 0)
        total_batches = state.get("total_batches", 0)

        start_time = metadata.get("start_time")
        end_time = metadata.get("end_time")
        if end_time:
            total_elapsed = end_time - start_time
        elif start_time:
            total_elapsed = time.time() - start_time
        else:
            total_elapsed = None

        val_mean_time = estimates.get("validation_mean_time")

        # Extract nested timing estimates
        run_estimates = estimates.get("run", {})
        epoch_estimates = estimates.get("epoch", {})
        training_estimates = estimates.get("training", {})
        validation_estimates = estimates.get("validation", {})

        # Run timing
        run_elapsed = run_estimates.get("elapsed")
        run_remaining = run_estimates.get("remaining")
        run_total = run_estimates.get("total")

        # Epoch timing
        epoch_elapsed = epoch_estimates.get("elapsed")
        epoch_remaining = epoch_estimates.get("remaining")
        epoch_total = epoch_estimates.get("total")

        # Training timing
        training_elapsed = training_estimates.get("elapsed")
        training_remaining = training_estimates.get("remaining")
        training_total = training_estimates.get("total")

        # Validation timing
        validation_elapsed = validation_estimates.get("elapsed")
        validation_remaining = validation_estimates.get("remaining")
        validation_total = validation_estimates.get("total")

        epoch_str = f"{epoch}/{total_epochs}" if total_epochs > 0 else str(epoch)
        batch_str = f"{batch}/{total_batches}" if total_batches > 0 else str(batch)

        if resources:
            procs_str = str(resources['processes'])
            threads_str = str(resources['threads'])
            cpu_str = f"{resources['cpu_percent']:6.1f}%"
            ram_str = f"{resources['ram_mb']:8.1f}MB"
            gpu_str = f"{resources['gpu_mem_mib']:6}MiB"
        else:
            procs_str = "Invalid"
            threads_str = "Invalid"
            cpu_str = "Invalid"
            ram_str = "Invalid"
            gpu_str = "Invalid"

        row = [
            wandb_run_name[:40],
            status,
            phase,
            epoch_str,
            batch_str,
            format_duration(run_elapsed),
            format_duration(run_remaining),
            format_duration(run_total),
            format_duration(epoch_elapsed),
            format_duration(epoch_remaining),
            format_duration(epoch_total),
            format_duration(training_elapsed),
            format_duration(training_remaining),
            format_duration(training_total),
            format_short_duration(val_mean_time),
            format_duration(validation_elapsed),
            format_duration(validation_remaining),
            format_duration(validation_total),
            cpu_str,
            ram_str,
            gpu_str,
            procs_str,
            threads_str,
            pid,
            wandb_url
        ]
        rows.append(row)

    return column_names, rows


def calculate_column_widths_from_matrix(column_names: List[str], rows: List[List[str]]) -> Dict[str, int]:
    """Calculate column widths from matrix data.

    Args:
        column_names: List of column names
        rows: List of data rows

    Returns:
        Dictionary mapping column names to widths
    """
    widths = {}

    for col_name in column_names:
        widths[col_name] = len(col_name)

    for row in rows:
        for col_name, value in zip(column_names, row):
            widths[col_name] = max(widths[col_name], len(value))

    return widths
