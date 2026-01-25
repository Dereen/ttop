"""
Variable editor for training monitor tracked variables.
Shows training statistics at the top and allows editing controllable variables.
"""

import json
import curses
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

from rci.ttop.color_scheme import ColorScheme
from rci.ttop.time_utils import format_duration
from rci.ttop.output_viewer import get_output_files_from_metadata, tail_both_files, print_ansi_colored_line
from rci.ttop.cpu_resources import get_process_tree_resources, get_user_cpu_ram_stats
from rci.ttop.gpu_resources import get_gpu_stats_for_pids
from collections import deque


class ViewMode(Enum):
    """View mode for the editor."""
    VARIABLES = "variables"
    THREADS = "threads"


class ThreadRowType(Enum):
    """Type of row in the threads display."""
    PROCESS = "process"
    THREAD = "thread"


@dataclass
class ThreadRow:
    """Represents a single row in the threads view."""
    row_type: ThreadRowType
    process_idx: int
    thread_idx: Optional[int] = None
    text: str = ""
    color_pair: int = ColorScheme.NORMAL


def convert_value(value_str: str, original_type: type) -> Optional[Any]:
    """Convert string value to the original type.

    Args:
        value_str: String representation of value
        original_type: Original type to convert to

    Returns:
        Converted value or None if conversion failed
    """
    try:
        if original_type == float:
            return float(value_str)
        elif original_type == int:
            return int(value_str)
        elif original_type == bool:
            if value_str.lower() in ('true', '1', 'yes'):
                return True
            elif value_str.lower() in ('false', '0', 'no'):
                return False
            else:
                return None
        else:
            return value_str
    except ValueError:
        return None


def pad_line(text: str, width: int) -> str:
    """Pad line with spaces to fill terminal width."""
    if len(text) >= width:
        return text[:width]
    return text + ' ' * (width - len(text))


def print_colored_line(stdscr, row: int, col: int, text: str, color_pair: int = ColorScheme.NORMAL):
    """Print a line with color, handling errors gracefully."""
    try:
        stdscr.addstr(row, col, text, curses.color_pair(color_pair))
    except curses.error:
        pass


def get_process_resources(pid: int) -> Optional[Dict]:
    """Get resource usage for a specific training process.

    Args:
        pid: Process ID

    Returns:
        Dictionary with resource information or None if process not found
    """
    if not pid:
        return None

    from rci.ttop.cpu_resources import get_process_tree
    from rci.ttop.gpu_resources import get_gpu_usage_for_pids

    tree_resources = get_process_tree_resources(pid)
    if tree_resources is None:
        return None

    pids = get_process_tree(pid)
    gpu_mem_mib = get_gpu_usage_for_pids(pids)

    return {
        "processes": tree_resources["num_processes"],
        "threads": tree_resources["total_threads"],
        "cpu_percent": tree_resources["total_cpu_percent"],
        "ram_mb": tree_resources["total_ram_mb"],
        "gpu_mem_mib": gpu_mem_mib,
        "detailed_processes": tree_resources["processes"]
    }


def get_overall_resources() -> Dict:
    """Get overall system resource usage.

    Returns:
        Dictionary with overall resource statistics
    """
    cpu_ram_stats = get_user_cpu_ram_stats()
    gpu_stats = get_gpu_stats_for_pids(pids=cpu_ram_stats["user_pids"])

    return {
        "cpu_used": cpu_ram_stats["cpu_used"],
        "max_cpus": cpu_ram_stats["max_cpus"],
        "cpu_usage_percent": cpu_ram_stats["cpu_usage_percent"],
        "ram_used_gb": cpu_ram_stats["ram_used_gb"],
        "max_ram_gb": cpu_ram_stats["max_ram_gb"],
        "ram_usage_percent": cpu_ram_stats["ram_usage_percent"],
        "gpu_used_gb": gpu_stats["gpu_used_gb"],
        "max_gpu_gb": gpu_stats["max_gpu_gb"],
        "gpu_usage_percent": gpu_stats["gpu_usage_percent"]
    }


def format_training_stats(file_data: dict) -> Dict[str, any]:
    """Format training statistics for display.

    Args:
        file_data: Training data dictionary from JSON

    Returns:
        Dictionary with grouped statistics
    """
    metadata = file_data.get("metadata", {})
    state = file_data.get("state", {})
    estimates = file_data.get("estimates", {})
    metrics = file_data.get("metrics", {})

    # Extract nested timing estimates
    run_estimates = estimates.get("run", {})
    epoch_estimates = estimates.get("epoch", {})
    training_estimates = estimates.get("training", {})
    validation_estimates = estimates.get("validation", {})

    # Metrics
    loss = metrics.get("loss")
    loss_str = f"{loss:.4f}" if loss else "N/A"

    lr = metrics.get("learning_rate")
    lr_str = f"{lr:.6f}" if lr else "N/A"

    return {
        "status_info": {
            "Status": file_data.get("status", "UNKNOWN"),
            "Phase": state.get("phase", "N/A"),
            "Epoch": f"{state.get('epoch', 0)}/{state.get('total_epochs', 0)}",
            "Batch": f"{state.get('batch', 0)}/{state.get('total_batches', 0)}",
        },
        "run_timing": {
            "Elapsed": format_duration(run_estimates.get("elapsed"), "N/A"),
            "Remaining": format_duration(run_estimates.get("remaining"), "N/A"),
            "Total": format_duration(run_estimates.get("total"), "N/A"),
        },
        "epoch_timing": {
            "Elapsed": format_duration(epoch_estimates.get("elapsed"), "N/A"),
            "Remaining": format_duration(epoch_estimates.get("remaining"), "N/A"),
            "Total": format_duration(epoch_estimates.get("total"), "N/A"),
        },
        "training_timing": {
            "Elapsed": format_duration(training_estimates.get("elapsed"), "N/A"),
            "Remaining": format_duration(training_estimates.get("remaining"), "N/A"),
            "Total": format_duration(training_estimates.get("total"), "N/A"),
        },
        "validation_timing": {
            "Elapsed": format_duration(validation_estimates.get("elapsed"), "N/A"),
            "Remaining": format_duration(validation_estimates.get("remaining"), "N/A"),
            "Total": format_duration(validation_estimates.get("total"), "N/A"),
        },
        "metrics": {
            "Loss": loss_str,
            "Learning Rate": lr_str,
        }
    }


def build_thread_display(resources: Dict, cursor_position: int, expanded_processes: set) -> List[ThreadRow]:
    """Build thread display rows.

    Args:
        resources: Resource dictionary with detailed_processes
        cursor_position: Current cursor position
        expanded_processes: Set of expanded process indices

    Returns:
        List of ThreadRow objects
    """
    rows = []
    if not resources or 'detailed_processes' not in resources:
        return rows

    processes = resources['detailed_processes']
    for proc_idx, proc_info in enumerate(processes):
        proc_name = proc_info.get('name', 'unknown')
        proc_pid = proc_info.get('pid', 0)
        proc_cpu = proc_info.get('cpu_percent', 0.0)
        proc_ram_mb = proc_info.get('ram_mb', 0.0)
        proc_gpu_mib = proc_info.get('gpu_mem_mib', 0)
        proc_threads_count = proc_info.get('num_threads', 0)

        is_cursor = (len(rows) == cursor_position)
        is_expanded = proc_idx in expanded_processes

        expand_indicator = "[-]" if is_expanded else "[+]"

        gpu_text = f" GPU: {proc_gpu_mib}MiB" if proc_gpu_mib > 0 else ""
        proc_text = f"{expand_indicator} {proc_name} (PID: {proc_pid}) - CPU: {proc_cpu:.1f}% RAM: {proc_ram_mb:.0f}MB{gpu_text} Threads: {proc_threads_count}"

        color_pair = ColorScheme.PROCESS_CURSOR if is_cursor else ColorScheme.PROCESS
        rows.append(ThreadRow(
            row_type=ThreadRowType.PROCESS,
            process_idx=proc_idx,
            text=proc_text,
            color_pair=color_pair
        ))

        if is_expanded:
            threads = proc_info.get('threads', [])
            for thread_idx, thread_info in enumerate(threads):
                thread_id = thread_info.get('thread_id', 0)
                thread_name = thread_info.get('thread_name', 'unknown')

                is_cursor = (len(rows) == cursor_position)
                thread_text = f"  [T] {thread_name} (TID: {thread_id})"

                color_pair = ColorScheme.THREAD_CURSOR if is_cursor else ColorScheme.THREAD
                rows.append(ThreadRow(
                    row_type=ThreadRowType.THREAD,
                    process_idx=proc_idx,
                    thread_idx=thread_idx,
                    text=thread_text,
                    color_pair=color_pair
                ))

    return rows


def render_resources_panel(stdscr, start_row: int, start_col: int, width: int, height: int,
                           resources: Optional[Dict], overall_resources: Optional[Dict]) -> int:
    """Render resources panel on the right side.

    Args:
        stdscr: Curses screen object
        start_row: Starting row for rendering
        start_col: Starting column for rendering
        width: Panel width
        height: Maximum panel height
        resources: Training-specific resources
        overall_resources: Overall system resources

    Returns:
        Number of rows rendered
    """
    row = start_row
    max_row = start_row + height

    # Panel header
    if row < max_row:
        header = "Resources"
        print_colored_line(stdscr, row, start_col, pad_line(header, width), ColorScheme.HEADER)
        row += 1

    if row < max_row:
        print_colored_line(stdscr, row, start_col, pad_line("-" * width, width), ColorScheme.SEPARATOR)
        row += 1

    # Training resources
    if resources:
        if row < max_row:
            print_colored_line(stdscr, row, start_col, pad_line("Training Resources:", width), ColorScheme.HEADER)
            row += 1

        if row < max_row:
            line = f"  CPU: {resources['cpu_percent']:.1f}%"
            print_colored_line(stdscr, row, start_col, pad_line(line, width), ColorScheme.VALUE)
            row += 1

        if row < max_row:
            line = f"  RAM: {resources['ram_mb']:.0f} MB"
            print_colored_line(stdscr, row, start_col, pad_line(line, width), ColorScheme.VALUE)
            row += 1

        if row < max_row:
            gpu_mem_mib = resources.get('gpu_mem_mib', 0)
            line = f"  GPU: {gpu_mem_mib} MiB"
            print_colored_line(stdscr, row, start_col, pad_line(line, width), ColorScheme.VALUE)
            row += 1

        if row < max_row:
            line = f"  Processes: {resources['processes']}"
            print_colored_line(stdscr, row, start_col, pad_line(line, width), ColorScheme.VALUE)
            row += 1

        if row < max_row:
            line = f"  Threads: {resources['threads']}"
            print_colored_line(stdscr, row, start_col, pad_line(line, width), ColorScheme.VALUE)
            row += 1

        if row < max_row:
            print_colored_line(stdscr, row, start_col, pad_line("", width), ColorScheme.NORMAL)
            row += 1
    else:
        if row < max_row:
            print_colored_line(stdscr, row, start_col, pad_line("Training Resources: N/A", width), ColorScheme.ERROR)
            row += 1

    # Overall resources
    if overall_resources:
        if row < max_row:
            print_colored_line(stdscr, row, start_col, pad_line("Overall Resources:", width), ColorScheme.HEADER)
            row += 1

        if row < max_row:
            cpu_line = f"  CPU: {overall_resources['cpu_used']:.1f}% / {overall_resources['max_cpus']} cores ({overall_resources['cpu_usage_percent']:.1f}%)"
            print_colored_line(stdscr, row, start_col, pad_line(cpu_line, width), ColorScheme.VALUE)
            row += 1

        if row < max_row:
            ram_line = f"  RAM: {overall_resources['ram_used_gb']:.1f}GB / {overall_resources['max_ram_gb']:.1f}GB ({overall_resources['ram_usage_percent']:.1f}%)"
            print_colored_line(stdscr, row, start_col, pad_line(ram_line, width), ColorScheme.VALUE)
            row += 1

        if row < max_row:
            gpu_line = f"  GPU: {overall_resources['gpu_used_gb']:.1f}GB / {overall_resources['max_gpu_gb']:.1f}GB ({overall_resources['gpu_usage_percent']:.1f}%)"
            print_colored_line(stdscr, row, start_col, pad_line(gpu_line, width), ColorScheme.VALUE)
            row += 1

    return row - start_row


def render_variables_view(stdscr, start_row: int, start_col: int, width: int, height: int,
                          tracked_vars: Dict, keys: List[str], current_index: int,
                          editing: bool, edit_buffer: str, error_message: str) -> int:
    """Render variables editor view.

    Args:
        stdscr: Curses screen object
        start_row: Starting row for rendering
        start_col: Starting column for rendering
        width: Panel width
        height: Maximum panel height
        tracked_vars: Dictionary of tracked variables
        keys: List of variable keys
        current_index: Currently selected variable index
        editing: Whether in editing mode
        edit_buffer: Current edit buffer
        error_message: Error message to display

    Returns:
        Number of rows rendered
    """
    row = start_row
    max_row = start_row + height

    # Panel header
    if row < max_row:
        print_colored_line(stdscr, row, start_col, pad_line("Controllable Variables", width), ColorScheme.HEADER)
        row += 1

    if row < max_row:
        if editing:
            instruction = "Type value, Enter=save, Esc=cancel"
        else:
            instruction = "Up/Down=select, Enter=edit"
        print_colored_line(stdscr, row, start_col, pad_line(instruction, width), ColorScheme.STATUS)
        row += 1

    if row < max_row:
        print_colored_line(stdscr, row, start_col, pad_line("-" * width, width), ColorScheme.SEPARATOR)
        row += 1

    # Display variables
    for idx, key in enumerate(keys):
        if row >= max_row:
            break

        value = tracked_vars[key]
        value_type = type(value).__name__

        if idx == current_index:
            if editing:
                display_value = edit_buffer + "_"
                prefix = f"> {key} ({value_type}): "
                try:
                    stdscr.addstr(row, start_col, prefix[:width], curses.color_pair(ColorScheme.PROCESS_CURSOR))
                    if start_col + len(prefix) < start_col + width:
                        stdscr.addstr(row, start_col + len(prefix), display_value[:width-len(prefix)], curses.color_pair(ColorScheme.HEADER))
                except curses.error:
                    pass
            else:
                line = f"> {key} ({value_type}): {value}"
                print_colored_line(stdscr, row, start_col, pad_line(line, width), ColorScheme.PROCESS_CURSOR)
        else:
            line = f"  {key} ({value_type}): {value}"
            print_colored_line(stdscr, row, start_col, pad_line(line, width), ColorScheme.NORMAL)

        row += 1

    # Show error message if any
    if error_message and row < max_row:
        row += 1
        if row < max_row:
            print_colored_line(stdscr, row, start_col, pad_line(error_message, width), ColorScheme.ERROR)
            row += 1

    return row - start_row


def render_threads_view(stdscr, start_row: int, start_col: int, width: int, height: int,
                        resources: Optional[Dict], threads_cursor: int,
                        expanded_processes: set, scroll_offset: int) -> tuple:
    """Render threads view with processes and threads.

    Args:
        stdscr: Curses screen object
        start_row: Starting row for rendering
        start_col: Starting column for rendering
        width: Panel width
        height: Maximum panel height
        resources: Resource dictionary with detailed_processes
        threads_cursor: Current cursor position
        expanded_processes: Set of expanded process indices
        scroll_offset: Vertical scroll offset

    Returns:
        Tuple of (rows_rendered, total_rows, current_row_info)
    """
    row = start_row
    max_row = start_row + height

    # Panel header
    if row < max_row:
        print_colored_line(stdscr, row, start_col, pad_line("Processes & Threads", width), ColorScheme.HEADER)
        row += 1

    if row < max_row:
        instruction = "Up/Down=navigate, Enter=expand/collapse"
        print_colored_line(stdscr, row, start_col, pad_line(instruction, width), ColorScheme.STATUS)
        row += 1

    if row < max_row:
        print_colored_line(stdscr, row, start_col, pad_line("-" * width, width), ColorScheme.SEPARATOR)
        row += 1

    content_start_row = row
    available_rows = max_row - row

    # Build thread display
    thread_rows = build_thread_display(resources, threads_cursor, expanded_processes)

    if not thread_rows:
        if row < max_row:
            print_colored_line(stdscr, row, start_col, pad_line("No processes found", width), ColorScheme.ERROR)
            row += 1
        return (row - start_row, 0, None)

    # Render visible rows with scrolling
    visible_rows = thread_rows[scroll_offset:scroll_offset + available_rows]
    current_row_info = None

    for thread_row in visible_rows:
        if row >= max_row:
            break

        # Truncate text to fit width
        display_text = thread_row.text[:width]
        print_colored_line(stdscr, row, start_col, pad_line(display_text, width), thread_row.color_pair)
        row += 1

        # Track current row info for returning
        if thread_row.color_pair in [ColorScheme.PROCESS_CURSOR, ColorScheme.THREAD_CURSOR]:
            current_row_info = thread_row

    return (row - start_row, len(thread_rows), current_row_info)


def show_variable_editor(stdscr, training_folder: Path, file_data: dict):
    """Show variable editor for tracked variables.

    Args:
        stdscr: Curses screen object
        training_folder: Path to the training folder
        file_data: Training data dictionary
    """
    # Define file paths
    control_file = training_folder / "control.json"
    progress_file = training_folder / "progress.json"

    # Get tracked variables from control.json
    tracked_vars = file_data.get("control", {})

    if not tracked_vars:
        # Show error message
        stdscr.clear()
        term_width = curses.COLS
        print_colored_line(stdscr, 0, 0, pad_line("=" * 200, term_width), ColorScheme.SEPARATOR)
        print_colored_line(stdscr, 1, 0, pad_line("Variable Editor", term_width), ColorScheme.HEADER)
        print_colored_line(stdscr, 2, 0, pad_line("=" * 200, term_width), ColorScheme.SEPARATOR)
        print_colored_line(stdscr, 3, 0, pad_line("No tracked variables found for this training.", term_width), ColorScheme.ERROR)
        print_colored_line(stdscr, 4, 0, pad_line("Press ESC to return", term_width), ColorScheme.NORMAL)
        stdscr.refresh()

        # Wait for ESC
        while True:
            key = stdscr.getch()
            if key == 27:  # ESC
                break
            time.sleep(0.01)
        return

    keys = list(tracked_vars.keys())
    current_index = 0
    editing = False
    edit_buffer = ""
    error_message = ""
    last_stats_update = 0
    stats_update_interval = 0.1

    # View mode state
    view_mode = ViewMode.VARIABLES
    threads_cursor = 0
    threads_scroll = 0
    expanded_processes = set()
    last_resources_update = 0
    resources_update_interval = 0.5

    # Output viewer state
    show_output = True  # Always show output by default
    output_scroll = 0
    output_auto_scroll = True
    output_lines = deque(maxlen=1000)
    last_output_update = 0
    output_update_interval = 0.5

    # Get output file paths
    metadata = file_data.get("metadata", {})
    stdout_file, stderr_file = get_output_files_from_metadata(metadata)

    # Get PID for resource tracking
    pid = metadata.get("pid")
    resources = None
    overall_resources = None

    while True:
        current_time = time.time()

        # Reload training statistics at regular intervals
        if current_time - last_stats_update >= stats_update_interval:
            try:
                # Read progress file (includes metadata)
                if progress_file.exists():
                    with open(progress_file, 'r') as f:
                        progress_data = json.load(f)

                    # Extract metadata from progress.json
                    metadata = progress_data.get("metadata", {})

                    # Merge data (same structure as BackgroundDataCollector)
                    file_data = {
                        "metadata": metadata,
                        "state": progress_data.get("state", {}),
                        "timing": progress_data.get("timing", {}),
                        "estimates": progress_data.get("estimates", {}),
                        "metrics": progress_data.get("metrics", {}),
                        "status": progress_data.get("status", "UNKNOWN"),
                        "end_time": progress_data.get("end_time"),
                        "error_message": progress_data.get("error_message"),
                        "control": tracked_vars
                    }

                    # Reload control.json to get updated tracked variables
                    if control_file.exists():
                        with open(control_file, 'r') as f:
                            tracked_vars = json.load(f)
                            file_data["control"] = tracked_vars

                stats = format_training_stats(file_data)
                last_stats_update = current_time
            except Exception:
                stats = format_training_stats(file_data)

        # Update resources at regular intervals
        if current_time - last_resources_update >= resources_update_interval:
            try:
                resources = get_process_resources(pid)
                overall_resources = get_overall_resources()
                last_resources_update = current_time
            except Exception:
                pass

        # Update output lines at regular intervals
        if show_output and (current_time - last_output_update >= output_update_interval):
            if stdout_file or stderr_file:
                output_lines = tail_both_files(stdout_file, stderr_file, max_lines=1000)
                last_output_update = current_time

                # Auto-scroll to bottom
                if output_auto_scroll:
                    # Will be calculated based on available space
                    pass

        stdscr.clear()
        term_height, term_width = stdscr.getmaxyx()

        # Calculate split layout - 60% left for editor/threads, 40% right for resources
        left_panel_width = int(term_width * 0.6)
        right_panel_width = term_width - left_panel_width - 1  # -1 for separator

        row = 0

        # Header (full width)
        mode_indicator = "[VARIABLES]" if view_mode == ViewMode.VARIABLES else "[THREADS]"
        header_text = f"Variable Editor - {mode_indicator} | ← Variables | → Threads | ESC Exit"
        print_colored_line(stdscr, row, 0, pad_line("=" * 200, term_width), ColorScheme.SEPARATOR); row += 1
        print_colored_line(stdscr, row, 0, pad_line(header_text, term_width), ColorScheme.HEADER); row += 1
        print_colored_line(stdscr, row, 0, pad_line(f"Training: {training_folder.name}", term_width), ColorScheme.VALUE); row += 1
        print_colored_line(stdscr, row, 0, pad_line("=" * 200, term_width), ColorScheme.SEPARATOR); row += 1

        # Training statistics section (full width)
        value_width = 12

        # Status and Progress
        status_info = stats["status_info"]
        line = f"Status: {status_info['Status']:<10} Phase: {status_info['Phase']:<10} Epoch: {status_info['Epoch']:<8} Batch: {status_info['Batch']}"
        print_colored_line(stdscr, row, 0, pad_line(line, term_width), ColorScheme.VALUE); row += 1

        # Timing - compact format
        run_timing = stats["run_timing"]
        epoch_timing = stats["epoch_timing"]
        training_timing = stats["training_timing"]
        validation_timing = stats["validation_timing"]

        line = f"Run: {run_timing['Elapsed']:<{value_width}} / {run_timing['Total']:<{value_width}} | Epoch: {epoch_timing['Elapsed']:<{value_width}} / {epoch_timing['Total']:<{value_width}}"
        print_colored_line(stdscr, row, 0, pad_line(line, term_width), ColorScheme.VALUE); row += 1

        line = f"Training: {training_timing['Elapsed']:<{value_width}} / {training_timing['Total']:<{value_width}} | Validation: {validation_timing['Elapsed']:<{value_width}} / {validation_timing['Total']:<{value_width}}"
        print_colored_line(stdscr, row, 0, pad_line(line, term_width), ColorScheme.VALUE); row += 1

        # Metrics
        metrics = stats["metrics"]
        line = f"Loss: {metrics['Loss']:<10} LR: {metrics['Learning Rate']}"
        print_colored_line(stdscr, row, 0, pad_line(line, term_width), ColorScheme.VALUE); row += 1

        print_colored_line(stdscr, row, 0, pad_line("-" * 200, term_width), ColorScheme.SEPARATOR); row += 1

        header_end_row = row

        # Calculate available space for panels (reserve space for output section at bottom)
        output_section_height = min(15, term_height // 3)  # 1/3 of screen or 15 lines max
        panel_height = term_height - header_end_row - output_section_height - 3  # -3 for separators

        # Render left panel (Variables OR Threads view)
        if view_mode == ViewMode.VARIABLES:
            render_variables_view(
                stdscr, header_end_row, 0, left_panel_width, panel_height,
                tracked_vars, keys, current_index, editing, edit_buffer, error_message
            )
        else:  # THREADS mode
            rows_rendered, total_rows, current_row_info = render_threads_view(
                stdscr, header_end_row, 0, left_panel_width, panel_height,
                resources, threads_cursor, expanded_processes, threads_scroll
            )

            # Auto-scroll to keep cursor visible
            available_content_rows = panel_height - 3  # -3 for header lines
            if threads_cursor < threads_scroll:
                threads_scroll = threads_cursor
            elif threads_cursor >= threads_scroll + available_content_rows:
                threads_scroll = threads_cursor - available_content_rows + 1

            # Clamp scroll
            max_scroll = max(0, total_rows - available_content_rows)
            threads_scroll = max(0, min(threads_scroll, max_scroll))

        # Draw vertical separator
        for sep_row in range(header_end_row, header_end_row + panel_height):
            if sep_row < term_height:
                print_colored_line(stdscr, sep_row, left_panel_width, "|", ColorScheme.SEPARATOR)

        # Render right panel (Resources)
        render_resources_panel(
            stdscr, header_end_row, left_panel_width + 1, right_panel_width, panel_height,
            resources, overall_resources
        )

        # Output viewer section at bottom
        output_row = header_end_row + panel_height
        if show_output and output_row < term_height:
            print_colored_line(stdscr, output_row, 0, pad_line("=" * 200, term_width), ColorScheme.SEPARATOR)
            output_row += 1

            if output_row < term_height:
                if stdout_file or stderr_file:
                    stdout_str = f"stdout: {stdout_file.name}" if stdout_file else ""
                    stderr_str = f"stderr: {stderr_file.name}" if stderr_file else ""
                    files_str = " | ".join(filter(None, [stdout_str, stderr_str]))
                    print_colored_line(stdscr, output_row, 0, pad_line(f"Training Output - {files_str}", term_width), ColorScheme.HEADER)
                else:
                    print_colored_line(stdscr, output_row, 0, pad_line("Training Output - No output files", term_width), ColorScheme.HEADER)
                output_row += 1

            if output_row < term_height:
                print_colored_line(stdscr, output_row, 0, pad_line("-" * 200, term_width), ColorScheme.SEPARATOR)
                output_row += 1

            # Display output lines
            output_start_row = output_row
            available_output_rows = term_height - output_row - 1  # -1 for footer

            if stdout_file or stderr_file and output_lines:
                # Auto-scroll to bottom
                max_scroll = max(0, len(output_lines) - available_output_rows)
                if output_auto_scroll:
                    output_scroll = max_scroll

                # Clamp scroll position
                output_scroll = max(0, min(output_scroll, max_scroll))

                # Display output lines
                visible_lines = list(output_lines)[output_scroll:output_scroll + available_output_rows]
                for timestamp, line_text, is_stderr in visible_lines:
                    if output_row >= term_height - 1:
                        break
                    time_str = timestamp.strftime("%H:%M:%S")
                    prefix = f"[{time_str}] "
                    full_line = prefix + line_text

                    default_color = ColorScheme.ERROR if is_stderr else ColorScheme.NORMAL
                    print_ansi_colored_line(stdscr, output_row, 0, full_line, term_width, default_color)
                    output_row += 1

                # Footer with scroll info
                if output_row < term_height:
                    scroll_info = f"Lines {output_scroll + 1}-{min(output_scroll + available_output_rows, len(output_lines))} of {len(output_lines)}"
                    auto_scroll_indicator = "[AUTO-SCROLL]" if output_auto_scroll else ""
                    footer = f"{scroll_info} {auto_scroll_indicator}"
                    print_colored_line(stdscr, term_height - 1, 0, pad_line(footer, term_width), ColorScheme.VALUE)
            else:
                # Show message about no output
                if output_row < term_height - 1:
                    print_colored_line(stdscr, output_row, 0, pad_line("No output available", term_width), ColorScheme.ERROR)

        stdscr.refresh()

        # Handle input
        key = stdscr.getch()

        if editing:
            # Editing mode (only in VARIABLES view)
            if key == 27:  # ESC
                editing = False
                edit_buffer = ""
                error_message = ""
            elif key == 10 or key == 13:  # Enter
                current_key = keys[current_index]
                original_type = type(tracked_vars[current_key])
                new_value = convert_value(edit_buffer, original_type)

                if new_value is not None:
                    # Update the tracked variable in local copy
                    tracked_vars[current_key] = new_value

                    # Write back to control.json
                    try:
                        with open(control_file, 'w') as f:
                            json.dump(tracked_vars, f, indent=2)

                        editing = False
                        edit_buffer = ""
                        error_message = ""
                    except Exception as e:
                        error_message = f"Error writing to control file: {e}"
                else:
                    error_message = f"Error: Invalid {original_type.__name__} value"

            elif key == curses.KEY_BACKSPACE or key == 127 or key == 8:
                if edit_buffer:
                    edit_buffer = edit_buffer[:-1]
                error_message = ""
            elif 32 <= key <= 126:  # Printable characters
                edit_buffer += chr(key)
                error_message = ""
        else:
            # Navigation mode
            if key == 27:  # ESC
                break
            elif key == curses.KEY_LEFT:
                # Switch to VARIABLES view
                view_mode = ViewMode.VARIABLES
                error_message = ""
            elif key == curses.KEY_RIGHT:
                # Switch to THREADS view
                view_mode = ViewMode.THREADS
                error_message = ""
            elif key == curses.KEY_UP:
                if view_mode == ViewMode.VARIABLES:
                    current_index = (current_index - 1) % len(keys)
                    error_message = ""
                else:  # THREADS mode
                    if threads_cursor > 0:
                        threads_cursor -= 1
            elif key == curses.KEY_DOWN:
                if view_mode == ViewMode.VARIABLES:
                    current_index = (current_index + 1) % len(keys)
                    error_message = ""
                else:  # THREADS mode
                    # Get total rows to limit cursor
                    if resources and 'detailed_processes' in resources:
                        thread_rows = build_thread_display(resources, threads_cursor, expanded_processes)
                        if threads_cursor < len(thread_rows) - 1:
                            threads_cursor += 1
            elif key == 10 or key == 13:  # Enter
                if view_mode == ViewMode.VARIABLES:
                    # Start editing variable
                    editing = True
                    edit_buffer = str(tracked_vars[keys[current_index]])
                    error_message = ""
                else:  # THREADS mode
                    # Toggle process expansion
                    if resources and 'detailed_processes' in resources:
                        thread_rows = build_thread_display(resources, threads_cursor, expanded_processes)
                        if threads_cursor < len(thread_rows):
                            current_row = thread_rows[threads_cursor]
                            if current_row.row_type == ThreadRowType.PROCESS:
                                proc_idx = current_row.process_idx
                                if proc_idx in expanded_processes:
                                    expanded_processes.remove(proc_idx)
                                else:
                                    expanded_processes.add(proc_idx)
            elif key == ord('t') or key == ord('T'):
                # Alternative key for toggling process expansion in THREADS mode
                if view_mode == ViewMode.THREADS and resources and 'detailed_processes' in resources:
                    thread_rows = build_thread_display(resources, threads_cursor, expanded_processes)
                    if threads_cursor < len(thread_rows):
                        current_row = thread_rows[threads_cursor]
                        if current_row.row_type == ThreadRowType.PROCESS:
                            proc_idx = current_row.process_idx
                            if proc_idx in expanded_processes:
                                expanded_processes.remove(proc_idx)
                            else:
                                expanded_processes.add(proc_idx)

        time.sleep(0.01)
