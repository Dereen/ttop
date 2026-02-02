#!/usr/bin/env python3
"""
Monitor training files created by train/utils/performance/training_monitor.py
and display their data along with system resource usage in a single line.
"""

import sys
import os
from pathlib import Path

# Add project root to path if running as script
if __name__ == "__main__" and __package__ is None:
    project_root = Path(__file__).resolve().parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

import time
import json
import psutil
import curses
import argparse
import threading
import re
import signal
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import queue

from ttop.time_utils import format_duration, format_short_duration
from ttop.background_data_collector import BackgroundDataCollector
from ttop.data_formatting import build_data_matrix, calculate_column_widths_from_matrix
from ttop.color_scheme import ColorScheme
from ttop.output_viewer import show_output_window, get_output_files_from_metadata
from ttop.variable_editor import show_variable_editor
from dataclasses import dataclass
from enum import Enum


class RowType(Enum):
    """Type of row in the display matrix."""
    TRAINING = "training"
    PROCESS = "process"
    THREAD = "thread"


@dataclass
class DisplayRow:
    """Represents a single row in the display with its associated action."""
    row_type: RowType
    training_idx: int  # Index in filtered_indices
    data_idx: int  # Index in data['training_data']
    process_idx: Optional[int] = None  # Index in processes list (for PROCESS/THREAD rows)
    thread_idx: Optional[int] = None  # Index in threads list (for THREAD rows)
    text: str = ""
    color_pair: int = ColorScheme.NORMAL

    def get_state_key(self) -> str:
        """Generate unique state key for this row."""
        if self.row_type == RowType.TRAINING:
            return f"training_{self.training_idx}"
        elif self.row_type == RowType.PROCESS:
            return f"training_{self.training_idx}_process_{self.process_idx}"
        elif self.row_type == RowType.THREAD:
            return f"training_{self.training_idx}_process_{self.process_idx}_thread_{self.thread_idx}"
        return ""


class RowState:
    """Manages state for all rows in the display."""

    def __init__(self):
        self.expanded_states: Dict[str, bool] = {}

    def is_expanded(self, row: DisplayRow) -> bool:
        """Check if a row is expanded."""
        return self.expanded_states.get(row.get_state_key(), False)

    def toggle(self, row: DisplayRow):
        """Toggle the expanded state of a row."""
        key = row.get_state_key()
        self.expanded_states[key] = not self.expanded_states.get(key, False)

    def set_expanded(self, row: DisplayRow, expanded: bool):
        """Set the expanded state of a row."""
        self.expanded_states[row.get_state_key()] = expanded

    def clear(self):
        """Clear all states."""
        self.expanded_states.clear()


def format_process_thread_row(name: str, pid: int, cpu: Optional[float],
                              ram_mb: Optional[float], gpu_mib: Optional[int],
                              threads: Optional[int],
                              column_widths: Dict[str, int]) -> str:
    """Format a process or thread row using table columns.

    Args:
        name: Process/thread name (with [P]/[T] prefix and indentation)
        pid: Process ID or Thread ID
        cpu: CPU percentage (None for threads)
        ram_mb: RAM in MB (None for threads)
        gpu_mib: GPU memory in MiB (None for threads)
        threads: Thread count (None for threads)
        column_widths: Column widths dictionary

    Returns:
        Formatted row string
    """
    # Build row with same column structure as training data
    # Columns: Run Name, Status, Phase, Epoch, Batch,
    #          Run Elapsed, Run Remaining, Run Total,
    #          Epoch Elapsed, Epoch Remaining, Epoch Total,
    #          Train Elapsed, Train Remaining, Train Total,
    #          Val Time, Val Elapsed, Val Remaining, Val Total,
    #          CPU, RAM, GPU, Procs, Threads,
    #          PID, Wandb URL

    # Empty strings for non-applicable columns (17 empty columns from Status to Val Total)
    empty_cols = [""] * 17  # Status, Phase, Epoch, Batch (4),
                            # Run Elapsed, Run Remaining, Run Total (3),
                            # Epoch Elapsed, Epoch Remaining, Epoch Total (3),
                            # Train Elapsed, Train Remaining, Train Total (3),
                            # Val Time, Val Elapsed, Val Remaining, Val Total (4)

    # CPU column
    cpu_str = f"{cpu:.1f}%" if cpu is not None else ""

    # RAM column
    ram_str = f"{ram_mb:.0f}MB" if ram_mb is not None else ""

    # GPU column
    gpu_str = f"{gpu_mib}MiB" if gpu_mib is not None and gpu_mib > 0 else ""

    # Procs column - empty for process/thread rows
    procs_str = ""

    # Threads column
    threads_str = str(threads) if threads is not None else ""

    # PID column (for processes/threads, show their PID/TID here)
    pid_str = str(pid) if pid is not None else ""

    # Empty Wandb URL
    wandb_str = ""

    # Build the row data matching the exact column structure
    row_data = [name] + empty_cols + [cpu_str, ram_str, gpu_str, procs_str, threads_str, pid_str, wandb_str]

    # Use the same column names from data_formatting.py
    column_names = [
        "Run Name", "Status", "Phase", "Epoch", "Batch",
        "Run Elapsed", "Run Remaining", "Run Total",
        "Epoch Elapsed", "Epoch Remaining", "Epoch Total",
        "Train Elapsed", "Train Remaining", "Train Total",
        "Val Time", "Val Elapsed", "Val Remaining", "Val Total",
        "CPU", "RAM", "GPU", "Procs", "Threads",
        "PID", "Wandb URL"
    ]

    return format_row(row_data, column_names, column_widths)


def get_column_positions(column_names: List[str], widths: Dict[str, int]) -> List[int]:
    """Get the starting positions of each scrollable column based on dynamic widths."""
    positions = [0]
    pos = 0

    # Skip Run Name (first column, always fixed)
    for col_name in column_names[1:]:
        positions.append(pos)
        pos += widths.get(col_name, 10) + 3  # 3 for " | "

    return positions


def get_column_headers(column_names: List[str], widths: Dict[str, int]) -> str:
    """Get column headers with dynamic widths."""
    headers = []
    for col_name in column_names:
        width = widths.get(col_name, len(col_name))
        headers.append(f"{col_name:{width}}")
    return " | ".join(headers)


def format_row(row: List[str], column_names: List[str], widths: Dict[str, int]) -> str:
    """Format a data row with proper column widths.

    Args:
        row: List of cell values
        column_names: List of column names
        widths: Dictionary mapping column names to widths

    Returns:
        Formatted row string
    """
    parts = []
    for col_name, value in zip(column_names, row):
        width = widths.get(col_name, len(col_name))
        parts.append(f"{value:{width}}")
    return " | ".join(parts)


def init_curses(stdscr):
    """Initialize curses settings."""
    curses.curs_set(0)  # Hide cursor
    stdscr.nodelay(1)  # Non-blocking getch()
    stdscr.timeout(50)  # 50ms timeout for getch()
    curses.noecho()
    curses.cbreak()

    # Initialize colors using ColorScheme
    ColorScheme.init_colors()

    # Set background
    stdscr.bkgd(' ', curses.color_pair(ColorScheme.NORMAL))

    return stdscr


def cleanup_curses():
    """Cleanup curses settings."""
    curses.nocbreak()
    curses.echo()
    curses.endwin()


def print_colored_line(stdscr, row: int, col: int, text: str, color_pair: int = ColorScheme.NORMAL):
    """Print a line with color, handling errors gracefully."""
    try:
        stdscr.addstr(row, col, text, curses.color_pair(color_pair))
    except curses.error:
        pass


def build_display_matrix(data: Dict, filtered_indices: List[int],
                         row_state: RowState,
                         column_names: List[str],
                         column_widths: Dict[str, int],
                         cursor_position: int) -> List[DisplayRow]:
    """Build the display matrix with all rows and their actions.

    Args:
        data: Data dictionary from background collector
        filtered_indices: List of filtered training indices
        row_state: RowState object tracking expanded/collapsed states
        column_names: Column names
        column_widths: Column widths
        cursor_position: Current cursor position

    Returns:
        List of DisplayRow objects
    """
    display_rows = []
    data_rows = data.get('data_rows', [])

    for training_idx, idx in enumerate(filtered_indices):
        if idx >= len(data_rows):
            continue

        # Main training row
        row_data = data_rows[idx]
        output = format_row(row_data, column_names, column_widths)

        json_file, file_data, resources = data['training_data'][idx]
        phase = file_data.get("state", {}).get("phase", "Unknown")
        status = file_data.get("metadata", {}).get("status", "Unknown")

        is_cursor = (len(display_rows) == cursor_position)
        color_pair = ColorScheme.get_phase_color(phase, status, cursor=is_cursor)

        training_row = DisplayRow(
            row_type=RowType.TRAINING,
            training_idx=training_idx,
            data_idx=idx,
            text=output,
            color_pair=color_pair
        )
        display_rows.append(training_row)

        # Add process/thread details if training is expanded
        if row_state.is_expanded(training_row) and resources and 'detailed_processes' in resources:
            processes = resources['detailed_processes']

            for proc_idx, proc_info in enumerate(processes):
                proc_name = proc_info.get('name', 'unknown')
                proc_pid = proc_info.get('pid', 0)
                proc_cpu = proc_info.get('cpu_percent', 0.0)
                proc_ram_mb = proc_info.get('ram_mb', 0.0)
                proc_gpu_mib = proc_info.get('gpu_mem_mib', 0)
                proc_threads_count = proc_info.get('num_threads', 0)

                # Format process row using table-like columns
                # Name column is indented with [P] prefix
                proc_name_col = f"  [P] {proc_name}"
                proc_line = format_process_thread_row(
                    proc_name_col, proc_pid, proc_cpu, proc_ram_mb, proc_gpu_mib, proc_threads_count,
                    column_widths
                )

                is_cursor = (len(display_rows) == cursor_position)
                color_pair = ColorScheme.PROCESS_CURSOR if is_cursor else ColorScheme.PROCESS

                process_row = DisplayRow(
                    row_type=RowType.PROCESS,
                    training_idx=training_idx,
                    data_idx=idx,
                    process_idx=proc_idx,
                    text=proc_line,
                    color_pair=color_pair
                )
                display_rows.append(process_row)

                # Add thread rows if process is expanded
                if row_state.is_expanded(process_row):
                    threads = proc_info.get('threads', [])
                    for thread_idx, thread_info in enumerate(threads):
                        thread_id = thread_info.get('thread_id', 0)
                        thread_name = thread_info.get('thread_name', 'unknown')

                        # Format thread row using table-like columns
                        # Thread name is more indented with [T] prefix, TID goes in PID column
                        thread_name_col = f"    [T] {thread_name}"
                        thread_line = format_process_thread_row(
                            thread_name_col, thread_id, None, None, None, None,
                            column_widths
                        )

                        is_cursor = (len(display_rows) == cursor_position)
                        color_pair = ColorScheme.THREAD_CURSOR if is_cursor else ColorScheme.THREAD

                        display_rows.append(DisplayRow(
                            row_type=RowType.THREAD,
                            training_idx=training_idx,
                            data_idx=idx,
                            process_idx=proc_idx,
                            thread_idx=thread_idx,
                            text=thread_line,
                            color_pair=color_pair
                        ))

    return display_rows


def pad_line(text: str, width: int, offset: int = 0, keep_prefix: int = 0) -> str:
    """Pad line with spaces to fill terminal width, with horizontal scrolling offset.

    Args:
        text: The text to pad
        width: Terminal width
        offset: Horizontal scroll offset
        keep_prefix: Number of characters to keep fixed at the start (not scrolled)
    """
    if keep_prefix > 0 and offset > 0:
        prefix = text[:keep_prefix]
        scrollable = text[keep_prefix:]

        if offset < len(scrollable):
            scrollable = scrollable[offset:]
        else:
            scrollable = ""

        text = prefix + scrollable
    elif offset > 0:
        text = text[offset:]

    if len(text) >= width:
        return text[:width]
    return text + ' ' * (width - len(text))


def show_signal_window(stdscr, pid: int, target_name: str) -> Optional[int]:
    """Show signal selection window and return selected signal number.

    Args:
        stdscr: Curses screen
        pid: Process ID to send signal to
        target_name: Name of the target process/thread

    Returns:
        Signal number if selected, None if canceled
    """
    # Available signals in desired order (SIGKILL first)
    available_signals = [
        (signal.SIGKILL, "SIGKILL", "Kill immediately (cannot be caught)"),
        (signal.SIGTERM, "SIGTERM", "Terminate gracefully"),
        (signal.SIGINT, "SIGINT", "Interrupt (Ctrl+C)"),
        (signal.SIGHUP, "SIGHUP", "Hangup"),
        (signal.SIGQUIT, "SIGQUIT", "Quit with core dump"),
        (signal.SIGSTOP, "SIGSTOP", "Stop process (cannot be caught)"),
        (signal.SIGCONT, "SIGCONT", "Continue if stopped"),
    ]

    cursor = 0
    term_height, term_width = stdscr.getmaxyx()

    while True:
        stdscr.clear()

        # Calculate window dimensions
        window_width = min(70, term_width - 4)
        window_height = min(len(available_signals) + 6, term_height - 4)
        start_y = (term_height - window_height) // 2
        start_x = (term_width - window_width) // 2

        # Draw window border
        for y in range(start_y, start_y + window_height):
            if y == start_y or y == start_y + window_height - 1:
                print_colored_line(stdscr, y, start_x, "=" * window_width, ColorScheme.SEPARATOR)
            else:
                print_colored_line(stdscr, y, start_x, " " * window_width, ColorScheme.NORMAL)

        # Title
        title = f"Send Signal to PID {pid} ({target_name})"
        print_colored_line(stdscr, start_y + 1, start_x + 2, title[:window_width-4], ColorScheme.HEADER)

        # Instructions
        instructions = "Press Enter to send | Press 'k' or ESC to cancel"
        print_colored_line(stdscr, start_y + 2, start_x + 2, instructions[:window_width-4], ColorScheme.VALUE)

        # Signal list
        for idx, (sig_num, sig_name, sig_desc) in enumerate(available_signals):
            y_pos = start_y + 4 + idx
            if y_pos >= start_y + window_height - 1:
                break

            signal_line = f"{sig_name:12} - {sig_desc}"
            if idx == cursor:
                print_colored_line(stdscr, y_pos, start_x + 2, f"> {signal_line[:window_width-6]}", ColorScheme.PROCESS_CURSOR)
            else:
                print_colored_line(stdscr, y_pos, start_x + 2, f"  {signal_line[:window_width-6]}", ColorScheme.NORMAL)

        stdscr.refresh()

        # Handle input
        key = stdscr.getch()
        if key == curses.KEY_UP:
            cursor = max(0, cursor - 1)
        elif key == curses.KEY_DOWN:
            cursor = min(len(available_signals) - 1, cursor + 1)
        elif key == 10 or key == 13:  # Enter
            return available_signals[cursor][0]
        elif key == ord('k') or key == ord('K') or key == 27:  # k or ESC
            return None








def scan_folder_curses(stdscr, folder_path: Path, interval: float = 2.0,
                        max_cpus: Optional[int] = None,
                        max_ram_gb: Optional[float] = None,
                        max_gpu_mem_gb: Optional[float] = None,
                        csv_output_interval: Optional[float] = None,
                        csv_output_dir: Optional[Path] = None):
    """Scan folder for training monitor files and display their data."""
    stdscr = init_curses(stdscr)

    status_filters = ["all", "RUNNING", "FINISHED", "CANCELED", "FAILED", "KILLED"]
    current_filter_index = 0
    scroll_offset = 0
    current_column_index = 0
    vertical_scroll = 0
    cursor_position = 0
    name_filter = ""
    entering_filter = False
    row_state = RowState()  # Track expanded/collapsed state of all rows

    # Start background data collector
    collector = BackgroundDataCollector(
        folder_path, interval, max_cpus, max_ram_gb, max_gpu_mem_gb,
        csv_output_interval, csv_output_dir
    )
    collector.start()

    # Wait for first data collection
    while collector.get_latest_data() is None:
        time.sleep(0.1)

    last_ui_update = 0
    ui_update_interval = 0.05  # Update UI every 50ms
    column_positions = []  # Will be updated dynamically
    toggle_details_key_pressed = False

    try:
        while True:
            current_time = time.time()

            # Handle keyboard input continuously
            key = stdscr.getch()
            toggle_details_key_pressed = False
            if key != -1:
                if entering_filter:
                    if key == 27:  # ESC
                        entering_filter = False
                        name_filter = ""
                    elif key == 10 or key == 13:  # Enter
                        entering_filter = False
                    elif key == curses.KEY_BACKSPACE or key == 127 or key == 8:
                        if name_filter:
                            name_filter = name_filter[:-1]
                    elif 32 <= key <= 126:  # Printable characters
                        name_filter += chr(key)
                else:
                    if key == ord('s') or key == ord('S'):
                        current_filter_index = (current_filter_index + 1) % len(status_filters)
                        vertical_scroll = 0
                        cursor_position = 0
                    elif key == ord('f') or key == ord('F'):
                        entering_filter = True
                        name_filter = ""
                    elif key == ord('t') or key == ord('T'):
                        # Mark that toggle was requested - will handle after building display matrix
                        toggle_details_key_pressed = True
                    elif key == ord('o') or key == ord('O'):
                        # Show output window for selected training
                        data = collector.get_latest_data()
                        if data:
                            # Apply filters to get filtered_indices
                            import fnmatch
                            filtered_indices = []
                            for idx, (json_file, file_data, resources) in enumerate(data['training_data']):
                                status = file_data.get("metadata", {}).get("status", "UNKNOWN")
                                if current_filter != "all" and status != current_filter:
                                    continue
                                if name_filter and not fnmatch.fnmatch(json_file.name.lower(), f"*{name_filter}*"):
                                    continue
                                filtered_indices.append(idx)

                            # Get metadata and find output files
                            if cursor_position < len(filtered_indices):
                                idx = filtered_indices[cursor_position]
                                if idx < len(data['training_data']):
                                    json_file, file_data, resources = data['training_data'][idx]
                                    metadata = file_data.get("metadata", {})
                                    stdout_file, stderr_file = get_output_files_from_metadata(metadata)
                                    # Show output window
                                    show_output_window(stdscr, stdout_file, stderr_file, metadata)

                            # Reinitialize curses after returning from output window
                            stdscr = init_curses(stdscr)
                    elif key == ord('e') or key == ord('E'):
                        # Show variable editor for selected training
                        data = collector.get_latest_data()
                        if data:
                            # Apply filters to get filtered_indices
                            import fnmatch
                            filtered_indices = []
                            for idx, (json_file, file_data, resources) in enumerate(data['training_data']):
                                status = file_data.get("metadata", {}).get("status", "UNKNOWN")
                                if current_filter != "all" and status != current_filter:
                                    continue
                                if name_filter and not fnmatch.fnmatch(json_file.name.lower(), f"*{name_filter}*"):
                                    continue
                                filtered_indices.append(idx)

                            # Open editor for selected training
                            if cursor_position < len(filtered_indices):
                                idx = filtered_indices[cursor_position]
                                if idx < len(data['training_data']):
                                    training_folder, file_data, resources = data['training_data'][idx]
                                    # Show editor window
                                    show_variable_editor(stdscr, training_folder, file_data)

                            # Reinitialize curses after returning from editor
                            stdscr = init_curses(stdscr)
                    elif key == ord('l') or key == ord('L'):
                        # l: Send signal to all RUNNING training runs
                        data = collector.get_latest_data()
                        if data:
                            # Collect all RUNNING training PIDs
                            running_pids = []
                            for idx, (json_file, file_data, resources) in enumerate(data['training_data']):
                                status = file_data.get("metadata", {}).get("status", "UNKNOWN")
                                if status == "RUNNING":
                                    pid = file_data.get("metadata", {}).get("pid")
                                    if pid:
                                        running_pids.append((pid, json_file.name))

                            if running_pids:
                                # Show signal selection window
                                target_name = f"{len(running_pids)} RUNNING training runs"
                                selected_signal = show_signal_window(stdscr, running_pids[0][0], target_name)
                                if selected_signal is not None:
                                    # Send signal to all RUNNING training runs
                                    for pid, name in running_pids:
                                        try:
                                            os.kill(pid, selected_signal)
                                        except Exception:
                                            pass

                            # Reinitialize curses after returning from signal window
                            stdscr = init_curses(stdscr)
                    elif key == ord('k'):
                        # k: Show signal selection window for selected process/thread
                        data = collector.get_latest_data()
                        if data:
                            # Apply filters to get filtered_indices
                            import fnmatch
                            filtered_indices = []
                            for idx, (json_file, file_data, resources) in enumerate(data['training_data']):
                                status = file_data.get("metadata", {}).get("status", "UNKNOWN")
                                if current_filter != "all" and status != current_filter:
                                    continue
                                if name_filter and not fnmatch.fnmatch(json_file.name.lower(), f"*{name_filter}*"):
                                    continue
                                filtered_indices.append(idx)

                            # Build display matrix to get current row
                            display_rows = build_display_matrix(
                                data, filtered_indices, row_state,
                                column_names, column_widths, cursor_position
                            )

                            if cursor_position < len(display_rows):
                                current_row = display_rows[cursor_position]
                                target_pid = None
                                target_name = ""

                                if current_row.row_type == RowType.TRAINING:
                                    # Get main process PID
                                    idx = current_row.data_idx
                                    if idx < len(data['training_data']):
                                        json_file, file_data, resources = data['training_data'][idx]
                                        target_pid = file_data.get("metadata", {}).get("pid")
                                        target_name = json_file.name
                                elif current_row.row_type == RowType.PROCESS:
                                    # Get specific process PID
                                    idx = current_row.data_idx
                                    if idx < len(data['training_data']):
                                        json_file, file_data, resources = data['training_data'][idx]
                                        if resources and 'detailed_processes' in resources:
                                            processes = resources['detailed_processes']
                                            if current_row.process_idx < len(processes):
                                                proc_info = processes[current_row.process_idx]
                                                target_pid = proc_info.get('pid')
                                                target_name = proc_info.get('name', 'unknown')
                                elif current_row.row_type == RowType.THREAD:
                                    # Get thread ID (which is treated as PID for signals)
                                    idx = current_row.data_idx
                                    if idx < len(data['training_data']):
                                        json_file, file_data, resources = data['training_data'][idx]
                                        if resources and 'detailed_processes' in resources:
                                            processes = resources['detailed_processes']
                                            if current_row.process_idx < len(processes):
                                                proc_info = processes[current_row.process_idx]
                                                threads = proc_info.get('threads', [])
                                                if current_row.thread_idx < len(threads):
                                                    thread_info = threads[current_row.thread_idx]
                                                    target_pid = thread_info.get('thread_id')
                                                    target_name = thread_info.get('thread_name', 'unknown')

                                if target_pid:
                                    # Show signal selection window
                                    selected_signal = show_signal_window(stdscr, target_pid, target_name)
                                    if selected_signal is not None:
                                        try:
                                            os.kill(target_pid, selected_signal)
                                        except Exception:
                                            pass

                            # Reinitialize curses after returning from signal window
                            stdscr = init_curses(stdscr)
                    elif key == ord('c') or key == ord('C'):
                        # Delete selected training folder
                        data = collector.get_latest_data()
                        if data:
                            # Apply filters to get filtered_indices
                            import fnmatch
                            import shutil
                            filtered_indices = []
                            for idx, (json_file, file_data, resources) in enumerate(data['training_data']):
                                status = file_data.get("metadata", {}).get("status", "UNKNOWN")
                                if current_filter != "all" and status != current_filter:
                                    continue
                                if name_filter and not fnmatch.fnmatch(json_file.name.lower(), f"*{name_filter}*"):
                                    continue
                                filtered_indices.append(idx)

                            # Delete the selected training folder
                            if cursor_position < len(filtered_indices):
                                idx = filtered_indices[cursor_position]
                                if idx < len(data['training_data']):
                                    training_folder, file_data, resources = data['training_data'][idx]
                                    try:
                                        # Delete the entire training folder
                                        shutil.rmtree(training_folder)
                                        # Adjust cursor position if needed
                                        if cursor_position >= len(filtered_indices) - 1:
                                            cursor_position = max(0, cursor_position - 1)
                                    except Exception:
                                        pass
                    elif key == 27:  # ESC - clear filter
                        name_filter = ""
                        vertical_scroll = 0
                        cursor_position = 0
                        row_state.clear()
                    elif key == curses.KEY_UP:
                        if cursor_position > 0:
                            cursor_position -= 1
                    elif key == curses.KEY_DOWN:
                        cursor_position += 1
                    elif key == curses.KEY_RIGHT:
                        if current_column_index < len(column_positions) - 1:
                            current_column_index += 1
                            scroll_offset = column_positions[current_column_index]
                    elif key == curses.KEY_LEFT:
                        if current_column_index > 0:
                            current_column_index -= 1
                            scroll_offset = column_positions[current_column_index]

            # Update UI at regular intervals
            if current_time - last_ui_update >= ui_update_interval:
                # Get latest data from background thread
                data = collector.get_latest_data()
                if data is None:
                    continue

                stdscr.clear()
                current_filter = status_filters[current_filter_index]
                term_width = curses.COLS

                scroll_indicator = ""
                if scroll_offset > 0:
                    scroll_indicator = f" [← Column {current_column_index}/{len(column_positions)-1}]"

                filter_indicator = ""
                if name_filter:
                    filter_indicator = f" | Name filter: '{name_filter}'"
                elif entering_filter:
                    filter_indicator = f" | Entering filter: '{name_filter}_'"

                row = 0
                header_height = 10  # Increased to accommodate color legend

                # Top banner
                print_colored_line(stdscr, row, 0, pad_line("=" * 200, term_width), ColorScheme.SEPARATOR); row += 1
                print_colored_line(stdscr, row, 0, pad_line(f"Training Monitor - 's' status | 'f' filter name | 't' toggle details | 'e' edit variables | 'o' output | 'k' kill signal | 'l' kill all RUNNING | 'c' delete | '↑↓' move | '←→' columns{scroll_indicator}{filter_indicator}", term_width), ColorScheme.HEADER); row += 1
                print_colored_line(stdscr, row, 0, pad_line(f"Status: {current_filter} | Filters: {' -> '.join(status_filters)}", term_width), ColorScheme.STATUS); row += 1

                # Color legend
                try:
                    legend = "Colors: "
                    stdscr.addstr(row, 0, legend, curses.color_pair(ColorScheme.NORMAL))
                    col_pos = len(legend)
                    color_legends = [
                        ("Loading", ColorScheme.LOADING),
                        ("Batch", ColorScheme.BATCH),
                        ("Validation", ColorScheme.VALIDATION),
                        ("Finished", ColorScheme.FINISHED),
                        ("Failed", ColorScheme.FAILED),
                        ("Canceled", ColorScheme.CANCELED),
                        ("Killed", ColorScheme.KILLED)
                    ]
                    for label, color_pair in color_legends:
                        if col_pos + len(label) + 3 < term_width:
                            stdscr.addstr(row, col_pos, f" {label} ", curses.color_pair(color_pair))
                            col_pos += len(label) + 3
                except curses.error:
                    pass  # Handle terminal edge cases
                row += 1

                print_colored_line(stdscr, row, 0, pad_line(f"Press Ctrl+C to stop | Press ESC to clear filter", term_width), ColorScheme.NORMAL); row += 1

                # CSV output info
                if collector.csv_writer:
                    csv_abs_path = collector.csv_writer.get_file_path().absolute()
                    print_colored_line(stdscr, row, 0, pad_line(f"CSV Output: {csv_abs_path}", term_width), ColorScheme.VALUE); row += 1

                print_colored_line(stdscr, row, 0, pad_line("=" * 200, term_width), ColorScheme.SEPARATOR); row += 1

                # Get matrix data from background collector
                column_names = data.get('column_names', [])
                data_rows = data.get('data_rows', [])
                column_widths = data.get('column_widths', {})

                # Recalculate column positions based on dynamic widths
                if column_names and column_widths:
                    column_positions = get_column_positions(column_names, column_widths)

                # System resources
                overall_resources = data['overall_resources']
                resources_line = (f"User Resources: CPUs: {overall_resources['cpu_used']:.1f}% / {overall_resources['max_cpus']} cores ({overall_resources['cpu_usage_percent']:.1f}% of allocated) | "
                      f"RAM: {overall_resources['ram_used_gb']:.1f}GB / {overall_resources['max_ram_gb']:.1f}GB ({overall_resources['ram_usage_percent']:.1f}%) | "
                      f"GPU: {overall_resources['gpu_used_gb']:.1f}GB / {overall_resources['max_gpu_gb']:.1f}GB ({overall_resources['gpu_usage_percent']:.1f}%)")
                print_colored_line(stdscr, row, 0, pad_line(resources_line, term_width), ColorScheme.VALUE); row += 1
                print_colored_line(stdscr, row, 0, pad_line("=" * 200, term_width), ColorScheme.SEPARATOR); row += 1

                # Column headers (scrollable)
                headers = get_column_headers(column_names, column_widths) if column_names else ""
                header_keep_prefix = column_widths.get('Run Name', 40) + 3  # +3 for " | "
                print_colored_line(stdscr, row, 0, pad_line(headers, term_width, scroll_offset, keep_prefix=header_keep_prefix), ColorScheme.HEADER); row += 1
                print_colored_line(stdscr, row, 0, pad_line("-" * 200, term_width), ColorScheme.SEPARATOR); row += 1

                # Apply filters
                import fnmatch
                filtered_indices = []
                for idx, (json_file, file_data, resources) in enumerate(data['training_data']):
                    status = file_data.get("metadata", {}).get("status", "UNKNOWN")
                    if current_filter != "all" and status != current_filter:
                        continue

                    if name_filter and not fnmatch.fnmatch(json_file.name.lower(), f"*{name_filter}*"):
                        continue

                    filtered_indices.append(idx)

                max_visible_rows = curses.LINES - header_height - 2

                # Build display matrix with all rows
                display_rows = build_display_matrix(
                    data, filtered_indices, row_state,
                    column_names, column_widths, cursor_position
                )

                # Handle toggle details key press
                if toggle_details_key_pressed and cursor_position < len(display_rows):
                    current_row = display_rows[cursor_position]
                    row_state.toggle(current_row)
                    # Rebuild matrix after toggle
                    display_rows = build_display_matrix(
                        data, filtered_indices, row_state,
                        column_names, column_widths, cursor_position
                    )

                total_rows = len(display_rows)

                # Clamp cursor position
                if cursor_position >= total_rows:
                    cursor_position = max(0, total_rows - 1)
                if cursor_position < 0:
                    cursor_position = 0

                # Smart scrolling: only scroll when cursor reaches edges
                if cursor_position < vertical_scroll:
                    vertical_scroll = cursor_position
                elif cursor_position >= vertical_scroll + max_visible_rows:
                    vertical_scroll = cursor_position - max_visible_rows + 1

                # Clamp vertical scroll
                if vertical_scroll > max(0, total_rows - max_visible_rows):
                    vertical_scroll = max(0, total_rows - max_visible_rows)
                if vertical_scroll < 0:
                    vertical_scroll = 0

                if not data['json_files']:
                    timestamp = datetime.now().strftime('%H:%M:%S')
                    print_colored_line(stdscr, row, 0, pad_line("", term_width), ColorScheme.NORMAL); row += 1
                    print_colored_line(stdscr, row, 0, pad_line(f"{timestamp} | No training files found in {folder_path}", term_width), ColorScheme.ERROR); row += 1
                else:
                    # Calculate keep_prefix based on dynamic Run Name width
                    keep_prefix = column_widths.get('Run Name', 40) + 3  # +3 for " | "

                    # Render visible rows from display matrix
                    visible_rows = display_rows[vertical_scroll:vertical_scroll + max_visible_rows]
                    for display_row in visible_rows:
                        print_colored_line(stdscr, row, 0, pad_line(display_row.text, term_width, scroll_offset, keep_prefix=keep_prefix), display_row.color_pair)
                        row += 1

                    timestamp = datetime.now().strftime('%H:%M:%S')
                    scroll_info = ""
                    if total_rows > max_visible_rows:
                        scroll_info = f" | Rows {vertical_scroll + 1}-{min(vertical_scroll + max_visible_rows, total_rows)} of {total_rows}"

                    cursor_info = f" | Cursor: {cursor_position + 1}/{total_rows}" if total_rows > 0 else ""

                    data_age = time.time() - data['timestamp']
                    age_str = f" | Data: {data_age:.1f}s ago"

                    print_colored_line(stdscr, row, 0, pad_line("=" * 200, term_width), ColorScheme.SEPARATOR); row += 1
                    print_colored_line(stdscr, row, 0, pad_line(f"{timestamp} | Showing {len(filtered_indices)} training files | {total_rows} total rows{scroll_info}{cursor_info}{age_str}", term_width), ColorScheme.VALUE)

                stdscr.refresh()
                last_ui_update = current_time

            # Small sleep to prevent CPU spinning
            time.sleep(0.01)

    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        collector.stop()


def run_headless(folder_path: Path, interval: float = 2.0,
                  max_cpus: Optional[int] = None,
                  max_ram_gb: Optional[float] = None,
                  max_gpu_mem_gb: Optional[float] = None,
                  csv_output_interval: Optional[float] = None,
                  csv_output_dir: Optional[Path] = None):
    """Run data collector without UI, only outputting CSV data."""
    # Start background data collector
    collector = BackgroundDataCollector(
        folder_path, interval, max_cpus, max_ram_gb, max_gpu_mem_gb,
        csv_output_interval, csv_output_dir
    )
    collector.start()

    # Wait for first data collection
    while collector.get_latest_data() is None:
        time.sleep(0.1)

    print(f"Collecting data from {folder_path}")
    if csv_output_interval:
        print(f"CSV output interval: {csv_output_interval}s")
        print(f"CSV output directory: {csv_output_dir}")

    try:
        while True:
            data = collector.get_latest_data()
            if data:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                num_trainings = len(data['training_data'])
                print(f"[{timestamp}] Found {num_trainings} training(s)", flush=True)
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nStopping collector...")
    finally:
        collector.stop()
        collector.join(timeout=5)
        print("Collector stopped")


def main():
    parser = argparse.ArgumentParser(
        description="Monitor training files and display resource usage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s ./training_logs
  %(prog)s ./training_logs --interval 2.0
  %(prog)s ./training_logs --max-cpus 16 --max-ram 64 --max-gpu 24
  %(prog)s ./training_logs -i 1.5 --max-cpus 16 --max-ram 64 --max-gpu 24
  %(prog)s ./training_logs --headless --csv-interval 5.0 --csv-dir ./stats
        """
    )

    parser.add_argument(
        'folder_path',
        type=str,
        help='Path to folder containing training JSON files'
    )

    parser.add_argument(
        '-i', '--interval',
        type=float,
        default=2.0,
        help='Refresh interval in seconds (default: 2.0)'
    )

    parser.add_argument(
        '--max-cpus',
        type=int,
        default=None,
        help='Maximum CPU cores allocated (default: all available)'
    )

    parser.add_argument(
        '--max-ram',
        type=float,
        default=None,
        help='Maximum RAM in GB allocated (default: total system RAM)'
    )

    parser.add_argument(
        '--max-gpu',
        type=float,
        default=None,
        help='Maximum GPU memory in GB allocated (default: total GPU memory)'
    )

    parser.add_argument(
        '--csv-interval',
        type=float,
        default=None,
        help='CSV output interval in seconds (default: disabled)'
    )

    parser.add_argument(
        '--csv-dir',
        type=str,
        default='./train/statistics/data',
        help='Directory for CSV output (default: ./train/statistics/data)'
    )

    parser.add_argument(
        '--headless',
        action='store_true',
        help='Run without UI, only collect data to CSV (useful for servers)'
    )

    args = parser.parse_args()

    folder_path = Path(args.folder_path)
    csv_output_dir = Path(args.csv_dir) if args.csv_interval else None

    if not folder_path.exists():
        print(f"Error: Folder '{folder_path}' does not exist")
        sys.exit(1)

    if not folder_path.is_dir():
        print(f"Error: '{folder_path}' is not a directory")
        sys.exit(1)

    if args.headless:
        run_headless(folder_path, args.interval, args.max_cpus, args.max_ram, args.max_gpu, args.csv_interval, csv_output_dir)
    else:
        curses.wrapper(scan_folder_curses, folder_path, args.interval, args.max_cpus, args.max_ram, args.max_gpu, args.csv_interval, csv_output_dir)


if __name__ == "__main__":
    main()
