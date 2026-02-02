"""
Output file viewer for training monitor.
Shows stdout and stderr with timestamps and color coding.
"""

import re
import time
import curses
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import deque

from ttop.color_scheme import ColorScheme


ANSI_COLOR_PATTERN = re.compile(r'\033\[([0-9;]+)m')

ANSI_TO_CURSES_MAP = {
    '30': curses.COLOR_BLACK,
    '31': curses.COLOR_RED,
    '32': curses.COLOR_GREEN,
    '33': curses.COLOR_YELLOW,
    '34': curses.COLOR_BLUE,
    '35': curses.COLOR_MAGENTA,
    '36': curses.COLOR_CYAN,
    '37': curses.COLOR_WHITE,
    '90': curses.COLOR_BLACK,
    '91': curses.COLOR_RED,
    '92': curses.COLOR_GREEN,
    '93': curses.COLOR_YELLOW,
    '94': curses.COLOR_BLUE,
    '95': curses.COLOR_MAGENTA,
    '96': curses.COLOR_CYAN,
    '97': curses.COLOR_WHITE,
}

ANSI_COLOR_PAIRS_CACHE = {}


def get_ansi_color_pair(fg_code: Optional[str], bold: bool = False) -> int:
    """Get or create a curses color pair for ANSI color code.

    Args:
        fg_code: ANSI foreground color code (e.g., '31' for red)
        bold: Whether the text should be bold

    Returns:
        Curses color pair index
    """
    if fg_code is None or fg_code not in ANSI_TO_CURSES_MAP:
        return ColorScheme.NORMAL

    cache_key = (fg_code, bold)
    if cache_key in ANSI_COLOR_PAIRS_CACHE:
        return ANSI_COLOR_PAIRS_CACHE[cache_key]

    fg_color = ANSI_TO_CURSES_MAP[fg_code]

    pair_idx = 100 + len(ANSI_COLOR_PAIRS_CACHE)
    if pair_idx < curses.COLOR_PAIRS:
        try:
            curses.init_pair(pair_idx, fg_color, -1)
            ANSI_COLOR_PAIRS_CACHE[cache_key] = pair_idx
            return pair_idx
        except:
            return ColorScheme.NORMAL

    return ColorScheme.NORMAL


def parse_ansi_text(text: str) -> List[Tuple[str, Optional[str], bool]]:
    """Parse text with ANSI color codes into segments.

    Args:
        text: Text with ANSI escape sequences

    Returns:
        List of (text_segment, color_code, is_bold) tuples
    """
    segments = []
    current_color = None
    current_bold = False
    last_end = 0

    for match in ANSI_COLOR_PATTERN.finditer(text):
        if match.start() > last_end:
            segments.append((text[last_end:match.start()], current_color, current_bold))

        codes = match.group(1).split(';')
        for code in codes:
            if code == '0':
                current_color = None
                current_bold = False
            elif code == '1':
                current_bold = True
            elif code in ANSI_TO_CURSES_MAP:
                current_color = code

        last_end = match.end()

    if last_end < len(text):
        segments.append((text[last_end:], current_color, current_bold))

    return segments


def pad_line(text: str, width: int) -> str:
    """Pad line with spaces to fill terminal width.

    Args:
        text: The text to pad
        width: Terminal width

    Returns:
        Padded string
    """
    if len(text) >= width:
        return text[:width]
    return text + ' ' * (width - len(text))


def print_colored_line(stdscr, row: int, col: int, text: str, color_pair: int = ColorScheme.NORMAL):
    """Print a line with color, handling errors gracefully."""
    try:
        stdscr.addstr(row, col, text, curses.color_pair(color_pair))
    except curses.error:
        pass


def print_ansi_colored_line(stdscr, row: int, col: int, text: str, width: int, default_color: int = ColorScheme.NORMAL):
    """Print a line with ANSI color support.

    Args:
        stdscr: Curses screen object
        row: Row position
        col: Column position
        text: Text with ANSI escape sequences
        width: Terminal width for padding
        default_color: Default color pair for non-colored text
    """
    segments = parse_ansi_text(text)
    current_col = col
    remaining_width = width

    for segment_text, color_code, is_bold in segments:
        if remaining_width <= 0:
            break

        display_text = segment_text[:remaining_width]
        color_pair = get_ansi_color_pair(color_code, is_bold) if color_code else default_color

        try:
            attrs = curses.color_pair(color_pair)
            if is_bold:
                attrs |= curses.A_BOLD
            stdscr.addstr(row, current_col, display_text, attrs)
        except curses.error:
            pass

        current_col += len(display_text)
        remaining_width -= len(display_text)

    if remaining_width > 0:
        try:
            stdscr.addstr(row, current_col, ' ' * remaining_width, curses.color_pair(default_color))
        except curses.error:
            pass




def tail_both_files(stdout_path: Path, stderr_path: Path, max_lines: int = 1000) -> deque:
    """Read and merge stdout and stderr files with timestamps.

    Args:
        stdout_path: Path to stdout file
        stderr_path: Path to stderr file
        max_lines: Maximum number of lines to keep

    Returns:
        Deque containing tuples of (timestamp, line_text, is_stderr)
    """
    all_lines = []

    # Check if both paths point to the same file
    if stdout_path and stderr_path:
        try:
            if stdout_path.resolve() == stderr_path.resolve():
                stderr_path = None
        except (OSError, PermissionError):
            pass

    # Read stdout
    if stdout_path and stdout_path.exists() and stdout_path.is_file():
        try:
            stat_info = stdout_path.stat()
            file_mtime = stat_info.st_mtime

            with open(stdout_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
                lines_list = content.split('\n')
                total_lines = len(lines_list)

                for idx, line in enumerate(lines_list):
                    if '\r' in line:
                        line = line.split('\r')[-1]

                    line = line.replace('\0', '').rstrip()
                    if not line and idx == total_lines - 1:
                        continue

                    if total_lines > 1:
                        line_ratio = idx / (total_lines - 1)
                    else:
                        line_ratio = 1.0

                    estimated_time = file_mtime - (1 - line_ratio) * 3600
                    timestamp = datetime.fromtimestamp(estimated_time)

                    all_lines.append((timestamp, line, False))
        except Exception as e:
            all_lines.append((datetime.now(), f"Error reading stdout: {e}", True))

    # Read stderr
    if stderr_path and stderr_path.exists() and stderr_path.is_file():
        try:
            stat_info = stderr_path.stat()
            file_mtime = stat_info.st_mtime

            with open(stderr_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
                lines_list = content.split('\n')
                total_lines = len(lines_list)

                for idx, line in enumerate(lines_list):
                    if '\r' in line:
                        line = line.split('\r')[-1]

                    line = line.replace('\0', '').rstrip()
                    if not line and idx == total_lines - 1:
                        continue

                    if total_lines > 1:
                        line_ratio = idx / (total_lines - 1)
                    else:
                        line_ratio = 1.0

                    estimated_time = file_mtime - (1 - line_ratio) * 3600
                    timestamp = datetime.fromtimestamp(estimated_time)

                    all_lines.append((timestamp, line, True))
        except Exception as e:
            all_lines.append((datetime.now(), f"Error reading stderr: {e}", True))

    # Sort by timestamp to interleave stdout and stderr chronologically
    all_lines.sort(key=lambda x: x[0])

    # Keep only last max_lines
    lines = deque(all_lines[-max_lines:] if len(all_lines) > max_lines else all_lines, maxlen=max_lines)
    return lines


def get_output_files_from_metadata(metadata: dict) -> Tuple[Optional[Path], Optional[Path]]:
    """Get stdout and stderr file paths from training metadata.

    Args:
        metadata: Metadata dictionary from training file

    Returns:
        Tuple of (stdout_path, stderr_path) - only regular files, not terminals
    """
    stdout_file = None
    stderr_file = None

    pid = metadata.get("pid")

    # Try to get stdout
    stdout_path = metadata.get("stdout_path")
    if stdout_path:
        stdout_file = Path(stdout_path)
        # Only accept regular files
        if not (stdout_file.exists() and stdout_file.is_file()):
            stdout_file = None

            # Try /proc/<pid>/fd/1 to see if it resolves to a real file
            if pid:
                proc_fd_path = Path(f"/proc/{pid}/fd/1")
                try:
                    if proc_fd_path.exists():
                        resolved = proc_fd_path.resolve()
                        if resolved.is_file():
                            stdout_file = resolved
                except (OSError, PermissionError):
                    pass

    # Try to get stderr
    stderr_path = metadata.get("stderr_path")
    if stderr_path:
        stderr_file = Path(stderr_path)
        # Only accept regular files
        if not (stderr_file.exists() and stderr_file.is_file()):
            stderr_file = None

            # Try /proc/<pid>/fd/2 to see if it resolves to a real file
            if pid:
                proc_fd_path = Path(f"/proc/{pid}/fd/2")
                try:
                    if proc_fd_path.exists():
                        resolved = proc_fd_path.resolve()
                        if resolved.is_file():
                            stderr_file = resolved
                except (OSError, PermissionError):
                    pass

    return stdout_file, stderr_file


def get_output_files_from_pid(pid: int) -> Tuple[Optional[Path], Optional[Path]]:
    """Get stdout and stderr file paths using PID and /proc filesystem.

    DEPRECATED: Use get_output_files_from_metadata instead.
    This function is kept for backwards compatibility.

    Args:
        pid: Process ID

    Returns:
        Tuple of (stdout_path, stderr_path) or (None, None) if not found
    """
    stdout_file = None
    stderr_file = None

    # Use /proc/<PID>/fd/1 to find stdout file
    try:
        fd_stdout = Path(f"/proc/{pid}/fd/1")
        if fd_stdout.exists():
            stdout_file = fd_stdout.resolve()
            if not (stdout_file.exists() and stdout_file.is_file()):
                stdout_file = None
    except (OSError, PermissionError):
        pass

    # Use /proc/<PID>/fd/2 to find stderr file
    try:
        fd_stderr = Path(f"/proc/{pid}/fd/2")
        if fd_stderr.exists():
            stderr_file = fd_stderr.resolve()
            if not (stderr_file.exists() and stderr_file.is_file()):
                stderr_file = None
    except (OSError, PermissionError):
        pass

    return stdout_file, stderr_file


def show_output_window(stdscr, stdout_file: Optional[Path], stderr_file: Optional[Path], metadata: Optional[dict] = None):
    """Show output file in a separate window with tail functionality.

    Args:
        stdscr: Curses screen object
        stdout_file: Path to stdout file
        stderr_file: Path to stderr file
        metadata: Optional metadata dict to show paths that were rejected
    """
    if not stdout_file and not stderr_file:
        # Show error message
        stdscr.clear()
        term_width = curses.COLS
        row = 0
        print_colored_line(stdscr, row, 0, pad_line("=" * 200, term_width), ColorScheme.SEPARATOR); row += 1
        print_colored_line(stdscr, row, 0, pad_line("Output File Viewer", term_width), ColorScheme.HEADER); row += 1
        print_colored_line(stdscr, row, 0, pad_line("=" * 200, term_width), ColorScheme.SEPARATOR); row += 1
        print_colored_line(stdscr, row, 0, pad_line("No readable output files found for selected training.", term_width), ColorScheme.ERROR); row += 1
        row += 1

        # Show what paths were in metadata (if available)
        if metadata:
            stdout_path = metadata.get("stdout_path")
            stderr_path = metadata.get("stderr_path")
            if stdout_path or stderr_path:
                print_colored_line(stdscr, row, 0, pad_line("Found paths (not regular files):", term_width), ColorScheme.NORMAL); row += 1
                if stdout_path:
                    print_colored_line(stdscr, row, 0, pad_line(f"  stdout: {stdout_path}", term_width), ColorScheme.VALUE); row += 1
                if stderr_path:
                    print_colored_line(stdscr, row, 0, pad_line(f"  stderr: {stderr_path}", term_width), ColorScheme.VALUE); row += 1
                row += 1
                print_colored_line(stdscr, row, 0, pad_line("Tip: Redirect training output to log files:", term_width), ColorScheme.NORMAL); row += 1
                print_colored_line(stdscr, row, 0, pad_line("  python script.py > stdout.log 2> stderr.log", term_width), ColorScheme.VALUE); row += 1

        row += 1
        print_colored_line(stdscr, row, 0, pad_line("Press ESC to return", term_width), ColorScheme.NORMAL)
        stdscr.refresh()

        # Wait for ESC
        while True:
            key = stdscr.getch()
            if key == 27:  # ESC
                break
            time.sleep(0.01)
        return

    # Initialize scroll position
    vertical_scroll = 0
    auto_scroll = True  # Auto-scroll to bottom
    last_line_count = 0

    while True:
        # Read file content from both stdout and stderr
        lines = tail_both_files(stdout_file, stderr_file)

        # Auto-scroll to bottom if new lines appeared
        if auto_scroll and len(lines) > last_line_count:
            vertical_scroll = max(0, len(lines) - (curses.LINES - 6))
        last_line_count = len(lines)

        stdscr.clear()
        term_width = curses.COLS

        row = 0
        header_height = 8

        # Header
        print_colored_line(stdscr, row, 0, pad_line("=" * 200, term_width), ColorScheme.SEPARATOR); row += 1
        print_colored_line(stdscr, row, 0, pad_line(f"Output File Viewer", term_width), ColorScheme.HEADER); row += 1

        # Show file paths
        stdout_str = f"stdout: {stdout_file}" if stdout_file else "stdout: (none)"
        stderr_str = f"stderr: {stderr_file}" if stderr_file else "stderr: (none)"
        print_colored_line(stdscr, row, 0, pad_line(stdout_str, term_width), ColorScheme.VALUE); row += 1
        print_colored_line(stdscr, row, 0, pad_line(stderr_str, term_width), ColorScheme.VALUE); row += 1

        auto_scroll_indicator = "[AUTO-SCROLL ON]" if auto_scroll else "[AUTO-SCROLL OFF]"
        print_colored_line(stdscr, row, 0, pad_line(f"'↑↓' scroll | 'a' toggle auto-scroll {auto_scroll_indicator} | ESC return", term_width), ColorScheme.STATUS); row += 1

        print_colored_line(stdscr, row, 0, pad_line(f"Total lines: {len(lines)}", term_width), ColorScheme.VALUE); row += 1
        print_colored_line(stdscr, row, 0, pad_line("=" * 200, term_width), ColorScheme.SEPARATOR); row += 1

        # Calculate visible area
        max_visible_rows = curses.LINES - header_height - 2

        # Clamp scroll position
        max_scroll = max(0, len(lines) - max_visible_rows)
        if vertical_scroll > max_scroll:
            vertical_scroll = max_scroll
        if vertical_scroll < 0:
            vertical_scroll = 0

        # Display lines
        if not lines:
            print_colored_line(stdscr, row, 0, pad_line("(empty file)", term_width), ColorScheme.NORMAL)
            row += 1
        else:
            visible_lines = list(lines)[vertical_scroll:vertical_scroll + max_visible_rows]
            for timestamp, line_text, is_stderr in visible_lines:
                # Format timestamp
                time_str = timestamp.strftime("%H:%M:%S")
                # Prefix with timestamp
                prefix = f"[{time_str}] "
                full_line = prefix + line_text

                # Use red color for stderr lines, normal for stdout
                default_color = ColorScheme.ERROR if is_stderr else ColorScheme.NORMAL

                # Print with ANSI color support
                print_ansi_colored_line(stdscr, row, 0, full_line, term_width, default_color)
                row += 1

        # Footer with scroll info
        if len(lines) > max_visible_rows:
            scroll_info = f"Lines {vertical_scroll + 1}-{min(vertical_scroll + max_visible_rows, len(lines))} of {len(lines)}"
        else:
            scroll_info = f"{len(lines)} lines"

        footer_row = curses.LINES - 1
        print_colored_line(stdscr, footer_row, 0, pad_line(f"[{scroll_info}]", term_width), ColorScheme.VALUE)

        stdscr.refresh()

        # Handle input
        key = stdscr.getch()
        if key == 27:  # ESC
            break
        elif key == ord('a') or key == ord('A'):
            auto_scroll = not auto_scroll
            if auto_scroll:
                vertical_scroll = max(0, len(lines) - max_visible_rows)
        elif key == curses.KEY_UP:
            if vertical_scroll > 0:
                vertical_scroll -= 1
                auto_scroll = False
        elif key == curses.KEY_DOWN:
            if vertical_scroll < max_scroll:
                vertical_scroll += 1
        elif key == curses.KEY_PPAGE:  # Page Up
            vertical_scroll = max(0, vertical_scroll - max_visible_rows)
            auto_scroll = False
        elif key == curses.KEY_NPAGE:  # Page Down
            vertical_scroll = min(max_scroll, vertical_scroll + max_visible_rows)
        elif key == curses.KEY_HOME:
            vertical_scroll = 0
            auto_scroll = False
        elif key == curses.KEY_END:
            vertical_scroll = max_scroll
            auto_scroll = True

        time.sleep(0.05)
