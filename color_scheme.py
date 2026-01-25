"""Color scheme management for ttop."""

import curses


class ColorScheme:
    """Centralized color management for ttop."""

    # Color pair indices
    HEADER = 1
    FIELD_NAME = 2
    VALUE = 3
    NORMAL = 4
    ERROR = 5
    STATUS = 6
    SEPARATOR = 7

    # Phase colors (normal)
    LOADING = 10
    BATCH = 11
    VALIDATION = 12
    FINISHED = 13
    FAILED = 14
    CANCELED = 15
    KILLED = 16
    UNKNOWN = 17

    # Cursor highlight colors
    LOADING_CURSOR = 20
    BATCH_CURSOR = 21
    VALIDATION_CURSOR = 22
    FINISHED_CURSOR = 23
    FAILED_CURSOR = 24
    CANCELED_CURSOR = 25
    KILLED_CURSOR = 26
    UNKNOWN_CURSOR = 27

    # Process/Thread colors
    PROCESS = 30
    THREAD = 31
    PROCESS_CURSOR = 40
    THREAD_CURSOR = 41

    @staticmethod
    def init_colors():
        """Initialize all color pairs."""
        curses.start_color()
        curses.use_default_colors()

        # Basic UI colors
        curses.init_pair(ColorScheme.HEADER, curses.COLOR_CYAN, -1)
        curses.init_pair(ColorScheme.FIELD_NAME, curses.COLOR_GREEN, -1)
        curses.init_pair(ColorScheme.VALUE, curses.COLOR_YELLOW, -1)
        curses.init_pair(ColorScheme.NORMAL, curses.COLOR_WHITE, -1)
        curses.init_pair(ColorScheme.ERROR, curses.COLOR_RED, -1)
        curses.init_pair(ColorScheme.STATUS, curses.COLOR_MAGENTA, -1)
        curses.init_pair(ColorScheme.SEPARATOR, curses.COLOR_BLUE, -1)

        # Phase-based colors
        curses.init_pair(ColorScheme.LOADING, curses.COLOR_YELLOW, curses.COLOR_BLACK)
        curses.init_pair(ColorScheme.BATCH, curses.COLOR_YELLOW, curses.COLOR_BLACK)
        curses.init_pair(ColorScheme.VALIDATION, curses.COLOR_GREEN, curses.COLOR_BLACK)
        curses.init_pair(ColorScheme.FINISHED, curses.COLOR_BLUE, curses.COLOR_BLACK)
        curses.init_pair(ColorScheme.FAILED, curses.COLOR_RED, curses.COLOR_BLACK)
        curses.init_pair(ColorScheme.CANCELED, curses.COLOR_WHITE, curses.COLOR_BLACK)
        curses.init_pair(ColorScheme.KILLED, curses.COLOR_RED, curses.COLOR_BLACK)
        curses.init_pair(ColorScheme.UNKNOWN, curses.COLOR_MAGENTA, curses.COLOR_BLACK)

        # Cursor highlight colors
        curses.init_pair(ColorScheme.LOADING_CURSOR, curses.COLOR_BLACK, curses.COLOR_YELLOW)
        curses.init_pair(ColorScheme.BATCH_CURSOR, curses.COLOR_BLACK, curses.COLOR_YELLOW)
        curses.init_pair(ColorScheme.VALIDATION_CURSOR, curses.COLOR_BLACK, curses.COLOR_GREEN)
        curses.init_pair(ColorScheme.FINISHED_CURSOR, curses.COLOR_BLACK, curses.COLOR_BLUE)
        curses.init_pair(ColorScheme.FAILED_CURSOR, curses.COLOR_BLACK, curses.COLOR_RED)
        curses.init_pair(ColorScheme.CANCELED_CURSOR, curses.COLOR_BLACK, curses.COLOR_WHITE)
        curses.init_pair(ColorScheme.KILLED_CURSOR, curses.COLOR_BLACK, curses.COLOR_RED)
        curses.init_pair(ColorScheme.UNKNOWN_CURSOR, curses.COLOR_BLACK, curses.COLOR_MAGENTA)

        # Process/Thread colors
        curses.init_pair(ColorScheme.PROCESS, curses.COLOR_CYAN, curses.COLOR_BLACK)
        curses.init_pair(ColorScheme.THREAD, curses.COLOR_MAGENTA, curses.COLOR_BLACK)
        curses.init_pair(ColorScheme.PROCESS_CURSOR, curses.COLOR_BLACK, curses.COLOR_CYAN)
        curses.init_pair(ColorScheme.THREAD_CURSOR, curses.COLOR_BLACK, curses.COLOR_MAGENTA)

    @staticmethod
    def get_phase_color(phase: str, status: str, cursor: bool = False) -> int:
        """Get color pair for a phase/status combination.

        Args:
            phase: Training phase
            status: Training status
            cursor: Whether this is for cursor highlighting

        Returns:
            Color pair index
        """
        status_upper = status.upper() if status else ""
        if status_upper == "FINISHED":
            base = ColorScheme.FINISHED
        elif status_upper == "FAILED":
            base = ColorScheme.FAILED
        elif status_upper == "CANCELED":
            base = ColorScheme.CANCELED
        elif status_upper == "KILLED":
            base = ColorScheme.KILLED
        else:
            phase_lower = phase.lower() if phase else ""
            if "loading" in phase_lower:
                base = ColorScheme.LOADING
            elif "batch" in phase_lower or "training" in phase_lower:
                base = ColorScheme.BATCH
            elif "validation" in phase_lower or "validating" in phase_lower:
                base = ColorScheme.VALIDATION
            else:
                base = ColorScheme.UNKNOWN

        if cursor:
            return base + 10  # Cursor colors are +10 from base
        return base
