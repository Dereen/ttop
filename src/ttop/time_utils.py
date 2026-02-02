"""
Time duration formatting utilities.

Provides compact, space-efficient time formatting for converting seconds to
human-readable representations with appropriate time units and precision.
"""

from typing import Optional


def format_duration(seconds: Optional[float], default: str = "Invalid") -> str:
    """Format seconds into compact human-readable duration string.

    Converts seconds into compact notation using months, days, hours, minutes, and seconds.
    Uses abbreviated units (mo/d/h/m/s) for terminal and log display.

    Args:
        seconds: Number of seconds to format, or None
        default: String to return if seconds is None (default: "Invalid")

    Returns:
        Formatted string like "1mo2d", "2d5h", "1h30m", "45m20s", "30s"

    Examples:
        >>> format_duration(0)
        '0s'
        >>> format_duration(45)
        '45s'
        >>> format_duration(125)
        '2m5s'
        >>> format_duration(3665)
        '1h1m'
        >>> format_duration(86400)
        '1d0h'
        >>> format_duration(2592000)
        '1mo0d'
        >>> format_duration(None)
        'Invalid'
        >>> format_duration(None, "N/A")
        'N/A'
    """
    if seconds is None:
        return default

    # Define time units in seconds
    MONTH = 30 * 24 * 3600  # Approximate month as 30 days
    DAY = 24 * 3600
    HOUR = 3600
    MINUTE = 60

    months = int(seconds // MONTH)
    remaining = seconds % MONTH

    days = int(remaining // DAY)
    remaining = remaining % DAY

    hours = int(remaining // HOUR)
    remaining = remaining % HOUR

    minutes = int(remaining // MINUTE)
    secs = int(remaining % MINUTE)

    if months > 0:
        return f"{months}mo{days}d"
    elif days > 0:
        return f"{days}d{hours}h"
    elif hours > 0:
        return f"{hours}h{minutes}m"
    elif minutes > 0:
        return f"{minutes}m{secs}s"
    else:
        return f"{secs}s"


def format_short_duration(seconds: Optional[float], default: str = "N/A") -> str:
    """Format seconds with automatic unit selection (s/ms/μs).

    Converts time duration to the most appropriate unit: seconds, milliseconds,
    or microseconds. Useful for performance metrics and iteration times.

    Args:
        seconds: Time duration in seconds, or None/invalid
        default: String to return if seconds is invalid (default: "N/A")

    Returns:
        Formatted string with appropriate unit like "1.50s", "250.00ms", "500.00μs"

    Examples:
        >>> format_short_duration(1.5)
        '1.50s'
        >>> format_short_duration(0.25)
        '250.00ms'
        >>> format_short_duration(0.0005)
        '500.00μs'
        >>> format_short_duration(None)
        'N/A'
        >>> format_short_duration(0)
        'N/A'
    """
    if seconds is None or seconds <= 0:
        return default

    try:
        if seconds >= 1.0:
            return f"{seconds:.2f}s"
        elif seconds >= 0.001:
            return f"{seconds * 1000:.2f}ms"
        else:
            return f"{seconds * 1000000:.2f}μs"
    except (ValueError, TypeError):
        return default
