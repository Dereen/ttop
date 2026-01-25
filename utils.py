#!/usr/bin/env python3
"""
Utility functions for ttop.
"""

import re
from pathlib import Path
from typing import List
from dataclasses import dataclass


def natural_sort_key(path: Path) -> List:
    """
    Generate a natural sort key for a file path.
    Converts numbers in the filename to integers for proper numeric sorting.
    Example: file_1, file_2, file_10 instead of file_1, file_10, file_2
    """
    def convert(text):
        return int(text) if text.isdigit() else text.lower()

    return [convert(c) for c in re.split(r'(\d+)', path.name)]


@dataclass
class MemoryInfo:
    """Memory information with unit conversions."""
    bytes_value: int

    @property
    def mb(self) -> float:
        """Memory in megabytes."""
        return self.bytes_value / (1024 * 1024)

    @property
    def gb(self) -> float:
        """Memory in gigabytes."""
        return self.bytes_value / (1024 ** 3)


def mib_to_gb(mib_value: int) -> float:
    """Convert MiB to GB."""
    return mib_value / 1024
