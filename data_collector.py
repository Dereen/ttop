#!/usr/bin/env python3
"""
Data collection and extraction logic for training monitor files.
Handles folder iteration and JSON parsing.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

from rci.ttop.utils import natural_sort_key


def read_training_file(file_path: Path) -> Optional[Dict]:
    """Read and parse a training monitor JSON file.

    Returns:
        Dictionary containing all data from the JSON file, or None if read fails
    """
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return None


def collect_training_data(folder_path: Path) -> List[Dict]:
    """Collect all training data from JSON files in folder.

    Args:
        folder_path: Path to folder containing JSON files

    Returns:
        List of dictionaries, each containing:
            - 'file_path': Path object of the JSON file
            - 'file_name': Name of the file
            - 'data': Complete dictionary of all data from the JSON file
    """
    json_files = sorted(folder_path.glob("*.json"), key=natural_sort_key)
    training_data = []

    for json_file in json_files:
        data = read_training_file(json_file)
        if data:
            training_data.append({
                'file_path': json_file,
                'file_name': json_file.name,
                'data': data
            })

    return training_data