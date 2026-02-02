"""
Generic CSV writer for storing matrices with column names.

Provides flexible CSV output functionality for storing data matrices with
column headers. Handles automatic file creation, column management, and
appending data rows.
"""

import csv
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional


class MatrixCSVWriter:
    """Generic CSV writer for storing matrices with column names.

    Supports:
    - Creating CSV files with custom column headers
    - Appending data rows to existing files
    - Automatic header writing on first write
    - File renaming with start/end datetime stamps
    """

    def __init__(self, output_dir: Path, file_prefix: str = "data"):
        """Initialize CSV writer.

        Args:
            output_dir: Directory where CSV file will be created
            file_prefix: Prefix for the CSV filename (datetime will be appended)
        """
        self.output_dir = Path(output_dir)
        self.file_prefix = file_prefix
        self.csv_file_path = None
        self.start_datetime = datetime.now().strftime('%Y%m%d_%H%M%S')
        self._initialize_file()

    def _initialize_file(self):
        """Initialize CSV file path."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.csv_file_path = self.output_dir / f"{self.start_datetime}.csv"

    def write_row(self, column_names: List[str], data_row: List[str]):
        """Write a single row to CSV file.

        Automatically writes header on first call with given column names.

        Args:
            column_names: List of column header names
            data_row: List of data values (must match column_names length)

        Raises:
            ValueError: If data_row length doesn't match column_names length
        """
        if len(data_row) != len(column_names):
            raise ValueError(
                f"Data row length ({len(data_row)}) doesn't match "
                f"column names length ({len(column_names)})"
            )

        try:
            file_exists = self.csv_file_path.exists()

            with open(self.csv_file_path, 'a', newline='') as f:
                writer = csv.writer(f)

                # Write header if file is new
                if not file_exists:
                    writer.writerow(column_names)

                # Write data row
                writer.writerow(data_row)

        except Exception as e:
            import sys
            print(f"CSV write error: {e}", file=sys.stderr)

    def write_rows(self, column_names: List[str], data_rows: List[List[str]]):
        """Write multiple rows to CSV file.

        Automatically writes header on first call with given column names.

        Args:
            column_names: List of column header names
            data_rows: List of data rows (each row values must match column_names length)

        Raises:
            ValueError: If any data row length doesn't match column_names length
        """
        if not data_rows:
            return

        # Validate all rows
        for i, row in enumerate(data_rows):
            if len(row) != len(column_names):
                raise ValueError(
                    f"Data row {i} length ({len(row)}) doesn't match "
                    f"column names length ({len(column_names)})"
                )

        try:
            file_exists = self.csv_file_path.exists()

            with open(self.csv_file_path, 'a', newline='') as f:
                writer = csv.writer(f)

                # Write header if file is new
                if not file_exists:
                    writer.writerow(column_names)

                # Write all data rows
                for row in data_rows:
                    writer.writerow(row)

        except Exception as e:
            import sys
            print(f"CSV write error: {e}", file=sys.stderr)

    def write_with_prefix_columns(self, column_names: List[str], data_row: List[str],
                                  prefix_columns: Dict[str, str]):
        """Write a row with additional prefix columns prepended.

        Args:
            column_names: List of column header names for main data
            data_row: List of data values for main data
            prefix_columns: Dict of {column_name: value} for columns to prepend

        Raises:
            ValueError: If data_row length doesn't match column_names length
        """
        if len(data_row) != len(column_names):
            raise ValueError(
                f"Data row length ({len(data_row)}) doesn't match "
                f"column names length ({len(column_names)})"
            )

        # Prepare full header and row with prefix
        prefix_names = list(prefix_columns.keys())
        prefix_values = list(prefix_columns.values())
        full_column_names = prefix_names + column_names
        full_data_row = prefix_values + data_row

        try:
            file_exists = self.csv_file_path.exists()

            with open(self.csv_file_path, 'a', newline='') as f:
                writer = csv.writer(f)

                # Write header if file is new
                if not file_exists:
                    writer.writerow(full_column_names)

                # Write data row
                writer.writerow(full_data_row)

        except Exception as e:
            import sys
            print(f"CSV write error: {e}", file=sys.stderr)

    def get_file_path(self) -> Path:
        """Get the current CSV file path."""
        return self.csv_file_path

    def finalize(self):
        """Finalize the CSV file with end datetime.

        Renames the CSV file to include both start and end datetime.
        """
        if self.csv_file_path and self.csv_file_path.exists():
            end_datetime = datetime.now().strftime('%Y%m%d_%H%M%S')
            new_path = self.csv_file_path.parent / f"{self.start_datetime}_{end_datetime}.csv"
            try:
                self.csv_file_path.rename(new_path)
                self.csv_file_path = new_path
            except Exception:
                pass
