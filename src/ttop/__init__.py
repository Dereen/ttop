"""
ttop - Training Top

A curses-based monitor for ML training progress.
Displays real-time training statistics, resource usage, and allows
editing of tracked variables during training.
"""

__version__ = "0.1.0"

from ttop.main import main
from ttop.training_monitor import TrainingMonitor
from ttop.async_file_writer import AsyncFileWriter

__all__ = ["main", "__version__", "TrainingMonitor", "AsyncFileWriter"]
