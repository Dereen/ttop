import portalocker
import json
import threading
import time
from pathlib import Path
from typing import Any, Optional


class AsyncFileWriter:
    """Asynchronous file writer that writes data at fixed intervals (100ms).

    Data updates are buffered and written to file every 100ms regardless of
    how frequently write() is called.
    """

    def __init__(self, output_file: str, write_interval: float = 0.1):
        """Initialize AsyncFileWriter.

        Args:
            output_file: Path to output JSON file
            write_interval: Time in seconds between writes (default 100ms)
        """
        self.output_file = Path(output_file)
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

        # Delete existing file to avoid locking issues
        if self.output_file.exists():
            try:
                self.output_file.unlink()
            except Exception as e:
                print(f"Warning: Failed to delete existing file {self.output_file}: {e}")

        self.write_interval = write_interval
        self.pending_data: Optional[Any] = None
        self.data_lock = threading.Lock()
        self.running = True
        self.has_pending = False
        self.is_shutdown = False

        # Start background write thread
        self.write_thread = threading.Thread(target=self._write_loop, daemon=True)
        self.write_thread.start()

    def write(self, data: Any):
        """Buffer data for writing.

        The data will be written to file within the next write_interval seconds.
        Multiple calls to write() between flushes will result in only the latest
        data being written.

        Args:
            data: Data to write to file
        """
        # If already shutdown, write immediately to avoid losing final data
        if self.is_shutdown:
            try:
                self._write_safe(data)
            except Exception as e:
                # Suppress "No locks available" errors during shutdown/interrupt
                if "No locks available" not in str(e):
                    print(f"Warning: Failed to write after shutdown: {e}")
            return

        with self.data_lock:
            self.pending_data = data
            self.has_pending = True

    def _write_loop(self):
        """Background thread loop that periodically writes buffered data."""
        while self.running:
            time.sleep(self.write_interval)

            # Check if there's data to write
            with self.data_lock:
                if self.has_pending and self.pending_data is not None:
                    data_to_write = self.pending_data
                    self.has_pending = False
                else:
                    continue

            # Write outside lock to avoid blocking other threads
            try:
                self._write_safe(data_to_write)
            except Exception as e:
                # Suppress "No locks available" errors during shutdown/interrupt
                if "No locks available" not in str(e):
                    print(f"Warning: Failed to write to {self.output_file}: {e}")

    def _write_safe(self, data: Any):
        """Write data to file with proper locking.

        Args:
            data: Data to write to file
        """
        with open(self.output_file, "w") as f:
            portalocker.lock(f, portalocker.LOCK_EX)
            try:
                json.dump(data, f, indent=2)  # type: ignore[arg-type]
                f.flush()
            finally:
                portalocker.unlock(f)

    def read(self) -> Any:
        """Read data from file with proper locking.

        Returns:
            Data read from file
        """
        with open(self.output_file, "r") as f:
            portalocker.lock(f, portalocker.LOCK_SH)
            try:
                data = json.load(f)
            finally:
                portalocker.unlock(f)
            return data

    def shutdown(self):
        """Shutdown the writer and flush any pending data."""
        # Mark as shutdown first to prevent new writes
        self.is_shutdown = True

        # Stop the background thread
        self.running = False

        # Wait for thread to finish with timeout
        self.write_thread.join(timeout=1.0)

        # Write any remaining pending data
        with self.data_lock:
            if self.has_pending and self.pending_data is not None:
                try:
                    self._write_safe(self.pending_data)
                except Exception as e:
                    # Suppress "No locks available" errors during shutdown/interrupt
                    if "No locks available" not in str(e):
                        print(f"Warning: Failed to write final data: {e}")
