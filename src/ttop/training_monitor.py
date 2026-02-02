import time
import os
import sys
import socket
import signal
import traceback
import json
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Callable, List
from .async_file_writer import AsyncFileWriter



class TrainingMonitor:
    def __init__(self, output_file: str, wandb_run_name: Optional[str] = None):
        self.wandb_run_name = wandb_run_name

        # Create folder structure: output_file is now a folder path
        # Extract folder from output_file path (remove .json extension if present)
        output_path = Path(output_file)
        if output_path.suffix == '.json':
            output_path = output_path.with_suffix('')

        self.run_folder = output_path
        self.run_folder.mkdir(parents=True, exist_ok=True)

        # Define file paths within the folder
        self.progress_file = self.run_folder / "progress.json"
        self.control_file = self.run_folder / "control.json"

        # Create writer for progress file (metadata merged into progress.json)
        self.progress_writer = AsyncFileWriter(str(self.progress_file))

        # Setup signal log file path
        script_path = Path(sys.argv[0]).resolve()
        script_dir = script_path.parent
        log_dir = script_dir / "log"
        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = wandb_run_name if wandb_run_name else "unknown"
        self.signal_log_file = log_dir / f"{timestamp}_{run_name}_signal.json"

        # Dataset sizes for ratio calculation
        self.train_dataset_size = None
        self.val_dataset_size = None

        # Variable tracking and callbacks
        self.tracked_variables = {}
        self.general_callbacks = []
        self.specific_callbacks = {}

        # Background monitoring thread
        self._monitor_thread = None
        self._monitor_running = False
        self._monitor_lock = threading.Lock()
        self._cached_changes = None
        self._new_data = False

        stdout_path, stderr_path = self._detect_output_files()

        # Static metadata (written once)
        self.metadata = {
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "script_name": sys.argv[0],
            "python_version": sys.version,
            "start_time": time.time(),
            "wandb_run_name": wandb_run_name,
            "stdout_path": stdout_path,
            "stderr_path": stderr_path
        }

        # Dynamic progress data (updated frequently)
        # Include metadata in progress.json for single-file operation
        self.progress_data = {
            "metadata": self.metadata,
            "status": "RUNNING",
            "end_time": None,
            "error_message": None,
            "state": {
                "phase": None,
                "epoch": 0,
                "batch": 0,
                "total_epochs": 0,
                "total_batches": 0
            },
            "timing": {
                "epoch_start": None,
                "validation_start": None,
            },
            "estimates": {
                "run" : {
                    "rate": 0,
                    "elapsed": 0,
                    "remaining": 0,
                    "total": 0,
                },
                "epoch": {
                    "rate": 0,
                    "elapsed": 0,
                    "remaining": 0,
                    "total": 0,
                    "durations": []
                },
                "training": {
                    "rate": 0,
                    "elapsed": 0,
                    "remaining": 0,
                    "total": 0,
                    "durations": []
                },
                "validation": {
                    "rate": 0,
                    "elapsed": 0,
                    "remaining": 0,
                    "total": 0,
                    "durations": []
                },
                "dataset_ratio": None,
            },
            "metrics": {
                "loss": None,
                "learning_rate": None,
                "samples_processed": 0
            }
        }

        # Control data (written by training, read by monitor)
        self.control_data = {}

        # Initialize control.json if it doesn't exist
        if not self.control_file.exists():
            with open(self.control_file, 'w') as f:
                json.dump(self.control_data, f, indent=2)

        self._setup_signal_handlers()
        self.progress_writer.write(self.progress_data)

    def _detect_output_files(self) -> tuple[Optional[str], Optional[str]]:
        """Detect stdout and stderr file paths using /proc filesystem.

        Returns:
            Tuple of (stdout_path, stderr_path) or (None, None) if not found
        """
        pid = os.getpid()
        stdout_path = None
        stderr_path = None

        try:
            fd_stdout = Path(f"/proc/{pid}/fd/1")
            if fd_stdout.exists():
                resolved_stdout = fd_stdout.resolve()
                stdout_path = str(resolved_stdout)
        except (OSError, PermissionError):
            pass

        try:
            fd_stderr = Path(f"/proc/{pid}/fd/2")
            if fd_stderr.exists():
                resolved_stderr = fd_stderr.resolve()
                stderr_path = str(resolved_stderr)
        except (OSError, PermissionError):
            pass

        return stdout_path, stderr_path

    def _setup_signal_handlers(self):
        self._signal_received = False
        signal.signal(signal.SIGINT, self._handle_interrupt)
        signal.signal(signal.SIGTERM, self._handle_interrupt)

    def _handle_interrupt(self, signum, frame):
        import inspect

        # Only handle the first signal, then restore default handlers
        if self._signal_received:
            return
        self._signal_received = True

        # Restore default signal handlers immediately to prevent catching cleanup exceptions
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        signal.signal(signal.SIGTERM, signal.SIG_DFL)

        # Collect signal information
        signal_name = "SIGINT" if signum == signal.SIGINT else "SIGTERM" if signum == signal.SIGTERM else f"Signal {signum}"
        timestamp = time.time()

        signal_info = {
            "signal_number": signum,
            "signal_name": signal_name,
            "pid": os.getpid(),
            "ppid": os.getppid(),
            "timestamp": timestamp,
            "timestamp_readable": datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S.%f'),
            "hostname": socket.gethostname(),
            "script_name": sys.argv[0],
            "run_name": self.wandb_run_name,
            "frame_file": frame.f_code.co_filename if frame else None,
            "frame_function": frame.f_code.co_name if frame else None,
            "frame_line": frame.f_lineno if frame else None,
            "training_phase": self.progress_data["state"]["phase"],
            "epoch": self.progress_data["state"]["epoch"],
            "batch": self.progress_data["state"]["batch"]
        }

        # Log to stderr (minimal output to avoid interfering with cleanup)
        print(f"\n[TrainingMonitor] Received {signal_name}, cleaning up...", file=sys.stderr)

        # Write signal information to dedicated log file
        try:
            with open(self.signal_log_file, 'w') as f:
                json.dump(signal_info, f, indent=2)
        except Exception:
            pass

        try:
            self.set_canceled()
        except Exception:
            pass
        try:
            self.shutdown()
        except Exception:
            pass

        # Raise KeyboardInterrupt to let Python's exception handling work
        raise KeyboardInterrupt()

    def set_dataset_sizes(self, train_size: int, val_size: int):
        """Set dataset sizes for validation estimation.

        Args:
            train_size: Number of samples in training dataset
            val_size: Number of samples in validation dataset
        """
        self.train_dataset_size = train_size
        self.val_dataset_size = val_size
        if train_size > 0:
            ratio = val_size / train_size
            self.progress_data["estimates"]["dataset_ratio"] = ratio
            print(f"[MONITOR] Dataset ratio set: val/train = {val_size}/{train_size} = {ratio:.4f}")
            self.progress_writer.write(self.progress_data)

    def start_phase(self, phase: str):
        self.progress_data["state"]["phase"] = phase.lower()
        self.progress_writer.write(self.progress_data)

    def start_batch(self):
        self.progress_data["state"]["batch"] += 1

    def start_epoch(self, epoch: int, total_batches: int, total_epochs: Optional[int] = None):
        self.progress_data["state"]["epoch"] = epoch
        self.progress_data["state"]["batch"] = 0
        self.progress_data["state"]["total_batches"] = total_batches
        self.progress_data["state"]["total_epochs"] = total_epochs

        self.progress_data["timing"]["epoch_start"] = time.time()
        self.progress_writer.write(self.progress_data)

    def end_epoch(self):
        if self.progress_data["timing"]["epoch_start"] is not None:
            duration = time.time() - self.progress_data["timing"]["epoch_start"]
            self.progress_data["estimates"]["epoch"]["durations"].append(duration)
            del self.progress_data["timing"]["epoch_start"]

            self.progress_writer.write(self.progress_data)

    def start_training(self):
        self.progress_data["timing"]["training_start"] = time.time()
        self.progress_writer.write(self.progress_data)

    def end_training(self):
        if self.progress_data["timing"]["training_start"] is not None:
            duration = time.time() - self.progress_data["timing"]["training_start"]
            self.progress_data["estimates"]["training"]["durations"].append(duration)
            del self.progress_data["timing"]["training_start"]

            self.progress_writer.write(self.progress_data)

    def start_validation(self):
        self.progress_data["timing"]["validation_start"] = time.time()
        self.progress_writer.write(self.progress_data)

    def end_validation(self):
        if self.progress_data["timing"]["validation_start"] is not None:
            duration = time.time() - self.progress_data["timing"]["validation_start"]
            self.progress_data["estimates"]["validation"]["durations"].append(duration)
            del self.progress_data["timing"]["validation_start"]

            self.progress_writer.write(self.progress_data)

    def update_metrics(self, loss: Optional[float] = None,
                      learning_rate: Optional[float] = None,
                      samples_processed: Optional[int] = None):
        if loss is not None:
            self.progress_data["metrics"]["loss"] = loss
        if learning_rate is not None:
            self.progress_data["metrics"]["learning_rate"] = learning_rate
        if samples_processed is not None:
            self.progress_data["metrics"]["samples_processed"] = samples_processed
        self.progress_writer.write(self.progress_data)

    def update_tqdm_stats(self, rate: float, elapsed: float, remaining: float):
        """Update estimates with tqdm statistics.

        Args:
            rate: Iterations per second from tqdm (it/s)
            elapsed: Elapsed time in seconds from tqdm
            remaining: Estimated remaining time in seconds from tqdm
        """
        phase = self.progress_data["state"]["phase"]

        if phase == "training":
            self.progress_data["estimates"]["training"]["rate"] = rate
            self.progress_data["estimates"]["training"]["elapsed"] = elapsed
            self.progress_data["estimates"]["training"]["remaining"] = remaining
            self.progress_data["estimates"]["training"]["total"] = elapsed + remaining
        elif phase == "validation":
            self.progress_data["estimates"]["validation"]["rate"] = rate
            self.progress_data["estimates"]["validation"]["elapsed"] = elapsed
            self.progress_data["estimates"]["validation"]["remaining"] = remaining
            self.progress_data["estimates"]["validation"]["total"] = elapsed + remaining

        self.update_time_estimates()

        self.progress_writer.write(self.progress_data)

    def update_time_estimates(self):
        training_durations = self.progress_data["estimates"]["training"]["durations"]
        validation_durations = self.progress_data["estimates"]["validation"]["durations"]
        epoch_durations = self.progress_data["estimates"]["epoch"]["durations"]

        phase = self.progress_data["state"]["phase"]
        if phase == "training":
            current_training_total = self.progress_data["estimates"]["training"]["total"]
        elif training_durations:
            current_training_total = sum(training_durations) / len(training_durations)

        else:
            current_training_total = 0

        if phase == "validation":
            current_validation_total = self.progress_data["estimates"]["validation"]["total"]
        elif validation_durations:
            current_validation_total = sum(validation_durations) / len(validation_durations)
        else:
            ratio = self.progress_data["estimates"]["dataset_ratio"]
            current_validation_total = current_training_total * ratio

        current_epoch_total = current_training_total + current_validation_total

        current_training_elapsed = self.progress_data["estimates"]["training"]["elapsed"]
        current_validation_elapsed = self.progress_data["estimates"]["validation"]["elapsed"]
        current_epoch_elapsed = current_training_elapsed + current_validation_elapsed

        epoch_count = self.progress_data["state"]["total_epochs"]

        epoch_rate = current_epoch_elapsed / current_epoch_total if current_epoch_total > 0 else 0
        self.progress_data["estimates"]["epoch"]["rate"] = epoch_rate
        self.progress_data["estimates"]["epoch"]["elapsed"] = current_epoch_elapsed
        self.progress_data["estimates"]["epoch"]["remaining"] = current_epoch_total - current_epoch_elapsed
        self.progress_data["estimates"]["epoch"]["total"] = current_epoch_total

        if not epoch_durations:
            run_total = current_epoch_total * epoch_count
            run_elapsed = current_epoch_elapsed
        else:
            run_total = sum(epoch_durations) + current_epoch_elapsed
            run_elapsed = current_epoch_elapsed

        run_rate = run_elapsed / run_total if run_total > 0 else 0
        self.progress_data["estimates"]["run"]["rate"] = run_rate
        self.progress_data["estimates"]["run"]["elapsed"] = run_elapsed
        self.progress_data["estimates"]["run"]["remaining"] = run_total - run_elapsed
        self.progress_data["estimates"]["run"]["total"] = run_total

    def set_finished(self):
        self.progress_data["status"] = "FINISHED"
        self.progress_data["end_time"] = time.time()
        self.progress_writer.write(self.progress_data)

    def set_failed(self, error_message: str):
        self.progress_data["status"] = "FAILED"
        self.progress_data["end_time"] = time.time()
        self.progress_data["error_message"] = error_message
        self.progress_writer.write(self.progress_data)

    def set_canceled(self):
        self.progress_data["status"] = "CANCELED"
        self.progress_data["end_time"] = time.time()
        self.progress_writer.write(self.progress_data)

    def shutdown(self):
        self.stop_monitoring()
        self.progress_writer.shutdown()

    def track_variables(self, variables: Dict[str, Any]):
        """Store variables and their values to track for changes.
        Writes to control.json so ttop can edit them.

        Args:
            variables: Dictionary mapping variable names to their current values
        """
        for name, value in variables.items():
            if name not in self.tracked_variables:
                print(f"[MONITOR] Registered: {name} = {value}")
            self.tracked_variables[name] = value
            self.control_data[name] = value

        # Write to control.json
        with open(self.control_file, 'w') as f:
            json.dump(self.control_data, f, indent=2)

    def _monitor_control_file(self):
        """Background thread function that monitors control file for changes at 1s intervals."""
        while self._monitor_running:
            changes = self._detect_changes_background()

            if changes:
                # Update tracked variables
                for name, (old_value, new_value) in changes.items():
                    self.tracked_variables[name] = new_value

                # Call registered callbacks in background thread
                for name, (old_value, new_value) in changes.items():
                    for callback in self.general_callbacks:
                        callback(name, old_value, new_value)

                    if name in self.specific_callbacks:
                        for callback in self.specific_callbacks[name]:
                            callback(name, old_value, new_value)

                with self._monitor_lock:
                    self._cached_changes = changes
                    self._new_data = True

            time.sleep(1.0)

    def _detect_changes_background(self) -> Dict[str, tuple[Any, Any]]:
        """Detect changes in background by reading control.json.
        Internal method that detects changes without calling callbacks.

        Returns:
            Dictionary mapping variable names to (old_value, new_value) tuples.
            Returns empty dict if no changes detected.
        """
        changes = {}

        # Read current values from control.json
        try:
            with open(self.control_file, 'r') as f:
                control_data = json.load(f)
        except FileNotFoundError:
            control_data = {}
        except json.JSONDecodeError as e:
            control_data = {}

        for name in self.tracked_variables.keys():
            old_value = self.tracked_variables[name]
            new_value = control_data.get(name, old_value)

            if old_value != new_value:
                changes[name] = (old_value, new_value)

        return changes

    def start_monitoring(self):
        """Start the background monitoring thread."""
        if self._monitor_thread is not None and self._monitor_thread.is_alive():
            print("[MONITOR] Background monitoring already running")
            return

        self._monitor_running = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_control_file,
            daemon=True,
            name="ControlFileMonitor"
        )
        self._monitor_thread.start()
        print(f"[MONITOR] Background monitoring thread started: {self._monitor_thread.name}")
        print(f"[MONITOR] Thread is alive: {self._monitor_thread.is_alive()}")
        print(f"[MONITOR] Thread is daemon: {self._monitor_thread.daemon}")

    def stop_monitoring(self):
        """Stop the background monitoring thread."""
        if self._monitor_thread is None:
            return

        self._monitor_running = False
        self._monitor_thread.join(timeout=2.0)
        self._monitor_thread = None
        print("[MONITOR] Background monitoring thread stopped")

    def is_change(self) -> Optional[Dict[str, tuple[Any, Any]]]:
        """Check for changes in tracked variables detected by background thread.
        This allows ttop to modify variables and the training to pick them up.

        Note: Callbacks are invoked in the background thread when changes are detected,
        so this method only returns the changes without calling callbacks again.

        Returns:
            Dictionary mapping variable names to (old_value, new_value) tuples if changes detected.
            Returns None if no changes detected.

        Example:
            changes = monitor.is_change()
            if changes:
                for name, (old_val, new_val) in changes.items():
                    print(f"[CHANGE] {name}: {old_val} -> {new_val}")
                    # Handle restart if needed
        """
        with self._monitor_lock:
            if not self._new_data:
                return None

            changes = self._cached_changes
            self._new_data = False
            self._cached_changes = None

        return changes

    def check_changes(self) -> List[tuple[str, Any, Any]]:
        """Check tracked variables for changes by reading control.json.
        DEPRECATED: Use is_change() instead for background detection.
        This allows ttop to modify variables and the training to pick them up.

        Returns:
            List of tuples (variable_name, old_value, new_value) for changed variables
        """
        changes_dict = self._detect_changes_background()
        changes = []

        for name, (old_value, new_value) in changes_dict.items():
            changes.append((name, old_value, new_value))
            self.tracked_variables[name] = new_value

            print(f"[MONITOR] Changed: {name} = {old_value} -> {new_value}")

            for callback in self.general_callbacks:
                callback(name, old_value, new_value)

            if name in self.specific_callbacks:
                for callback in self.specific_callbacks[name]:
                    callback(name, old_value, new_value)

        return changes

    def register_general_callback(self, callback: Callable[[str, Any, Any], None]):
        """Register a callback to be called on any variable change.

        Args:
            callback: Function with signature callback(name: str, old_value: Any, new_value: Any)
        """
        self.general_callbacks.append(callback)

    def register_callback(self, variable_name: str, callback: Callable[[str, Any, Any], None]):
        """Register a callback for a specific variable.

        Args:
            variable_name: Name of the variable to monitor
            callback: Function with signature callback(name: str, old_value: Any, new_value: Any)
        """
        if variable_name not in self.specific_callbacks:
            self.specific_callbacks[variable_name] = []
        self.specific_callbacks[variable_name].append(callback)

    def save_checkpoint(self, checkpoint_data: Dict[str, Any]) -> str:
        """Save a training checkpoint with decision state.

        Args:
            checkpoint_data: Dictionary containing model, optimizer, and training state

        Returns:
            Path to saved checkpoint file
        """
        import torch
        checkpoint_file = self.run_folder / "checkpoint_latest.pth"
        print(f"\n[CHECKPOINT] Saving checkpoint to: {checkpoint_file}")
        torch.save(checkpoint_data, checkpoint_file)
        print(f"[CHECKPOINT] Checkpoint saved successfully")
        return str(checkpoint_file)

    def load_checkpoint(self, checkpoint_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Load a training checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file. If None, uses latest checkpoint.

        Returns:
            Checkpoint data dictionary or None if not found
        """
        import torch
        if checkpoint_path is None:
            checkpoint_path = self.run_folder / "checkpoint_latest.pth"
        else:
            checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            print(f"[CHECKPOINT] No checkpoint found at: {checkpoint_path}")
            return None

        print(f"[CHECKPOINT] Loading checkpoint from: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path)
            print(f"[CHECKPOINT] Checkpoint loaded successfully")
            return checkpoint
        except Exception as e:
            print(f"[CHECKPOINT] ERROR: Failed to load checkpoint: {e}")
            return None

    def should_restart_dataloader(self, changes: Dict[str, tuple[Any, Any]]) -> bool:
        """Determine if DataLoader restart is needed based on changes.

        Args:
            changes: Dictionary of detected changes

        Returns:
            True if batch_size or num_workers changed
        """
        critical_params = ['batch_size', 'num_workers']
        return any(param in changes for param in critical_params)

    def get_change_summary(self, changes: Optional[Dict[str, tuple[Any, Any]]]) -> str:
        """Generate full decision stack summary for detected changes.

        Args:
            changes: Dictionary of detected changes or None

        Returns:
            Formatted string with full decision stack
        """
        if not changes:
            return "[DECISION] No changes detected"

        summary = []
        summary.append("\n" + "="*80)
        summary.append("[DECISION STACK] ANALYZING CONFIGURATION CHANGES")
        summary.append("="*80)

        critical_params = ['batch_size', 'num_workers']
        critical_changes = {k: v for k, v in changes.items() if k in critical_params}
        other_changes = {k: v for k, v in changes.items() if k not in critical_params}

        summary.append(f"\n[DECISION] Total changes detected: {len(changes)}")

        if critical_changes:
            summary.append(f"[DECISION] Critical changes: {len(critical_changes)}")
            for var_name, (old_val, new_val) in critical_changes.items():
                summary.append(f"  - {var_name}: {old_val} -> {new_val}")
        else:
            summary.append("[DECISION] Critical changes: 0")

        if other_changes:
            summary.append(f"[DECISION] Non-critical changes: {len(other_changes)}")
            for var_name, (old_val, new_val) in other_changes.items():
                summary.append(f"  - {var_name}: {old_val} -> {new_val}")

        requires_restart = self.should_restart_dataloader(changes)
        summary.append(f"\n[DECISION] DataLoader restart required: {requires_restart}")

        if requires_restart:
            summary.append("[DECISION] RESTART DECISION TREE:")
            summary.append("  1. DETECT CHANGE -> Completed")
            summary.append("  2. VALIDATE CHANGE -> Proceeding")
            summary.append("  3. SAVE CHECKPOINT -> Required")
            summary.append("  4. RECREATE DATALOADERS -> Required")
            summary.append("  5. LOAD CHECKPOINT -> If available")
            summary.append("  6. RESUME TRAINING -> Continue with new config")
        else:
            summary.append("[DECISION] Only parameter updates (no restart needed)")

        summary.append("="*80 + "\n")

        return "\n".join(summary)
