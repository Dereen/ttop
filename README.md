# ttop - Real-time Job Monitor

![Claude Code Opus 4.5 coded | Unreviewed](https://img.shields.io/badge/Claude%20Code%20Opus%204.5%20coded-Unreviewed-grey?logo=claude&logoColor=white&labelColor=D97757)

A real-time terminal UI for monitoring long-running jobs, system resources, and process details in a single view.

## Overview

`ttop` monitors progress files created by jobs using `TrainingMonitor` and displays their data alongside system resource usage (CPU, RAM, GPU). It provides an interactive interface for exploring job metadata, filtering runs, and managing processes.

The tool connects to progress JSON files in a specified folder, continuously collects and displays:
- Job metadata (status, current phase, progress information)
- Progress metrics (elapsed, remaining, total times)
- System resource usage per job
- Process and thread-level resource details
- Custom metrics reported by the job

## Features

- **Real-time Monitoring**: Displays job status and system resources with configurable refresh intervals
- **Interactive UI**: Terminal-based interface with color-coded status indicators
- **Filtering**: Filter jobs by status or name pattern
- **Process Explorer**: Expand jobs to view child processes and thread-level details
- **Resource Tracking**: Monitor CPU, RAM, and GPU usage per job
- **Process Management**: Send signals (SIGTERM, SIGKILL, etc.) to running processes
- **CSV Export**: Optional CSV output for data logging and analysis
- **Headless Mode**: Run without UI for server environments, outputting only to CSV
- **Variable Editor**: Edit job variables directly from the monitor
- **Output Viewer**: View job stdout/stderr logs in real-time

## Installation

Ensure you have Python 3.6+ with the required dependencies:

```bash
pip install psutil
```

## Usage

### Basic Usage

Monitor a progress folder with default settings:

```bash
python ttop.py ./progress
```

### With Custom Refresh Interval

```bash
python ttop.py ./progress --interval 1.5
```

### With Resource Limits

Specify maximum resource allocations to track usage as percentage:

```bash
python ttop.py ./progress \
  --max-cpus 16 \
  --max-ram 64 \
  --max-gpu 24
```

### CSV Output

Collect data to CSV files at regular intervals:

```bash
python ttop.py ./progress \
  --csv-interval 5.0 \
  --csv-dir ./statistics
```

### Headless Mode

Run without UI, only outputting to CSV (ideal for servers):

```bash
python ttop.py ./progress \
  --headless \
  --csv-interval 5.0 \
  --csv-dir ./statistics
```

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `s` | Cycle through status filters (all, RUNNING, FINISHED, CANCELED, FAILED, KILLED) |
| `f` | Enter name filter mode (type to filter, Enter to confirm, ESC to cancel) |
| `t` | Toggle expanded view for selected job (shows processes and threads) |
| `e` | Edit variables for selected job |
| `o` | View stdout/stderr output for selected job |
| `k` | Send signal (kill/terminate/etc.) to selected process/thread |
| `l` | Send signal to all RUNNING jobs |
| `c` | Delete selected job folder |
| `↑↓` | Navigate rows |
| `←→` | Scroll columns horizontally |
| `ESC` | Clear name filter and reset view |
| `Ctrl+C` | Exit the monitor |

## Command-line Options

```
positional arguments:
  folder_path              Path to folder containing training JSON files

optional arguments:
  -h, --help              Show this help message and exit
  -i, --interval SECONDS  Refresh interval in seconds (default: 2.0)
  --max-cpus CORES        Maximum CPU cores allocated (default: all available)
  --max-ram GB            Maximum RAM in GB allocated (default: total system RAM)
  --max-gpu GB            Maximum GPU memory in GB allocated (default: total GPU memory)
  --csv-interval SECONDS  CSV output interval in seconds (default: disabled)
  --csv-dir PATH          Directory for CSV output (default: ./train/statistics/data)
  --headless              Run without UI, only collect data to CSV (useful for servers)
```

## Color Legend

The display uses color coding to indicate job status:

- **Loading**: Job initializing
- **Batch**: Currently processing items
- **Validation**: Running validation/checkpoint phase
- **Finished**: Job completed successfully
- **Failed**: Job failed
- **Canceled**: Job was canceled
- **Killed**: Job was killed

When a row is selected/highlighted, the color becomes darker to indicate focus.

## Display Columns

### Main Job Row

| Column | Description |
|--------|-------------|
| Run Name | Name of the job |
| Status | Job status (RUNNING, FINISHED, etc.) |
| Phase | Current execution phase |
| Epoch | Epoch/iteration counter (if applicable) |
| Batch | Batch/step counter (if applicable) |
| Run Elapsed/Remaining/Total | Total job time breakdown |
| Epoch Elapsed/Remaining/Total | Current phase time breakdown |
| Train Elapsed/Remaining/Total | Primary phase time breakdown |
| Val Time | Validation phase time |
| Val Elapsed/Remaining/Total | Validation time breakdown |
| CPU | CPU usage percentage |
| RAM | RAM usage in MB |
| GPU | GPU memory usage in MiB |
| Procs | Number of child processes |
| Threads | Number of threads |
| PID | Process ID |
| Wandb URL | Link to external service (if available) |

### Process/Thread Rows

When expanded, child processes and threads show with indentation:
- `[P]` prefix for processes
- `[T]` prefix for threads

Process rows display CPU, RAM, GPU usage, and thread count. Thread rows show only the thread ID.

## Examples

### Monitor jobs with resource limits and custom refresh interval

```bash
python ttop.py ./progress \
  --interval 5.0 \
  --max-cpus 8 \
  --max-ram 16 \
  --max-gpu 8
```

### Run headless, logging to CSV every 10 seconds

```bash
python ttop.py ./training_data \
  --headless \
  --csv-interval 10.0 \
  --csv-dir ./metrics
```

### Monitor with aggressive updates and view process details

```bash
python ttop.py ./jobs \
  --interval 0.5 \
  --max-cpus 32 \
  --max-ram 128 \
  --max-gpu 48
```

Then use `t` to expand runs and see process/thread details.

## Tutorial: Integration and Usage

### Overview

`ttop` is designed to work with scripts that use `TrainingMonitor`. The monitor writes progress data to JSON files in a designated folder, and `ttop` reads and displays this data in real-time alongside system resource usage.

### Step 1: Integrate TrainingMonitor into Your Script

In your script, create a `TrainingMonitor` instance and use it throughout execution:

```python
from train.utils.performance import TrainingMonitor

# Initialize monitor (writes to progress/{run_name}.json)
monitor = TrainingMonitor(
    output_file=f"progress/{run_name}.json"
)

# Track variables you want to be editable during execution (optional)
monitor.track_variables({
    'param1': value1,
    'param2': value2
})

# Start background monitoring
monitor.start_monitoring()
```

### Step 2: Update Monitor During Execution

Call monitor methods at key points:

```python
# Start phases of execution
monitor.start_phase("phase_name")

# Track iterations (call before each iteration)
for iteration in range(total_iterations):
    monitor.start_batch()
    # ... your code ...
    monitor.update_metrics(metric1=value1, metric2=value2)

# End phases
monitor.end_phase()

# Report final status
monitor.set_finished()         # Completed successfully
monitor.set_failed(error_msg)  # Failed with error
monitor.set_canceled()         # Interrupted by user
```

### Step 3: Run Your Script and Monitor with ttop

In one terminal, start your script:

```bash
python your_script.py --run_name my_run
```

This creates: `progress/my_run.json`

In another terminal, start ttop to monitor:

```bash
python ttop.py ./progress --interval 1.0
```

### Step 4: Interactive Monitoring Workflow

Once ttop is running, you have full control:

#### Monitor Progress
- View run status, current phase, and progress metrics
- See real-time CPU, RAM, and GPU usage
- Watch metrics update as they're reported

#### View Process Details
- Press `t` to expand a run and see child processes
- Inspect subprocesses and threads
- Monitor individual process resource usage

#### Edit Variables
- Press `e` to open the variable editor
- Modify tracked variables during execution
- Script detects changes and adapts accordingly

#### View Output
- Press `o` to open the output viewer
- See real-time stdout/stderr from the run
- Useful for debugging or checking messages

#### Manage Processes
- Press `k` to send signals (SIGTERM, SIGKILL, etc.) to processes
- Press `l` to send signals to all RUNNING jobs
- Gracefully stop or kill when needed

#### Filter and Navigate
- Press `s` to filter by status (RUNNING, FINISHED, FAILED, etc.)
- Press `f` to filter by run name
- Use arrow keys to navigate, `←→` to scroll columns

### Practical Workflow Example

Here's a complete example workflow:

```bash
# Terminal 1: Start your script
$ python your_script.py --run_name job_001

# Terminal 2: Monitor with ttop
$ python ttop.py ./progress --max-cpus 32 --max-ram 128 --max-gpu 48

# In ttop (Terminal 2):
# - Watch the run appear as it starts
# - Press 't' to expand and see subprocesses
# - Press 'o' to check stdout/stderr
# - If parameters need adjustment:
#   - Press 'e' to edit
#   - Modify tracked variables
#   - Script detects change and adapts
# - If something fails:
#   - Status changes to FAILED
#   - Press 'o' to view error messages

# Terminal 3 (optional): Log metrics to CSV
$ python ttop.py ./progress \
    --headless \
    --csv-interval 30.0 \
    --csv-dir ./metrics
```

### Monitoring Multiple Jobs

`ttop` handles multiple concurrent jobs:

```bash
# Terminal 1: Start first job
$ python your_script.py --run_name job_001 &

# Terminal 2: Start second job
$ python your_script.py --run_name job_002 &

# Terminal 3: Monitor all with ttop
$ python ttop.py ./progress

# In ttop:
# - Both jobs appear in the table
# - Filter by status with 's' or name with 'f'
# - Expand individual jobs with 't' to see their processes
# - Edit variables independently for each job with 'e'
```

### Headless Monitoring for Servers

For server environments without a display, use headless mode:

```bash
# Run your script normally
$ python your_script.py --run_name my_job

# In another terminal, collect metrics without UI
$ python ttop.py ./progress \
    --headless \
    --csv-interval 10.0 \
    --csv-dir ./metrics

# Metrics are saved to CSV for later analysis
```

### Data Files Created

`ttop` and your script create:

```
progress/
├── {run_name}.json                 # Main progress file
├── control_{run_name}.json         # Control file for variable edits
└── checkpoint_{run_name}.pth       # Optional checkpoint (if using)

metrics/
├── {run_name}_matrix.csv           # Resource metrics over time
└── {run_name}_overall.csv          # Overall statistics
```

### Tips for Best Results

1. **Set Accurate Resource Limits**: Use `--max-cpus`, `--max-ram`, `--max-gpu` to see usage percentages
2. **Adjust Refresh Interval**: Use `--interval 0.5` for faster updates, `2.0` for lower CPU usage
3. **Monitor Early**: Start ttop as soon as your job begins
4. **Use Filters**: Press `s` and `f` to focus on specific runs
5. **Check Expansion Details**: Press `t` on RUNNING jobs to see their processes
6. **Check Metrics**: Press `e` to view or edit variables, `o` to view output
7. **Use CSV Export**: Run headless mode in parallel to log metrics for analysis

## Architecture

The tool consists of:

- **ttop.py**: Main UI and event loop
- **background_data_collector.py**: Background thread for continuous data collection
- **cpu_resources.py**: CPU and process tree analysis
- **gpu_resources.py**: GPU resource tracking
- **data_formatting.py**: Formatting progress data into tabular display
- **color_scheme.py**: Terminal color management
- **output_viewer.py**: Stdout/stderr viewer window
- **variable_editor.py**: Variable editor window
- **csv_writer.py**: CSV data export
- **time_utils.py**: Duration formatting utilities

## Requirements

- Python 3.6+
- psutil library for system resource monitoring
- curses library (built-in on Unix-like systems)

## Known Issues & TODO

- [ ] Wandb URL doesn't work
- [ ] Include RAM, CPU, GPU resources (per process, all threads) into progress.json in TrainingMonitor
- [ ] Hide/show stdout/stderr output in editor and output viewer
- [ ] Rewrite ttop to use tmux tabs and panes
- [ ] Editor doesn't refresh PID and hardware resources are unavailable until exiting editor and returning
- [ ] Missing rate in main table and in editor
- [ ] Editor formatting is difficult to read
- [ ] Main table doesn't show validation batch
- [ ] Load more than 1000 lines of outputs
  - [ ] Implement smarter scrolling for output viewer

## License

See LICENSE file for details.
