import os
import subprocess
import time
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import argparse
import sys
import re
import queue
from datetime import datetime
import signal

def get_cpu_affinity_groups():
    """Get CPU core groups, each group contains 8 cores, considering parent process CPU binding"""
    try:
        # Get current process CPU affinity (if bound by taskset)
        with open('/proc/self/status', 'r') as f:
            status = f.read()
        cpus_allowed = re.search(r'Cpus_allowed:\s*([0-9a-f,]+)', status)
        if cpus_allowed:
            # Parse hexadecimal mask
            mask_str = cpus_allowed.group(1).replace(',', '')
            mask = int(mask_str, 16)
            # Convert mask to CPU list
            available_cores = []
            for i in range(604):
                if mask & (1 << i):
                    available_cores.append(i)
        else:
            # If not bound, use all cores
            available_cores = list(range(os.cpu_count()))
    except:
        # If NUMA information cannot be obtained, use sequential allocation
        available_cores = list(range(os.cpu_count()))
    
    # If no cores are available, use all cores
    if not available_cores:
        available_cores = list(range(os.cpu_count()))
    # Maximum of 160 cores used simultaneously
    execution_core = len(available_cores)
    # Create core groups (each group contains 8 cores)
    affinity_groups = []
    for i in range(0, execution_core, 8):
        group = available_cores[i:i+8]
        if len(group) == 8:
            affinity_groups.append(group)

    # If insufficient cores, try grouping
    if not affinity_groups:
        for i in range(0, len(available_cores), 8):
            group = available_cores[i:i+8]
            if len(group) > 0:
                affinity_groups.append(group)
    
    return len(available_cores), affinity_groups

def clean_command(command):
    """Clean command: remove OMP_NUM_THREADS settings"""
    # Use regular expression to remove OMP_NUM_THREADS=xx
    cleaned = re.sub(r'OMP_NUM_THREADS\s*=\s*\d+\s*', '', command)
    # Remove extra spaces
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

def get_mem_usage_percent():
    """
    Read /proc/meminfo and return system memory usage percentage (0~100).
    """
    meminfo = {}
    with open("/proc/meminfo", "r") as f:
        for line in f:
            key, val = line.split(":", 1)
            meminfo[key] = int(val.strip().split()[0])
    total = meminfo.get("MemTotal", 0)
    avail = meminfo.get("MemAvailable", 0)
    if total == 0:
        return 0.0
    used = total - avail
    percent = used / total * 100.0
    return used / total * 100.0

def run_command(command, log_file, affinity_group, working_dir):
    """Execute a single command and manage logs, memory usage, and timeouts"""
    cleaned_command = clean_command(command)
    affinity_str = ','.join(map(str, affinity_group))
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(len(affinity_group))
    start_time = time.time()
    status = "SUCCESS"
    process = None
    output_lines = []

    try:
        with open(log_file, 'a') as f:
            # Write execution context information
            f.write(f"\n{'='*80}\n")
            f.write(f"Command: {command}\nCleaned: {cleaned_command}\n")
            f.write(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
                    f"Working Directory: {working_dir}\n")
            f.write(f"CPU Affinity: {affinity_str} | OMP_NUM_THREADS=8\n")
            f.write(f"{'='*80}\n")
            f.flush()

            full_command = f"cd {working_dir} && taskset -c {affinity_str} {cleaned_command}"
            process = subprocess.Popen(
                full_command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                text=True,
                bufsize=1,
                env=env,
                preexec_fn=os.setsid
            )

            try:
                stdout_data, _ = process.communicate(timeout=2400)
                output_lines = stdout_data.splitlines()
                for line in output_lines:
                    f.write(line + "\n")
                status = "SUCCESS"
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                try:
                    stdout_data, _ = process.communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                output_lines = stdout_data.splitlines() if 'stdout_data' in locals() else []
                status = "TIMEOUT"

            return_code = process.returncode
            if return_code != 0 and status != "TIMEOUT":
                raise subprocess.CalledProcessError(return_code, cleaned_command)

    except subprocess.CalledProcessError:
        status = "FAILED"
    except Exception:
        status = "ERROR"
    finally:
        duration = time.time() - start_time
        with open(log_file, 'a') as f:
            f.write(f"\n{'='*80}\nStatus: {status} | Duration: {duration:.2f}s\n")
            if status != "SUCCESS":
                f.write(f"Error Output:\n{'='*40}\n")
                for line in output_lines:
                    f.write(line + "\n")
                f.write(f"{'='*40}\n")
            f.write(f"{'='*80}\n\n")
            f.flush()

        if process and process.poll() is None:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except Exception:
                pass

    output_str = "\n".join(output_lines)
    if "FAILED" not in output_str and status == "FAILED":
        # If "FAILED" not in the output, the command might was killed by OS.
        # Mark as timeout, retry in sequence at the end. 
        return (command, cleaned_command, "TIMEOUT", duration, affinity_str)

    if status == "SUCCESS":
        return (command, cleaned_command, "SUCCESS", duration, affinity_str)

    return (command, cleaned_command, status, duration, affinity_str)

def command_worker(command_queue, result_queue, log_file, affinity_group, working_dir):
    """Worker process function, continuously processes commands until the queue is empty"""
    while True:
        try:
            # —— Memory check here —— #
            mem_threshold = 40
            mem_check_interval = 10
            while True:
                used = get_mem_usage_percent()
                if used < mem_threshold:
                    break
                time.sleep(mem_check_interval)
            # Try to get a command from the queue (with timeout)
            command = command_queue.get(timeout=1)
            # Execute command
            result = run_command(command, log_file, affinity_group, working_dir)
            # Put result into result queue
            result_queue.put(result)
            # Mark task as done
            command_queue.task_done()
        except queue.Empty:
            # Exit when queue is empty
            break
        except Exception as e:
            print(f"Command execution exception: {e}")
            result_queue.put((command, "", "ERROR", 0, affinity_group))
            command_queue.task_done()

def main():
    # Automatically get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Set default command file path
    default_command_file = os.path.join(script_dir, "daily_build_cases.txt")
    
    # Set working directory to two levels above the script directory
    working_dir = os.path.abspath(os.path.join(script_dir, "../../oneDNN-3.4/build/tests/benchdnn"))
    
    # Set single log file path
    log_file = os.path.join(script_dir, "test_report.log")
    
    parser = argparse.ArgumentParser(description='Parallel test scheduling system')
    parser.add_argument('--command_file', default=default_command_file, 
                        help=f'Text file containing test commands (default: {default_command_file})')
    parser.add_argument('--working_dir', default=working_dir,
                        help=f'Command execution directory (default: {working_dir})')
    parser.add_argument('--log_file', default=log_file,
                        help=f'Log file (default: {log_file})')
    args = parser.parse_args()

    # Check if command file exists
    if not os.path.exists(args.command_file):
        print(f"Error: Command file does not exist - {args.command_file}")
        print("Please create a text file with test commands or specify using --command_file")
        sys.exit(1)
    
    # Check if working directory exists
    if not os.path.exists(args.working_dir):
        print(f"Error: Working directory does not exist - {args.working_dir}")
        print("Please create the directory or specify a valid directory using --working_dir")
        sys.exit(1)
    
    # Read command file
    with open(args.command_file, 'r') as f:
        commands = [line.strip() for line in f if line.strip()]
    
    if not commands:
        print("Error: No valid commands found")
        sys.exit(1)
    
    total_commands = len(commands)
    print(f"Loaded {total_commands} test commands")
    print(f"Command file: {args.command_file}")
    print(f"Working directory: {args.working_dir}")
    print(f"Log file: {args.log_file}")
    
    # Get CPU affinity groups (considering parent process binding)
    num_available_cores, affinity_groups = get_cpu_affinity_groups()
    max_workers = len(affinity_groups)

    print(f"Available cores: {num_available_cores} | Maximum concurrent processes: {max_workers}")
    print(f"Each task configuration: OMP_NUM_THREADS=8 + CPU affinity group (8 cores per group)")
    print(f"Starting tests... (Press Ctrl+C to interrupt)")
    print("-" * 60)

    # Create command queue and result queue
    command_queue = multiprocessing.JoinableQueue()
    result_queue = multiprocessing.Queue()
    
    # Fill command queue
    for cmd in commands:
        command_queue.put(cmd)
    
    # Start worker processes
    start_time = time.time()
    success_count = 0
    failed_commands = []
    timeout_commands = []
    
    # Initialize log file
    with open(args.log_file, 'w') as f:
        f.write(f"{'='*80}\n")
        f.write(f"Test Report\n")
        f.write(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Command file: {args.command_file}\n")
        f.write(f"Working directory: {args.working_dir}\n")
        f.write(f"Total commands: {total_commands}\n")
        f.write(f"Available cores: {num_available_cores}\n")
        f.write(f"Maximum concurrent processes: {affinity_groups}\n")
        f.write(f"{'='*80}\n\n")
        f.flush()
    
    # Progress display variables
    last_update = time.time()
    completed = 0
    success_count = 0
    failed_count = 0
    timeout_count = 0
    
    # Single-line progress display function
    def update_progress():
        """Update single-line progress display"""
        elapsed = time.time() - start_time
        progress = completed / total_commands * 100
        mem_usage_percent = get_mem_usage_percent()
        mem_message = ""
        if mem_usage_percent > 40: mem_message = "\033[94m[High memory usage. Concurrency reduced.]\033[0m"
        sys.stdout.write("\r")
        sys.stdout.write('\x1b[K') 
        sys.stdout.write(f"Progress: {progress:.1f}% | Completed: {completed}/{total_commands} | Success: {success_count} | Failed: {failed_count} | Timeout: {timeout_count} | Memory usage: {mem_usage_percent:.1f}% {mem_message} | Elapsed: {elapsed:.0f}s")
        sys.stdout.flush()
    
    # Initial progress display
    update_progress()
    
    try:
        # Use ThreadPoolExecutor to manage worker processes
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Assign a unique CPU affinity group to each worker process
            futures = []
            for group in affinity_groups:
                future = executor.submit(
                    command_worker, 
                    command_queue,
                    result_queue,
                    args.log_file,  # All commands share the same log file
                    group,
                    args.working_dir
                )
                futures.append(future)
            
            # Process results in real-time and update progress
            while completed < total_commands:
                try:
                    # Get result from result queue (with timeout)
                    command, cleaned_command, status, duration, affinity_str = result_queue.get(timeout=1)
                    completed += 1
                    
                    if status == "SUCCESS":
                        success_count += 1
                    elif status == "TIMEOUT":
                        timeout_count += 1
                        # Display timeout on a new line
                        print(f"\nTimeout: {cleaned_command}  [Will retry at the end]")
                        # Save timeout command information
                        timeout_commands.append((cleaned_command, affinity_str))
                    else:
                        failed_count += 1
                        # Display failure on a new line
                        print(f"\nFailed: {cleaned_command}  [Will retry at the end]")
                        # Save failed command information
                        failed_commands.append((cleaned_command, affinity_str))
                    
                    # Update progress display
                    update_progress()
                    
                    # Periodically write to log
                    if time.time() - last_update > 10:
                        with open(args.log_file, 'a') as f:
                            f.write(f"Completed: {completed}/{total_commands} | Success: {success_count} | Failed: {failed_count} | Timeout: {timeout_count}\n")
                            f.flush()
                        last_update = time.time()
            
                except queue.Empty:
                    # Check if all worker processes have finished
                    if all(f.done() for f in futures):
                        print("\nAll worker processes have finished, but not all results have been received")
                        break
                    time.sleep(0.5)
                except Exception as e:
                    print(f"Exception while processing results: {e}")
                    break
            
            # Wait for all tasks to complete
            command_queue.join()
    except KeyboardInterrupt:
        print("\nInterrupt signal detected, stopping tests...")
        # Clear queue to stop all worker processes
        while not command_queue.empty():
            command_queue.get()
            command_queue.task_done()
        print("Tests have been interrupted")
    
    # Final progress display
    print("\n")  # End single-line display

    # Write final report to log file
    with open(args.log_file, 'a') as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"Total commands: {total_commands}\n")
        f.write(f"Success: {success_count} | Failed: {failed_count} | Timeout: {timeout_count}\n")
        f.write(f"Success rate: {success_count/total_commands*100:.2f}%\n")

        if failed_commands:
            f.write(f"\n{'='*80}\n")
            f.write(f"Failed commands list:\n")
            f.write(f"{'='*80}\n")
            for cmd, affinity in failed_commands:
                f.write(f" - [Core group: {affinity}] {cmd}\n")
            f.write(f"{'='*80}\n")
        
        all_usable_core = [core for affinity_group in affinity_groups for core in affinity_group]
        if timeout_commands:
            f.write(f"\n{'='*80}\n")
            f.write(f"Timeout commands list:\n")
            f.write(f"{'='*80}\n")
            print("Retrying timeout commands serially...\n")
            for cmd, affinity in timeout_commands:
                result = run_command(cmd, args.log_file, all_usable_core, args.working_dir)
                if result[2] != "SUCCESS":
                    print(f"Failed on retry: {cmd}")
                    f.write(f" - [Core group: {affinity}] {cmd}\n")
                else:
                    print(f"Passed on retry: {cmd}")
                    timeout_count -= 1
                    success_count += 1
            f.write(f"{'='*80}\n")
        
        f.write(f"\n{'='*80}\n")
        f.write(f"Report generated time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*80}\n")
        f.flush()
    
    # Generate final report
    total_time = time.time() - start_time
    avg_time = total_time / total_commands if total_commands > 0 else 0

    # Console output of final report
    print(f"\n{'='*80}")
    print(f"Tests completed! Total time: {total_time:.2f} seconds")
    print(f"Total commands: {total_commands}")
    print(f"Success: {success_count} | Failed: {failed_count} | Timeout: {timeout_count}")
    print(f"Success rate: {success_count/total_commands*100:.2f}%")
    print(f"Average time per command: {avg_time:.2f} seconds")

    print(f"\nDetailed report please check: {args.log_file}")

if __name__ == "__main__":
    main()