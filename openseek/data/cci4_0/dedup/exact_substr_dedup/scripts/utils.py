import os
import glob
import time
import numpy as np
from rich.console import Console
from rich.table import Table

class Timer:
    def __init__(self):
        self.times = []
        self.descriptions = []  # Store descriptions
        self.start()

    def start(self):
        self.tik = time.time()

    def stop(self, description=""):
        elapsed_time = time.time() - self.tik
        self.times.append(elapsed_time)
        self.descriptions.append(description)  # Store the description
        return elapsed_time

    def avg(self):
        return sum(self.times) / len(self.times) if self.times else 0

    def sum(self):
        return sum(self.times)

    def cumsum(self):
        return np.array(self.times).cumsum().tolist()
    
    def pretty_print(self):
        """Prints the timing results using rich.table."""
        if not self.times:
            print("No timing data recorded yet.")
            return

        console = Console()
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Index", style="dim", width=6, justify="right")
        table.add_column("Description", min_width=20)
        table.add_column("Time (s)", justify="right")
        table.add_column("Cumulative Time (s)", justify="right")

        cumulative_times = self.cumsum()
        for i, (time_val, desc, cum_time) in enumerate(zip(self.times, self.descriptions, cumulative_times)):
            table.add_row(
                str(i + 1),
                desc,
                f"{time_val:.4f}",
                f"{cum_time:.4f}"
            )

        console.print(table)


def human_readable_size(size_bytes):
    """
    Converts a size in bytes to a human-readable string.

    Args:
        size_bytes: The size in bytes (integer).

    Returns:
        A human-readable string representation of the size (e.g., "1.23 KB").
        Returns an empty string if input is invalid.
    """
    if not isinstance(size_bytes, (int, float)):
      return ""  # or raise TypeError("size_bytes must be a number")
    if size_bytes < 0:
        return "-" + human_readable_size(-size_bytes)  # Handle negative sizes
    if size_bytes == 0:
      return "0 B"

    suffixes = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = 0
    while size_bytes >= 1024 and i < len(suffixes) - 1:
        size_bytes /= 1024.0
        i += 1
    return f"{size_bytes:.5f} {suffixes[i]}"


def get_size(path):
    """
    Gets the total size of a directory or file in a human-readable format,
    similar to `du -sh`.

    Args:
        path: The path to the directory or file.

    Returns:
        A string representing the size in a human-readable format (e.g., "1.23 KB").
        Returns "0 B" if the path doesn't exist or is inaccessible.
        Returns an empty string if input is invalid
    """
    try:
        total_size = 0
        if os.path.isfile(path):
            total_size = os.path.getsize(path)
        elif os.path.isdir(path):
            for dirpath, dirnames, filenames in os.walk(path):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    # Skip if it is symbolic link to avoid infinite recursion
                    if not os.path.islink(fp):
                        total_size += os.path.getsize(fp)
        else:
            return 0  # Path doesn't exist or is inaccessible
        return total_size

    except OSError:
        return 0  # Handle potential errors like permission issues
    

def assert_files_exit(folder_path, patterns):
    for pattern in patterns:
        files = glob.glob(os.path.join(folder_path, pattern))
        files = [f for f in files if os.path.isfile(f)]
        assert files, f"No {pattern} files in {folder_path}."



# Example usage (including how to use the new features):
if __name__ == '__main__':
    timer = Timer()
    time.sleep(1)  # Simulate some work
    timer.stop("Initialization")

    for i in range(3):
        timer.start()  # Restart the timer
        time.sleep(0.5 * (i + 1)) # Simulate varying workloads
        timer.stop(f"Loop iteration {i+1}")

    timer.start()
    time.sleep(0.2)
    timer.stop("Final step")
    
    print(f"Average time: {timer.avg():.4f} s")
    print(f"Total time: {timer.sum():.4f} s")
    print(f"Cumulative times: {timer.cumsum()}")

    timer.pretty_print()  # Use the new pretty_print method
    
    #Example with empty times
    timer2 = Timer()
    timer2.pretty_print()

    timer3 = Timer()
    timer3.stop("Only one record")
    timer3.pretty_print()