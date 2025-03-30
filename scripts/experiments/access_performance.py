import os
import csv
import time
import tempfile
import shutil
from pathlib import Path


def setup_test_environment(num_files=1000, content_size=100):
    """
    Create a test directory with individual files and a CSV file.

    Args:
        num_files: Number of files to create
        content_size: Size of content in each file (in characters)

    Returns:
        tuple: (temp_dir_path, csv_file_path)
    """
    print(f"Setting up test environment with {num_files} files...")

    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    temp_dir_path = Path(temp_dir)

    # Create CSV file
    csv_path = temp_dir_path / "data.csv"

    # Generate content of specified size
    base_content = "X" * content_size

    # Create individual files and CSV data
    with open(csv_path, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["id", "content"])  # Header

        for i in range(num_files):
            # Unique content for this file
            content = f"File {i}: {base_content}"

            # Write to CSV
            csv_writer.writerow([i, content])

            # Create individual file
            file_path = temp_dir_path / f"file_{i}.txt"
            with open(file_path, "w") as f:
                f.write(content)

    print(
        f"Created {num_files} individual files and a CSV with {num_files} rows"
    )
    return temp_dir_path, csv_path


def read_individual_files(directory, num_runs=5):
    """
    Time the reading of all .txt files in the directory.
    """
    total_time = 0

    for run in range(num_runs):
        start_time = time.time()

        file_contents = []
        for file_path in sorted(directory.glob("file_*.txt")):
            with open(file_path, "r") as f:
                content = f.read()
                file_contents.append(content)

        end_time = time.time()
        run_time = end_time - start_time
        total_time += run_time

        print(
            f"Run {run+1}: Read {len(file_contents)} individual files in {run_time:.5f} seconds"
        )

    avg_time = total_time / num_runs
    print(f"\nAverage time to read individual files: {avg_time:.5f} seconds")
    return avg_time


def read_csv_file(csv_path, num_runs=5):
    """
    Time the reading of all rows from a CSV file.
    """
    total_time = 0

    for run in range(num_runs):
        start_time = time.time()

        rows = []
        with open(csv_path, "r", newline="") as csvfile:
            csv_reader = csv.reader(csvfile)
            next(csv_reader)  # Skip header
            for row in csv_reader:
                rows.append(row)

        end_time = time.time()
        run_time = end_time - start_time
        total_time += run_time

        print(
            f"Run {run+1}: Read {len(rows)} CSV rows in {run_time:.5f} seconds"
        )

    avg_time = total_time / num_runs
    print(f"\nAverage time to read CSV file: {avg_time:.5f} seconds")
    return avg_time


def compare_performance(num_files=1000, content_size=100, num_runs=5):
    """
    Compare the performance of reading individual files vs a CSV file.
    """
    # Setup test environment
    temp_dir_path, csv_path = setup_test_environment(num_files, content_size)

    try:
        print("\nTesting performance of reading individual files...")
        individual_time = read_individual_files(temp_dir_path, num_runs)

        print("\nTesting performance of reading CSV file...")
        csv_time = read_csv_file(csv_path, num_runs)

        # Compare results
        ratio = individual_time / csv_time

        print("\nResults:")
        print(
            f"Average time to read {num_files} individual files: {individual_time:.5f} seconds"
        )
        print(
            f"Average time to read {num_files} rows from CSV: {csv_time:.5f} seconds"
        )
        print(
            f"Ratio: Reading individual files is {ratio:.2f}x slower than reading a CSV file"
        )

        return {
            "individual_time": individual_time,
            "csv_time": csv_time,
            "ratio": ratio,
        }

    finally:
        # Clean up
        shutil.rmtree(temp_dir_path)


if __name__ == "__main__":
    # Run the performance comparison
    compare_performance(num_files=60000, content_size=100, num_runs=5)
