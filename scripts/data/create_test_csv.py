import csv
import os

from phylo_gnn import get_data_path


def create_subset_csv(csv_path, n_rows, new_file_name=None):
    """
    Creates a new CSV file containing only the first n rows from the original CSV.

    Parameters:
    -----------
    csv_path : str
        Path to the original CSV file
    n_rows : int
        Number of rows to include in the new CSV (including header)
    new_file_name : str, optional
        Name for the new CSV file. If None, '_subset_n' will be appended to the original filename

    Returns:
    --------
    str
        Path to the newly created CSV file
    """
    # Validate input
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    if n_rows <= 0:
        raise ValueError("Number of rows must be positive")

    # Generate new file name if not provided
    if new_file_name is None:
        base_name, ext = os.path.splitext(csv_path)
        new_file_name = f"{base_name}_subset_{n_rows}{ext}"

    # Read the original CSV and write first n rows to new file
    rows_written = 0

    with open(csv_path, "r", newline="") as input_file:
        reader = csv.reader(input_file)

        with open(new_file_name, "w", newline="") as output_file:
            writer = csv.writer(output_file)

            # Write rows until we reach n_rows or run out of rows
            for row in reader:
                writer.writerow(row)
                rows_written += 1

                if rows_written >= n_rows:
                    break

    print(f"Created subset CSV with {rows_written} rows at: {new_file_name}")
    return new_file_name


if __name__ == "__main__":
    data_path = get_data_path() / "raw" / "classification"
    csv_path = data_path / "87_10k_nwk.csv"
    n_rows = 1000
    new_file_name = create_subset_csv(csv_path, n_rows)
    print(f"New CSV file created: {new_file_name}")
