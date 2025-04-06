import random
import os
import pandas as pd

from phylo_gnn import get_data_path


def create_subset_csv(csv_path, n_rows, new_file_name=None, random_seed=None):
    """
    Creates a new CSV file containing a random sample of n rows from the
    original CSV.

    Parameters:
    -----------
    csv_path : str
        Path to the original CSV file
    n_rows : int
        Number of rows to include in the new CSV (including header)
    new_file_name : str, optional
        Name for the new CSV file. If None, '_subset_n' will be appended to the
        original filename
    random_seed : int, optional
        Seed for random number generator for reproducible results

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

    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_path)

    # Check if we're requesting more rows than available (minus header)
    available_rows = len(df)
    if n_rows > available_rows + 1:  # +1 for header
        print(
            f"Warning: Requested {n_rows} rows but only {available_rows} "
            "rows available (plus header)"
        )
        n_rows = available_rows + 1

    # Random sample of rows (n_rows - 1 to account for header which is always
    # included)
    sample_size = min(n_rows - 1, available_rows)

    if random_seed is not None:
        random.seed(random_seed)
        sampled_df = df.sample(n=sample_size, random_state=random_seed)
    else:
        sampled_df = df.sample(n=sample_size)

    # Write the sampled DataFrame to CSV
    sampled_df.to_csv(new_file_name, index=False)

    print(
        f"Created random subset CSV with {sample_size + 1} rows at: "
        f"{new_file_name}"
    )
    return new_file_name


if __name__ == "__main__":
    create_subset_csv(
        csv_path=get_data_path() / "raw" / "87_10k_nwk.csv",
        n_rows=1000,
        random_seed=42,
    )
