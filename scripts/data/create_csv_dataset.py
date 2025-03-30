import os
import csv
import glob


def create_dataset_csv(classification_folder, output_file="tree_dataset.csv"):
    # Dictionary to map labels (subfolder names) to integers
    label_mapping: dict[int, str] = {}

    # Find all subfolders
    subfolders = [
        f
        for f in os.listdir(classification_folder)
        if os.path.isdir(os.path.join(classification_folder, f))
    ]

    # Create label mapping
    for i, subfolder in enumerate(sorted(subfolders)):
        label_mapping[i] = subfolder

    # Create CSV file
    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["nwk", "label"])  # Write header

        # Process each subfolder
        for label_int, subfolder_name in label_mapping.items():
            subfolder_path = os.path.join(
                classification_folder, subfolder_name
            )
            nwk_files = glob.glob(os.path.join(subfolder_path, "*.nwk"))

            # Add each tree file to the CSV
            for nwk_file in nwk_files:
                with open(nwk_file, "r") as f:
                    nwk_content = f.read().strip()
                writer.writerow([nwk_content, label_int])

    print(f"CSV file created at: {output_file}")
    print("Label mapping:")
    for label_int, label_name in label_mapping.items():
        print(f"{label_int}: {label_name}")

    return label_mapping


# Usage
if __name__ == "__main__":
    # Replace with your actual folder path
    from phylo_gnn import get_data_path, get_project_path

    data_path = get_data_path() / "raw" / "classification"
    print(get_project_path())
    num_tips = [87, 489, 674]
    for n in num_tips:
        classification_folder = data_path / f"{n}_10k_nwk"
        output_file = data_path / f"{n}_10k_nwk.csv"
        create_dataset_csv(classification_folder, output_file)
