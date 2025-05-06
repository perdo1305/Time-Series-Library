import os
import pandas as pd
import glob
import shutil


def separate_by_cycle(input_dir=".", output_dir="cycles"):
    """
    Separates battery data by cycle index and saves each cycle in a separate CSV file.

    Args:
        input_dir: Directory containing the CSV files
        output_dir: Directory where cycle data will be saved
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get all CSV files
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))

    # Filter out files that might already be in the output format
    csv_files = [f for f in csv_files if not os.path.basename(f).startswith("cycle_")]

    # Process each CSV file
    for csv_file in csv_files:
        print(f"Processing {csv_file}...")

        # Extract the base filename without extension
        base_name = os.path.basename(csv_file).split(".")[0]

        try:
            # Read the CSV file
            df = pd.read_csv(csv_file)

            # Check if 'Cycle_Index' column exists
            if "Cycle_Index" not in df.columns:
                print(
                    f"Warning: 'Cycle_Index' column not found in {csv_file}. Skipping file."
                )
                continue

            # Get unique cycle indices
            unique_cycles = df["Cycle_Index"].unique()

            # Create a directory for this file's cycles
            file_cycle_dir = os.path.join(output_dir, base_name)
            os.makedirs(file_cycle_dir, exist_ok=True)

            # Separate data by cycle index and save to individual files
            for cycle in unique_cycles:
                cycle_data = df[df["Cycle_Index"] == cycle]

                # Save to a new CSV file
                cycle_file = os.path.join(file_cycle_dir, f"cycle_{int(cycle)}.csv")
                cycle_data.to_csv(cycle_file, index=False)
                print(
                    f"  Saved cycle {int(cycle)} with {len(cycle_data)} data points to {cycle_file}"
                )

        except Exception as e:
            print(f"Error processing {csv_file}: {e}")

    print("Done!")


if __name__ == "__main__":
    separate_by_cycle()
