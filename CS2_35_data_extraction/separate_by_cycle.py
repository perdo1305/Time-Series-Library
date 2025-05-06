import os
import pandas as pd
import glob
import shutil


def separate_by_cycle(input_file="organized_battery_data.csv", output_dir="cycles"):
    """
    Separates organized battery data by cycle index and saves each cycle in a separate CSV file.

    Args:
        input_file: Path to the organized_battery_data.csv file
        output_dir: Directory where cycle data will be saved
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # If input_file is a directory, assume it contains organized_battery_data.csv
    if os.path.isdir(input_file):
        input_file = os.path.join(input_file, "organized_battery_data.csv")

    print(f"Processing {input_file}...")

    try:
        # Read the CSV file
        df = pd.read_csv(input_file)

        # Drop the Test_Time column if it exists
        if "Test_Time(s)" in df.columns:
            df = df.drop(columns=["Test_Time(s)"])
            print("  Dropped 'Test_Time(s)' column from the data")
        elif "Test_Time" in df.columns:
            df = df.drop(columns=["Test_Time"])
            print("  Dropped 'Test_Time' column from the data")

        # Continue with dropping other columns

        # if "Cycle_Index" in df.columns:
        #   df = df.drop(columns=["Cycle_Index"])
        #    print("  Dropped 'Cycle_Index' column from the data")
        if "Date_Time" in df.columns:
            df = df.drop(columns=["Date_Time"])
            print("  Dropped 'Date_Time' column from the data")
        if "source_file" in df.columns:
            df = df.drop(columns=["source_file"])
            print("  Dropped 'source_file' column from the data")

        # Check if 'Cumulative_Cycle_Index' column exists
        cycle_column = (
            "Cumulative_Cycle_Index"
            if "Cumulative_Cycle_Index" in df.columns
            else "Cycle_Index"
        )
        if cycle_column not in df.columns:
            print(
                f"Warning: Neither 'Cumulative_Cycle_Index' nor 'Cycle_Index' column found in {input_file}. Exiting."
            )
            return

        # Get unique cycle indices
        unique_cycles = df[cycle_column].unique()

        # Separate data by cycle index and save to individual files
        for cycle in unique_cycles:
            cycle_data = df[df[cycle_column] == cycle]

            # Save to a new CSV file with simple numbering
            cycle_file = os.path.join(output_dir, f"{int(cycle)}.csv")
            cycle_data.to_csv(cycle_file, index=False)
            print(
                f"  Saved cycle {int(cycle)} with {len(cycle_data)} data points to {cycle_file}"
            )

    except Exception as e:
        print(f"Error processing {input_file}: {e}")

    print("Done!")


if __name__ == "__main__":
    separate_by_cycle()
