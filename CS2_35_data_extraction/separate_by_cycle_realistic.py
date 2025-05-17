import os
import pandas as pd
import glob
import shutil


def separate_by_cycle_filtered(
    input_file="organized_battery_data.csv", output_dir="realistic_cycles"
):
    """
    Separates organized battery data by cycle index and saves each cycle in a separate CSV file.
    Only keeps Current(A), Voltage(V), Capacity_Difference, and Cumulative_Cycle_Index columns.

    Args:
        input_file: Path to the organized_battery_data.csv file
        output_dir: Directory where filtered cycle data will be saved
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

        # Filter to only keep the specified columns
        columns_to_keep = [
            "Current(A)",
            "Voltage(V)",
            "Capacity_Difference",
            cycle_column,
        ]

        # Check if all required columns exist
        missing_columns = [
            col
            for col in columns_to_keep
            if col != cycle_column and col not in df.columns
        ]
        if missing_columns:
            print(
                f"Warning: The following required columns are missing: {missing_columns}"
            )
            # Keep only the columns that exist
            columns_to_keep = [
                col
                for col in columns_to_keep
                if col in df.columns or col == cycle_column
            ]

        # Filter the dataframe to only include desired columns
        filtered_df = df[columns_to_keep]

        # Get unique cycle indices
        unique_cycles = filtered_df[cycle_column].unique()

        # Separate data by cycle index and save to individual files
        for cycle in unique_cycles:
            cycle_data = filtered_df[filtered_df[cycle_column] == cycle]

            # Keep the cycle column in the output (don't drop it)
            output_data = cycle_data

            # Save to a new CSV file with simple numbering
            cycle_file = os.path.join(output_dir, f"{int(cycle)}.csv")
            output_data.to_csv(cycle_file, index=False)
            print(
                f"  Saved filtered cycle {int(cycle)} with {len(cycle_data)} data points to {cycle_file}"
            )

        print(
            f"Successfully extracted the following columns: {', '.join(columns_to_keep)}"
        )

    except Exception as e:
        print(f"Error processing {input_file}: {e}")

    print("Done!")


if __name__ == "__main__":
    separate_by_cycle_filtered()
