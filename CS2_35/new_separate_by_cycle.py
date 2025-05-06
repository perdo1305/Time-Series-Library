#!/usr/bin/env python3
import os
import pandas as pd
import argparse


def process_file_by_cycle(input_file, output_dir=None):
    """
    Process a battery data file to split into individual cycle files.
    Each file contains cumulative charge/discharge capacities for that cycle.

    Args:
        input_file (str): Path to the input CSV file
        output_dir (str, optional): Directory to save output files. If None, use same directory as input file.
    """
    # Read the input file
    print(f"Reading input file: {input_file}")
    try:
        data = pd.read_csv(input_file)
        print(f"Successfully loaded data with {len(data)} rows")
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # If output directory not specified, use the same directory as input file
    if output_dir is None:
        output_dir = os.path.dirname(input_file)
        if not output_dir:
            output_dir = "."

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get unique cycle indices
    cycles = data["Cycle_Index"].unique()
    cycles.sort()

    print(f"Found {len(cycles)} unique cycles")

    # Process each cycle
    for cycle in cycles:
        # Filter data for this cycle
        cycle_data = data[data["Cycle_Index"] == cycle].copy()

        # Save to a new CSV file named by cycle number
        output_file = os.path.join(output_dir, f"{int(cycle)}.csv")
        cycle_data.to_csv(output_file, index=False)
        print(f"Created cycle file: {output_file} with {len(cycle_data)} rows")


def main():
    parser = argparse.ArgumentParser(
        description="Split battery data by cycle into separate CSV files"
    )
    parser.add_argument("input_file", help="Input CSV file containing battery data")
    parser.add_argument(
        "--output-dir",
        help="Directory to save output files (default: same as input file)",
    )

    args = parser.parse_args()

    process_file_by_cycle(args.input_file, args.output_dir)


if __name__ == "__main__":
    main()
