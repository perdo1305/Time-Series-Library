import os
import glob
import pandas as pd
import re
import numpy as np
from datetime import datetime


def organize_battery_data(
    input_dir=None, output_file=None, rated_capacity=1.1, correction_factor=0.95
):
    """
    Organize battery data CSV files:
    1. Sort files chronologically by the date in the 3rd column (Date_Time)
    2. Create a cumulative cycle index
    3. Add columns for corrected capacity metrics
    4. Add a new column for State of Charge (SOC) based on rated capacity
    5. Combine everything into a single output file

    Args:
        input_dir: Directory containing the CSV files
        output_file: Path for the output file
        rated_capacity: The rated capacity of the battery in Ah (default: 1.1 Ah)
        correction_factor: Factor to account for coulombic inefficiency (default: 0.95)
    """
    # Use current directory if none specified
    if input_dir is None:
        input_dir = os.path.dirname(os.path.abspath(__file__))

    if output_file is None:
        output_file = os.path.join(
            input_dir, "organized_battery_data_1st_cycles_removed.csv"
        )

    # Find all sheet2 CSV files in the data directory
    csv_files = glob.glob(os.path.join(input_dir, "Data", "*_sheet2.csv"))

    if not csv_files:
        print(f"No sheet2 CSV files found in {input_dir}")
        return

    # Read all CSV files to get their first date for sorting
    dated_files = []
    for csv_file in csv_files:
        try:
            # Only read first few rows to extract date
            df = pd.read_csv(csv_file, nrows=1)
            # Extract date from the 3rd column (Date_Time)
            if "Date_Time" in df.columns:
                date_str = df["Date_Time"].iloc[0]
                date = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
                dated_files.append((csv_file, date))
            else:
                print(f"Warning: Date_Time column not found in {csv_file}. Skipping.")
        except Exception as e:
            print(f"Error reading {csv_file}: {str(e)}")

    if not dated_files:
        print("No valid CSV files with dates were found. Exiting.")
        return

    # Sort files by date
    dated_files.sort(key=lambda x: x[1])

    print(f"Found {len(dated_files)} CSV files, sorted chronologically.")

    # Process files in chronological order
    combined_df = pd.DataFrame()
    max_cycle_index = 0  # Track max cycle index for cumulative calculation

    for i, (csv_file, date) in enumerate(dated_files):
        print(
            f"Processing file {i + 1}/{len(dated_files)}: {os.path.basename(csv_file)}"
        )

        try:
            # Read the CSV file
            df = pd.read_csv(csv_file)

            # Create new columns for reset capacities
            df["Reset_Charge_Capacity"] = 0.0
            df["Reset_Discharge_Capacity"] = 0.0

            # Initialize variables to track capacities
            current_charge = 0.0
            current_discharge = 0.0

            # Process rows to reset capacities at step index 9
            for j in range(len(df)):
                if j > 0 and df.loc[j, "Step_Index"] == 9:
                    # Reset the counters when step index is 9
                    current_charge = 0.0
                    current_discharge = 0.0

                # Increment based on differences from previous row
                if j > 0:
                    charge_diff = (
                        df.loc[j, "Charge_Capacity(Ah)"]
                        - df.loc[j - 1, "Charge_Capacity(Ah)"]
                    )
                    discharge_diff = (
                        df.loc[j, "Discharge_Capacity(Ah)"]
                        - df.loc[j - 1, "Discharge_Capacity(Ah)"]
                    )

                    if charge_diff > 0:
                        current_charge += charge_diff
                    if discharge_diff > 0:
                        current_discharge += discharge_diff

                # Store the current values
                df.loc[j, "Reset_Charge_Capacity"] = current_charge
                df.loc[j, "Reset_Discharge_Capacity"] = current_discharge

            # Add column for charge-discharge capacity difference
            df["Capacity_Difference"] = (
                df["Reset_Charge_Capacity"] - df["Reset_Discharge_Capacity"]
            )

            # remove unnecessary columns
            # data_point, test_time, source_file, and cycle_index
            df = df.drop(
                columns=[
                    "Data_Point",
                    # "Test_Time",
                    # "Source_File",
                    # "Cycle_Index",
                    "Is_FC_Data",
                    "ACI_Phase_Angle(Deg)",
                    "AC_Impedance(Ohm)",
                ],
                errors="ignore",
            )

            # Add source file information
            df["source_file"] = os.path.basename(csv_file)

            # Identify the first cycle number in this file
            first_cycle = df["Cycle_Index"].min()

            # Create cumulative cycle index by adding max_cycle_index to current Cycle_Index
            df["Cumulative_Cycle_Index"] = df["Cycle_Index"] + max_cycle_index

            # After processing, update max_cycle_index for next file
            # Still count the first cycle in the cumulative index even though we'll drop it
            file_max_cycle = df["Cycle_Index"].max()
            max_cycle_index += file_max_cycle

            # Drop the rows corresponding to the first cycle
            df = df[df["Cycle_Index"] > first_cycle]

            # For the first file, use it as the base
            if i == 0:
                combined_df = df
            else:
                # For subsequent files, append them
                combined_df = pd.concat([combined_df, df], ignore_index=True)

        except Exception as e:
            print(f"Error processing {csv_file}: {str(e)}")

    # Double-check that everything is sorted by date
    combined_df["Date_Time"] = pd.to_datetime(combined_df["Date_Time"])
    combined_df = combined_df.sort_values(by="Date_Time")

    # Save the combined dataframe
    combined_df.to_csv(output_file, index=False)
    print(f"Successfully processed all CSV files into {output_file}")
    print(f"Total number of data points: {len(combined_df)}")
    print(
        f"Date range: {combined_df['Date_Time'].min()} to {combined_df['Date_Time'].max()}"
    )
    print(f"Total cycles: {combined_df['Cumulative_Cycle_Index'].max()}")


if __name__ == "__main__":
    organize_battery_data()
