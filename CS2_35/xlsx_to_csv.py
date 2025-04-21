import os
import glob
import pandas as pd
import re
from datetime import datetime


def extract_date_from_filename(filename):
    """Try to extract date from filename using regex"""
    # Look for common date patterns in the filename
    date_pattern = re.compile(
        r"(\d{4}[-_/]?\d{1,2}[-_/]?\d{1,2})|(\d{1,2}[-_/]?\d{1,2}[-_/]?\d{4})"
    )
    match = date_pattern.search(os.path.basename(filename))

    if match:
        try:
            date_str = match.group(0)
            # Try different date formats
            for fmt in [
                "%Y-%m-%d",
                "%Y_%m_%d",
                "%Y/%m/%d",
                "%d-%m-%Y",
                "%d_%m_%Y",
                "%d/%m/%Y",
            ]:
                try:
                    return datetime.strptime(date_str, fmt)
                except ValueError:
                    continue
        except Exception:
            pass
    return None


def extract_date_from_content(csv_path):
    """Try to extract date from CSV content"""
    try:
        df = pd.read_csv(csv_path)
        # Look for date columns - common names for date columns
        date_columns = [
            col
            for col in df.columns
            if "date" in col.lower() or "time" in col.lower() or "day" in col.lower()
        ]

        if date_columns and not df[date_columns[0]].empty:
            # Use first date column found
            date_val = df[date_columns[0]].iloc[0]
            if isinstance(date_val, str):
                # Try different date formats
                for fmt in ["%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%d/%m/%Y"]:
                    try:
                        return datetime.strptime(date_val, fmt)
                    except ValueError:
                        continue
    except Exception:
        pass
    return None


def get_file_date(csv_path):
    """Get date from file - try multiple methods"""
    # Try extract from content
    date = extract_date_from_content(csv_path)
    if date:
        return date

    # Try extract from filename
    date = extract_date_from_filename(csv_path)
    if date:
        return date

    # Fall back to file modification time
    return datetime.fromtimestamp(os.path.getmtime(csv_path))


def xlsx_to_csv_and_combine(input_dir=None, output_file=None):
    """
    1. Convert second sheet of all .xlsx files to individual CSV files
    2. Combine all CSVs into one file, ordered by date
    """
    # Use current directory if none specified
    if input_dir is None:
        input_dir = os.path.dirname(os.path.abspath(__file__))

    if output_file is None:
        output_file = os.path.join(input_dir, "combined_data.csv")

    # Find all .xlsx files
    xlsx_files = glob.glob(os.path.join(input_dir, "*.xlsx"))

    if not xlsx_files:
        print(f"No .xlsx files found in {input_dir}")
        return

    temp_csv_files = []

    # Step 1: Convert second sheet of each Excel file to CSV
    for xlsx_file in xlsx_files:
        # Get the base filename without extension
        base_filename = os.path.basename(xlsx_file)
        name_without_ext = os.path.splitext(base_filename)[0]

        # Create output CSV filename
        csv_file = os.path.join(input_dir, f"{name_without_ext}_sheet2.csv")

        try:
            # Read the Excel file, specifically the second sheet
            # Note: Excel sheets are 0-indexed in pandas
            xl = pd.ExcelFile(xlsx_file)

            # Check if there are at least 2 sheets
            if len(xl.sheet_names) < 2:
                print(f"Warning: {xlsx_file} doesn't have a second sheet. Skipping.")
                continue

            # Get the name of the second sheet
            second_sheet_name = xl.sheet_names[1]

            # Read the second sheet
            df = pd.read_excel(xlsx_file, sheet_name=second_sheet_name)

            # Save as CSV
            df.to_csv(csv_file, index=False)

            print(f"Converted second sheet of {xlsx_file} to {csv_file}")
            temp_csv_files.append(csv_file)
        except Exception as e:
            print(f"Error converting {xlsx_file}: {str(e)}")

    if not temp_csv_files:
        print("No CSV files were created. Exiting.")
        return

    # Step 2: Combine all CSVs into one, ordered by date
    try:
        # Create a list of (file_path, date) tuples
        dated_files = []
        for csv_file in temp_csv_files:
            date = get_file_date(csv_file)
            dated_files.append((csv_file, date))

        # Sort by date
        dated_files.sort(key=lambda x: x[1])

        # Combine the CSV files
        combined_df = pd.DataFrame()

        for i, (csv_file, date) in enumerate(dated_files):
            df = pd.read_csv(csv_file)

            # Add a source column to track which file the data came from
            df["source_file"] = os.path.basename(csv_file)
            df["file_date"] = date

            # For the first file, use it as the base
            if i == 0:
                combined_df = df
            else:
                # For subsequent files, append them
                combined_df = pd.concat([combined_df, df], ignore_index=True)

        # Save the combined dataframe
        combined_df.to_csv(output_file, index=False)
        print(f"Successfully combined all CSV files into {output_file}")

    except Exception as e:
        print(f"Error combining CSV files: {str(e)}")


if __name__ == "__main__":
    xlsx_to_csv_and_combine()
