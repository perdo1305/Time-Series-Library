import os
import csv
import glob


def txt_to_csv(input_dir=None):
    """
    Convert all .txt files in input_dir to .csv files
    If input_dir is None, use the current directory
    """
    # Use current directory if none specified
    if input_dir is None:
        input_dir = os.path.dirname(os.path.abspath(__file__))

    # Find all .txt files
    txt_files = glob.glob(os.path.join(input_dir, "*.txt"))

    if not txt_files:
        print(f"No .txt files found in {input_dir}")
        return

    converted_count = 0
    for txt_file in txt_files:
        # Get the base filename without extension
        base_filename = os.path.basename(txt_file)
        name_without_ext = os.path.splitext(base_filename)[0]

        # Create output CSV filename
        csv_file = os.path.join(input_dir, f"{name_without_ext}.csv")

        try:
            # Convert the file
            with (
                open(txt_file, "r", errors="replace") as infile,
                open(csv_file, "w", newline="") as outfile,
            ):
                # Create CSV writer
                csv_writer = csv.writer(outfile)

                # Process each line
                for line in infile:
                    # Skip empty lines
                    if not line.strip():
                        continue

                    # Split by tabs
                    fields = line.strip().split("\t")

                    # Write to CSV
                    csv_writer.writerow(fields)

            print(f"Converted {txt_file} to {csv_file}")
            converted_count += 1
        except Exception as e:
            print(f"Error converting {txt_file}: {str(e)}")

    print(f"Conversion complete: {converted_count} files converted")


if __name__ == "__main__":
    txt_to_csv()
