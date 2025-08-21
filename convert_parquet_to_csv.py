import pandas as pd
import argparse
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def convert_parquet_to_csv(parquet_path):
    """
    Converts a Parquet file to a CSV file in the same directory.

    Args:
        parquet_path (str): The path to the input Parquet file.
    """
    if not os.path.exists(parquet_path):
        logging.error(f"File not found: {parquet_path}")
        return

    try:
        # Read the Parquet file
        df = pd.read_parquet(parquet_path)

        # Define the output CSV path
        directory = os.path.dirname(parquet_path)
        filename = os.path.splitext(os.path.basename(parquet_path))[0]
        csv_path = os.path.join(directory, f"{filename}.csv")

        # Convert to CSV
        df.to_csv(csv_path, index=False)
        logging.info(f"Successfully converted {parquet_path} to {csv_path}")

    except Exception as e:
        logging.error(f"An error occurred during conversion: {e}")

def main():
    parser = argparse.ArgumentParser(description='Convert a Parquet file to a CSV file.')
    parser.add_argument('parquet_file', type=str, help='The path to the Parquet file to convert.')
    
    args = parser.parse_args()
    
    convert_parquet_to_csv(args.parquet_file)

if __name__ == "__main__":
    main()