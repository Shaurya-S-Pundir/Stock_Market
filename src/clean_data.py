import pandas as pd
from pathlib import Path

INPUT_DIR = Path("data/raw/stock_prices")
OUTPUT_DIR = Path("data/processed/clean")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def clean_dataset(file_path):

    print(f"Cleaning → {file_path.name}")

    # Skip Yahoo metadata rows
    df = pd.read_csv(file_path, skiprows=2)

    # Standardize column names
    df.columns = ["Date", "Close", "High", "Low", "Open", "Volume"]

    # Convert date
    df["Date"] = pd.to_datetime(df["Date"])

    # Sort
    df = df.sort_values("Date")

    # Remove duplicates
    df = df.drop_duplicates()

    # Handle missing values
    df = df.ffill().bfill()

    return df


def main():

    files = list(INPUT_DIR.glob("*.csv"))

    print(f"Found {len(files)} stock files")

    for file in files:

        cleaned = clean_dataset(file)

        output_file = OUTPUT_DIR / file.name

        cleaned.to_csv(output_file, index=False)

        print(f"Saved → {output_file}")


if __name__ == "__main__":
    main()