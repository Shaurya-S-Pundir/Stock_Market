import pandas as pd
from pathlib import Path

INPUT_DIR = Path("data/processed/combined")
OUTPUT_DIR = Path("data/processed/clean")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def clean_dataset(file_path):
    df = pd.read_csv(file_path)

    # Ensure Date column exists
    if "Date" not in df.columns:
        print(f"Skipping {file_path.name} (no Date column)")
        return

    # Convert date
    df["Date"] = pd.to_datetime(df["Date"])

    # Sort by date
    df = df.sort_values("Date")

    # Remove duplicates
    df = df.drop_duplicates(subset=["Date"])

    # Reset index
    df = df.reset_index(drop=True)

    # Forward fill missing values
    df = df.ffill()

    # Drop rows still containing NaN
    df = df.dropna()

    return df


def main():
    files = list(INPUT_DIR.glob("*.csv"))

    if not files:
        print("No datasets found")
        return

    for file in files:
        print(f"Cleaning {file.name}")

        cleaned = clean_dataset(file)

        if cleaned is None:
            continue

        output_file = OUTPUT_DIR / file.name
        cleaned.to_csv(output_file, index=False)

        print(f"Saved → {output_file}")


if __name__ == "__main__":
    main()