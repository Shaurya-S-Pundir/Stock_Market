import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/processed/combined")

files = list(DATA_PATH.glob("*.csv"))

print("\nFound datasets:")
for f in files:
    print(f.name)

print("\n--- DATA VALIDATION REPORT ---\n")

for file in files:
    print(f"\nChecking {file.name}")
    print("-" * 40)

    df = pd.read_csv(file)

    print("Shape:", df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nMissing values:")
    print(df.isna().sum())

    print("\nDuplicate rows:", df.duplicated().sum())

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        print("\nDate range:", df["Date"].min(), "→", df["Date"].max())

    print("\nFirst rows:")
    print(df.head())

    print("\nLast rows:")
    print(df.tail())

print("\nValidation complete.")