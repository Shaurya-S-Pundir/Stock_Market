import pandas as pd
import numpy as np
from pathlib import Path

INPUT_DIR = Path("data/processed/features")
OUTPUT_DIR = Path("data/processed/ml_features")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def add_ml_features(file_path):
    print(f"Feature Engineering → {file_path.name}")

    df = pd.read_csv(file_path)

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # -------------------------
    # RETURNS
    # -------------------------

    df["returns"] = df["Close"].pct_change()

    # SAFE log return (prevents inf)
    df["log_returns"] = np.log(df["Close"]).diff()

    # -------------------------
    # VOLATILITY
    # -------------------------

    df["volatility_7"] = df["returns"].rolling(7).std()
    df["volatility_14"] = df["returns"].rolling(14).std()

    # -------------------------
    # MOMENTUM
    # -------------------------

    df["momentum_5"] = df["Close"].diff(5)
    df["momentum_10"] = df["Close"].diff(10)

    # -------------------------
    # LAG FEATURES
    # -------------------------

    df["close_lag_1"] = df["Close"].shift(1)
    df["close_lag_3"] = df["Close"].shift(3)
    df["close_lag_5"] = df["Close"].shift(5)

    df["returns_lag_1"] = df["returns"].shift(1)
    df["returns_lag_3"] = df["returns"].shift(3)

    # -------------------------
    # TARGET
    # -------------------------

    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    # -------------------------
    # HARD CLEAN (REAL FIX)
    # -------------------------

    # Replace inf first (critical)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Single authoritative drop
    df = df.dropna().copy()

    # -------------------------
    # DATA INTEGRITY CHECK (never remove this)
    # -------------------------

    assert not df.isna().values.any(), "NaNs still present after cleaning"
    assert not np.isinf(df.select_dtypes(include=np.number)).values.any(), "Inf still present"

    return df


def main():
    files = list(INPUT_DIR.glob("*.csv"))
    print(f"Processing {len(files)} feature datasets")

    for file in files:
        df = add_ml_features(file)
        output_file = OUTPUT_DIR / file.name
        df.to_csv(output_file, index=False)
        print(f"Saved → {output_file}")


if __name__ == "__main__":
    main()