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

    df["ret_1"] = df["Close"].pct_change()
    df["ret_5"] = df["Close"].pct_change(5)
    df["ret_10"] = df["Close"].pct_change(10)

    df["log_ret"] = np.log(df["Close"]).diff()

    # -------------------------
    # VOLATILITY
    # -------------------------

    df["vol_10"] = df["ret_1"].rolling(10).std()
    df["vol_20"] = df["ret_1"].rolling(20).std()

    df["vol_ratio"] = df["vol_10"] / (df["vol_20"] + 1e-6)

    # volatility-adjusted return
    df["scaled_ret_1"] = df["ret_1"] / (df["vol_10"] + 1e-6)

    # -------------------------
    # MOMENTUM (stationary version)
    # -------------------------

    df["mom_5"] = df["ret_5"]
    df["mom_10"] = df["ret_10"]

    df["mom_diff"] = df["mom_5"] - df["mom_10"]

    # -------------------------
    # MOVING AVERAGES (trend)
    # -------------------------

    df["ma_5"] = df["Close"].rolling(5).mean()
    df["ma_10"] = df["Close"].rolling(10).mean()
    df["ma_20"] = df["Close"].rolling(20).mean()

    df["ma_ratio_5_10"] = df["ma_5"] / df["ma_10"]
    df["ma_ratio_10_20"] = df["ma_10"] / df["ma_20"]

    df["trend_strength"] = (df["ma_5"] - df["ma_20"]) / df["Close"]

    # -------------------------
    # VOLUME FEATURES
    # -------------------------

    if "Volume" in df.columns:
        df["volume_chg"] = df["Volume"].pct_change()
        df["volume_ma_10"] = df["Volume"].rolling(10).mean()
        df["volume_ratio"] = df["Volume"] / (df["volume_ma_10"] + 1e-6)

    # -------------------------
    # LAG FEATURES (expanded)
    # -------------------------

    for lag in [1, 2, 3, 5, 7, 10, 14]:
        df[f"ret_lag_{lag}"] = df["ret_1"].shift(lag)

    # -------------------------
    # TIME FEATURES
    # -------------------------

    df["day_of_week"] = df["Date"].dt.dayofweek
    df["month"] = df["Date"].dt.month
    df["is_month_start"] = df["Date"].dt.is_month_start.astype(int)
    df["is_month_end"] = df["Date"].dt.is_month_end.astype(int)

    # -------------------------
    # TARGET (less noisy)
    # -------------------------

    future_ret = df["Close"].pct_change().shift(-1)
    df["target"] = (future_ret > 0.001).astype(int)

    # -------------------------
    # CLEANING
    # -------------------------

    df = df.replace([np.inf, -np.inf], np.nan)

    df = df.dropna().copy()

    # -------------------------
    # SAFETY CHECKS
    # -------------------------

    assert not df.isna().values.any(), "NaNs still present after cleaning"
    assert not np.isinf(df.select_dtypes(include=np.number)).values.any(), "Inf still present"

    return df


def main():
    files = list(INPUT_DIR.glob("*.csv"))
    print(f"Processing {len(files)} feature datasets")

    for file in files:
        df = add_ml_features(file)
        output_file = OUTPUT_DIR / f"xg_{file.name}"
        df.to_csv(output_file, index=False)
        print(f"Saved → {output_file}")


if __name__ == "__main__":
    main()