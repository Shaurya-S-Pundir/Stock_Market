import pandas as pd
import pandas_ta as ta
from pathlib import Path

INPUT_DIR = Path("data/processed/clean")
OUTPUT_DIR = Path("data/processed/features")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def add_indicators(file_path):
    df = pd.read_csv(file_path)

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    # Basic indicators
    df["SMA_20"] = ta.sma(df["Close"], length=20)
    df["SMA_50"] = ta.sma(df["Close"], length=50)

    df["EMA_12"] = ta.ema(df["Close"], length=12)
    df["EMA_26"] = ta.ema(df["Close"], length=26)

    # RSI
    df["RSI"] = ta.rsi(df["Close"], length=14)

    # MACD
    macd = ta.macd(df["Close"])
    df["MACD"] = macd["MACD_12_26_9"]
    df["MACD_SIGNAL"] = macd["MACDs_12_26_9"]

    # Bollinger Bands
   # Bollinger Bands
    bb = ta.bbands(df["Close"])

    df["BB_LOWER"] = bb.iloc[:, 0]
    df["BB_MIDDLE"] = bb.iloc[:, 1]
    df["BB_UPPER"] = bb.iloc[:, 2]

    return df


def main():
    files = list(INPUT_DIR.glob("*.csv"))

    for file in files:
        print(f"Adding indicators → {file.name}")

        df = add_indicators(file)

        output_file = OUTPUT_DIR / file.name
        df.to_csv(output_file, index=False)

        print(f"Saved → {output_file}")


if __name__ == "__main__":
    main()