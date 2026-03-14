import pandas as pd
import pandas_ta as ta
from pathlib import Path

INPUT_DIR = Path("data/processed/clean")
OUTPUT_DIR = Path("data/processed/features")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def add_indicators(file_path):

    print(f"Adding indicators → {file_path.name}")

    df = pd.read_csv(file_path)

    # RSI
    df["RSI"] = ta.rsi(df["Close"], length=14)

    # Moving averages
    df["SMA_20"] = ta.sma(df["Close"], length=20)
    df["EMA_20"] = ta.ema(df["Close"], length=20)

    # MACD
    macd = ta.macd(df["Close"])
    df["MACD"] = macd["MACD_12_26_9"]
    df["MACD_SIGNAL"] = macd["MACDs_12_26_9"]

    # Bollinger Bands
    bb = ta.bbands(df["Close"], length=20)

    df["BB_LOWER"] = bb.iloc[:, 0]
    df["BB_MIDDLE"] = bb.iloc[:, 1]
    df["BB_UPPER"] = bb.iloc[:, 2]

    return df


def main():

    files = list(INPUT_DIR.glob("*.csv"))

    print(f"Processing {len(files)} cleaned datasets")

    for file in files:

        df = add_indicators(file)

        output_file = OUTPUT_DIR / file.name

        df.to_csv(output_file, index=False)

        print(f"Saved → {output_file}")


if __name__ == "__main__":
    main()