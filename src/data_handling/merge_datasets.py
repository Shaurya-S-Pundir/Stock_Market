import pandas as pd
from pathlib import Path

STOCK_DIR = Path("data/raw/stock_prices")
SENTIMENT_DIR = Path("data/processed/news_sentiment")
OUTPUT_DIR = Path("data/processed/merged")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def merge_data(stock_file):

    ticker = stock_file.stem

    sentiment_file = SENTIMENT_DIR / f"{ticker}_sentiment.csv"

    print(f"Merging → {ticker}")

    stock = pd.read_csv(stock_file, skiprows=2)
    sentiment = pd.read_csv(sentiment_file)

    stock["Date"] = pd.to_datetime(stock["Date"])
    sentiment["Date"] = pd.to_datetime(sentiment["Date"])
    print(stock.head())
    merged = pd.merge(stock, sentiment, on="Date", how="left")

    # forward fill sentiment
    merged["sentiment"] = merged["sentiment"].ffill().fillna(0)

    return merged


def main():

    files = list(STOCK_DIR.glob("*.csv"))

    for file in files:

        df = merge_data(file)

        output_file = OUTPUT_DIR / file.name

        df.to_csv(output_file, index=False)

        print(f"Saved → {output_file}")


if __name__ == "__main__":
    main()