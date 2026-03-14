import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from pathlib import Path

INPUT_DIR = Path("data/raw/news_sentiment")
OUTPUT_DIR = Path("data/processed/news_sentiment")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

analyzer = SentimentIntensityAnalyzer()


def compute_sentiment(file_path):
    df = pd.read_csv(file_path)

    df["date"] = pd.to_datetime(df["date"])

    # Compute sentiment for each headline
    df["sentiment"] = df["headline"].apply(
        lambda x: analyzer.polarity_scores(str(x))["compound"]
    )

    # Aggregate sentiment per day
    daily_sentiment = (
        df.groupby(df["date"].dt.date)["sentiment"]
        .mean()
        .reset_index()
        .rename(columns={"date": "Date"})
    )

    return daily_sentiment


def main():
    files = list(INPUT_DIR.glob("*.csv"))

    for file in files:
        print(f"Processing sentiment → {file.name}")

        sentiment_df = compute_sentiment(file)

        output_file = OUTPUT_DIR / file.name.replace("_raw", "_sentiment")

        sentiment_df.to_csv(output_file, index=False)

        print(f"Saved → {output_file}")


if __name__ == "__main__":
    main()