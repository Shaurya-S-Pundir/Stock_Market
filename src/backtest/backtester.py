import pandas as pd
from pathlib import Path


def backtest_from_signals(signals_csv):
    df = pd.read_csv(signals_csv, parse_dates=["Date"])

    results = []

    for stock in df["stock"].unique():
        stock_df = df[df["stock"] == stock].sort_values("Date").copy()

        stock_df["position"] = stock_df["signal"].map({"BUY": 1, "SELL": -1})

        # FIX 1 — ensure no NaN positions
        stock_df["position"] = stock_df["position"].fillna(0)

        # Daily returns
        stock_df["market_return"] = stock_df["Close"].pct_change()

        # Strategy return
        stock_df["strategy_return"] = stock_df["position"].shift(1) * stock_df["market_return"]

        # FIX 2 — remove NaN PnL rows correctly
        stock_df["strategy_return"] = stock_df["strategy_return"].fillna(0)

        # Equity
        stock_df["equity_curve"] = (1 + stock_df["strategy_return"]).cumprod()

        final_equity = stock_df["equity_curve"].iloc[-1]

        print(f"{stock} Final Equity: {final_equity:.3f}")
        results.append(final_equity)
        #print(stock, stock_df["position"].value_counts())

    print("\nAverage Equity Across Stocks:", sum(results) / len(results))