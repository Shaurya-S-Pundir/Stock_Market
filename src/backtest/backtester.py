import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = ROOT / "reports" / "signals"

def backtest(file):

    df = pd.read_csv(file)

    df["position"] = df["position"]
    df["position"] = df["position"].shift(1).fillna(0)

    # shift to avoid lookahead
    df["position"] = df["position"].shift(1).fillna(0)

    df["strategy_ret"] = df["position"] * df["ret_1"]

    df["trade"] = df["position"].diff().abs().fillna(0)
    df["strategy_ret"] -= df["trade"] * 0.0005

    df["equity"] = (1 + df["strategy_ret"]).cumprod()

    final_equity = df["equity"].iloc[-1]
    sharpe = np.sqrt(252) * df["strategy_ret"].mean() / (df["strategy_ret"].std() + 1e-9)

    print(f"\n{file.stem}")
    print("Equity:", final_equity)
    print("Sharpe:", sharpe)

    out = file.with_name(file.stem + "_backtest.csv")
    df.to_csv(out, index=False)

def random_baseline(df):
    df = df.copy()

    df["position"] = np.random.choice([0, 1], size=len(df))

    df["position"] = df["position"].shift(1).fillna(0)

    df["strategy_ret"] = df["position"] * df["ret_1"]

    equity = (1 + df["strategy_ret"]).cumprod().iloc[-1]

    sharpe = np.sqrt(252) * df["strategy_ret"].mean() / (df["strategy_ret"].std() + 1e-9)

    return equity, sharpe

if __name__ == "__main__":
    for f in REPORTS_DIR.glob("*signals.csv"):
        backtest(f)
