import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = ROOT / "reports" / "signals"


# -----------------------------
# MODEL MAP
# -----------------------------
MODEL_MAP = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost"
}


# -----------------------------
# METRICS
# -----------------------------
def compute_metrics(df):
    equity = df["equity"].iloc[-1]

    sharpe = np.sqrt(252) * df["strategy_ret"].mean() / (df["strategy_ret"].std() + 1e-9)

    roll_max = df["equity"].cummax()
    drawdown = (df["equity"] - roll_max) / roll_max
    max_drawdown = drawdown.min()

    win_rate = float((df["strategy_ret"] > 0).mean())
    total_trades = int(df["trade"].sum())

    return equity, sharpe, max_drawdown, win_rate, total_trades


# -----------------------------
# BACKTEST CORE
# -----------------------------
def backtest(file_path, run_id, model_name):

    df = pd.read_csv(file_path).copy()

    stock = file_path.stem.replace("_signals", "")
    model_name = MODEL_MAP.get(model_name, model_name)

    if "ret_1" not in df.columns:
        raise ValueError(f"Missing ret_1 in {file_path}")

    # -------------------------
    # SIGNAL PROCESSING
    # -------------------------
    df["position"] = df["position"].shift(1).fillna(0)

    df["strategy_ret"] = df["position"] * df["ret_1"]

    df["trade"] = df["position"].diff().abs().fillna(0)
    df["strategy_ret"] -= df["trade"] * 0.0005

    df["equity"] = (1 + df["strategy_ret"]).cumprod()

    # -------------------------
    # METRICS
    # -------------------------
    final_equity, sharpe, max_dd, win_rate, trades = compute_metrics(df)

    # -------------------------
    # DATE HANDLING
    # -------------------------
    raw = pd.read_csv(file_path)

    if "date" in raw.columns:
        df["date"] = pd.to_datetime(raw["date"])
    else:
        df["date"] = pd.NaT

    # -------------------------
    # ADD METADATA
    # -------------------------
    df["run_id"] = run_id
    df["stock"] = stock
    df["model"] = model_name

    # -------------------------
    # SIGNAL OUTPUT (ROW LEVEL)
    # -------------------------
    signals = df[
        [
            "run_id",
            "stock",
            "model",
            "date",
            "position",
            "ret_1",
            "strategy_ret",
            "equity",
        ]
    ].rename(columns={"ret_1": "returns"}).to_dict(orient="records")

    # -------------------------
    # SUMMARY OUTPUT
    # -------------------------
    summary = {
        "run_id": run_id,
        "stock": stock,
        "model": model_name,
        "final_equity": float(final_equity),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_dd),
        "win_rate": float(win_rate),
        "total_trades": int(trades),
    }

    return {
        "signals": signals,
        "summary": [summary],
    }


# -----------------------------
# RUNNER (MULTI MODEL + STOCK)
# -----------------------------
def run_backtester(run_id=None, model_names=None):
    print("RUN_BACKTESTER CALLED")
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M")

    if model_names is None:
        model_names = ["xgboost", "logistic_regression", "random_forest"]

    all_signals = []
    all_summary = []

    for model_name in model_names:
        for file in REPORTS_DIR.glob("*signals.csv"):

            result = backtest(file, run_id, model_name)

            all_signals.extend(result["signals"])
            all_summary.extend(result["summary"])

    return {
        "run_id": run_id,
        "signals": all_signals,
        "summary": all_summary
    }


# -----------------------------
# ENTRYPOINT (n8n)
# -----------------------------
if __name__ == "__main__":
    import json
    import sys

    output = run_backtester()

    sys.stdout.write(json.dumps(output))
    sys.stdout.flush()