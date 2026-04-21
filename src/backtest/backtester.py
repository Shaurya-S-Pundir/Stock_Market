import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = ROOT / "reports" / "signals"

MODEL_MAP = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost"
}


def compute_metrics(df):
    equity = df["equity"].iloc[-1] if len(df) else 1.0

    ret = df["strategy_ret"].replace([np.inf, -np.inf], 0).fillna(0)

    sharpe = (
        np.sqrt(252) * ret.mean() / (ret.std() + 1e-9)
        if ret.std() != 0 else 0.0
    )

    roll_max = df["equity"].cummax()
    drawdown = (df["equity"] - roll_max) / (roll_max + 1e-9)
    max_drawdown = drawdown.min() if len(df) else 0.0

    win_rate = float((ret > 0).mean()) if len(ret) else 0.0
    total_trades = int(df["trade"].sum()) if "trade" in df else 0

    return equity, sharpe, max_drawdown, win_rate, total_trades


def backtest(file_path, run_id, model_name):
    df = pd.read_csv(file_path).copy()

    # -------------------------
    # CLEAN CORE DATA
    # -------------------------
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # ensure required columns exist
    required_cols = ["position", "ret_1"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"{c} missing in {file_path}")

    stock = file_path.stem.replace("_signals", "")
    model_name = MODEL_MAP.get(model_name, model_name)

    # -------------------------
    # SORT SAFETY (VERY IMPORTANT)
    # -------------------------
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)

    # -------------------------
    # SHIFT POSITION (NO LOOKAHEAD)
    # -------------------------
    df["position"] = df["position"].shift(1).fillna(0)

    # -------------------------
    # STRATEGY RETURNS
    # -------------------------
    df["strategy_ret"] = df["position"] * df["ret_1"]

    # transaction cost
    df["trade"] = df["position"].diff().abs().fillna(0)
    df["strategy_ret"] -= df["trade"] * 0.0005

    # clean returns again
    df["strategy_ret"] = df["strategy_ret"].replace([np.inf, -np.inf], 0).fillna(0)

    # -------------------------
    # EQUITY CURVE
    # -------------------------
    if len(df) == 0:
        df["equity"] = 1.0
    else:
        df["equity"] = (1 + df["strategy_ret"]).cumprod()

    df["equity"] = df["equity"].replace([np.inf, -np.inf], np.nan)

    if df["equity"].isna().any():
        df["equity"] = df["equity"].fillna(1.0)
    # -------------------------
    # METRICS
    # -------------------------
    final_equity, sharpe, max_dd, win_rate, trades = compute_metrics(df)

    # -------------------------
    # DATE FIELD SAFE HANDLING
    # -------------------------
    if "Date" in df.columns:
        df["date"] = df["Date"]
    else:
        df["date"] = pd.NaT

    # -------------------------
    # METADATA
    # -------------------------
    df["run_id"] = run_id
    df["stock"] = stock
    df["model"] = model_name

    # -------------------------
    # SIGNAL OUTPUT
    # -------------------------
    signals = df[[
        "run_id",
        "stock",
        "model",
        "date",
        "position",
        "ret_1",
        "strategy_ret",
        "equity"
    ]].rename(columns={"ret_1": "returns"}).to_dict(orient="records")

    # -------------------------
    # SUMMARY OUTPUT (SAFE JSON)
    # -------------------------
    summary = {
        "run_id": run_id,
        "stock": stock,
        "model": model_name,
        "final_equity": float(final_equity) if np.isfinite(final_equity) else 1.0,
        "sharpe": float(sharpe) if np.isfinite(sharpe) else 0.0,
        "max_drawdown": float(max_dd) if np.isfinite(max_dd) else 0.0,
        "win_rate": float(win_rate) if np.isfinite(win_rate) else 0.0,
        "total_trades": int(trades)
    }

    return {
        "signals": signals,
        "summary": [summary]
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