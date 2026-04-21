import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# -----------------------
# PATHS
# -----------------------
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed" / "ml_features"
REPORTS_DIR = ROOT / "reports" / "signals"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

DROP_COLS = ["Date", "target", "Open", "High", "Low", "Close", "stock"]


# -----------------------
# MODEL FACTORY
# -----------------------
def get_model(model_type: str):
    if model_type == "lr":
        return LogisticRegression(max_iter=1000, class_weight="balanced")

    elif model_type == "rf":
        return RandomForestRegressor(
            n_estimators=300,
            max_depth=10,
            min_samples_split=20,
            random_state=42,
            n_jobs=-1
        )

    elif model_type == "xgb":
        return XGBRegressor(
            n_estimators=600,
            max_depth=6,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1
        )

    else:
        raise ValueError(f"Unknown model_type: {model_type}")


# -----------------------
# CORE TRAIN LOOP
# -----------------------
def run_for_model(model_type: str):
    files = list(DATA_DIR.glob("xg_*.csv"))
    reports = []

    for file in files:
        df = pd.read_csv(file)
        df["stock"] = file.stem

        split = int(len(df) * 0.8)
        train_df = df.iloc[:split].copy()
        test_df = df.iloc[split:].copy()

        train_df = train_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["target"])
        test_df = test_df.replace([np.inf, -np.inf], np.nan).dropna()

        X_train = train_df.drop(columns=DROP_COLS).fillna(0)
        y_train = train_df["target"]
        X_test = test_df.drop(columns=DROP_COLS).fillna(0)
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

        # Scaling only for LR
        if model_type == "lr":
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        model = get_model(model_type)
        model.fit(X_train, y_train)

        pred = model.predict(X_test)

        # Direction logic
        direction_pred = (pred > 0.5).astype(int)
        direction_true = test_df["target"].astype(int)
        direction_accuracy = (direction_pred == direction_true).mean()

        # Save signals
        test_df["pred"] = pred
        test_df["position"] = pred / (np.std(pred) + 1e-9)

        out_path = REPORTS_DIR / f"{file.stem}_{model_type}_signals.csv"
        test_df.to_csv(out_path, index=False)

        reports.append({
            "stock": file.stem,
            "model": model_type.upper(),
            "num_rows": int(len(test_df)),
            "direction_accuracy": float(direction_accuracy) * 100,
            "pred_mean": float(np.mean(pred)) * 100,
            "pred_std": float(np.std(pred)) * 100
        })

    return reports


# -----------------------
# RUN ALL MODELS
# -----------------------
def train_and_generate_signals():
    all_reports = []

    for m in ["lr", "rf", "xgb"]:
        print(f"\nRunning model: {m.upper()}")
        reports = run_for_model(m)
        for r in reports:
            print(r)
        all_reports.extend(reports)

    return all_reports


if __name__ == "__main__":
    train_and_generate_signals()