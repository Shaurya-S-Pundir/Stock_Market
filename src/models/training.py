import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor

# -----------------------
# PATHS
# -----------------------
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed" / "ml_features"
REPORTS_DIR = ROOT / "reports" / "signals"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

files = list(DATA_DIR.glob("xg_*.csv"))

drop_cols = ["Date", "target", "Open", "High", "Low", "Close", "stock"]

MODEL_TYPE = "lr"   # "lr" or "rf"


# -----------------------
# MODEL SETUP
# -----------------------
def get_model():
    if MODEL_TYPE == "lr":
        return LogisticRegression(max_iter=1000, class_weight="balanced")
    else:
        return RandomForestRegressor(
            n_estimators=300,
            max_depth=10,
            min_samples_split=20,
            random_state=42,
            n_jobs=-1
        )


# -----------------------
# MAIN LOOP (PER STOCK)
# -----------------------
def train_and_generate_signals(model_type: str = "rf"):
    files = list(DATA_DIR.glob("xg_*.csv"))
    drop_cols = ["Date", "target", "Open", "High", "Low", "Close", "stock"]

    def get_model():
        if model_type == "lr":
            return LogisticRegression(max_iter=1000, class_weight="balanced")
        else:
            return RandomForestRegressor(
                n_estimators=300,
                max_depth=10,
                min_samples_split=20,
                random_state=42,
                n_jobs=-1
            )

    saved_files = []

    for file in files:
        df = pd.read_csv(file)
        df["stock"] = file.stem

        split = int(len(df) * 0.8)
        train_df = df.iloc[:split].copy()
        test_df = df.iloc[split:].copy()

        train_df = train_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["target"])
        test_df = test_df.replace([np.inf, -np.inf], np.nan).dropna()

        X_train = train_df.drop(columns=drop_cols).fillna(0)
        y_train = train_df["target"]
        X_test = test_df.drop(columns=drop_cols).fillna(0)
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

        scaler = None
        if model_type == "lr":
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        model = get_model()
        model.fit(X_train, y_train)

        pred = model.predict(X_test)
        # Convert prediction into direction (buy/sell)
        direction_pred = np.sign(pred)
        direction_true = np.sign(test_df["ret_1"])

        direction_accuracy = (direction_pred == direction_true).mean()

        pred_mean = float(np.mean(pred))
        pred_std = float(np.std(pred))

        test_df["pred"] = pred
        test_df["position"] = pred / (np.std(pred) + 1e-9)

        if "ret_1" not in test_df.columns:
            raise ValueError(f"ret_1 missing in {file.stem}")

        out_path = REPORTS_DIR / f"{file.stem}_{model_type}_signals.csv"
        test_df.to_csv(out_path, index=False)

        if MODEL_TYPE == "rf":
            model_used = "Random Forest"
        else: model_used = "Logistic Regression"

        report = {
            "stock": file.stem,
            "model": model_used,
            #"signals_path": str(out_path),
            "num_rows": int(len(test_df)),
            "direction_accuracy": float(direction_accuracy) * 100,
            "pred_mean": pred_mean * 100,
            "pred_std": pred_std * 100
        }

        saved_files.append(report)

    return saved_files