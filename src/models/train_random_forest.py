import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# -----------------------------
# PATH SETUP
# -----------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed" / "ml_features"
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

files = list(DATA_DIR.glob("*.csv"))

print("Found files:", len(files))

# -----------------------------
# TRAIN / TEST SPLIT PER STOCK
# -----------------------------
train_dfs = []
test_dfs = []

for file in files:
    df = pd.read_csv(file)
    df["stock"] = file.stem

    split = int(len(df) * 0.8)
    train_dfs.append(df.iloc[:split])
    test_dfs.append(df.iloc[split:])

train_data = pd.concat(train_dfs, ignore_index=True)
test_data = pd.concat(test_dfs, ignore_index=True)

print("Train shape:", train_data.shape)
print("Test shape:", test_data.shape)

# -----------------------------
# PREPARE X AND y
# -----------------------------
drop_cols = ["Date", "target", "Open", "High", "Low", "Close", "stock"]

X_train = train_data.drop(columns=drop_cols)
y_train = train_data["target"]

X_test = test_data.drop(columns=drop_cols)
y_test = test_data["target"]

# -----------------------------
# SCALING
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# MODEL
# -----------------------------
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_split=20,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train_scaled, y_train)


# -----------------------------
# EVALUATION
# -----------------------------
y_pred = model.predict(X_test_scaled)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# -----------------------------
# FEATURE IMPORTANCE
# -----------------------------
importance = pd.Series(
    model.coef_[0],
    index=X_train.columns
).sort_values(key=abs, ascending=False)

print("\nTop 15 Important Features:")
print(importance.head(15))

# -----------------------------
# PROBABILITY → SIGNAL
# -----------------------------
probs = model.predict_proba(X_test_scaled)[:, 1]
test_data["prob_up"] = probs

test_data["signal"] = np.where(
    test_data["prob_up"] > 0.5,
    "BUY",
    "SELL"
)

# -----------------------------
# SAVE SIGNALS FOR BACKTEST
# -----------------------------
out_path = REPORTS_DIR / "rf_signals.csv"
test_data.to_csv(out_path, index=False)

print(f"\nSignals saved to: {out_path}")