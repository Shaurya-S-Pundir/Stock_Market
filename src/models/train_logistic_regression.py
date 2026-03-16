import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# -----------------------------
# PATH
# -----------------------------
DATA_DIR = Path("data/processed/ml_features")

mainList = {}
# -----------------------------
# TRAIN FUNCTION
# -----------------------------
def train_model(file_path):

    print(f"\nTraining on {file_path.name}")
    print("-" * 40)

    df = pd.read_csv(file_path)

    # Drop columns not needed for ML
    X = df.drop(columns=[
    "Date",
    "target",
    "Open",
    "High",
    "Low",
    "Close"
    ])
    y = df["target"]

    # Time series split (80/20)
    split = int(len(df) * 0.8)

    X_train = X[:split]
    X_test = X[split:]

    y_train = y[:split]
    y_test = y[split:]

    # Scaling
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Model
    model = LogisticRegression(max_iter=1000)

    model.fit(X_train_scaled, y_train)

    # Predictions
    y_pred = model.predict(X_test_scaled)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    mainList[file_path.name] = accuracy
    print("Accuracy:", accuracy)
    print(classification_report(y_test, y_pred))

    # Feature importance
    importance = pd.Series(
        model.coef_[0],
        index=X.columns
    ).sort_values(key=abs, ascending=False)

    print("\nTop Features:")
    print(importance.head(10))


# -----------------------------
# MAIN LOOP
# -----------------------------
def main():

    from tqdm import tqdm


def main():

    files = list(DATA_DIR.glob("*.csv"))

    print(f"Found {len(files)} datasets\n")

    for file in tqdm(files, desc="Training Models"):
        train_model(file)
    print(mainList)

if __name__ == "__main__":
    main()