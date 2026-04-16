import sys
from pathlib import Path
import pandas as pd


# -----------------------
# PROJECT ROOT SETUP
# -----------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.backtest.backtester import backtest, random_baseline

# -----------------------
# SIGNALS DIRECTORY
# -----------------------
REPORTS_DIR = ROOT / "reports" / "signals"

# Safety check
if not REPORTS_DIR.exists():
    raise FileNotFoundError(f"Signals directory not found: {REPORTS_DIR}")

# -----------------------
# OPTIONAL: FILTER PATTERN
# -----------------------
FILE_PATTERN = "*_signals.csv"

# -----------------------
# RUN BACKTESTS
# -----------------------
def run_all_backtests():
    files = sorted(REPORTS_DIR.glob(FILE_PATTERN))

    if not files:
        raise FileNotFoundError(f"No signal files found in {REPORTS_DIR}")

    print(f"\nFound {len(files)} signal files")
    print("Starting backtests...\n")

    results = []

    for f in files:
        try:
            print(f"Running backtest: {f.name}")
            result = backtest(f)   # we will upgrade this later to return metrics
            results.append((f.name, result))
            real = backtest(f)
            random_eq, random_sharpe = random_baseline(pd.read_csv(f))

            print("\n--- EDGE TEST ---")
            print("Real Sharpe:", real)
            print("Random Sharpe:", random_sharpe)
        except Exception as e:
            print(f"[ERROR] Failed on {f.name}: {e}")

    print("\nAll backtests completed.")
    return results
    


# -----------------------
# MAIN
# -----------------------
if __name__ == "__main__":
    run_all_backtests()