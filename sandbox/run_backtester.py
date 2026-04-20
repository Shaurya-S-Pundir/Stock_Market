import sys
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.backtest.backtester import run_backtester

# -----------------------
# BACKTEST PIPELINE ONLY
# -----------------------
def run_backtester_pipeline(run_id=None):

    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M")

    # -------------------------
    # CORE BACKTEST EXECUTION
    # -------------------------
    result = run_backtester(run_id=run_id)

    # -------------------------
    # RETURN ONLY RESULTS
    # -------------------------
    return {
        "run_id": run_id,
        "summary": result.get("summary", [])
    }


# -----------------------
# ENTRYPOINT (FOR N8N)
# -----------------------
if __name__ == "__main__":
    output = run_backtester_pipeline()

    print("\nPIPELINE COMPLETE")
    print(f"Run ID: {output['run_id']}")
    print(f"Rows: {len(output['rows'])}")
    print(f"Models: {len(output['model_summary'])}")
    print(f"Baselines: {len(output['baseline_summary'])}")