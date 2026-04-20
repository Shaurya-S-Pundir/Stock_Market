from fastapi import FastAPI
import sys
import os

# Fix imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.training import train_and_generate_signals
from sandbox.run_backtester import run_backtester_pipeline

app = FastAPI(title="Stock Market ML Experiment Service")


# -----------------------
# CORE PIPELINE
# -----------------------
def experiment_pipeline(mode: str = "train"):

    training_reports = train_and_generate_signals("rf")

    if mode == "train":
        return {
            "status": "success",
            "stage": "training",
            "mode": mode,
            "training_reports": training_reports
        }

    else:
        backtest_result = run_backtester_pipeline()

        return {
            "status": "success", 
            "stage": "experiment",
            "mode": mode,
            "backtest_result": backtest_result
        }

# -----------------------
# ROUTES
# -----------------------

@app.post("/train")
def train():
    return experiment_pipeline(mode="train")


@app.post("/run-experiment")
def run_experiment():
    return experiment_pipeline(mode="experiment")