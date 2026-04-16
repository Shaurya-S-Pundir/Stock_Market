from fastapi import FastAPI
from experiment_service.app.pipeline_runner import run_experiment

app = FastAPI()

@app.post("/experiment/create-run")
def create_run(config: dict):
    result = run_experiment(config)
    return {
        "status": "success",
        "result": result
    }