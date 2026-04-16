from src.models.train_XGBoost import train_xgboost
from src.models.train_random_forest import train_random_forest
from src.backtest.backtester import run_backtest
from src.utils.experiment_logger import log_experiment


def run_experiment(config):
    results = []

    for model in config["models"]:

        if model["model_name"] == "xgboost":
            trained_model = train_xgboost(model, config)

        elif model["model_name"] == "random_forest":
            trained_model = train_random_forest(model, config)

        backtest_result = run_backtest(trained_model, config)

        # 👇 extract metrics you already print today
        experiment_data = {
            "model_name": model["model_name"],
            "symbol": config["symbol"],
            "feature_set": config["feature_set"],

            "accuracy": backtest_result["accuracy"],
            "precision": backtest_result["precision"],
            "recall": backtest_result["recall"],
            "f1_score": backtest_result["f1"],

            "final_equity": backtest_result["final_equity"],
            "sharpe_ratio": backtest_result["sharpe"],
            "max_drawdown": backtest_result["max_drawdown"]
        }

        log_experiment(experiment_data)

        results.append({
            "model": model["model_name"],
            "backtest": backtest_result
        })

    return results