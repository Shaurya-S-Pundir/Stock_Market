import psycopg2

conn = psycopg2.connect(
    dbname="stock_market",
    user="shaurya",
    password="[PASSWORD]",
    host="localhost",
    port="5432"
)

def log_experiment(data):
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO experiment_runs (
            model_name,
            symbol,
            feature_set,
            accuracy,
            precision,
            recall,
            f1_score,
            final_equity,
            sharpe_ratio,
            max_drawdown
        )
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s);
    """, (
        data["model_name"],
        data["symbol"],
        data["feature_set"],
        data["accuracy"],
        data["precision"],
        data["recall"],
        data["f1_score"],
        data["final_equity"],
        data["sharpe_ratio"],
        data["max_drawdown"]
    ))
    conn.commit()