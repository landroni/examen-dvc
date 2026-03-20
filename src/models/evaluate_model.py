import pandas as pd 
import numpy as np
from joblib import load
import json
from pathlib import Path
import os

from sklearn.metrics import mean_squared_error, mean_absolute_error

X_train_scaled = pd.read_csv('data/scaled/X_train_scaled.csv')
X_test_scaled = pd.read_csv('data/scaled/X_test_scaled.csv')
y_train = pd.read_csv('data/processed/y_train.csv')
y_test = pd.read_csv('data/processed/y_test.csv')

def main(repo_path):
    model = load(repo_path / "models/trained_model.joblib")
    y_test_pred = model.predict(X_test_scaled)
    m_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    m_mae = mean_absolute_error(y_test, y_test_pred)
    metrics = {"rmse": m_rmse, "mae": m_mae}
    print(metrics)

    ##save metrics
    metrics_path = repo_path / "metrics"
    os.makedirs(metrics_path, exist_ok=True)
    metrics_file = metrics_path / "scores.json"
    metrics_file.write_text(json.dumps(metrics))
    print(f"Saved metrics to {metrics_file}")

    ##save predictions
    predictions_path = repo_path / "data/predictions"
    os.makedirs(predictions_path, exist_ok=True)
    predictions_file = predictions_path / "predictions.csv"
    print(pd.DataFrame(y_test_pred))
    pd.DataFrame(y_test_pred).to_csv(predictions_file)
    print(f"Saved predictions to {predictions_file}")


if __name__ == "__main__":
    repo_path = Path(__file__).parent.parent.parent
    main(repo_path)