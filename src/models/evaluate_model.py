import pandas as pd 
import numpy as np
from joblib import load
import json
from pathlib import Path

from sklearn.metrics import mean_squared_error, mean_absolute_error

X_train_scaled = pd.read_csv('data/processed/X_train_scaled.csv')
X_test_scaled = pd.read_csv('data/processed/X_test_scaled.csv')
y_train = pd.read_csv('data/processed/y_train.csv')
y_test = pd.read_csv('data/processed/y_test.csv')

def main(repo_path):
    model = load(repo_path / "models/trained_model.joblib")
    y_test_pred = model.predict(X_test_scaled)
    m_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    m_mae = mean_absolute_error(y_test, y_test_pred)

    metrics = {"rmse": m_rmse, "mae": m_mae}
    print(metrics)
    metrics_path = repo_path / "metrics/scores.json"
    metrics_path.write_text(json.dumps(metrics))
    print(f"Saved metrics to {metrics_path}")

if __name__ == "__main__":
    repo_path = Path(__file__).parent.parent.parent
    main(repo_path)