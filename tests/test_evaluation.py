"""Unit tests for model evaluation."""
import pandas as pd
from sklearn.linear_model import LinearRegression
from evaluation.evaluation import ModelEvaluator


def test_evaluate_model_returns_metrics():
    x = pd.DataFrame({"study_hours": [1, 2, 3, 4]})
    y = pd.Series([50, 60, 70, 80])
    model = LinearRegression().fit(x, y)
    metrics = ModelEvaluator.evaluate_model(model, x, y)
    assert "MAE" in metrics
    assert "R2 Score" in metrics
