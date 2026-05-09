"""Unit tests for machine learning model training."""
import pandas as pd
from models.ml_models import StudentPerformanceModelTrainer


def test_train_models_returns_trained_models():
    x_train = pd.DataFrame({"study_hours": [1, 2, 3, 4], "attendance_percentage": [70, 80, 90, 95]})
    y_train = pd.Series([60, 70, 80, 90])
    trainer = StudentPerformanceModelTrainer()
    trained_models = trainer.train_models(x_train, y_train)
    assert "Linear Regression" in trained_models
    assert "Random Forest" in trained_models
