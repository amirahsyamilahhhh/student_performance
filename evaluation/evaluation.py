"""Evaluation metrics for regression models."""
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class ModelEvaluator:
    """Evaluate trained models using MAE, MSE, RMSE, and R2 score."""

    @staticmethod
    def evaluate_model(model, x_test, y_test) -> dict:
        """Evaluate one model and return metric results."""
        predictions = model.predict(x_test)
        mse = mean_squared_error(y_test, predictions)
        return {
            "MAE": mean_absolute_error(y_test, predictions),
            "MSE": mse,
            "RMSE": mse ** 0.5,
            "R2 Score": r2_score(y_test, predictions),
        }

    def evaluate_all_models(self, trained_models: dict, x_test, y_test) -> pd.DataFrame:
        """Evaluate multiple trained models and return a sorted result table."""
        results = []
        for model_name, model in trained_models.items():
            metrics = self.evaluate_model(model, x_test, y_test)
            metrics["Model"] = model_name
            results.append(metrics)

        return pd.DataFrame(results).sort_values(by="R2 Score", ascending=False)
