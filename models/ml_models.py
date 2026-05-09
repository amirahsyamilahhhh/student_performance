"""Machine learning model definitions and training functions."""
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from config import RANDOM_STATE


class StudentPerformanceModelTrainer:
    """Train several regression models for predicting overall student score."""

    def __init__(self):
        self.models = {
            "Linear Regression": LinearRegression(),
            "Decision Tree": DecisionTreeRegressor(random_state=RANDOM_STATE),
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE),
        }

    def train_models(self, x_train, y_train) -> dict:
        """Train all selected machine learning models."""
        trained_models = {}
        for name, model in self.models.items():
            model.fit(x_train, y_train)
            trained_models[name] = model
        return trained_models

    @staticmethod
    def predict(model, x_test):
        """Generate prediction values from a trained model."""
        return model.predict(x_test)
