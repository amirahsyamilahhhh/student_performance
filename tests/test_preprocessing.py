"""Unit tests for preprocessing functions."""
import pandas as pd
from preprocessing.preprocessing import StudentPerformancePreprocessor


def test_clean_data_maps_yes_no_values():
    df = pd.DataFrame({
        "internet_access": ["yes", "no"],
        "extra_activities": ["no", "yes"],
        "overall_score": [88, 75],
    })
    cleaned = StudentPerformancePreprocessor.clean_data(df)
    assert cleaned["internet_access"].tolist() == [1, 0]
    assert cleaned["extra_activities"].tolist() == [0, 1]


def test_prepare_features_target_separates_target():
    df = pd.DataFrame({"study_hours": [2, 3], "overall_score": [70, 80]})
    preprocessor = StudentPerformancePreprocessor()
    x, y = preprocessor.prepare_features_target(df)
    assert "overall_score" not in x.columns
    assert y.tolist() == [70, 80]
