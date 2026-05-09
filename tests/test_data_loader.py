"""Unit tests for data loading utilities."""
import pandas as pd
from data.data_loader import StudentPerformanceDataLoader


def test_get_basic_summary_returns_correct_shape():
    df = pd.DataFrame({"age": [16, 17], "overall_score": [80, 90]})
    summary = StudentPerformanceDataLoader.get_basic_summary(df)
    assert summary["shape"] == (2, 2)
    assert summary["duplicate_rows"] == 0
