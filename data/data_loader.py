"""Functions for loading and inspecting the student performance dataset."""
from pathlib import Path
import pandas as pd

from config import KAGGLE_DATA_PATH, LOCAL_DATA_PATH


class StudentPerformanceDataLoader:
    """Load the Student Performance dataset from Kaggle or a local folder."""

    def __init__(self, data_path: str | Path | None = None):
        self.data_path = Path(data_path) if data_path else self._find_default_path()

    @staticmethod
    def _find_default_path() -> Path:
        """Return the available default dataset path."""
        if LOCAL_DATA_PATH.exists():
            return LOCAL_DATA_PATH
        if KAGGLE_DATA_PATH.exists():
            return KAGGLE_DATA_PATH
        raise FileNotFoundError(
            "Dataset not found. Place Student_Performance.csv inside the data folder "
            "or run this project in Kaggle with the dataset attached."
        )

    def load_data(self) -> pd.DataFrame:
        """Load the CSV file into a pandas DataFrame."""
        return pd.read_csv(self.data_path)

    @staticmethod
    def get_basic_summary(df: pd.DataFrame) -> dict:
        """Return basic information about the dataset."""
        return {
            "shape": df.shape,
            "columns": list(df.columns),
            "missing_values": df.isnull().sum().to_dict(),
            "duplicate_rows": int(df.duplicated().sum()),
        }
