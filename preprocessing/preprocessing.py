"""Preprocessing functions for cleaning and preparing data for modelling."""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from config import RANDOM_STATE, TARGET_COLUMN, TEST_SIZE


class StudentPerformancePreprocessor:
    """Clean data, encode categorical columns, and split features and target."""

    def __init__(self, target_column: str = TARGET_COLUMN):
        self.target_column = target_column
        self.label_encoders: dict[str, LabelEncoder] = {}

    @staticmethod
    def clean_data(df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate records and standardise yes/no columns."""
        clean_df = df.copy().drop_duplicates()

        yes_no_columns = ["internet_access", "extra_activities"]
        for column in yes_no_columns:
            if column in clean_df.columns:
                clean_df[column] = clean_df[column].map({"yes": 1, "no": 0}).fillna(clean_df[column])

        return clean_df

    def encode_categorical_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode text-based categorical columns using LabelEncoder."""
        encoded_df = df.copy()
        categorical_columns = encoded_df.select_dtypes(include=["object"]).columns

        for column in categorical_columns:
            encoder = LabelEncoder()
            encoded_df[column] = encoder.fit_transform(encoded_df[column].astype(str))
            self.label_encoders[column] = encoder

        return encoded_df

    def prepare_features_target(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Separate independent variables and the target variable."""
        if self.target_column not in df.columns:
            raise KeyError(f"Target column '{self.target_column}' was not found in the dataset.")

        x = df.drop(columns=[self.target_column])
        y = df[self.target_column]
        return x, y

    @staticmethod
    def split_data(x: pd.DataFrame, y: pd.Series):
        """Split data into training and testing sets."""
        return train_test_split(x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    def preprocess(self, df: pd.DataFrame):
        """Run the complete preprocessing workflow."""
        clean_df = self.clean_data(df)
        encoded_df = self.encode_categorical_columns(clean_df)
        x, y = self.prepare_features_target(encoded_df)
        return self.split_data(x, y)
