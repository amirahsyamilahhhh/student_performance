"""Configuration settings for the Student Performance project."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# The program checks the local path first, then the Kaggle dataset path.
LOCAL_DATA_PATH = BASE_DIR / "data" / "Student_Performance.csv"
KAGGLE_DATA_PATH = Path("/kaggle/input/student-performance-dataset/Student_Performance.csv")

TARGET_COLUMN = "overall_score"
RANDOM_STATE = 42
TEST_SIZE = 0.2

REPORT_DIR = BASE_DIR / "reports"
FIGURE_DIR = REPORT_DIR / "figures"
