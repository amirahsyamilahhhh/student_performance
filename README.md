# student_performance
# Student Performance Machine Learning Project

This project restructures a Kaggle-style student performance analysis notebook into a modular Python project. The coursework requires code to be separated into clear machine learning workflow components such as data loading, preprocessing, model training, evaluation, visualization, utility functions, and testing.

## Project Structure

```text
student_performance_project/
├── main.py
├── config.py
├── requirements.txt
├── README.md
├── data/
│   ├── __init__.py
│   └── data_loader.py
├── preprocessing/
│   ├── __init__.py
│   └── preprocessing.py
├── models/
│   ├── __init__.py
│   └── ml_models.py
├── evaluation/
│   ├── __init__.py
│   └── evaluation.py
├── utils/
│   ├── __init__.py
│   └── visualization.py
├── tests/
│   ├── __init__.py
│   ├── test_data_loader.py
│   ├── test_preprocessing.py
│   ├── test_ml_models.py
│   └── test_evaluation.py
└── reports/
    └── figures/
```

## Purpose of Each File

| File/Folder | Purpose |
|---|---|
| `main.py` | Runs the complete workflow: load data, clean data, generate visualizations, train models, and evaluate results. |
| `config.py` | Stores project settings such as dataset path, target column, test size, random state, and output folders. |
| `data/data_loader.py` | Loads the student performance dataset and gives a basic dataset summary. |
| `preprocessing/preprocessing.py` | Removes duplicates, converts yes/no values, encodes categorical columns, separates features and target, and splits the data. |
| `models/ml_models.py` | Defines and trains machine learning regression models. |
| `evaluation/evaluation.py` | Calculates MAE, MSE, RMSE, and R2 Score for model evaluation. |
| `utils/visualization.py` | Contains visualization functions adapted from the original Kaggle notebook. Graphs are saved inside `reports/figures`. |
| `tests/` | Contains unit tests to verify that the main modules work correctly. |

## Dataset

The program checks for the dataset in this order:

1. `data/Student_Performance.csv`
2. `/kaggle/input/student-performance-dataset/Student_Performance.csv`

If you run this project in GitHub Codespaces, upload `Student_Performance.csv` into the `data/` folder.

## How to Run

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the main program:

```bash
python main.py
```

Run the tests:

```bash
pytest
```

## Models Used

This project trains three regression models to predict `overall_score`:

1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor

## Evaluation Metrics

The models are evaluated using:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R2 Score

## Notes for Coursework Presentation

During the presentation, explain the code flow in this order:

1. `main.py` starts the workflow.
2. `data_loader.py` loads the dataset.
3. `preprocessing.py` cleans and prepares the dataset.
4. `visualization.py` saves graphs from exploratory data analysis.
5. `ml_models.py` trains machine learning models.
6. `evaluation.py` compares model performance.
7. `tests/` checks whether each important module works correctly.
