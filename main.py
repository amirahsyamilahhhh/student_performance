
"""Main script to run the Student Performance machine learning workflow."""
from data.data_loader import StudentPerformanceDataLoader
from evaluation.evaluation import ModelEvaluator
from models.ml_models import StudentPerformanceModelTrainer
from preprocessing.preprocessing import StudentPerformancePreprocessor
from utils.visualization import generate_all_plots


def main():
    """Run data loading, preprocessing, model training, evaluation, and visualization."""
    loader = StudentPerformanceDataLoader()
    df = loader.load_data()

    print("Dataset summary:")
    print(loader.get_basic_summary(df))

    preprocessor = StudentPerformancePreprocessor()
    clean_df = preprocessor.clean_data(df)

    # Save figures from the exploratory analysis section.
    generate_all_plots(clean_df)

    x_train, x_test, y_train, y_test = preprocessor.preprocess(df)

    trainer = StudentPerformanceModelTrainer()
    trained_models = trainer.train_models(x_train, y_train)

    evaluator = ModelEvaluator()
    results = evaluator.evaluate_all_models(trained_models, x_test, y_test)

    print("\nModel Evaluation Results:")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
