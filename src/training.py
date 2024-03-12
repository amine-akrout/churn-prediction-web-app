"""
This module contains functions for training and evaluating models for customer churn prediction.

"""

from typing import List, Tuple, Any
import pandas as pd
from pycaret.classification import (
    setup,
    compare_models,
    tune_model,
    finalize_model,
    save_model,
    automl,
    stack_models,
    create_model,
    create_app,
    create_docker,
    create_api,
)


def read_data(train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reads the training and test datasets from the specified file paths.

    Parameters:
    - train_path: Path to the training data file.
    - test_path: Path to the test data file.

    Returns:
    - Tuple containing the training and test data as pandas DataFrames.
    """
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test


def setup_model(train_data: pd.DataFrame) -> Any:
    """
    Sets up the environment in PyCaret for model training.

    Parameters:
    - train_data: The training data as a pandas DataFrame.

    Returns:
    - The setup object from PyCaret after initialization.
    """
    setting_up = setup(
        data=train_data,
        target="churn",
        fix_imbalance=True,
        remove_outliers=True,
        normalize=True,
        train_size=0.9,
        log_experiment=True,
        experiment_name="churn_smote",
    )
    return setting_up


def train_models() -> List[Any]:
    """
    Trains multiple models and selects the top performing models.

    Returns:
    - A list of top performing models.
    """
    models = compare_models(
        n_select=3, fold=5, round=2, exclude=["qda", "gpc"], turbo=False
    )
    return models


def tune_models(models: List[Any]) -> List[Any]:
    """
    Tunes the given models.

    Parameters:
    - models: A list of models to be tuned.

    Returns:
    - A list of tuned models.
    """
    tuned_models = [
        tune_model(
            i,
            fold=5,
            round=2,
            optimize="F1",
            search_library="tune-sklearn",
            search_algorithm="hyperopt",
            early_stopping=True,
            choose_better=True,
            early_stopping_max_iters=3,
        )
        for i in models
    ]
    return tuned_models


def create_ensemble_model(tuned_models: List[Any]) -> Any:
    """
    Creates an ensemble model using the given models.

    Parameters:
    - tuned_models: A list of models to be used in the ensemble.

    Returns:
    - The ensemble model.
    """
    ada = create_model("ada", fold=5, round=2)
    ensemble = stack_models(
        tuned_models, fold=5, round=2, meta_model=ada, optimize="F1", choose_better=True
    )
    return ensemble


def select_and_finalize_best_model() -> Any:
    """
    Selects the best model based on performance and finalizes it.

    Returns:
    - The finalized best model.
    """
    best = automl(optimize="F1")
    final_model = finalize_model(best)
    return final_model


def save_final_model(model: Any, filename: str) -> None:
    """
    Saves the given model to a file.

    Parameters:
    - model: The model to be saved.
    - filename: The filename to save the model.
    """
    save_model(model, filename)


def main() -> None:
    """
    Main function to orchestrate the model training and evaluation process.
    """
    train_data, _ = read_data("../data/train.csv", "../data/test.csv")
    setup_model(train_data)
    models = train_models()
    tuned_models = tune_models(models)
    _ = create_ensemble_model(tuned_models)
    final_model = select_and_finalize_best_model()
    save_final_model(final_model, "../artifacts/model")


if __name__ == "__main__":
    main()
