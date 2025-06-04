"""
Optuna benchmark for wine quality dataset.
Tests both classification and regression tasks using Random Forest models.
"""

import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import optuna
from optuna.samplers import TPESampler
import warnings

warnings.filterwarnings("ignore")


def load_and_prepare_data(data_path, task_type="classification", random_seed=42):
    """Load and prepare wine quality data for the specified task."""
    df = pd.read_csv(data_path, sep=";")

    # Features
    X = df.drop("quality", axis=1)

    if task_type == "classification":
        # Convert to binary classification: quality >= 7 is "good", < 7 is "not good"
        y = (df["quality"] >= 7).astype(int)
    else:  # regression
        y = df["quality"]

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_seed,
        stratify=y if task_type == "classification" else None,
    )


def optuna_optimization(
    X_train,
    y_train,
    X_test,
    y_test,
    task_type="classification",
    n_trials=50,
    random_seed=42,
):
    """Perform hyperparameter optimization using Optuna."""

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_categorical(
                "n_estimators", [50, 100, 200, 300]
            ),
            "max_depth": trial.suggest_categorical(
                "max_depth", [3, 5, 10, 15, 20, None]
            ),
            "min_samples_split": trial.suggest_categorical(
                "min_samples_split", [2, 5, 10]
            ),
            "min_samples_leaf": trial.suggest_categorical(
                "min_samples_leaf", [1, 2, 4]
            ),
            "max_features": trial.suggest_categorical(
                "max_features", ["sqrt", "log2", None]
            ),
        }

        if task_type == "classification":
            model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
        else:
            model = RandomForestRegressor(**params, random_state=42, n_jobs=-1)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if task_type == "classification":
            # Optuna maximizes by default, so return accuracy directly
            score = accuracy_score(y_test, y_pred)
        else:
            # For regression, minimize MSE (return negative MSE to maximize)
            score = -mean_squared_error(y_test, y_pred)

        return score

    # Create study
    direction = "maximize"
    study = optuna.create_study(direction=direction, sampler=TPESampler(seed=42))

    start_time = time.time()

    # Optimize
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    optimization_time = time.time() - start_time

    # Train final model with best parameters
    best_params = study.best_params

    if task_type == "classification":
        best_model = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
    else:
        best_model = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)

    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)

    if task_type == "classification":
        final_score = accuracy_score(y_test, y_pred)
    else:
        final_score = mean_squared_error(y_test, y_pred)

    return {
        "best_params": best_params,
        "final_score": final_score,
        "optimization_time": optimization_time,
        "n_trials": len(study.trials),
        "best_value": study.best_value,
    }


def run_optuna_benchmark(data_paths, n_trials=50, random_seed=42):
    """Run complete Optuna benchmark on wine datasets."""
    results = {}

    for data_name, data_path in data_paths.items():
        print(f"\n=== Running Optuna benchmark on {data_name} ===")
        results[data_name] = {}

        for task_type in ["classification", "regression"]:
            print(f"\nTask: {task_type}")

            # Load and prepare data
            X_train, X_test, y_train, y_test = load_and_prepare_data(
                data_path, task_type, random_seed
            )

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Run optimization
            result = optuna_optimization(
                X_train_scaled,
                y_train,
                X_test_scaled,
                y_test,
                task_type=task_type,
                n_trials=n_trials,
                random_seed=random_seed,
            )

            results[data_name][task_type] = result

            print(f"Best score: {result['final_score']:.4f}")
            print(f"Optimization time: {result['optimization_time']:.2f}s")
            print(f"Number of trials: {result['n_trials']}")

    return results


if __name__ == "__main__":
    data_paths = {
        "red_wine": "data/winequality-red.csv",
        "white_wine": "data/winequality-white.csv",
    }

    print("Starting Optuna benchmark...")
    results = run_optuna_benchmark(data_paths, n_trials=50)

    print("\n" + "=" * 60)
    print("OPTUNA BENCHMARK RESULTS")
    print("=" * 60)

    for dataset, tasks in results.items():
        print(f"\nDataset: {dataset}")
        for task, result in tasks.items():
            print(f"  {task}:")
            print(f"    Score: {result['final_score']:.4f}")
            print(f"    Time: {result['optimization_time']:.2f}s")
            print(f"    Trials: {result['n_trials']}")
