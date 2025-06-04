"""
Scikit-learn GridSearchCV and RandomizedSearchCV benchmark for wine quality dataset.
Tests both classification and regression tasks using Random Forest models.
"""

import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
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


def sklearn_gridsearch_optimization(
    X_train, y_train, X_test, y_test, task_type="classification", random_seed=42
):
    """Perform hyperparameter optimization using GridSearchCV."""  # Define parameter grid (smaller for Grid Search due to computational cost)
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [5, 10, 15, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
    }

    if task_type == "classification":
        base_model = RandomForestClassifier(random_state=random_seed, n_jobs=-1)
        scoring = "accuracy"
    else:
        base_model = RandomForestRegressor(random_state=random_seed, n_jobs=-1)
        scoring = "neg_mean_squared_error"

    start_time = time.time()

    # Perform Grid Search
    grid_search = GridSearchCV(
        base_model, param_grid, cv=3, scoring=scoring, n_jobs=-1, verbose=0
    )

    grid_search.fit(X_train, y_train)

    optimization_time = time.time() - start_time

    # Evaluate best model
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    if task_type == "classification":
        final_score = accuracy_score(y_test, y_pred)
    else:
        final_score = mean_squared_error(y_test, y_pred)

    return {
        "best_params": grid_search.best_params_,
        "final_score": final_score,
        "optimization_time": optimization_time,
        "n_trials": len(grid_search.cv_results_["params"]),
        "method": "GridSearchCV",
    }


def sklearn_randomsearch_optimization(
    X_train,
    y_train,
    X_test,
    y_test,
    task_type="classification",
    n_iter=50,
    random_seed=42,
):
    """Perform hyperparameter optimization using RandomizedSearchCV."""  # Define parameter distribution
    param_distributions = {
        "n_estimators": [50, 100, 200, 300],
        "max_depth": [3, 5, 10, 15, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None],
    }

    if task_type == "classification":
        base_model = RandomForestClassifier(random_state=random_seed, n_jobs=-1)
        scoring = "accuracy"
    else:
        base_model = RandomForestRegressor(random_state=random_seed, n_jobs=-1)
        scoring = "neg_mean_squared_error"

    start_time = time.time()

    # Perform Randomized Search
    random_search = RandomizedSearchCV(
        base_model,
        param_distributions,
        n_iter=n_iter,
        cv=3,
        scoring=scoring,
        n_jobs=-1,
        random_state=random_seed,
        verbose=0,
    )

    random_search.fit(X_train, y_train)

    optimization_time = time.time() - start_time

    # Evaluate best model
    best_model = random_search.best_estimator_
    y_pred = best_model.predict(X_test)

    if task_type == "classification":
        final_score = accuracy_score(y_test, y_pred)
    else:
        final_score = mean_squared_error(y_test, y_pred)

    return {
        "best_params": random_search.best_params_,
        "final_score": final_score,
        "optimization_time": optimization_time,
        "n_trials": n_iter,
        "method": "RandomizedSearchCV",
    }


def run_sklearn_benchmark(data_paths, n_iter=50, random_seed=42):
    """Run complete sklearn benchmark on wine datasets."""
    results = {}

    for data_name, data_path in data_paths.items():
        print(f"\n=== Running sklearn benchmark on {data_name} ===")
        results[data_name] = {}

        for task_type in ["classification", "regression"]:
            print(f"\nTask: {task_type}")
            results[data_name][task_type] = {}

            # Load and prepare data
            X_train, X_test, y_train, y_test = load_and_prepare_data(
                data_path, task_type, random_seed
            )

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Run GridSearchCV
            print("  Running GridSearchCV...")
            grid_result = sklearn_gridsearch_optimization(
                X_train_scaled, y_train, X_test_scaled, y_test, task_type, random_seed
            )
            results[data_name][task_type]["GridSearchCV"] = grid_result

            print(
                f"    GridSearch - Score: {grid_result['final_score']:.4f}, Time: {grid_result['optimization_time']:.2f}s"
            )

            # Run RandomizedSearchCV
            print("  Running RandomizedSearchCV...")
            random_result = sklearn_randomsearch_optimization(
                X_train_scaled,
                y_train,
                X_test_scaled,
                y_test,
                task_type,
                n_iter,
                random_seed,
            )
            results[data_name][task_type]["RandomizedSearchCV"] = random_result

            print(
                f"    RandomSearch - Score: {random_result['final_score']:.4f}, Time: {random_result['optimization_time']:.2f}s"
            )

    return results


if __name__ == "__main__":
    data_paths = {
        "red_wine": "data/winequality-red.csv",
        "white_wine": "data/winequality-white.csv",
    }

    print("Starting sklearn benchmark...")
    results = run_sklearn_benchmark(data_paths, n_iter=50)

    print("\n" + "=" * 60)
    print("SKLEARN BENCHMARK RESULTS")
    print("=" * 60)

    for dataset, tasks in results.items():
        print(f"\nDataset: {dataset}")
        for task, methods in tasks.items():
            print(f"  {task}:")
            for method, result in methods.items():
                print(f"    {method}:")
                print(f"      Score: {result['final_score']:.4f}")
                print(f"      Time: {result['optimization_time']:.2f}s")
                print(f"      Trials: {result['n_trials']}")
