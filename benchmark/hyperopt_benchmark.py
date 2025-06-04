"""
Hyperopt benchmark for wine quality dataset.
Tests both classification and regression tasks using Random Forest models.
"""

import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
from sklearn.preprocessing import StandardScaler
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
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


def hyperopt_optimization(
    X_train,
    y_train,
    X_test,
    y_test,
    task_type="classification",
    max_evals=50,
    random_seed=42,
):
    """Perform hyperparameter optimization using Hyperopt."""

    # Define search space
    space = {
        "n_estimators": hp.choice("n_estimators", [50, 100, 200, 300]),
        "max_depth": hp.choice("max_depth", [3, 5, 10, 15, 20, None]),
        "min_samples_split": hp.choice("min_samples_split", [2, 5, 10]),
        "min_samples_leaf": hp.choice("min_samples_leaf", [1, 2, 4]),
        "max_features": hp.choice("max_features", ["sqrt", "log2", None]),
    }

    def objective(params):
        if task_type == "classification":
            model = RandomForestClassifier(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                min_samples_split=params["min_samples_split"],
                min_samples_leaf=params["min_samples_leaf"],
                max_features=params["max_features"],
                random_state=42,
                n_jobs=-1,
            )
        else:
            model = RandomForestRegressor(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                min_samples_split=params["min_samples_split"],
                min_samples_leaf=params["min_samples_leaf"],
                max_features=params["max_features"],
                random_state=42,
                n_jobs=-1,
            )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if task_type == "classification":
            # Return negative accuracy (hyperopt minimizes)
            score = -accuracy_score(y_test, y_pred)
        else:
            # Return MSE (hyperopt minimizes)
            score = mean_squared_error(y_test, y_pred)

        return {"loss": score, "status": STATUS_OK}

    trials = Trials()
    start_time = time.time()

    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials,
        rstate=np.random.default_rng(42),
    )

    optimization_time = time.time() - start_time

    # Train final model with best parameters
    if task_type == "classification":
        best_model = RandomForestClassifier(
            n_estimators=[50, 100, 200, 300][best["n_estimators"]],
            max_depth=[3, 5, 10, 15, 20, None][best["max_depth"]],
            min_samples_split=[2, 5, 10][best["min_samples_split"]],
            min_samples_leaf=[1, 2, 4][best["min_samples_leaf"]],
            max_features=["sqrt", "log2", None][best["max_features"]],
            random_state=random_seed,
            n_jobs=-1,
        )
    else:
        best_model = RandomForestRegressor(
            n_estimators=[50, 100, 200, 300][best["n_estimators"]],
            max_depth=[3, 5, 10, 15, 20, None][best["max_depth"]],
            min_samples_split=[2, 5, 10][best["min_samples_split"]],
            min_samples_leaf=[1, 2, 4][best["min_samples_leaf"]],
            max_features=["sqrt", "log2", None][best["max_features"]],
            random_state=random_seed,
            n_jobs=-1,
        )

    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)

    if task_type == "classification":
        final_score = accuracy_score(y_test, y_pred)
    else:
        final_score = mean_squared_error(y_test, y_pred)

    return {
        "best_params": best,
        "final_score": final_score,
        "optimization_time": optimization_time,
        "n_trials": len(trials.trials),
    }


def run_hyperopt_benchmark(data_paths, max_evals=50, random_seed=42):
    """Run complete Hyperopt benchmark on wine datasets."""
    results = {}

    for data_name, data_path in data_paths.items():
        print(f"\n=== Running Hyperopt benchmark on {data_name} ===")
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
            result = hyperopt_optimization(
                X_train_scaled,
                y_train,
                X_test_scaled,
                y_test,
                task_type=task_type,
                max_evals=max_evals,
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

    print("Starting Hyperopt benchmark...")
    results = run_hyperopt_benchmark(data_paths, max_evals=50)

    print("\n" + "=" * 60)
    print("HYPEROPT BENCHMARK RESULTS")
    print("=" * 60)

    for dataset, tasks in results.items():
        print(f"\nDataset: {dataset}")
        for task, result in tasks.items():
            print(f"  {task}:")
            print(f"    Score: {result['final_score']:.4f}")
            print(f"    Time: {result['optimization_time']:.2f}s")
            print(f"    Trials: {result['n_trials']}")
