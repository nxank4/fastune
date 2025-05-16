#!/usr/bin/env python
"""
Benchmark script comparing fastune's PBTSearchCV with Optuna
for hyperparameter optimization.

Before running this script, make sure to install the required dependencies:
pip install numpy scipy matplotlib scikit-learn optuna
"""

import time
import sys
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, load_diabetes, fetch_california_housing
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from argparse import ArgumentParser
from tabulate import tabulate  # Optional for pretty printing
import warnings
import os
import psutil
from scipy.stats import ttest_rel

# Import fastune
from fastune.pbt_search import PBTSearchCV

# Import optuna
import optuna
import warnings

# Set random state for reproducibility
RANDOM_STATE = 42

# Filter experimental warnings from Optuna
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)


# Load dataset
def load_data(dataset="breast_cancer", random_state=None):
    """
    Load a dataset with an optional random state for the train/test split

    Parameters:
    -----------
    dataset : str
        Name of the dataset to load. Options are:
        - "breast_cancer" (classification)
        - "diabetes" (regression)
        - "california" (regression)
    random_state : int or None
        Random state for the train/test split
    """
    seed = random_state if random_state is not None else RANDOM_STATE

    if dataset == "breast_cancer":
        X, y = load_breast_cancer(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed
        )
        task_type = "classification"

    elif dataset == "diabetes":
        X, y = load_diabetes(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed
        )
        task_type = "regression"

    elif dataset == "california":
        try:
            X, y = fetch_california_housing(return_X_y=True)
            # For California housing, take a subset to speed up benchmarks
            indices = np.random.RandomState(seed).permutation(len(X))[:5000]
            X, y = X[indices], y[indices]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=seed
            )
            task_type = "regression"
        except Exception as e:
            print(f"Error loading California housing dataset: {e}")
            print("Falling back to diabetes dataset")
            X, y = load_diabetes(return_X_y=True)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=seed
            )
            task_type = "regression"

    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    return X_train, X_test, y_train, y_test, task_type


# Define parameter space
def get_param_dist():
    return {
        "n_estimators": stats.randint(10, 100),
        "max_depth": stats.randint(2, 20),
        "min_samples_split": stats.randint(2, 10),
        "min_samples_leaf": stats.randint(1, 10),
    }


# Optuna objective function
def optuna_objective(
    trial, X_train, y_train, X_test, y_test, task_type="classification"
):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 10, 100),
        "max_depth": trial.suggest_int("max_depth", 2, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
    }

    if task_type == "classification":
        model = RandomForestClassifier(random_state=RANDOM_STATE, **params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return accuracy_score(y_test, y_pred)
    else:  # regression
        model = RandomForestRegressor(random_state=RANDOM_STATE, **params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        # Return negative MSE for maximization (Optuna defaults to maximizing)
        return -mean_squared_error(y_test, y_pred)


def run_fastune_search(
    X_train,
    y_train,
    X_test,
    y_test,
    param_dist,
    task_type="classification",
    population_size=10,
    generations=5,
    random_state=None,
):
    print("Running fastune PBTSearchCV...")
    start_time = time.time()

    # Use provided random state or default
    seed = random_state if random_state is not None else RANDOM_STATE

    # Select estimator based on task type
    if task_type == "classification":
        base_estimator = RandomForestClassifier(random_state=seed)
    else:  # regression
        base_estimator = RandomForestRegressor(random_state=seed)

    # Create PBTSearchCV object
    pbt = PBTSearchCV(
        estimator=base_estimator,
        param_dist=param_dist,
        population_size=population_size,
        generations=generations,
        cv=3,
        random_state=seed,
    )  # Fit the search
    pbt.fit(X_train, y_train)

    # Evaluate on test set
    if task_type == "classification":
        best_model = RandomForestClassifier(
            random_state=RANDOM_STATE, **pbt.best_params_
        )
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)
        test_score = accuracy_score(y_test, y_pred)
    else:  # regression
        best_model = RandomForestRegressor(
            random_state=RANDOM_STATE, **pbt.best_params_
        )
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)
        test_score = -mean_squared_error(
            y_test, y_pred
        )  # Negative MSE for consistency with Optuna

    elapsed_time = time.time() - start_time

    return {
        "best_params": pbt.best_params_,
        "best_score": pbt.best_score_,
        "test_score": test_score,
        "time": elapsed_time,
        "history": pbt.history_,
        "task_type": task_type,
    }


def run_optuna_search(
    X_train,
    y_train,
    X_test,
    y_test,
    task_type="classification",
    n_trials=50,
    random_state=None,
):
    print("Running Optuna optimization...")
    start_time = time.time()

    # Use provided random state or default
    seed = random_state if random_state is not None else RANDOM_STATE

    # Create a study object with seed
    sampler = optuna.samplers.RandomSampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    # Run optimization
    study.optimize(
        lambda trial: optuna_objective(
            trial, X_train, y_train, X_test, y_test, task_type
        ),
        n_trials=n_trials,
    )

    # Evaluate on test set
    if task_type == "classification":
        best_model = RandomForestClassifier(
            random_state=RANDOM_STATE, **study.best_params
        )
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)
        test_score = accuracy_score(y_test, y_pred)
    else:  # regression
        best_model = RandomForestRegressor(
            random_state=RANDOM_STATE, **study.best_params
        )
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)
        test_score = -mean_squared_error(y_test, y_pred)  # Negative MSE for consistency

    elapsed_time = time.time() - start_time

    return {
        "best_params": study.best_params,
        "best_score": study.best_value,
        "test_score": test_score,
        "time": elapsed_time,
        "study": study,
        "task_type": task_type,
    }


def plot_comparison(fastune_result, optuna_result):
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot fastune results
    gens = list(range(len(fastune_result["history"])))
    for gen, scores in enumerate(fastune_result["history"]):
        ax1.scatter([gen] * len(scores), scores, alpha=0.6)

    # Plot best per generation
    bests = [max(scores) for scores in fastune_result["history"]]
    ax1.plot(gens, bests, "-o", color="red", label="best")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Score")
    ax1.set_title("fastune PBT Optimization Progress")
    ax1.legend()
    ax1.grid(True)

    try:
        # First approach: Try to use the ax parameter if supported
        optuna.visualization.matplotlib.plot_optimization_history(
            optuna_result["study"], ax=ax2
        )
    except TypeError:
        # Fallback approach: Create a separate figure and copy the data
        fig_optuna = plt.figure()
        ax_optuna = optuna.visualization.matplotlib.plot_optimization_history(
            optuna_result["study"]
        )

        # Extract data from the Optuna plot
        lines = ax_optuna.get_lines()

        # Copy the data to our subplot
        for line in lines:
            ax2.plot(line.get_xdata(), line.get_ydata(), color=line.get_color())

        # Close the temporary figure
        plt.close(fig_optuna)

    ax2.set_title("Optuna Optimization Progress")
    ax2.set_xlabel("Number of Trials")
    ax2.set_ylabel("Objective Value")
    ax2.grid(True)

    # Show plot
    plt.tight_layout()
    plt.show()

    # Print comparison table
    print("\n=== Performance Comparison ===")
    print(f"{'Method':<10} | {'Test Score':<15} | {'Time (s)':<10}")
    print("-" * 40)
    print(
        f"{'fastune':<10} | {fastune_result['test_score']:.4f} | {fastune_result['time']:.2f}"
    )
    print(
        f"{'Optuna':<10} | {optuna_result['test_score']:.4f} | {optuna_result['time']:.2f}"
    )


def run_multiple_benchmarks(n_runs=5):
    """Run multiple benchmarks with different random seeds and collect results"""
    all_results = {
        "fastune": {
            "test_accuracy": [],
            "time": [],
            "best_score": [],
            "run_results": [],
        },
        "optuna": {
            "test_accuracy": [],
            "time": [],
            "best_score": [],
            "run_results": [],
        },
    }

    for run in range(n_runs):
        print(f"\n=== Running benchmark {run + 1}/{n_runs} ===")
        run_seed = RANDOM_STATE + run

        # Load data with current seed
        X_train, X_test, y_train, y_test, task_type = load_data(random_state=run_seed)
        param_dist = get_param_dist()

        # Run both optimizers
        fastune_result = run_fastune_search(
            X_train,
            y_train,
            X_test,
            y_test,
            param_dist,
            task_type=task_type,
            random_state=run_seed,
        )
        optuna_result = run_optuna_search(
            X_train,
            y_train,
            X_test,
            y_test,
            task_type=task_type,
            n_trials=50,
            random_state=run_seed,
        )

        # Store results
        all_results["fastune"]["test_accuracy"].append(fastune_result["test_score"])
        all_results["fastune"]["time"].append(fastune_result["time"])
        all_results["fastune"]["best_score"].append(fastune_result["best_score"])
        all_results["fastune"]["run_results"].append(fastune_result)

        all_results["optuna"]["test_accuracy"].append(optuna_result["test_score"])
        all_results["optuna"]["time"].append(optuna_result["time"])
        all_results["optuna"]["best_score"].append(optuna_result["best_score"])
        all_results["optuna"]["run_results"].append(optuna_result)

        # Print individual run results
        print(f"\n--- Run {run + 1} Results ---")
        print(f"{'Method':<10} | {'Test Score':<15} | {'Time (s)':<10}")
        print("-" * 40)
        print(
            f"{'fastune':<10} | {fastune_result['test_score']:.4f} | {fastune_result['time']:.2f}"
        )
        print(
            f"{'Optuna':<10} | {optuna_result['test_score']:.4f} | {optuna_result['time']:.2f}"
        )

    return all_results


def plot_aggregate_results(all_results, save_path=None):
    """Plot aggregated results from multiple benchmark runs and save to file"""
    fig, axes = plt.subplots(1, 4, figsize=(28, 6))

    # Accuracy comparison
    labels = ["fastune", "Optuna"]
    accuracy_means = [
        np.mean(all_results["fastune"]["test_accuracy"]),
        np.mean(all_results["optuna"]["test_accuracy"]),
    ]
    accuracy_stds = [
        np.std(all_results["fastune"]["test_accuracy"]),
        np.std(all_results["optuna"]["test_accuracy"]),
    ]

    axes[0].bar(labels, accuracy_means, yerr=accuracy_stds, alpha=0.7, capsize=10)
    axes[0].set_ylabel("Test Accuracy")
    axes[0].set_title("Average Test Accuracy")
    axes[0].grid(True, axis="y")

    # Time comparison
    time_means = [
        np.mean(all_results["fastune"]["time"]),
        np.mean(all_results["optuna"]["time"]),
    ]
    time_stds = [
        np.std(all_results["fastune"]["time"]),
        np.std(all_results["optuna"]["time"]),
    ]

    axes[1].bar(labels, time_means, yerr=time_stds, alpha=0.7, capsize=10)
    axes[1].set_ylabel("Time (seconds)")
    axes[1].set_title("Average Runtime")
    axes[1].grid(True, axis="y")

    # Convergence plot
    # For each run, plot the best score found up to each evaluation
    for run_idx, fastune_result in enumerate(all_results["fastune"]["run_results"]):
        gens = list(range(len(fastune_result["history"])))
        bests = [
            max(max(scores) for scores in fastune_result["history"][: i + 1])
            for i in range(len(fastune_result["history"]))
        ]
        axes[2].plot(gens, bests, "-", alpha=0.3, color="blue")

    for run_idx, optuna_result in enumerate(all_results["optuna"]["run_results"]):
        try:
            # Extract trial history from Optuna study
            trials = optuna_result["study"].trials
            values = [t.value if t.value is not None else 0 for t in trials]
            best_values = [max(values[: i + 1]) for i in range(len(values))]
            trial_numbers = list(range(len(values)))
            axes[2].plot(trial_numbers, best_values, "-", alpha=0.3, color="orange")
        except (AttributeError, KeyError):
            # Skip if trial history is not available
            pass

    # Add average convergence lines
    # Calculate average fastune convergence across runs
    avg_fastune = []
    for gen in range(5):  # Assuming 5 generations
        scores_at_gen = []
        for run_result in all_results["fastune"]["run_results"]:
            if gen < len(run_result["history"]):
                # Best score up to this generation
                best_so_far = max(
                    max(scores) for scores in run_result["history"][: gen + 1]
                )
                scores_at_gen.append(best_so_far)
        if scores_at_gen:
            avg_fastune.append(np.mean(scores_at_gen))

    # Plot average lines if data exists
    if avg_fastune:
        axes[2].plot(
            range(len(avg_fastune)),
            avg_fastune,
            "-o",
            linewidth=2,
            color="blue",
            label="fastune (avg)",
        )

    # Try to calculate average Optuna convergence
    max_trials = 50  # Assuming 50 trials
    avg_optuna = []

    for trial in range(max_trials):
        scores_at_trial = []
        for run_result in all_results["optuna"]["run_results"]:
            try:
                trials = run_result["study"].trials[: trial + 1]
                if trials:
                    values = [t.value if t.value is not None else 0 for t in trials]
                    best_so_far = max(values)
                    scores_at_trial.append(best_so_far)
            except (AttributeError, KeyError, IndexError):
                pass

        if scores_at_trial:
            avg_optuna.append(np.mean(scores_at_trial))

    if avg_optuna:
        axes[2].plot(
            range(len(avg_optuna)),
            avg_optuna,
            "-o",
            linewidth=2,
            color="orange",
            label="Optuna (avg)",
        )

    axes[2].set_xlabel("Iterations")
    axes[2].set_ylabel("Best Score")
    axes[2].set_title("Convergence Comparison")
    axes[2].legend()
    axes[2].grid(True)

    # axes[3]: Score vs. Time plot with average and std shading
    axes[3].set_title("Score vs. Time (avg ± std)")
    axes[3].set_xlabel("Time (seconds)")
    axes[3].set_ylabel("Best Score So Far")
    # Fastune runs
    fastune_curves = []
    for run_result in all_results["fastune"]["run_results"]:
        times = []
        bests = []
        total_time = 0.0
        for gen, scores in enumerate(run_result["history"]):
            gen_time = run_result["time"] / max(1, len(run_result["history"]))
            total_time += gen_time
            times.append(total_time)
            bests.append(
                max(max(scores) for scores in run_result["history"][: gen + 1])
            )
        fastune_curves.append((np.array(times), np.array(bests)))
    # Interpolate to common time grid for averaging
    if fastune_curves:
        max_time = max(curve[0][-1] for curve in fastune_curves)
        time_grid = np.linspace(0, max_time, 100)
        interp_scores = [
            np.interp(time_grid, curve[0], curve[1]) for curve in fastune_curves
        ]
        mean_curve = np.mean(interp_scores, axis=0)
        std_curve = np.std(interp_scores, axis=0)
        axes[3].plot(time_grid, mean_curve, color="blue", label="fastune (avg)")
        axes[3].fill_between(
            time_grid,
            mean_curve - std_curve,
            mean_curve + std_curve,
            color="blue",
            alpha=0.2,
        )
    # Optuna runs
    optuna_curves = []
    for run_result in all_results["optuna"]["run_results"]:
        trials = run_result["study"].trials
        if not trials:
            continue
        times = []
        bests = []
        total_time = 0.0
        trial_time = run_result["time"] / max(1, len(trials))
        values = [t.value if t.value is not None else 0 for t in trials]
        for i, v in enumerate(values):
            total_time += trial_time
            times.append(total_time)
            bests.append(max(values[: i + 1]))
        optuna_curves.append((np.array(times), np.array(bests)))
    if optuna_curves:
        max_time = max(curve[0][-1] for curve in optuna_curves)
        time_grid = np.linspace(0, max_time, 100)
        interp_scores = [
            np.interp(time_grid, curve[0], curve[1]) for curve in optuna_curves
        ]
        mean_curve = np.mean(interp_scores, axis=0)
        std_curve = np.std(interp_scores, axis=0)
        axes[3].plot(time_grid, mean_curve, color="orange", label="Optuna (avg)")
        axes[3].fill_between(
            time_grid,
            mean_curve - std_curve,
            mean_curve + std_curve,
            color="orange",
            alpha=0.2,
        )
    axes[3].legend()
    axes[3].grid(True)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.close(fig)


def print_aggregate_results(all_results, resource_stats=None):
    """Print aggregated statistics from multiple benchmark runs, including resource usage and t-test."""
    print("\n=== Aggregate Results ===")
    print(f"Number of benchmark runs: {len(all_results['fastune']['test_accuracy'])}")

    fastune_acc_mean = np.mean(all_results["fastune"]["test_accuracy"])
    fastune_acc_std = np.std(all_results["fastune"]["test_accuracy"])
    fastune_time_mean = np.mean(all_results["fastune"]["time"])
    fastune_time_std = np.std(all_results["fastune"]["time"])

    optuna_acc_mean = np.mean(all_results["optuna"]["test_accuracy"])
    optuna_acc_std = np.std(all_results["optuna"]["test_accuracy"])
    optuna_time_mean = np.mean(all_results["optuna"]["time"])
    optuna_time_std = np.std(all_results["optuna"]["time"])

    print("\n--- Test Accuracy ---")
    print(f"{'Method':<10} | {'Mean':<10} | {'Std Dev':<10}")
    print("-" * 35)
    print(f"{'fastune':<10} | {fastune_acc_mean:.4f} | {fastune_acc_std:.4f}")
    print(f"{'Optuna':<10} | {optuna_acc_mean:.4f} | {optuna_acc_std:.4f}")

    print("\n--- Runtime (seconds) ---")
    print(f"{'Method':<10} | {'Mean':<10} | {'Std Dev':<10}")
    print("-" * 35)
    print(f"{'fastune':<10} | {fastune_time_mean:.2f} | {fastune_time_std:.2f}")
    print(f"{'Optuna':<10} | {optuna_time_mean:.2f} | {optuna_time_std:.2f}")

    # Resource usage
    if resource_stats:
        print("\n--- Resource Usage (peak per run, MB) ---")
        print(f"{'Method':<10} | {'Max RSS':<10} | {'Mean RSS':<10}")
        print("-" * 35)
        for method in ["fastune", "optuna"]:
            rss_list = resource_stats[method]
            print(f"{method:<10} | {max(rss_list):<10.1f} | {np.mean(rss_list):<10.1f}")

    # Statistical significance
    if len(all_results["fastune"]["test_accuracy"]) > 1:
        t_stat, p_val = ttest_rel(
            all_results["fastune"]["test_accuracy"],
            all_results["optuna"]["test_accuracy"],
        )
        print("\n--- Paired t-test (fastune vs Optuna, test accuracy) ---")
        print(f"t-statistic: {t_stat:.4f}, p-value: {p_val:.4g}")
        if np.isnan(t_stat) or np.isnan(p_val):
            print(
                "Result: t-test not defined (identical or constant values across runs)."
            )
        elif p_val < 0.05:
            print("Result: Statistically significant difference (p < 0.05)")
        else:
            print("Result: No statistically significant difference (p >= 0.05)")
    else:
        print("\n--- Paired t-test (fastune vs Optuna, test accuracy) ---")
        print("Not enough runs for statistical test (need at least 2).")


def main():
    # Parse command-line arguments
    parser = ArgumentParser(description="Benchmark fastune vs Optuna")
    parser.add_argument(
        "-r",
        "--runs",
        type=int,
        default=1,
        help="Number of benchmark runs (default: 1)",
    )
    parser.add_argument(
        "-s",
        "--single",
        action="store_true",
        help="Run a single comparison instead of multiple runs",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="breast_cancer",
        choices=["breast_cancer", "diabetes", "california"],
        help="Dataset to use: breast_cancer (classification), diabetes (regression), california (regression)",
    )
    parser.add_argument(
        "--population_size",
        type=int,
        default=10,
        help="Population size for fastune PBTSearchCV (default: 10)",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=5,
        help="Number of generations for fastune PBTSearchCV (default: 5)",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=50,
        help="Number of trials for Optuna (default: 50)",
    )
    args = parser.parse_args()

    if args.single:
        print(f"Running single benchmark comparison on dataset: {args.dataset}")
        X_train, X_test, y_train, y_test, task_type = load_data(dataset=args.dataset)
        param_dist = get_param_dist()
        fastune_result = run_fastune_search(
            X_train,
            y_train,
            X_test,
            y_test,
            param_dist,
            task_type=task_type,
            population_size=args.population_size,
            generations=args.generations,
            random_state=RANDOM_STATE,
        )
        optuna_result = run_optuna_search(
            X_train,
            y_train,
            X_test,
            y_test,
            task_type=task_type,
            n_trials=args.n_trials,
            random_state=RANDOM_STATE,
        )
        plot_comparison(fastune_result, optuna_result)
        print("\n=== Best Parameters ===")
        print("fastune:", fastune_result["best_params"])
        print("Optuna:", optuna_result["best_params"])
    else:
        print(f"Running {args.runs} benchmark comparisons on dataset: {args.dataset}")
        resource_stats = {"fastune": [], "optuna": []}

        def run_multi():
            all_results = {
                "fastune": {
                    "test_accuracy": [],
                    "time": [],
                    "best_score": [],
                    "run_results": [],
                },
                "optuna": {
                    "test_accuracy": [],
                    "time": [],
                    "best_score": [],
                    "run_results": [],
                },
            }
            for run in range(args.runs):
                print(f"\n=== Running benchmark {run + 1}/{args.runs} ===")
                run_seed = RANDOM_STATE + run
                X_train, X_test, y_train, y_test, task_type = load_data(
                    dataset=args.dataset, random_state=run_seed
                )
                param_dist = get_param_dist()
                # Resource usage tracking
                process = psutil.Process()
                rss_before = process.memory_info().rss / 1024 / 1024
                cpu_before = process.cpu_times().user
                fastune_result = run_fastune_search(
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    param_dist,
                    task_type=task_type,
                    population_size=args.population_size,
                    generations=args.generations,
                    random_state=run_seed,
                )
                rss_after = process.memory_info().rss / 1024 / 1024
                cpu_after = process.cpu_times().user
                resource_stats["fastune"].append(rss_after)
                # Optuna
                rss_before = process.memory_info().rss / 1024 / 1024
                cpu_before = process.cpu_times().user
                optuna_result = run_optuna_search(
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    task_type=task_type,
                    n_trials=args.n_trials,
                    random_state=run_seed,
                )
                rss_after = process.memory_info().rss / 1024 / 1024
                cpu_after = process.cpu_times().user
                resource_stats["optuna"].append(rss_after)
                all_results["fastune"]["test_accuracy"].append(
                    fastune_result["test_score"]
                )
                all_results["fastune"]["time"].append(fastune_result["time"])
                all_results["fastune"]["best_score"].append(
                    fastune_result["best_score"]
                )
                all_results["fastune"]["run_results"].append(fastune_result)
                all_results["optuna"]["test_accuracy"].append(
                    optuna_result["test_score"]
                )
                all_results["optuna"]["time"].append(optuna_result["time"])
                all_results["optuna"]["best_score"].append(optuna_result["best_score"])
                all_results["optuna"]["run_results"].append(optuna_result)
                print(f"\n--- Run {run + 1} Results ---")
                print(f"{'Method':<10} | {'Test Score':<15} | {'Time (s)':<10}")
                print("-" * 40)
                print(
                    f"{'fastune':<10} | {fastune_result['test_score']:.4f} | {fastune_result['time']:.2f}"
                )
                print(
                    f"{'Optuna':<10} | {optuna_result['test_score']:.4f} | {optuna_result['time']:.2f}"
                )
            return all_results

        all_results = run_multi()
        results_dir = os.path.join(
            os.path.dirname(__file__), "..", "results", "benchmark"
        )
        os.makedirs(results_dir, exist_ok=True)
        plot_aggregate_results(
            all_results, save_path=os.path.join(results_dir, "aggregate_results.png")
        )
        print_aggregate_results(all_results, resource_stats=resource_stats)


if __name__ == "__main__":
    main()
