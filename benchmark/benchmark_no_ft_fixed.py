"""
Comprehensive benchmark comparing hyperparameter tuning libraries for wine quality prediction.
This script benchmarks Hyperopt, Optuna, and sklearn GridSearch/RandomSearch
on wine quality datasets for both classification and regression tasks.
"""

import time
import json
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add the benchmark directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


# Import benchmark modules directly
try:
    from hyperopt_benchmark import run_hyperopt_benchmark

    HYPEROPT_AVAILABLE = True
except ImportError as e:
    print(f"Hyperopt benchmark not available: {e}")
    HYPEROPT_AVAILABLE = False
    run_hyperopt_benchmark = None

try:
    from optuna_benchmark import run_optuna_benchmark

    OPTUNA_AVAILABLE = True
except ImportError as e:
    print(f"Optuna benchmark not available: {e}")
    OPTUNA_AVAILABLE = False
    run_optuna_benchmark = None

try:
    from sklearn_benchmark import run_sklearn_benchmark

    SKLEARN_AVAILABLE = True
except ImportError as e:
    print(f"Sklearn benchmark not available: {e}")
    SKLEARN_AVAILABLE = False
    run_sklearn_benchmark = None


def check_data_files(data_paths):
    """Check if data files exist."""
    missing_files = []
    for name, path in data_paths.items():
        if not os.path.exists(path):
            missing_files.append(f"{name}: {path}")

    if missing_files:
        print("Missing data files:")
        for file in missing_files:
            print(f"  - {file}")
        return False
    return True


def save_results(results, filename="results/benchmark/benchmark_results.json"):
    """Save benchmark results to JSON file."""

    # Convert numpy types to Python types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        else:
            return obj

    results_serializable = convert_numpy(results)

    with open(filename, "w") as f:
        json.dump(results_serializable, f, indent=2)
    print(f"\nResults saved to {filename}")


def create_summary_table(results):
    """Create a summary table of all benchmark results."""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE BENCHMARK SUMMARY")
    print("=" * 80)

    # Collect all data for comparison
    summary_data = []

    for library, lib_results in results.items():
        if "error" in lib_results:
            continue

        for dataset, dataset_results in lib_results.items():
            for task, task_results in dataset_results.items():
                if isinstance(task_results, dict) and "GridSearchCV" in task_results:
                    # sklearn has nested structure
                    for method, method_results in task_results.items():
                        if "final_score" in method_results:
                            summary_data.append(
                                {
                                    "Library": f"{library}_{method}",
                                    "Dataset": dataset,
                                    "Task": task,
                                    "Score": method_results["final_score"],
                                    "Time(s)": method_results["optimization_time"],
                                    "Trials": method_results.get("n_trials", "N/A"),
                                }
                            )
                elif "final_score" in task_results:
                    summary_data.append(
                        {
                            "Library": library,
                            "Dataset": dataset,
                            "Task": task,
                            "Score": task_results["final_score"],
                            "Time(s)": task_results["optimization_time"],
                            "Trials": task_results.get("n_trials", "N/A"),
                        }
                    )

    if not summary_data:
        print("No valid results to display.")
        return

    # Create DataFrame for better formatting
    df = pd.DataFrame(summary_data)

    # Group by task type and show results
    for task in ["classification", "regression"]:
        task_data = df[df["Task"] == task]
        if task_data.empty:
            continue

        print(f"\n{task.upper()} RESULTS:")
        print("-" * 60)

        for dataset in task_data["Dataset"].unique():
            dataset_data = task_data[task_data["Dataset"] == dataset]
            print(f"\n{dataset}:")

            # Sort by score (descending for classification, ascending for regression)
            if task == "classification":
                dataset_data = dataset_data.sort_values("Score", ascending=False)
                score_label = "Accuracy"
            else:
                dataset_data = dataset_data.sort_values("Score", ascending=True)
                score_label = "MSE"

            for _, row in dataset_data.iterrows():
                print(
                    f"  {row['Library']:<20} | {score_label}: {row['Score']:.4f} | Time: {row['Time(s)']:>6.2f}s | Trials: {row['Trials']}"
                )


def run_comprehensive_benchmark():
    """Run all available benchmarks."""
    print("Starting comprehensive hyperparameter tuning benchmark...")
    print("=" * 60)

    # Define data paths (relative to the project root)
    data_paths = {
        "red_wine": "data/winequality-red.csv",
        "white_wine": "data/winequality-white.csv",
    }

    # Check if data files exist
    print("📁 Checking data files...")
    if not check_data_files(data_paths):
        print("Please ensure data files are available before running benchmarks.")
        return None
    print("✅ All data files found!")

    # Configuration
    n_trials = 50  # Number of trials/iterations for each method
    random_seed = 42  # For reproducibility
    print(f"🎯 Configuration: {n_trials} trials per optimization method")
    print(f"🎲 Random seed: {random_seed} (for reproducibility)")

    # Count available benchmarks
    available_benchmarks = []
    if HYPEROPT_AVAILABLE:
        available_benchmarks.append("Hyperopt")
    if OPTUNA_AVAILABLE:
        available_benchmarks.append("Optuna")
    if SKLEARN_AVAILABLE:
        available_benchmarks.append("Sklearn")

    print(f"🔧 Available libraries: {', '.join(available_benchmarks)}")
    print(
        f"📊 Total benchmarks to run: {len(available_benchmarks)} libraries × 2 datasets × 2 tasks = {len(available_benchmarks) * 4} individual tests"
    )
    print("")

    results = {}
    total_start_time = time.time()
    library_count = 0
    total_libraries = len(available_benchmarks)

    # Run Hyperopt benchmark
    if HYPEROPT_AVAILABLE:
        library_count += 1
        print(f"\n🔍 [{library_count}/{total_libraries}] Running Hyperopt benchmark...")
        print("   📋 Testing: TPE (Tree-structured Parzen Estimator) optimization")
        hyperopt_start = time.time()
        try:
            print("   🍷 Processing red wine dataset...")
            print("   🍾 Processing white wine dataset...")
            print("   📊 Running classification tasks...")
            print("   📈 Running regression tasks...")
            hyperopt_results = run_hyperopt_benchmark(
                data_paths, max_evals=n_trials, random_seed=random_seed
            )
            hyperopt_time = time.time() - hyperopt_start
            print(f"   ✅ Hyperopt completed in {hyperopt_time:.2f} seconds")
            results["hyperopt"] = hyperopt_results
        except Exception as e:
            hyperopt_time = time.time() - hyperopt_start
            print(
                f"   ❌ Error running Hyperopt benchmark after {hyperopt_time:.2f}s: {e}"
            )
            results["hyperopt"] = {"error": str(e)}
    else:
        print(f"\n❌ [1/{total_libraries}] Hyperopt not available, skipping...")
        results["hyperopt"] = {"error": "Library not installed"}

    # Run Optuna benchmark
    if OPTUNA_AVAILABLE:
        library_count += 1
        print(f"\n🎯 [{library_count}/{total_libraries}] Running Optuna benchmark...")
        print("   📋 Testing: TPE + Multi-objective optimization")
        optuna_start = time.time()
        try:
            print("   🍷 Processing red wine dataset...")
            print("   🍾 Processing white wine dataset...")
            print("   📊 Running classification tasks...")
            print("   📈 Running regression tasks...")
            optuna_results = run_optuna_benchmark(
                data_paths, n_trials=n_trials, random_seed=random_seed
            )
            optuna_time = time.time() - optuna_start
            print(f"   ✅ Optuna completed in {optuna_time:.2f} seconds")
            results["optuna"] = optuna_results
        except Exception as e:
            optuna_time = time.time() - optuna_start
            print(f"   ❌ Error running Optuna benchmark after {optuna_time:.2f}s: {e}")
            results["optuna"] = {"error": str(e)}
    else:
        print(f"\n❌ [2/{total_libraries}] Optuna not available, skipping...")
        results["optuna"] = {"error": "Library not installed"}

    # Run sklearn benchmark
    if SKLEARN_AVAILABLE:
        library_count += 1
        print(f"\n🧪 [{library_count}/{total_libraries}] Running sklearn benchmark...")
        print("   📋 Testing: GridSearchCV + RandomizedSearchCV")
        sklearn_start = time.time()
        try:
            print("   🍷 Processing red wine dataset...")
            print("   🍾 Processing white wine dataset...")
            print("   📊 Running classification tasks...")
            print("   📈 Running regression tasks...")
            print("   🔍 GridSearchCV: Exhaustive search over parameter grid")
            print("   🎲 RandomizedSearchCV: Random sampling from parameter space")
            sklearn_results = run_sklearn_benchmark(
                data_paths, n_iter=n_trials, random_seed=random_seed
            )
            sklearn_time = time.time() - sklearn_start
            print(f"   ✅ Sklearn benchmark completed in {sklearn_time:.2f} seconds")
            results["sklearn"] = sklearn_results
        except Exception as e:
            sklearn_time = time.time() - sklearn_start
            print(
                f"   ❌ Error running sklearn benchmark after {sklearn_time:.2f}s: {e}"
            )
            results["sklearn"] = {"error": str(e)}
    else:
        print(
            f"\n❌ [3/{total_libraries}] Sklearn benchmarks not available, skipping..."
        )
        results["sklearn"] = {"error": "Library not installed"}

    # Total time
    total_time = time.time() - total_start_time

    print(f"\n" + "=" * 60)
    print(f"🏁 BENCHMARK COMPLETE!")
    print(
        f"⏱️  Total execution time: {total_time:.2f} seconds ({total_time / 60:.1f} minutes)"
    )
    print(f"📊 Completed {library_count}/{total_libraries} libraries")

    # Show success/failure summary
    successful_benchmarks = []
    failed_benchmarks = []
    for lib, lib_results in results.items():
        if "error" in lib_results:
            failed_benchmarks.append(lib)
        else:
            successful_benchmarks.append(lib)

    if successful_benchmarks:
        print(f"✅ Successful: {', '.join(successful_benchmarks)}")
    if failed_benchmarks:
        print(f"❌ Failed: {', '.join(failed_benchmarks)}")
    print("=" * 60)

    # Create summary
    create_summary_table(results)

    # Save results
    save_results(results)

    return results


if __name__ == "__main__":
    print("Wine Quality Hyperparameter Tuning Benchmark")
    print("Comparing: Hyperopt, Optuna, and Scikit-learn")
    print("Tasks: Classification (quality >= 7) and Regression (quality score)")
    print("Model: Random Forest with hyperparameter optimization")

    results = run_comprehensive_benchmark()

    if results:
        print("\n🎉 Benchmark completed successfully!")
        print("Check 'results/benchmark/benchmark_results.json' for detailed results.")
    else:
        print("\n❌ Benchmark failed to run.")
