"""
Utility script to analyze and visualize benchmark results.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np


def load_benchmark_results(filepath="results/benchmark/benchmark_results.json"):
    """Load benchmark results from JSON file."""
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Results file {filepath} not found. Please run the benchmark first.")
        return None
    except json.JSONDecodeError:
        print(f"Invalid JSON in {filepath}")
        return None


def create_results_dataframe(results):
    """Convert benchmark results to pandas DataFrame for analysis."""
    data = []

    for library, lib_results in results.items():
        if "error" in lib_results:
            continue

        for dataset, dataset_results in lib_results.items():
            for task, task_results in dataset_results.items():
                if isinstance(task_results, dict) and "GridSearchCV" in task_results:
                    # sklearn has nested structure
                    for method, method_results in task_results.items():
                        if "final_score" in method_results:
                            data.append(
                                {
                                    "library": f"{library}_{method}",
                                    "dataset": dataset,
                                    "task": task,
                                    "score": method_results["final_score"],
                                    "time": method_results["optimization_time"],
                                    "trials": method_results.get("n_trials", 0),
                                    "method": method,
                                }
                            )
                elif "final_score" in task_results:
                    data.append(
                        {
                            "library": library,
                            "dataset": dataset,
                            "task": task,
                            "score": task_results["final_score"],
                            "time": task_results["optimization_time"],
                            "trials": task_results.get("n_trials", 0),
                            "method": library,
                        }
                    )

    return pd.DataFrame(data)


def plot_performance_comparison(df, save_path="results/benchmark_plots"):
    """Create performance comparison plots."""
    Path(save_path).mkdir(exist_ok=True)

    # Set style
    plt.style.use("seaborn-v0_8")
    sns.set_palette("husl")

    # 1. Score comparison by library and task
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Hyperparameter Tuning Libraries Performance Comparison", fontsize=16)

    tasks = ["classification", "regression"]
    datasets = ["red_wine", "white_wine"]

    for i, task in enumerate(tasks):
        for j, dataset in enumerate(datasets):
            ax = axes[i, j]
            task_data = df[(df["task"] == task) & (df["dataset"] == dataset)]

            if not task_data.empty:
                # Sort by score
                if task == "classification":
                    task_data = task_data.sort_values("score", ascending=False)
                    score_label = "Accuracy"
                else:
                    task_data = task_data.sort_values("score", ascending=True)
                    score_label = "MSE"

                bars = ax.bar(
                    range(len(task_data)),
                    task_data["score"],
                    color=sns.color_palette("husl", len(task_data)),
                )
                ax.set_xticks(range(len(task_data)))
                ax.set_xticklabels(task_data["library"], rotation=45, ha="right")
                ax.set_ylabel(score_label)
                ax.set_title(f"{task.title()} - {dataset.replace('_', ' ').title()}")

                # Add value labels on bars
                for bar, score in zip(bars, task_data["score"]):
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height,
                        f"{score:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )

    plt.tight_layout()
    plt.savefig(f"{save_path}/performance_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Time comparison
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Optimization Time Comparison", fontsize=16)

    for i, task in enumerate(tasks):
        ax = axes[i]
        task_data = df[df["task"] == task]

        if not task_data.empty:
            # Group by library and calculate mean time
            time_data = task_data.groupby("library")["time"].mean().sort_values()

            bars = ax.bar(
                range(len(time_data)),
                time_data.values,
                color=sns.color_palette("viridis", len(time_data)),
            )
            ax.set_xticks(range(len(time_data)))
            ax.set_xticklabels(time_data.index, rotation=45, ha="right")
            ax.set_ylabel("Time (seconds)")
            ax.set_title(f"{task.title()} Task")

            # Add value labels
            for bar, time_val in zip(bars, time_data.values):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{time_val:.1f}s",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    plt.tight_layout()
    plt.savefig(f"{save_path}/time_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 3. Score vs Time scatter plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Performance vs Time Trade-off", fontsize=16)

    for i, task in enumerate(tasks):
        ax = axes[i]
        task_data = df[df["task"] == task]

        if not task_data.empty:
            # Create scatter plot
            for library in task_data["library"].unique():
                lib_data = task_data[task_data["library"] == library]
                ax.scatter(
                    lib_data["time"], lib_data["score"], label=library, s=100, alpha=0.7
                )

            ax.set_xlabel("Time (seconds)")
            if task == "classification":
                ax.set_ylabel("Accuracy")
            else:
                ax.set_ylabel("MSE")
            ax.set_title(f"{task.title()} Task")
            ax.legend()
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_path}/performance_vs_time.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Plots saved to {save_path}/ directory")


def generate_summary_report(df, results):
    """Generate a detailed summary report."""
    report = []
    report.append("# Wine Quality Hyperparameter Tuning Benchmark Report")
    report.append("=" * 60)
    report.append("")

    # Overall statistics
    report.append("## Overall Statistics")
    report.append(f"- Total experiments run: {len(df)}")
    report.append(f"- Libraries tested: {', '.join(df['library'].unique())}")
    report.append(f"- Datasets: {', '.join(df['dataset'].unique())}")
    report.append(f"- Tasks: {', '.join(df['task'].unique())}")
    report.append("")

    # Best performers by task
    report.append("## Best Performers by Task")

    for task in df["task"].unique():
        task_data = df[df["task"] == task]
        report.append(f"\n### {task.title()}")

        for dataset in df["dataset"].unique():
            dataset_task_data = task_data[task_data["dataset"] == dataset]
            if not dataset_task_data.empty:
                if task == "classification":
                    best = dataset_task_data.loc[dataset_task_data["score"].idxmax()]
                    metric = "Accuracy"
                else:
                    best = dataset_task_data.loc[dataset_task_data["score"].idxmin()]
                    metric = "MSE"

                report.append(f"**{dataset.replace('_', ' ').title()}:**")
                report.append(
                    f"- Best: {best['library']} ({metric}: {best['score']:.4f}, Time: {best['time']:.2f}s)"
                )

    # Time efficiency
    report.append("\n## Time Efficiency")
    time_summary = (
        df.groupby("library")["time"].agg(["mean", "std"]).sort_values("mean")
    )
    for library, (mean_time, std_time) in time_summary.iterrows():
        report.append(f"- {library}: {mean_time:.2f}±{std_time:.2f}s")

    # Recommendations
    report.append("\n## Recommendations")

    # Best for accuracy
    classification_data = df[df["task"] == "classification"]
    if not classification_data.empty:
        best_acc_lib = classification_data.groupby("library")["score"].mean().idxmax()
        report.append(f"- **Best for Classification Accuracy**: {best_acc_lib}")

    # Best for regression
    regression_data = df[df["task"] == "regression"]
    if not regression_data.empty:
        best_reg_lib = regression_data.groupby("library")["score"].mean().idxmin()
        report.append(f"- **Best for Regression Performance**: {best_reg_lib}")

    # Fastest
    fastest_lib = df.groupby("library")["time"].mean().idxmin()
    report.append(f"- **Fastest Optimization**: {fastest_lib}")

    # Best balance
    report.append("\n### Performance-Time Balance")
    report.append("Consider your use case:")
    report.append("- For quick prototyping: Choose fastest library")
    report.append("- For best performance: Choose top-performing library for your task")
    report.append("- For production: Consider both performance and optimization time")

    return "\n".join(report)


def analyze_benchmark_results(results_file="results/benchmark/benchmark_results.json"):
    """Main function to analyze benchmark results."""
    print("Loading benchmark results...")
    results = load_benchmark_results(results_file)

    if results is None:
        return

    print("Converting to DataFrame...")
    df = create_results_dataframe(results)

    if df.empty:
        print("No valid results found in the benchmark data.")
        return

    print(f"Found {len(df)} valid benchmark results")

    # Display basic statistics
    print("\nBasic Statistics:")
    print(df.describe())

    # Create visualizations
    print("\nGenerating visualizations...")
    plot_performance_comparison(df)

    # Generate report
    print("\nGenerating summary report...")
    report = generate_summary_report(df, results)

    # Save report
    with open("results/benchmark/benchmark_summary_report.md", "w") as f:
        f.write(report)

    print("\nAnalysis complete!")
    print("- Plots saved to 'benchmark_plots/' directory")
    print("- Summary report saved to 'benchmark_summary_report.md'")

    return df, results


if __name__ == "__main__":
    analyze_benchmark_results()
