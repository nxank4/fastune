from fastune.pbt_search import PBTSearchCV
from sklearn.linear_model import LogisticRegression  # type: ignore
from scipy.stats import uniform  # type: ignore
from sklearn.datasets import make_classification  # type: ignore
import numpy as np


def test_pbt_search_basic():
    X, y = make_classification(n_samples=200, n_features=5, random_state=0)
    estimator = LogisticRegression(solver="liblinear")
    param_dist = {"C": uniform(0.1, 1)}
    search = PBTSearchCV(
        estimator, param_dist, population_size=4, generations=2, cv=2, random_state=0
    )
    search.fit(X, y)
    assert hasattr(search, "best_params_")
    assert hasattr(search, "best_score_")
    assert 0.0 <= search.best_score_ <= 1.0


def test_visualize_runs_without_error(monkeypatch):
    # Use n_features > n_informative + n_redundant (default 2+2=4)
    X, y = make_classification(n_samples=100, n_features=5, random_state=1)
    estimator = LogisticRegression(solver="liblinear")
    param_dist = {"C": [0.1, 1]}
    search = PBTSearchCV(
        estimator, param_dist, population_size=3, generations=1, cv=2, random_state=1
    )
    search.fit(X, y)
    import matplotlib.pyplot as plt  # type: ignore

    called = {}

    def fake_show():
        called["show"] = True

    monkeypatch.setattr(plt, "show", fake_show)
    search.visualize()
    assert called.get("show", False)


def test_hybrid_mutation_changes_params():
    X, y = make_classification(n_samples=100, n_features=4, random_state=42)
    estimator = LogisticRegression(solver="liblinear")
    param_dist = {"C": uniform(0.1, 1), "max_iter": [100, 200]}
    search = PBTSearchCV(
        estimator, param_dist, population_size=6, generations=1, cv=2, random_state=42
    )
    search.fit(X, y)
    # After one generation, check that at least one child differs from its parent in at least one param
    initial_params = [ind["params"] for ind in search.history_[0]]
    next_params = [ind["params"] for ind in search.history_[1]]
    # Only check children (not elites)
    mutated = False
    for child in next_params[len(initial_params) // 2 :]:
        if not any(child == parent for parent in initial_params):
            mutated = True
            break
    assert mutated, "Hybrid mutation did not change any offspring parameters."


def test_early_stopping():
    X, y = make_classification(n_samples=100, n_features=4, random_state=123)
    estimator = LogisticRegression(solver="liblinear")
    param_dist = {"C": uniform(0.1, 1)}
    # Set patience low to trigger early stopping
    search = PBTSearchCV(
        estimator,
        param_dist,
        population_size=4,
        generations=20,
        cv=2,
        random_state=123,
        patience=1,
    )
    search.fit(X, y)
    # Should not run all generations if early stopping works
    assert len(search.history_) < 21


def test_reproducibility():
    X, y = make_classification(n_samples=100, n_features=4, random_state=99)
    estimator = LogisticRegression(solver="liblinear")
    param_dist = {"C": uniform(0.1, 1)}
    search1 = PBTSearchCV(
        estimator, param_dist, population_size=4, generations=3, cv=2, random_state=99
    )
    search2 = PBTSearchCV(
        estimator, param_dist, population_size=4, generations=3, cv=2, random_state=99
    )
    search1.fit(X, y)
    search2.fit(X, y)
    assert search1.best_params_ == search2.best_params_
    assert search1.best_score_ == search2.best_score_
    assert search1.history_ == search2.history_


def test_best_params_and_score_are_valid():
    X, y = make_classification(n_samples=100, n_features=4, random_state=7)
    estimator = LogisticRegression(solver="liblinear")
    param_dist = {"C": uniform(0.1, 1)}
    search = PBTSearchCV(
        estimator, param_dist, population_size=4, generations=2, cv=2, random_state=7
    )
    search.fit(X, y)
    assert isinstance(search.best_params_, dict)
    assert isinstance(search.best_score_, float)
    assert 0.0 <= search.best_score_ <= 1.0


def test_categorical_mutation():
    X, y = make_classification(n_samples=100, n_features=4, random_state=55)
    estimator = LogisticRegression(solver="liblinear")
    param_dist = {"C": [0.1, 1, 10], "max_iter": [100, 200]}
    search = PBTSearchCV(
        estimator, param_dist, population_size=6, generations=1, cv=2, random_state=55
    )
    search.fit(X, y)
    initial_params = [ind["params"] for ind in search.history_[0]]
    next_params = [ind["params"] for ind in search.history_[1]]
    mutated = False
    for child in next_params[len(initial_params) // 2 :]:
        if not any(child == parent for parent in initial_params):
            mutated = True
            break
    assert mutated, "Categorical mutation did not change any offspring parameters."


def test_parallel_evaluation_runs_and_returns_results():
    import time
    from sklearn.base import BaseEstimator, ClassifierMixin

    class SlowEstimator(BaseEstimator, ClassifierMixin):
        def __init__(self, dummy=None):
            self.dummy = dummy

        def fit(self, X, y):
            time.sleep(0.1)  # Simulate slow fit
            return self

        def predict(self, X):
            return np.zeros(X.shape[0], dtype=int)

    X, y = make_classification(n_samples=40, n_features=4, random_state=123)
    estimator = SlowEstimator()
    param_dist = {"dummy": [1, 2, 3, 4]}
    search = PBTSearchCV(
        estimator, param_dist, population_size=4, generations=2, cv=2, random_state=123
    )
    start = time.time()
    search.fit(X, y)
    elapsed = (
        time.time() - start
    )  # Should not take 0.1 * population_size * generations seconds (should be faster due to parallelism)
    assert elapsed < 0.1 * 4 * 2 * 1.5  # Allow more overhead for parallelism
    assert hasattr(search, "best_params_")
    assert hasattr(search, "best_score_")


def test_logging_creates_csv_and_logs_content(tmp_path):
    import pandas as pd

    X, y = make_classification(n_samples=40, n_features=4, random_state=42)
    estimator = LogisticRegression(solver="liblinear")
    param_dist = {"C": [0.1, 1, 10]}
    log_file = tmp_path / "pbt_log.csv"
    search = PBTSearchCV(
        estimator,
        param_dist,
        population_size=3,
        generations=1,
        cv=2,
        random_state=42,
        log_to=str(log_file),
    )
    search.fit(X, y)
    # Check file exists and has expected columns
    df = pd.read_csv(log_file)
    assert set(
        ["generation", "individual", "C", "score", "elapsed_sec", "memory_mb"]
    ).issubset(df.columns)
    # There should be at least one row per individual per generation
    assert len(df) >= 3 * 2
    # Scores should be floats and in [0, 1]
    assert df["score"].between(0, 1).all()
