from fastune.pbt_search import PBTSearchCV
from sklearn.linear_model import LogisticRegression  # type: ignore
from scipy.stats import uniform  # type: ignore
from sklearn.datasets import make_classification  # type: ignore


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
