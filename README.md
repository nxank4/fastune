# Fastune

Fastune: High-Performance Hyperparameter Tuning in Python
Fastune is a Python library for hyperparameter tuning, offering improved performance over scikit-learn's `GridSearchCV` through optimized parallel processing. It integrates seamlessly with scikit-learn models.

## Installation

```bash
pip install fast-tune
```

## Example: Grid Search with Scikit-Learn

```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from fast_tune import GridSearchCV

X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
estimator = LogisticRegression(solver="liblinear")
param_grid = {"C": [0.1, 1.0, 10.0], "penalty": ["l1", "l2"]}
grid = GridSearchCV(estimator, param_grid, cv=5)
grid.fit(X, y)
print("Best Parameters:", grid.best_params)
print("Best Score:", grid.best_score)
```

## Features

- Drop-in replacement for scikit-learn's GridSearchCV
- Optimized parallel processing with Python's multiprocessing
- Compatible with all scikit-learn estimators
- Simple, Pythonic API

## Status

Under active development. See [Issues](https://github.com/<your-username>/Fastune/issues) for tasks.

## License

MIT
