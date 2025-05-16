import random
from copy import deepcopy
from sklearn.base import clone  # type: ignore
from sklearn.model_selection import cross_val_score  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np


class PBTSearchCV:
    def __init__(
        self,
        estimator,
        param_dist,
        population_size=10,
        generations=10,
        cv=3,
        random_state=None,
        patience=5,  # Early stopping patience
    ):
        self.estimator = estimator
        self.param_dist = param_dist
        self.population_size = population_size
        self.generations = generations
        self.cv = cv
        self.random_state = random_state
        self.patience = patience
        self.history_ = []
        if self.random_state is not None:
            self._rng = np.random.RandomState(self.random_state)
        else:
            self._rng = np.random

    def _sample_params(self):
        params = {}
        for key, dist in self.param_dist.items():
            if hasattr(dist, "rvs"):
                params[key] = dist.rvs(random_state=self._rng)
            elif isinstance(dist, (list, tuple)):
                params[key] = self._rng.choice(dist)
            else:
                params[key] = dist
        return params

    def _evaluate(self, params, X, y):
        model = clone(self.estimator)
        model.set_params(**params)
        scores = cross_val_score(model, X, y, cv=self.cv)
        return float(scores.mean())

    def _tournament_select(self, pop, k=3, num_select=None):
        # Tournament selection: pick k random, select best, repeat
        if num_select is None:
            num_select = len(pop) // 2
        selected = []
        for _ in range(num_select):
            competitors = random.sample(pop, k)
            winner = max(competitors, key=lambda ind: ind["score"])
            selected.append(winner)
        return selected

    def _adaptive_mutation_prob(self, scores, min_prob=0.1, max_prob=0.5):
        # Higher stddev = more diversity = less mutation needed
        std = np.std(scores)
        norm_std = min(std / (np.mean(scores) + 1e-8), 1.0)
        # Invert: more diversity, less mutation
        return max_prob - (max_prob - min_prob) * norm_std

    def fit(self, X, y):
        # initialize population
        pop = []
        for _ in range(self.population_size):
            params = self._sample_params()
            score = self._evaluate(params, X, y)
            pop.append({"params": params, "score": score})
        self.history_.append([deepcopy(ind) for ind in pop])
        best_score = None
        best_gen = 0
        for gen in range(self.generations):
            # Tournament selection for diversity
            elites = self._tournament_select(
                pop, k=3, num_select=self.population_size // 2
            )
            # Adaptive mutation probability
            scores = [ind["score"] for ind in pop]
            mut_prob = self._adaptive_mutation_prob(scores)
            new_pop = elites.copy()
            for i in range(self.population_size - len(elites)):
                src = self._rng.choice(elites)
                child_params = deepcopy(src["params"])
                # Hybrid mutation: perturb numeric, re-sample categorical
                for key, dist in self.param_dist.items():
                    if self._rng.rand() < mut_prob:
                        val = child_params[key]
                        # Numeric: perturb
                        if hasattr(dist, "rvs") and isinstance(val, (int, float)):
                            # Small random factor (0.8-1.2)
                            factor = self._rng.uniform(0.8, 1.2)
                            new_val = val * factor
                            # Clamp to distribution bounds if possible
                            if hasattr(dist, "a") and hasattr(dist, "b"):
                                new_val = max(dist.a, min(dist.b, new_val))
                            child_params[key] = type(val)(new_val)
                        # Categorical: re-sample
                        elif isinstance(dist, (list, tuple)):
                            child_params[key] = self._rng.choice(dist)
                score = self._evaluate(child_params, X, y)
                new_pop.append({"params": child_params, "score": score})
            pop = new_pop
            self.history_.append([deepcopy(ind) for ind in pop])
            best_gen_score = max([ind["score"] for ind in pop])
            if best_score is None or best_gen_score > best_score:
                best_score = best_gen_score
                best_gen = gen
            elif gen - best_gen >= self.patience:
                # Early stopping
                break
        best = max(pop, key=lambda ind: ind["score"])
        self.best_params_ = best["params"]
        self.best_score_ = best["score"]
        return self

    def visualize(self):
        # plot score trajectories
        plt.figure(figsize=(8, 4))
        for gen, pop in enumerate(self.history_):
            scores = [ind["score"] for ind in pop]
            plt.scatter([gen] * len(scores), scores, alpha=0.6)
        # plot best per generation
        bests = [max([ind["score"] for ind in pop]) for pop in self.history_]
        plt.plot(range(len(self.history_)), bests, "-o", color="red", label="best")
        plt.xlabel("Generation")
        plt.ylabel("Score")
        plt.title("PBT Optimization Progress")
        plt.legend()
        plt.grid(True)
        plt.show()
        plt.close()
