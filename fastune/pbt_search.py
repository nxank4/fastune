import random
from copy import deepcopy
from sklearn.base import clone  # type: ignore
from sklearn.model_selection import cross_val_score  # type: ignore
import matplotlib.pyplot as plt  # type: ignore


class PBTSearchCV:
    def __init__(
        self,
        estimator,
        param_dist,
        population_size=10,
        generations=10,
        cv=3,
        random_state=None,
    ):
        self.estimator = estimator
        self.param_dist = param_dist
        self.population_size = population_size
        self.generations = generations
        self.cv = cv
        self.random_state = random_state
        self.history_ = []
        if self.random_state is not None:
            random.seed(self.random_state)

    def _sample_params(self):
        params = {}
        for key, dist in self.param_dist.items():
            if hasattr(dist, "rvs"):
                params[key] = dist.rvs(random_state=self.random_state)
            elif isinstance(dist, (list, tuple)):
                params[key] = random.choice(dist)
            else:
                params[key] = dist
        return params

    def _evaluate(self, params, X, y):
        model = clone(self.estimator)
        model.set_params(**params)
        scores = cross_val_score(model, X, y, cv=self.cv)
        return float(scores.mean())

    def fit(self, X, y):
        # initialize population
        pop = []
        for _ in range(self.population_size):
            params = self._sample_params()
            score = self._evaluate(params, X, y)
            pop.append({"params": params, "score": score})
        # record history
        self.history_.append([ind["score"] for ind in pop])
        # PBT loop
        half = self.population_size // 2
        for gen in range(self.generations):
            # exploit: select top half
            pop = sorted(pop, key=lambda ind: ind["score"], reverse=True)
            elites = pop[:half]
            # replace worst half with copies of elites
            new_pop = elites.copy()
            for i in range(self.population_size - half):
                src = random.choice(elites)
                child_params = deepcopy(src["params"])
                # explore: re-sample each param with prob 0.3
                for key, dist in self.param_dist.items():
                    if random.random() < 0.3:
                        if hasattr(dist, "rvs"):
                            child_params[key] = dist.rvs(random_state=self.random_state)
                        elif isinstance(dist, (list, tuple)):
                            child_params[key] = random.choice(dist)
                score = self._evaluate(child_params, X, y)
                new_pop.append({"params": child_params, "score": score})
            pop = new_pop
            self.history_.append([ind["score"] for ind in pop])
        # finalize best
        best = max(pop, key=lambda ind: ind["score"])
        self.best_params_ = best["params"]
        self.best_score_ = best["score"]
        return self

    def visualize(self):
        # plot score trajectories
        plt.figure(figsize=(8, 4))
        for gen, scores in enumerate(self.history_):
            plt.scatter([gen] * len(scores), scores, alpha=0.6)
        # plot best per generation
        bests = [max(scores) for scores in self.history_]
        plt.plot(range(len(self.history_)), bests, "-o", color="red", label="best")
        plt.xlabel("Generation")
        plt.ylabel("Score")
        plt.title("PBT Optimization Progress")
        plt.legend()
        plt.grid(True)
        plt.show()
        plt.close()
