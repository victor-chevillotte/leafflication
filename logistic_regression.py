from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt


class LogisticRegression:

    def __init__(
        self,
        learning_rate: float,
        max_iterations: int,
    ):
        self.learning_rate: float = learning_rate
        self.max_iterations: int = max_iterations
        self.weights: list = []
        self.costs: Dict[str, List[float]] = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> list:
        X = np.insert(X, 0, 1, axis=1)

        for house in np.unique(y):
            current_house_vs_all = np.where(y == house, 1, 0)
            house_weights = np.zeros(X.shape[1])
            house_cost = []

            for _ in range(self.max_iterations):
                output = np.dot(X, house_weights)
                p = self._sigmoid(output)
                errors = current_house_vs_all - p
                gradient = np.dot(X.T, errors)
                house_weights += self.learning_rate * gradient
                house_cost.append(self._cost_function(p, current_house_vs_all))

            self.weights.append((house_weights, house))
            self.costs[house] = house_cost
        return self.weights

    def _predict(self, X: np.ndarray) -> np.ndarray:
        return [self._predict_one(i) for i in np.insert(X, 0, 1, axis=1)]

    def _predict_one(self, grades: np.ndarray) -> str:
        max_probability = (-10, 0)

        for weight, house in self.weights:
            if (grades.dot(weight), house) > max_probability:
                max_probability = (grades.dot(weight), house)

        return max_probability[1]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return sum(self._predict(X) == y) / len(y)

    def _sigmoid(self, X: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-X))

    def _cost_function(self, h, y):
        m = len(y)
        # prevent division by zero (log(0) is undefined)
        h = np.clip(h, 1e-10, 1 - 1e-10)
        cost = (1 / m) * (
            np.sum(-y.T.dot(np.log(h)) - (1 - y).T.dot(np.log(1 - h)))
        )
        return cost

    def plot_cost(self):
        for house in self.costs:
            plt.plot(self.costs[house], label=house)
        plt.xlabel("Number of iterations")
        plt.ylabel("Cost")
        plt.legend()
        try:
            plt.show()
        except KeyboardInterrupt:
            print("Interrupted by user")
