"""Standard synthetic optimisation benchmarks, written as maximisation problems.

These are the functions the optimisation literature uses to compare search
strategies. Including them means a planner is not judged on one hand-made
problem: the benchmark harness can ask whether an advantage holds across
landscapes with different character --

  branin     2D, smooth, three equally good global optima
  hartmann6  6D, moderately rugged, one sharp global optimum
  ackley     nD, a narrow needle at the origin surrounded by a near-flat plain
  rastrigin  nD, extremely multimodal, a regular lattice of local optima

Each exposes its exact known optimum, so "fraction of optimum reached" is a real
number rather than a Monte-Carlo estimate.
"""

from __future__ import annotations

import numpy as np

from ..objective import Objective, register_objective


@register_objective
class Branin(Objective):
    """Branin-Hoo, negated. Exact optimum 0 is reached at three distinct points."""

    name = "branin"
    noise_std = 0.5

    @property
    def var_names(self) -> list[str]:
        return ["x1", "x2"]

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return np.array([-5.0, 0.0]), np.array([10.0, 15.0])

    def evaluate(self, x: np.ndarray) -> float:
        x1, x2 = x
        a, b, c = 1.0, 5.1 / (4 * np.pi**2), 5.0 / np.pi
        r, s, t = 6.0, 10.0, 1.0 / (8 * np.pi)
        val = a * (x2 - b * x1**2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s
        return float(-val)  # maximise

    def reference_optimum(self, n_scan: int = 0, seed: int = 0) -> float:
        return -0.397887


@register_objective
class Hartmann6(Objective):
    """Hartmann 6-dimensional. Exact optimum 3.32237."""

    name = "hartmann6"
    noise_std = 0.05

    _ALPHA = np.array([1.0, 1.2, 3.0, 3.2])
    _A = np.array(
        [
            [10.0, 3.0, 17.0, 3.5, 1.7, 8.0],
            [0.05, 10.0, 17.0, 0.1, 8.0, 14.0],
            [3.0, 3.5, 1.7, 10.0, 17.0, 8.0],
            [17.0, 8.0, 0.05, 10.0, 0.1, 14.0],
        ]
    )
    _P = 1e-4 * np.array(
        [
            [1312, 1696, 5569, 124, 8283, 5886],
            [2329, 4135, 8307, 3736, 1004, 9991],
            [2348, 1451, 3522, 2883, 3047, 6650],
            [4047, 8828, 8732, 5743, 1091, 381],
        ]
    )

    @property
    def var_names(self) -> list[str]:
        return [f"x{i + 1}" for i in range(6)]

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros(6), np.ones(6)

    def evaluate(self, x: np.ndarray) -> float:
        inner = np.sum(self._A * (x - self._P) ** 2, axis=1)
        return float(np.sum(self._ALPHA * np.exp(-inner)))  # maximise

    def reference_optimum(self, n_scan: int = 0, seed: int = 0) -> float:
        return 3.32237


@register_objective
class Ackley(Objective):
    """Ackley, negated. Exact optimum 0 at the origin. A needle in a flat plain."""

    name = "ackley"
    noise_std = 0.1

    def __init__(self, dim: int = 4) -> None:
        self._dim = dim

    @property
    def var_names(self) -> list[str]:
        return [f"x{i + 1}" for i in range(self._dim)]

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return np.full(self._dim, -32.768), np.full(self._dim, 32.768)

    def evaluate(self, x: np.ndarray) -> float:
        n = len(x)
        s1 = np.sum(x**2) / n
        s2 = np.sum(np.cos(2 * np.pi * x)) / n
        val = -20.0 * np.exp(-0.2 * np.sqrt(s1)) - np.exp(s2) + 20.0 + np.e
        return float(-val)  # maximise

    def reference_optimum(self, n_scan: int = 0, seed: int = 0) -> float:
        return 0.0


@register_objective
class Rastrigin(Objective):
    """Rastrigin, negated. Exact optimum 0 at the origin. Densely multimodal."""

    name = "rastrigin"
    noise_std = 0.5

    def __init__(self, dim: int = 4) -> None:
        self._dim = dim

    @property
    def var_names(self) -> list[str]:
        return [f"x{i + 1}" for i in range(self._dim)]

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return np.full(self._dim, -5.12), np.full(self._dim, 5.12)

    def evaluate(self, x: np.ndarray) -> float:
        val = 10.0 * len(x) + np.sum(x**2 - 10.0 * np.cos(2 * np.pi * x))
        return float(-val)  # maximise

    def reference_optimum(self, n_scan: int = 0, seed: int = 0) -> float:
        return 0.0
