"""Acquisition functions: turn (mean, std) predictions into a priority score.

Each function scores a pool of candidates so the top-k can be selected for the
next round. They trade off exploitation (high predicted value) against
exploration (high uncertainty). A planner chooses which one to use, and how hard
to explore, each round.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def ucb(mean: np.ndarray, std: np.ndarray, *, kappa: float = 1.5, **_) -> np.ndarray:
    """Upper Confidence Bound: mean + kappa * std. Higher kappa = more exploration."""
    return mean + kappa * std


def expected_improvement(
    mean: np.ndarray, std: np.ndarray, *, best: float, xi: float = 0.01, **_
) -> np.ndarray:
    """Expected Improvement over the best value seen so far (maximisation)."""
    std = np.maximum(std, 1e-9)
    imp = mean - best - xi
    z = imp / std
    return imp * norm.cdf(z) + std * norm.pdf(z)


def greedy(mean: np.ndarray, std: np.ndarray, **_) -> np.ndarray:
    """Pure exploitation: trust the surrogate's mean prediction."""
    return mean


def pure_explore(mean: np.ndarray, std: np.ndarray, **_) -> np.ndarray:
    """Pure exploration: go where the model is most uncertain."""
    return std


def random_score(
    mean: np.ndarray, std: np.ndarray, *, rng: np.random.Generator | None = None, **_
) -> np.ndarray:
    """Ignore the model entirely -- the random-search baseline, inside the same loop.

    Scoring randomly makes the top-k selection a uniform random draw from the
    pool, so random search goes through exactly the same machinery (pool
    sampling, batching, budget accounting) as every other planner. That keeps
    the comparison fair: the only thing that differs is the decision rule.
    """
    rng = rng or np.random.default_rng()
    return rng.random(len(mean))


ACQUISITIONS = {
    "ucb": ucb,
    "ei": expected_improvement,
    "greedy": greedy,
    "explore": pure_explore,
    "random": random_score,
}


def score(
    name: str,
    mean,
    std,
    *,
    best: float,
    kappa: float = 1.5,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    if name not in ACQUISITIONS:
        raise ValueError(f"unknown acquisition {name!r}; choose from {list(ACQUISITIONS)}")
    return ACQUISITIONS[name](mean, std, kappa=kappa, best=best, rng=rng)
