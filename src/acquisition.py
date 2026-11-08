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


# --------------------------------------------------------------------------
# Cost-aware acquisition.
#
# Every function above assumes each evaluation costs the same, so maximising
# value per experiment is the same as maximising value per unit of budget. Once
# a campaign can choose between methods whose prices differ by orders of
# magnitude that stops being true, and the ranking has to be per unit cost --
# otherwise the most expensive method always looks best, because it is the most
# informative, and the budget is gone after a handful of calls.
# --------------------------------------------------------------------------


def per_cost(scores: np.ndarray, cost: float, exponent: float = 1.0) -> np.ndarray:
    """Divide an acquisition score by what it costs to obtain it.

    `exponent` tunes how hard cost is penalised: 0 ignores price entirely and
    recovers the cost-blind behaviour, 1 is strict value-per-unit-cost.
    """
    cost = max(float(cost), 1e-9)
    return np.asarray(scores, dtype=float) / (cost ** float(exponent))


def information_per_cost(std: np.ndarray, fidelity: float, cost: float) -> np.ndarray:
    """Expected uncertainty reduction per unit cost, for choosing *which method*.

    A measurement at correlation `rho` can at best remove a `rho**2` share of
    the variance at a point. Dividing that by price gives a comparable number
    across rungs of the ladder: it is what says a near-free ranking pass over
    many compounds can beat one expensive co-folding run, and also when it
    cannot.
    """
    return (np.asarray(std, dtype=float) ** 2) * float(fidelity) ** 2 / max(float(cost), 1e-9)
