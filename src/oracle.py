"""The expensive black-box experiment.

This module defines the search space and a simulated "experiment" that is
costly to evaluate. The agent never sees the ground-truth function: it may
only call `Oracle.run`, which returns a noisy measurement and increments a
budget counter. That is exactly the setting of a real screening campaign,
where every evaluation (a wet-lab assay, a long simulation) costs time or money.

Example domain (benign): concrete compressive strength as a function of the
mix design. The ground-truth response below is a smooth, physically-motivated
surrogate inspired by Abrams' water-to-cement law and the log-time strength
gain of curing concrete. It is NOT a published dataset; it is a self-contained
function so the repository runs with no network access and so the optimum is
known and the agent's performance can be scored honestly.

To retarget the framework to another problem, replace `SEARCH_SPACE` and
`Oracle._ground_truth` with your own variables and evaluator.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Each variable: (name, low, high, unit). Continuous bounded design space.
SEARCH_SPACE: list[tuple[str, float, float, str]] = [
    ("cement", 120.0, 540.0, "kg/m^3"),
    ("slag", 0.0, 360.0, "kg/m^3"),
    ("fly_ash", 0.0, 200.0, "kg/m^3"),
    ("water", 120.0, 250.0, "kg/m^3"),
    ("superplasticizer", 0.0, 32.0, "kg/m^3"),
    ("coarse_agg", 800.0, 1145.0, "kg/m^3"),
    ("fine_agg", 590.0, 992.0, "kg/m^3"),
    ("age", 3.0, 365.0, "days"),
]

VAR_NAMES = [v[0] for v in SEARCH_SPACE]
LOW = np.array([v[1] for v in SEARCH_SPACE], dtype=float)
HIGH = np.array([v[2] for v in SEARCH_SPACE], dtype=float)
DIM = len(SEARCH_SPACE)


def sample_candidates(n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw `n` uniform random points from the search space (a mini 'library')."""
    return rng.uniform(LOW, HIGH, size=(n, DIM))


def local_candidates(
    center: np.ndarray, n: int, scale: float, rng: np.random.Generator
) -> np.ndarray:
    """Draw `n` points near `center` by Gaussian perturbation (a trust region).

    `scale` is a fraction of each variable's range (e.g. 0.1 = 10%). This lets
    the agent *refine around promising hits* instead of only sampling globally,
    which is what makes it possible to home in on a narrow optimum.
    """
    span = HIGH - LOW
    pts = center + rng.normal(0.0, scale, size=(n, DIM)) * span
    return np.clip(pts, LOW, HIGH)


def is_feasible(x: np.ndarray) -> bool:
    """A cheap, benign feasibility constraint checked before an experiment.

    Mirrors the 'property filter' step of a screening pipeline: candidates
    that violate a simple domain rule are discarded without spending budget.
    Here: total binder must be sane and the water-to-binder ratio physical.
    """
    cement, slag, fly_ash, water = x[0], x[1], x[2], x[3]
    binder = cement + slag + fly_ash
    if binder <= 0:
        return False
    w_b = water / binder
    return bool(0.25 <= w_b <= 0.90)


@dataclass
class Oracle:
    """A budget-limited, noisy evaluator of the expensive experiment."""

    noise_std: float = 1.5  # measurement noise, in MPa
    seed: int = 0
    n_calls: int = field(default=0, init=False)
    _rng: np.random.Generator = field(init=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    @staticmethod
    def _ground_truth(x: np.ndarray) -> float:
        """Noise-free response in MPa. The agent must not call this directly."""
        cement, slag, fly_ash, water, sp, coarse, fine, age = x
        binder = cement + slag + fly_ash
        w_b = water / max(binder, 1e-6)

        # Abrams-style decay with water/binder ratio.
        base = 95.0 * np.exp(-2.2 * w_b)
        # Supplementary cementitious materials help, with diminishing returns.
        scm = 10.0 * np.tanh((0.5 * slag + 0.7 * fly_ash) / 150.0)
        # Superplasticizer lets you keep workability at low w/b: mild bonus.
        plast = 4.0 * np.tanh(sp / 12.0)
        # Log-time strength gain from curing.
        age_factor = 0.35 + 0.65 * np.log1p(age) / np.log1p(365.0)
        # Gentle penalty for a poorly graded aggregate skeleton.
        agg_ratio = fine / max(coarse, 1e-6)
        agg_penalty = 10.0 * (agg_ratio - 0.65) ** 2

        # A localised "sweet spot": peak strength needs a specific low water/binder
        # ratio, enough superplasticizer to stay workable there, and long curing.
        # Random sampling rarely satisfies all three at once, but the surrogate can
        # learn the trend and climb to it -- which is where active learning pays off.
        sweet = 18.0 * np.exp(
            -0.5 * (((w_b - 0.32) / 0.035) ** 2 + ((sp - 20.0) / 5.0) ** 2)
        ) * (0.3 + 0.7 * np.log1p(age) / np.log1p(365.0))

        strength = (base + scm + plast) * age_factor - agg_penalty + sweet
        return float(max(strength, 0.0))

    def run(self, x: np.ndarray) -> float:
        """Evaluate one candidate; returns a noisy measurement, costs 1 budget."""
        x = np.asarray(x, dtype=float)
        self.n_calls += 1
        true = self._ground_truth(x)
        return true + float(self._rng.normal(0.0, self.noise_std))

    def run_batch(self, X: np.ndarray) -> np.ndarray:
        return np.array([self.run(x) for x in np.atleast_2d(X)])

    def reference_optimum(self, n_scan: int = 200_000, seed: int = 12345) -> float:
        """Monte-Carlo estimate of the best achievable (noise-free) strength.

        Used only for reporting, to express the agent's result as a % of the
        best found by a huge brute-force scan. Never used during the campaign.
        """
        rng = np.random.default_rng(seed)
        X = rng.uniform(LOW, HIGH, size=(n_scan, DIM))
        best = 0.0
        for x in X:
            if is_feasible(x):
                best = max(best, self._ground_truth(x))
        return best
