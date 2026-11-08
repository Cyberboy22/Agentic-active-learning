"""The pluggable objective interface -- the seam a lab plugs its own science into.

An `Objective` describes *what* is being optimised: the variables, their bounds,
any feasibility constraint, and how to evaluate a design. It knows nothing about
agents, surrogates or budgets.

An `Oracle` wraps an objective and adds the two things that make a campaign a
campaign: a **budget counter** (every evaluation costs) and **measurement noise**.

Separating these means the whole framework -- planners, surrogates, acquisition,
provenance, reporting -- is domain-agnostic. To use it on your own problem you
implement one class with two methods; nothing else changes. See
`examples/custom_objective.py`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


class Objective(ABC):
    """A black-box function to maximise over a bounded, optionally constrained space."""

    #: short identifier used in reports and the registry
    name: str = "objective"
    #: typical measurement noise of this experiment (same units as `evaluate`)
    noise_std: float = 0.0

    @property
    @abstractmethod
    def var_names(self) -> list[str]:
        """Names of the design variables, in order."""

    @property
    @abstractmethod
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """(low, high) arrays, each of length `dim`."""

    @abstractmethod
    def evaluate(self, x: np.ndarray) -> float:
        """Noise-free ground-truth value of design `x`. Higher is better."""

    # -- optional hooks ----------------------------------------------------

    def is_feasible(self, x: np.ndarray) -> bool:
        """Cheap constraint checked before spending budget. Default: all feasible."""
        return True

    @property
    def units(self) -> list[str]:
        """Optional units for reporting, one per variable."""
        return [""] * self.dim

    @property
    def value_unit(self) -> str:
        """Optional unit of the objective value, for reporting."""
        return ""

    # -- derived -----------------------------------------------------------

    @property
    def dim(self) -> int:
        low, _ = self.bounds
        return len(low)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Uniform random designs from the box (not filtered for feasibility)."""
        low, high = self.bounds
        return rng.uniform(low, high, size=(n, self.dim))

    def sample_feasible(self, n: int, rng: np.random.Generator, max_tries: int = 50) -> np.ndarray:
        """Uniform random *feasible* designs. Falls back to the box if constraints bite."""
        out: list[np.ndarray] = []
        for _ in range(max_tries):
            for x in self.sample(max(n * 2, 16), rng):
                if self.is_feasible(x):
                    out.append(x)
                    if len(out) == n:
                        return np.array(out)
        # Constraint too tight to satisfy by rejection: return what we have.
        return np.array(out) if out else self.sample(n, rng)

    def local_sample(
        self, center: np.ndarray, n: int, scale: float, rng: np.random.Generator
    ) -> np.ndarray:
        """Designs near `center` (trust region). `scale` is a fraction of each range."""
        low, high = self.bounds
        pts = center + rng.normal(0.0, scale, size=(n, self.dim)) * (high - low)
        return np.clip(pts, low, high)

    def reference_optimum(self, n_scan: int = 200_000, seed: int = 12345) -> float:
        """Monte-Carlo estimate of the best achievable noise-free value.

        Reporting only -- never used during a campaign. Override with the exact
        optimum when it is known analytically.
        """
        rng = np.random.default_rng(seed)
        best = -np.inf
        for x in self.sample(n_scan, rng):
            if self.is_feasible(x):
                best = max(best, self.evaluate(x))
        return float(best)


@dataclass
class Oracle:
    """Budget-limited, noisy access to an objective. The agent only ever sees this."""

    objective: Objective
    seed: int = 0
    noise_std: float | None = None  # overrides objective.noise_std when set
    n_calls: int = field(default=0, init=False)
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        if self.noise_std is None:
            self.noise_std = self.objective.noise_std

    def run(self, x: np.ndarray) -> float:
        """Spend one unit of budget; return a noisy measurement."""
        self.n_calls += 1
        value = self.objective.evaluate(np.asarray(x, dtype=float))
        if self.noise_std:
            value += float(self._rng.normal(0.0, self.noise_std))
        return float(value)

    def run_batch(self, X: np.ndarray) -> np.ndarray:
        return np.array([self.run(x) for x in np.atleast_2d(X)])


# --------------------------------------------------------------------------
# Registry: look objectives up by name from the CLI / benchmark harness.
# --------------------------------------------------------------------------

_REGISTRY: dict[str, type[Objective]] = {}


def register_objective(cls: type[Objective]) -> type[Objective]:
    """Class decorator; makes an objective available by `cls.name`."""
    _REGISTRY[cls.name] = cls
    return cls


def get_objective(name: str, **kwargs) -> Objective:
    if name not in _REGISTRY:
        raise ValueError(f"unknown objective {name!r}; available: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


def available_objectives() -> list[str]:
    return sorted(_REGISTRY)


# --------------------------------------------------------------------------
# Pool objectives: the discrete half of the interface.
#
# A box objective lets you invent a design anywhere inside its bounds. That is
# the wrong model for screening, where the molecules that exist are the
# molecules you can order and the decision is *which rows of a fixed table* to
# pay for. A PoolObjective keeps exactly the same surface the rest of the
# framework consumes -- var_names, bounds, sample, local_sample, evaluate -- so
# every planner, surrogate, acquisition function and provenance record works
# against it unchanged, while underneath it only ever returns real library
# members.
#
# Two things get *better* in the discrete setting rather than worse:
#   * the optimum is known exactly, so no Monte-Carlo reference scan is needed;
#   * "refine around the best result" becomes analogue search -- the nearest
#     catalogue entries to a hit -- which is a chemically meaningful move, where
#     a Gaussian nudge in descriptor space is not.
# --------------------------------------------------------------------------


class PoolObjective(Objective):
    """A finite candidate library. Designs are rows of a fixed feature matrix."""

    name = "pool"

    def __init__(self, pool, true_values: np.ndarray, name: str | None = None,
                 noise_std: float = 0.0):
        self.pool = pool
        self.true_values = np.asarray(true_values, dtype=float)
        if len(self.true_values) != len(pool):
            raise ValueError("true_values must have one entry per pool member")
        if name:
            self.name = name
        self.noise_std = noise_std

    # -- the Objective surface --------------------------------------------

    @property
    def var_names(self) -> list[str]:
        return list(self.pool.feature_names)

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Feature extent of the library. Advisory only -- nothing is generated."""
        return self.pool.features.min(axis=0), self.pool.features.max(axis=0)

    @property
    def dim(self) -> int:
        return self.pool.dim

    def evaluate(self, x: np.ndarray) -> float:
        return float(self.true_values[self.pool.index_of(x)])

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw n library members uniformly, without replacement where possible."""
        size = min(n, len(self.pool))
        idx = rng.choice(len(self.pool), size=size, replace=False)
        return self.pool.features[idx]

    def local_sample(self, center: np.ndarray, n: int, scale: float,
                     rng: np.random.Generator) -> np.ndarray:
        """Analogue search: the library members most similar to `center`.

        `scale` widens the neighbourhood rather than the step size -- a larger
        scale considers more distant analogues, which is the discrete
        counterpart of a bigger trust region.
        """
        width = max(int(round(n * (1.0 + 10.0 * scale))), n)
        pool_idx = self.pool.neighbours(center, width, rng)
        if len(pool_idx) > n:
            pool_idx = rng.choice(pool_idx, size=n, replace=False)
        return self.pool.features[pool_idx]

    def reference_optimum(self, n_scan: int = 0, seed: int = 0) -> float:
        """Exact -- in a finite library the best value is simply known."""
        return float(self.true_values.max())

    # -- pool-specific -----------------------------------------------------

    def indices_of(self, X: np.ndarray) -> list[int]:
        return [self.pool.index_of(row) for row in np.atleast_2d(X)]

    def true_top_k(self, k: int) -> np.ndarray:
        return np.argsort(self.true_values)[::-1][:k]

    def smiles_at(self, index: int) -> str:
        return self.pool.smiles[int(index)]
