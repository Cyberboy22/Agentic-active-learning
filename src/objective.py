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
