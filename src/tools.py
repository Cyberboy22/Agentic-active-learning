"""The tool registry the agent orchestrates.

Keeping the capabilities behind a small, uniform tool interface is what makes
the agent's job clean: it reasons over *which tool to use next*, exactly as in
ChemCrow's Thought -> Action -> Observation loop, rather than hard-coding the
pipeline. Each tool is a plain function with a documented signature; the agent
selects among them by name.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from . import acquisition, oracle
from .surrogate import make_surrogate


@dataclass
class Tool:
    name: str
    description: str
    fn: Callable


def build_toolbox(seed: int = 0) -> dict[str, Tool]:
    """Return the dict of tools available to the agent, with descriptions.

    The descriptions are also what gets shown to the LLM, so it knows the menu.
    """

    def fit_surrogate(X, y, kind="random_forest"):
        """Fit a surrogate model to all experiments run so far."""
        return make_surrogate(kind, seed=seed).fit(np.asarray(X), np.asarray(y))

    def predict(model, X):
        """Predict mean strength and uncertainty for a pool of candidates."""
        return model.predict(np.atleast_2d(X))

    def propose_candidates(n, rng):
        """Sample a feasible candidate pool from the search space (a mini library)."""
        pool = oracle.sample_candidates(n, rng)
        return np.array([x for x in pool if oracle.is_feasible(x)])

    def score_candidates(name, mean, std, best, kappa):
        """Rank candidates by an acquisition function."""
        return acquisition.score(name, mean, std, best=best, kappa=kappa)

    def run_experiment(orc: oracle.Oracle, X):
        """Spend budget: measure the true (noisy) response for selected designs."""
        return orc.run_batch(np.atleast_2d(X))

    tools = [
        Tool("fit_surrogate", "Fit RF or GP surrogate to observed data.", fit_surrogate),
        Tool("predict", "Predict mean + uncertainty for candidates.", predict),
        Tool("propose_candidates", "Sample a feasible candidate pool.", propose_candidates),
        Tool("score_candidates", "Rank candidates by acquisition (ucb/ei/greedy/explore).", score_candidates),
        Tool("run_experiment", "Evaluate candidates on the expensive oracle.", run_experiment),
    ]
    return {t.name: t for t in tools}
