"""A tool registry with schemas and call logging.

Any Python callable can be registered as a tool with a description and a
parameter schema. The campaign then performs *all* of its work through
`registry.call(...)`, which means two things:

1. **The menu is data.** `registry.describe()` renders the tool list as text an
   LLM can be shown, so adding a capability to the agent is one `@register`
   away -- the lab's own docking wrapper, assay client or property predictor
   plugs in here without touching the planner.

2. **Every invocation is recorded.** Each call logs its name, a compact argument
   summary, duration and success. Combined with the per-round rationales, that
   gives a complete provenance trail: not just what the agent decided, but what
   it actually ran, in order, and how long it took.
"""

from __future__ import annotations

import time
import traceback
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

import numpy as np


def _summarise(value: Any) -> str:
    """Compact, log-safe rendering of an argument (never dumps a whole array)."""
    if isinstance(value, np.ndarray):
        return f"ndarray{tuple(value.shape)}"
    if isinstance(value, (list, tuple)):
        return f"{type(value).__name__}[{len(value)}]"
    if isinstance(value, (int, float, bool, str)) or value is None:
        text = repr(value)
        return text if len(text) <= 40 else text[:37] + "..."
    return type(value).__name__


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, str]
    fn: Callable = field(repr=False)

    def signature(self) -> str:
        params = ", ".join(f"{k}: {v}" for k, v in self.parameters.items())
        return f"{self.name}({params})"


@dataclass
class ToolCall:
    name: str
    args: dict[str, str]
    duration_ms: float
    ok: bool
    error: str | None = None


class ToolRegistry:
    """Holds the tools an agent may use, and the log of what it used."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}
        self.call_log: list[ToolCall] = []

    # -- registration ------------------------------------------------------

    def register(self, name: str, description: str, parameters: dict[str, str] | None = None):
        """Decorator: register a callable as a tool.

        >>> registry = ToolRegistry()
        >>> @registry.register("dock", "Dock a ligand.", {"ligand": "path to file"})
        ... def dock(ligand): ...
        """

        def wrap(fn: Callable) -> Callable:
            self._tools[name] = ToolSpec(name, description, parameters or {}, fn)
            return fn

        return wrap

    def add(self, name: str, description: str, fn: Callable, parameters: dict[str, str] | None = None) -> None:
        """Imperative form of `register`, for wiring up tools at runtime."""
        self._tools[name] = ToolSpec(name, description, parameters or {}, fn)

    # -- use ---------------------------------------------------------------

    def call(self, name: str, /, **kwargs):
        """Invoke a tool by name, recording the call.

        `name` is positional-only so a tool is free to have its own `name`
        parameter without colliding with this one.
        """
        if name not in self._tools:
            raise KeyError(f"unknown tool {name!r}; registered: {sorted(self._tools)}")
        args = {k: _summarise(v) for k, v in kwargs.items()}
        start = time.perf_counter()
        try:
            result = self._tools[name].fn(**kwargs)
        except Exception as exc:
            self.call_log.append(
                ToolCall(name, args, (time.perf_counter() - start) * 1e3, False,
                         f"{type(exc).__name__}: {exc}")
            )
            traceback.clear_frames(exc.__traceback__) if exc.__traceback__ else None
            raise
        self.call_log.append(
            ToolCall(name, args, (time.perf_counter() - start) * 1e3, True)
        )
        return result

    # -- introspection -----------------------------------------------------

    def names(self) -> list[str]:
        return sorted(self._tools)

    def describe(self) -> str:
        """Render the tool menu as text (what you would show an LLM)."""
        return "\n".join(
            f"- {spec.signature()}\n    {spec.description}"
            for _, spec in sorted(self._tools.items())
        )

    def log_as_dicts(self) -> list[dict]:
        return [asdict(c) for c in self.call_log]

    def summary(self) -> dict[str, dict[str, float]]:
        """Per-tool call count and total time -- a cheap profile of the campaign."""
        out: dict[str, dict[str, float]] = {}
        for call in self.call_log:
            entry = out.setdefault(call.name, {"calls": 0, "total_ms": 0.0, "failures": 0})
            entry["calls"] += 1
            entry["total_ms"] += call.duration_ms
            entry["failures"] += 0 if call.ok else 1
        for entry in out.values():
            entry["total_ms"] = round(entry["total_ms"], 2)
        return out


def build_default_registry(objective, oracle, seed: int = 0) -> ToolRegistry:
    """Wire up the standard campaign toolset against one objective and oracle."""
    from .acquisition import score as acq_score
    from .surrogate import make_surrogate

    reg = ToolRegistry()

    reg.add(
        "fit_surrogate",
        "Fit a surrogate model (with uncertainty) to all observations so far.",
        lambda X, y, kind="random_forest": make_surrogate(kind, seed=seed).fit(X, y),
        {"X": "observed designs", "y": "observed values", "kind": "random_forest|gaussian_process"},
    )
    reg.add(
        "predict",
        "Predict mean value and uncertainty for a pool of candidate designs.",
        lambda model, X: model.predict(X),
        {"model": "fitted surrogate", "X": "candidate designs"},
    )
    reg.add(
        "propose_candidates",
        "Sample a feasible candidate pool: global exploration plus local refinement "
        "around the best designs found so far.",
        lambda n, rng, centers=None, scale=0.05: _propose(objective, n, rng, centers, scale),
        {"n": "pool size", "rng": "random generator", "centers": "designs to refine around",
         "scale": "local perturbation, fraction of each variable's range"},
    )
    reg.add(
        "score_candidates",
        "Rank candidates by an acquisition function (ucb|ei|greedy|explore|random).",
        lambda acquisition, mean, std, best, kappa, rng=None: acq_score(
            acquisition, mean, std, best=best, kappa=kappa, rng=rng
        ),
        {"acquisition": "ucb|ei|greedy|explore|random", "mean": "predicted means",
         "std": "uncertainties", "best": "best value so far",
         "kappa": "exploration strength"},
    )
    reg.add(
        "run_experiment",
        "Spend budget: measure the true (noisy) value of the selected designs.",
        lambda X: oracle.run_batch(X),
        {"X": "designs to evaluate"},
    )
    return reg


def _propose(objective, n: int, rng, centers, scale: float) -> np.ndarray:
    """Global pool + optional trust-region pool around promising designs."""
    pool = objective.sample(n, rng)
    if centers is not None and len(centers):
        local = np.vstack(
            [objective.local_sample(c, max(n // 4, 1), scale, rng) for c in np.atleast_2d(centers)]
        )
        pool = np.vstack([pool, local])
    feasible = np.array([x for x in pool if objective.is_feasible(x)])
    return feasible if len(feasible) else pool
