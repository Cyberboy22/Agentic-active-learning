#!/usr/bin/env python3
"""Plug your own experiment into the framework.

This is the whole integration surface: implement `var_names`, `bounds` and
`evaluate`, optionally `is_feasible`, and every planner, surrogate, acquisition
function, provenance record and report works against it unchanged.

`evaluate` is where your real experiment goes -- a simulation, a call to an
instrument queue, a scoring function, a cluster job. The framework only ever
reaches it through the budget-counting `Oracle`, so each call is accounted for.

Run:  python examples/custom_objective.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from src.campaign import run_campaign  # noqa: E402
from src.objective import Objective, register_objective  # noqa: E402


@register_objective
class ReactorYield(Objective):
    """A toy process-optimisation problem: maximise yield of a batch process.

    Three knobs (temperature, pressure, residence time) with a physical trade-off:
    hotter is faster but degrades the product, and a long residence time only pays
    off at moderate temperature.
    """

    name = "reactor_yield"
    noise_std = 0.8  # percentage points of yield

    @property
    def var_names(self) -> list[str]:
        return ["temperature", "pressure", "residence_time"]

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return np.array([60.0, 1.0, 5.0]), np.array([220.0, 20.0, 180.0])

    @property
    def units(self) -> list[str]:
        return ["°C", "bar", "min"]

    @property
    def value_unit(self) -> str:
        return "% yield"

    def is_feasible(self, x: np.ndarray) -> bool:
        """A safety envelope: high temperature is only allowed at low pressure."""
        temperature, pressure, _ = x
        return bool(temperature <= 180.0 or pressure <= 8.0)

    def evaluate(self, x: np.ndarray) -> float:
        temperature, pressure, time = x
        conversion = 68.0 / (1.0 + np.exp(-(temperature - 130.0) / 18.0))
        degradation = 0.02 * np.exp((temperature - 150.0) / 25.0) * np.sqrt(time)
        pressure_gain = 6.0 * np.tanh(pressure / 6.0)
        time_gain = 12.0 * (1.0 - np.exp(-time / 45.0))
        return float(np.clip(conversion + pressure_gain + time_gain - degradation, 0.0, 100.0))


if __name__ == "__main__":
    result = run_campaign(
        objective="reactor_yield", planner="heuristic",
        budget=40, n_init=8, n_rounds=10, seed=0,
    )
    print(f"\nBest yield found: {result.best_value:.2f} %")
    print("Best conditions:")
    for name, value, unit in zip(result.var_names, result.best_x, result.units):
        print(f"  {name:<16} {value:8.2f} {unit}")
    print(f"\nReference optimum: {result.reference_optimum:.2f} %")
    print(f"Tool calls made: {sum(int(s['calls']) for s in result.tool_summary.values())}")
