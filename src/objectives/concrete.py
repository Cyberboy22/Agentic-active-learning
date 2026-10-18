"""A benign materials-engineering objective: concrete mix design.

Maximise compressive strength by choosing the mix proportions and curing age.

The response is a self-contained, physically-motivated function -- Abrams'
water-to-binder law, the log-time strength gain of curing concrete, a
supplementary-cementitious-material term, an aggregate-grading penalty, and a
narrow "sweet spot" where a low water/binder ratio is kept workable by
superplasticizer. It is **not** a fitted engineering model and not a published
dataset; it exists so the repository runs with no network and so the optimum is
knowable, letting a planner's result be scored honestly.
"""

from __future__ import annotations

import numpy as np

from ..objective import Objective, register_objective


@register_objective
class ConcreteStrength(Objective):
    name = "concrete"
    noise_std = 1.5  # MPa

    _SPEC = [
        ("cement", 120.0, 540.0, "kg/m^3"),
        ("slag", 0.0, 360.0, "kg/m^3"),
        ("fly_ash", 0.0, 200.0, "kg/m^3"),
        ("water", 120.0, 250.0, "kg/m^3"),
        ("superplasticizer", 0.0, 32.0, "kg/m^3"),
        ("coarse_agg", 800.0, 1145.0, "kg/m^3"),
        ("fine_agg", 590.0, 992.0, "kg/m^3"),
        ("age", 3.0, 365.0, "days"),
    ]

    @property
    def var_names(self) -> list[str]:
        return [s[0] for s in self._SPEC]

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.array([s[1] for s in self._SPEC], dtype=float),
            np.array([s[2] for s in self._SPEC], dtype=float),
        )

    @property
    def units(self) -> list[str]:
        return [s[3] for s in self._SPEC]

    @property
    def value_unit(self) -> str:
        return "MPa"

    def is_feasible(self, x: np.ndarray) -> bool:
        """Binder must be present and the water/binder ratio physically sensible."""
        cement, slag, fly_ash, water = x[0], x[1], x[2], x[3]
        binder = cement + slag + fly_ash
        if binder <= 0:
            return False
        return bool(0.25 <= water / binder <= 0.90)

    def evaluate(self, x: np.ndarray) -> float:
        cement, slag, fly_ash, water, sp, coarse, fine, age = x
        binder = cement + slag + fly_ash
        w_b = water / max(binder, 1e-6)

        # Abrams-style decay with water/binder ratio.
        base = 95.0 * np.exp(-2.2 * w_b)
        # Supplementary cementitious materials help, with diminishing returns.
        scm = 10.0 * np.tanh((0.5 * slag + 0.7 * fly_ash) / 150.0)
        # Superplasticizer keeps a low-w/b mix workable: mild bonus.
        plast = 4.0 * np.tanh(sp / 12.0)
        # Log-time strength gain from curing.
        age_factor = 0.35 + 0.65 * np.log1p(age) / np.log1p(365.0)
        # Gentle penalty for a poorly graded aggregate skeleton.
        agg_penalty = 10.0 * (fine / max(coarse, 1e-6) - 0.65) ** 2
        # A localised optimum: specific low w/b, enough superplasticizer, long cure.
        sweet = 18.0 * np.exp(
            -0.5 * (((w_b - 0.32) / 0.035) ** 2 + ((sp - 20.0) / 5.0) ** 2)
        ) * (0.3 + 0.7 * np.log1p(age) / np.log1p(365.0))

        return float(max((base + scm + plast) * age_factor - agg_penalty + sweet, 0.0))
