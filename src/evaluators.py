"""Evaluators: the methods a campaign can pay to learn something with.

The original framework had one oracle and counted calls. Real screening has a
*ladder* of methods sitting at different points on a cost-accuracy curve --
embedding retrieval costs almost nothing and only ranks; docking costs real CPU
and returns a pose; a co-folding model costs more again; an assay costs days.
No single rung is correct everywhere, which is the entire reason a campaign
needs something deciding which rung to stand on.

So budget here is **cost**, not a count of calls, and a planner's decision
includes *which evaluator to spend it with*.

How fidelity is modelled
------------------------
Each evaluator reports an estimate of the true value. Writing both in
standardised units, an evaluator with fidelity ``rho`` returns

    y = rho * z_true + sqrt(1 - rho^2) * e ,     e ~ N(0, 1)

so ``rho`` is exactly the correlation between what the method reports and the
truth, and the ladder degrades smoothly from ``rho = 1`` (the assay) down to a
crude ranking. ``rho`` is **per target**, not global: the co-folding rung is
excellent on one target and little better than chance on another, which is the
behaviour the Mirgaux co-folding paper reports and measures (concordance
ranging from essentially nothing to strong, depending on the system).

That per-target variation is not decoration. It is the thing that makes method
selection a real decision rather than a fixed ordering.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Evaluator:
    """One rung of the cost-accuracy ladder."""

    name: str
    cost: float
    fidelity: float
    description: str = ""
    #: an evaluator at or above this fidelity is trusted enough to *confirm* a hit
    confirms: bool = False

    def measure(self, z_true: float, rng: np.random.Generator) -> float:
        """A standardised observation of a standardised truth."""
        rho = float(np.clip(self.fidelity, 0.0, 1.0))
        return rho * z_true + np.sqrt(max(1.0 - rho**2, 0.0)) * rng.normal()


#: The default ladder, named after the four approaches on the reading list.
#: Costs are in abstract "compute units" calibrated so one docking run is 1.0.
DEFAULT_LADDER: list[Evaluator] = [
    Evaluator("retrieval", cost=0.002, fidelity=0.45,
              description="Embedding dot product (DrugCLIP-style): ranks only, no pose, near-free."),
    Evaluator("ml_score", cost=0.05, fidelity=0.65,
              description="Learned scoring function: cheap, no pose, moderate agreement."),
    Evaluator("docking", cost=1.0, fidelity=0.85, confirms=True,
              description="Classical docking (Vina-class): pose plus score, the workhorse cost unit."),
    Evaluator("cofolding", cost=8.0, fidelity=0.75, confirms=True,
              description="Co-folding model (AF3/Boltz-class): expensive, excellent on some targets and near-useless on others."),
    Evaluator("assay", cost=400.0, fidelity=1.0, confirms=True,
              description="Wet-lab measurement: ground truth, and the reason the rest of the ladder exists."),
]


def ladder_for_target(target: str, base: list[Evaluator] | None = None) -> list[Evaluator]:
    """The ladder as it behaves on one target.

    Only the two structure-based rungs are target-dependent, and deliberately in
    opposite directions: a target where docking does well is not necessarily one
    where co-folding does. A campaign that always reaches for the most expensive
    method is therefore wrong roughly half the time, and a campaign that always
    reaches for the cheapest is wrong the rest of it.
    """
    base = base or DEFAULT_LADDER
    # Deterministic per-target perturbation: stable across runs, unknown to the
    # planner. hashlib rather than hash(), because Python salts string hashing
    # per process and this must not move between runs.
    digest = hashlib.sha256(f"fidelity:{target}".encode()).digest()[:8]
    rng = np.random.default_rng(int.from_bytes(digest, "big") % (2**32))

    # The two structure-based rungs are drawn from *overlapping* ranges, and
    # independently of each other. This is the one calibration choice in the
    # whole simulator that decides whether the project's premise is even
    # testable: if co-folding could never beat docking, docking would dominate
    # by construction and "which method for this target" would not be a
    # question. The ranges are set to straddle, in the spirit of the co-folding
    # benchmark that reports concordance with experiment swinging from roughly
    # no correlation to strong depending on the system.
    #
    # Cost is left genuinely punitive: co-folding is eight docking runs. So a
    # higher fidelity does not automatically justify it, and the planner has a
    # real trade to make rather than a ranking to read off.
    fidelities = {
        "docking": rng.uniform(0.55, 0.92),
        "cofolding": rng.uniform(0.45, 0.97),
    }
    out = []
    for ev in base:
        value = fidelities.get(ev.name, ev.fidelity)
        out.append(Evaluator(ev.name, ev.cost, float(np.clip(value, 0.05, 1.0)),
                             ev.description, ev.confirms))
    return out


@dataclass
class CostLedger:
    """Budget accounting in cost units, with a per-evaluator breakdown.

    A count of experiments says nothing when the experiments differ in price by
    five orders of magnitude. This is what replaces it, and it is also what the
    screening literature reports improvements in -- "n-fold less compute".
    """

    total: float
    spent: float = 0.0
    by_evaluator: dict[str, float] = field(default_factory=dict)
    calls: dict[str, int] = field(default_factory=dict)

    @property
    def remaining(self) -> float:
        return max(self.total - self.spent, 0.0)

    @property
    def exhausted(self) -> bool:
        return self.remaining <= 1e-12

    def can_afford(self, evaluator: Evaluator, n: int = 1) -> bool:
        return evaluator.cost * n <= self.remaining + 1e-12

    def affordable_count(self, evaluator: Evaluator) -> int:
        if evaluator.cost <= 0:
            return 1_000_000
        return int(self.remaining // evaluator.cost)

    def spend(self, evaluator: Evaluator, n: int = 1) -> float:
        amount = evaluator.cost * n
        self.spent += amount
        self.by_evaluator[evaluator.name] = self.by_evaluator.get(evaluator.name, 0.0) + amount
        self.calls[evaluator.name] = self.calls.get(evaluator.name, 0) + n
        return amount

    def as_dict(self) -> dict:
        return {
            "total": self.total,
            "spent": round(self.spent, 4),
            "remaining": round(self.remaining, 4),
            "by_evaluator": {k: round(v, 4) for k, v in sorted(self.by_evaluator.items())},
            "calls": dict(sorted(self.calls.items())),
        }


@dataclass
class Observation:
    """One measurement: what was measured, by what, at what fidelity and price."""

    index: int
    value: float
    evaluator: str
    fidelity: float
    cost: float
    round_idx: int

    @property
    def weight(self) -> float:
        """How much a surrogate should trust this point.

        Squared correlation is the share of variance the measurement actually
        explains, which is the natural weight for a least-squares fit.
        """
        return float(self.fidelity**2)


class MultiFidelityOracle:
    """Budget-limited access to a pool objective through a ladder of methods.

    The agent never touches the objective directly; it asks this for a
    measurement with a named evaluator, and this spends the cost, adds the
    fidelity-appropriate error, and records what happened.
    """

    def __init__(self, objective, ladder: list[Evaluator], budget: float, seed: int = 0):
        self.objective = objective
        self.ladder = {ev.name: ev for ev in ladder}
        self.ledger = CostLedger(total=float(budget))
        self.observations: list[Observation] = []
        self._rng = np.random.default_rng(seed)
        values = np.asarray(objective.true_values, dtype=float)
        self._mu, self._sigma = float(values.mean()), float(values.std() or 1.0)

    # -- scale conversion --------------------------------------------------

    def _z(self, value: float) -> float:
        return (value - self._mu) / self._sigma

    def _unz(self, z: float) -> float:
        return z * self._sigma + self._mu

    # -- measuring ---------------------------------------------------------

    def measure(self, index: int, evaluator_name: str, round_idx: int = -1) -> float:
        ev = self.ladder[evaluator_name]
        self.ledger.spend(ev)
        z = ev.measure(self._z(float(self.objective.true_values[index])), self._rng)
        value = self._unz(z)
        self.observations.append(
            Observation(int(index), float(value), ev.name, ev.fidelity, ev.cost, round_idx)
        )
        return value

    def measure_batch(self, indices, evaluator_name: str, round_idx: int = -1) -> np.ndarray:
        return np.array([self.measure(int(i), evaluator_name, round_idx) for i in indices])

    # -- what the campaign knows -------------------------------------------

    def confirmed_indices(self) -> set[int]:
        """Molecules measured by a method trustworthy enough to call a hit.

        A compound ranked highly by a retrieval model is a lead, not a hit. This
        distinction is what stops a campaign from claiming credit for finding
        something it never actually checked.
        """
        return {
            o.index for o in self.observations
            if self.ladder[o.evaluator].confirms
        }

    def evaluated_indices(self) -> set[int]:
        return {o.index for o in self.observations}

    def best_estimate_per_index(self) -> dict[int, tuple[float, float]]:
        """Highest-fidelity measurement held for each molecule: {index: (value, fidelity)}."""
        best: dict[int, tuple[float, float]] = {}
        for o in self.observations:
            if o.index not in best or o.fidelity > best[o.index][1]:
                best[o.index] = (o.value, o.fidelity)
        return best

    def training_set(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(X, y, weights) for fitting a surrogate on everything measured so far."""
        best = self.best_estimate_per_index()
        idx = sorted(best)
        X = self.objective.pool.features[idx]
        y = np.array([best[i][0] for i in idx])
        w = np.array([max(best[i][1], 1e-3) ** 2 for i in idx])
        return X, y, w
