"""Screening metrics: the numbers this field actually reports.

"Gap closed toward a reference optimum" is the right score for a continuous
optimiser and means nothing to a computational chemist. A retrospective
screening benchmark is judged on how much of the known good stuff you recovered
for how little of the library you paid to look at:

* **Shortlist recall** -- the headline. Hand a chemist the k compounds the
  campaign rates highest on confirmed evidence; how many of the genuinely best k
  are in that list? This is the number that matters, and it is deliberately
  harsher than counting what the campaign merely *measured*: an inaccurate
  evaluator fills the shortlist with the wrong compounds and is punished for it,
  which is exactly the pressure that makes choosing a method a real decision.
* **Coverage recall** -- of the true top-k, how many did the campaign evaluate
  at confirming fidelity at all? Reported as a diagnostic, because a campaign
  can have touched the right compounds and still have failed to recognise them.
* **Enrichment factor** -- how many times more hits are in the top slice of your
  ranking than chance would put there? EF1% is the field's default headline.
* **Cost to recall** -- how much compute to reach a given recall. This is the
  axis AdaptiveFlow reports its "n-fold less compute" on, and the one an
  orchestration layer has to move to justify itself.

All three need ground truth, so all three are *retrospective*: they score a
campaign after the fact against answers the campaign never saw.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def true_top_k(true_values: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k genuinely best compounds in the library."""
    return np.argsort(np.asarray(true_values))[::-1][:k]


def recall_at(found: set[int] | np.ndarray, target_set: np.ndarray) -> float:
    """Fraction of the true top-k the campaign actually has in hand."""
    target = set(int(i) for i in target_set)
    if not target:
        return float("nan")
    got = set(int(i) for i in found)
    return len(got & target) / len(target)


def enrichment_factor(ranking: np.ndarray, actives: np.ndarray, fraction: float = 0.01) -> float:
    """EF at the top `fraction` of a ranked library.

    EF = (hit rate in the top slice) / (hit rate in the library). EF = 1 is
    chance; EF = 20 at 1% means the ranking put twenty times as many actives in
    the first percentile as random ordering would.
    """
    ranking = np.asarray(ranking, dtype=int)
    active = set(int(i) for i in actives)
    n = len(ranking)
    if n == 0 or not active:
        return float("nan")
    cut = max(int(round(n * fraction)), 1)
    in_slice = sum(1 for i in ranking[:cut] if i in active)
    baseline = len(active) / n
    if baseline <= 0:
        return float("nan")
    return (in_slice / cut) / baseline


def bedroc(ranking: np.ndarray, actives: np.ndarray, alpha: float = 20.0) -> float:
    """BEDROC: an early-recognition score that weights the top of the ranking.

    Included because enrichment at a single cut point is brittle -- it moves a
    lot when one compound crosses the boundary -- and reviewers ask for it.
    """
    ranking = np.asarray(ranking, dtype=int)
    active = set(int(i) for i in actives)
    n, m = len(ranking), len(active)
    if n == 0 or m == 0 or m == n:
        return float("nan")
    ratio = m / n
    ranks = np.array([i + 1 for i, idx in enumerate(ranking) if idx in active], dtype=float)
    s = np.sum(np.exp(-alpha * ranks / n))
    ra = ratio
    random_sum = ra * (1 - np.exp(-alpha)) / (np.exp(alpha / n) - 1)
    factor = (ra * np.sinh(alpha / 2) / (np.cosh(alpha / 2) - np.cosh(alpha / 2 - alpha * ra)))
    const = 1 / (1 - np.exp(alpha * (1 - ra)))
    return float(s / random_sum * factor + const)


@dataclass
class RecallCurve:
    """Recall of the true top-k as a function of cost spent."""

    cost: list[float]
    recall: list[float]

    def cost_to_recall(self, target: float) -> float:
        """Cost at which recall first reached `target`; inf if it never did."""
        for c, r in zip(self.cost, self.recall):
            if r >= target - 1e-12:
                return c
        return float("inf")

    def final(self) -> float:
        return self.recall[-1] if self.recall else float("nan")


def recall_curve(observations, true_values: np.ndarray, k: int,
                 confirming: set[str] | None = None) -> RecallCurve:
    """Walk the observation log and track recall of the true top-k against cost.

    Only measurements from a confirming evaluator count toward recall: a cheap
    ranking says a compound is interesting, it does not establish that it is a
    hit. Costs from every rung still count against the budget, which is what
    makes the cheap rungs a real decision rather than free information.
    """
    target = set(int(i) for i in true_top_k(true_values, k))
    found: set[int] = set()
    cost, recall = [0.0], [0.0]
    running = 0.0
    for o in observations:
        running += o.cost
        if confirming is None or o.evaluator in confirming:
            if o.index in target:
                found.add(o.index)
        cost.append(running)
        recall.append(len(found) / len(target) if target else float("nan"))
    return RecallCurve(cost, recall)


def shortlist(oracle, k: int, confirming: set[str] | None = None) -> list[int]:
    """The k compounds a campaign would actually hand over.

    Only confirmed measurements are eligible. A compound a retrieval model liked
    is a lead, not something you put in front of a chemist as a result, and a
    campaign that never confirmed anything has nothing to hand over -- which is
    the correct score for it, not an inconvenience to work around.
    """
    best: dict[int, tuple[float, float]] = {}
    for o in oracle.observations:
        if confirming is not None and o.evaluator not in confirming:
            continue
        if o.index not in best or o.fidelity > best[o.index][1]:
            best[o.index] = (o.value, o.fidelity)
    return sorted(best, key=lambda i: -best[i][0])[:k]


def json_safe(value):
    """Replace non-finite floats with None, recursively.

    `cost_to_recall` returns infinity when a recall target is never reached,
    which is the right answer in Python and a broken document in JSON:
    `json.dumps` happily writes a bare `Infinity`, and every browser's
    `JSON.parse` then rejects the *entire* file. Anything written out for a
    browser goes through here first.
    """
    import math

    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def summarise_campaign(oracle, objective, k: int = 50) -> dict:
    """The headline numbers for one screening campaign."""
    true_values = np.asarray(objective.true_values, dtype=float)
    confirming = {name for name, ev in oracle.ladder.items() if ev.confirms}
    curve = recall_curve(oracle.observations, true_values, k, confirming)
    target = true_top_k(true_values, k)
    confirmed = oracle.confirmed_indices()

    # Final ranking: what the campaign would hand a chemist, best estimate first.
    best = oracle.best_estimate_per_index()
    ranked_seen = sorted(best, key=lambda i: -best[i][0])
    unseen = [i for i in range(len(true_values)) if i not in best]
    ranking = np.array(ranked_seen + unseen, dtype=int)

    picked = shortlist(oracle, k, confirming)

    return {
        "k": k,
        # The headline: of the k compounds the campaign would hand over, how
        # many are genuinely in the top k?
        "recall_at_k": recall_at(picked, target),
        "shortlist_size": len(picked),
        # Diagnostic: did it even look at the right compounds?
        "coverage_recall": recall_at(confirmed, target),
        "confirmed_in_top_k": len(set(int(i) for i in confirmed) & set(int(i) for i in target)),
        "n_confirmed": len(confirmed),
        "n_evaluated": len(oracle.evaluated_indices()),
        "library_size": len(true_values),
        "fraction_confirmed": len(confirmed) / max(len(true_values), 1),
        "ef_1pct": enrichment_factor(ranking, target, 0.01),
        "ef_5pct": enrichment_factor(ranking, target, 0.05),
        "bedroc_20": bedroc(ranking, target, 20.0),
        "cost_spent": round(oracle.ledger.spent, 4),
        "cost_to_recall_25": curve.cost_to_recall(0.25),
        "cost_to_recall_50": curve.cost_to_recall(0.50),
        "ledger": oracle.ledger.as_dict(),
    }
