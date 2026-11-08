"""Paired, multi-seed comparison of screening strategies -- with the spread.

The thing none of the four papers on the reading list does is run anything
twice. Every comparison in them rests on a single number: one enrichment
factor, one hit rate, one concordance index. That is also the mistake this
repository's own first benchmark made, where an advantage that looked decisive
over ten seeds turned out to be noise over twenty.

So this harness exists to make the spread impossible to ignore:

* every strategy sees the **same seed**, so runs are paired and the difference
  between them is the decision rule rather than the draw;
* differences are tested with a **Wilcoxon signed-rank test** on the pairs;
* effect sizes come with a **bootstrap confidence interval**, because a p-value
  alone says whether an effect exists and not whether it matters;
* the headline is **recall per unit cost**, the axis this field improves on.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from .evaluators import DEFAULT_LADDER, ladder_for_target
from .objectives.screening import VirtualScreeningObjective
from .screening_campaign import run_screening_campaign


@dataclass
class ScreeningRecord:
    target: str
    planner: str
    seed: int
    recall: float
    ef_1pct: float
    bedroc_20: float
    cost: float
    n_confirmed: int
    cost_to_recall_25: float
    cost_to_recall_50: float
    by_evaluator: dict


def run_grid(targets: list[str], planners: list[str], seeds: int = 10,
             budget: float = 250.0, n_rounds: int = 16, n_decoys: int = 2500,
             top_k: int = 50, diversity: float | None = None,
             reviewer_kind: str = "none", planner_factory=None,
             verbose: bool = True, log=print) -> list[ScreeningRecord]:
    """Every planner on every target across `seeds` paired seeds."""
    from .human import make_reviewer

    records: list[ScreeningRecord] = []
    for target in targets:
        objective = VirtualScreeningObjective(target, n_decoys=n_decoys)
        ladder = ladder_for_target(target, DEFAULT_LADDER)
        if verbose:
            fid = ", ".join(f"{e.name} {e.fidelity:.2f}" for e in ladder)
            log(f"\n=== {target}  library={len(objective.pool)}  "
                f"actives={int(objective.known_actives.sum())}  fidelities: {fid}")
        for planner in planners:
            for seed in range(seeds):
                obj = planner_factory(planner, seed) if planner_factory else None
                result = run_screening_campaign(
                    objective=objective,
                    planner=planner if obj is None else "heuristic",
                    planner_obj=obj,
                    budget=budget, n_rounds=n_rounds, top_k=top_k, seed=seed,
                    diversity=diversity, reviewer=make_reviewer(reviewer_kind),
                    verbose=False,
                )
                m = result.metrics
                records.append(ScreeningRecord(
                    target=target, planner=planner, seed=seed,
                    recall=m["recall_at_k"], ef_1pct=m["ef_1pct"],
                    bedroc_20=m["bedroc_20"], cost=m["cost_spent"],
                    n_confirmed=m["n_confirmed"],
                    cost_to_recall_25=m["cost_to_recall_25"],
                    cost_to_recall_50=m["cost_to_recall_50"],
                    by_evaluator=m["ledger"]["by_evaluator"],
                ))
            if verbose:
                vals = [r.recall for r in records if r.target == target and r.planner == planner]
                log(f"  {planner:22s} recall {np.mean(vals):.3f} +- {np.std(vals):.3f}")
    return records


def bootstrap_ci(deltas: np.ndarray, n_boot: int = 10_000, seed: int = 0,
                 alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap interval for a mean paired difference."""
    deltas = np.asarray(deltas, dtype=float)
    if len(deltas) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = rng.choice(deltas, size=(n_boot, len(deltas)), replace=True).mean(axis=1)
    return (float(np.percentile(means, 100 * alpha / 2)),
            float(np.percentile(means, 100 * (1 - alpha / 2))))


def paired_comparison(records: list[ScreeningRecord], a: str, b: str,
                      target: str, metric: str = "recall") -> dict:
    """Compare planner `a` against planner `b` on one target, seed by seed."""
    by_seed = {}
    for r in records:
        if r.target == target and r.planner in (a, b):
            by_seed.setdefault(r.seed, {})[r.planner] = getattr(r, metric)
    pairs = [(v[a], v[b]) for v in by_seed.values() if a in v and b in v]
    if len(pairs) < 2:
        return {"n": len(pairs), "delta": float("nan"), "p": float("nan")}
    va, vb = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
    delta = va - vb
    try:
        p = float(wilcoxon(va, vb).pvalue) if np.any(delta != 0) else 1.0
    except ValueError:
        p = 1.0
    lo, hi = bootstrap_ci(delta)
    return {"n": len(pairs), "mean_a": float(va.mean()), "mean_b": float(vb.mean()),
            "delta": float(delta.mean()), "ci_low": lo, "ci_high": hi,
            "wins": int((delta > 0).sum()), "p": p}


def best_fixed_method(records: list[ScreeningRecord], target: str,
                      metric: str = "recall") -> tuple[str, float]:
    """The strongest always-use-one-method policy on this target."""
    scores: dict[str, list[float]] = {}
    for r in records:
        if r.target == target and r.planner.startswith("always_"):
            scores.setdefault(r.planner, []).append(getattr(r, metric))
    if not scores:
        return ("", float("nan"))
    means = {k: float(np.mean(v)) for k, v in scores.items()}
    best = max(means, key=means.get)
    return best, means[best]


def summarise(records: list[ScreeningRecord], adaptive: list[str] | None = None,
              metric: str = "recall") -> str:
    """Markdown report: per-target table, then the method-selection question."""
    targets = sorted({r.target for r in records})
    planners = sorted({r.planner for r in records})
    adaptive = adaptive or [p for p in planners if not p.startswith("always_")]
    seeds = len({r.seed for r in records})

    lines = ["# Screening strategy benchmark", ""]
    lines.append(f"{len(targets)} targets x {len(planners)} strategies x {seeds} paired seeds. "
                 f"Metric: **{metric}** (recall of the true top-k at the cost budget).")
    lines.append("")
    lines.append("## Per target")
    lines.append("")
    header = "| strategy | " + " | ".join(targets) + " | mean |"
    lines.append(header)
    lines.append("|---" * (len(targets) + 2) + "|")
    for planner in planners:
        cells, allv = [], []
        for t in targets:
            vals = [getattr(r, metric) for r in records if r.target == t and r.planner == planner]
            if vals:
                cells.append(f"{np.mean(vals):.3f} ± {np.std(vals):.3f}")
                allv.extend(vals)
            else:
                cells.append("—")
        mean = f"**{np.mean(allv):.3f}**" if allv else "—"
        lines.append(f"| `{planner}` | " + " | ".join(cells) + f" | {mean} |")

    lines += ["", "## Does choosing the method per target beat committing to one?", "",
              "The project's premise is that no single method generalises. If that is true, "
              "an adaptive strategy should beat the *best fixed method on each target* — and "
              "crucially, the best fixed method should not be the same one every time. "
              "If one method wins everywhere, the orchestration layer is unnecessary and "
              "this table is the honest place to find that out.", ""]
    lines.append("| target | best fixed method | its " + metric + " | " +
                 " | ".join(f"{a} Δ (p)" for a in adaptive) + " |")
    lines.append("|---" * (3 + len(adaptive)) + "|")
    winners = []
    for t in targets:
        best, score = best_fixed_method(records, t, metric)
        winners.append(best)
        cells = []
        for a in adaptive:
            c = paired_comparison(records, a, best, t, metric)
            delta, p = c.get("delta", float("nan")), c.get("p", float("nan"))
            star = "**" if p < 0.05 else ""
            cells.append(f"{star}{delta:+.3f}{star} (p={p:.3f})" if np.isfinite(delta) else "—")
        lines.append(f"| {t} | `{best}` | {score:.3f} | " + " | ".join(cells) + " |")

    distinct = len(set(w for w in winners if w))
    lines += ["", f"**Best fixed method differs across targets: "
                  f"{'yes' if distinct > 1 else 'no'}** "
                  f"({distinct} distinct winner(s) across {len(targets)} targets). "
              + ("This is the premise holding up: the right method depends on the target, "
                 "so something has to choose."
                 if distinct > 1 else
                 "On these targets one fixed method was best everywhere. That is evidence "
                 "*against* needing an orchestration layer here, and it should be reported "
                 "as such rather than explained away."), ""]
    lines += ["Δ is the paired mean difference against the best fixed method, with a "
              "Wilcoxon signed-rank p-value. Bold marks p < 0.05. With "
              f"{seeds} seeds these intervals are wide; treat anything above p = 0.05 "
              "as 'not shown', not as 'slightly worse'.", ""]
    return "\n".join(lines)


def save(records: list[ScreeningRecord], out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "records.json").write_text(json.dumps([asdict(r) for r in records], indent=2))
    return out
