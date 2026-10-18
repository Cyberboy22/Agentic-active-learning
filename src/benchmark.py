"""Benchmark harness: does agentic adaptivity actually beat the standard method?

That question is the reason this module exists. It is easy to build an LLM-driven
planner and assume the intelligence helps; it is harder, and much more useful, to
measure whether it beats textbook Bayesian optimisation and plain random search
on the same budget. This harness runs every planner against every objective
across many seeds and reports the comparison with statistics attached.

Design choices that make the comparison fair:

* **Paired runs.** For a given seed, every planner starts from the *same* initial
  design and the same oracle noise draw, because both generators are seeded
  identically. Differences therefore come from the decision rule, not from luck
  in the starting sample -- which also licenses a paired significance test.
* **Sign-safe normalisation.** Objectives have optima that are positive, zero or
  negative, so "percent of optimum" is meaningless. The harness reports the
  standard BO **gap closed**, G = (best − init_best) / (reference − init_best),
  which is 0 when a planner learns nothing beyond its initial design and 1 when
  it reaches the optimum.
* **Sample efficiency**, not just final value: how many experiments were needed
  to close 90% of the gap, counting runs that never got there.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .campaign import run_campaign
from .objective import get_objective

DEFAULT_OBJECTIVES = ["concrete", "hartmann6", "ackley"]
DEFAULT_PLANNERS = ["random", "classical_bo", "heuristic"]
GAP_TARGET = 0.90


@dataclass
class RunRecord:
    objective: str
    planner: str
    seed: int
    init_best: float
    final_best: float
    reference: float
    regret: float
    gap: float
    budget_used: int
    experiments_to_target: float  # nan when the target was never reached

    @staticmethod
    def from_result(res, n_init: int) -> "RunRecord":
        hist = res.history_best
        init_best = float(hist[min(n_init, len(hist)) - 1])
        ref = float(res.reference_optimum)
        span = ref - init_best
        gap = 1.0 if span <= 1e-12 else float(np.clip((res.best_value - init_best) / span, 0.0, 1.0))

        to_target = float("nan")
        if span > 1e-12:
            for i, v in enumerate(hist, start=1):
                if (v - init_best) / span >= GAP_TARGET:
                    to_target = float(i)
                    break
        return RunRecord(
            objective=res.objective_name, planner=res.planner_name, seed=-1,
            init_best=init_best, final_best=float(res.best_value), reference=ref,
            regret=float(ref - res.best_value), gap=gap,
            budget_used=int(res.budget_used), experiments_to_target=to_target,
        )


def run_benchmark(
    *,
    objectives: list[str] | None = None,
    planners: list[str] | None = None,
    seeds: int = 10,
    budget: int = 50,
    n_init: int = 10,
    n_rounds: int = 12,
    pool_size: int = 1500,
    max_batch: int = 6,
    log=print,
) -> list[RunRecord]:
    objectives = objectives or DEFAULT_OBJECTIVES
    planners = planners or DEFAULT_PLANNERS
    records: list[RunRecord] = []
    total = len(objectives) * len(planners) * seeds
    done = 0

    for obj_name in objectives:
        # Warm the reference-optimum cache once per objective.
        get_objective(obj_name)
        for planner in planners:
            for seed in range(seeds):
                res = run_campaign(
                    objective=obj_name, planner=planner, budget=budget, n_init=n_init,
                    n_rounds=n_rounds, pool_size=pool_size, max_batch=max_batch,
                    seed=seed, verbose=False,
                )
                rec = RunRecord.from_result(res, n_init)
                rec.seed = seed
                records.append(rec)
                done += 1
            if log:
                log(f"  [{done:>4}/{total}] {obj_name:<10} {planner:<13} done")
    return records


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------


def _paired(records: list[RunRecord], objective: str, a: str, b: str, key: str = "gap"):
    """Return the paired (a, b) metric vectors, matched by seed."""
    ra = {r.seed: getattr(r, key) for r in records if r.objective == objective and r.planner == a}
    rb = {r.seed: getattr(r, key) for r in records if r.objective == objective and r.planner == b}
    seeds = sorted(set(ra) & set(rb))
    return np.array([ra[s] for s in seeds]), np.array([rb[s] for s in seeds])


def wilcoxon_p(a: np.ndarray, b: np.ndarray) -> float:
    """Paired Wilcoxon signed-rank p-value; nan when the test is not applicable."""
    try:
        from scipy.stats import wilcoxon

        if len(a) < 5 or np.allclose(a, b):
            return float("nan")
        return float(wilcoxon(a, b).pvalue)
    except Exception:
        return float("nan")


def summarise(records: list[RunRecord], reference_planner: str = "classical_bo") -> str:
    """Markdown summary: per-objective table plus paired tests against a baseline."""
    objectives = sorted({r.objective for r in records})
    planners = sorted({r.planner for r in records}, key=lambda p: (p != "random", p))
    n_seeds = len({r.seed for r in records})

    L: list[str] = []
    L.append("# Planner benchmark\n")
    L.append(
        f"Each planner was run on every objective across **{n_seeds} seeds** at an "
        f"equal experiment budget. For a given seed all planners share the same "
        f"initial design, so the runs are paired and the differences reflect the "
        f"decision rule alone.\n"
    )
    L.append(
        "**Gap closed** G = (best − init_best) / (reference − init_best): 0 means a "
        "planner never improved on its starting sample, 1 means it reached the "
        "optimum. **Exp. to 90%** is how many experiments were needed to close 90% "
        f"of that gap, with the number of runs that got there at all in brackets.\n"
    )

    for obj in objectives:
        L.append(f"## {obj}\n")
        L.append("| planner | gap closed | final best | regret | exp. to 90% |")
        L.append("|---|---:|---:|---:|---:|")
        for p in planners:
            rs = [r for r in records if r.objective == obj and r.planner == p]
            if not rs:
                continue
            gaps = np.array([r.gap for r in rs])
            finals = np.array([r.final_best for r in rs])
            regrets = np.array([r.regret for r in rs])
            reached = [r.experiments_to_target for r in rs if r.experiments_to_target == r.experiments_to_target]
            eff = f"{np.mean(reached):.0f} ({len(reached)}/{len(rs)})" if reached else f"– (0/{len(rs)})"
            L.append(
                f"| {p} | {gaps.mean():.3f} ± {gaps.std():.3f} "
                f"| {finals.mean():.4g} ± {finals.std():.3g} "
                f"| {regrets.mean():.4g} | {eff} |"
            )
        L.append("")

        # Paired comparisons against the reference planner.
        others = [p for p in planners if p != reference_planner]
        if reference_planner in planners and others:
            L.append(f"**Paired comparison vs `{reference_planner}` (gap closed):**\n")
            L.append("| planner | mean Δgap | wins | p (Wilcoxon) |")
            L.append("|---|---:|---:|---:|")
            for p in others:
                a, b = _paired(records, obj, p, reference_planner)
                if not len(a):
                    continue
                wins = int(np.sum(a > b))
                L.append(f"| {p} | {np.mean(a - b):+.3f} | {wins}/{len(a)} "
                         f"| {wilcoxon_p(a, b):.3f} |")
            L.append("")

    # Aggregate across objectives.
    L.append("## Aggregate across objectives\n")
    L.append("| planner | mean gap closed | mean regret (normalised) |")
    L.append("|---|---:|---:|")
    for p in planners:
        rs = [r for r in records if r.planner == p]
        if not rs:
            continue
        gaps = np.array([r.gap for r in rs])
        L.append(f"| {p} | {gaps.mean():.3f} | {1 - gaps.mean():.3f} |")
    L.append(
        "\n> A caveat worth stating: these are low-dimensional, cheap-to-evaluate "
        "benchmarks with a modest budget. They show whether a decision rule is "
        "sound, not how it behaves on a real campaign with expensive, noisy "
        "evaluations and far larger candidate spaces.\n"
    )
    return "\n".join(L)


def write_csv(records: list[RunRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(asdict(records[0])))
        writer.writeheader()
        for r in records:
            writer.writerow(asdict(r))


def plot_benchmark(records: list[RunRecord], path: Path) -> None:
    """Grouped bars: mean gap closed per planner, per objective, with error bars."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    objectives = sorted({r.objective for r in records})
    planners = sorted({r.planner for r in records}, key=lambda p: (p != "random", p))

    fig, ax = plt.subplots(figsize=(1.9 * len(objectives) + 4, 4.4), dpi=130)
    width = 0.8 / max(len(planners), 1)
    xs = np.arange(len(objectives))

    for i, p in enumerate(planners):
        means, errs = [], []
        for obj in objectives:
            gaps = np.array([r.gap for r in records if r.objective == obj and r.planner == p])
            means.append(gaps.mean() if len(gaps) else 0.0)
            errs.append(gaps.std() if len(gaps) else 0.0)
        ax.bar(xs + i * width - 0.4 + width / 2, means, width, yerr=errs,
               capsize=3, label=p)

    ax.set_xticks(xs)
    ax.set_xticklabels(objectives)
    ax.set_ylabel("gap closed  (1.0 = reached the optimum)")
    ax.set_title("Planner comparison across objectives (mean ± std over seeds)")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
