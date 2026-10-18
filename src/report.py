"""Reporting: convergence plot, Markdown write-up and machine-readable log.

The report is deliberately an *audit trail*, not a summary. It records the
configuration and environment the result depended on, which tools ran and for
how long, and -- for every round -- what the planner decided, why, and what the
chosen experiments actually measured.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .campaign import CampaignResult  # noqa: E402


def plot_convergence(
    res: CampaignResult, path: Path, baseline: CampaignResult | None = None
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=130)
    ax.plot(range(1, len(res.history_best) + 1), res.history_best, lw=2.2,
            label=f"{res.planner_name} (active learning)")
    if baseline is not None:
        ax.plot(range(1, len(baseline.history_best) + 1), baseline.history_best,
                lw=1.8, ls="--", label=f"{baseline.planner_name} baseline")
    ref = res.reference_optimum
    if ref == ref:  # not NaN
        ax.axhline(ref, color="grey", lw=1.0, ls=":",
                   label=f"reference optimum ≈ {ref:.3g}")
    unit = f" ({res.value_unit})" if res.value_unit else ""
    ax.set_xlabel("experiments run (budget spent)")
    ax.set_ylabel(f"best value found so far{unit}")
    ax.set_title(f"{res.objective_name}: agentic active learning vs baseline")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_report(
    res: CampaignResult, out_dir: Path, baseline: CampaignResult | None = None
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_convergence(res, out_dir / "convergence.png", baseline)

    unit = f" {res.value_unit}" if res.value_unit else ""
    ref = res.reference_optimum
    frac = res.fraction_of_reference()
    pct = f"{100 * frac:.1f}%" if frac == frac else "n/a"
    man = res.manifest

    L: list[str] = []
    L.append(f"# Campaign report — {res.objective_name}\n")
    L.append(
        f"Planner **{res.planner_name}** chose which experiments to run, round by "
        f"round, to maximise the objective under a fixed budget of "
        f"{res.budget_used} experiments.\n"
    )
    if res.resumed_from_round is not None:
        L.append(f"> This campaign was **resumed** from a checkpoint at round "
                 f"{res.resumed_from_round}.\n")

    L.append("## Result\n")
    L.append(f"- **Best value found:** {res.best_value:.4g}{unit}")
    if baseline is not None:
        delta = res.best_value - baseline.best_value
        L.append(f"- **{baseline.planner_name} baseline (same budget):** "
                 f"{baseline.best_value:.4g}{unit}  (**{delta:+.4g}{unit}**)")
    L.append(f"- **Reference optimum (noise-free):** {ref:.4g}{unit}")
    L.append(f"- **Fraction of reference reached:** {pct}")
    L.append(f"- **Experiments spent:** {res.budget_used}")
    L.append(
        "\n> The reference optimum is the noise-free ground-truth maximum. "
        "Measurements include experimental noise, so a single measurement can sit "
        "slightly above it. The meaningful comparison is against the baseline at "
        "the same budget.\n"
    )

    L.append("**Best design found:**\n")
    L.append("| variable | value | unit |")
    L.append("|---|---:|---|")
    for name, value, u in zip(res.var_names, res.best_x, res.units):
        L.append(f"| {name} | {value:.4g} | {u} |")

    # -- provenance --------------------------------------------------------
    if man is not None:
        L.append("\n## Provenance\n")
        L.append(f"- **Config hash:** `{man.config_hash}`")
        L.append(f"- **Started / finished (UTC):** {man.started_at} → {man.finished_at}")
        if man.git_commit:
            L.append(f"- **Git commit:** `{man.git_commit}`")
        env = man.environment
        L.append(f"- **Python:** {env.get('python')} on {env.get('platform')}")
        pkgs = ", ".join(f"{k} {v}" for k, v in (env.get("packages") or {}).items())
        L.append(f"- **Packages:** {pkgs}")
        L.append("\n<details><summary>Full configuration</summary>\n")
        L.append("```json")
        L.append(json.dumps(man.config, indent=2))
        L.append("```\n</details>\n")

    # -- tool usage --------------------------------------------------------
    if res.tool_summary:
        L.append("## Tool usage\n")
        L.append("| tool | calls | total ms | failures |")
        L.append("|---|---:|---:|---:|")
        for name, st in sorted(res.tool_summary.items()):
            L.append(f"| `{name}` | {int(st['calls'])} | {st['total_ms']:.1f} "
                     f"| {int(st['failures'])} |")
        L.append("")

    # -- the decision trail ------------------------------------------------
    L.append("## Per-round decisions (the explainable trail)\n")
    for rl in res.rounds:
        d = rl.decision
        L.append(f"### Round {rl.round_idx} — best so far {rl.best_value:.4g}{unit} "
                 f"(budget used {rl.budget_used})")
        L.append(f"- **Decision [{d['source']}]:** acquisition=`{d['acquisition']}`, "
                 f"surrogate=`{d['surrogate']}`, kappa={d['kappa']}, batch={d['batch_size']}")
        L.append(f"- **Rationale:** {d['rationale']}")
        L.append("- **Selected experiments:**\n")
        L.append("  | predicted | uncertainty ± | acq score | measured |")
        L.append("  |---:|---:|---:|---:|")
        for s in rl.selected:
            L.append(f"  | {s['predicted_mean']:.4g} | {s['predicted_std']:.4g} "
                     f"| {s['acq_score']:.4g} | {s['measured']:.4g} |")
        L.append("")

    (out_dir / "report.md").write_text("\n".join(L))

    payload = {
        "objective": res.objective_name,
        "planner": res.planner_name,
        "best_value": res.best_value,
        "best_design": {n: float(v) for n, v in zip(res.var_names, res.best_x)},
        "reference_optimum": ref,
        "fraction_of_reference": frac,
        "budget_used": res.budget_used,
        "history_best": res.history_best,
        "baseline": None if baseline is None else {
            "planner": baseline.planner_name,
            "best_value": baseline.best_value,
            "history_best": baseline.history_best,
        },
        "manifest": None if man is None else asdict(man),
        "tool_summary": res.tool_summary,
        "rounds": [vars(r) for r in res.rounds],
    }
    (out_dir / "campaign_log.json").write_text(json.dumps(payload, indent=2, default=str))
    (out_dir / "tool_calls.json").write_text(json.dumps(res.tool_calls, indent=2, default=str))
