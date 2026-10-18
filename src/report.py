"""Reporting: a convergence plot and a Markdown write-up with rationales.

The report is deliberately "evidence-based": for every round it records what
the agent decided, why, and what the chosen experiments actually measured.
That is the transparent, auditable trail a screening campaign needs.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import oracle  # noqa: E402
from .campaign import CampaignResult  # noqa: E402


def plot_convergence(res: CampaignResult, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=130)
    agent_x = range(1, len(res.history_best) + 1)
    ax.plot(agent_x, res.history_best, lw=2.2, label="agent (active learning)")
    base_x = range(1, len(res.baseline_best) + 1)
    ax.plot(base_x, res.baseline_best, lw=1.8, ls="--", label="random baseline")
    if res.reference_optimum == res.reference_optimum:  # not NaN
        ax.axhline(
            res.reference_optimum,
            color="grey",
            lw=1.0,
            ls=":",
            label=f"brute-force optimum ≈ {res.reference_optimum:.1f} MPa",
        )
    ax.set_xlabel("experiments run (budget spent)")
    ax.set_ylabel("best strength found so far (MPa)")
    ax.set_title("Agentic active learning vs random search")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_report(res: CampaignResult, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_convergence(res, out_dir / "convergence.png")

    best_mix = {n: round(float(v), 2) for n, v in zip(oracle.VAR_NAMES, res.best_x)}
    ref = res.reference_optimum
    pct = f"{100 * res.best_value / ref:.1f}%" if ref == ref and ref > 0 else "n/a"
    base_final = res.baseline_best[-1] if res.baseline_best else float("nan")

    lines = []
    lines.append("# Campaign report\n")
    lines.append(
        "An LLM/heuristic agent chose which experiments to run, round by round, "
        "to maximise concrete compressive strength under a fixed experiment budget.\n"
    )
    lines.append("## Result\n")
    lines.append(f"- **Best strength found:** {res.best_value:.2f} MPa")
    lines.append(f"- **Random baseline (same budget):** {base_final:.2f} MPa")
    lines.append(f"- **Brute-force reference optimum (noise-free):** {ref:.2f} MPa")
    lines.append(f"- **Fraction of reference reached:** {pct}")
    lines.append(f"- **Experiments spent:** {res.rounds[-1].budget_used if res.rounds else 0}")
    lines.append(
        "\n> The reference optimum is the *noise-free* ground-truth maximum from a "
        "200k-point brute-force scan. Measured values include experimental noise "
        "(σ = 1.5 MPa), so a single measurement can sit slightly above it. The point "
        "is the gap to the random baseline, found with the same number of experiments.\n"
    )
    lines.append("\n**Best mix design found:**\n")
    lines.append("| variable | value | unit |")
    lines.append("|---|---:|---|")
    for (name, _, _, unit), v in zip(oracle.SEARCH_SPACE, res.best_x):
        lines.append(f"| {name} | {v:.2f} | {unit} |")

    lines.append("\n## Per-round decisions (the explainable trail)\n")
    for rl in res.rounds:
        d = rl.decision
        lines.append(
            f"### Round {rl.round_idx} — best so far {rl.best_value:.2f} MPa "
            f"(budget used {rl.budget_used})"
        )
        lines.append(
            f"- **Decision [{d['source']}]:** acquisition=`{d['acquisition']}`, "
            f"surrogate=`{d['surrogate']}`, kappa={d['kappa']}, batch={d['batch_size']}"
        )
        lines.append(f"- **Rationale:** {d['rationale']}")
        lines.append("- **Selected experiments:**\n")
        lines.append("  | predicted (MPa) | uncertainty ± | acq score | measured (MPa) |")
        lines.append("  |---:|---:|---:|---:|")
        for s in rl.selected:
            lines.append(
                f"  | {s['predicted_mean']} | {s['predicted_std']} | "
                f"{s['acq_score']} | {s['measured']} |"
            )
        lines.append("")

    (out_dir / "report.md").write_text("\n".join(lines))

    # Machine-readable log for downstream analysis.
    payload = {
        "best_value": res.best_value,
        "best_mix": best_mix,
        "reference_optimum": ref,
        "history_best": res.history_best,
        "baseline_best": res.baseline_best,
        "rounds": [
            {
                "round_idx": rl.round_idx,
                "budget_used": rl.budget_used,
                "best_value": rl.best_value,
                "decision": rl.decision,
                "selected": rl.selected,
            }
            for rl in res.rounds
        ],
    }
    (out_dir / "campaign_log.json").write_text(json.dumps(payload, indent=2))
