#!/usr/bin/env python3
"""When, if ever, can the expensive method pay for itself?

The method-selection experiment returned a clean negative: docking won on every
landscape, including the ones where co-folding is substantially more accurate.
Before concluding that accuracy does not matter, it is worth asking whether the
*measurement* can see it at all.

The suspicion is structural. Recall of the true top-k needs coverage: to put 50
specific compounds in front of a chemist you have to have measured something
like 50 of them. Co-folding costs eight docking runs, so at a fixed budget it
measures an eighth as many compounds. A method that is more accurate per
measurement but gets eight times fewer measurements cannot win a metric that
rewards breadth — no matter how accurate it is.

If that is the explanation, then shrinking k (ask for a shortlist of 10 instead
of 50) or raising the budget should let accuracy matter, and the winner should
change on landscapes where co-folding is the better model. If the winner never
changes, the cost ratio simply dominates and that is the finding.

Run:
    python experiments/cost_sensitivity.py --seeds 4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from src.evaluators import DEFAULT_LADDER, ladder_for_target  # noqa: E402
from src.objectives.screening import VirtualScreeningObjective, target_variants  # noqa: E402
from src.screening_campaign import run_screening_campaign  # noqa: E402


def fidelity_gap(target: str) -> float:
    lad = {e.name: e.fidelity for e in ladder_for_target(target, DEFAULT_LADDER)}
    return lad["cofolding"] - lad["docking"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--decoys", type=int, default=1200)
    ap.add_argument("--landscapes", type=int, default=2,
                    help="how many co-folding-favouring landscapes to test")
    ap.add_argument("--ks", nargs="+", type=int, default=[10, 50])
    ap.add_argument("--budgets", nargs="+", type=float, default=[240.0, 960.0])
    ap.add_argument("--out", default="examples/cost_sensitivity")
    args = ap.parse_args()

    # Pick the landscapes where co-folding is most clearly the better model.
    candidates = target_variants(["egfr", "cdk2", "cmet"], 4)
    ranked = sorted(candidates, key=fidelity_gap, reverse=True)[: args.landscapes]

    rows = []
    for target in ranked:
        lad = {e.name: e.fidelity for e in ladder_for_target(target, DEFAULT_LADDER)}
        print(f"\n=== {target}: docking fidelity {lad['docking']:.2f}, "
              f"co-folding {lad['cofolding']:.2f} at 8x the price")
        objective = VirtualScreeningObjective(target, n_decoys=args.decoys)
        for budget in args.budgets:
            for k in args.ks:
                line = {"target": target, "budget": budget, "k": k}
                for planner in ("always_docking", "always_cofolding"):
                    vals = []
                    for seed in range(args.seeds):
                        r = run_screening_campaign(
                            objective=objective, planner=planner, budget=budget,
                            n_rounds=18, top_k=k, seed=seed, verbose=False)
                        vals.append(r.metrics["recall_at_k"])
                        line[planner + "_n"] = r.metrics["n_confirmed"]
                    line[planner] = float(np.mean(vals))
                winner = ("cofolding" if line["always_cofolding"] > line["always_docking"]
                          else "docking")
                line["winner"] = winner
                rows.append(line)
                print(f"  budget {budget:6.0f}  k={k:3d} | "
                      f"docking {line['always_docking']:.3f} "
                      f"({line['always_docking_n']} confirmed) | "
                      f"co-folding {line['always_cofolding']:.3f} "
                      f"({line['always_cofolding_n']} confirmed) | winner: {winner}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    flips = [r for r in rows if r["winner"] == "cofolding"]
    lines = ["# Can the expensive method ever pay for itself?", "",
             "Landscapes chosen because co-folding is the *more accurate* model on them. "
             "It still costs eight docking runs per compound.", "",
             "| target | budget | k | docking | co-folding | winner |", "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['target']} | {r['budget']:.0f} | {r['k']} | "
                     f"{r['always_docking']:.3f} | {r['always_cofolding']:.3f} | {r['winner']} |")
    lines += ["", f"Co-folding won {len(flips)} of {len(rows)} settings.", ""]
    lines.append(
        "Coverage, not accuracy, is what a recall-of-top-k metric rewards: putting k compounds "
        "in front of a chemist requires having measured about k of them, and an eight-times "
        "pricier method measures an eighth as many. Where co-folding wins, it is because k is "
        "small enough or the budget large enough that both methods can cover the shortlist and "
        "accuracy decides. Where it loses, no amount of accuracy compensates."
        if flips else
        "Co-folding never won, at any budget or shortlist size tested. On this cost model the "
        "eight-fold price difference dominates any accuracy advantage it has, and 'use the more "
        "accurate method' is simply the wrong instinct here. That is a finding about the cost "
        "model as much as about the methods, and it is the honest reading of the "
        "method-selection result rather than an excuse for it.")
    (out / "report.md").write_text("\n".join(lines))
    print(f"\nco-folding won {len(flips)}/{len(rows)} settings -> {out}/report.md")


if __name__ == "__main__":
    main()
