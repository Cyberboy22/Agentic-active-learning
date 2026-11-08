#!/usr/bin/env python3
"""Run one virtual-screening campaign.

Examples
--------
A cascade campaign against EGFR, with the medicinal-chemistry reviewer on:
    python run_screening.py --target egfr --planner cascade --reviewer rules

Commit to one method, for comparison:
    python run_screening.py --target egfr --planner always_docking

Run part of a long campaign now and the rest later (what the cluster needs):
    python run_screening.py --target egfr --out runs/egfr --stop-after-rounds 5
    python run_screening.py --target egfr --out runs/egfr --resume
"""
from __future__ import annotations

import argparse
import json

from src.evaluators import DEFAULT_LADDER, ladder_for_target
from src.human import make_reviewer
from src.objectives.screening import VirtualScreeningObjective
from src.planners import available_planners
from src.screening_campaign import run_screening_campaign


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", default="egfr",
                    help="egfr | cdk2 | cmet, optionally with a synthetic variant: egfr#2")
    ap.add_argument("--planner", default="cascade",
                    help=f"{sorted(available_planners())} or always_<evaluator>")
    ap.add_argument("--budget", type=float, default=260.0, help="budget in cost units")
    ap.add_argument("--rounds", type=int, default=18)
    ap.add_argument("--decoys", type=int, default=2500)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reviewer", default="none", choices=["none", "rules"])
    ap.add_argument("--diversity", type=float, default=None,
                    help="override the planner's diversity weight (0 = plain top-k)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--stop-after-rounds", type=int, default=None)
    args = ap.parse_args()

    objective = VirtualScreeningObjective(args.target, n_decoys=args.decoys)
    ladder = ladder_for_target(args.target, DEFAULT_LADDER)
    print(f"{objective.description} · library {len(objective.pool)} · "
          f"{int(objective.known_actives.sum())} known actives")
    print("evaluator ladder: " + ", ".join(
        f"{e.name} (cost {e.cost:g}, fidelity {e.fidelity:.2f})" for e in ladder))

    result = run_screening_campaign(
        objective=objective, planner=args.planner, budget=args.budget,
        n_rounds=args.rounds, top_k=args.top_k, seed=args.seed,
        reviewer=make_reviewer(args.reviewer), diversity=args.diversity,
        out_dir=args.out, resume=args.resume, stop_after_rounds=args.stop_after_rounds,
    )

    m = result.metrics
    print(f"\nrecall@{m['k']}  {m['recall_at_k']:.3f}   "
          f"(coverage {m['coverage_recall']:.3f})")
    print(f"EF 1%        {m['ef_1pct']:.1f}        BEDROC20 {m['bedroc_20']:.3f}")
    print(f"cost spent   {m['cost_spent']:.1f} of {args.budget:g}")
    print(f"spend        {json.dumps(m['ledger']['by_evaluator'])}")
    if result.overrides:
        print(f"vetoed       {len(result.overrides)} compounds by the reviewer")
    if args.out:
        print(f"written to   {args.out}/")


if __name__ == "__main__":
    main()
