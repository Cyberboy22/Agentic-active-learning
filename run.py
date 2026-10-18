#!/usr/bin/env python3
"""Run one active-learning campaign and write a full report.

Examples
--------
Fully offline, deterministic planner:
    python run.py --planner heuristic --objective concrete

Let Claude plan each round (needs ANTHROPIC_API_KEY):
    export ANTHROPIC_API_KEY=sk-...
    python run.py --planner llm

Compare against the classical Bayesian-optimisation baseline instead of random:
    python run.py --planner heuristic --baseline classical_bo

Resume an interrupted campaign (same --out and same settings):
    python run.py --planner heuristic --out runs/my_campaign --resume
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.benchmark import DEFAULT_OBJECTIVES  # noqa: F401  (keeps names discoverable)
from src.campaign import run_campaign
from src.objective import available_objectives
from src.objectives import *  # noqa: F401,F403  (registers built-in objectives)
from src.planners import available_planners
from src.report import write_report


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--objective", default="concrete",
                   help=f"what to optimise; one of {available_objectives()}")
    p.add_argument("--planner", default="auto",
                   help=f"decision rule; one of {available_planners()} or 'auto'")
    p.add_argument("--baseline", default="random",
                   help="planner to compare against on the plot ('none' to skip)")
    p.add_argument("--budget", type=int, default=50, help="total experiment budget")
    p.add_argument("--n-init", type=int, default=10, help="size of the initial design")
    p.add_argument("--n-rounds", type=int, default=12, help="active-learning rounds")
    p.add_argument("--pool-size", type=int, default=1500, help="candidates sampled per round")
    p.add_argument("--max-batch", type=int, default=6, help="max experiments per round")
    p.add_argument("--seed", type=int, default=0, help="random seed")
    p.add_argument("--out", type=Path, default=Path("examples/demo_output"),
                   help="output directory (also holds the checkpoint)")
    p.add_argument("--resume", action="store_true",
                   help="continue from the checkpoint in --out, if present")
    p.add_argument("--quiet", action="store_true", help="suppress per-round logging")
    args = p.parse_args()

    common = dict(
        objective=args.objective, budget=args.budget, n_init=args.n_init,
        n_rounds=args.n_rounds, pool_size=args.pool_size, max_batch=args.max_batch,
        seed=args.seed,
    )

    res = run_campaign(
        planner=args.planner, out_dir=args.out, resume=args.resume,
        verbose=not args.quiet, **common,
    )

    baseline = None
    if args.baseline and args.baseline.lower() != "none" and args.baseline != args.planner:
        baseline = run_campaign(planner=args.baseline, verbose=False, **common)

    write_report(res, args.out, baseline=baseline)

    unit = f" {res.value_unit}" if res.value_unit else ""
    print(f"\nBest value found: {res.best_value:.4g}{unit}", end="")
    if baseline is not None:
        print(f"  ({res.best_value - baseline.best_value:+.4g}{unit} vs "
              f"{baseline.planner_name} baseline)", end="")
    print(f"\nReference optimum: {res.reference_optimum:.4g}{unit}")
    print(f"Report:  {args.out / 'report.md'}")
    print(f"Plot:    {args.out / 'convergence.png'}")
    print(f"Manifest:{args.out / 'manifest.json'}  (config {res.manifest.config_hash})")


if __name__ == "__main__":
    main()
