#!/usr/bin/env python3
"""Compare planners across objectives and seeds, with paired statistics.

Examples
--------
Default comparison (random vs classical BO vs the adaptive heuristic):
    python benchmark.py --seeds 10

A quick smoke run:
    python benchmark.py --seeds 3 --budget 30 --objectives concrete

Include the Claude-driven planner (needs ANTHROPIC_API_KEY; makes one API call
per round, so keep the seed count small):
    python benchmark.py --seeds 5 --planners random classical_bo heuristic llm
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.benchmark import (
    DEFAULT_OBJECTIVES,
    DEFAULT_PLANNERS,
    plot_benchmark,
    run_benchmark,
    summarise,
    write_csv,
)
from src.objective import available_objectives
from src.objectives import *  # noqa: F401,F403
from src.planners import available_planners


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--objectives", nargs="+", default=DEFAULT_OBJECTIVES,
                   help=f"objectives to test; available: {available_objectives()}")
    p.add_argument("--planners", nargs="+", default=DEFAULT_PLANNERS,
                   help=f"planners to compare; available: {available_planners()}")
    p.add_argument("--seeds", type=int, default=10, help="independent runs per cell")
    p.add_argument("--budget", type=int, default=50, help="experiment budget per run")
    p.add_argument("--n-init", type=int, default=10, help="size of the initial design")
    p.add_argument("--n-rounds", type=int, default=12, help="active-learning rounds")
    p.add_argument("--reference", default="classical_bo",
                   help="planner the paired tests compare against")
    p.add_argument("--out", type=Path, default=Path("examples/benchmark_output"))
    args = p.parse_args()

    print(f"Running {len(args.objectives) * len(args.planners) * args.seeds} campaigns "
          f"({len(args.objectives)} objectives x {len(args.planners)} planners x "
          f"{args.seeds} seeds)...")

    records = run_benchmark(
        objectives=args.objectives, planners=args.planners, seeds=args.seeds,
        budget=args.budget, n_init=args.n_init, n_rounds=args.n_rounds,
    )

    report = summarise(records, reference_planner=args.reference)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "benchmark.md").write_text(report)
    write_csv(records, args.out / "benchmark_runs.csv")
    plot_benchmark(records, args.out / "benchmark.png")

    print("\n" + report)
    print(f"Written: {args.out / 'benchmark.md'}, benchmark_runs.csv, benchmark.png")


if __name__ == "__main__":
    main()
