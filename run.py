#!/usr/bin/env python3
"""Command-line entry point for an active-learning campaign.

Examples
--------
Run fully offline (deterministic heuristic planner):
    python run.py --agent heuristic --seed 0

Let Claude do the planning (needs ANTHROPIC_API_KEY in the environment):
    export ANTHROPIC_API_KEY=sk-...
    python run.py --agent llm

Auto: use the LLM if a key is present, else fall back to the heuristic:
    python run.py --agent auto
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.campaign import run_campaign
from src.report import write_report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--agent", choices=["auto", "llm", "heuristic"], default="auto",
                   help="planning backend (default: auto)")
    p.add_argument("--budget", type=int, default=80, help="total experiment budget")
    p.add_argument("--n-init", type=int, default=12, help="size of the initial random design")
    p.add_argument("--n-rounds", type=int, default=10, help="number of active-learning rounds")
    p.add_argument("--pool-size", type=int, default=2000, help="candidate pool sampled per round")
    p.add_argument("--max-batch", type=int, default=10, help="max experiments per round")
    p.add_argument("--seed", type=int, default=0, help="random seed")
    p.add_argument("--out", type=Path, default=Path("examples/demo_output"), help="output directory")
    p.add_argument("--quiet", action="store_true", help="suppress per-round logging")
    args = p.parse_args()

    res = run_campaign(
        budget=args.budget,
        n_init=args.n_init,
        n_rounds=args.n_rounds,
        pool_size=args.pool_size,
        max_batch=args.max_batch,
        agent_kind=args.agent,
        seed=args.seed,
        verbose=not args.quiet,
    )

    write_report(res, args.out)

    uplift = ""
    if res.baseline_best:
        gain = res.best_value - res.baseline_best[-1]
        uplift = f"  (+{gain:.2f} MPa over random baseline)"
    print(f"\nBest strength found: {res.best_value:.2f} MPa{uplift}")
    print(f"Report written to: {args.out / 'report.md'}")
    print(f"Plot written to:   {args.out / 'convergence.png'}")


if __name__ == "__main__":
    main()
