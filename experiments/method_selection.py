#!/usr/bin/env python3
"""Does choosing the method per target beat committing to one method?

This is the project's central claim, written as an experiment that can fail.

The reading list argues that brute-force docking, tranche pruning, co-folding
and retrieval sit at different points on a cost-accuracy curve and that none
generalises. If that is right, a strategy that picks per target should beat the
best fixed method -- and, more tellingly, the best fixed method should not be
the same one on every target. If one method wins everywhere, an orchestration
layer is unnecessary here and this script is where that gets found out.

Run:
    python experiments/method_selection.py --variants 4 --seeds 8
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.objectives.screening import target_variants  # noqa: E402
from src.screening_benchmark import run_grid, save, summarise  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--targets", nargs="+", default=["egfr", "cdk2", "cmet"])
    ap.add_argument("--variants", type=int, default=4,
                    help="synthetic landscape variants per real target (0 = real targets only)")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--budget", type=float, default=240.0)
    ap.add_argument("--rounds", type=int, default=16)
    ap.add_argument("--decoys", type=int, default=1500)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--out", default="examples/method_selection")
    ap.add_argument("--planners", nargs="+", default=[
        "always_docking", "always_cofolding", "always_ml_score",
        "cascade", "bandit",
    ])
    args = ap.parse_args()

    targets = (target_variants(args.targets, args.variants) if args.variants
               else list(args.targets))
    print(f"{len(targets)} landscapes x {len(args.planners)} strategies x {args.seeds} seeds "
          f"= {len(targets) * len(args.planners) * args.seeds} campaigns")

    records = run_grid(targets, args.planners, seeds=args.seeds, budget=args.budget,
                       n_rounds=args.rounds, n_decoys=args.decoys, top_k=args.top_k)

    out = save(records, args.out)
    report = summarise(records, adaptive=[p for p in args.planners if not p.startswith("always_")])
    (out / "report.md").write_text(report)
    print("\n" + report)
    print(f"\nwritten to {out}/report.md")


if __name__ == "__main__":
    main()
