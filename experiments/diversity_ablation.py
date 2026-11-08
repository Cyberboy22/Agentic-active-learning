#!/usr/bin/env python3
"""Does diversity-aware batching beat taking the top-k?

Acquisition functions are smooth, so the k highest scorers in a library tend to
be k close analogues of each other. The received wisdom is that this wastes a
batch. This checks whether it does, here, by varying one number and nothing
else.

Run:
    python experiments/diversity_ablation.py --seeds 8
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from src.screening_benchmark import bootstrap_ci, run_grid, save  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--targets", nargs="+", default=["egfr", "cdk2", "cmet"])
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--budget", type=float, default=240.0)
    ap.add_argument("--levels", nargs="+", type=float, default=[0.0, 0.3, 0.6, 1.0])
    ap.add_argument("--out", default="examples/diversity_ablation")
    args = ap.parse_args()

    all_records = []
    for lam in args.levels:
        recs = run_grid(args.targets, ["always_docking"], seeds=args.seeds,
                        budget=args.budget, n_rounds=16, n_decoys=1500,
                        diversity=lam, verbose=False)
        for r in recs:
            r.planner = f"lambda={lam}"
        all_records.extend(recs)
        print(f"lambda={lam}: recall {np.mean([r.recall for r in recs]):.3f}")

    baseline = [r.recall for r in all_records if r.planner == f"lambda={args.levels[0]}"]
    lines = ["# Diversity-aware batching", "",
             f"`always_docking` on {len(args.targets)} targets x {args.seeds} seeds, "
             "varying only the diversity weight. lambda=0 is plain top-k selection.", "",
             "| lambda | recall | Δ vs top-k | 95% CI |", "|---|---|---|---|"]
    for lam in args.levels:
        vals = [r.recall for r in all_records if r.planner == f"lambda={lam}"]
        delta = np.array(vals) - np.array(baseline)
        lo, hi = bootstrap_ci(delta)
        lines.append(f"| {lam} | {np.mean(vals):.3f} ± {np.std(vals):.3f} | "
                     f"{delta.mean():+.3f} | [{lo:+.3f}, {hi:+.3f}] |")
    lines += ["", "A confidence interval spanning zero means this did not help here. "
                  "That is a result, not a failed experiment.", ""]
    out = save(all_records, args.out)
    (out / "report.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
