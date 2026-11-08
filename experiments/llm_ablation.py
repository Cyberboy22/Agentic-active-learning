#!/usr/bin/env python3
"""Is the LLM planner actually reading the campaign state?

A language model asked to plan a round will always return a confident strategy
and a fluent justification for it. That proves nothing. The question is whether
the decision depends on the state it was shown.

So run the same campaigns three ways:

  real      the planner sees the true campaign state
  shuffled  it sees the same fields with the values permuted -- plausible
            magnitudes, wrong numbers
  withheld  it sees no state at all, only the menu of options

If `real` does not beat `shuffled`, the planner is not using the state, and its
per-round rationales are narration written after the fact. That would be worth
knowing before putting an LLM in charge of a compute budget, and it is the kind
of control that almost nobody runs on their own agent.

Needs ANTHROPIC_API_KEY for the first pass. Every prompt and reply is cached to
disk, so repeat runs are free and exactly reproducible -- which is what makes
running this across enough seeds affordable at all.

Run:
    python experiments/llm_ablation.py --seeds 6 --cache .llm_cache
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from src.planners import LLMPlanner  # noqa: E402
from src.screening_benchmark import paired_comparison, run_grid, save  # noqa: E402

MODES = ["real", "shuffled", "withheld"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--targets", nargs="+", default=["egfr", "cdk2", "cmet"])
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--budget", type=float, default=240.0)
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--decoys", type=int, default=1500)
    ap.add_argument("--cache", default=".llm_cache")
    ap.add_argument("--model", default="claude-sonnet-5-5")
    ap.add_argument("--out", default="examples/llm_ablation")
    args = ap.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY") and not Path(args.cache).exists():
        print("No ANTHROPIC_API_KEY and no cache: every planner would silently fall back\n"
              "to the heuristic and all three arms would be identical. Refusing to run a\n"
              "comparison that cannot mean anything.")
        raise SystemExit(1)

    def factory(name: str, seed: int):
        return LLMPlanner(model=args.model, cache_dir=f"{args.cache}/{name}",
                          state_mode=name, temperature=0.0)

    records = run_grid(args.targets, MODES, seeds=args.seeds, budget=args.budget,
                       n_rounds=args.rounds, n_decoys=args.decoys,
                       planner_factory=factory)
    out = save(records, args.out)

    lines = ["# Is the LLM planner reading the state?", "",
             "Same campaigns, three state conditions, paired by seed.", "",
             "| target | real | shuffled | withheld | real − shuffled | p |",
             "|---|---|---|---|---|---|"]
    for t in args.targets:
        means = {m: np.mean([r.recall for r in records if r.target == t and r.planner == m])
                 for m in MODES}
        c = paired_comparison(records, "real", "shuffled", t, "recall")
        lines.append(f"| {t} | {means['real']:.3f} | {means['shuffled']:.3f} | "
                     f"{means['withheld']:.3f} | {c['delta']:+.3f} | {c['p']:.3f} |")

    overall = {m: np.mean([r.recall for r in records if r.planner == m]) for m in MODES}
    gap = overall["real"] - overall["shuffled"]
    lines += ["", f"Overall: real {overall['real']:.3f}, shuffled {overall['shuffled']:.3f}, "
                  f"withheld {overall['withheld']:.3f}.", "",
              ("**The planner is using the state**: corrupting it costs measurable performance."
               if gap > 0.02 else
               "**No evidence the planner is using the state.** Feeding it shuffled numbers "
               "changed almost nothing, so the rationales it produced each round were "
               "fluent narration rather than reasoning over the campaign. On this "
               "evidence an LLM planner is not earning its API calls here, and saying so "
               "is more useful than shipping it."), ""]
    report = "\n".join(lines)
    (out / "report.md").write_text(report)
    print("\n" + report)


if __name__ == "__main__":
    main()
