#!/usr/bin/env python3
"""Honest multi-seed benchmark: agent vs random search.

Runs the campaign across several seeds and reports, for each strategy:
  - mean final best value (± std),
  - mean number of experiments to reach 95% of the brute-force optimum
    (sample efficiency -- the metric that actually matters when each
    experiment is expensive).

Usage:
    python benchmark.py --seeds 10 --budget 50
"""

from __future__ import annotations

import argparse

import numpy as np

from src import oracle
from src.campaign import run_campaign


def experiments_to_threshold(history: list[float], threshold: float) -> float:
    for i, v in enumerate(history, start=1):
        if v >= threshold:
            return i
    return float("nan")  # never reached within budget


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--budget", type=int, default=50)
    p.add_argument("--agent", choices=["auto", "llm", "heuristic"], default="heuristic")
    args = p.parse_args()

    ref = oracle.Oracle(seed=0).reference_optimum()
    thresh = 0.95 * ref

    agent_final, base_final = [], []
    agent_eff, base_eff = [], []

    for s in range(args.seeds):
        res = run_campaign(
            budget=args.budget, n_init=10, n_rounds=12, pool_size=1500,
            max_batch=6, agent_kind=args.agent, seed=s, verbose=False,
        )
        agent_final.append(res.best_value)
        base_final.append(res.baseline_best[-1])
        agent_eff.append(experiments_to_threshold(res.history_best, thresh))
        base_eff.append(experiments_to_threshold(res.baseline_best, thresh))

    def success_rate(xs):
        reached = [x for x in xs if x == x]
        return len(reached), len(xs)

    def mean_eff(xs):
        reached = [x for x in xs if x == x]
        return f"{np.mean(reached):.1f}" if reached else "–"

    a_hit, n = success_rate(agent_eff)
    b_hit, _ = success_rate(base_eff)

    print(f"Reference optimum (brute force): {ref:.2f} MPa")
    print(f"95% threshold:                   {thresh:.2f} MPa")
    print(f"Seeds: {args.seeds} | budget: {args.budget} | agent: {args.agent}\n")
    print(f"{'strategy':<10} {'final best (MPa)':<20} {'reached 95%':<16} {'mean exp. to 95%':<16}")
    print("-" * 66)
    print(f"{'agent':<10} {np.mean(agent_final):6.2f} ± {np.std(agent_final):<11.2f}"
          f"{a_hit}/{n} runs{'':<8} {mean_eff(agent_eff)}")
    print(f"{'random':<10} {np.mean(base_final):6.2f} ± {np.std(base_final):<11.2f}"
          f"{b_hit}/{n} runs{'':<8} {mean_eff(base_eff)}")
    gain = np.mean(agent_final) - np.mean(base_final)
    print(f"\nMean uplift from active learning: {gain:+.2f} MPa over random search "
          f"at the same budget ({n} seeds).")


if __name__ == "__main__":
    main()
