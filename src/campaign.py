"""The active-learning campaign loop.

Ties everything together:

  initial design  ->  [ fit surrogate -> agent decides -> score pool ->
                        select batch -> run experiments -> log ]  x rounds

A random-selection campaign with the same budget is run alongside as a
baseline, so the value added by the agent is measured, not assumed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import acquisition, oracle
from .agent import CampaignState, make_agent
from .surrogate import make_surrogate


@dataclass
class RoundLog:
    round_idx: int
    budget_used: int
    best_value: float
    decision: dict
    selected: list[dict]  # each: {x, predicted_mean, predicted_std, acq_score, measured}


@dataclass
class CampaignResult:
    history_best: list[float]  # best-so-far after each experiment
    rounds: list[RoundLog]
    best_x: np.ndarray
    best_value: float
    baseline_best: list[float]  # random baseline best-so-far per experiment
    reference_optimum: float = field(default=float("nan"))


def _run_baseline(orc: oracle.Oracle, budget: int, rng: np.random.Generator) -> list[float]:
    best = -np.inf
    hist = []
    taken = 0
    while taken < budget:
        x = oracle.sample_candidates(1, rng)[0]
        if not oracle.is_feasible(x):
            continue
        y = orc.run(x)
        best = max(best, y)
        hist.append(best)
        taken += 1
    return hist


def run_campaign(
    *,
    budget: int = 80,
    n_init: int = 12,
    n_rounds: int = 10,
    pool_size: int = 2000,
    max_batch: int = 10,
    agent_kind: str = "auto",
    seed: int = 0,
    verbose: bool = True,
    log=print,
) -> CampaignResult:
    rng = np.random.default_rng(seed)
    orc = oracle.Oracle(seed=seed)
    agent = make_agent(agent_kind)

    if verbose:
        log(f"Agent backend: {type(agent).__name__} (source={getattr(agent, 'source', '?')})")

    # --- Initial design: a feasible random seed set ---
    X_obs, y_obs = [], []
    while len(X_obs) < n_init:
        x = oracle.sample_candidates(1, rng)[0]
        if oracle.is_feasible(x):
            X_obs.append(x)
            y_obs.append(orc.run(x))
    X_obs = np.array(X_obs)
    y_obs = np.array(y_obs)

    history_best = list(np.maximum.accumulate(y_obs))
    rounds: list[RoundLog] = []
    prev_best = float(y_obs.max())
    rounds_since_improvement = 0

    for r in range(n_rounds):
        if orc.n_calls >= budget:
            break

        state = CampaignState(
            round_idx=r,
            n_rounds=n_rounds,
            budget_used=orc.n_calls,
            budget_total=budget,
            best_value=float(max(history_best)),
            last_improvement=float(max(history_best) - prev_best),
            rounds_since_improvement=rounds_since_improvement,
            n_observations=len(X_obs),
        )
        remaining = budget - orc.n_calls
        decision = agent.decide(state, max_batch=min(max_batch, remaining))

        # Fit the surrogate the agent chose.
        surrogate = make_surrogate(decision.surrogate, seed=seed).fit(X_obs, y_obs)

        # Propose a fresh candidate pool: global random exploration PLUS local
        # refinement around the best hits found so far (a trust-region approach).
        # The local scale shrinks as exploration cools (lower kappa -> tighter
        # search), so the agent zooms in on promising regions over time.
        global_pool = oracle.sample_candidates(pool_size, rng)
        local_scale = 0.02 + 0.03 * decision.kappa  # ~2%-8% of each var's range
        top_obs = X_obs[np.argsort(y_obs)[::-1][:3]]
        local_pool = np.vstack(
            [oracle.local_candidates(c, pool_size // 4, local_scale, rng) for c in top_obs]
        )
        pool = np.vstack([global_pool, local_pool])
        pool = np.array([x for x in pool if oracle.is_feasible(x)])
        mean, std = surrogate.predict(pool)
        best_so_far = float(max(history_best))
        scores = acquisition.score(
            decision.acquisition, mean, std, best=best_so_far, kappa=decision.kappa
        )

        # Select the top-k distinct candidates for this round.
        k = min(decision.batch_size, remaining, len(pool))
        top = np.argsort(scores)[::-1][:k]

        selected_logs = []
        for idx in top:
            x = pool[idx]
            measured = orc.run(x)
            X_obs = np.vstack([X_obs, x])
            y_obs = np.append(y_obs, measured)
            history_best.append(max(history_best[-1], measured))
            selected_logs.append(
                {
                    "x": {n: round(float(v), 2) for n, v in zip(oracle.VAR_NAMES, x)},
                    "predicted_mean": round(float(mean[idx]), 2),
                    "predicted_std": round(float(std[idx]), 2),
                    "acq_score": round(float(scores[idx]), 3),
                    "measured": round(float(measured), 2),
                }
            )

        new_best = float(max(history_best))
        improved = new_best > best_so_far + 1e-6
        rounds_since_improvement = 0 if improved else rounds_since_improvement + 1
        prev_best = best_so_far

        rounds.append(
            RoundLog(
                round_idx=r,
                budget_used=orc.n_calls,
                best_value=new_best,
                decision={
                    "acquisition": decision.acquisition,
                    "surrogate": decision.surrogate,
                    "kappa": round(decision.kappa, 2),
                    "batch_size": k,
                    "rationale": decision.rationale,
                    "source": decision.source,
                },
                selected=selected_logs,
            )
        )
        if verbose:
            log(
                f"round {r:>2} | used {orc.n_calls:>3}/{budget} | best {new_best:6.2f} MPa "
                f"| {decision.acquisition:<7} k={decision.kappa:<4} {decision.surrogate:<16} "
                f"| {decision.rationale[:70]}"
            )

    best_idx = int(np.argmax(y_obs))

    # Baseline with an identical budget.
    baseline = _run_baseline(oracle.Oracle(seed=seed + 1000), orc.n_calls, np.random.default_rng(seed + 1))

    return CampaignResult(
        history_best=history_best,
        rounds=rounds,
        best_x=X_obs[best_idx],
        best_value=float(y_obs[best_idx]),
        baseline_best=baseline,
        reference_optimum=orc.reference_optimum(),
    )
