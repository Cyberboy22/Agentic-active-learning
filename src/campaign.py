"""The active-learning campaign loop.

    initial design ─▶ ┌──────────────────────────────────────────┐
                      │ fit surrogate (mean + uncertainty)       │
                      │ planner decides the round's strategy     │
                      │ propose candidates (global + local)      │
                      │ score and select the top-k               │
                      │ spend budget: run the experiments        │
                      │ checkpoint                               │
                      └────────────────┬─────────────────────────┘
                                       ▼ repeat
                              result + provenance

Every step runs through the tool registry, so the log records what was actually
executed; every round is checkpointed, so an interrupted campaign resumes
exactly where it stopped; and the whole configuration is hashed into a manifest,
so a result can be traced back to the setup that produced it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .objective import Objective, Oracle, get_objective
from .objectives import *  # noqa: F401,F403  (registers the built-in objectives)
from .planners import CampaignState, make_planner
from .provenance import Checkpoint, RunManifest, config_hash, restore_rng, rng_state
from .tools import build_default_registry

_REFERENCE_CACHE: dict[str, float] = {}


def reference_optimum(objective: Objective) -> float:
    """Best achievable noise-free value, cached per objective (reporting only)."""
    key = f"{objective.name}:{objective.dim}"
    if key not in _REFERENCE_CACHE:
        _REFERENCE_CACHE[key] = objective.reference_optimum()
    return _REFERENCE_CACHE[key]


@dataclass
class RoundLog:
    round_idx: int
    budget_used: int
    best_value: float
    decision: dict
    selected: list[dict]


@dataclass
class CampaignResult:
    objective_name: str
    planner_name: str
    history_best: list[float]
    rounds: list[RoundLog]
    best_x: np.ndarray
    best_value: float
    X_obs: np.ndarray
    y_obs: np.ndarray
    var_names: list[str]
    units: list[str]
    value_unit: str
    reference_optimum: float
    manifest: RunManifest | None = None
    tool_summary: dict = field(default_factory=dict)
    tool_calls: list[dict] = field(default_factory=list)
    resumed_from_round: int | None = None

    @property
    def budget_used(self) -> int:
        return self.rounds[-1].budget_used if self.rounds else len(self.y_obs)

    def fraction_of_reference(self) -> float:
        ref = self.reference_optimum
        if not np.isfinite(ref) or ref == 0:
            return float("nan")
        return self.best_value / ref


def run_campaign(
    *,
    objective: str | Objective = "concrete",
    planner: str = "auto",
    budget: int = 50,
    n_init: int = 10,
    n_rounds: int = 12,
    pool_size: int = 1500,
    max_batch: int = 6,
    seed: int = 0,
    out_dir: str | Path | None = None,
    resume: bool = False,
    stop_after_rounds: int | None = None,
    compute_reference: bool = True,
    verbose: bool = True,
    log=print,
) -> CampaignResult:
    """Run (or continue) one campaign.

    `stop_after_rounds` limits how many rounds *this session* executes, then
    returns with the checkpoint saved. It deliberately does not enter the config
    hash: it describes how the work was chunked, not what the campaign is. That
    is what lets a long campaign run in pieces under a wall-time limit and still
    produce exactly the result an uninterrupted run would have.
    """
    obj = get_objective(objective) if isinstance(objective, str) else objective
    planner_obj = make_planner(planner)
    planner_name = getattr(planner_obj, "name", planner)

    config = {
        "objective": obj.name, "dim": obj.dim, "planner": planner, "budget": budget,
        "n_init": n_init, "n_rounds": n_rounds, "pool_size": pool_size,
        "max_batch": max_batch, "seed": seed,
    }
    cfg_hash = config_hash(config)
    out_path = Path(out_dir) if out_dir else None
    ckpt_path = out_path / "checkpoint.json" if out_path else None

    rng = np.random.default_rng(seed)
    oracle = Oracle(obj, seed=seed)
    registry = build_default_registry(obj, oracle, seed=seed)

    X_obs: np.ndarray
    y_obs: np.ndarray
    rounds: list[RoundLog] = []
    resumed_from: int | None = None

    # ---- resume, or start a fresh initial design -------------------------
    if resume and ckpt_path and Checkpoint.exists(ckpt_path):
        ck = Checkpoint.load(ckpt_path)
        if ck.config_hash != cfg_hash:
            raise ValueError(
                f"checkpoint at {ckpt_path} was written for a different configuration "
                f"({ck.config_hash} != {cfg_hash}); delete it or change --out"
            )
        X_obs = np.array(ck.X_obs, dtype=float)
        y_obs = np.array(ck.y_obs, dtype=float)
        history_best = list(ck.history_best)
        rounds = [RoundLog(**r) for r in ck.rounds]
        rng = restore_rng(ck.rng_state)
        oracle._rng = restore_rng(ck.oracle_rng_state)
        oracle.n_calls = ck.oracle_calls
        resumed_from = len(rounds)
        if verbose:
            log(f"Resumed from {ckpt_path} at round {resumed_from} "
                f"({oracle.n_calls}/{budget} budget already spent)")
    else:
        X_init = obj.sample_feasible(n_init, rng)
        y_init = registry.call("run_experiment", X=X_init)
        X_obs, y_obs = X_init, np.asarray(y_init, dtype=float)
        history_best = list(np.maximum.accumulate(y_obs))

    manifest = RunManifest(config=config, config_hash=cfg_hash)
    if verbose:
        log(f"objective={obj.name}(d={obj.dim})  planner={planner_name}  "
            f"budget={budget}  config={cfg_hash}")

    # Rebuild the planner-visible history exactly as it evolved, so a resumed
    # campaign sees the same state an uninterrupted one would have at this point.
    # best_before[i] is the best value known *before* round i ran.
    best_before = [float(history_best[min(n_init, len(history_best)) - 1])]
    best_before += [float(r.best_value) for r in rounds]
    rounds_since_improvement = 0
    for i in range(len(rounds)):
        improved = best_before[i + 1] > best_before[i] + 1e-9
        rounds_since_improvement = 0 if improved else rounds_since_improvement + 1
    prev_best = best_before[len(rounds) - 1] if rounds else best_before[0]

    # ---- the loop --------------------------------------------------------
    rounds_this_session = 0
    for r in range(len(rounds), n_rounds):
        if oracle.n_calls >= budget:
            break
        if stop_after_rounds is not None and rounds_this_session >= stop_after_rounds:
            if verbose:
                log(f"stopping after {rounds_this_session} rounds this session; "
                    f"checkpoint saved, resume with --resume")
            break
        rounds_this_session += 1
        remaining = budget - oracle.n_calls
        best_so_far = float(max(history_best))

        state = CampaignState(
            round_idx=r, n_rounds=n_rounds, budget_used=oracle.n_calls,
            budget_total=budget, best_value=best_so_far,
            last_improvement=best_so_far - prev_best,
            rounds_since_improvement=rounds_since_improvement,
            n_observations=len(X_obs),
        )
        decision = planner_obj.decide(state, max_batch=min(max_batch, remaining))

        surrogate = registry.call(
            "fit_surrogate", X=X_obs, y=y_obs, kind=decision.surrogate
        )
        # Local scale shrinks as exploration cools, so the search zooms in.
        centers = X_obs[np.argsort(y_obs)[::-1][:3]]
        pool = registry.call(
            "propose_candidates", n=pool_size, rng=rng, centers=centers,
            scale=0.02 + 0.03 * decision.kappa,
        )
        mean, std = registry.call("predict", model=surrogate, X=pool)
        scores = registry.call(
            "score_candidates", acquisition=decision.acquisition, mean=mean, std=std,
            best=best_so_far, kappa=decision.kappa, rng=rng,
        )

        k = int(min(decision.batch_size, remaining, len(pool)))
        top = np.argsort(scores)[::-1][:k]
        measured = registry.call("run_experiment", X=pool[top])

        selected = []
        for slot, idx in enumerate(top):
            X_obs = np.vstack([X_obs, pool[idx]])
            y_obs = np.append(y_obs, measured[slot])
            history_best.append(max(history_best[-1], float(measured[slot])))
            selected.append({
                "x": {n: round(float(v), 4) for n, v in zip(obj.var_names, pool[idx])},
                "predicted_mean": round(float(mean[idx]), 4),
                "predicted_std": round(float(std[idx]), 4),
                "acq_score": round(float(scores[idx]), 4),
                "measured": round(float(measured[slot]), 4),
            })

        new_best = float(max(history_best))
        rounds_since_improvement = 0 if new_best > best_so_far + 1e-9 else rounds_since_improvement + 1
        prev_best = best_so_far

        rounds.append(RoundLog(
            round_idx=r, budget_used=oracle.n_calls, best_value=new_best,
            decision={
                "acquisition": decision.acquisition, "surrogate": decision.surrogate,
                "kappa": round(decision.kappa, 3), "batch_size": k,
                "rationale": decision.rationale, "source": decision.source,
            },
            selected=selected,
        ))

        if ckpt_path:
            Checkpoint(
                X_obs=X_obs.tolist(), y_obs=y_obs.tolist(), history_best=history_best,
                rounds=[vars(x) for x in rounds], rng_state=rng_state(rng),
                oracle_rng_state=rng_state(oracle._rng), oracle_calls=oracle.n_calls,
                config_hash=cfg_hash,
            ).save(ckpt_path)

        if verbose:
            log(f"round {r:>2} | used {oracle.n_calls:>3}/{budget} | best {new_best:9.4f} "
                f"| {decision.acquisition:<7} k={decision.kappa:<4} {decision.surrogate:<16} "
                f"| {decision.rationale[:64]}")

    best_idx = int(np.argmax(y_obs))
    result = CampaignResult(
        objective_name=obj.name, planner_name=planner_name,
        history_best=history_best, rounds=rounds,
        best_x=X_obs[best_idx], best_value=float(y_obs[best_idx]),
        X_obs=X_obs, y_obs=y_obs,
        var_names=obj.var_names, units=obj.units, value_unit=obj.value_unit,
        reference_optimum=reference_optimum(obj) if compute_reference else float("nan"),
        manifest=manifest.finish(),
        tool_summary=registry.summary(), tool_calls=registry.log_as_dicts(),
        resumed_from_round=resumed_from,
    )
    if out_path:
        result.manifest.save(out_path / "manifest.json")
    return result
