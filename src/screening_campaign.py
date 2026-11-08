"""The screening campaign: active learning over a fixed library under a cost budget.

This is the pool-based, multi-fidelity sibling of `campaign.run_campaign`. The
box campaign is left exactly as it was -- it is covered by tests that assert
bit-exact resume and determinism, and the honest thing is to extend alongside it
rather than rewrite underneath it.

What changes relative to the box loop
-------------------------------------
* **Candidates are library rows, not invented points.** Proposal means choosing
  which unevaluated compounds to look at, and "refine around the best" means
  analogue search in the catalogue.
* **Budget is cost.** A round spends `batch_size x evaluator.cost`, and no round
  may commit more than a fixed share of what is left -- which is what makes a
  cheap wide pass and one expensive measurement genuinely comparable moves
  rather than one obviously dominating.
* **The planner chooses the method.** `Decision.evaluator` names the rung of the
  ladder this round's budget is spent on.
* **Measurements carry fidelity.** The surrogate is fitted with weights, so a
  cheap noisy ranking informs the model without being mistaken for a result.
* **A reviewer can intervene.** The batch passes a human checkpoint before any
  budget is spent, and every veto is recorded.
* **Success is recall, not a maximum.** Finding one good compound is not the
  goal; recovering the good region of the library for as little compute as
  possible is.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .acquisition import per_cost
from .acquisition import score as acq_score
from .diversity import batch_diversity, greedy_maxmin
from .evaluators import DEFAULT_LADDER, MultiFidelityOracle, ladder_for_target
from .human import NoReviewer
from .metrics import json_safe, summarise_campaign, true_top_k
from .planners import CampaignState, make_planner
from .provenance import RunManifest, config_hash, rng_state
from .surrogate import make_surrogate
from .tools import ToolRegistry

#: No single round may commit more than this share of the remaining budget.
#: Without a cap the planner can spend everything in round one and the campaign
#: stops being adaptive; with it, a cheap method naturally affords a wide batch
#: and an expensive one does not.
ROUND_SPEND_CAP = 0.15


def max_batch_for(evaluator, ledger, hard_cap: int = 512) -> int:
    """How many compounds this evaluator may be run on this round."""
    affordable = int((ledger.remaining * ROUND_SPEND_CAP) // max(evaluator.cost, 1e-9))
    return max(1, min(hard_cap, affordable))


@dataclass
class ScreeningRound:
    round_idx: int
    evaluator: str
    batch_size: int
    cost_spent: float
    cumulative_cost: float
    best_value: float
    recall: float
    confirmed_hits: int
    batch_diversity: float
    n_vetoed: int
    decision: dict
    selected: list[dict] = field(default_factory=list)


@dataclass
class ScreeningResult:
    target: str
    planner_name: str
    rounds: list[ScreeningRound]
    metrics: dict
    ledger: dict
    overrides: list[dict]
    manifest: RunManifest | None = None
    tool_calls: list[dict] = field(default_factory=list)
    top_compounds: list[dict] = field(default_factory=list)
    resumed_from_round: int | None = None

    @property
    def recall(self) -> float:
        return self.metrics.get("recall_at_k", float("nan"))

    @property
    def cost(self) -> float:
        return self.metrics.get("cost_spent", float("nan"))


def _evaluator_menu(ladder, ledger) -> list[str]:
    """The menu as the planner sees it: price, fidelity, and what it may buy now."""
    out = []
    for ev in ladder:
        out.append(
            f"{ev.name}: cost {ev.cost:g}/compound, fidelity {ev.fidelity:.2f}, "
            f"max batch this round {max_batch_for(ev, ledger)}, "
            f"{'confirms a hit' if ev.confirms else 'ranking only, cannot confirm a hit'}"
            f" -- {ev.description}"
        )
    return out


def run_screening_campaign(
    *,
    objective,
    planner: str = "heuristic",
    budget: float = 250.0,
    n_init: int = 12,
    n_rounds: int = 20,
    init_evaluator: str = "docking",
    candidate_pool_size: int = 1200,
    hard_max_batch: int = 512,
    top_k: int = 50,
    seed: int = 0,
    cost_aware: bool = True,
    diversity: float | None = None,
    reviewer=None,
    ladder=None,
    out_dir: str | Path | None = None,
    resume: bool = False,
    stop_after_rounds: int | None = None,
    planner_obj=None,
    verbose: bool = True,
    log=print,
) -> ScreeningResult:
    """Run one screening campaign and score it retrospectively.

    `diversity` overrides whatever the planner asks for, which is what makes the
    "does diversity batching help?" comparison a clean one-variable experiment.
    """
    rng = np.random.default_rng(seed)
    target = getattr(objective, "target", objective.name)
    ladder = ladder or ladder_for_target(target, DEFAULT_LADDER)
    oracle = MultiFidelityOracle(objective, ladder, budget=budget, seed=seed)
    reviewer = reviewer or NoReviewer()
    planner_obj = planner_obj or make_planner(planner)
    planner_name = getattr(planner_obj, "name", planner)

    config = {
        "objective": objective.name, "target": target, "planner": planner_name,
        "budget": budget, "n_init": n_init, "n_rounds": n_rounds,
        "candidate_pool_size": candidate_pool_size, "top_k": top_k, "seed": seed,
        "cost_aware": cost_aware, "diversity_override": diversity,
        "reviewer": getattr(reviewer, "name", "none"),
        "pool_hash": objective.pool.fingerprint_hash(),
        "ladder": {ev.name: [ev.cost, round(ev.fidelity, 4)] for ev in ladder},
    }
    manifest = RunManifest(config=config, config_hash=config_hash(config))

    ckpt_path = Path(out_dir) / "checkpoint.json" if out_dir else None
    registry = ToolRegistry()
    registry.add("fit_surrogate", "Fit a surrogate with per-point fidelity weights.",
                 lambda X, y, w, kind: make_surrogate(kind, seed=seed).fit(X, y, weights=w),
                 {"X": "features", "y": "measurements", "w": "fidelity weights", "kind": "surrogate name"})
    registry.add("predict", "Predict value and uncertainty for candidate compounds.",
                 lambda model, X: model.predict(X), {"model": "surrogate", "X": "candidates"})
    registry.add("propose_candidates",
                 "Draw library members this evaluator has not yet settled, plus analogues of the best hits.",
                 lambda n, rng, centers, fidelity: _propose(objective, oracle, n, rng, centers,
                                                            fidelity, reviewer),
                 {"n": "pool size", "rng": "generator", "centers": "best compounds so far",
                  "fidelity": "fidelity of the evaluator about to be used"})
    registry.add("score_candidates", "Rank candidates by an acquisition function.",
                 lambda acquisition, mean, std, best, kappa, rng: acq_score(
                     acquisition, mean, std, best=best, kappa=kappa, rng=rng),
                 {"acquisition": "ucb|ei|greedy|explore|random", "mean": "predictions",
                  "std": "uncertainties", "best": "best so far", "kappa": "exploration strength"})
    registry.add("select_batch", "Choose the batch, trading acquisition score against chemical diversity.",
                 lambda features, scores, k, lambda_div: greedy_maxmin(features, scores, k, lambda_div),
                 {"features": "candidate features", "scores": "acquisition scores",
                  "k": "batch size", "lambda_div": "diversity weight"})
    registry.add("review_batch", "Human/medicinal-chemistry checkpoint before spending budget.",
                 lambda indices, round_idx: reviewer.review(indices, objective, round_idx),
                 {"indices": "proposed compounds", "round_idx": "round number"})
    registry.add("measure", "Spend budget: measure compounds with a named evaluator.",
                 lambda indices, evaluator, round_idx: oracle.measure_batch(indices, evaluator, round_idx),
                 {"indices": "compounds", "evaluator": "ladder rung", "round_idx": "round number"})
    if hasattr(planner_obj, "registry"):
        planner_obj.registry = registry

    target_set = set(int(i) for i in true_top_k(objective.true_values, top_k))
    rounds: list[ScreeningRound] = []
    init_ev = oracle.ladder.get(init_evaluator, ladder[0])
    resumed_from = None

    # ---- resume, or lay down the initial design --------------------------
    if resume and ckpt_path and ckpt_path.exists():
        state_in = _load_checkpoint(ckpt_path, manifest.config_hash)
        _restore(oracle, reviewer, rng, state_in)
        rounds = [ScreeningRound(**r) for r in state_in["rounds"]]
        prev_best = state_in["prev_best"]
        resumed_from = len(rounds)
        if verbose:
            log(f"resumed {target} from round {resumed_from}, "
                f"{oracle.ledger.spent:.1f}/{budget:g} cost already spent")
    else:
        n_init = min(n_init, oracle.ledger.affordable_count(init_ev), len(objective.pool))
        init_idx = rng.choice(len(objective.pool), size=max(n_init, 1), replace=False)
        registry.call("measure", indices=init_idx, evaluator=init_ev.name, round_idx=-1)
        prev_best = max(o.value for o in oracle.observations)

    if verbose:
        log(f"target={target} planner={planner_name} budget={budget:g} "
            f"library={len(objective.pool)} config={manifest.config_hash}")

    # ---- the loop ---------------------------------------------------------
    rounds_this_session = 0
    for r in range(len(rounds), n_rounds):
        if oracle.ledger.exhausted:
            break
        if stop_after_rounds is not None and rounds_this_session >= stop_after_rounds:
            if verbose:
                log(f"stopping after {rounds_this_session} rounds this session; "
                    f"checkpoint written, resume with --resume")
            break
        rounds_this_session += 1
        best_so_far = max(o.value for o in oracle.observations)
        confirmed = oracle.confirmed_indices()

        state = CampaignState(
            round_idx=r, n_rounds=n_rounds,
            budget_used=int(oracle.ledger.spent), budget_total=int(budget),
            best_value=float(best_so_far),
            last_improvement=float(best_so_far - prev_best),
            rounds_since_improvement=_stall(rounds),
            n_observations=len(oracle.best_estimate_per_index()),
            cost_spent=round(oracle.ledger.spent, 4), cost_total=float(budget),
            cost_by_evaluator=dict(oracle.ledger.by_evaluator),
            evaluator_menu=_evaluator_menu(ladder, oracle.ledger),
            pool_remaining=len(objective.pool) - len(oracle.evaluated_indices()),
            confirmed_hits=len(confirmed & target_set),
        )

        decision = planner_obj.decide(state, max_batch=hard_max_batch)
        ev = oracle.ladder.get(decision.evaluator or init_ev.name, init_ev)
        lam = decision.diversity if diversity is None else diversity

        X, y, w = oracle.training_set()
        if len(np.unique(y)) < 2:
            y = y + rng.normal(0, 1e-6, size=len(y))
        surrogate = registry.call("fit_surrogate", X=X, y=y, w=w, kind=decision.surrogate)

        best_idx = sorted(oracle.best_estimate_per_index().items(),
                          key=lambda kv: -kv[1][0])[:3]
        centers = objective.pool.features[[i for i, _ in best_idx]] if best_idx else None
        cand = registry.call("propose_candidates", n=candidate_pool_size, rng=rng,
                             centers=centers, fidelity=ev.fidelity)
        if len(cand) == 0:
            # This rung has nothing left to add -- a cheap pass can exhaust the
            # whole library for a rounding error, which is precisely its appeal.
            # That is not the end of the campaign: climb to the most accurate
            # rung that is still affordable and still has compounds to settle.
            for better in sorted(ladder, key=lambda e: -e.fidelity):
                if not oracle.ledger.can_afford(better):
                    continue
                alt = registry.call("propose_candidates", n=candidate_pool_size, rng=rng,
                                    centers=centers, fidelity=better.fidelity)
                if len(alt):
                    ev, cand = better, alt
                    decision.rationale = (f"[{ev.name} substituted: no compounds left for "
                                          f"{decision.evaluator}] " + decision.rationale)
                    break
            if len(cand) == 0:
                break

        mean, std = registry.call("predict", model=surrogate, X=objective.pool.features[cand])
        scores = registry.call("score_candidates", acquisition=decision.acquisition,
                               mean=mean, std=std, best=float(best_so_far),
                               kappa=decision.kappa, rng=rng)
        if cost_aware:
            # Ranking *within* one evaluator is unaffected by its price, but the
            # magnitude matters once the planner is comparing what a round buys.
            scores = per_cost(scores, ev.cost, exponent=0.0)

        k = min(decision.batch_size, max_batch_for(ev, oracle.ledger, hard_max_batch),
                oracle.ledger.affordable_count(ev), len(cand))
        if k < 1:
            break
        local = registry.call("select_batch", features=objective.pool.features[cand],
                              scores=scores, k=k, lambda_div=lam)
        chosen = [int(cand[i]) for i in local]

        review = registry.call("review_batch", indices=chosen, round_idx=r)
        kept = review.kept
        if not kept:
            # Everything was vetoed: record the round and move on rather than
            # silently spending nothing and looping forever.
            rounds.append(ScreeningRound(r, ev.name, 0, 0.0, oracle.ledger.spent,
                                         best_so_far, _recall(oracle, target_set),
                                         len(oracle.confirmed_indices() & target_set),
                                         0.0, review.n_vetoed, _decision_dict(decision, ev, lam)))
            continue

        before = oracle.ledger.spent
        values = registry.call("measure", indices=kept, evaluator=ev.name, round_idx=r)
        spent = oracle.ledger.spent - before

        new_best = max(best_so_far, float(np.max(values)))
        if hasattr(planner_obj, "observe"):
            planner_obj.observe(ev.name, new_best - best_so_far, spent)

        selected = [
            {"index": int(i), "smiles": objective.smiles_at(int(i)),
             "measured": round(float(v), 4), "evaluator": ev.name,
             "predicted": round(float(mean[list(cand).index(i)]), 4) if i in list(cand) else None,
             "is_true_top_k": int(i) in target_set}
            for i, v in zip(kept, np.atleast_1d(values))
        ]
        rounds.append(ScreeningRound(
            r, ev.name, len(kept), round(spent, 4), round(oracle.ledger.spent, 4),
            round(new_best, 4), _recall(oracle, target_set),
            len(oracle.confirmed_indices() & target_set),
            round(batch_diversity(objective.pool.features, kept), 4),
            review.n_vetoed, _decision_dict(decision, ev, lam), selected))

        if verbose:
            log(f"  round {r:2d} {ev.name:10s} n={len(kept):4d} cost={spent:7.2f} "
                f"total={oracle.ledger.spent:8.2f} best={new_best:.2f} "
                f"recall={rounds[-1].recall:.2f} vetoed={review.n_vetoed}")

        prev_best = best_so_far
        if ckpt_path:
            _save_checkpoint(ckpt_path, manifest.config_hash, oracle, reviewer, rng,
                             rounds, prev_best)

    metrics = summarise_campaign(oracle, objective, k=top_k)
    best = sorted(oracle.best_estimate_per_index().items(), key=lambda kv: -kv[1][0])[:25]
    top_compounds = [
        {"index": int(i), "smiles": objective.smiles_at(int(i)),
         "estimate": round(float(v), 4), "fidelity": round(float(f), 3),
         "true_value": round(float(objective.true_values[int(i)]), 4),
         "is_true_top_k": int(i) in target_set,
         "is_known_active": bool(getattr(objective, "known_actives", np.zeros(1))[int(i)])
         if hasattr(objective, "known_actives") else False}
        for i, (v, f) in best
    ]

    result = ScreeningResult(
        target=target, planner_name=planner_name, rounds=rounds, metrics=metrics,
        resumed_from_round=resumed_from,
        ledger=oracle.ledger.as_dict(),
        overrides=[o.as_dict() for o in getattr(reviewer, "overrides", [])],
        manifest=manifest, tool_calls=registry.log_as_dicts(), top_compounds=top_compounds,
    )

    if out_dir:
        _write_outputs(Path(out_dir), result, objective, oracle)
    return result


# --------------------------------------------------------------------------
# Checkpointing.
#
# A screening campaign on a cluster will not finish inside one wall-time slot,
# so it has to survive being cut in half. Everything that makes a run
# reproducible is written out: every measurement, the cost ledger, the
# reviewer's blocklist and the internal state of the random generator. Resuming
# with a different configuration is refused rather than silently producing a
# mixture of two experiments.
# --------------------------------------------------------------------------


def _save_checkpoint(path: Path, cfg_hash: str, oracle, reviewer, rng, rounds, prev_best) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config_hash": cfg_hash,
        "prev_best": float(prev_best),
        "rng_state": rng_state(rng),
        "oracle_rng_state": rng_state(oracle._rng),
        "ledger": {"total": oracle.ledger.total, "spent": oracle.ledger.spent,
                   "by_evaluator": oracle.ledger.by_evaluator, "calls": oracle.ledger.calls},
        "observations": [vars(o) for o in oracle.observations],
        "blocked": sorted(getattr(reviewer, "blocked", set()) or set()),
        "overrides": [o.as_dict() for o in getattr(reviewer, "overrides", [])],
        "rounds": [r.__dict__ for r in rounds],
    }
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, default=str))
    tmp.replace(path)


def _load_checkpoint(path: Path, cfg_hash: str) -> dict:
    import json

    state = json.loads(path.read_text())
    if state.get("config_hash") != cfg_hash:
        raise ValueError(
            f"checkpoint in {path} was written for configuration "
            f"{state.get('config_hash')}, but this run is {cfg_hash}. Refusing to "
            "resume: the result would be a mixture of two different experiments."
        )
    return state


def _restore(oracle, reviewer, rng, state: dict) -> None:
    """Put both generators back exactly where they were, not merely re-seeded."""
    from .evaluators import Observation

    # restore_rng builds a fresh generator, so assign the state in place: the
    # caller holds a reference to `rng` and must keep using the same object.
    rng.bit_generator.state = state["rng_state"]
    oracle._rng.bit_generator.state = state["oracle_rng_state"]
    oracle.observations = [Observation(**o) for o in state["observations"]]
    oracle.ledger.total = state["ledger"]["total"]
    oracle.ledger.spent = state["ledger"]["spent"]
    oracle.ledger.by_evaluator = dict(state["ledger"]["by_evaluator"])
    oracle.ledger.calls = dict(state["ledger"]["calls"])
    if hasattr(reviewer, "blocked"):
        reviewer.blocked = set(state.get("blocked", []))


def _decision_dict(decision, ev, lam) -> dict:
    return {"acquisition": decision.acquisition, "surrogate": decision.surrogate,
            "kappa": round(decision.kappa, 3), "batch_size": decision.batch_size,
            "evaluator": ev.name, "diversity": round(float(lam), 3),
            "source": decision.source, "rationale": decision.rationale}


def _stall(rounds: list[ScreeningRound]) -> int:
    count = 0
    for i in range(len(rounds) - 1, 0, -1):
        if rounds[i].best_value > rounds[i - 1].best_value + 1e-9:
            break
        count += 1
    return count


def _recall(oracle, target_set: set[int]) -> float:
    if not target_set:
        return float("nan")
    return round(len(oracle.confirmed_indices() & target_set) / len(target_set), 4)


def _propose(objective, oracle, n: int, rng, centers, fidelity: float,
             reviewer=None) -> np.ndarray:
    """Library members this evaluator could still add something to.

    The exclusion rule has to be **per fidelity**, not per compound. A compound
    that a near-free ranking pass has looked at is not finished: the entire
    point of a cheap triage pass is to decide what to spend the expensive
    method on next. An earlier version excluded anything already measured at
    all, which quietly made triage self-defeating -- every compound the cascade
    ranked became permanently ineligible for the docking run it was ranked for.
    So a compound is only out of scope once it has been measured at equal or
    better fidelity than the method about to be used.
    """
    settled = {
        o.index for o in oracle.observations if o.fidelity >= fidelity - 1e-9
    }
    blocked = getattr(reviewer, "blocked", set()) or set()
    available = np.array([i for i in range(len(objective.pool))
                          if i not in settled and i not in blocked], dtype=int)
    if len(available) == 0:
        return available
    take = min(n, len(available))
    pool = rng.choice(available, size=take, replace=False)
    if centers is not None and len(centers):
        local = []
        for c in np.atleast_2d(centers):
            local.extend(objective.pool.neighbours(c, max(n // 8, 1), rng, pool_subset=available))
        pool = np.unique(np.concatenate([pool, np.array(local, dtype=int)]))
    return pool


def _write_outputs(out: Path, result: ScreeningResult, objective, oracle) -> None:
    import json

    out.mkdir(parents=True, exist_ok=True)
    def dump(name: str, payload) -> None:
        (out / name).write_text(json.dumps(json_safe(payload), indent=2, default=str))

    dump("manifest.json", {"config": result.manifest.config,
                           "config_hash": result.manifest.config_hash,
                           "ledger": result.ledger, "metrics": result.metrics})
    dump("rounds.json", [r.__dict__ for r in result.rounds])
    dump("top_compounds.json", result.top_compounds)
    dump("overrides.json", result.overrides)
    dump("tool_calls.json", result.tool_calls)
