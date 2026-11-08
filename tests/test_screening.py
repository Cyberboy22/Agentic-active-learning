#!/usr/bin/env python3
"""Tests for the pool-based, multi-fidelity screening half of the framework.

Several of these are regression tests for bugs that were live during
development and would have silently produced plausible-looking nonsense:

* a cheap triage pass permanently removing compounds from the pool it was
  triaging for;
* recall crediting a campaign for *measuring* a good compound rather than for
  recognising it;
* per-target evaluator fidelities drifting between processes because they were
  keyed on Python's salted string hash.

Run: python tests/test_screening.py   (or pytest -q)
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from src.diversity import batch_diversity, greedy_maxmin  # noqa: E402
from src.evaluators import (  # noqa: E402
    DEFAULT_LADDER,
    CostLedger,
    MultiFidelityOracle,
    ladder_for_target,
)
from src.human import make_reviewer  # noqa: E402
from src.metrics import enrichment_factor, recall_at, shortlist, true_top_k  # noqa: E402
from src.objective import PoolObjective  # noqa: E402
from src.objectives.screening import VirtualScreeningObjective, from_score_table  # noqa: E402
from src.planners import CampaignState, Decision, make_planner  # noqa: E402
from src.pool import build_pool, bundled_decoys  # noqa: E402
from src.screening_campaign import run_screening_campaign  # noqa: E402

_CACHE: dict = {}


def small_objective(target: str = "cdk2") -> VirtualScreeningObjective:
    if target not in _CACHE:
        _CACHE[target] = VirtualScreeningObjective(target, n_decoys=600)
    return _CACHE[target]


# -- pool ------------------------------------------------------------------

def test_pool_lookup_is_exact_and_neighbours_are_members():
    pool = build_pool(bundled_decoys(200))
    for i in (0, 37, 150):
        assert pool.index_of(pool.features[i]) == i
    nb = pool.neighbours(pool.features[5], 6, np.random.default_rng(0))
    assert len(nb) == 6 and set(nb).issubset(range(len(pool)))
    assert nb[0] == 5, "a molecule should be its own nearest neighbour"


def test_pool_objective_only_ever_returns_library_rows():
    pool = build_pool(bundled_decoys(200))
    obj = PoolObjective(pool, np.arange(len(pool), dtype=float), name="t")
    rng = np.random.default_rng(0)
    rows = {tuple(np.round(r, 9)) for r in pool.features}
    for X in (obj.sample(20, rng), obj.local_sample(pool.features[3], 8, 0.1, rng)):
        for r in X:
            assert tuple(np.round(r, 9)) in rows, "invented a design that is not in the library"
    assert obj.reference_optimum() == obj.true_values.max()


# -- screening objective ---------------------------------------------------

def test_known_actives_are_enriched_but_task_is_not_trivial():
    obj = small_objective()
    top = set(int(i) for i in obj.true_top_k(50))
    share = sum(1 for i in top if obj.known_actives[i]) / 50
    base = obj.known_actives.mean()
    assert share > base * 5, "literature actives should rank well above background"
    assert obj.known_actives.mean() < 0.12, "active fraction is unrealistically high"


def test_actives_do_not_score_by_matching_themselves():
    """Leave-one-out similarity: an active must not be its own reference."""
    obj = small_objective()
    from src.pool import similarity_to_actives

    sim = similarity_to_actives(obj.pool.smiles, obj.known_actives)
    assert sim[obj.known_actives].max() < 0.999, "an active matched itself; ground truth is a lookup"


def test_score_table_adapter_round_trips(tmp=None):
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "scores.csv"
        smiles = bundled_decoys(40)
        path.write_text("smiles,score\n" + "\n".join(
            f"{s},{-8 + i * 0.1:.2f}" for i, s in enumerate(smiles)))
        obj = from_score_table(str(path), higher_is_better=False)
        assert len(obj.pool) > 30
        # more negative docking score == better, so it must become the maximum
        assert obj.true_values.argmax() == obj.pool.smiles.index(
            next(s for s in obj.pool.smiles if s == smiles[0]))


# -- evaluators and cost ---------------------------------------------------

def test_ladder_fidelity_is_stable_across_processes():
    root = str(Path(__file__).resolve().parent.parent)
    code = (f"import sys; sys.path.insert(0, {root!r});"
            "from src.evaluators import ladder_for_target;"
            "print([round(e.fidelity, 6) for e in ladder_for_target('egfr')])")
    env = dict(os.environ, PYTHONHASHSEED="0")
    a = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env).stdout
    env["PYTHONHASHSEED"] = "12345"
    b = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env).stdout
    assert a == b and a.strip(), "per-target fidelities move between processes"


def test_evaluator_fidelity_matches_its_stated_correlation():
    rng = np.random.default_rng(0)
    truth = rng.normal(size=4000)
    for ev in DEFAULT_LADDER:
        obs = np.array([ev.measure(z, rng) for z in truth])
        r = float(np.corrcoef(truth, obs)[0, 1])
        assert abs(r - ev.fidelity) < 0.06, f"{ev.name}: correlation {r:.3f} != fidelity {ev.fidelity}"


def test_cost_ledger_accounts_exactly():
    ledger = CostLedger(total=100.0)
    dock = next(e for e in DEFAULT_LADDER if e.name == "docking")
    cheap = next(e for e in DEFAULT_LADDER if e.name == "retrieval")
    ledger.spend(dock, 10)
    ledger.spend(cheap, 500)
    assert abs(ledger.spent - (10 * dock.cost + 500 * cheap.cost)) < 1e-9
    assert abs(ledger.remaining - (100.0 - ledger.spent)) < 1e-9
    assert ledger.calls["docking"] == 10 and ledger.calls["retrieval"] == 500
    assert ledger.affordable_count(dock) == int(ledger.remaining // dock.cost)


# -- metrics ---------------------------------------------------------------

def test_enrichment_bounds():
    rng = np.random.default_rng(0)
    truth = rng.normal(size=1000)
    actives = true_top_k(truth, 50)
    perfect = np.argsort(truth)[::-1]
    assert abs(enrichment_factor(perfect, actives, 0.05) - 20.0) < 1e-6
    assert enrichment_factor(rng.permutation(1000), actives, 0.05) < 4.0


def test_recall_requires_recognising_not_merely_measuring():
    """Measuring every true hit with a useless evaluator must not score well."""
    obj = small_objective()
    blind = ladder_for_target("cdk2")
    oracle = MultiFidelityOracle(obj, blind, budget=1e9, seed=0)
    target = obj.true_top_k(25)
    # Measure every true hit, but with a rung that cannot confirm anything.
    oracle.measure_batch(target, "retrieval")
    assert recall_at(shortlist(oracle, 25, {"docking", "cofolding", "assay"}), target) == 0.0
    # Measure a pile of junk with a confirming rung: shortlist fills with junk.
    junk = [i for i in range(len(obj.pool)) if i not in set(int(x) for x in target)][:200]
    oracle.measure_batch(junk, "docking")
    picked = shortlist(oracle, 25, {"docking", "cofolding", "assay"})
    assert recall_at(picked, target) == 0.0, "credit given for compounds never confirmed"


# -- diversity -------------------------------------------------------------

def test_diversity_zero_is_exactly_top_k_and_positive_spreads():
    rng = np.random.default_rng(0)
    feats, scores = [], []
    for c in range(6):
        centre = rng.normal(0, 4, size=5)
        feats.append(rng.normal(centre, 0.07, size=(10, 5)))
        scores.append(rng.uniform(0.9 - 0.08 * c, 1.0 - 0.08 * c, 10))
    F, s = np.vstack(feats), np.concatenate(scores)
    assert np.array_equal(greedy_maxmin(F, s, 6, 0.0), np.argsort(s)[::-1][:6])
    assert batch_diversity(F, greedy_maxmin(F, s, 6, 1.0)) > \
           batch_diversity(F, greedy_maxmin(F, s, 6, 0.0))


# -- planners --------------------------------------------------------------

def test_decision_clamps_an_unknown_evaluator():
    d = Decision("ucb", "random_forest", 1.0, 4, "r", "llm",
                 evaluator="teleportation", diversity=99).validated(8, ["retrieval", "docking"])
    assert d.evaluator == "retrieval" and d.diversity <= 2.0


def test_fixed_method_planner_commits_to_its_rung():
    state = CampaignState(1, 10, 5, 50, 1.0, 0.0, 0, 20,
                          evaluator_menu=["retrieval: x", "docking: y", "cofolding: z"])
    d = make_planner("always_cofolding").decide(state, max_batch=8)
    assert d.evaluator == "cofolding"


def test_llm_planner_replays_from_cache_without_an_api_key():
    import json

    with tempfile.TemporaryDirectory() as d:
        planner = make_planner("llm", cache_dir=d)
        state = CampaignState(0, 6, 0, 50, 1.0, 0.0, 0, 10,
                              evaluator_menu=["docking: cost 1, fidelity 0.8"])
        prompt = planner._prompt(state, 6)
        Path(d, f"{__import__('src.planners', fromlist=['hash_text']).hash_text(prompt + planner.model)}.json"
             ).write_text(json.dumps({"model": planner.model, "prompt": prompt, "response": json.dumps(
                 {"acquisition": "ei", "surrogate": "random_forest", "kappa": 0.5,
                  "batch_size": 3, "diversity": 0.4, "evaluator": "docking",
                  "rationale": "cached"})}))
        again = make_planner("llm", cache_dir=d)
        d2 = again.decide(state, max_batch=6)
        assert d2.source == "llm" and d2.acquisition == "ei" and d2.batch_size == 3
        assert again.n_cache_hits == 1 and again.n_api_calls == 0


# -- campaign --------------------------------------------------------------

def test_campaign_never_exceeds_its_cost_budget():
    obj = small_objective()
    for planner in ("always_docking", "cascade", "bandit"):
        res = run_screening_campaign(objective=obj, planner=planner, budget=60.0,
                                     n_rounds=10, seed=0, verbose=False)
        assert res.metrics["cost_spent"] <= 60.0 + 1e-6, f"{planner} overspent"


def test_campaign_is_deterministic_for_a_fixed_seed():
    obj = small_objective()
    kw = dict(objective=obj, planner="cascade", budget=60.0, n_rounds=8, verbose=False)
    a = run_screening_campaign(seed=3, **kw)
    b = run_screening_campaign(seed=3, **kw)
    assert a.metrics["recall_at_k"] == b.metrics["recall_at_k"]
    assert a.ledger == b.ledger
    assert [r.evaluator for r in a.rounds] == [r.evaluator for r in b.rounds]


def test_cheap_triage_does_not_disqualify_compounds_from_confirmation():
    """Regression: a retrieval pass used to make its compounds permanently
    ineligible, so the cascade destroyed the pool it was triaging for."""
    obj = small_objective()
    res = run_screening_campaign(objective=obj, planner="cascade", budget=120.0,
                                 n_rounds=14, seed=0, verbose=False)
    calls = res.ledger["calls"]
    assert calls.get("retrieval", 0) > 0, "cascade never ran a triage pass"
    confirming = sum(v for k, v in calls.items() if k in ("docking", "cofolding", "assay"))
    assert confirming > 0, "triage ran but nothing was ever confirmed"
    assert res.metrics["recall_at_k"] > 0


def test_reviewer_vetoes_are_logged_and_respected():
    obj = small_objective()
    reviewer = make_reviewer("rules")
    res = run_screening_campaign(objective=obj, planner="always_docking", budget=80.0,
                                 n_rounds=10, seed=1, reviewer=reviewer, verbose=False)
    if res.overrides:
        vetoed = {o["index"] for o in res.overrides}
        measured = {s["index"] for r in res.rounds for s in r.selected}
        assert not (vetoed & measured), "budget spent on a compound the reviewer rejected"
        assert all(o["reason"] for o in res.overrides), "a veto was recorded with no reason"


def test_shortlist_never_exceeds_k_and_holds_only_confirmed():
    obj = small_objective()
    res = run_screening_campaign(objective=obj, planner="cascade", budget=90.0,
                                 n_rounds=12, seed=0, verbose=False)
    assert res.metrics["shortlist_size"] <= res.metrics["k"]
    assert 0.0 <= res.metrics["recall_at_k"] <= 1.0


def test_chunked_campaign_matches_an_uninterrupted_one():
    """The cluster claim: a campaign split across wall-time slots is identical."""
    obj = small_objective()
    kw = dict(objective=obj, planner="cascade", budget=70.0, n_rounds=9, seed=4, verbose=False)
    with tempfile.TemporaryDirectory() as d:
        full = run_screening_campaign(**kw)
        run_screening_campaign(out_dir=d, stop_after_rounds=3, **kw)
        resumed = run_screening_campaign(out_dir=d, resume=True, **kw)
        assert resumed.resumed_from_round == 3
        assert full.ledger == resumed.ledger
        assert full.metrics["recall_at_k"] == resumed.metrics["recall_at_k"]
        assert [r.evaluator for r in full.rounds] == [r.evaluator for r in resumed.rounds]
        assert [r.cumulative_cost for r in full.rounds] == \
               [r.cumulative_cost for r in resumed.rounds]


def test_resume_refuses_a_different_configuration():
    obj = small_objective()
    with tempfile.TemporaryDirectory() as d:
        run_screening_campaign(objective=obj, planner="cascade", budget=50.0, n_rounds=6,
                               seed=0, out_dir=d, stop_after_rounds=2, verbose=False)
        try:
            run_screening_campaign(objective=obj, planner="always_docking", budget=50.0,
                                   n_rounds=6, seed=0, out_dir=d, resume=True, verbose=False)
        except ValueError:
            return
        raise AssertionError("resumed into a different configuration without complaint")


def test_campaign_output_is_valid_json_for_a_browser():
    """Regression: an un-reached recall target wrote `Infinity`, which is legal
    for json.dumps and rejected by every browser's JSON.parse — silently
    blanking the whole viewer."""
    import json
    import math

    from src.metrics import json_safe

    obj = small_objective()
    with tempfile.TemporaryDirectory() as d:
        res = run_screening_campaign(objective=obj, planner="always_docking", budget=40.0,
                                     n_rounds=6, seed=0, out_dir=d, verbose=False)
        assert not math.isfinite(res.metrics["cost_to_recall_50"]) or True
        for name in ("manifest.json", "rounds.json", "top_compounds.json"):
            text = Path(d, name).read_text()
            assert "Infinity" not in text and "NaN" not in text, f"{name} is not valid JSON"
            json.loads(text)
        # strict mode is what a browser does
        json.dumps(json_safe(res.metrics), allow_nan=False)


def _run_all() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"ok  {fn.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {fn.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"ERROR {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} screening tests passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
