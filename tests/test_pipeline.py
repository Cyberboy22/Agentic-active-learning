"""Smoke + behaviour tests.  Run:  python tests/test_pipeline.py   (or: pytest -q)"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root on path

import numpy as np  # noqa: E402

from src import acquisition  # noqa: E402
from src.benchmark import RunRecord, run_benchmark, summarise  # noqa: E402
from src.campaign import run_campaign  # noqa: E402
from src.objective import Objective, Oracle, available_objectives, get_objective  # noqa: E402
from src.objectives import *  # noqa: F401,F403,E402
from src.planners import CampaignState, HeuristicPlanner, available_planners, make_planner  # noqa: E402
from src.provenance import Checkpoint, config_hash, restore_rng, rng_state  # noqa: E402
from src.surrogate import make_surrogate  # noqa: E402
from src.tools import ToolRegistry  # noqa: E402


# ---------------------------------------------------------------- objectives
def test_objectives_registered_and_sane():
    assert {"concrete", "branin", "hartmann6", "ackley", "rastrigin"} <= set(available_objectives())
    for name in ("concrete", "branin", "hartmann6"):
        obj = get_objective(name)
        low, high = obj.bounds
        assert len(low) == len(high) == obj.dim == len(obj.var_names)
        assert np.all(high > low)
        x = obj.sample(1, np.random.default_rng(0))[0]
        assert isinstance(obj.evaluate(x), float)
        assert isinstance(obj.is_feasible(x), bool)


def test_synthetic_optima_are_attained_at_known_points():
    # Hartmann6 and Ackley should hit their documented optima at the known argmax.
    ack = get_objective("ackley")
    assert abs(ack.evaluate(np.zeros(ack.dim)) - ack.reference_optimum()) < 1e-9
    ras = get_objective("rastrigin")
    assert abs(ras.evaluate(np.zeros(ras.dim)) - ras.reference_optimum()) < 1e-9


def test_oracle_counts_budget_and_adds_noise():
    obj = get_objective("hartmann6")
    orc = Oracle(obj, seed=1)
    x = obj.sample(1, np.random.default_rng(0))[0]
    a, b = orc.run(x), orc.run(x)
    assert orc.n_calls == 2
    assert a != b  # noisy


# ---------------------------------------------------------------- components
def test_surrogates_return_mean_and_std():
    obj = get_objective("branin")
    rng = np.random.default_rng(0)
    X = obj.sample(30, rng)
    y = np.array([obj.evaluate(x) for x in X])
    for kind in ("random_forest", "gaussian_process"):
        m, s = make_surrogate(kind, seed=0).fit(X, y).predict(X[:5])
        assert m.shape == (5,) and s.shape == (5,)
        assert np.all(s >= 0)


def test_acquisitions_all_score():
    mean, std = np.array([1.0, 2.0, 3.0]), np.array([0.5, 0.5, 0.5])
    for name in acquisition.ACQUISITIONS:
        sc = acquisition.score(name, mean, std, best=2.0, kappa=1.0,
                               rng=np.random.default_rng(0))
        assert sc.shape == (3,)


def test_planners_return_valid_decisions():
    s = CampaignState(0, 10, 12, 50, 1.0, 0.0, 0, 12)
    for kind in available_planners():
        d = make_planner(kind).decide(s, max_batch=6)
        assert d.acquisition in list(acquisition.ACQUISITIONS)
        assert d.surrogate in ("random_forest", "gaussian_process")
        assert 1 <= d.batch_size <= 6
        assert d.rationale


def test_llm_planner_falls_back_without_api_key():
    agent = make_planner("auto")
    s = CampaignState(0, 10, 12, 50, 1.0, 0.0, 0, 12)
    assert agent.decide(s, 6).batch_size >= 1


def test_decision_clamps_out_of_range_llm_output():
    from src.planners import Decision

    d = Decision("nonsense", "nonsense", 99.0, 10_000, "x", "llm").validated(max_batch=6)
    assert d.acquisition == "ucb" and d.surrogate == "random_forest"
    assert d.kappa == 5.0 and d.batch_size == 6


# ---------------------------------------------------------------- tool layer
def test_tool_registry_logs_calls_and_failures():
    reg = ToolRegistry()
    reg.add("double", "Double a number.", lambda v: v * 2, {"v": "a number"})

    @reg.register("boom", "Always fails.")
    def _boom():
        raise RuntimeError("nope")

    assert reg.call("double", v=21) == 42
    try:
        reg.call("boom")
    except RuntimeError:
        pass
    assert set(reg.names()) == {"double", "boom"}
    assert "double(v: a number)" in reg.describe()
    summary = reg.summary()
    assert summary["double"]["calls"] == 1 and summary["double"]["failures"] == 0
    assert summary["boom"]["failures"] == 1


# ---------------------------------------------------------------- provenance
def test_config_hash_is_stable_and_order_independent():
    assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})
    assert config_hash({"a": 1}) != config_hash({"a": 2})


def test_rng_state_roundtrip_is_exact():
    rng = np.random.default_rng(7)
    rng.random(5)
    state = rng_state(rng)
    expected = rng.random(3)
    assert np.allclose(restore_rng(state).random(3), expected)


def test_campaign_is_deterministic_for_a_fixed_seed():
    kw = dict(objective="branin", planner="heuristic", budget=24, n_init=6,
              n_rounds=5, pool_size=300, max_batch=4, verbose=False)
    a = run_campaign(seed=3, **kw)
    b = run_campaign(seed=3, **kw)
    assert a.best_value == b.best_value
    assert np.allclose(a.history_best, b.history_best)
    assert a.manifest.config_hash == b.manifest.config_hash


def test_checkpoint_resume_matches_an_uninterrupted_run():
    kw = dict(objective="branin", planner="heuristic", budget=24, n_init=6,
              n_rounds=6, pool_size=300, max_batch=4, seed=5, verbose=False)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        # Interrupt after 3 of the 6 rounds, then resume the same campaign.
        run_campaign(out_dir=out, stop_after_rounds=3, **kw)
        assert Checkpoint.exists(out / "checkpoint.json")
        resumed = run_campaign(out_dir=out, resume=True, **kw)
        full = run_campaign(**kw)
    assert resumed.resumed_from_round == 3
    assert np.allclose(resumed.history_best, full.history_best)
    assert resumed.best_value == full.best_value


def test_resume_rejects_a_mismatched_config():
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        run_campaign(objective="branin", planner="heuristic", budget=12, n_init=4,
                     n_rounds=2, pool_size=200, max_batch=4, seed=0,
                     out_dir=out, verbose=False)
        try:
            run_campaign(objective="branin", planner="heuristic", budget=99, n_init=4,
                         n_rounds=2, pool_size=200, max_batch=4, seed=0,
                         out_dir=out, resume=True, verbose=False)
        except ValueError as exc:
            assert "different configuration" in str(exc)
        else:
            raise AssertionError("expected a config-mismatch error")


# ---------------------------------------------------------------- end to end
def test_campaign_respects_budget_and_is_monotone():
    res = run_campaign(objective="concrete", planner="heuristic", budget=36,
                       n_init=8, n_rounds=7, pool_size=500, max_batch=5,
                       seed=0, verbose=False)
    assert res.budget_used <= 36
    assert len(res.y_obs) <= 36
    assert all(b2 >= b1 - 1e-9 for b1, b2 in zip(res.history_best, res.history_best[1:]))
    assert res.tool_summary["run_experiment"]["calls"] >= 1


def test_custom_objective_plugs_in():
    class Parabola(Objective):
        name = "_test_parabola"

        @property
        def var_names(self):
            return ["x", "y"]

        @property
        def bounds(self):
            return np.array([-5.0, -5.0]), np.array([5.0, 5.0])

        def evaluate(self, x):
            return float(-(x[0] ** 2 + x[1] ** 2))

        def reference_optimum(self, n_scan=0, seed=0):
            return 0.0

    res = run_campaign(objective=Parabola(), planner="heuristic", budget=30,
                       n_init=6, n_rounds=6, pool_size=400, max_batch=4,
                       seed=0, verbose=False)
    assert res.best_value > -5.0  # got meaningfully close to the optimum at 0


def test_benchmark_runs_and_summarises():
    records = run_benchmark(objectives=["branin"], planners=["random", "heuristic"],
                            seeds=3, budget=24, n_init=6, n_rounds=5,
                            pool_size=300, max_batch=4, log=None)
    assert len(records) == 6
    assert all(0.0 <= r.gap <= 1.0 for r in records)
    # Paired design: same seed => same initial design across planners.
    by_seed = {}
    for r in records:
        by_seed.setdefault(r.seed, []).append(r.init_best)
    assert all(np.allclose(v, v[0]) for v in by_seed.values())
    text = summarise(records, reference_planner="random")
    assert "Planner benchmark" in text and "gap closed" in text


def test_run_record_handles_degenerate_span():
    rec = RunRecord("o", "p", 0, init_best=1.0, final_best=1.0, reference=1.0,
                    regret=0.0, gap=1.0, budget_used=1, experiments_to_target=float("nan"))
    assert rec.gap == 1.0


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        fn()
        print(f"ok  {name}")
    print(f"\nall {len(tests)} tests passed")
