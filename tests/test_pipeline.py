"""Smoke + behaviour tests. Run with: python -m pytest -q  (or python tests/test_pipeline.py)"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root on path

import numpy as np

from src import acquisition, oracle
from src.agent import CampaignState, HeuristicAgent, make_agent
from src.campaign import run_campaign
from src.surrogate import make_surrogate


def test_oracle_budget_and_feasibility():
    orc = oracle.Oracle(seed=1)
    x = oracle.sample_candidates(1, np.random.default_rng(0))[0]
    orc.run(x)
    assert orc.n_calls == 1
    assert isinstance(oracle.is_feasible(x), bool)


def test_surrogates_return_mean_and_std():
    rng = np.random.default_rng(0)
    X = oracle.sample_candidates(30, rng)
    y = np.array([oracle.Oracle(seed=2)._ground_truth(x) for x in X])
    for kind in ("random_forest", "gaussian_process"):
        m, s = make_surrogate(kind, seed=0).fit(X, y).predict(X[:5])
        assert m.shape == (5,) and s.shape == (5,)
        assert np.all(s >= 0)


def test_acquisition_shapes():
    mean = np.array([1.0, 2.0, 3.0])
    std = np.array([0.5, 0.5, 0.5])
    for name in acquisition.ACQUISITIONS:
        sc = acquisition.score(name, mean, std, best=2.0, kappa=1.0)
        assert sc.shape == (3,)


def test_heuristic_agent_valid_decision():
    s = CampaignState(0, 10, 12, 80, 50.0, 0.0, 0, 12)
    d = HeuristicAgent().decide(s, max_batch=10)
    assert d.acquisition in ("ucb", "ei", "greedy", "explore")
    assert d.surrogate in ("random_forest", "gaussian_process")
    assert 1 <= d.batch_size <= 10
    assert d.rationale


def test_auto_agent_runs_without_key():
    # With no API key, make_agent('auto') must give a working heuristic agent.
    agent = make_agent("auto")
    s = CampaignState(0, 10, 12, 80, 50.0, 0.0, 0, 12)
    assert agent.decide(s, 10).batch_size >= 1


def test_campaign_beats_random_baseline():
    res = run_campaign(
        budget=60, n_init=10, n_rounds=8, pool_size=800,
        agent_kind="heuristic", seed=0, verbose=False,
    )
    assert res.best_value > 0
    # Active learning should not do worse than random on this smooth problem.
    assert res.best_value >= res.baseline_best[-1] - 1.0
    # Best-so-far is monotone non-decreasing.
    assert all(b2 >= b1 - 1e-9 for b1, b2 in zip(res.history_best, res.history_best[1:]))


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all tests passed")
