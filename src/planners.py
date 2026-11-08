"""Planners: the decision layer. Given campaign state, choose the next move.

Every planner answers the same question -- *how should this round be run?* --
and returns the same `Decision`: which surrogate to fit, which acquisition
function to rank with, how hard to explore, how many experiments to spend, and
a one-line **rationale** recording why. The rationale is what makes a campaign
auditable rather than a black box.

Four planners ship, and they form a ladder that lets the benchmark harness ask a
real question: *does agentic adaptivity actually beat the standard approach?*

  random       ignores the model; the honest floor
  classical_bo textbook Bayesian optimisation -- GP + Expected Improvement, fixed
  heuristic    an adaptive schedule: explore early, refine late, break plateaus
  bandit       UCB1 over the evaluator menu -- the non-LLM way to choose a method
  llm          Claude reads the campaign state and chooses the strategy itself

When the campaign has a *ladder* of evaluators rather than a single oracle, a
decision also names which one to spend budget with. That is the orchestration
question proper: not "what should I measure next" but "what should I measure it
*with*, at what price". `bandit` exists so that question has a principled
non-LLM baseline, because "better than nothing" is not the bar.

`llm` falls back to `heuristic` whenever no API key is set or a call fails, so a
run never breaks and the repository is fully testable offline.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path


def hash_text(text: str) -> str:
    """Stable short hash -- used to key the LLM decision cache."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]

ACQ_CHOICES = ["ucb", "ei", "greedy", "explore"]
SURR_CHOICES = ["random_forest", "gaussian_process", "ensemble_mlp"]


@dataclass
class CampaignState:
    """What a planner is allowed to know when it decides."""

    round_idx: int
    n_rounds: int
    budget_used: int
    budget_total: int
    best_value: float
    last_improvement: float
    rounds_since_improvement: int
    n_observations: int
    #: Cost view. Present only for multi-fidelity campaigns; a single-oracle
    #: campaign leaves these at their defaults and nothing reads them.
    cost_spent: float = 0.0
    cost_total: float = 0.0
    cost_by_evaluator: dict = field(default_factory=dict)
    evaluator_menu: list = field(default_factory=list)
    pool_remaining: int = 0
    confirmed_hits: int = 0


@dataclass
class Decision:
    acquisition: str
    surrogate: str
    kappa: float
    batch_size: int
    rationale: str
    source: str
    #: Which evaluator to spend this round's budget with. None means the
    #: campaign has a single oracle and there is nothing to choose.
    evaluator: str | None = None
    #: How much acquisition score to trade for a chemically spread-out batch.
    diversity: float = 0.0

    def validated(self, max_batch: int, evaluators: list[str] | None = None) -> Decision:
        """Clamp a decision into the legal range -- the guard rail around the LLM.

        Everything a planner returns passes through here, so a model that
        hallucinates an acquisition function, asks for a thousand experiments or
        names an evaluator that does not exist degrades to a legal decision
        instead of crashing a campaign that may be hours in.
        """
        if self.acquisition not in ACQ_CHOICES + ["random"]:
            self.acquisition = "ucb"
        if self.surrogate not in SURR_CHOICES:
            self.surrogate = "random_forest"
        self.kappa = float(min(max(self.kappa, 0.0), 5.0))
        self.batch_size = int(min(max(self.batch_size, 1), max_batch))
        self.diversity = float(min(max(self.diversity, 0.0), 2.0))
        if evaluators:
            if self.evaluator not in evaluators:
                self.evaluator = evaluators[0]
        return self


def _even_batch(s: CampaignState, max_batch: int) -> int:
    """Spread the remaining budget evenly over the remaining rounds."""
    remaining_rounds = max(s.n_rounds - s.round_idx, 1)
    return max(1, min(max_batch, round((s.budget_total - s.budget_used) / remaining_rounds)))


class RandomPlanner:
    """The floor: spend the budget on uniformly random feasible designs."""

    name = "random"
    source = "random"

    def decide(self, s: CampaignState, max_batch: int) -> Decision:
        return Decision(
            acquisition="random",
            surrogate="random_forest",  # fitted but ignored by the random acquisition
            kappa=0.0,
            batch_size=_even_batch(s, max_batch),
            rationale="random search baseline: candidates drawn uniformly, model ignored",
            source=self.source,
        ).validated(max_batch)


class ClassicalBOPlanner:
    """Textbook Bayesian optimisation: Gaussian process + Expected Improvement.

    Deliberately non-adaptive -- the same strategy every round. This is the
    baseline the agentic planners have to beat to justify their complexity.
    """

    name = "classical_bo"
    source = "classical_bo"

    def decide(self, s: CampaignState, max_batch: int) -> Decision:
        return Decision(
            acquisition="ei",
            surrogate="gaussian_process",
            kappa=0.0,
            batch_size=_even_batch(s, max_batch),
            rationale="classical BO: GP surrogate with Expected Improvement, fixed strategy",
            source=self.source,
        ).validated(max_batch)


class HeuristicPlanner:
    """An adaptive schedule encoding standard active-learning intuition.

    Explore early, refine late, and fire a single exploration burst when progress
    stalls -- alternating rather than staying hot, so the search can escape a
    basin without abandoning the good region it already found. Deterministic, so
    it is fully reproducible and testable.
    """

    name = "heuristic"
    source = "heuristic"

    def decide(self, s: CampaignState, max_batch: int) -> Decision:
        progress = s.round_idx / max(s.n_rounds - 1, 1)

        if progress < 0.34:
            acq, kappa = "ucb", 2.0
            why = "early phase: map the space, so explore with a high UCB kappa"
        elif progress < 0.70:
            acq, kappa = "ucb", 1.0
            why = "mid phase: balance explore/exploit with a moderate UCB kappa"
        else:
            acq, kappa = "ei", 0.0
            why = "late phase: refine the best region with Expected Improvement"

        if s.rounds_since_improvement >= 2 and s.rounds_since_improvement % 2 == 0:
            acq, kappa = "ucb", 2.5
            why = (
                f"stalled for {s.rounds_since_improvement} rounds: one exploration "
                "burst (raise kappa) to escape the current basin"
            )

        # GP is better in the low-data regime; hand over to RF once data grows.
        surrogate = "gaussian_process" if s.n_observations < 40 else "random_forest"
        batch = _even_batch(s, max_batch)

        return Decision(
            acq, surrogate, kappa, batch,
            f"{why}; using {surrogate} (n={s.n_observations} observations); batch={batch}.",
            self.source,
        ).validated(max_batch)


class LLMPlanner:
    """Claude reads the campaign state and chooses the strategy, with a rationale.

    Three things here are deliberate and worth reading before trusting it.

    **The menu is data.** The tool list in the prompt is rendered from the live
    `ToolRegistry`, not written into the prompt by hand. Registering a lab's
    docking wrapper therefore changes what the agent can choose *without anyone
    editing this file* -- which is the claim the registry exists to support, and
    which an earlier version of this class quietly failed to honour by
    hard-coding the menu.

    **Decisions are cached.** Prompt and reply are written to disk keyed by a
    hash of the prompt, so a benchmark can replay a campaign exactly, for free,
    as many times as it needs. Without this the LLM planner cannot be run across
    twenty seeds, which means the central claim of the project -- that the agent
    helps -- is the one thing never measured.

    **The state can be withheld.** `state_mode` corrupts or removes the campaign
    state before the model sees it. If a planner given shuffled state performs
    as well as one given real state, it is not reading the state, and its
    rationales are post-hoc narration. That experiment is in
    `experiments/llm_ablation.py`.
    """

    name = "llm"
    source = "llm"

    def __init__(self, model: str = "claude-sonnet-5-5", fallback=None,
                 cache_dir: str | None = None, state_mode: str = "real",
                 registry=None, temperature: float = 0.0):
        self.model = model
        self.fallback = fallback or HeuristicPlanner()
        self.state_mode = state_mode
        self.registry = registry
        self.temperature = temperature
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.n_cache_hits = 0
        self.n_api_calls = 0
        self._client = None
        try:
            import anthropic  # noqa: F401

            if os.environ.get("ANTHROPIC_API_KEY"):
                self._client = anthropic.Anthropic()
        except Exception:
            self._client = None

    @property
    def available(self) -> bool:
        return self._client is not None or self._cache_only()

    def _cache_only(self) -> bool:
        return bool(self.cache_dir and any(self.cache_dir.glob("*.json")))

    # -- the prompt --------------------------------------------------------

    def _state_payload(self, s: CampaignState) -> dict:
        """What the model is shown -- possibly degraded, for the ablation."""
        payload = {k: v for k, v in asdict(s).items() if not isinstance(v, (dict, list))}
        payload["evaluator_menu"] = list(s.evaluator_menu)
        if self.state_mode == "real":
            return payload
        if self.state_mode == "withheld":
            return {"evaluator_menu": list(s.evaluator_menu),
                    "note": "campaign state deliberately withheld"}
        if self.state_mode == "shuffled":
            # Same keys, same plausible magnitudes, wrong values. A planner that
            # is genuinely reading the state should do measurably worse here.
            rng = random.Random(hash_text(json.dumps(payload, sort_keys=True, default=str)))
            numeric = {k: v for k, v in payload.items() if isinstance(v, (int, float))}
            shuffled_values = list(numeric.values())
            rng.shuffle(shuffled_values)
            for key, value in zip(numeric, shuffled_values):
                payload[key] = value
            return payload
        raise ValueError(f"unknown state_mode {self.state_mode!r}")

    def _prompt(self, s: CampaignState, max_batch: int) -> str:
        menu = self.registry.describe() if self.registry is not None else "(no tool registry attached)"
        evaluator_text = ""
        if s.evaluator_menu:
            evaluator_text = (
                "\nEvaluators you may spend budget with this round "
                "(name, cost per compound, fidelity 0-1):\n"
                + "\n".join(f"  - {e}" for e in s.evaluator_menu)
                + "\n  Budget is measured in cost, not in number of experiments. A cheap\n"
                  "  low-fidelity pass over many compounds and one expensive high-fidelity\n"
                  "  measurement can cost the same; choosing between them is your job.\n"
            )
        return (
            "You are the planner of an active-learning campaign that optimises an "
            "expensive experiment by choosing which candidates to measure next.\n\n"
            f"State:\n{json.dumps(self._state_payload(s), indent=2, default=str)}\n\n"
            f"Tools available (rendered from the live registry):\n{menu}\n"
            f"{evaluator_text}\n"
            "Strategy choices:\n"
            f"- surrogate: one of {SURR_CHOICES} "
            "(gaussian_process is better with little data; random_forest is robust later; "
            "ensemble_mlp scales to large pools and high-dimensional features)\n"
            f"- acquisition: one of {ACQ_CHOICES} "
            "(ucb explores via mean+kappa*std; ei refines near the best; "
            "greedy pure-exploits; explore maximises uncertainty)\n"
            "- kappa: float 0..5 (UCB exploration strength)\n"
            f"- batch_size: int 1..{max_batch} (compounds to measure this round)\n"
            "- diversity: float 0..2 (how much acquisition score to trade for a "
            "chemically spread-out batch; 0 takes the top scorers even if they are "
            "near-identical analogues)\n"
            + ("- evaluator: one of the names listed above\n" if s.evaluator_menu else "")
            + "\nPick a strategy appropriate to the phase of the campaign and whether "
            "progress has stalled. Reply with ONLY a JSON object with keys "
            '"acquisition","surrogate","kappa","batch_size","diversity"'
            + (',"evaluator"' if s.evaluator_menu else "")
            + ',"rationale". '
            'The "rationale" is one sentence of evidence-based justification.'
        )

    # -- caching -----------------------------------------------------------

    def _cache_path(self, prompt: str) -> Path | None:
        if not self.cache_dir:
            return None
        return self.cache_dir / f"{hash_text(prompt + self.model)}.json"

    def _cached(self, prompt: str) -> str | None:
        path = self._cache_path(prompt)
        if path and path.exists():
            try:
                return json.loads(path.read_text())["response"]
            except Exception:
                return None
        return None

    def _store(self, prompt: str, response: str) -> None:
        path = self._cache_path(prompt)
        if path:
            path.write_text(json.dumps(
                {"model": self.model, "prompt": prompt, "response": response}, indent=2
            ))

    # -- deciding ----------------------------------------------------------

    def decide(self, s: CampaignState, max_batch: int) -> Decision:
        evaluators = [e.split()[0].rstrip(":") for e in s.evaluator_menu] or None
        prompt = self._prompt(s, max_batch)

        text = self._cached(prompt)
        if text is not None:
            self.n_cache_hits += 1
        elif self._client is None:
            d = self.fallback.decide(s, max_batch)
            d.source = "heuristic(no-api-key)"
            return d.validated(max_batch, evaluators)
        else:
            try:
                msg = self._client.messages.create(
                    model=self.model, max_tokens=400, temperature=self.temperature,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
                self.n_api_calls += 1
                self._store(prompt, text)
            except Exception as exc:
                d = self.fallback.decide(s, max_batch)
                d.rationale = f"[LLM unavailable: {type(exc).__name__}; used heuristic] " + d.rationale
                d.source = "heuristic(fallback)"
                return d.validated(max_batch, evaluators)

        try:
            start, end = text.find("{"), text.rfind("}") + 1
            data = json.loads(text[start:end])
            return Decision(
                acquisition=str(data["acquisition"]),
                surrogate=str(data["surrogate"]),
                kappa=float(data["kappa"]),
                batch_size=int(data["batch_size"]),
                rationale=str(data.get("rationale", "")).strip() or "(no rationale given)",
                source=self.source,
                evaluator=str(data["evaluator"]) if data.get("evaluator") else None,
                diversity=float(data.get("diversity", 0.0)),
            ).validated(max_batch, evaluators)
        except Exception as exc:
            d = self.fallback.decide(s, max_batch)
            d.rationale = f"[unparseable LLM reply: {type(exc).__name__}; used heuristic] " + d.rationale
            d.source = "heuristic(fallback)"
            return d.validated(max_batch, evaluators)


class BanditPlanner:
    """UCB1 over the evaluator menu -- the baseline an LLM planner has to beat.

    Before claiming a language model is good at choosing methods, it is worth
    knowing what a textbook bandit does with the same choice. This keeps a
    running reward per evaluator -- improvement achieved per unit cost -- and
    picks by UCB1. It is three lines of theory, it has regret guarantees, and in
    my experience on this framework it is a genuinely hard baseline.

    The acquisition schedule is delegated to the heuristic planner, so the only
    thing being compared is the *method-selection* decision.
    """

    name = "bandit"
    source = "bandit"

    def __init__(self, c: float = 1.4):
        self.c = c
        self.counts: dict[str, int] = {}
        self.rewards: dict[str, float] = {}
        self._schedule = HeuristicPlanner()
        self._last: tuple[str, float] | None = None

    def observe(self, evaluator: str, improvement: float, cost: float) -> None:
        """Feed back what the last choice actually bought, per unit cost.

        `improvement` must be progress the campaign can *act on* -- a rise in
        the best **confirmed** value. An earlier version of this rewarded any
        improvement per unit cost, which a near-free ranking pass wins trivially
        by producing a tiny gain for a tiny price. The bandit duly learned to
        spend the whole campaign on a method that cannot confirm anything. That
        is a reward-specification bug, not a finding about bandits, and it is
        worth leaving in the comments because it is the obvious trap here.
        """
        gain = max(improvement, 0.0) / max(cost, 1e-9)
        self.counts[evaluator] = self.counts.get(evaluator, 0) + 1
        n = self.counts[evaluator]
        prev = self.rewards.get(evaluator, 0.0)
        self.rewards[evaluator] = prev + (gain - prev) / n

    def decide(self, s: CampaignState, max_batch: int) -> Decision:
        d = self._schedule.decide(s, max_batch)
        names = [e.split()[0].rstrip(":") for e in s.evaluator_menu]
        if not names:
            d.source = self.source
            return d.validated(max_batch)

        total = max(sum(self.counts.get(n, 0) for n in names), 1)
        best, best_score = names[0], -float("inf")
        for name in names:
            count = self.counts.get(name, 0)
            if count == 0:
                best, best_score = name, float("inf")
                break
            score = self.rewards[name] + self.c * math.sqrt(math.log(total) / count)
            if score > best_score:
                best, best_score = name, score
        d.evaluator = best
        d.source = self.source
        tried = {n: round(self.rewards.get(n, 0.0), 4) for n in names if self.counts.get(n)}
        d.rationale = (f"UCB1 over evaluators picked {best} "
                       f"(mean gain per unit cost so far: {tried or 'nothing tried yet'}); "
                       + d.rationale)
        return d.validated(max_batch, names)


class FixedMethodPlanner:
    """Always spend with the same evaluator. The "pick one method and commit" strategy.

    Every paper on the reading list is, in effect, one of these: brute-force
    docking everything, or ranking everything by retrieval, or co-folding
    everything. They are the right comparison for an orchestration layer,
    because the claim being tested is precisely that no fixed choice is correct
    across targets. If a fixed method matches the agent everywhere, the agent is
    not earning its complexity.
    """

    source = "fixed_method"

    def __init__(self, evaluator: str, schedule=None):
        self.evaluator = evaluator
        self.name = f"always_{evaluator}"
        self._schedule = schedule or HeuristicPlanner()

    def decide(self, s: CampaignState, max_batch: int) -> Decision:
        d = self._schedule.decide(s, max_batch)
        d.evaluator = self.evaluator
        d.source = self.source
        d.rationale = f"fixed policy: always use {self.evaluator}; " + d.rationale
        names = [e.split()[0].rstrip(":") for e in s.evaluator_menu] or None
        return d.validated(max_batch, names)


class CascadePlanner:
    """Triage cheaply, confirm expensively -- the pipeline the papers describe.

    This is what DrugCLIP and AdaptiveFlow actually argue for, written as a
    decision rule: use a near-free method to rank a large slice of the library,
    then spend the confirming budget only on what survived. The cheap pass buys
    no hits by itself; it buys a better-informed surrogate, which is what makes
    the expensive pass land on the right compounds.

    It alternates by design -- a triage round, then one or more confirm rounds --
    and widens the triage as the library gets picked over. The `triage_share`
    argument sets how much of the budget may go to the cheap rung in total,
    which is the one knob that decides whether this behaves like DrugCLIP
    (nearly all triage) or like brute force (none).
    """

    name = "cascade"
    source = "cascade"

    def __init__(self, triage: str = "retrieval", confirm: str = "docking",
                 triage_share: float = 0.15, confirm_every: int = 2):
        self.triage = triage
        self.confirm = confirm
        self.triage_share = triage_share
        self.confirm_every = confirm_every
        self._schedule = HeuristicPlanner()

    def decide(self, s: CampaignState, max_batch: int) -> Decision:
        d = self._schedule.decide(s, max_batch)
        names = [e.split()[0].rstrip(":") for e in s.evaluator_menu]
        d.source = self.source
        if not names:
            return d.validated(max_batch)

        triage = self.triage if self.triage in names else names[0]
        confirm = self.confirm if self.confirm in names else names[-1]
        spent_on_triage = s.cost_by_evaluator.get(triage, 0.0)
        triage_budget_left = (s.cost_total * self.triage_share) - spent_on_triage

        do_triage = (s.round_idx % self.confirm_every == 0) and triage_budget_left > 0
        if do_triage:
            d.evaluator = triage
            d.acquisition = "explore"       # the cheap pass is for information
            d.batch_size = max_batch
            d.diversity = 1.0               # cover the space, do not cluster
            d.rationale = (f"cascade triage: a wide, near-free {triage} pass to inform the "
                           f"surrogate before paying for {confirm}; "
                           f"{triage_budget_left:.1f} cost units of triage budget left")
        else:
            d.evaluator = confirm
            d.diversity = 0.3
            d.rationale = (f"cascade confirm: spending on {confirm}, the cheapest rung that "
                           f"can establish a hit, guided by the triage pass; " + d.rationale)
        return d.validated(max_batch, names)


PLANNERS = {
    "random": RandomPlanner,
    "cascade": CascadePlanner,
    "classical_bo": ClassicalBOPlanner,
    "heuristic": HeuristicPlanner,
    "bandit": BanditPlanner,
    "llm": LLMPlanner,
}


def make_planner(kind: str = "auto", model: str = "claude-sonnet-5-5", **kw):
    """Factory. 'auto' uses the LLM planner if a key or a replay cache is present."""
    if kind == "auto":
        p = LLMPlanner(model=model, **kw)
        return p if p.available else HeuristicPlanner()
    if kind.startswith("always_"):
        return FixedMethodPlanner(kind[len("always_"):])
    if kind not in PLANNERS:
        raise ValueError(
            f"unknown planner {kind!r}; choose from {sorted(PLANNERS)}, "
            "'always_<evaluator>' or 'auto'"
        )
    if kind == "llm":
        return LLMPlanner(model=model, **kw)
    return PLANNERS[kind]()


def available_planners() -> list[str]:
    return sorted(PLANNERS)
