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
  llm          Claude reads the campaign state and chooses the strategy itself

`llm` falls back to `heuristic` whenever no API key is set or a call fails, so a
run never breaks and the repository is fully testable offline.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass

ACQ_CHOICES = ["ucb", "ei", "greedy", "explore"]
SURR_CHOICES = ["random_forest", "gaussian_process"]


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


@dataclass
class Decision:
    acquisition: str
    surrogate: str
    kappa: float
    batch_size: int
    rationale: str
    source: str

    def validated(self, max_batch: int) -> "Decision":
        """Clamp a decision into the legal range -- the guard rail around the LLM."""
        if self.acquisition not in ACQ_CHOICES + ["random"]:
            self.acquisition = "ucb"
        if self.surrogate not in SURR_CHOICES:
            self.surrogate = "random_forest"
        self.kappa = float(min(max(self.kappa, 0.0), 5.0))
        self.batch_size = int(min(max(self.batch_size, 1), max_batch))
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

    The model is given the state, the menu of tools, and a strict JSON contract.
    Its answer is clamped by `Decision.validated`, so a bad reply degrades to a
    legal decision rather than breaking the run; any API or parsing failure falls
    back to the heuristic planner.
    """

    name = "llm"
    source = "llm"
    #: Planning is a small structured decision, so a mid-tier model is the
    #: sensible default; pass `model=` to use another.

    def __init__(self, model: str = "claude-sonnet-5-5", fallback=None):
        self.model = model
        self.fallback = fallback or HeuristicPlanner()
        self._client = None
        try:
            import anthropic  # noqa: F401

            if os.environ.get("ANTHROPIC_API_KEY"):
                self._client = anthropic.Anthropic()
        except Exception:
            self._client = None

    @property
    def available(self) -> bool:
        return self._client is not None

    def _prompt(self, s: CampaignState, max_batch: int) -> str:
        return (
            "You are the planner of an active-learning campaign that optimises an "
            "expensive experiment by choosing which candidates to measure next.\n\n"
            f"State:\n{json.dumps(asdict(s), indent=2)}\n\n"
            "Tools you may choose:\n"
            f"- surrogate: one of {SURR_CHOICES} "
            "(gaussian_process is better with little data; random_forest is robust later)\n"
            f"- acquisition: one of {ACQ_CHOICES} "
            "(ucb explores via mean+kappa*std; ei refines near the best; "
            "greedy pure-exploits; explore maximises uncertainty)\n"
            "- kappa: float 0..5 (UCB exploration strength)\n"
            f"- batch_size: int 1..{max_batch} (experiments to run this round)\n\n"
            "Pick a strategy appropriate to the phase of the campaign and whether "
            "progress has stalled. Reply with ONLY a JSON object with keys "
            '"acquisition","surrogate","kappa","batch_size","rationale". '
            'The "rationale" is one sentence of evidence-based justification.'
        )

    def decide(self, s: CampaignState, max_batch: int) -> Decision:
        if not self._client:
            d = self.fallback.decide(s, max_batch)
            d.source = "heuristic(no-api-key)"
            return d
        try:
            msg = self._client.messages.create(
                model=self.model,
                max_tokens=400,
                messages=[{"role": "user", "content": self._prompt(s, max_batch)}],
            )
            text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
            start, end = text.find("{"), text.rfind("}") + 1
            data = json.loads(text[start:end])
            return Decision(
                acquisition=str(data["acquisition"]),
                surrogate=str(data["surrogate"]),
                kappa=float(data["kappa"]),
                batch_size=int(data["batch_size"]),
                rationale=str(data.get("rationale", "")).strip() or "(no rationale given)",
                source=self.source,
            ).validated(max_batch)
        except Exception as exc:
            d = self.fallback.decide(s, max_batch)
            d.rationale = f"[LLM unavailable: {type(exc).__name__}; used heuristic] " + d.rationale
            d.source = "heuristic(fallback)"
            return d


PLANNERS = {
    "random": RandomPlanner,
    "classical_bo": ClassicalBOPlanner,
    "heuristic": HeuristicPlanner,
    "llm": LLMPlanner,
}


def make_planner(kind: str = "auto", model: str = "claude-sonnet-5-5"):
    """Factory. 'auto' uses the LLM planner if a key is present, else heuristic."""
    if kind == "auto":
        p = LLMPlanner(model=model)
        return p if p.available else HeuristicPlanner()
    if kind not in PLANNERS:
        raise ValueError(f"unknown planner {kind!r}; choose from {sorted(PLANNERS)} or 'auto'")
    return LLMPlanner(model=model) if kind == "llm" else PLANNERS[kind]()


def available_planners() -> list[str]:
    return sorted(PLANNERS)
