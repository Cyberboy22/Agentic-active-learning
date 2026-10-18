"""The orchestration agent: decides the strategy for each screening round.

This is the "autonomous planning" layer. Given the state of the campaign
(budget spent, best value so far, whether progress has stalled, model fit),
the agent chooses:

  - which surrogate to trust (random_forest vs gaussian_process),
  - which acquisition function to use (ucb / ei / greedy / explore),
  - the exploration strength (kappa),
  - the batch size for this round,

and returns a short, human-readable *rationale* for the choice. That rationale
is what makes the pipeline explainable: every decision comes with its evidence.

Two backends, same output contract (a `Decision`):

  * LLMAgent   - asks Claude to make the call, with the state summarised as
                 text and a strict JSON reply. Requires ANTHROPIC_API_KEY.
  * HeuristicAgent - a transparent rule-based planner with no dependencies,
                 so the repository runs and is testable fully offline.

The LLM backend falls back to the heuristic one on any error, so a reviewer
who clones the repo without an API key still gets a working, deterministic run.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass

ACQ_CHOICES = ["ucb", "ei", "greedy", "explore"]
SURR_CHOICES = ["random_forest", "gaussian_process"]


@dataclass
class CampaignState:
    round_idx: int
    n_rounds: int
    budget_used: int
    budget_total: int
    best_value: float
    last_improvement: float  # improvement in best_value at the previous round
    rounds_since_improvement: int
    n_observations: int


@dataclass
class Decision:
    acquisition: str
    surrogate: str
    kappa: float
    batch_size: int
    rationale: str
    source: str  # "llm" or "heuristic"

    def validated(self, max_batch: int) -> "Decision":
        if self.acquisition not in ACQ_CHOICES:
            self.acquisition = "ucb"
        if self.surrogate not in SURR_CHOICES:
            self.surrogate = "random_forest"
        self.kappa = float(min(max(self.kappa, 0.0), 5.0))
        self.batch_size = int(min(max(self.batch_size, 1), max_batch))
        return self


class HeuristicAgent:
    """Rule-based planner: explore early, exploit late, break plateaus.

    The schedule encodes standard active-learning intuition so the behaviour
    is fully explainable and reproducible without any model call.
    """

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

        # Plateau breaker: if stalled, fire a single exploration burst, then go
        # back to refining -- alternating rather than staying hot, so the search
        # escapes the basin without abandoning the good region it has found.
        if s.rounds_since_improvement >= 2 and s.rounds_since_improvement % 2 == 0:
            acq, kappa = "ucb", 2.5
            why = (
                f"stalled for {s.rounds_since_improvement} rounds: one exploration "
                "burst (raise kappa) to escape the current basin"
            )

        # GP is better in the low-data regime; hand over to RF once data grows.
        surrogate = "gaussian_process" if s.n_observations < 40 else "random_forest"

        batch = max(1, min(max_batch, round((s.budget_total - s.budget_used) / max(s.n_rounds - s.round_idx, 1))))

        rationale = (
            f"{why}; using {surrogate} (n={s.n_observations} observations); "
            f"batch={batch}."
        )
        return Decision(acq, surrogate, kappa, batch, rationale, self.source).validated(max_batch)


class LLMAgent:
    """Claude-backed planner. Falls back to the heuristic agent on any failure."""

    source = "llm"

    def __init__(self, model: str = "claude-opus-4-8", fallback: HeuristicAgent | None = None):
        self.model = model
        self.fallback = fallback or HeuristicAgent()
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
            f"Tools you may choose:\n"
            f"- surrogate: one of {SURR_CHOICES} "
            "(gaussian_process is better with little data; random_forest is robust later)\n"
            f"- acquisition: one of {ACQ_CHOICES} "
            "(ucb explores via mean+kappa*std; ei refines near the best; "
            "greedy pure-exploits; explore maximises uncertainty)\n"
            f"- kappa: float 0..5 (UCB exploration strength)\n"
            f"- batch_size: int 1..{max_batch} (experiments to run this round)\n\n"
            "Pick a strategy appropriate to the phase of the campaign and whether "
            "progress has stalled. Reply with ONLY a JSON object with keys "
            '"acquisition","surrogate","kappa","batch_size","rationale". '
            'The "rationale" is one sentence of evidence-based justification.'
        )

    def decide(self, s: CampaignState, max_batch: int) -> Decision:
        if not self._client:
            return self.fallback.decide(s, max_batch)
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
        except Exception as exc:  # any API/parse error -> deterministic fallback
            d = self.fallback.decide(s, max_batch)
            d.rationale = f"[LLM unavailable: {type(exc).__name__}; used heuristic] " + d.rationale
            d.source = "heuristic(fallback)"
            return d


def make_agent(kind: str = "auto", model: str = "claude-opus-4-8"):
    """Factory. 'auto' uses the LLM if a key is present, else the heuristic."""
    if kind == "heuristic":
        return HeuristicAgent()
    if kind == "llm":
        return LLMAgent(model=model)
    agent = LLMAgent(model=model)
    return agent if agent.available else HeuristicAgent()
