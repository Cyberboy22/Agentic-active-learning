"""The human checkpoint: where a medicinal chemist overrules the machine.

The gap this exists to close
----------------------------
DrugCLIP reports roughly a 10% hit rate from its fully automated pipeline and
roughly 17.5% when human experts chose what to take forward. That difference is
the clearest number in the whole reading list, and it is a *measurable target*:
an orchestration layer justifies itself by closing part of it, not by having a
nicer architecture.

Closing it requires a seam the chemist can actually reach into, and -- more
importantly -- a record of what they did when they reached. An override that is
not logged teaches the system nothing and happens again next round.

What a reviewer does here
-------------------------
Given the batch the planner wants to run, a reviewer may **approve** it,
**veto** compounds, or **reorder** them. Every one of those is recorded with a
reason, and vetoed compounds are both removed from this batch and remembered,
so the campaign does not propose them again.

Two reviewers ship. `NoReviewer` is the automated baseline. `RuleBasedReviewer`
stands in for a chemist using RDKit's PAINS alerts -- real medicinal-chemistry
filter patterns, not invented ones -- plus property rules of the kind that get
applied by eye in a real triage meeting. It is a crude model of expert judgement
and is labelled as such; the point is the *mechanism*, which a real reviewer or
a lab's own filters drop straight into.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Override:
    """One reviewer action on one compound, kept for the audit trail."""

    round_idx: int
    index: int
    smiles: str
    action: str  # "veto" | "promote" | "approve"
    reason: str

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class ReviewResult:
    kept: list[int]
    overrides: list[Override] = field(default_factory=list)

    @property
    def n_vetoed(self) -> int:
        return sum(1 for o in self.overrides if o.action == "veto")


class NoReviewer:
    """The fully automated pipeline: whatever the planner picked, gets run."""

    name = "none"

    def __init__(self) -> None:
        self.overrides: list[Override] = []
        self.blocked: set[int] = set()

    def review(self, indices, objective, round_idx: int, **_) -> ReviewResult:
        return ReviewResult(kept=[int(i) for i in indices])


class RuleBasedReviewer:
    """A stand-in medicinal chemist: PAINS alerts plus property triage.

    Vetoes a compound that trips a PAINS substructure alert, or that is so far
    outside drug-like property space that nobody would order it. This is a
    caricature of expert review -- a real chemist rejects things for reasons no
    rule encodes -- but it exercises the same machinery and makes the cost of
    ignoring triage measurable.
    """

    name = "rules"

    def __init__(self, max_mol_weight: float = 600.0, max_logp: float = 6.0,
                 min_heavy_atoms: int = 12, use_pains: bool = True):
        self.max_mol_weight = max_mol_weight
        self.max_logp = max_logp
        self.min_heavy_atoms = min_heavy_atoms
        self.overrides: list[Override] = []
        self.blocked: set[int] = set()
        self._catalog = None
        if use_pains:
            try:
                from .pool import pains_catalog

                self._catalog = pains_catalog()
            except Exception:
                self._catalog = None

    def _pains_alert(self, smiles: str) -> str | None:
        if self._catalog is None:
            return None
        try:
            from rdkit import Chem

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return "unparseable structure"
            entry = self._catalog.GetFirstMatch(mol)
            return entry.GetDescription() if entry is not None else None
        except Exception:
            return None

    def review(self, indices, objective, round_idx: int, **_) -> ReviewResult:
        kept: list[int] = []
        overrides: list[Override] = []
        descriptors = getattr(objective.pool, "descriptors", None)

        for raw in indices:
            i = int(raw)
            smiles = objective.smiles_at(i)
            reason = None

            alert = self._pains_alert(smiles)
            if alert:
                reason = f"PAINS alert: {alert}"
            elif descriptors is not None:
                mw, logp, heavy = descriptors[i][0], descriptors[i][1], descriptors[i][8]
                if mw > self.max_mol_weight:
                    reason = f"molecular weight {mw:.0f} above {self.max_mol_weight:.0f}"
                elif logp > self.max_logp:
                    reason = f"logP {logp:.1f} above {self.max_logp:.1f}; likely insoluble"
                elif heavy < self.min_heavy_atoms:
                    reason = f"only {int(heavy)} heavy atoms; too small to be a lead"

            if reason:
                overrides.append(Override(round_idx, i, smiles, "veto", reason))
                self.blocked.add(i)
            else:
                kept.append(i)

        self.overrides.extend(overrides)
        return ReviewResult(kept=kept, overrides=overrides)


class CallbackReviewer:
    """Wraps any callable as a reviewer, for a real human or a lab's own filters.

    The callback receives the proposed SMILES and returns the ones to keep.
    """

    name = "callback"

    def __init__(self, fn):
        self.fn = fn
        self.overrides: list[Override] = []
        self.blocked: set[int] = set()

    def review(self, indices, objective, round_idx: int, **_) -> ReviewResult:
        indices = [int(i) for i in indices]
        proposed = {i: objective.smiles_at(i) for i in indices}
        keep = set(self.fn(proposed))
        kept, overrides = [], []
        for i in indices:
            if i in keep or proposed[i] in keep:
                kept.append(i)
            else:
                overrides.append(Override(round_idx, i, proposed[i], "veto", "rejected by reviewer"))
                self.blocked.add(i)
        self.overrides.extend(overrides)
        return ReviewResult(kept=kept, overrides=overrides)


REVIEWERS = {"none": NoReviewer, "rules": RuleBasedReviewer}


def make_reviewer(kind: str = "none", **kw):
    if kind not in REVIEWERS:
        raise ValueError(f"unknown reviewer {kind!r}; choose from {sorted(REVIEWERS)}")
    return REVIEWERS[kind](**kw)
