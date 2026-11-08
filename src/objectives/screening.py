"""Virtual screening as a retrospective benchmark on real molecules.

What is real here, and what is not
----------------------------------
**Real:** the molecules. The actives are published ligand sets for three
targets -- EGFR, CDK2 and c-Met -- that ship inside RDKit, and the decoy bulk is
the NCI library. The structural similarity that drives the ground truth is
computed with Morgan fingerprints exactly as it would be in practice, and the
c-Met set carries measured free energies, which are used directly where present.

**Not real:** the affinity number assigned to every other molecule. There is no
docking engine in this repository and no experimental measurement for the NCI
compounds, so the objective defines an explicit, documented affinity model
instead (below). It produces a landscape with the right *character* -- a few
strong binders, a structure-activity relationship a surrogate can partly learn,
and a long tail of weak ones -- which is what is needed to compare decision
rules. It is not a prediction about any of these compounds, and no number out
of this module should ever be read as one.

The affinity model
------------------
Scored as a pseudo-pKd, higher is better::

    pKd = base
        + w_sim  * max Tanimoto similarity to the target's known actives
        + w_lipo * hydrophobic complementarity (distance from a per-target logP optimum)
        + w_hb   * hydrogen-bonding match to a per-target preference
        + w_size * ligand-efficiency term, penalising mass that buys no affinity
        - penalties for excessive polarity and conformational flexibility

Every weight is per target, so the three targets present genuinely different
landscapes: one rewards recognising a known chemotype, another rewards getting
the physical properties right. A surrogate that works on one is not guaranteed
to work on the next -- which is the premise the orchestration layer exists to
handle, made concrete enough to measure.

Swapping in real docking scores
-------------------------------
`from_score_table` takes a CSV of SMILES and measured or docked values and
builds the identical objective from it. That is the one-call path to running
this whole framework on DOCKSTRING's score matrix, on a lab's own Vina output,
or on in-house assay data -- no other part of the framework changes.
"""

from __future__ import annotations

import numpy as np

from ..objective import PoolObjective, register_objective
from ..pool import (
    build_pool,
    bundled_actives,
    bundled_decoys,
    similarity_to_actives,
    target_description,
)

#: Per-target weights for the documented affinity model. The spread between
#: targets is the point: EGFR rewards chemotype recognition, c-Met rewards
#: physical property match, CDK2 sits between them.
#: Calibrated so a typical decoy lands near pKd 4.5-5.5 and a good binder near
#: 7-9 -- the range experimental affinities actually occupy. This matters:
#: c-Met's measured free energies are converted with the real thermodynamic
#: relation, so if the synthetic scale drifted upward the literature ligands
#: would rank *below* invented decoy scores and the benchmark would be rubbish.
TARGET_PROFILES: dict[str, dict] = {
    "egfr": dict(base=3.85, w_sim=5.0, w_lipo=0.55, w_hb=0.35, w_size=0.45,
                 logp_opt=3.6, hbd_opt=1.0, hba_opt=5.0, tpsa_max=110.0, rot_max=8),
    "cdk2": dict(base=3.70, w_sim=3.8, w_lipo=0.95, w_hb=0.70, w_size=0.55,
                 logp_opt=2.4, hbd_opt=2.0, hba_opt=5.0, tpsa_max=130.0, rot_max=7),
    "cmet": dict(base=3.60, w_sim=2.9, w_lipo=1.30, w_hb=0.85, w_size=0.70,
                 logp_opt=4.1, hbd_opt=1.0, hba_opt=6.0, tpsa_max=100.0, rot_max=10),
}

_DESC = {name: i for i, name in enumerate([
    "mol_weight", "logp", "tpsa", "h_donors", "h_acceptors", "rotatable_bonds",
    "rings", "aromatic_rings", "heavy_atoms", "fraction_csp3", "heteroatoms",
    "formal_charge", "molar_refractivity", "ring_fraction", "sp3_nitrogens",
    "max_ring_size",
])}


def affinity_model(descriptors: np.ndarray, similarity: np.ndarray,
                   profile: dict) -> np.ndarray:
    """The documented pseudo-pKd for a whole library at once. See module docstring."""
    d = descriptors
    logp = d[:, _DESC["logp"]]
    tpsa = d[:, _DESC["tpsa"]]
    hbd = d[:, _DESC["h_donors"]]
    hba = d[:, _DESC["h_acceptors"]]
    rot = d[:, _DESC["rotatable_bonds"]]
    heavy = np.maximum(d[:, _DESC["heavy_atoms"]], 1.0)
    arom = d[:, _DESC["aromatic_rings"]]

    # Chemotype recognition: looking like something that already works.
    sim_term = profile["w_sim"] * similarity

    # Hydrophobic complementarity: a window, not a monotone preference.
    lipo_term = profile["w_lipo"] * np.exp(-0.5 * ((logp - profile["logp_opt"]) / 1.6) ** 2)

    # Hydrogen bonding: donors and acceptors near the pocket's preference.
    hb_term = profile["w_hb"] * (
        np.exp(-0.5 * ((hbd - profile["hbd_opt"]) / 1.5) ** 2)
        + np.exp(-0.5 * ((hba - profile["hba_opt"]) / 2.5) ** 2)
    ) / 2.0

    # Ligand efficiency: affinity should be earned per heavy atom, not by bulk.
    size_term = profile["w_size"] * np.exp(-0.5 * ((heavy - 28.0) / 11.0) ** 2)

    # Aromatic rings help up to a point, then stop.
    arom_term = 0.15 * np.minimum(arom, 3.0)

    penalty = (
        0.012 * np.maximum(tpsa - profile["tpsa_max"], 0.0)
        + 0.10 * np.maximum(rot - profile["rot_max"], 0.0)
    )
    return profile["base"] + sim_term + lipo_term + hb_term + size_term + arom_term - penalty


def variant_profile(base_target: str, variant: int) -> dict:
    """A deterministic perturbation of a target's affinity profile.

    Three chemistry targets is far too small a sample to ask whether a method
    generalises -- which is the same sample-size trap that made this project's
    original benchmark report an advantage that was not there. These variants
    reuse the real molecules but tilt the landscape: a different hydrophobic
    optimum, a different balance between recognising a known chemotype and
    getting the physics right.

    They are explicitly **synthetic targets**, not additional proteins, and
    results on them say something about how a decision rule behaves across a
    *range* of landscapes, not about chemistry. The three real targets remain
    the realistic cases.
    """
    import hashlib

    base = dict(TARGET_PROFILES[base_target])
    digest = hashlib.sha256(f"profile:{base_target}:{variant}".encode()).digest()[:8]
    rng = np.random.default_rng(int.from_bytes(digest, "big") % (2**32))
    base["w_sim"] = float(np.clip(base["w_sim"] * rng.uniform(0.55, 1.45), 1.0, 7.0))
    base["w_lipo"] = float(np.clip(base["w_lipo"] * rng.uniform(0.5, 2.2), 0.2, 3.0))
    base["w_hb"] = float(np.clip(base["w_hb"] * rng.uniform(0.5, 2.2), 0.1, 2.5))
    base["logp_opt"] = float(base["logp_opt"] + rng.uniform(-1.3, 1.3))
    base["hba_opt"] = float(base["hba_opt"] + rng.uniform(-1.5, 1.5))
    return base


def parse_target(name: str) -> tuple[str, int | None]:
    """`egfr` -> ('egfr', None); `egfr#3` -> ('egfr', 3)."""
    if "#" in name:
        base, _, idx = name.partition("#")
        return base, int(idx)
    return name, None


def target_variants(base_targets: list[str], n_per_target: int) -> list[str]:
    """Names for a grid of synthetic landscapes built on the real molecule sets."""
    return [f"{t}#{i}" for t in base_targets for i in range(n_per_target)]


class VirtualScreeningObjective(PoolObjective):
    """One target: a library of real molecules and a documented affinity landscape."""

    def __init__(self, target: str = "egfr", n_decoys: int = 4999,
                 features: str = "descriptors", seed: int = 0,
                 use_experimental: bool = True, max_active_fraction: float = 0.02):
        base_target, variant = parse_target(target)
        if base_target not in TARGET_PROFILES:
            raise ValueError(f"unknown target {base_target!r}; available: {sorted(TARGET_PROFILES)}")
        self.target = target
        self.base_target = base_target
        self.variant = variant
        self.profile = (TARGET_PROFILES[base_target] if variant is None
                        else variant_profile(base_target, variant))
        target = base_target

        actives, experimental = bundled_actives(target)
        decoys = bundled_decoys(n_decoys)

        # Real screening libraries are overwhelmingly inactive -- hit rates are
        # fractions of a percent. Keeping every bundled ligand would make the
        # actives a fifth of the library and the task trivially easy, so the
        # active set is capped at a realistic share. Molecules with a measured
        # value are kept first, because real data is the scarce thing.
        cap = max(int(len(decoys) * max_active_fraction), 8)
        if len(actives) > cap:
            rng = np.random.default_rng(seed)
            measured = [s for s in actives if s in experimental]
            rest = [s for s in actives if s not in experimental]
            take = max(cap - len(measured), 0)
            chosen = measured[:cap] + list(rng.choice(rest, size=min(take, len(rest)), replace=False))
            actives = sorted(set(chosen))
        # Deduplicate while preserving the actives, which are the point.
        seen, smiles, is_active = set(), [], []
        for smi, flag in [(s, True) for s in actives] + [(s, False) for s in decoys]:
            if smi in seen:
                continue
            seen.add(smi)
            smiles.append(smi)
            is_active.append(flag)

        pool = build_pool(smiles, features=features)
        # build_pool drops unparseable entries, so realign the active flags.
        kept = {s: i for i, s in enumerate(pool.smiles)}
        active_flags = np.zeros(len(pool), dtype=bool)
        for smi, flag in zip(smiles, is_active):
            if flag and smi in kept:
                active_flags[kept[smi]] = True
        self.known_actives = active_flags

        # Similarity is measured against the actives *excluding each molecule
        # itself*, so an active does not score highly merely for being in the
        # reference set. Without this the task is trivially solvable by lookup.
        sim = similarity_to_actives(pool.smiles, active_flags)

        values = affinity_model(pool.descriptors, sim, self.profile)

        # Where a real measurement exists, it wins over the model.
        self.n_experimental = 0
        if use_experimental and experimental:
            for i, smi in enumerate(pool.smiles):
                if smi in experimental:
                    # Experimental dG (kcal/mol, negative) -> pKd-like, higher better.
                    values[i] = -float(experimental[smi]) / 1.364
                    self.n_experimental += 1

        super().__init__(pool, values, name=f"screen_{self.target}")
        self.description = target_description(base_target)
        self.seed = seed

    # -- reporting ---------------------------------------------------------

    @property
    def value_unit(self) -> str:
        return "pseudo-pKd"

    def hit_rate_of_known_actives(self, k: int) -> float:
        """What share of the true top-k are literature actives rather than decoys?

        A sanity check on the affinity model: if the known ligands for a target
        do not come out near the top, the landscape is not modelling anything
        recognisable and the benchmark is meaningless.
        """
        top = set(int(i) for i in self.true_top_k(k))
        return sum(1 for i in top if self.known_actives[i]) / max(k, 1)


def from_score_table(csv_path: str, smiles_column: str = "smiles",
                     value_column: str = "score", name: str = "custom_screen",
                     features: str = "descriptors",
                     higher_is_better: bool = True) -> PoolObjective:
    """Build a screening objective from a table of real scores.

    This is the adapter for DOCKSTRING's score matrix, a lab's own docking
    output, or assay data: one column of SMILES, one column of values. Docking
    scores are usually more-negative-is-better, so pass
    `higher_is_better=False` and they are flipped.
    """
    import csv as _csv

    smiles: list[str] = []
    values: list[float] = []
    with open(csv_path, newline="") as fh:
        for row in _csv.DictReader(fh):
            try:
                value = float(row[value_column])
            except (TypeError, ValueError, KeyError):
                continue
            smiles.append(row[smiles_column])
            values.append(value if higher_is_better else -value)

    if not smiles:
        raise ValueError(f"no usable rows in {csv_path}")

    pool = build_pool(smiles, features=features)
    kept = {s: v for s, v in zip(smiles, values)}
    aligned = np.array([kept[s] for s in pool.smiles], dtype=float)
    return PoolObjective(pool, aligned, name=name)


@register_objective
class ScreenEGFR(VirtualScreeningObjective):
    name = "screen_egfr"

    def __init__(self, **kw):
        super().__init__(target="egfr", **kw)


@register_objective
class ScreenCDK2(VirtualScreeningObjective):
    name = "screen_cdk2"

    def __init__(self, **kw):
        super().__init__(target="cdk2", **kw)


@register_objective
class ScreenCMet(VirtualScreeningObjective):
    name = "screen_cmet"

    def __init__(self, **kw):
        super().__init__(target="cmet", **kw)
