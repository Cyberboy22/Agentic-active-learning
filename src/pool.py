"""Building a candidate *pool*: the finite library a screening campaign selects from.

A box-bounded optimiser invents designs. A screening campaign cannot: the
molecules that exist are the molecules you can order, and the job is to choose
which rows of that fixed table are worth paying to evaluate. This module turns a
list of SMILES into the three things the rest of the framework needs:

* a **feature matrix** -- one row per molecule, so a surrogate can learn on it;
* a **nearest-neighbour index** -- so "refine around the best hit" means
  *analogue search* rather than a Gaussian nudge in coordinate space;
* a **ground-truth column** -- the value the campaign is trying to discover,
  used only for scoring the campaign afterwards, never during it.

Two featurisations ship. `descriptors` is a compact vector of interpretable
physicochemical properties -- small enough that a Gaussian process still works,
and the same axes AdaptiveFlow slices chemical space along. `fingerprint` is a
folded Morgan bit vector, which is what the field actually uses at scale and
which needs the ensemble surrogate rather than a GP.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

# RDKit is imported lazily so the rest of the framework keeps working without it.
_RDKIT_ERROR = (
    "RDKit is required for molecular pools. Install it with `pip install rdkit`, "
    "or use a non-molecular PoolObjective by passing your own feature matrix."
)

DESCRIPTOR_NAMES = [
    "mol_weight", "logp", "tpsa", "h_donors", "h_acceptors", "rotatable_bonds",
    "rings", "aromatic_rings", "heavy_atoms", "fraction_csp3", "heteroatoms",
    "formal_charge", "molar_refractivity", "ring_fraction", "sp3_nitrogens",
    "max_ring_size",
]


def _require_rdkit():
    try:
        from rdkit import Chem, RDLogger
        from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors

        RDLogger.DisableLog("rdApp.*")
        return Chem, Descriptors, Crippen, Lipinski, rdMolDescriptors
    except ImportError as exc:  # pragma: no cover - exercised only without rdkit
        raise ImportError(_RDKIT_ERROR) from exc


def descriptor_vector(smiles: str) -> np.ndarray | None:
    """Physicochemical descriptor vector for one SMILES, or None if unparseable."""
    Chem, Descriptors, Crippen, Lipinski, rdMolDescriptors = _require_rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    ring_info = mol.GetRingInfo()
    ring_sizes = [len(r) for r in ring_info.AtomRings()] or [0]
    heavy = mol.GetNumHeavyAtoms() or 1
    n_ring_atoms = sum(1 for a in mol.GetAtoms() if a.IsInRing())
    sp3_n = sum(
        1 for a in mol.GetAtoms()
        if a.GetSymbol() == "N" and str(a.GetHybridization()) == "SP3"
    )
    return np.array([
        Descriptors.MolWt(mol),
        Crippen.MolLogP(mol),
        rdMolDescriptors.CalcTPSA(mol),
        Lipinski.NumHDonors(mol),
        Lipinski.NumHAcceptors(mol),
        Lipinski.NumRotatableBonds(mol),
        ring_info.NumRings(),
        rdMolDescriptors.CalcNumAromaticRings(mol),
        heavy,
        rdMolDescriptors.CalcFractionCSP3(mol),
        sum(1 for a in mol.GetAtoms() if a.GetSymbol() not in ("C", "H")),
        Chem.GetFormalCharge(mol),
        Crippen.MolMR(mol),
        n_ring_atoms / heavy,
        sp3_n,
        max(ring_sizes),
    ], dtype=float)


def fingerprint_vector(smiles: str, n_bits: int = 1024, radius: int = 2) -> np.ndarray | None:
    """Folded Morgan fingerprint as a float array, or None if unparseable."""
    Chem, *_ = _require_rdkit()
    from rdkit.Chem import rdFingerprintGenerator

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    return np.array(gen.GetFingerprint(mol), dtype=float)


def canonical_smiles(smiles: str) -> str | None:
    Chem, *_ = _require_rdkit()
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol is not None else None


@dataclass
class MoleculePool:
    """A fixed library: SMILES, a feature matrix, and a nearest-neighbour index.

    `features` has one row per molecule in `smiles`. Rows are looked up by an
    exact hash of their bytes, which is what lets the rest of the framework pass
    plain feature vectors around (as it does for box objectives) while this class
    maps them back to the molecule they came from.
    """

    smiles: list[str]
    features: np.ndarray
    feature_names: list[str]
    descriptors: np.ndarray | None = None
    descriptor_names: list[str] = field(default_factory=lambda: list(DESCRIPTOR_NAMES))
    _index: dict[bytes, int] = field(default_factory=dict, repr=False, init=False)

    def __post_init__(self) -> None:
        if len(self.smiles) != len(self.features):
            raise ValueError("smiles and features must have the same length")
        self.features = np.ascontiguousarray(self.features, dtype=float)
        for i, row in enumerate(self.features):
            self._index.setdefault(self._key(row), i)

    # -- lookup ------------------------------------------------------------

    @staticmethod
    def _key(row: np.ndarray) -> bytes:
        return np.ascontiguousarray(row, dtype=float).tobytes()

    def index_of(self, row: np.ndarray) -> int:
        """Row index of a feature vector. Falls back to nearest row if not exact."""
        key = self._key(row)
        if key in self._index:
            return self._index[key]
        return int(np.argmin(((self.features - np.asarray(row, float)) ** 2).sum(axis=1)))

    def __len__(self) -> int:
        return len(self.smiles)

    @property
    def dim(self) -> int:
        return self.features.shape[1]

    # -- neighbourhoods ----------------------------------------------------

    def neighbours(self, row: np.ndarray, n: int, rng: np.random.Generator,
                   pool_subset: np.ndarray | None = None) -> np.ndarray:
        """Indices of the `n` library members most similar to `row`.

        This is the discrete replacement for a trust region: you cannot nudge a
        molecule by 5% of its range, but you can ask which catalogue entries are
        most like the one that worked -- an analogue search.
        """
        candidates = np.arange(len(self)) if pool_subset is None else np.asarray(pool_subset)
        if len(candidates) == 0:
            return candidates
        dist = ((self.features[candidates] - np.asarray(row, float)) ** 2).sum(axis=1)
        take = min(n, len(candidates))
        # Random tie-breaking keeps the selection reproducible but unbiased.
        order = np.lexsort((rng.random(len(candidates)), dist))
        return candidates[order[:take]]

    def fingerprint_hash(self) -> str:
        """Stable hash of the library, so provenance can prove which pool was used."""
        h = hashlib.sha256()
        h.update(str(len(self.smiles)).encode())
        for s in self.smiles[:5000]:
            h.update(s.encode())
        h.update(np.round(self.features, 6).tobytes())
        return h.hexdigest()[:12]


def build_pool(
    smiles: list[str],
    features: str = "descriptors",
    n_bits: int = 1024,
    standardise: bool = True,
) -> MoleculePool:
    """Featurise a list of SMILES into a `MoleculePool`, dropping unparseable entries."""
    kept_smiles: list[str] = []
    rows: list[np.ndarray] = []
    desc_rows: list[np.ndarray] = []

    for smi in smiles:
        desc = descriptor_vector(smi)
        if desc is None or not np.all(np.isfinite(desc)):
            continue
        if features == "descriptors":
            row = desc
        elif features == "fingerprint":
            fp = fingerprint_vector(smi, n_bits=n_bits)
            if fp is None:
                continue
            row = fp
        else:
            raise ValueError(f"unknown featurisation {features!r}")
        kept_smiles.append(smi)
        rows.append(row)
        desc_rows.append(desc)

    if not rows:
        raise ValueError("no parseable molecules in the input")

    X = np.vstack(rows)
    D = np.vstack(desc_rows)
    names = list(DESCRIPTOR_NAMES) if features == "descriptors" else [f"bit_{i}" for i in range(X.shape[1])]

    if standardise and features == "descriptors":
        mu, sigma = X.mean(axis=0), X.std(axis=0)
        sigma[sigma < 1e-9] = 1.0
        X = (X - mu) / sigma

    return MoleculePool(
        smiles=kept_smiles, features=X, feature_names=names,
        descriptors=D, descriptor_names=list(DESCRIPTOR_NAMES),
    )


# --------------------------------------------------------------------------
# Bundled chemistry: real molecules that ship with RDKit, so the screening
# benchmark runs offline and on genuine medicinal chemistry rather than
# invented strings.
# --------------------------------------------------------------------------

#: Real ligand sets distributed inside RDKit's Contrib tree, keyed by target.
_ACTIVE_SETS = {
    "egfr": ("Contrib/PBF/testData/egfr.sdf", "EGFR kinase inhibitors"),
    "cdk2": ("Contrib/Fastcluster/testdata/cdk2.sdf", "CDK2 inhibitors"),
    "cmet": ("Contrib/FreeWilson/data/cmet_ligands.sdf", "c-Met inhibitors (with experimental dG)"),
}
_DECOY_FILE = "Data/NCI/first_5K.smi"
_PAINS_FILE = "Data/Pains/wehi_pains.csv"


def _rdkit_base() -> str:
    import os

    from rdkit import RDConfig

    return os.path.dirname(RDConfig.RDDataDir)


def available_targets() -> list[str]:
    return sorted(_ACTIVE_SETS)


def target_description(target: str) -> str:
    return _ACTIVE_SETS[target][1]


def bundled_actives(target: str) -> tuple[list[str], dict[str, float]]:
    """Real known ligands for `target`, plus any experimental values they carry.

    Returns (smiles, {smiles: experimental_value}). Only the c-Met set ships with
    measured free energies; for the others the dictionary is empty and the
    screening objective falls back to its structural affinity model.
    """
    import os

    Chem, *_ = _require_rdkit()
    if target not in _ACTIVE_SETS:
        raise ValueError(f"unknown target {target!r}; available: {available_targets()}")
    path = os.path.join(_rdkit_base(), _ACTIVE_SETS[target][0])
    smiles: list[str] = []
    experimental: dict[str, float] = {}
    for mol in Chem.SDMolSupplier(path):
        if mol is None:
            continue
        smi = Chem.MolToSmiles(mol)
        smiles.append(smi)
        props = mol.GetPropsAsDict()
        for key in ("r_exp_dg", "exp_dg", "dG"):
            if key in props:
                try:
                    experimental[smi] = float(props[key])
                except (TypeError, ValueError):
                    pass
                break
    return smiles, experimental


def bundled_decoys(limit: int | None = None) -> list[str]:
    """Real, structurally diverse molecules used as the screening library's bulk."""
    import os

    path = os.path.join(_rdkit_base(), _DECOY_FILE)
    with open(path) as fh:
        smiles = [line.split()[0] for line in fh if line.strip()]
    return smiles[:limit] if limit else smiles


def pains_catalog():
    """RDKit's PAINS filter catalog -- real medicinal-chemistry alert patterns.

    Used by the human-override channel to model the kind of veto a medicinal
    chemist applies to an automated pipeline's suggestions.
    """
    from rdkit.Chem import FilterCatalog

    params = FilterCatalog.FilterCatalogParams()
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
    return FilterCatalog.FilterCatalog(params)


def tanimoto_to_set(smiles: list[str], reference: list[str], n_bits: int = 2048,
                    radius: int = 2) -> np.ndarray:
    """Max Tanimoto similarity of each molecule to any molecule in `reference`.

    This is the standard "how like a known active is it?" measure, and it is the
    backbone of the screening objective's ground truth.
    """
    Chem, *_ = _require_rdkit()
    from rdkit import DataStructs
    from rdkit.Chem import rdFingerprintGenerator

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)

    def fps(items: list[str]):
        out = []
        for smi in items:
            mol = Chem.MolFromSmiles(smi)
            out.append(gen.GetFingerprint(mol) if mol is not None else None)
        return out

    query_fps, ref_fps = fps(smiles), [f for f in fps(reference) if f is not None]
    if not ref_fps:
        return np.zeros(len(smiles))
    out = np.zeros(len(smiles))
    for i, fp in enumerate(query_fps):
        if fp is None:
            continue
        out[i] = max(DataStructs.BulkTanimotoSimilarity(fp, ref_fps))
    return out


def similarity_to_actives(pool_smiles: list[str], active_flags: np.ndarray,
                          n_bits: int = 2048, radius: int = 2) -> np.ndarray:
    """Max Tanimoto of every pool member to the actives, excluding itself.

    Computed in one pass with cached fingerprints. The leave-one-out rule
    matters: without it an active scores 1.0 for being identical to itself, the
    ground truth becomes a lookup of the answer key, and any surrogate "solves"
    the benchmark without learning chemistry.
    """
    Chem, *_ = _require_rdkit()
    from rdkit import DataStructs
    from rdkit.Chem import rdFingerprintGenerator

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fps = []
    for smi in pool_smiles:
        mol = Chem.MolFromSmiles(smi)
        fps.append(gen.GetFingerprint(mol) if mol is not None else None)

    active_idx = [int(i) for i in np.flatnonzero(active_flags)]
    active_fps = [fps[i] for i in active_idx if fps[i] is not None]
    active_pos = {i: j for j, i in enumerate(i for i in active_idx if fps[i] is not None)}
    if not active_fps:
        return np.zeros(len(pool_smiles))

    out = np.zeros(len(pool_smiles))
    for i, fp in enumerate(fps):
        if fp is None:
            continue
        sims = DataStructs.BulkTanimotoSimilarity(fp, active_fps)
        if i in active_pos:
            sims[active_pos[i]] = -1.0  # exclude self
        out[i] = max(sims) if sims else 0.0
    return np.clip(out, 0.0, 1.0)
