#!/usr/bin/env python3
"""Export one screening campaign as data the 3D viewer can render.

Writes a single JSON file holding, per target: the campaign's shortlist, what
each compound cost to establish, whether it turned out to be a genuine top-k
compound, and a 3D conformer for every molecule so the viewer has something to
dock. Conformers are generated with RDKit's ETKDG and MMFF-minimised -- real
3D chemistry, computed here rather than invented in the browser.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from rdkit import Chem, RDLogger  # noqa: E402
from rdkit.Chem import AllChem, Descriptors  # noqa: E402

from src.human import make_reviewer  # noqa: E402
from src.metrics import json_safe  # noqa: E402
from src.objectives.screening import VirtualScreeningObjective  # noqa: E402
from src.screening_campaign import run_screening_campaign  # noqa: E402

RDLogger.DisableLog("rdApp.*")

#: A reference structure per target, used by the viewer to show the real
#: protein and locate its binding site. Fetched from RCSB by the browser.
REFERENCE_STRUCTURES = {
    "egfr": {"pdb": "1M17", "note": "EGFR kinase domain with erlotinib bound", "ligand": "AQ4"},
    "cdk2": {"pdb": "1AQ1", "note": "CDK2 with staurosporine bound", "ligand": "STU"},
    "cmet": {"pdb": "3LQ8", "note": "c-Met kinase domain with a small-molecule inhibitor", "ligand": None},
}


def conformer_sdf(smiles: str, seed: int = 0) -> str | None:
    """A single MMFF-minimised 3D conformer as an SDF block."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    if AllChem.EmbedMolecule(mol, params) != 0:
        return None
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    except Exception:
        pass
    mol = Chem.RemoveHs(mol)
    return Chem.MolToMolBlock(mol)


def describe(smiles: str) -> dict:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}
    return {
        "mw": round(Descriptors.MolWt(mol), 1),
        "logp": round(Descriptors.MolLogP(mol), 2),
        "hbd": Descriptors.NumHDonors(mol),
        "hba": Descriptors.NumHAcceptors(mol),
        "tpsa": round(Descriptors.TPSA(mol), 1),
        "heavy_atoms": mol.GetNumHeavyAtoms(),
        "rot_bonds": Descriptors.NumRotatableBonds(mol),
    }


def build_target(target: str, budget: float, n_show: int, decoys: int, seed: int) -> dict:
    objective = VirtualScreeningObjective(target, n_decoys=decoys)
    result = run_screening_campaign(
        objective=objective, planner="cascade", budget=budget, n_rounds=18,
        seed=seed, reviewer=make_reviewer("rules"), verbose=False,
    )
    top_set = set(int(i) for i in objective.true_top_k(result.metrics["k"]))

    compounds = []
    for entry in result.top_compounds[:n_show]:
        sdf = conformer_sdf(entry["smiles"], seed=seed)
        if sdf is None:
            continue
        compounds.append({
            **entry,
            "properties": describe(entry["smiles"]),
            "sdf": sdf,
            "rank_true": int(np.where(np.argsort(objective.true_values)[::-1]
                                      == entry["index"])[0][0]) + 1,
        })

    rounds = [{
        "round": r.round_idx, "evaluator": r.evaluator, "n": r.batch_size,
        "cost": r.cost_spent, "cumulative_cost": r.cumulative_cost,
        "recall": r.recall, "vetoed": r.n_vetoed,
        "rationale": r.decision.get("rationale", ""),
    } for r in result.rounds]

    return {
        "target": target,
        "description": objective.description,
        "reference": REFERENCE_STRUCTURES.get(target, {}),
        "library_size": len(objective.pool),
        "known_actives": int(objective.known_actives.sum()),
        "n_experimental": int(getattr(objective, "n_experimental", 0)),
        "metrics": result.metrics,
        "ledger": result.ledger,
        "rounds": rounds,
        "compounds": compounds,
        "overrides": result.overrides[:20],
        "true_top_k_size": len(top_set),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", default=["egfr", "cdk2", "cmet"])
    ap.add_argument("--budget", type=float, default=260.0)
    ap.add_argument("--show", type=int, default=14)
    ap.add_argument("--decoys", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="examples/viewer_data.json")
    args = ap.parse_args()

    data = {"targets": []}
    for t in args.targets:
        print(f"building {t} ...", flush=True)
        data["targets"].append(build_target(t, args.budget, args.show, args.decoys, args.seed))
        last = data["targets"][-1]
        print(f"  {len(last['compounds'])} compounds, recall {last['metrics']['recall_at_k']:.2f}, "
              f"cost {last['metrics']['cost_spent']:.0f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(json_safe(data), allow_nan=False))
    print(f"wrote {out} ({out.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
