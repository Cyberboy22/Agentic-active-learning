# Third-party notices

This project bundles or depends on the following. Each is used under its own
licence, which is **not** changed by this project's licence.

## Bundled in the web app (`site/`)

**3Dmol.js 2.5.5** — BSD-3-Clause.
Copyright (c) 2014, University of Pittsburgh and contributors.
Shipped at `site/vendor/3Dmol-min.js`; full licence text in
`site/vendor/3Dmol-LICENSE.txt`. BSD-3-Clause requires that this copyright
notice and licence be retained, which is why both are included rather than the
script alone. Verified against npm's published integrity hash
`sha512-kqNHouGqq3YfW58174tdERvm0XYTmP0tavQKOqIw1ouc2OJ7epkXEFrtEkVXV0clBZT2Ze2xHRC/qxX0u0qCdw==`.

## Bundled structural data

**ABL1 kinase (PDB 1IEP) receptor and the imatinib ligand**, at
`data/structures/` and `site/data/`. Taken from the AutoDock Vina example data
(<https://github.com/ccsb-scripps/AutoDock-Vina>, Apache-2.0). The underlying
structural data is from the RCSB Protein Data Bank, which places its archive in
the public domain (CC0); please still cite the original structure paper if you
publish anything based on it.

Structures for EGFR (1M17), CDK2 (1AQ1) and c-Met (3LQ8) are **not** bundled —
the web app fetches them from RCSB at runtime.

## Python dependencies

| package | licence | used for |
|---|---|---|
| RDKit | BSD-3-Clause | molecules, fingerprints, conformers, PAINS alerts |
| NumPy | BSD-3-Clause | arrays and numerics |
| SciPy | BSD-3-Clause | Wilcoxon signed-rank test |
| scikit-learn | BSD-3-Clause | surrogate models |
| matplotlib | PSF-based (BSD-compatible) | plots |
| anthropic (optional) | MIT | the Claude-backed planner |

## Molecule sets

The active sets (EGFR, CDK2, c-Met) and the NCI decoy library are distributed
inside RDKit's own source tree and are used here under RDKit's BSD-3-Clause
licence. They originate from published medicinal-chemistry work and the NCI
open database; cite the primary sources if you publish results that rely on
them.
