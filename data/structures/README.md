# Bundled structures

`1iep_receptor_clean.pdb` — the ABL1 kinase domain from PDB entry **1IEP**,
and `1iep_ligand.sdf` — the co-crystallised **imatinib** ligand.

Both come from the AutoDock Vina basic-docking example
(<https://github.com/ccsb-scripps/AutoDock-Vina>), which is why this particular
system is here: it is a real kinase–inhibitor complex with a binding site that
is already known to be set up correctly for docking, so the browser engine in
`viewer/` can be validated by redocking the crystal ligand and measuring how
far the recovered pose sits from the crystallographic one.

The receptor file has been reduced to heavy atoms, with waters and alternate
conformations dropped, which is all the scoring function uses and keeps the
file small enough to embed in a page.

Note that ABL1 is **not** one of the campaign targets (EGFR, CDK2, c-Met). It
is here because it is the kinase that could be bundled with a validated set-up.
All four are protein kinases with similar ATP sites, so placing campaign
compounds in this pocket is chemically meaningful — but a pose produced there
is not a prediction about any campaign target.
