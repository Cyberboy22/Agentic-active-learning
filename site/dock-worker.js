/* Worker wrapper around the docking engine. Keeps the search off the UI thread
   so the page stays responsive while a pocket is mapped and a ligand placed. */
importScripts("./dock.js");

let MAP = null;
let CRYSTAL = null;
let POCKET = null;

function centroid(lig) {
  const c = [0, 0, 0];
  for (let i = 0; i < lig.nAtoms; i++) {
    c[0] += lig.coords[i * 3];
    c[1] += lig.coords[i * 3 + 1];
    c[2] += lig.coords[i * 3 + 2];
  }
  return c.map((v) => v / lig.nAtoms);
}

self.onmessage = function (e) {
  const m = e.data;
  try {
    if (m.type === "init") {
      const receptor = DockEngine.parsePDB(m.receptor);
      let centre;
      if (m.ligand) {
        CRYSTAL = DockEngine.parseSDF(m.ligand);
        centre = centroid(CRYSTAL);
      } else {
        CRYSTAL = null;
        centre = m.center;
      }
      POCKET = centre;
      MAP = DockEngine.buildGrids(receptor, centre, m.boxSize || 24,
        ["C", "N", "O", "S"], {
          onProgress: (f) => self.postMessage({ type: "progress", fraction: f }),
        });
      self.postMessage({
        type: "ready",
        center: centre,
        cavity: MAP.cavityCount,
        receptorAtoms: receptor.length,
        crystalScore: CRYSTAL ? DockEngine.scorePose(MAP, CRYSTAL.coords, CRYSTAL.elements) : null,
        crystalAtoms: CRYSTAL ? CRYSTAL.nAtoms : null,
        crystalRot: CRYSTAL ? CRYSTAL.rotatableBonds : null,
      });
    } else if (m.type === "dock") {
      if (!MAP) throw new Error("pocket maps are not ready yet");
      const lig = m.useCrystal ? CRYSTAL : DockEngine.parseSDF(m.sdf);
      if (!lig || !lig.nAtoms) throw new Error("could not read that structure");
      const t0 = Date.now();
      const res = DockEngine.dock(MAP, lig, { seed: m.seed });
      self.postMessage({
        type: "result",
        score: res.score,
        normalised: res.normalised,
        pdb: DockEngine.toPDBBlock(res.coords, lig.elements, "LIG"),
        ms: Date.now() - t0,
        atoms: lig.nAtoms,
        rot: lig.rotatableBonds,
        rmsd: m.useCrystal && CRYSTAL ? DockEngine.rmsd(res.coords, CRYSTAL.coords) : null,
        isCrystal: !!m.useCrystal,
        label: m.label || "",
      });
    }
  } catch (err) {
    self.postMessage({ type: "error", message: String(err && err.message ? err.message : err) });
  }
};
