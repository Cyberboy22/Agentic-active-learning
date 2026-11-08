/*
 * A small rigid-body docking engine.
 *
 * This is a real calculation, not an animation: an empirical scoring function
 * evaluated on precomputed interaction grids, searched by simulated annealing
 * over the ligand's position and orientation. It is deliberately modest and
 * the limits are stated rather than hidden:
 *
 *   - The ligand is RIGID. Its internal conformation is whatever was handed in,
 *     so torsional flexibility -- which is most of what makes docking hard -- is
 *     not searched. Real engines sample it.
 *   - The receptor is RIGID too, and desolvation and electrostatics are absent.
 *   - Scores are in arbitrary units on the same scale as the Vina-style terms
 *     they are built from. They are NOT binding free energies in kcal/mol and
 *     must not be read as affinity predictions.
 *
 * What it is good for is showing, concretely and interactively, what "dock a
 * molecule into a pocket and score the fit" actually means -- and it can be
 * checked: redocking a crystal ligand should return it close to where
 * crystallography found it, which is the standard sanity test and is wired up
 * as a button rather than asserted in prose.
 *
 * Functional terms follow the published Vina form (two gaussians, a repulsion
 * term, a hydrophobic term and a hydrogen-bond term over surface distances).
 */
(function (root) {
  "use strict";

  // Van der Waals radii used for surface distances, in angstroms.
  const RADII = { C: 1.9, N: 1.8, O: 1.7, S: 2.0, P: 2.1, F: 1.5, CL: 1.8, BR: 2.0, I: 2.2 };
  const HYDROPHOBIC = new Set(["C", "F", "CL", "BR", "I"]);
  const POLAR = new Set(["N", "O"]);

  // Weights of the empirical terms.
  const W = { gauss1: -0.0356, gauss2: -0.00516, repulsion: 0.840,
              hydrophobic: -0.0351, hbond: -0.587 };
  const CUTOFF = 8.0;          // angstroms, beyond which pairs are ignored
  const SPACING = 0.5;          // grid spacing (0.5 A keeps the build near a second)

  function element(sym) {
    const e = (sym || "C").toUpperCase();
    return RADII[e] !== undefined ? e : "C";
  }

  /** Pair energy between a ligand atom type and a protein atom, by surface distance. */
  function pairEnergy(d, ligEl, protEl) {
    let e = 0;
    e += W.gauss1 * Math.exp(-Math.pow(d / 0.5, 2));
    e += W.gauss2 * Math.exp(-Math.pow((d - 3.0) / 2.0, 2));
    if (d < 0) e += W.repulsion * d * d;
    if (HYDROPHOBIC.has(ligEl) && HYDROPHOBIC.has(protEl)) {
      let h = 0;
      if (d < 0.5) h = 1; else if (d < 1.5) h = 1.5 - d;
      e += W.hydrophobic * h;
    }
    if (POLAR.has(ligEl) && POLAR.has(protEl)) {
      let hb = 0;
      if (d < -0.7) hb = 1; else if (d < 0) hb = d / -0.7;
      e += W.hbond * hb;
    }
    return e;
  }

  /**
   * Precompute one energy grid per ligand element type over the search box.
   * This is what makes the search affordable: scoring a pose then costs one
   * trilinear interpolation per atom instead of a sweep over the receptor.
   */
  function buildGrids(receptor, center, boxSize, types, options) {
    const half = boxSize / 2;
    const n = Math.max(2, Math.round(boxSize / SPACING) + 1);
    const origin = [center[0] - half, center[1] - half, center[2] - half];

    // Keep only receptor atoms that can reach the box.
    const reach = half + CUTOFF + 2.5;
    const near = [];
    for (let i = 0; i < receptor.length; i++) {
      const a = receptor[i];
      if (Math.abs(a.x - center[0]) < reach &&
          Math.abs(a.y - center[1]) < reach &&
          Math.abs(a.z - center[2]) < reach) near.push(a);
    }

    // Bucket those atoms so each grid point only visits plausible neighbours.
    const cell = CUTOFF;
    const buckets = new Map();
    const key = (i, j, k) => i + "," + j + "," + k;
    for (const a of near) {
      const bk = key(Math.floor(a.x / cell), Math.floor(a.y / cell), Math.floor(a.z / cell));
      if (!buckets.has(bk)) buckets.set(bk, []);
      buckets.get(bk).push(a);
    }

    // Flatten to typed arrays and hoist everything out of the inner loop: this
    // runs once per pocket but it is tens of millions of pair evaluations, and
    // a per-point object allocation is the difference between one second and
    // ten.
    const nt = types.length;
    const grids = {};
    const flat = [];
    for (const t of types) { grids[t] = new Float32Array(n * n * n); flat.push(grids[t]); }
    const typeRadius = types.map((t) => RADII[t]);
    const typeHydro = types.map((t) => HYDROPHOBIC.has(t));
    const typePolar = types.map((t) => POLAR.has(t));

    const ax = new Float64Array(near.length), ay = new Float64Array(near.length);
    const az = new Float64Array(near.length), ar = new Float64Array(near.length);
    const ahydro = new Uint8Array(near.length), apolar = new Uint8Array(near.length);
    for (let i = 0; i < near.length; i++) {
      ax[i] = near[i].x; ay[i] = near[i].y; az[i] = near[i].z;
      ar[i] = RADII[near[i].el];
      ahydro[i] = HYDROPHOBIC.has(near[i].el) ? 1 : 0;
      apolar[i] = POLAR.has(near[i].el) ? 1 : 0;
    }
    const bucketIdx = new Map();
    for (const [bkey, list] of buckets) {
      bucketIdx.set(bkey, Int32Array.from(list.map((a) => near.indexOf(a))));
    }

    const acc = new Float64Array(nt);
    const cutoff2 = CUTOFF * CUTOFF;
    for (let ix = 0; ix < n; ix++) {
      const px = origin[0] + ix * SPACING;
      const bi = Math.floor(px / cell);
      for (let iy = 0; iy < n; iy++) {
        const py = origin[1] + iy * SPACING;
        const bj = Math.floor(py / cell);
        for (let iz = 0; iz < n; iz++) {
          const pz = origin[2] + iz * SPACING;
          const bk = Math.floor(pz / cell);
          acc.fill(0);
          for (let di = -1; di <= 1; di++)
            for (let dj = -1; dj <= 1; dj++)
              for (let dk = -1; dk <= 1; dk++) {
                const list = bucketIdx.get(key(bi + di, bj + dj, bk + dk));
                if (!list) continue;
                for (let li = 0; li < list.length; li++) {
                  const a = list[li];
                  const dx = px - ax[a], dy = py - ay[a], dz = pz - az[a];
                  const r2 = dx * dx + dy * dy + dz * dz;
                  if (r2 > cutoff2) continue;
                  const r = Math.sqrt(r2);
                  for (let t = 0; t < nt; t++) {
                    const d = r - (typeRadius[t] + ar[a]);
                    let e = W.gauss1 * Math.exp(-(d / 0.5) * (d / 0.5));
                    const g2 = (d - 3.0) / 2.0;
                    e += W.gauss2 * Math.exp(-g2 * g2);
                    if (d < 0) e += W.repulsion * d * d;
                    if (typeHydro[t] && ahydro[a]) {
                      if (d < 0.5) e += W.hydrophobic;
                      else if (d < 1.5) e += W.hydrophobic * (1.5 - d);
                    }
                    if (typePolar[t] && apolar[a]) {
                      if (d < -0.7) e += W.hbond;
                      else if (d < 0) e += W.hbond * (d / -0.7);
                    }
                    acc[t] += e;
                  }
                }
              }
          const idx = (ix * n + iy) * n + iz;
          for (let t = 0; t < nt; t++) flat[t][idx] = acc[t];
        }
      }
      if (options && options.onProgress && ix % 8 === 0) options.onProgress(ix / n);
    }
    // Candidate centroid positions: grid points that are buried enough to be
    // inside a cavity but not already occupied by protein. Sampling ligand
    // placements from these instead of uniformly over the box is the single
    // biggest factor in whether the search finds the pocket at all -- most of
    // a 24 angstrom box is either solvent or solid protein.
    const probe = grids[types[0]];
    const cavity = [];
    for (let ix = 1; ix < n - 1; ix++)
      for (let iy = 1; iy < n - 1; iy++)
        for (let iz = 1; iz < n - 1; iz++) {
          const v = probe[(ix * n + iy) * n + iz];
          if (v < -0.08 && v > -3.0) {
            cavity.push(origin[0] + ix * SPACING, origin[1] + iy * SPACING,
                        origin[2] + iz * SPACING);
          }
        }

    return { grids, origin, n, spacing: SPACING, boxSize, center,
             cavity: Float32Array.from(cavity), cavityCount: cavity.length / 3 };
  }

  /** Trilinear interpolation, with a steep penalty for leaving the box. */
  function sample(map, t, x, y, z) {
    const { grids, origin, n } = map;
    const g = grids[t];
    const fx = (x - origin[0]) / map.spacing;
    const fy = (y - origin[1]) / map.spacing;
    const fz = (z - origin[2]) / map.spacing;
    if (fx < 0 || fy < 0 || fz < 0 || fx > n - 1 || fy > n - 1 || fz > n - 1) return 5.0;
    const i0 = Math.floor(fx), j0 = Math.floor(fy), k0 = Math.floor(fz);
    const i1 = Math.min(i0 + 1, n - 1), j1 = Math.min(j0 + 1, n - 1), k1 = Math.min(k0 + 1, n - 1);
    const tx = fx - i0, ty = fy - j0, tz = fz - k0;
    const at = (i, j, k) => g[(i * n + j) * n + k];
    const c00 = at(i0, j0, k0) * (1 - tx) + at(i1, j0, k0) * tx;
    const c01 = at(i0, j0, k1) * (1 - tx) + at(i1, j0, k1) * tx;
    const c10 = at(i0, j1, k0) * (1 - tx) + at(i1, j1, k0) * tx;
    const c11 = at(i0, j1, k1) * (1 - tx) + at(i1, j1, k1) * tx;
    const c0 = c00 * (1 - ty) + c10 * ty;
    const c1 = c01 * (1 - ty) + c11 * ty;
    return c0 * (1 - tz) + c1 * tz;
  }

  function scorePose(map, coords, elements) {
    let total = 0;
    for (let i = 0; i < elements.length; i++) {
      total += sample(map, elements[i], coords[i * 3], coords[i * 3 + 1], coords[i * 3 + 2]);
    }
    return total;
  }

  // -- quaternion helpers -------------------------------------------------

  function randomQuaternion(rng) {
    const u1 = rng(), u2 = rng(), u3 = rng();
    const s1 = Math.sqrt(1 - u1), s2 = Math.sqrt(u1);
    return [s1 * Math.sin(2 * Math.PI * u2), s1 * Math.cos(2 * Math.PI * u2),
            s2 * Math.sin(2 * Math.PI * u3), s2 * Math.cos(2 * Math.PI * u3)];
  }

  function perturbQuaternion(q, amount, rng) {
    const dq = [q[0] + amount * (rng() - 0.5), q[1] + amount * (rng() - 0.5),
                q[2] + amount * (rng() - 0.5), q[3] + amount * (rng() - 0.5)];
    const norm = Math.hypot(dq[0], dq[1], dq[2], dq[3]) || 1;
    return dq.map((v) => v / norm);
  }

  function quaternionMatrix(q) {
    const [x, y, z, w] = q;
    return [
      1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
      2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
      2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
    ];
  }

  /** Apply rotation about the ligand centroid, then translation. */
  function transform(local, q, t, out) {
    const m = quaternionMatrix(q);
    for (let i = 0; i < local.length / 3; i++) {
      const x = local[i * 3], y = local[i * 3 + 1], z = local[i * 3 + 2];
      out[i * 3] = m[0] * x + m[1] * y + m[2] * z + t[0];
      out[i * 3 + 1] = m[3] * x + m[4] * y + m[5] * z + t[1];
      out[i * 3 + 2] = m[6] * x + m[7] * y + m[8] * z + t[2];
    }
    return out;
  }

  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  /**
   * Search for the best placement of a rigid ligand in the pocket.
   *
   * The energy landscape here is extremely sharp -- shifting a 37-atom ligand
   * by a single angstrom inside a closed pocket swings the score by nearly a
   * hundred units, because atoms start overlapping. Plain annealing from random
   * starts therefore almost never finds the narrow funnel; it looks like it
   * works only if a lucky seed is chosen, which is exactly the kind of result
   * worth distrusting.
   *
   * So the search runs in two stages, which is roughly what production docking
   * engines do:
   *
   *   1. **Coarse scan.** Score a large number of random positions and
   *      orientations cheaply, and keep the most promising handful. This is
   *      about finding the funnel, not the minimum.
   *   2. **Refinement.** From each surviving start, anneal with shrinking moves
   *      and finish with a greedy coordinate descent over the seven degrees of
   *      freedom, which is what actually lands the pose.
   */
  function dock(map, ligand, options) {
    const opts = Object.assign(
      { coarse: 120000, keep: 40, steps: 2500, seed: 1, startTemp: 2.0, endTemp: 0.05,
        maxShift: 1.2, refineRounds: 80 },
      options || {}
    );
    const rng = mulberry32(opts.seed);
    const nAtoms = ligand.elements.length;
    const els = ligand.elements;

    // Work in a centroid-centred frame so rotation is about the ligand's centre.
    const centroid = [0, 0, 0];
    for (let i = 0; i < nAtoms; i++) {
      centroid[0] += ligand.coords[i * 3];
      centroid[1] += ligand.coords[i * 3 + 1];
      centroid[2] += ligand.coords[i * 3 + 2];
    }
    centroid.forEach((_, i) => (centroid[i] /= nAtoms));
    const local = new Float32Array(nAtoms * 3);
    for (let i = 0; i < nAtoms; i++) {
      local[i * 3] = ligand.coords[i * 3] - centroid[0];
      local[i * 3 + 1] = ligand.coords[i * 3 + 1] - centroid[1];
      local[i * 3 + 2] = ligand.coords[i * 3 + 2] - centroid[2];
    }

    const work = new Float32Array(nAtoms * 3);
    const half = map.boxSize / 2 - 2;
    const evaluate = (q, t) => scorePose(map, transform(local, q, t, work), els);

    // -- stage 1: coarse scan, biased to cavity space ---------------------
    const cav = map.cavity, nCav = map.cavityCount || 0;
    const sampleCentre = () => {
      if (nCav > 0) {
        const i = Math.floor(rng() * nCav) * 3;
        return [cav[i] + (rng() - 0.5) * map.spacing,
                cav[i + 1] + (rng() - 0.5) * map.spacing,
                cav[i + 2] + (rng() - 0.5) * map.spacing];
      }
      return [map.center[0] + (rng() - 0.5) * 2 * half,
              map.center[1] + (rng() - 0.5) * 2 * half,
              map.center[2] + (rng() - 0.5) * 2 * half];
    };

    const starts = [];
    for (let i = 0; i < opts.coarse; i++) {
      const q = randomQuaternion(rng);
      const t = sampleCentre();
      starts.push({ score: evaluate(q, t), q, t });
    }
    starts.sort((a, b) => a.score - b.score);

    // Keep starts that are spread out, not twenty variations of one placement.
    const seeds = [];
    for (const cand of starts) {
      if (seeds.length >= opts.keep) break;
      let tooClose = false;
      for (const kept of seeds) {
        const dx = cand.t[0] - kept.t[0], dy = cand.t[1] - kept.t[1], dz = cand.t[2] - kept.t[2];
        if (dx * dx + dy * dy + dz * dz < 1.5 * 1.5) { tooClose = true; break; }
      }
      if (!tooClose) seeds.push(cand);
    }

    // -- stage 2: anneal, then descend ------------------------------------
    let best = { score: Infinity, coords: null, trace: [] };
    for (const seed of seeds) {
      let q = seed.q.slice(), t = seed.t.slice();
      let current = seed.score;
      let localBest = { score: current, q: q.slice(), t: t.slice() };
      const trace = [current];

      for (let s = 0; s < opts.steps; s++) {
        const frac = s / opts.steps;
        const temp = opts.startTemp * Math.pow(opts.endTemp / opts.startTemp, frac);
        const shift = opts.maxShift * (1 - 0.8 * frac);
        const nq = perturbQuaternion(q, 0.35 * (1 - 0.7 * frac), rng);
        const nt = [t[0] + (rng() - 0.5) * shift,
                    t[1] + (rng() - 0.5) * shift,
                    t[2] + (rng() - 0.5) * shift];
        const candidate = evaluate(nq, nt);
        if (candidate < current || rng() < Math.exp((current - candidate) / temp)) {
          current = candidate; q = nq; t = nt;
          if (candidate < localBest.score) localBest = { score: candidate, q: q.slice(), t: t.slice() };
        }
        if (s % 20 === 0) trace.push(localBest.score);
      }

      // Greedy descent: shrink the step until nothing improves.
      let bq = localBest.q.slice(), bt = localBest.t.slice(), bs = localBest.score;
      let step = 0.4, rot = 0.12;
      for (let r = 0; r < opts.refineRounds; r++) {
        let improved = false;
        for (let axis = 0; axis < 3; axis++) {
          for (const dir of [1, -1]) {
            const nt = bt.slice(); nt[axis] += dir * step;
            const sc = evaluate(bq, nt);
            if (sc < bs) { bs = sc; bt = nt; improved = true; }
          }
        }
        for (let k = 0; k < 4; k++) {
          for (const dir of [1, -1]) {
            const nq = bq.slice(); nq[k] += dir * rot;
            const norm = Math.hypot(nq[0], nq[1], nq[2], nq[3]) || 1;
            for (let j = 0; j < 4; j++) nq[j] /= norm;
            const sc = evaluate(nq, bt);
            if (sc < bs) { bs = sc; bq = nq; improved = true; }
          }
        }
        if (!improved) { step *= 0.5; rot *= 0.5; if (step < 0.01) break; }
      }
      trace.push(bs);

      if (bs < best.score) {
        const coords = new Float32Array(nAtoms * 3);
        transform(local, bq, bt, coords);
        best = { score: bs, coords, trace };
      }
    }

    // Vina normalises the summed terms by a torsional penalty. With a rigid
    // ligand this only rescales, but it keeps the number on a comparable scale
    // to published scores instead of silently on a different one.
    const rotB = ligand.rotatableBonds || 0;
    best.normalised = best.score / (1 + 0.0585 * rotB);
    return best;
  }

  function rmsd(a, b) {
    let sum = 0;
    const n = a.length / 3;
    for (let i = 0; i < n; i++) {
      const dx = a[i * 3] - b[i * 3];
      const dy = a[i * 3 + 1] - b[i * 3 + 1];
      const dz = a[i * 3 + 2] - b[i * 3 + 2];
      sum += dx * dx + dy * dy + dz * dz;
    }
    return Math.sqrt(sum / n);
  }

  // -- parsers ------------------------------------------------------------

  function parsePDB(text) {
    const atoms = [];
    for (const line of text.split("\n")) {
      if (!line.startsWith("ATOM") && !line.startsWith("HETATM")) continue;
      const el = element((line.slice(76, 78).trim() || line.slice(12, 16).trim()[0]));
      atoms.push({
        x: parseFloat(line.slice(30, 38)), y: parseFloat(line.slice(38, 46)),
        z: parseFloat(line.slice(46, 54)), el,
        resn: line.slice(17, 20).trim(), resi: parseInt(line.slice(22, 26), 10),
        name: line.slice(12, 16).trim(),
      });
    }
    return atoms;
  }

  /** Minimal SDF/molblock reader: heavy atoms, coordinates, bond count. */
  function parseSDF(text) {
    const lines = text.split("\n");
    const counts = lines[3];
    const nAtoms = parseInt(counts.slice(0, 3), 10);
    const nBonds = parseInt(counts.slice(3, 6), 10);
    const coords = [], elements = [], keep = [];
    for (let i = 0; i < nAtoms; i++) {
      const l = lines[4 + i];
      const sym = l.slice(31, 34).trim().toUpperCase();
      if (sym === "H") { keep.push(-1); continue; }
      keep.push(coords.length / 3);
      coords.push(parseFloat(l.slice(0, 10)), parseFloat(l.slice(10, 20)), parseFloat(l.slice(20, 30)));
      elements.push(element(sym));
    }
    // Rotatable-bond count: single, acyclic, between two non-terminal heavy atoms.
    const degree = new Array(nAtoms).fill(0);
    const bonds = [];
    for (let i = 0; i < nBonds; i++) {
      const l = lines[4 + nAtoms + i];
      const a = parseInt(l.slice(0, 3), 10) - 1, b = parseInt(l.slice(3, 6), 10) - 1;
      const order = parseInt(l.slice(6, 9), 10);
      if (keep[a] < 0 || keep[b] < 0) continue;
      degree[a]++; degree[b]++;
      bonds.push({ a, b, order });
    }
    // A bond inside a ring cannot rotate. Ring membership is exactly
    // "not a bridge" in graph terms, so find the bridges with a depth-first
    // search rather than guessing -- the naive version counted every ring bond
    // and reported 26 rotatable bonds for imatinib, which has about seven.
    const bridges = findBridges(nAtoms, bonds);
    let rot = 0;
    for (let i = 0; i < bonds.length; i++) {
      const bd = bonds[i];
      if (bd.order === 1 && degree[bd.a] > 1 && degree[bd.b] > 1 && bridges.has(i)) rot++;
    }
    return { coords: Float32Array.from(coords), elements, rotatableBonds: rot,
             nAtoms: elements.length };
  }

  /** Indices of bonds whose removal disconnects the graph: the non-ring bonds. */
  function findBridges(nAtoms, bonds) {
    const adj = Array.from({ length: nAtoms }, () => []);
    bonds.forEach((b, i) => { adj[b.a].push([b.b, i]); adj[b.b].push([b.a, i]); });
    const disc = new Int32Array(nAtoms).fill(-1);
    const low = new Int32Array(nAtoms);
    const bridges = new Set();
    let timer = 0;
    for (let start = 0; start < nAtoms; start++) {
      if (disc[start] !== -1) continue;
      const stack = [[start, -1, 0]];
      disc[start] = low[start] = timer++;
      while (stack.length) {
        const frame = stack[stack.length - 1];
        const [node, parentEdge, ptr] = frame;
        if (ptr < adj[node].length) {
          frame[2]++;
          const [next, edge] = adj[node][ptr];
          if (edge === parentEdge) continue;
          if (disc[next] === -1) {
            disc[next] = low[next] = timer++;
            stack.push([next, edge, 0]);
          } else {
            low[node] = Math.min(low[node], disc[next]);
          }
        } else {
          stack.pop();
          if (stack.length) {
            const parent = stack[stack.length - 1][0];
            low[parent] = Math.min(low[parent], low[node]);
            if (low[node] > disc[parent]) bridges.add(parentEdge);
          }
        }
      }
    }
    return bridges;
  }

  function toPDBBlock(coords, elements, name) {
    const lines = [];
    for (let i = 0; i < elements.length; i++) {
      const el = elements[i];
      lines.push(
        "HETATM" + String(i + 1).padStart(5) + " " +
        (el + String(i + 1)).padEnd(4).slice(0, 4) + " " +
        (name || "LIG") + " A   1    " +
        coords[i * 3].toFixed(3).padStart(8) +
        coords[i * 3 + 1].toFixed(3).padStart(8) +
        coords[i * 3 + 2].toFixed(3).padStart(8) +
        "  1.00  0.00          " + el.padStart(2)
      );
    }
    lines.push("END");
    return lines.join("\n");
  }

  root.DockEngine = { RADII, buildGrids, dock, scorePose, rmsd, parsePDB, parseSDF,
                      toPDBBlock, mulberry32, transform, findBridges };
})(typeof globalThis !== "undefined" ? globalThis : this);

if (typeof module !== "undefined" && module.exports) module.exports = globalThis.DockEngine;
