/* Docking Bench — campaign readout plus an in-browser docking calculation. */
(function () {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const fmt = (v, d = 2) => (v === null || v === undefined || !isFinite(v) ? "–" : Number(v).toFixed(d));
  const cssVar = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const status = (t) => { $("vpStatus").textContent = t; };

  /* Escape anything that reaches an innerHTML template.
     Compound identifiers and veto reasons come from the campaign data file,
     which is generated from whatever score table the pipeline was pointed at —
     a vendor CSV, a public dataset, a collaborator's export. Treating those
     strings as markup is how a poisoned data file turns into script execution
     in every visitor's browser, so nothing from the data is ever interpolated
     into HTML. Prefer el() / text below; this is for the few literal
     templates that remain. */
  const esc = (v) => String(v ?? "").replace(/[&<>"']/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  /** Create an element; `text` is always set as text, never parsed as markup. */
  function el(tag, opts) {
    const n = document.createElement(tag);
    if (!opts) return n;
    if (opts.className) n.className = opts.className;
    if (opts.text !== undefined) n.textContent = String(opts.text);
    if (opts.title !== undefined) n.title = String(opts.title);
    if (opts.style) n.setAttribute("style", opts.style);
    if (opts.attrs) for (const [k, v] of Object.entries(opts.attrs)) n.setAttribute(k, String(v));
    if (opts.children) for (const c of opts.children) if (c) n.appendChild(c);
    return n;
  }

  const clear = (node) => { while (node.firstChild) node.removeChild(node.firstChild); };

  /* Structures the viewer can load. ABL1 ships with the app so the page works
     offline and so the redocking check always has a crystal pose to compare
     against; the campaign targets are fetched from RCSB on demand. */
  const STRUCTURES = [
    { id: "abl1", label: "ABL1", pdb: "1IEP", bundled: true,
      receptor: "./data/1iep_receptor.pdb", ligand: "./data/1iep_ligand.sdf",
      title: "ABL1 kinase · PDB 1IEP", note: "imatinib co-crystal (bundled)" },
    { id: "egfr", label: "EGFR", pdb: "1M17", title: "EGFR kinase · PDB 1M17",
      note: "erlotinib co-crystal (fetched from RCSB)" },
    { id: "cdk2", label: "CDK2", pdb: "1AQ1", title: "CDK2 · PDB 1AQ1",
      note: "staurosporine co-crystal (fetched from RCSB)" },
    { id: "cmet", label: "c-Met", pdb: "3LQ8", title: "c-Met kinase · PDB 3LQ8",
      note: "inhibitor co-crystal (fetched from RCSB)" },
  ];
  const RCSB = (id) => `https://files.rcsb.org/download/${id}.pdb`;
  const SOLVENT = new Set(["HOH", "WAT", "DOD", "SO4", "PO4", "GOL", "EDO", "ACT", "CL", "NA",
    "MG", "ZN", "CA", "K", "MN", "PEG", "DMS", "TRS", "IOD", "BR", "NO3", "FMT", "MPD"]);

  let DATA = null, viewer = null, worker = null;
  let currentTarget = null, currentStruct = STRUCTURES[0];
  let selected = null, poseModel = null, ligandModelIdx = null;
  let ready = false, busy = false, seedCounter = 1;

  /* ---------- structure handling ---------- */

  /** Largest non-solvent HETATM group in a PDB: the likely bound ligand. */
  function findLigandSite(pdbText) {
    const groups = new Map();
    for (const line of pdbText.split("\n")) {
      if (!line.startsWith("HETATM")) continue;
      const resn = line.slice(17, 20).trim();
      if (SOLVENT.has(resn)) continue;
      const el = (line.slice(76, 78).trim() || "C").toUpperCase();
      if (el === "H") continue;
      const key = resn + "|" + line.slice(20, 27);
      if (!groups.has(key)) groups.set(key, { resn, atoms: [] });
      groups.get(key).atoms.push([
        parseFloat(line.slice(30, 38)), parseFloat(line.slice(38, 46)), parseFloat(line.slice(46, 54)),
      ]);
    }
    let best = null;
    for (const g of groups.values()) if (!best || g.atoms.length > best.atoms.length) best = g;
    if (!best || best.atoms.length < 6) return null;
    const c = [0, 0, 0];
    for (const a of best.atoms) { c[0] += a[0]; c[1] += a[1]; c[2] += a[2]; }
    return { resn: best.resn, nAtoms: best.atoms.length, center: c.map((v) => v / best.atoms.length) };
  }

  function initViewer(pdbText, ligandSDF, site) {
    if (typeof $3Dmol === "undefined") {
      status("3D library failed to load — the campaign panel still works.");
      return;
    }
    $("viewport").querySelectorAll("canvas").forEach((c) => c.remove());
    viewer = $3Dmol.createViewer($("viewport"), { backgroundColor: cssVar("--panel-2") || "#1d2733" });
    viewer.addModel(pdbText, "pdb");
    viewer.setStyle({}, {});
    viewer.setStyle({ hetflag: false }, { cartoon: { color: "spectrum", opacity: 0.92 } });
    ligandModelIdx = null;
    if (ligandSDF) {
      const m = viewer.addModel(ligandSDF, "sdf");
      m.setStyle({}, { stick: { radius: 0.17, colorscheme: "greenCarbon" } });
      ligandModelIdx = 1;
    } else if (site) {
      viewer.setStyle({ resn: site.resn, hetflag: true },
        { stick: { radius: 0.17, colorscheme: "greenCarbon" } });
    }
    addSurface(site);
    focusPocket(site);
    viewer.render();
  }

  function pocketSel(site) {
    if (ligandModelIdx !== null) return { model: ligandModelIdx };
    return site ? { resn: site.resn, hetflag: true } : null;
  }

  function addSurface(site) {
    const sel = pocketSel(site);
    if (!viewer || !sel) return;
    try {
      viewer.addSurface($3Dmol.SurfaceType.VDW,
        { opacity: 0.68, color: cssVar("--cyan") || "#68b7c9" },
        { hetflag: false, within: { distance: 9, sel } });
    } catch (e) { /* surface is decoration; never fail the page over it */ }
  }

  function focusPocket(site) {
    const sel = pocketSel(site);
    if (!viewer) return;
    // Frame the pocket with enough surrounding protein to read its shape.
    // zoom() multiplies, so a factor below 1 pulls back from the tight fit
    // zoomTo() gives the ligand alone.
    if (sel) { viewer.zoomTo(sel); viewer.zoom(0.62); } else { viewer.zoomTo(); }
  }

  function showPose(pdb, label) {
    if (!viewer) return;
    if (poseModel !== null) viewer.removeModel(poseModel);
    poseModel = viewer.addModel(pdb, "pdb");
    poseModel.setStyle({}, { stick: { radius: 0.21, color: cssVar("--accent") || "#e2a03f" } });
    viewer.render();
    status("Docked pose in amber" + (label ? " · " + label : ""));
  }

  /* ---------- worker ---------- */

  function startWorker(payload) {
    if (worker) worker.terminate();
    ready = false; setBusy(true);
    $("progBar").style.width = "0%";
    worker = new Worker("./dock-worker.js");
    worker.onmessage = onWorkerMessage;
    worker.onerror = (e) => {
      setBusy(true); ready = false;
      status("the docking worker failed to start: " + (e.message || "unknown error"));
    };
    worker.postMessage(payload);
  }

  function onWorkerMessage(e) {
    const m = e.data;
    if (m.type === "progress") {
      $("progBar").style.width = (m.fraction * 100).toFixed(0) + "%";
      return;
    }
    if (m.type === "ready") {
      ready = true; setBusy(false);
      $("progBar").style.width = "0%";
      $("btnValidate").disabled = m.crystalScore === null;
      status(`${currentStruct.title} · pocket mapped (${m.cavity.toLocaleString()} cavity points, `
        + `${m.receptorAtoms.toLocaleString()} receptor atoms)`);
      $("dockNote").innerHTML = m.crystalScore !== null
        ? `Crystal pose scores <b class="mono">${fmt(m.crystalScore)}</b> on these grids `
          + `(${m.crystalAtoms} heavy atoms, ${m.crystalRot} rotatable bonds). Scores are in the scoring `
          + `function's arbitrary units — <b>not</b> kcal/mol, and not affinity predictions.`
        : `Pocket centred on the bound ligand in this structure. No separate crystal ligand file, so the `
          + `redocking check is only available on the bundled ABL1 structure. Scores are in arbitrary `
          + `units — <b>not</b> kcal/mol.`;
      return;
    }
    if (m.type === "result") {
      setBusy(false);
      showPose(m.pdb, m.label);
      let html = `<span>score <b class="mono" style="color:var(--accent)">${fmt(m.score)}</b></span>`
        + `<span style="color:var(--muted)">normalised ${fmt(m.normalised)}</span>`
        + `<span style="color:var(--faint)">${m.atoms} atoms · ${m.rot} rot. bonds · ${m.ms} ms</span>`;
      if (m.isCrystal && m.rmsd !== null) {
        const good = m.rmsd < 2.0;
        html += `<span style="color:var(${good ? "--good" : "--bad"})">RMSD to crystal pose `
          + `<b>${fmt(m.rmsd)} Å</b> — ${good ? "pose recovered" : "wrong pocket, search failed this run"}</span>`;
      }
      $("scoreLine").style.color = "";
      $("scoreLine").innerHTML = html;  // numbers and fixed labels only
      return;
    }
    if (m.type === "error") {
      setBusy(false);
      $("scoreLine").textContent = m.message;
      $("scoreLine").style.color = "var(--bad)";
    }
  }

  function setBusy(b) {
    busy = b;
    $("btnDock").disabled = b || !ready;
    $("btnValidate").disabled = b || !ready || !currentStruct.bundled;
  }

  async function loadStructure(struct) {
    currentStruct = struct;
    renderStructTabs();
    $("structTitle").textContent = struct.title;
    status(`loading ${struct.pdb}…`);
    setBusy(true);
    try {
      let pdbText, ligandSDF = null, site = null;
      if (struct.bundled) {
        [pdbText, ligandSDF] = await Promise.all([
          fetch(struct.receptor).then((r) => r.text()),
          fetch(struct.ligand).then((r) => r.text()),
        ]);
        startWorker({ type: "init", receptor: pdbText, ligand: ligandSDF, boxSize: 24 });
      } else {
        const res = await fetch(RCSB(struct.pdb));
        if (!res.ok) throw new Error(`RCSB returned ${res.status}`);
        pdbText = await res.text();
        site = findLigandSite(pdbText);
        if (!site) throw new Error("no bound ligand found to centre a search box on");
        startWorker({ type: "init", receptor: pdbText, center: site.center, boxSize: 24 });
      }
      initViewer(pdbText, ligandSDF, site);
      status(`${struct.title} · mapping pocket…`);
    } catch (err) {
      setBusy(true);
      status(`could not load ${struct.pdb}: ${err.message}. The bundled ABL1 structure still works.`);
      $("dockNote").className = "note warn";
      $("dockNote").innerHTML = `<b>${esc(struct.pdb)} could not be fetched</b> (${esc(err.message)}). `
        + `Structures for the campaign targets are pulled from RCSB at load time; if that request is `
        + `blocked, pick ABL1, which ships with the page.`;
    }
  }

  /* ---------- campaign panel ---------- */

  function renderStructTabs() {
    const box = $("structTabs"); clear(box);
    STRUCTURES.forEach((s) => {
      const b = document.createElement("button");
      b.textContent = s.label;
      b.className = s.id === currentStruct.id ? "sel" : "";
      b.onclick = () => { if (!busy || ready) loadStructure(s); };
      box.appendChild(b);
    });
  }

  function renderTabs() {
    const box = $("targetTabs"); clear(box);
    DATA.targets.forEach((t) => {
      const b = document.createElement("button");
      b.textContent = t.target.toUpperCase();
      b.className = t === currentTarget ? "sel" : "";
      b.onclick = () => { currentTarget = t; selected = null; renderAll(); };
      box.appendChild(b);
    });
  }

  function renderMetrics() {
    const m = currentTarget.metrics;
    const cells = [
      ["recall@" + m.k, (m.recall_at_k * 100).toFixed(0) + "%"],
      ["EF 1%", fmt(m.ef_1pct, 1)],
      ["cost spent", fmt(m.cost_spent, 0)],
      ["confirmed", m.n_confirmed],
      ["library", currentTarget.library_size.toLocaleString()],
    ];
    const mb = $("metrics");
    clear(mb);
    cells.forEach(([k, v]) => mb.appendChild(el("div", { className: "metric", children: [
      el("span", { className: "v", text: v }),
      el("span", { className: "k", text: k }),
    ] })));
    $("targetDesc").textContent = `${currentTarget.description} · ${currentTarget.known_actives} known `
      + `actives in a ${currentTarget.library_size.toLocaleString()}-compound library`
      + (currentTarget.n_experimental ? ` · ${currentTarget.n_experimental} with measured ΔG` : "");
  }

  function renderLadder() {
    const by = (currentTarget.ledger && currentTarget.ledger.by_evaluator) || {};
    const calls = (currentTarget.ledger && currentTarget.ledger.calls) || {};
    const total = Object.values(by).reduce((a, b) => a + b, 0) || 1;
    const colour = { retrieval: "--cyan", ml_score: "--violet", docking: "--accent",
                     cofolding: "--bad", assay: "--good" };
    const lad = $("ladder");
    clear(lad);
    const names = Object.keys(by).sort((a, b) => by[b] - by[a]);
    if (!names.length) {
      lad.appendChild(el("span", { text: "no spend recorded",
                                   style: "color:var(--faint);font-size:.8rem" }));
      return;
    }
    names.forEach((k) => {
      const pct = ((by[k] / total) * 100).toFixed(1);
      // Only a key from the fixed colour map reaches a style attribute.
      const token = Object.prototype.hasOwnProperty.call(colour, k) ? colour[k] : "--muted";
      lad.appendChild(el("div", { className: "lad", children: [
        el("span", { className: "mono", text: k }),
        el("span", { className: "bar", children: [
          el("i", { style: `width:${pct}%;background:var(${token})` }),
        ] }),
        el("span", { className: "mono", style: "font-size:.72rem;color:var(--faint)",
                     text: `${calls[k] || 0} calls · ${by[k].toFixed(1)}` }),
      ] }));
    });
  }

  function renderCompounds() {
    const body = $("cmpBody");
    clear(body);
    currentTarget.compounds.forEach((c, i) => {
      const verdictCell = el("td", { children: [
        el("span", c.is_true_top_k
          ? { className: "pill hit", text: "top-" + currentTarget.metrics.k }
          : { className: "pill miss", text: "no" }),
        c.is_known_active ? document.createTextNode(" ") : null,
        c.is_known_active ? el("span", { className: "pill act", text: "known" }) : null,
      ] });
      const btn = el("button", { text: "Select", style: "padding:3px 9px;font-size:.74rem" });
      btn.dataset.i = String(i);
      btn.onclick = () => {
        selected = currentTarget.compounds[i];
        [...body.querySelectorAll("tr")].forEach((r, j) => r.classList.toggle("active", j === i));
        const pr = selected.properties || {};
        $("scoreLine").textContent = `Selected ${pr.mw} Da, ${pr.heavy_atoms} heavy atoms, `
          + `${pr.rot_bonds} rotatable bonds. Press Dock.`;
        $("scoreLine").style.color = "var(--faint)";
      };
      body.appendChild(el("tr", { children: [
        el("td", { className: "smi", text: c.smiles, title: c.smiles }),
        el("td", { className: "num", text: fmt(c.estimate) }),
        el("td", { className: "num", text: fmt(c.true_value) }),
        verdictCell,
        el("td", { children: [btn] }),
      ] }));
    });
  }

  function renderRounds() {
    const rb = $("roundBody");
    clear(rb);
    currentTarget.rounds.forEach((r) => {
      rb.appendChild(el("tr", { children: [
        el("td", { className: "num", text: r.round }),
        el("td", { className: "mono", text: r.evaluator, style: "font-size:.74rem" }),
        el("td", { className: "num", text: r.n }),
        el("td", { className: "num", text: fmt(r.cost, 1) }),
        el("td", { className: "num", text: (r.recall * 100).toFixed(0) + "%" }),
      ] }));
    });
    const last = currentTarget.rounds[currentTarget.rounds.length - 1];
    $("rationaleNote").textContent = last ? "Last rationale — " + last.rationale : "";
  }

  function renderVetoes() {
    const body = $("vetoBody");
    clear(body);
    if (!currentTarget.overrides.length) {
      body.appendChild(el("tr", { children: [
        el("td", { text: "Nothing vetoed in this run.", style: "color:var(--faint)",
                   attrs: { colspan: 2 } }),
      ] }));
      return;
    }
    currentTarget.overrides.forEach((o) => {
      body.appendChild(el("tr", { children: [
        el("td", { className: "smi", text: o.smiles, title: o.smiles }),
        el("td", { text: o.reason, style: "font-size:.76rem;color:var(--muted)" }),
      ] }));
    });
  }

  function renderAll() {
    renderTabs(); renderMetrics(); renderLadder();
    renderCompounds(); renderRounds(); renderVetoes();
  }

  /* ---------- wiring ---------- */

  $("btnDock").onclick = () => {
    if (!selected) {
      $("scoreLine").textContent = "Pick a compound from the shortlist first.";
      $("scoreLine").style.color = "var(--bad)";
      return;
    }
    setBusy(true);
    status("searching placements…");
    $("scoreLine").textContent = "docking…";
    $("scoreLine").style.color = "var(--muted)";
    worker.postMessage({ type: "dock", sdf: selected.sdf, seed: seedCounter++, label: "campaign compound" });
  };

  $("btnValidate").onclick = () => {
    setBusy(true);
    status("redocking the crystal ligand…");
    $("scoreLine").textContent = "redocking from a random start…";
    $("scoreLine").style.color = "var(--muted)";
    worker.postMessage({ type: "dock", useCrystal: true, seed: seedCounter++, label: "crystal-ligand redock" });
  };

  let surfaceOn = true;
  $("btnSurface").onclick = () => {
    if (!viewer) return;
    surfaceOn = !surfaceOn;
    $("btnSurface").classList.toggle("sel", surfaceOn);
    if (surfaceOn) addSurface(null); else viewer.removeAllSurfaces();
    viewer.render();
  };
  $("btnReset").onclick = () => { if (viewer) { focusPocket(null); viewer.render(); } };

  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
    if (viewer) { viewer.setBackgroundColor(cssVar("--panel-2")); viewer.render(); }
  });

  /* Expose a little state for automated checks. */
  window.__bench = {
    isReady: () => ready,
    lastScoreText: () => $("scoreLine").textContent.trim(),
    statusText: () => $("vpStatus").textContent.trim(),
    targets: () => (DATA ? DATA.targets.map((t) => t.target) : []),
    selectCompound: (i) => { const b = $("cmpBody").querySelectorAll("button")[i]; if (b) b.click(); },
    dock: () => $("btnDock").click(),
    validate: () => $("btnValidate").click(),
  };

  fetch("./data/campaigns.json")
    .then((r) => r.json())
    .then((d) => { DATA = d; currentTarget = d.targets[0]; renderAll(); })
    .catch((e) => { $("targetDesc").textContent = "campaign data failed to load: " + e.message; });

  renderStructTabs();
  loadStructure(STRUCTURES[0]);
})();
