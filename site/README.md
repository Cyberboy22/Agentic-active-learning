# Docking Bench — static web app

A virtual-screening campaign readout beside a rigid-body docking engine that
runs entirely in the visitor's browser. No backend, no build step, no API keys.

## Deploy to Netlify

Everything is static, so any of these works. The folder *is* the site.

**Drag and drop (fastest).** Go to <https://app.netlify.com/drop> and drop this
`site/` folder onto the page. You get a live URL in a few seconds.

**Netlify CLI.**

```bash
npm install -g netlify-cli
cd site
netlify deploy --dir . --prod
```

**From Git.** Point Netlify at the repository and set:

- Base directory: `site`
- Build command: *(leave empty)*
- Publish directory: `site`

`netlify.toml` already sets long cache headers for `data/` and `vendor/` and
keeps the HTML uncached.

It works the same on GitHub Pages, Cloudflare Pages or Vercel — it is just
static files. To preview locally:

```bash
cd site && python3 -m http.server 8777
# http://127.0.0.1:8777
```

Opening `index.html` as a `file://` URL will **not** work: the page fetches its
data files and starts a Web Worker, and browsers block both from `file://`.

## What is in here

```
index.html          the page
app.js              UI, structure loading, campaign rendering
dock.js             the docking engine (scoring function, grids, search)
dock-worker.js      runs the engine off the UI thread
vendor/3Dmol-min.js 3Dmol.js, vendored so the page has no CDN dependency
data/campaigns.json campaign results + 3D conformers, from tools/build_viewer_data.py
data/1iep_*.pdb|sdf ABL1 kinase and imatinib, bundled so the page works offline
netlify.toml        publish settings and cache headers
```

Regenerate the campaign data after changing a campaign:

```bash
python tools/build_viewer_data.py && cp examples/viewer_data.json site/data/campaigns.json
```

## Structures

**ABL1 (PDB 1IEP)** ships with the app, so the page works with no network and
the redocking check always has a crystal pose to compare against.

**EGFR, CDK2 and c-Met** are fetched from RCSB when you pick them. These are the
campaigns' actual targets. The search box is centred on the largest non-solvent
heteroatom group in the file, which is normally the co-crystallised ligand. If
the fetch is blocked or a structure has no bound ligand, the page says so and
ABL1 still works.

RCSB serves `Access-Control-Allow-Origin: *`, so the browser fetch is allowed.
If you would rather have no third-party request at all, download the three PDB
files into `data/` and point `STRUCTURES` in `app.js` at the local copies.

## What was tested, and how

Checked in headless Chrome against this exact folder:

- campaign data loads and renders for all three targets
- 3Dmol initialises and draws the structure
- the worker builds the pocket grids
- **redocking imatinib recovers the crystallographic pose to 0.26 Å in ~0.7 s**
- docking a campaign compound returns a pose and a score
- no horizontal scroll at 390 px
- light and dark themes both render
- the fetched-structure path, with the RCSB response stubbed locally

Not testable from here: the real RCSB request, and Google Fonts. Both are plain
network fetches that work from a normal browser; the page degrades to system
fonts and to ABL1 if either is unavailable.

## Honest limits

The ligand is rigid, so torsional flexibility is not searched. The receptor is
rigid too. Desolvation and electrostatics are absent. Scores are in the scoring
function's arbitrary units — **not** kcal/mol, and not affinity predictions.
The affinity values behind the campaign are a documented model, not measurements,
except for 24 c-Met ligands with experimental free energies.
