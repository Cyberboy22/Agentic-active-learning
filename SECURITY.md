# Security notes

What was checked before deployment, what was found, and what remains true.

## Audited

| area | result |
|---|---|
| Secrets in the repository | none. The Anthropic key is read from `ANTHROPIC_API_KEY` at runtime and never written to disk or into a cache file. |
| Personal data | none in the source or data files. |
| `eval` / `new Function` in shipped JavaScript | none. |
| Unsafe deserialisation in Python | none. Checkpoints are JSON, never pickle, so loading an untrusted checkpoint cannot execute code. |
| Shell injection | one `subprocess` call, in `provenance.py`, with a fixed argument list and no shell. |
| Vendored dependency | 3Dmol.js 2.5.5, verified byte-for-byte against npm's published SHA-512 integrity hash. |
| Content-Security-Policy | enforced, and verified not to break the app (see below). |

## One real vulnerability, found and fixed

**Stored XSS through campaign data.** Compound identifiers and medicinal-chemistry
veto reasons were interpolated straight into `innerHTML`, including inside a
`title="..."` attribute. A SMILES field containing

```
"><img src=x onerror="...">
```

executed script in every visitor's browser.

This was not hypothetical. `from_score_table` exists so the framework can be
pointed at an external score table — DOCKSTRING, a public dataset, a
collaborator's CSV — and whatever is in that file flows into `campaigns.json`
and then into the page. A poisoned or merely corrupt data file was enough.

Fixed by building those rows with DOM APIs and `textContent`, so nothing from
the data file is ever parsed as markup. Hostile strings now render literally,
which is the right behaviour: you can see that the data is bad.

Proven with a working exploit before the fix and re-run after:

```bash
./tools/xss_regression.sh     # needs node + puppeteer-core + a Chrome binary
```

Before: 3 injected elements, both payloads fired. After: 0 injected elements,
nothing fired.

## Defence in depth

`netlify.toml` sets a Content-Security-Policy with `script-src 'self'` and no
`'unsafe-inline'` for scripts, so even a future escaping bug cannot run an
injected handler. `connect-src` allows only this origin and `files.rcsb.org`;
framing, form submission and plugins are blocked outright. Verified in a real
browser with the headers applied: no violations, the Web Worker still runs, and
the redocking check still recovers the crystal pose to 0.26 Å.

## What the app does and does not touch

- No backend, no database, no accounts, no cookies, no analytics, no tracking.
- No user input is collected, stored or transmitted anywhere.
- One third-party request at runtime: the structure files fetched from RCSB
  when a non-bundled target is selected. To remove even that, download the
  three PDB files into `site/data/` and point `STRUCTURES` in `app.js` at the
  local copies.
- One third-party stylesheet: Google Fonts. The page degrades to system fonts
  if it is blocked.

## Not verified here

The live RCSB request and the Google Fonts request are blocked from the
sandbox this was built in. Both are ordinary browser fetches and the app
handles failure of either, but check them once after deploying.
