#!/usr/bin/env bash
# Build a poisoned copy of the site, serve it, and assert no script executes.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

cp -r "$ROOT/site" "$WORK/site"
python3 - "$WORK/site/data/campaigns.json" <<'PY'
import json, sys, pathlib
p = pathlib.Path(sys.argv[1]); d = json.loads(p.read_text())
d["targets"][0]["compounds"][0]["smiles"] = '"><img src=x onerror="window.__XSS_FIRED=true">'
d["targets"][0]["overrides"] = [{"index": 1, "smiles": "CCO", "action": "veto", "round_idx": 0,
    "reason": '<img src=x onerror="window.__XSS_FIRED2=true">'}]
p.write_text(json.dumps(d))
PY

(cd "$WORK/site" && python3 -m http.server 8791 --bind 127.0.0.1 >/dev/null 2>&1) &
SRV=$!
trap 'kill $SRV 2>/dev/null; rm -rf "$WORK"' EXIT
sleep 2
node "$ROOT/tools/xss_regression.js"
