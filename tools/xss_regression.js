/*
 * XSS regression test.
 *
 * Compound identifiers and veto reasons in the campaign data come from
 * whatever score table the pipeline was pointed at -- `from_score_table`
 * exists precisely so that can be an external CSV. An early version put
 * those strings straight into innerHTML, including inside a title="..."
 * attribute, and a SMILES field containing `"><img src=x onerror=...>`
 * executed in every visitor's browser. This test poisons the data and
 * asserts nothing runs.
 *
 * Usage: serve a copy of site/ whose data/campaigns.json carries the
 * payload (see tools/xss_regression.sh) on :8791, then run this.
 */
const puppeteer = require('puppeteer-core');
(async () => {
  const b = await puppeteer.launch({ executablePath: '/usr/bin/google-chrome', headless: 'new',
    args: ['--no-sandbox','--disable-dev-shm-usage','--disable-gpu','--use-gl=swiftshader','--enable-unsafe-swiftshader'] });
  const p = await b.newPage();
  await p.goto('http://127.0.0.1:8791/index.html', { waitUntil: 'networkidle2' });
  await new Promise(r => setTimeout(r, 2500));
  const fired  = await p.evaluate(() => !!window.__XSS_FIRED);
  const fired2 = await p.evaluate(() => !!window.__XSS_FIRED2);
  const imgs   = await p.evaluate(() => document.querySelectorAll('#cmpBody img, #vetoBody img').length);
  console.log('injected <img> elements in the DOM :', imgs);
  console.log('XSS via compound SMILES fired      :', fired);
  console.log('XSS via veto reason fired          :', fired2);
  console.log(fired || fired2 ? '>>> VULNERABLE' : '>>> not exploitable');
  await b.close();
})().catch(e => { console.error('PoC error:', e.message); process.exit(1); });
