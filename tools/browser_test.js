/*
 * End-to-end browser test for site/.
 *
 * Serve the site first:   cd site && python3 -m http.server 8777
 * Then:                   node tools/browser_test.js
 *
 * Needs a Chrome binary and puppeteer-core (npm i puppeteer-core).
 * This is what caught the invalid-JSON bug: Python will happily write
 * Infinity into a JSON file and every browser rejects the whole document,
 * which blanked the entire campaign panel while every Python test passed.
 */
const puppeteer = require('puppeteer-core');

(async () => {
  const browser = await puppeteer.launch({
    executablePath: '/usr/bin/google-chrome',
    headless: 'new',
    args: ['--no-sandbox', '--disable-dev-shm-usage', '--disable-gpu', '--use-gl=swiftshader',
           '--enable-unsafe-swiftshader', '--window-size=1280,900'],
  });
  const page = await browser.newPage();
  await page.setViewport({ width: 1280, height: 900 });

  const errors = [], failures = [];
  page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
  page.on('pageerror', (e) => errors.push('PAGEERROR: ' + e.message));
  page.on('requestfailed', (r) => failures.push(r.url() + ' -> ' + (r.failure()||{}).errorText));

  await page.goto('http://127.0.0.1:8777/index.html', { waitUntil: 'networkidle2', timeout: 60000 });

  // 1. campaign data rendered
  const targets = await page.evaluate(() => window.__bench.targets());
  const metricCount = await page.$$eval('.metric', (n) => n.length);
  const rows = await page.$$eval('#cmpBody tr', (n) => n.length);
  const rounds = await page.$$eval('#roundBody tr', (n) => n.length);
  const ladder = await page.$$eval('.lad', (n) => n.length);
  console.log('targets loaded      :', targets.join(', '));
  console.log('metric tiles        :', metricCount);
  console.log('shortlist rows      :', rows);
  console.log('round rows          :', rounds);
  console.log('ladder bars         :', ladder);

  // 2. 3Dmol rendered a canvas
  const canvas = await page.$$eval('#viewport canvas', (n) => n.map(c => c.width + 'x' + c.height));
  console.log('viewer canvases     :', JSON.stringify(canvas));

  // 3. worker built the pocket grids
  await page.waitForFunction(() => window.__bench.isReady(), { timeout: 120000 });
  console.log('pocket status       :', await page.evaluate(() => window.__bench.statusText()));

  // 4. redock the crystal ligand — the real validation
  await page.evaluate(() => window.__bench.validate());
  await page.waitForFunction(() => /RMSD|score/.test(window.__bench.lastScoreText()) &&
                                   !/redocking/.test(window.__bench.lastScoreText()), { timeout: 120000 });
  const redock = await page.evaluate(() => window.__bench.lastScoreText());
  console.log('redock result       :', redock.replace(/\s+/g, ' '));

  // 5. dock a campaign compound
  await page.evaluate(() => window.__bench.selectCompound(0));
  await page.evaluate(() => window.__bench.dock());
  await page.waitForFunction(() => /score/.test(window.__bench.lastScoreText()) &&
                                   !/docking…/.test(window.__bench.lastScoreText()), { timeout: 120000 });
  console.log('compound dock       :', (await page.evaluate(() => window.__bench.lastScoreText())).replace(/\s+/g,' '));

  await page.screenshot({ path: '/home/claude/shot-dark.png' });

  // 6. light theme
  await page.emulateMediaFeatures([{ name: 'prefers-color-scheme', value: 'light' }]);
  await new Promise(r => setTimeout(r, 600));
  await page.screenshot({ path: '/home/claude/shot-light.png' });

  // 7. phone width, check for horizontal overflow
  await page.setViewport({ width: 390, height: 844 });
  await new Promise(r => setTimeout(r, 800));
  const overflow = await page.evaluate(() =>
    ({ scroll: document.documentElement.scrollWidth, client: document.documentElement.clientWidth }));
  console.log('phone 390px         : scrollWidth', overflow.scroll, 'clientWidth', overflow.client,
              overflow.scroll <= overflow.client + 1 ? '(no horizontal scroll)' : '(OVERFLOW!)');
  await page.screenshot({ path: '/home/claude/shot-mobile.png' });

  console.log('console errors      :', errors.length ? errors.slice(0,6) : 'none');
  console.log('failed requests     :', failures.length ? failures.slice(0,6) : 'none');
  await browser.close();
})().catch(e => { console.error('TEST FAILED:', e.message); process.exit(1); });
