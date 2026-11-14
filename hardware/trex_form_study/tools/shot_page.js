// Screenshot a page fragment the way the Artifact publisher will show it, and report what breaks on a phone.
//
//   node tools/shot_page.js PAGE.html OUT_PREFIX [WIDTH ...]      (default widths: 1280 390)
//
// The publisher wraps the file in its own doctype, charset and device-width viewport; this does the same in a
// temporary copy, opens it in the pre-installed Chromium (Playwright; never run "playwright install"), waits for it to
// settle and writes OUT_PREFIX_<width>.png at deviceScaleFactor 1 (for checking, not for showing). It prints the page's
// scroll width against the viewport (wider means the page scrolls sideways, which the page contract forbids), its
// height and any script errors, and exits 1 if any width overflows or errs.
const { chromium } = require('playwright');
const fs = require('fs');
const os = require('os');
const path = require('path');

(async () => {
  const [src, prefix, ...ws] = process.argv.slice(2);
  if (!src || !prefix) {
    console.error('usage: node tools/shot_page.js PAGE.html OUT_PREFIX [WIDTH ...]');
    process.exit(2);
  }
  const widths = ws.length ? ws.map(Number) : [1280, 390];
  const page = path.resolve(src);
  // the wrapped copy sits beside the page, so that relative media paths still resolve
  const wrapped = path.join(path.dirname(page), `.shot_${process.pid}_${path.basename(page)}`);
  fs.writeFileSync(wrapped, '<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" '
    + 'content="width=device-width, initial-scale=1, viewport-fit=cover"></head><body>'
    + fs.readFileSync(page, 'utf8') + '</body></html>');
  let bad = 0;
  const browser = await chromium.launch();
  try {
    for (const w of widths) {
      const p = await browser.newPage({ viewport: { width: w, height: 1000 }, deviceScaleFactor: 1 });
      const errs = [];
      p.on('pageerror', (e) => errs.push(String(e)));
      await p.goto('file://' + wrapped);
      await p.waitForTimeout(800);
      const [sw, cw, h] = await p.evaluate(() => [document.documentElement.scrollWidth,
        document.documentElement.clientWidth, document.body.scrollHeight]);
      const out = `${prefix}_${w}.png`;
      await p.screenshot({ path: out });
      const over = sw > cw;
      if (over || errs.length) bad += 1;
      console.log(`${w}px: scrollWidth ${sw} / clientWidth ${cw}${over ? ' OVERFLOW' : ''}, height ${h}, `
        + `errors ${errs.length}${errs.length ? ': ' + errs.join(' | ') : ''} -> ${out}`);
      await p.close();
    }
  } finally {
    await browser.close();
    fs.unlinkSync(wrapped);
  }
  process.exit(bad ? 1 : 0);
})();
