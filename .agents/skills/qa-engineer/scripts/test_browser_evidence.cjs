const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { createRequire } = require('node:module');
const { setTimeout: delay } = require('node:timers/promises');
const repo = path.resolve(__dirname, '../../../..');
const frontendRequire = createRequire(path.join(repo, 'client/geist/package.json'));
const { chromium } = frontendRequire('@playwright/test');
const { BrowserEvidenceRecorder, reviewRecording } = require('./browser_evidence.cjs');
const out = fs.mkdtempSync(path.join(os.tmpdir(), 'geist-windows-recorder-'));
(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const checks = [];
  try {
    const page = await browser.newPage({ viewport: { width: 1100, height: 800 } });
    await page.setContent('<style>body{font:24px system-ui;padding:60px;background:#171a21;color:white}button{font:inherit;padding:12px;background:#8066ef;color:white;border:0}</style><h1>Windows recording helper smoke test</h1><p>This is a synthetic helper test, not Geist product QA.</p><button onclick="this.textContent=\'Action captured\'">Exercise browser action</button>');
    const common = { browser, page, heading:'Windows recorder helper smoke | no product QA claimed', provenance:'Synthetic test page | installed Edge', seconds:15 };
    assert.throws(() => new BrowserEvidenceRecorder({ ...common, seconds:301 }), /15–300/);
    assert.throws(() => new BrowserEvidenceRecorder({ ...common, seconds:1 }), /15–300/);
    await assert.rejects(new BrowserEvidenceRecorder({ ...common, output:'relative.mp4' }).start(), /absolute/);
    const existing = path.join(out, 'existing.mp4');
    fs.writeFileSync(existing, 'preserve this existing evidence');
    await assert.rejects(new BrowserEvidenceRecorder({ ...common, output:existing }).start(), /overwrite/);
    assert.equal(fs.readFileSync(existing, 'utf8'), 'preserve this existing evidence');
    checks.push('invalid limits/relative paths rejected; existing evidence preserved');

    const unsupported = await browser.newContext();
    await unsupported.addInitScript(() => { MediaRecorder.isTypeSupported = () => false; });
    await assert.rejects(new BrowserEvidenceRecorder({ ...common, browser:unsupported, output:path.join(out,'unsupported.mp4') }).start(), /cannot record MP4/);
    assert.equal(unsupported.pages().length, 0);
    await unsupported.close();
    checks.push('unsupported codec fails clearly and closes recorder page');

    const recorder = new BrowserEvidenceRecorder({ ...common, output:path.join(out,'smoke.mp4') });
    recorder.log('Starting helper validation with actual browser interaction.');
    await recorder.start();
    await delay(800);
    await page.getByRole('button', { name:'Exercise browser action' }).click();
    assert.equal(await page.getByRole('button').textContent(), 'Action captured');
    recorder.log('PASS: browser action observed; testing automatic 15-second capture cap.');
    await delay(16200);
    const capture = await recorder.stop();
    assert.deepEqual(capture.captureErrors, []);
    assert(capture.bytes > 1000);
    assert.deepEqual(await recorder.stop(), capture);
    assert.equal(page.isClosed(), false);
    const review = await reviewRecording({ browser, video:capture.path, outputDirectory:path.join(out,'review') });
    assert(review.playbackAdvanced);
    assert.equal(review.width, 1600); assert.equal(review.height, 900);
    assert(review.duration >= 14 && review.duration <= 16);
    assert.equal(review.frames.length, 5);
    await assert.rejects(reviewRecording({ browser, video:capture.path, outputDirectory:path.join(out,'review') }), /overwrite/);
    checks.push('actual MP4, automatic duration cap, playback, decoded frames, idempotent stop, caller page preserved');

    const failurePage = await browser.newPage();
    await failurePage.setContent('<h1>Intentional screenshot-loss fixture</h1>');
    const partial = new BrowserEvidenceRecorder({ ...common, page:failurePage, output:path.join(out,'partial.mp4') });
    await partial.start(); await delay(600); await failurePage.close(); await delay(1200);
    const failure = await partial.stop();
    assert(failure.captureErrors.length >= 3);
    checks.push('closed target yields reported capture errors and preserves partial recording');
    const result = { result:'PASS', out, browser:browser.version(), checks, capture, review, partial:failure };
    fs.writeFileSync(path.join(out,'self-test.json'), JSON.stringify(result,null,2));

    console.log(JSON.stringify(result,null,2));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); console.error('Evidence retained at '+out); process.exitCode=1; });
