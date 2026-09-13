# Windows browser evidence

Use this workflow for Geist's browser UI when Windows has the repository's
existing Node/Playwright dependencies and an installed browser with MP4
`MediaRecorder` support. The helper samples the test page into a canvas beside
timestamped QA output and records that canvas. It adds no dependency and does not
read configuration or launch an inference runtime.

## Coverage and prerequisites

- This is browser-only evidence from sampled screenshots, approximately a few
  frames per second. It covers the selected page, not the Windows desktop,
  Electron chrome, file pickers, or other OS dialogs. It does not establish smooth
  animation timing or prove brief states that fall between samples.
- Use an already available native recorder when desktop/OS behavior requires
  visual proof; otherwise mark that evidence requirement blocked. Browser dialog
  events can corroborate behavior, but they do not show an OS dialog's appearance.
- Reuse the installed `@playwright/test` from `client/geist/node_modules` and
  Microsoft Edge (`channel: 'msedge'`). If unavailable, an already installed
  compatible browser is acceptable. Do not invoke `npx` downloads or install a
  browser, codec, FFmpeg, or package without authorization.
- The helper probes MP4 support before starting. If none is supported, retain the
  error and mark recording blocked or use another available recorder. Renaming a
  WebM file to `.mp4` does not convert it.
- Use browser APIs only when the available browser-control instructions permit
  them. The helper accepts a Playwright browser and page owned by the QA script;
  it cannot directly accept a CUA tab handle. A separate isolated test browser
  may be used for the repeatable recorded checks when authorized.

## Paths and invocation

Use PowerShell throughout. Resolve the repo root from Git and create a unique
temporary evidence directory; do not embed an operator's username or checkout
path in scripts. Substitute the actual PR number:

```powershell
$qaPr = 123
$qaRepo = (git rev-parse --show-toplevel).Trim()
$qaOut = Join-Path ([IO.Path]::GetTempPath()) ('geist-qa-' + $qaPr + '-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $qaOut | Out-Null
& "$qaRepo/.venv/Scripts/python.exe" "$qaRepo/.agents/skills/qa-engineer/scripts/collect_pr_context.py" --pr $qaPr --output "$qaOut/context.json"
```

The context collector uses authenticated `gh`; follow the repository's Windows
keyring/elevation instructions. Stop on a failed preflight. Use the READY context's
base/head for subsequent inspection. Temporary paths belong in the subagent
handoff too, replacing the macOS `/private/tmp` examples.

## Connect recording to the actual QA checks

Import the checked-in helper from a task-owned Node script. Resolve Playwright
relative to the frontend package so scripts in the temporary directory can find
existing dependencies. This pattern assumes `selected-checks.cjs` contains the
independent, PR-specific checks and returns the skill's actual verdict:

```javascript
const fs = require('node:fs');
const path = require('node:path');
const { createRequire } = require('node:module');
const [repo, out, url] = process.argv.slice(2);
const frontendRequire = createRequire(path.join(repo, 'client/geist/package.json'));
const { chromium } = frontendRequire('@playwright/test');
const { BrowserEvidenceRecorder, reviewRecording } = require(
  path.join(repo, '.agents/skills/qa-engineer/scripts/browser_evidence.cjs')
);
const { runSelectedChecks } = require('./selected-checks.cjs');
const context = JSON.parse(fs.readFileSync(path.join(out, 'context.json'), 'utf8'));

async function main() {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  try {
    const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });
    await page.goto(url);
    const video = path.join(out, 'qa.mp4');
    const recorder = new BrowserEvidenceRecorder({
      browser, page, output: video, seconds: 240,
      heading: `PR #${context.pr.number} | ${context.pr.headRefOid}`,
      provenance: 'Isolated Geist browser | describe actual runtime and any fixtures',
    });
    await recorder.start();
    try {
      const result = await runSelectedChecks(page, message => recorder.log(message));
      recorder.log(`VERDICT: ${result.verdict}`);
      await page.waitForTimeout(1200); // Keep the final result visible in sampled frames.
    } catch (error) {
      recorder.log(`Check interrupted: ${error.message}`);
      throw error;
    } finally {
      const capture = await recorder.stop();
      fs.writeFileSync(path.join(out, 'capture.json'), JSON.stringify(capture, null, 2));
    }
    await reviewRecording({ browser, video, outputDirectory: path.join(out, 'review') });
  } finally {
    await browser.close();
  }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
```

Run it from PowerShell, passing the actual isolated QA origin:

```powershell
node "$qaOut/run-qa.cjs" "$qaRepo" "$qaOut" 'http://127.0.0.1:5368/chat'
```

Prepare the applicable runtime before recording. Use the runtime matrix to decide
whether real inference is required; a successful browser fixture does not prove a
model or runtime lane. Set the provenance text to the actual setup and log real
check results, correlated API output, and blockers. Avoid secrets in both the UI
and log messages. Name a new file/directory for each attempt; helpers refuse to
overwrite evidence.

The recorder closes only its own canvas page. The QA driver owns and closes the
test browser and any services. Always stop recording in `finally`; the recorder
also caps encoded duration at the configured 15–300 seconds. At that limit,
additional actions are no longer recorded. Keep the pathway near two to four
minutes and report any omitted required evidence. After an interrupted test,
review the preserved partial MP4 separately and retain the original failure.

## Decode, inspect, and report

`reviewRecording` opens a temporary page in the supplied browser, reads the
**complete** local MP4 into a Blob, checks dimensions/duration and advancing
playback, and exports five time-indexed PNGs plus `video-review.json`. It closes
that page afterward. Pass `timestamps: [2, 18, 40]` with observed times from the
actual recording when the default samples miss a critical state. Use a fresh
review directory for another attempt.

Loading the complete Blob avoids two observed pitfalls: incomplete duration
metadata during progressive loading, and tainted-canvas export errors from
`file://` video origins. The helper uses a local in-memory Blob; no media upload
or review server is needed.

Visually inspect the decoded frames, not just source screenshots. Confirm they
show the tested SHA, readable behavior and output, and final verdict. For brief
states such as modal feedback, also collect animation/DOM evidence or use an
appropriate higher-fidelity recorder. The helper does not certify visual quality
or decide the QA verdict.

Put the MP4 path in `artifacts.video_path`. Put capture method, browser/version,
codec, duration, fixture boundaries and limitations in `focused_tests[].evidence`
so the standard renderer includes them. Preserve missed-frame/capture errors
from `capture.json` and failed attempts in the report; an unreadable recording
or missing required visual proof remains blocked. Review and publish through
the existing report/publisher workflow and its authorization requirements. The
Windows capture method does not change runtime coverage or upload permissions.

## Validate changes to the helper

Run `node .agents/skills/qa-engineer/scripts/test_browser_evidence.cjs` from the
Geist checkout with the existing frontend dependencies and Edge available. It
uses an isolated synthetic page to test MP4 playback/frame export, the automatic
duration cap, output preservation, unsupported codecs, and capture failures.
It retains evidence in a unique system temporary directory and closes its test
browser. This helper smoke test is not a Geist product QA run.
