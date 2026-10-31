const fs = require('node:fs');
const path = require('node:path');
const { setTimeout: delay } = require('node:timers/promises');

const MP4_TYPES = ['video/mp4;codecs=avc1.42001f', 'video/mp4'];

function newOutput(output, extension) {
  if (!path.isAbsolute(output) || path.extname(output).toLowerCase() !== extension) {
    throw new Error(`Use an absolute ${extension} output path`);
  }
  if (fs.existsSync(output)) throw new Error(`Refusing to overwrite ${output}`);
  fs.mkdirSync(path.dirname(output), { recursive: true });
}

class BrowserEvidenceRecorder {
  constructor({ browser, page, output, heading, provenance, seconds = 240 }) {
    if (!Number.isInteger(seconds) || seconds < 15 || seconds > 300) {
      throw new Error('Recording duration must be 15–300 seconds');
    }
    if (!heading || !provenance) throw new Error('Supply the PR/head heading and capture provenance');
    Object.assign(this, { browser, page, output, heading, provenance, seconds });
    this.lines = [];
    this.errors = [];
  }

  log(message) {
    const line = `${new Date().toISOString().slice(11, 19)} ${message}`;
    this.lines.push(line);
    this.lines = this.lines.slice(-100);
    console.log(line);
  }

  async start() {
    if (this.capturePage) throw new Error('Recorder already started');
    newOutput(this.output, '.mp4');
    this.capturePage = await this.browser.newPage({ viewport: { width: 1600, height: 900 } });
    try {
      await this.capturePage.setContent('<!doctype html><title>QA browser evidence</title><canvas width="1600" height="900"></canvas>');
      this.mime = await this.capturePage.evaluate(({ types, seconds }) => {
        const mime = types.find(type => MediaRecorder.isTypeSupported(type));
        if (!mime) throw new Error('This browser cannot record MP4; do not rename WebM to .mp4');
        const canvas = document.querySelector('canvas');
        const stream = canvas.captureStream(4);
        const chunks = [];
        const recorder = new MediaRecorder(stream, { mimeType: mime, videoBitsPerSecond: 2800000 });
        const state = window.qaEvidence = { recorder, error: null, canvas };
        state.finished = new Promise(resolve => {
          recorder.ondataavailable = event => { if (event.data.size) chunks.push(event.data); };
          recorder.onerror = event => { state.error = event.error?.message || 'MediaRecorder failed'; };
          recorder.onstop = () => {
            clearTimeout(state.deadline);
            stream.getTracks().forEach(track => track.stop());
            resolve(new Blob(chunks, { type: mime }));
          };
        });
        recorder.start(1000);
        state.deadline = setTimeout(() => {
          if (recorder.state !== 'inactive') recorder.stop();
        }, seconds * 1000);
        return mime;
      }, { types: MP4_TYPES, seconds: this.seconds });
      this.started = Date.now();
      this.sampling = this.sample().catch(error => {
        this.errors.push(String(error));
        this.log(`Capture failed: ${error.message}`);
      });
    } catch (error) {
      await this.capturePage.close();
      throw error;
    }
  }

  async sample() {
    let missed = 0;
    while (!this.stopping && Date.now() - this.started < this.seconds * 1000) {
      let frame;
      try {
        frame = await this.page.screenshot({ type: 'jpeg', quality: 80, timeout: 2000 });
        missed = 0;
      } catch (error) {
        this.errors.push(String(error));
        this.log('Browser frame unavailable; retain this gap in the QA report.');
        if (++missed >= 3) return;
        await delay(250);
        continue;
      }
      await this.capturePage.evaluate(async ({ frame, heading, provenance, lines, elapsed }) => {
        const image = new Image();
        image.src = 'data:image/jpeg;base64,' + frame;
        await image.decode();
        const ctx = window.qaEvidence.canvas.getContext('2d');
        ctx.fillStyle = '#0b1020'; ctx.fillRect(0, 0, 1600, 900);
        const scale = Math.min(1100 / image.width, 800 / image.height);
        ctx.drawImage(image, (1100 - image.width * scale) / 2, 55, image.width * scale, image.height * scale);
        ctx.font = 'bold 15px monospace'; ctx.fillStyle = '#9ce0d0'; ctx.fillText(heading, 20, 28, 1560);
        ctx.font = '14px monospace'; ctx.fillStyle = '#cbd5e1';
        ctx.fillText(`${provenance} | sampled browser frames | ${elapsed}s`, 20, 883, 1560);
        ctx.font = 'bold 17px monospace'; ctx.fillText('Contemporaneous QA output', 1120, 80);
        const rows = lines.flatMap(line => line.match(/.{1,49}/g) || ['']);
        ctx.font = '14px monospace';
        rows.slice(-38).forEach((line, index) => ctx.fillText(line, 1120, 110 + index * 19));
      }, {
        frame: frame.toString('base64'), heading: this.heading, provenance: this.provenance,
        lines: this.lines, elapsed: Math.round((Date.now() - this.started) / 1000),
      });
      await delay(250);
    }
  }

  async stop() {
    if (!this.sampling) throw new Error('Recorder has not started');
    if (this.result) return this.result;
    this.stopping = true;
    await this.sampling;
    try {
      const recording = await this.capturePage.evaluate(async () => {
        const state = window.qaEvidence;
        if (state.recorder.state !== 'inactive') state.recorder.stop();
        const blob = await state.finished;
        const data = await new Promise((resolve, reject) => {
          const reader = new FileReader();
          reader.onerror = () => reject(new Error('Cannot read recorded MP4'));
          reader.onload = () => resolve(reader.result.split(',')[1]);
          reader.readAsDataURL(blob);
        });
        return { data, error: state.error };
      });
      if (recording.error) this.errors.push(recording.error);
      fs.writeFileSync(this.output, Buffer.from(recording.data, 'base64'), { flag: 'wx' });
      this.result = { path: this.output, mime: this.mime, bytes: fs.statSync(this.output).size,
        elapsedSeconds: (Date.now() - this.started) / 1000, captureErrors: [...this.errors] };
      return this.result;
    } finally {
      await this.capturePage.close();
    }
  }
}

async function reviewRecording({ browser, video, outputDirectory, timestamps }) {
  if (!path.isAbsolute(video) || !path.isAbsolute(outputDirectory)) throw new Error('Use absolute evidence paths');
  const indexPath = path.join(outputDirectory, 'video-review.json');
  newOutput(indexPath, '.json');
  const page = await browser.newPage({ viewport: { width: 1600, height: 900 } });
  try {
    await page.setContent('<!doctype html><title>QA playback review</title><video muted></video><canvas></canvas>');
    // Load the complete file to avoid partial progressive-download duration metadata and file:// canvas restrictions.
    const metadata = await page.evaluate(async data => {
      const bytes = Uint8Array.from(atob(data), char => char.charCodeAt(0));
      const video = document.querySelector('video');
      video.src = URL.createObjectURL(new Blob([bytes], { type: 'video/mp4' }));
      await new Promise((resolve, reject) => {
        const timer = setTimeout(() => reject(new Error('MP4 decode timed out')), 15000);
        video.onloadeddata = () => { clearTimeout(timer); resolve(); };
        video.onerror = () => { clearTimeout(timer); reject(new Error(video.error?.message || 'MP4 decode failed')); };
      });
      await video.play();
      await new Promise(resolve => setTimeout(resolve, 500));
      video.pause();
      return { duration: video.duration, width: video.videoWidth, height: video.videoHeight, playbackAdvanced: video.currentTime > 0 };
    }, fs.readFileSync(video).toString('base64'));
    if (!metadata.playbackAdvanced || !Number.isFinite(metadata.duration) || metadata.duration <= 0 || metadata.duration > 301) {
      throw new Error(`Invalid playback or recording exceeds five minutes: ${JSON.stringify(metadata)}`);
    }
    const times = timestamps || [0, metadata.duration / 4, metadata.duration / 2, metadata.duration * 3 / 4, Math.max(0, metadata.duration - 0.5)];
    if (!Array.isArray(times) || times.length === 0) throw new Error('Select at least one decoded frame');
    const frames = [];
    for (const [index, second] of times.entries()) {
      if (!Number.isFinite(second) || second < 0 || second >= metadata.duration) throw new Error(`Invalid frame time ${second}`);
      const output = path.join(outputDirectory, `frame-${String(index + 1).padStart(2, '0')}.png`);
      newOutput(output, '.png');
      const data = await page.evaluate(async second => {
        const video = document.querySelector('video');
        if (Math.abs(video.currentTime - second) > 0.01) {
          await new Promise((resolve, reject) => {
            const timer = setTimeout(() => reject(new Error('MP4 seek timed out')), 10000);
            video.onseeked = () => { clearTimeout(timer); resolve(); };
            video.currentTime = second;
          });
        }
        const canvas = document.querySelector('canvas');
        canvas.width = video.videoWidth; canvas.height = video.videoHeight;
        canvas.getContext('2d').drawImage(video, 0, 0);
        return canvas.toDataURL('image/png').split(',')[1];
      }, second);
      fs.writeFileSync(output, Buffer.from(data, 'base64'), { flag: 'wx' });
      frames.push({ second, path: output });
    }
    const result = { ...metadata, frames, source: video, browser: browser.version() };
    fs.writeFileSync(indexPath, JSON.stringify(result, null, 2), { flag: 'wx' });
    return result;
  } finally {
    await page.close();
  }
}

module.exports = { BrowserEvidenceRecorder, reviewRecording };
