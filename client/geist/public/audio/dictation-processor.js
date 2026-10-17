/* Audio capture runs on the audio rendering thread, not the React UI thread. */
class DictationProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.recording = true;
    this.frames = [];
    this.samples = 0;
    this.port.onmessage = ({ data }) => {
      if (data === 'stop') {
        this.recording = false;
        this.flush();
        this.port.postMessage({ type: 'stopped' });
      }
    };
  }

  flush() {
    if (!this.samples) return;
    const pcm = new Int16Array(this.samples);
    let offset = 0;
    for (const frame of this.frames) {
      for (const value of frame) {
        const sample = Math.max(-1, Math.min(1, value));
        pcm[offset++] = sample < 0 ? sample * 32768 : sample * 32767;
      }
    }
    this.port.postMessage({ type: 'audio', pcm: pcm.buffer }, [pcm.buffer]);
    this.frames = [];
    this.samples = 0;
  }

  process(inputs) {
    const audio = inputs[0]?.[0];
    if (this.recording && audio) {
      this.frames.push(audio.slice());
      this.samples += audio.length;
      if (this.samples >= 4096) this.flush();
    }
    return this.recording;
  }
}

registerProcessor('dictation-processor', DictationProcessor);
