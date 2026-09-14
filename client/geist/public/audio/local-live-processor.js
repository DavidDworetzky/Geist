class LocalLiveProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this.frameSamples = options.processorOptions.frameSamples;
    this.input = new Float32Array(this.frameSamples);
    this.offset = 0;
    this.playback = [];
    this.playbackOffset = 0;
    this.active = false;
    this.port.onmessage = ({ data }) => {
      if (data.type === 'ready') this.active = true;
      if (data.type === 'audio') {
        this.playback.push(new Float32Array(data.pcm));
        // Never let delayed playback grow into seconds of stale speech.
        if (this.playback.length > 10) {
          this.playback = [];
          this.playbackOffset = 0;
          this.port.postMessage({ type: 'overrun' });
        }
      }
    };
  }

  process(inputs, outputs) {
    const input = inputs[0]?.[0];
    const output = outputs[0]?.[0];
    if (!this.active) return true;
    let energy = 0;
    for (let i = 0; i < (output?.length || 128); i++) {
      const sample = input?.[i] || 0;
      this.input[this.offset++] = sample;
      let played = 0;
      if (this.playback.length) {
        played = this.playback[0][this.playbackOffset++];
        if (this.playbackOffset === this.playback[0].length) {
          this.playback.shift();
          this.playbackOffset = 0;
        }
      }
      if (output) output[i] = played;
      energy += Math.max(sample * sample, played * played);
      if (this.offset === this.frameSamples) {
        const pcm = this.input.buffer;
        this.port.postMessage({ type: 'audio', pcm }, [pcm]);
        this.input = new Float32Array(this.frameSamples);
        this.offset = 0;
        this.port.postMessage({ type: 'level', level: Math.min(1, Math.sqrt(energy / 128) * 5) });
      }
    }
    return true;
  }
}

registerProcessor('local-live-processor', LocalLiveProcessor);
