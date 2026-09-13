export function wavRecording(chunks: ArrayBuffer[]): Blob {
  const size = chunks.reduce((total, chunk) => total + chunk.byteLength, 0);
  const header = new ArrayBuffer(44);
  const view = new DataView(header);
  const write = (offset: number, value: string) => {
    for (let i = 0; i < value.length; i++) view.setUint8(offset + i, value.charCodeAt(i));
  };
  write(0, 'RIFF'); view.setUint32(4, 36 + size, true); write(8, 'WAVE');
  write(12, 'fmt '); view.setUint32(16, 16, true); view.setUint16(20, 1, true);
  view.setUint16(22, 1, true); view.setUint32(24, 16000, true);
  view.setUint32(28, 32000, true); view.setUint16(32, 2, true); view.setUint16(34, 16, true);
  write(36, 'data'); view.setUint32(40, size, true);
  return new Blob([header, ...chunks], { type: 'audio/wav' });
}

export class DictationSession {
  private stream: MediaStream | null = null;
  private context: AudioContext | null = null;
  private processor: AudioWorkletNode | null = null;
  private chunks: ArrayBuffer[] = [];
  private disposed = false;
  private stopping = false;
  private timer: ReturnType<typeof setTimeout> | undefined;
  private abort = new AbortController();

  constructor(private options: {
    provider: string;
    onReady: () => void;
    onProcessing: () => void;
    onText: (text: string) => void;
    onError: (text: string) => void;
    onClosed: () => void;
  }) {}

  async start() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      if (this.disposed) { stream.getTracks().forEach(track => track.stop()); return; }
      this.stream = stream;
      this.context = new AudioContext({ sampleRate: 16000 });
      await this.context.audioWorklet.addModule('/audio/dictation-processor.js');
      if (this.disposed) return;
      await this.context.resume();
      if (this.disposed) return;
      this.processor = new AudioWorkletNode(this.context, 'dictation-processor');
      this.processor.port.onmessage = ({ data }) => {
        if (this.disposed) return;
        if (data.type === 'audio') this.chunks.push(data.pcm);
        if (data.type === 'stopped') void this.transcribe();
      };
      this.context.createMediaStreamSource(stream).connect(this.processor);
      this.processor.connect(this.context.destination);
      this.timer = setTimeout(() => this.stop(), 119000);
      this.options.onReady();
    } catch (error) {
      if (!this.disposed) {
        this.options.onError(error instanceof Error ? error.message : 'Could not record dictation.');
        this.dispose();
      }
    }
  }

  stop() {
    if (this.disposed || this.stopping) return;
    if (!this.processor) { this.dispose(); return; }
    this.stopping = true;
    this.options.onProcessing();
    clearTimeout(this.timer);
    this.timer = setTimeout(() => {
      this.options.onError('Dictation timed out. Please try again.');
      this.dispose();
    }, 65000);
    this.processor.port.postMessage('stop');
  }

  private releaseMicrophone() {
    this.stream?.getTracks().forEach(track => track.stop());
    this.stream = null;
    this.processor?.disconnect();
    if (this.processor) this.processor.port.onmessage = null;
    this.processor = null;
    void this.context?.close();
    this.context = null;
  }

  private async transcribe() {
    this.releaseMicrophone();
    try {
      const form = new FormData();
      form.append('audio_file', wavRecording(this.chunks), 'dictation.wav');
      const response = await fetch(`/api/v1/voice/transcribe?provider=${encodeURIComponent(this.options.provider)}`, {
        method: 'POST', body: form, signal: this.abort.signal
      });
      const body = await response.json();
      if (!response.ok) throw new Error(typeof body.detail === 'string' ? body.detail : 'Transcription failed.');
      if (!this.disposed) this.options.onText(body.text || '');
    } catch (error) {
      if (!this.disposed) this.options.onError(error instanceof Error ? error.message : 'Transcription failed.');
    } finally { this.dispose(); }
  }

  dispose() {
    if (this.disposed) return;
    this.disposed = true;
    clearTimeout(this.timer);
    this.abort.abort();
    this.releaseMicrophone();
    this.chunks = [];
    this.options.onClosed();
  }
}
