export class LocalVoiceSession {
  private socket: WebSocket | null = null;
  private stream: MediaStream | null = null;
  private context: AudioContext | null = null;
  private processor: AudioWorkletNode | null = null;
  private disposed = false;
  private abort = new AbortController();
  private timer: ReturnType<typeof setTimeout> | undefined;

  constructor(private options: {
    onReady: () => void;
    onTranscript: (role: 'user' | 'assistant', text: string) => void;
    onAudioLevel: (level: number) => void;
    onError: (text: string) => void;
    onClosed: () => void;
  }) {}

  private fail(message: string) {
    if (this.disposed) return;
    this.options.onError(message);
    this.dispose();
  }

  async start() {
    try {
      this.timer = setTimeout(() => this.fail('Local voice configuration timed out.'), 10000);
      const response = await fetch('/api/v1/voice/live/local', { signal: this.abort.signal });
      const config = await response.json();
      if (this.disposed) return;
      clearTimeout(this.timer);
      if (!response.ok) throw new Error(typeof config.detail === 'string' ? config.detail : 'Could not load local voice configuration.');
      if (config.protocol !== 'geist-pcm-v1' || config.endpoint !== '/api/v1/voice/live/local' ||
          ![16000, 24000, 48000].includes(config.sample_rate) || !Number.isInteger(config.frame_samples) ||
          config.frame_samples < config.sample_rate / 100 || config.frame_samples > config.sample_rate / 10) {
        throw new Error('Unsupported local voice audio configuration.');
      }
      const frameBytes = config.frame_samples * 4;
      const stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true } });
      if (this.disposed) { stream.getTracks().forEach(track => track.stop()); return; }
      this.stream = stream;
      this.context = new AudioContext({ sampleRate: config.sample_rate });
      if (this.context.sampleRate !== config.sample_rate) {
        throw new Error(`Local voice requires ${config.sample_rate} Hz audio, but this browser uses ${this.context.sampleRate} Hz. Please try another browser or audio device.`);
      }
      await this.context.audioWorklet.addModule('/audio/local-live-processor.js');
      if (this.disposed) return;
      await this.context.resume();
      if (this.disposed) return;
      this.processor = new AudioWorkletNode(this.context, 'local-live-processor', { processorOptions: { frameSamples: config.frame_samples } });
      this.context.createMediaStreamSource(stream).connect(this.processor);
      this.processor.connect(this.context.destination);
      const scheme = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const socket = new WebSocket(`${scheme}//${window.location.host}${config.endpoint}`);
      this.socket = socket;
      socket.binaryType = 'arraybuffer';
      this.timer = setTimeout(() => this.fail('Local voice model loading timed out.'), 120000);
      this.processor.port.onmessage = ({ data }) => {
        if (this.disposed) return;
        if (data.type === 'audio' && socket.readyState === WebSocket.OPEN) {
          if (socket.bufferedAmount > frameBytes * 10) { this.fail('Local voice cannot keep up with live audio on this device.'); return; }
          socket.send(data.pcm);
        } else if (data.type === 'level') this.options.onAudioLevel(data.level);
        else if (data.type === 'overrun') this.fail('Local voice audio fell behind. Reconnect to start a fresh call.');
      };
      socket.onmessage = ({ data }) => {
        if (this.disposed) return;
        if (data instanceof ArrayBuffer) {
          if (data.byteLength !== frameBytes) { this.fail('Invalid local voice audio frame.'); return; }
          this.processor?.port.postMessage({ type: 'audio', pcm: data }, [data]);
          return;
        }
        try {
          const event = JSON.parse(data);
          if (event.type === 'ready') {
            clearTimeout(this.timer);
            this.processor?.port.postMessage({ type: 'ready' });
            this.options.onReady();
          } else if (event.type === 'transcript' && typeof event.text === 'string' && ['user', 'assistant'].includes(event.role || 'assistant')) this.options.onTranscript(event.role || 'assistant', event.text);
          else if (event.type === 'error') this.fail(event.message || 'Local voice could not start.');
        } catch { this.fail('Invalid local voice response.'); }
      };
      socket.onerror = () => this.fail('Could not connect to the local voice backend.');
      socket.onclose = () => this.fail('Local voice call ended.');
    } catch (error) {
      this.fail(error instanceof Error ? error.message : 'Could not start local voice.');
    }
  }

  close() { this.dispose(); }

  dispose() {
    if (this.disposed) return;
    this.disposed = true;
    clearTimeout(this.timer);
    this.abort.abort();
    this.stream?.getTracks().forEach(track => track.stop());
    if (this.processor) { this.processor.port.onmessage = null; this.processor.disconnect(); }
    void this.context?.close();
    if (this.socket) {
      this.socket.onmessage = null;
      this.socket.onclose = null;
      this.socket.onerror = null;
      this.socket.close();
    }
    this.options.onAudioLevel(0);
    this.options.onClosed();
  }
}
