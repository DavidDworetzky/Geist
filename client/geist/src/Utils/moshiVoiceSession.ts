export class MoshiVoiceSession {
  private socket: WebSocket | null = null;
  private stream: MediaStream | null = null;
  private context: AudioContext | null = null;
  private processor: AudioWorkletNode | null = null;
  private disposed = false;
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
      const stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true } });
      if (this.disposed) { stream.getTracks().forEach(track => track.stop()); return; }
      this.stream = stream;
      this.context = new AudioContext({ sampleRate: 24000 });
      if (this.context.sampleRate !== 24000) {
        throw new Error(`Moshi requires 24000 Hz audio, but this browser uses ${this.context.sampleRate} Hz. Please try another browser or audio device.`);
      }
      await this.context.audioWorklet.addModule('/audio/moshi-processor.js');
      if (this.disposed) return;
      await this.context.resume();
      if (this.disposed) return;
      this.processor = new AudioWorkletNode(this.context, 'moshi-processor');
      this.context.createMediaStreamSource(stream).connect(this.processor);
      this.processor.connect(this.context.destination);
      const scheme = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const socket = new WebSocket(`${scheme}//${window.location.host}/api/v1/voice/moshi`);
      this.socket = socket;
      socket.binaryType = 'arraybuffer';
      this.timer = setTimeout(() => this.fail('Moshi model loading timed out.'), 120000);
      this.processor.port.onmessage = ({ data }) => {
        if (this.disposed) return;
        if (data.type === 'audio' && socket.readyState === WebSocket.OPEN) {
          if (socket.bufferedAmount > 76800) { this.fail('Moshi cannot keep up with live audio on this device.'); return; }
          socket.send(data.pcm);
        } else if (data.type === 'level') this.options.onAudioLevel(data.level);
        else if (data.type === 'overrun') this.fail('Moshi audio fell behind. Reconnect to start a fresh call.');
      };
      socket.onmessage = ({ data }) => {
        if (this.disposed) return;
        if (data instanceof ArrayBuffer) {
          this.processor?.port.postMessage({ type: 'audio', pcm: data }, [data]);
          return;
        }
        try {
          const event = JSON.parse(data);
          if (event.type === 'ready') {
            clearTimeout(this.timer);
            this.processor?.port.postMessage({ type: 'ready' });
            this.options.onReady();
          } else if (event.type === 'transcript') this.options.onTranscript('assistant', event.text);
          else if (event.type === 'error') this.fail(event.message || 'Moshi could not start.');
        } catch { this.fail('Invalid Moshi response.'); }
      };
      socket.onerror = () => this.fail('Could not connect to the local Moshi backend.');
      socket.onclose = () => this.fail('Moshi voice call ended.');
    } catch (error) {
      this.fail(error instanceof Error ? error.message : 'Could not start Moshi.');
    }
  }

  close() { this.dispose(); }

  dispose() {
    if (this.disposed) return;
    this.disposed = true;
    clearTimeout(this.timer);
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
