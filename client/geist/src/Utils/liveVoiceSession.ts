interface LiveVoiceOptions {
  model?: string;
  voice?: string;
  onReady: () => void;
  onAudioLevel?: (level: number) => void;
  onTranscript: (role: 'user' | 'assistant', text: string) => void;
  onError: (message: string) => void;
  onClosed: () => void;
}

async function responseBody(response: Response): Promise<any> {
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(typeof body.detail === 'string' ? body.detail : 'GPT-Live request failed.');
  }
  return body;
}

export class LiveVoiceSession {
  private peer: RTCPeerConnection | null = null;
  private channel: RTCDataChannel | null = null;
  private microphone: MediaStream | null = null;
  private audio: HTMLAudioElement | null = null;
  private disposed = false;
  private closing = false;
  private ready = false;
  private timers = new Set<ReturnType<typeof setTimeout>>();
  private abort = new AbortController();
  private audioContext: AudioContext | null = null;
  private analysers: AnalyserNode[] = [];
  private frame: number | null = null;

  constructor(private options: LiveVoiceOptions) {}

  private timeout(callback: () => void, duration: number) {
    const timer = setTimeout(() => {
      this.timers.delete(timer);
      callback();
    }, duration);
    this.timers.add(timer);
    return timer;
  }

  private fail(message: string) {
    if (this.disposed) return;
    this.options.onError(message);
    this.dispose();
  }

  private send(event: object) {
    if (!this.disposed && this.channel?.readyState === 'open') {
      this.channel.send(JSON.stringify(event));
    }
  }

  async start(): Promise<void> {
    try {
      this.peer = new RTCPeerConnection();
      if (this.options.onAudioLevel) {
        this.audioContext = new AudioContext();
        void this.audioContext.resume().catch(() => this.fail('Audio playback was blocked. Allow sound and reconnect.'));
      }
      this.audio = new Audio();
      this.audio.autoplay = true;
      this.peer.ontrack = event => {
        if (this.disposed || !this.audio) return;
        const output = new MediaStream([event.track]);
        this.audio.srcObject = output;
        this.monitor(output);
        void this.audio.play().catch(() => this.fail('Audio playback was blocked. Allow sound and reconnect.'));
      };
      this.peer.onconnectionstatechange = () => {
        if (this.peer?.connectionState === 'failed') this.fail('GPT-Live audio connection failed.');
      };
      const microphone = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true }
      });
      if (this.disposed) {
        microphone.getTracks().forEach(track => track.stop());
        return;
      }
      this.microphone = microphone;
      this.monitor(microphone);
      if (this.options.onAudioLevel) this.animate();
      for (const track of microphone.getAudioTracks()) this.peer.addTrack(track, microphone);
      this.channel = this.peer.createDataChannel('oai-events');
      this.channel.onmessage = event => this.handleEvent(event.data);
      this.channel.onclose = () => this.fail('GPT-Live disconnected before confirming session closure.');
      this.channel.onerror = () => this.fail('GPT-Live event connection failed.');
      const startupTimer = this.timeout(() => this.fail('GPT-Live connection timed out.'), 45000);
      const offer = await this.peer.createOffer();
      if (this.disposed) return;
      await this.peer.setLocalDescription(offer);
      const peer = this.peer;
      if (peer.iceGatheringState !== 'complete') {
        await new Promise<void>((resolve, reject) => {
          const finish = (error?: Error) => {
            clearTimeout(timer);
            this.timers.delete(timer);
            peer.removeEventListener('icegatheringstatechange', check);
            this.abort.signal.removeEventListener('abort', cancel);
            error ? reject(error) : resolve();
          };
          const check = () => { if (peer.iceGatheringState === 'complete') finish(); };
          const cancel = () => finish(new Error('Connection canceled.'));
          const timer = this.timeout(() => finish(new Error('ICE gathering timed out.')), 10000);
          peer.addEventListener('icegatheringstatechange', check);
          this.abort.signal.addEventListener('abort', cancel, { once: true });
          check();
        });
      }
      if (this.disposed) return;
      const sdp = peer.localDescription?.sdp;
      if (!sdp) throw new Error('Missing microphone connection offer.');
      const response = await fetch('/api/v1/voice/live/session', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sdp, model: this.options.model || 'gpt-live-1', voice: this.options.voice || 'marin' }),
        signal: this.abort.signal
      });
      const result = await responseBody(response);
      if (this.disposed) return;
      await peer.setRemoteDescription({ type: 'answer', sdp: result.sdp });
      // Readiness is confirmed by session.started, not the SDP response.
      if (this.ready) {
        clearTimeout(startupTimer);
        this.timers.delete(startupTimer);
      }
    } catch (error) {
      if (!this.disposed) this.fail(error instanceof Error ? error.message : 'Could not start GPT-Live.');
    }
  }

  private handleEvent(data: string) {
    if (this.disposed) return;
    let event: any;
    try { event = JSON.parse(data); } catch { this.fail('Invalid GPT-Live event.'); return; }
    if (!event || typeof event.type !== 'string') return;
    if (event.type === 'session.started') {
      this.ready = true;
      this.timers.forEach(clearTimeout);
      this.timers.clear();
      this.options.onReady();
    } else if (event.type === 'session.closed') {
      this.dispose();
    } else if (event.type === 'error') {
      this.fail('OpenAI rejected a GPT-Live command. Please reconnect.');
    } else if (event.type === 'session.input_transcript.delta' || event.type === 'session.output_transcript.delta') {
      if (typeof event.delta !== 'string' || !event.delta) return;
      const role = event.type === 'session.input_transcript.delta' ? 'user' : 'assistant';
      this.options.onTranscript(role, event.delta);
    } else if (event.type === 'session.delegation.created' && !this.closing && this.ready) {
      // This mode owns only voice and captions; no text-agent request is started.
      const id = event.delegation?.id;
      if (event.delegation?.target === 'client' && typeof id === 'string' && id) {
        this.send({ type: 'session.thinking.append', delegation_id: id,
          content: 'No backend is connected in this voice-only call. Answer conversationally when possible, or explain that this requires the text chat.' });
      }
    }
  }

  private monitor(stream: MediaStream) {
    if (!this.audioContext) return;
    const analyser = this.audioContext.createAnalyser();
    analyser.fftSize = 256;
    this.audioContext.createMediaStreamSource(stream).connect(analyser);
    this.analysers.push(analyser);
  }

  private animate = () => {
    if (this.disposed) return;
    let level = 0;
    const samples = new Uint8Array(256);
    for (const analyser of this.analysers) {
      analyser.getByteTimeDomainData(samples);
      let energy = 0;
      for (let i = 0; i < samples.length; i++) energy += ((samples[i] - 128) / 128) ** 2;
      level = Math.max(level, Math.min(1, Math.sqrt(energy / samples.length) * 5));
    }
    this.options.onAudioLevel?.(level);
    this.frame = requestAnimationFrame(this.animate);
  };

  close() {
    if (this.disposed || this.closing) return;
    this.closing = true;
    if (!this.ready || this.channel?.readyState !== 'open') { this.dispose(); return; }
    // Mute capture immediately, keeping the track alive for graceful finalization.
    this.microphone?.getTracks().forEach(track => { track.enabled = false; });
    this.timeout(() => this.fail('GPT-Live did not confirm closure before the timeout.'), 15000);
    this.send({ type: 'session.close' });
  }

  dispose() {
    if (this.disposed) return;
    if (!this.closing && this.ready) this.send({ type: 'session.close' });
    this.disposed = true;
    this.abort.abort();
    if (this.frame !== null) cancelAnimationFrame(this.frame);
    void this.audioContext?.close();
    this.audioContext = null;
    this.analysers = [];
    this.options.onAudioLevel?.(0);
    this.timers.forEach(clearTimeout);
    this.timers.clear();
    this.microphone?.getTracks().forEach(track => track.stop());
    if (this.channel) {
      this.channel.onmessage = null;
      this.channel.onclose = null;
      this.channel.onerror = null;
      this.channel.close();
    }
    if (this.peer) {
      this.peer.ontrack = null;
      this.peer.onconnectionstatechange = null;
      this.peer.close();
    }
    if (this.audio) { this.audio.pause(); this.audio.srcObject = null; }
    this.options.onClosed();
  }
}
