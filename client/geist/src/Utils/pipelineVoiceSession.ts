import { createVoicePcmEncoder } from '../Hooks/voicePcm';

/** Speech-to-speech providers with their own transport; every other voice uses the pipeline. */
export const CONVERSATION_PROVIDERS = new Set(['openai_live', 'local_live']);

export interface PipelineVoiceOptions {
  sessionId: number;
  agentType?: string;
  sttProvider?: string;
  ttsProvider?: string;
  ttsModel?: string;
  ttsVoice?: string;
  ttsLanguage?: string;
  onReady: () => void;
  onUserText: (text: string) => void;
  onAssistantText: (text: string) => void;
  onProcessing: (processing: boolean) => void;
  onError: (message: string) => void;
  onClosed: () => void;
}

/** Half-duplex speech-to-text, chat agent and text-to-speech over /api/v1/voice/stream. */
export class PipelineVoiceSession {
  private socket: WebSocket | null = null;
  private stream: MediaStream | null = null;
  private capture: AudioContext | null = null;
  private playback: AudioContext | null = null;
  private processor: ScriptProcessorNode | null = null;
  private sources = new Set<AudioBufferSourceNode>();
  private nextPlaybackTime = 0;
  private sampleRate = 0;
  private ready = false;
  private processing = false;
  private assistantText = '';
  private disposed = false;

  constructor(private options: PipelineVoiceOptions) {}

  private fail(message: string) {
    if (this.disposed) return;
    this.options.onError(message);
    this.dispose();
  }

  private setProcessing(processing: boolean) {
    this.processing = processing;
    this.options.onProcessing(processing);
  }

  private clearPlayback() {
    this.sources.forEach(source => {
      source.onended = null;
      source.stop();
      source.disconnect();
    });
    this.sources.clear();
    this.nextPlaybackTime = 0;
  }

  async start(): Promise<void> {
    try {
      // Resume playback during the user gesture, not on a network callback.
      const AudioContextClass = window.AudioContext || (window as any).webkitAudioContext;
      const playback = new AudioContextClass();
      this.playback = playback;
      await playback.resume();
      if (this.disposed) return;
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true, channelCount: 1 },
      });
      if (this.disposed) {
        stream.getTracks().forEach(track => track.stop());
        return;
      }
      this.stream = stream;
      const capture = new AudioContextClass({ sampleRate: 16000 });
      this.capture = capture;
      const encodePcm = createVoicePcmEncoder(capture.sampleRate);
      const source = capture.createMediaStreamSource(stream);
      const processor = capture.createScriptProcessor(1024, 1, 1);
      this.processor = processor;
      processor.onaudioprocess = (event: AudioProcessingEvent) => {
        const socket = this.socket;
        if (!this.ready || this.processing || this.sources.size
          || !socket || socket.readyState !== WebSocket.OPEN) return;
        const pcm = encodePcm(event.inputBuffer.getChannelData(0));
        if (pcm.length) socket.send(pcm.buffer);
      };
      source.connect(processor);
      processor.connect(capture.destination);

      const options = this.options;
      const params = new URLSearchParams({
        session_id: String(options.sessionId),
        agent_type: options.agentType || 'online',
        stt_provider: options.sttProvider || 'mms',
        tts_provider: options.ttsProvider || 'sesame',
      });
      for (const [key, value] of Object.entries({
        tts_model: options.ttsModel,
        tts_voice: options.ttsVoice,
        tts_language: options.ttsLanguage,
      })) {
        if (value !== undefined) params.set(key, String(value));
      }
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const socket = new WebSocket(`${protocol}//${window.location.host}/api/v1/voice/stream?${params}`);
      socket.binaryType = 'arraybuffer';
      this.socket = socket;
      let messages = Promise.resolve();
      socket.onmessage = event => {
        // Preserve frame order across Blob conversion; dispose invalidates
        // pending callbacks before they can create any more audio.
        messages = messages
          .then(() => this.handleMessage(event.data))
          .catch(error => this.fail(error instanceof Error ? error.message : 'Voice playback failed'));
        return messages;
      };
      socket.onerror = () => this.fail('Voice connection failed');
      socket.onclose = () => this.dispose();
    } catch (error) {
      this.fail(error instanceof Error ? error.message : 'Microphone unavailable');
    }
  }

  private async handleMessage(raw: unknown): Promise<void> {
    if (this.disposed) return;
    const data = raw instanceof Blob ? await raw.arrayBuffer() : raw;
    if (this.disposed || !this.playback) return;
    if (data instanceof ArrayBuffer) {
      this.play(this.playback, data);
      return;
    }
    const message = JSON.parse(data as string);
    switch (message.type) {
      case 'ready':
        this.ready = true;
        this.options.onReady();
        break;
      case 'transcript_partial':
        this.options.onUserText(message.text || '');
        break;
      case 'transcript_final':
        this.options.onUserText(message.text || '');
        break;
      case 'processing':
      case 'text_start':
        this.setProcessing(true);
        this.assistantText = '';
        this.options.onAssistantText('');
        break;
      case 'text_chunk':
        this.assistantText = (this.assistantText + (message.text || '')).slice(-4000);
        this.options.onAssistantText(this.assistantText);
        break;
      case 'text_complete':
        this.assistantText = message.text || '';
        this.options.onAssistantText(this.assistantText);
        break;
      case 'audio_start':
        if (message.encoding !== 'pcm_s16le' || message.channels !== 1
          || !Number.isInteger(message.sample_rate) || message.sample_rate < 8000
          || message.sample_rate > 96000) throw new Error('Unsupported voice audio format');
        this.sampleRate = message.sample_rate;
        // Do not rewind: audio from the preceding turn may still be audible.
        break;
      case 'reset_complete':
        this.clearPlayback();
        this.sampleRate = 0;
        this.setProcessing(false);
        this.options.onUserText('');
        break;
      case 'done':
        this.setProcessing(false);
        break;
      case 'error':
        if (message.fatal) throw new Error(message.message || 'Voice processing failed');
        this.clearPlayback();
        this.setProcessing(false);
        this.options.onError(message.message || 'Voice processing failed');
        break;
    }
  }

  private play(playback: AudioContext, data: ArrayBuffer) {
    if (!this.sampleRate || data.byteLength % 2) throw new Error('Invalid voice PCM frame');
    if (!data.byteLength) return;
    const pcm = new Int16Array(data);
    const buffer = playback.createBuffer(1, pcm.length, this.sampleRate);
    const samples = buffer.getChannelData(0);
    for (let i = 0; i < pcm.length; i++) samples[i] = pcm[i] / 32768;
    const source = playback.createBufferSource();
    source.buffer = buffer;
    source.connect(playback.destination);
    this.sources.add(source);
    source.onended = () => {
      this.sources.delete(source);
      source.disconnect();
    };
    const startAt = Math.max(playback.currentTime + 0.02, this.nextPlaybackTime);
    source.start(startAt);
    this.nextPlaybackTime = startAt + buffer.duration;
  }

  close() {
    if (this.socket?.readyState === WebSocket.OPEN) this.socket.send(JSON.stringify({ type: 'stop' }));
    this.dispose();
  }

  dispose() {
    if (this.disposed) return;
    this.disposed = true;
    this.ready = false;
    const socket = this.socket;
    this.socket = null;
    if (socket) {
      socket.onmessage = null;
      socket.onclose = null;
      socket.onerror = null;
      socket.close();
    }
    if (this.processor) {
      this.processor.onaudioprocess = null;
      this.processor.disconnect();
      this.processor = null;
    }
    this.stream?.getTracks().forEach(track => track.stop());
    this.stream = null;
    this.clearPlayback();
    for (const context of [this.capture, this.playback]) {
      if (context) void context.close();
    }
    this.capture = null;
    this.playback = null;
    this.options.onClosed();
  }
}
