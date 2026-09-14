import { DictationSession } from '../dictationSession';
import { LocalVoiceSession } from '../localVoiceSession';

class FakeProcessor {
  static options: any;
  static latest: FakeProcessor;
  port = { onmessage: null as any, postMessage: jest.fn() };
  connect = jest.fn();
  disconnect = jest.fn();
  constructor(context: any, name: string, options: any) { FakeProcessor.latest = this; FakeProcessor.options = options; }
  emit(data: any) { this.port.onmessage?.({ data }); }
}
class FakeContext {
  static actualRate: number | undefined;
  static latest: FakeContext;
  sampleRate: number;
  constructor(options: AudioContextOptions) {
    this.sampleRate = FakeContext.actualRate ?? options.sampleRate!;
    FakeContext.latest = this;
  }
  audioWorklet = { addModule: jest.fn(async () => {}) };
  destination = {};
  createMediaStreamSource = () => ({ connect: jest.fn() });
  resume = jest.fn(async () => {});
  close = jest.fn(async () => {});
}
class FakeSocket {
  static latest: FakeSocket;
  static OPEN = 1;
  url: string;
  readyState = 1;
  bufferedAmount = 0;
  onmessage: any;
  onerror: any;
  onclose: any;
  send = jest.fn();
  close = jest.fn();
  constructor(url: string) { this.url = url; FakeSocket.latest = this; }
  emit(data: any) { this.onmessage?.({ data: typeof data === 'object' && !(data instanceof ArrayBuffer) ? JSON.stringify(data) : data }); }
}
const flush = async () => { for (let i = 0; i < 12; i++) await Promise.resolve(); };

describe('isolated audio sessions', () => {
  let options: any;
  let stop: jest.Mock;
  let session: DictationSession | LocalVoiceSession;
  beforeEach(() => {
    jest.useFakeTimers();
    FakeContext.actualRate = undefined;
    Object.assign(global, { AudioContext: FakeContext, AudioWorkletNode: FakeProcessor, WebSocket: FakeSocket });
    stop = jest.fn();
    Object.defineProperty(navigator, 'mediaDevices', { configurable: true, value: {
      getUserMedia: jest.fn(async () => ({ getTracks: () => [{ stop }] }))
    } });
    global.fetch = jest.fn(async (url) => ({ ok: true, json: async () => url === '/api/v1/voice/live/local' ? { protocol: 'geist-pcm-v1', endpoint: '/api/v1/voice/live/local', sample_rate: 24000, frame_samples: 1920 } : { text: 'Draft text' } })) as any;
    options = { provider: 'whisper', onReady: jest.fn(), onProcessing: jest.fn(), onText: jest.fn(),
      onTranscript: jest.fn(), onAudioLevel: jest.fn(), onError: jest.fn(), onClosed: jest.fn() };
  });
  afterEach(() => { session?.dispose(); jest.useRealTimers(); jest.restoreAllMocks(); });

  it.each([
    ['Dictation', DictationSession, 16000],
    ['Local voice', LocalVoiceSession, 24000],
  ] as const)('%s rejects a hardware sample-rate fallback and releases the microphone', async (name, Session, rate) => {
    FakeContext.actualRate = 48000;
    const previousSocket = FakeSocket.latest;
    session = new Session(options);
    await session.start();
    expect(options.onError).toHaveBeenCalledWith(expect.stringContaining(`${name} requires ${rate} Hz audio`));
    expect(options.onReady).not.toHaveBeenCalled();
    expect(options.onClosed).toHaveBeenCalledTimes(1);
    expect(stop).toHaveBeenCalledTimes(1);
    expect(FakeContext.latest.close).toHaveBeenCalledTimes(1);
    expect(FakeContext.latest.audioWorklet.addModule).not.toHaveBeenCalled();
    expect(FakeSocket.latest).toBe(previousSocket);
    expect(global.fetch).toHaveBeenCalledTimes(name === 'Dictation' ? 0 : 1);
  });

  it('dictation stops the mic before transcribing and never submits a chat turn', async () => {
    session = new DictationSession(options);
    await session.start();
    FakeProcessor.latest.emit({ type: 'audio', pcm: new Int16Array([1, -1]).buffer });
    session.stop();
    expect(global.fetch).not.toHaveBeenCalled();
    FakeProcessor.latest.emit({ type: 'stopped' });
    expect(stop).toHaveBeenCalledTimes(1);
    await flush();
    expect(global.fetch).toHaveBeenCalledTimes(1);
    expect((global.fetch as jest.Mock).mock.calls[0][0]).toBe('/api/v1/voice/transcribe?provider=whisper');
    expect(options.onText).toHaveBeenCalledWith('Draft text');
    expect(options.onClosed).toHaveBeenCalledTimes(1);
  });

  it('discards late dictation results after leaving the conversation', async () => {
    let finish: any;
    (global.fetch as jest.Mock).mockReturnValue(new Promise(resolve => { finish = resolve; }));
    session = new DictationSession(options);
    await session.start();
    session.stop();
    FakeProcessor.latest.emit({ type: 'stopped' });
    session.dispose();
    finish({ ok: true, json: async () => ({ text: 'Late text' }) });
    await flush();
    expect(options.onText).not.toHaveBeenCalled();
    expect((global.fetch as jest.Mock).mock.calls[0][1].signal.aborted).toBe(true);
  });


  it('uses the configured audio rate and frame size for another local engine', async () => {
    (global.fetch as jest.Mock).mockResolvedValue({ ok: true, json: async () => ({ protocol: 'geist-pcm-v1', endpoint: '/api/v1/voice/live/local', sample_rate: 16000, frame_samples: 320 }) });
    session = new LocalVoiceSession(options);
    await session.start();
    expect(FakeContext.latest.sampleRate).toBe(16000);
    expect(FakeProcessor.options.processorOptions.frameSamples).toBe(320);
    FakeSocket.latest.emit(new ArrayBuffer(1280));
    expect(FakeProcessor.latest.port.postMessage).toHaveBeenCalled();
    FakeSocket.latest.emit(new ArrayBuffer(7680));
    expect(options.onError).toHaveBeenCalledWith('Invalid local voice audio frame.');
    expect(stop).toHaveBeenCalledTimes(1);
  });

  it('rejects an unsupported transport before acquiring the microphone', async () => {
    (global.fetch as jest.Mock).mockResolvedValue({ ok: true, json: async () => ({ protocol: 'opus' }) });
    session = new LocalVoiceSession(options);
    await session.start();
    expect(navigator.mediaDevices.getUserMedia).not.toHaveBeenCalled();
    expect(options.onError).toHaveBeenCalledWith('Unsupported local voice audio configuration.');
  });

  it('cancels a pending configuration request before accessing the microphone', async () => {
    let finish: any;
    (global.fetch as jest.Mock).mockReturnValue(new Promise(resolve => { finish = resolve; }));
    session = new LocalVoiceSession(options);
    const starting = session.start();
    session.dispose();
    finish({ ok: true, json: async () => ({}) });
    await starting;
    expect((global.fetch as jest.Mock).mock.calls[0][1].signal.aborted).toBe(true);
    expect(navigator.mediaDevices.getUserMedia).not.toHaveBeenCalled();
    expect(options.onError).not.toHaveBeenCalled();
  });

  it('Local voice sends audio over its socket and renders independent voice captions', async () => {
    session = new LocalVoiceSession(options);
    await session.start();
    const socket = FakeSocket.latest;
    expect(options.onReady).not.toHaveBeenCalled();
    socket.emit({ type: 'ready' });
    expect(FakeProcessor.latest.port.postMessage).toHaveBeenCalledWith({ type: 'ready' });
    const pcm = new Float32Array(1920).buffer;
    FakeProcessor.latest.emit({ type: 'audio', pcm });
    expect(socket.send).toHaveBeenCalledWith(pcm);
    socket.emit(pcm);
    expect(FakeProcessor.latest.port.postMessage).toHaveBeenCalledWith({ type: 'audio', pcm }, [pcm]);
    socket.emit({ type: 'transcript', text: ' Hello' });
    expect(options.onTranscript).toHaveBeenCalledWith('assistant', ' Hello');
    expect(socket.url).toBe('ws://localhost/api/v1/voice/live/local');
    expect(global.fetch).toHaveBeenCalledTimes(1);
    socket.emit({ type: 'transcript', role: 'user', text: 'Hi' });
    expect(options.onTranscript).toHaveBeenCalledWith('user', 'Hi');
    session.close();
    expect(socket.close).toHaveBeenCalledTimes(1);
    expect(stop).toHaveBeenCalledTimes(1);
  });

  it('Local voice closes stalled audio before latency grows without bound', async () => {
    session = new LocalVoiceSession(options);
    await session.start();
    FakeSocket.latest.bufferedAmount = 80000;
    FakeProcessor.latest.emit({ type: 'audio', pcm: new ArrayBuffer(7680) });
    expect(options.onError).toHaveBeenCalledWith(expect.stringContaining('cannot keep up'));
    expect(FakeSocket.latest.send).not.toHaveBeenCalled();
    expect(stop).toHaveBeenCalled();
  });

  it('Local voice cancels model loading and ignores a late ready event', async () => {
    session = new LocalVoiceSession(options);
    await session.start();
    session.close();
    FakeSocket.latest.emit({ type: 'ready' });
    jest.advanceTimersByTime(120000);
    expect(options.onReady).not.toHaveBeenCalled();
    expect(options.onError).not.toHaveBeenCalled();
    expect(options.onClosed).toHaveBeenCalledTimes(1);
  });
});
