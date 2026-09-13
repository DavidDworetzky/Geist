import { DictationSession } from '../dictationSession';
import { MoshiVoiceSession } from '../moshiVoiceSession';

class FakeProcessor {
  static latest: FakeProcessor;
  port = { onmessage: null as any, postMessage: jest.fn() };
  connect = jest.fn();
  disconnect = jest.fn();
  constructor() { FakeProcessor.latest = this; }
  emit(data: any) { this.port.onmessage?.({ data }); }
}
class FakeContext {
  audioWorklet = { addModule: jest.fn(async () => {}) };
  destination = {};
  createMediaStreamSource = () => ({ connect: jest.fn() });
  resume = jest.fn(async () => {});
  close = jest.fn(async () => {});
}
class FakeSocket {
  static latest: FakeSocket;
  static OPEN = 1;
  readyState = 1;
  bufferedAmount = 0;
  onmessage: any;
  onerror: any;
  onclose: any;
  send = jest.fn();
  close = jest.fn();
  constructor() { FakeSocket.latest = this; }
  emit(data: any) { this.onmessage?.({ data: typeof data === 'object' && !(data instanceof ArrayBuffer) ? JSON.stringify(data) : data }); }
}
const flush = async () => { for (let i = 0; i < 12; i++) await Promise.resolve(); };

describe('isolated audio sessions', () => {
  let options: any;
  let stop: jest.Mock;
  let session: DictationSession | MoshiVoiceSession;
  beforeEach(() => {
    jest.useFakeTimers();
    Object.assign(global, { AudioContext: FakeContext, AudioWorkletNode: FakeProcessor, WebSocket: FakeSocket });
    stop = jest.fn();
    Object.defineProperty(navigator, 'mediaDevices', { configurable: true, value: {
      getUserMedia: jest.fn(async () => ({ getTracks: () => [{ stop }] }))
    } });
    global.fetch = jest.fn(async () => ({ ok: true, json: async () => ({ text: 'Draft text' }) })) as any;
    options = { provider: 'whisper', onReady: jest.fn(), onProcessing: jest.fn(), onText: jest.fn(),
      onTranscript: jest.fn(), onAudioLevel: jest.fn(), onError: jest.fn(), onClosed: jest.fn() };
  });
  afterEach(() => { session?.dispose(); jest.useRealTimers(); jest.restoreAllMocks(); });

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

  it('Moshi sends audio over its socket and renders independent voice captions', async () => {
    session = new MoshiVoiceSession(options);
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
    expect(global.fetch).not.toHaveBeenCalled();
    session.close();
    expect(socket.close).toHaveBeenCalledTimes(1);
    expect(stop).toHaveBeenCalledTimes(1);
  });

  it('Moshi closes stalled audio before latency grows without bound', async () => {
    session = new MoshiVoiceSession(options);
    await session.start();
    FakeSocket.latest.bufferedAmount = 80000;
    FakeProcessor.latest.emit({ type: 'audio', pcm: new ArrayBuffer(7680) });
    expect(options.onError).toHaveBeenCalledWith(expect.stringContaining('cannot keep up'));
    expect(FakeSocket.latest.send).not.toHaveBeenCalled();
    expect(stop).toHaveBeenCalled();
  });

  it('Moshi cancels model loading and ignores a late ready event', async () => {
    session = new MoshiVoiceSession(options);
    await session.start();
    session.close();
    FakeSocket.latest.emit({ type: 'ready' });
    jest.advanceTimersByTime(120000);
    expect(options.onReady).not.toHaveBeenCalled();
    expect(options.onError).not.toHaveBeenCalled();
    expect(options.onClosed).toHaveBeenCalledTimes(1);
  });
});
