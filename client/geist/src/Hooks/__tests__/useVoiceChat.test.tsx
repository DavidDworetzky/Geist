import { act, renderHook } from '@testing-library/react';
import useVoiceChat from '../useVoiceChat';


class FakeWebSocket {
  static OPEN = 1;
  static instances: FakeWebSocket[] = [];

  readyState = FakeWebSocket.OPEN;
  onopen: (() => void) | null = null;
  onmessage: ((event: MessageEvent) => Promise<void>) | null = null;
  onerror: ((event: Event) => void) | null = null;
  onclose: ((event: CloseEvent) => void) | null = null;
  send = jest.fn();
  close = jest.fn();

  constructor(public url: string) {
    FakeWebSocket.instances.push(this);
  }
}

class FakeAudioContext {
  static instances: FakeAudioContext[] = [];

  currentTime = 1;
  sampleRate = 16000;
  destination = {} as AudioDestinationNode;
  close = jest.fn();
  resume = jest.fn().mockResolvedValue(undefined);
  createBuffer = jest.fn((channels: number, length: number, sampleRate: number) => ({
    duration: length / sampleRate,
    getChannelData: () => new Float32Array(length),
  }));
  createBufferSource = jest.fn(() => ({
    buffer: null,
    connect: jest.fn(),
    start: jest.fn(),
    stop: jest.fn(),
    disconnect: jest.fn(),
    onended: null,
  }));
  createMediaStreamSource = jest.fn(() => ({ connect: jest.fn() }));
  createScriptProcessor = jest.fn(() => ({
    connect: jest.fn(),
    disconnect: jest.fn(),
    onaudioprocess: null,
  }));

  constructor(_options?: AudioContextOptions) {
    FakeAudioContext.instances.push(this);
  }
}

describe('useVoiceChat audio playback contract', () => {
  beforeEach(() => {
    FakeWebSocket.instances = [];
    FakeAudioContext.instances = [];
    Object.defineProperty(global, 'WebSocket', { value: FakeWebSocket, configurable: true });
    Object.defineProperty(window, 'AudioContext', {
      value: FakeAudioContext,
      configurable: true,
    });
    Object.defineProperty(navigator, 'mediaDevices', {
      value: {
        getUserMedia: jest.fn().mockResolvedValue({
          getTracks: () => [{ stop: jest.fn() }],
        }),
      },
      configurable: true,
    });
  });

  it('uses audio_start sample-rate metadata for streamed PCM playback', async () => {
    const { result, unmount } = renderHook(() => useVoiceChat({
      sessionId: 7,
      ttsProvider: 'magpie',
      ttsModel: 'nvidia/magpie_tts_multilingual_357m',
      ttsVoice: 'John',
      ttsLanguage: 'en-US',
    }));

    await act(async () => {
      await result.current.startRecording();
    });
    const socket = FakeWebSocket.instances[0];
    expect(socket.url).toContain('tts_provider=magpie');

    await act(async () => {
      await socket.onmessage?.({
        data: JSON.stringify({
          type: 'audio_start',
          encoding: 'pcm_s16le',
          channels: 1,
          sample_rate: 22050,
        }),
      } as MessageEvent);
      const pcm = new Blob([new Int16Array([0, 1000, -1000]).buffer]);
      Object.defineProperty(pcm, 'arrayBuffer', {
        value: async () => new Int16Array([0, 1000, -1000]).buffer,
      });
      await socket.onmessage?.({ data: pcm } as MessageEvent);
    });

    const playbackContext = FakeAudioContext.instances[0];
    expect(playbackContext.createBuffer).toHaveBeenCalledWith(1, 3, 22050);
    expect(playbackContext.createBufferSource.mock.results[0].value.start).toHaveBeenCalled();

    unmount();
  });

  it('schedules successive turns without overlap and stops every scheduled source', async () => {
    const { result, unmount } = renderHook(() => useVoiceChat({ sessionId: 7 }));
    await act(async () => { await result.current.startRecording(); });
    const socket = FakeWebSocket.instances[0];
    const playback = FakeAudioContext.instances[0];
    await act(async () => {
      for (let turn = 0; turn < 2; turn++) {
        await socket.onmessage?.({ data: JSON.stringify({
          type: 'audio_start', encoding: 'pcm_s16le', channels: 1, sample_rate: 24000,
        }) } as MessageEvent);
        await socket.onmessage?.({ data: new Int16Array(72000).buffer } as MessageEvent);
      }
    });
    const sources = playback.createBufferSource.mock.results.map(result => result.value);
    expect(sources[0].start).toHaveBeenCalledWith(1.02);
    expect(sources[1].start).toHaveBeenCalledWith(4.02);
    act(() => result.current.stopRecording());
    sources.forEach(source => expect(source.stop).toHaveBeenCalledTimes(1));
    expect(result.current.isProcessing).toBe(false);
    unmount();
  });

  it('drops a Blob that resolves after stop instead of restarting playback', async () => {
    const { result, unmount } = renderHook(() => useVoiceChat({ sessionId: 7 }));
    await act(async () => { await result.current.startRecording(); });
    const socket = FakeWebSocket.instances[0];
    let resolve!: (data: ArrayBuffer) => void;
    const blob = new Blob();
    Object.defineProperty(blob, 'arrayBuffer', { value: () => new Promise(r => { resolve = r; }) });
    let pending: Promise<void> | undefined;
    await act(async () => { pending = socket.onmessage?.({ data: blob } as MessageEvent); });
    act(() => result.current.stopRecording());
    await act(async () => { resolve(new Int16Array(100).buffer); await pending; });
    expect(FakeAudioContext.instances).toHaveLength(2);
    expect(FakeAudioContext.instances[0].createBufferSource).not.toHaveBeenCalled();
    unmount();
  });

  it('cleans up recording when the socket closes unexpectedly', async () => {
    const { result, unmount } = renderHook(() => useVoiceChat({ sessionId: 7 }));
    await act(async () => { await result.current.startRecording(); });
    act(() => FakeWebSocket.instances[0].onclose?.({ code: 1006 } as CloseEvent));
    expect(result.current.isRecording).toBe(false);
    FakeAudioContext.instances.forEach(context => expect(context.close).toHaveBeenCalled());
    unmount();
  });
});
