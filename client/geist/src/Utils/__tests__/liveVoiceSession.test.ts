import { LiveVoiceSession } from '../liveVoiceSession';

class FakeChannel {
  readyState = 'open';
  onmessage: any;
  onclose: any;
  onerror: any;
  send = jest.fn();
  close = jest.fn();
  emit(event: object) { this.onmessage?.({ data: JSON.stringify(event) }); }
}

class FakePeer {
  static latest: FakePeer;
  channel = new FakeChannel();
  iceGatheringState = 'complete';
  localDescription = { sdp: 'v=0\r\n' };
  ontrack: any;
  onconnectionstatechange: any;
  addTrack = jest.fn();
  createDataChannel = jest.fn(() => this.channel);
  createOffer = jest.fn(async () => this.localDescription);
  setLocalDescription = jest.fn(async () => {});
  setRemoteDescription = jest.fn(async () => {});
  close = jest.fn();
  constructor() { FakePeer.latest = this; }
}

const flush = async () => { for (let i = 0; i < 15; i++) await Promise.resolve(); };

describe('GPT-Live browser session', () => {
  let track: { stop: jest.Mock; enabled: boolean };
  let options: any;
  let session: LiveVoiceSession;
  beforeEach(() => {
    jest.useFakeTimers();
    Object.assign(global, { RTCPeerConnection: FakePeer });
    track = { stop: jest.fn(), enabled: true };
    Object.defineProperty(navigator, 'mediaDevices', { configurable: true, value: {
      getUserMedia: jest.fn(async () => ({ getTracks: () => [track], getAudioTracks: () => [track] }))
    } });
    jest.spyOn(HTMLMediaElement.prototype, 'pause').mockImplementation(() => {});
    global.fetch = jest.fn(async () => ({ ok: true, json: async () => ({ session_id: 'live_1', sdp: 'answer' }) })) as any;
    options = { onReady: jest.fn(), onTranscript: jest.fn(), onError: jest.fn(), onClosed: jest.fn() };
    session = new LiveVoiceSession(options);
  });
  afterEach(() => { session.dispose(); jest.useRealTimers(); jest.restoreAllMocks(); });

  it('negotiates media, waits for session.started, and never sends session.start', async () => {
    await session.start();
    expect(FakePeer.latest.createDataChannel).toHaveBeenCalledWith('oai-events');
    expect(FakePeer.latest.setRemoteDescription).toHaveBeenCalledWith({ type: 'answer', sdp: 'answer' });
    expect(options.onReady).not.toHaveBeenCalled();
    FakePeer.latest.channel.emit({ type: 'session.started' });
    expect(options.onReady).toHaveBeenCalledTimes(1);
    expect(FakePeer.latest.channel.send).not.toHaveBeenCalled();
    jest.advanceTimersByTime(46000);
    expect(options.onError).not.toHaveBeenCalled();
  });

  it('forwards captions independently without a text-agent request', async () => {
    await session.start();
    const channel = FakePeer.latest.channel;
    channel.emit({ type: 'session.started' });
    channel.emit({ type: 'session.input_transcript.delta', delta: 'Hello ' });
    channel.emit({ type: 'session.output_transcript.delta', delta: 'Hi' });
    channel.emit({ type: 'session.delegation.created', delegation: { target: 'client', id: 'item_1' } });
    await flush();
    expect(options.onTranscript.mock.calls).toEqual([['user', 'Hello '], ['assistant', 'Hi']]);
    expect(global.fetch).toHaveBeenCalledTimes(1);
    expect(channel.send.mock.calls[0][0]).toContain('session.thinking.append');
  });

  it('keeps media alive until graceful closure is confirmed', async () => {
    await session.start();
    const channel = FakePeer.latest.channel;
    channel.emit({ type: 'session.started' });
    session.close();
    await flush();
    expect(track.enabled).toBe(false);
    expect(channel.send).toHaveBeenCalledWith(JSON.stringify({ type: 'session.close' }));
    expect(track.stop).not.toHaveBeenCalled();
    channel.emit({ type: 'session.closed', usage: { seconds: 2 } });
    expect(track.stop).toHaveBeenCalledTimes(1);
    expect(FakePeer.latest.close).toHaveBeenCalledTimes(1);
    expect(options.onClosed).toHaveBeenCalledTimes(1);
  });

  it('stops a late microphone grant after cancellation', async () => {
    let grant: any;
    (navigator.mediaDevices.getUserMedia as jest.Mock).mockReturnValue(new Promise(resolve => { grant = resolve; }));
    const starting = session.start();
    session.close();
    grant({ getTracks: () => [track] });
    await starting;
    expect(track.stop).toHaveBeenCalledTimes(1);
    expect(global.fetch).not.toHaveBeenCalled();
  });

  it('releases microphone on provider failure and reports the safe error', async () => {
    (global.fetch as jest.Mock).mockResolvedValueOnce({ ok: false, json: async () => ({ detail: 'GPT-Live access required.' }) });
    await session.start();
    expect(options.onError).toHaveBeenCalledWith('GPT-Live access required.');
    expect(track.stop).toHaveBeenCalled();
    expect(options.onClosed).toHaveBeenCalled();
  });

  it('times out a connection that never starts', async () => {
    await session.start();
    jest.advanceTimersByTime(45000);
    expect(options.onError).toHaveBeenCalledWith('GPT-Live connection timed out.');
    expect(track.stop).toHaveBeenCalled();
  });

  it('cleans up when graceful closure times out', async () => {
    await session.start();
    FakePeer.latest.channel.emit({ type: 'session.started' });
    session.close();
    jest.advanceTimersByTime(15000);
    expect(track.stop).toHaveBeenCalled();
    expect(options.onError).toHaveBeenCalledWith('GPT-Live did not confirm closure before the timeout.');
  });
});
