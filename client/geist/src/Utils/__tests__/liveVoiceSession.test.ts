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

  it('delegates transcript context once and speaks the confirmed result with the same ID', async () => {
    options.onDelegate = jest.fn(async () => 'Found three matching files.');
    await session.start();
    expect(JSON.parse((global.fetch as jest.Mock).mock.calls[0][1].body).tools_enabled).toBe(true);
    const channel = FakePeer.latest.channel;
    channel.emit({ type: 'session.started' });
    channel.emit({ type: 'session.input_transcript.delta', delta: 'Find the report' });
    channel.emit({ type: 'session.output_transcript.delta', delta: 'Which project?' });
    channel.emit({ type: 'session.input_transcript.delta', delta: 'Geist, please.' });
    const event = { type: 'session.delegation.created', delegation: { target: 'client', id: 'task1' } };
    channel.emit(event);
    channel.emit(event);
    await flush();
    channel.emit({ ...event, delegation: { target: 'client', id: 'task2' } });
    await flush();
    expect(options.onDelegate).toHaveBeenCalledTimes(1);
    expect(options.onDelegate.mock.calls[0][0]).toBe('user: Find the report\nassistant: Which project?\nuser: Geist, please.');
    expect(channel.send).toHaveBeenCalledWith(JSON.stringify({ type: 'session.commentary.append',
      delegation_id: 'task1', content: 'Found three matching files.' }));
  });

  it('requires a user transcript and rejects a second task while work is pending', async () => {
    options.onDelegate = jest.fn(() => new Promise(() => {}));
    await session.start();
    const channel = FakePeer.latest.channel;
    channel.emit({ type: 'session.started' });
    channel.emit({ type: 'session.delegation.created', delegation: { target: 'client', id: 'empty' } });
    expect(options.onDelegate).not.toHaveBeenCalled();
    channel.emit({ type: 'session.input_transcript.delta', delta: '   ' });
    channel.emit({ type: 'session.delegation.created', delegation: { target: 'client', id: 'blank' } });
    expect(options.onDelegate).not.toHaveBeenCalled();
    channel.emit({ type: 'session.input_transcript.delta', delta: 'Find files' });
    channel.emit({ type: 'session.delegation.created', delegation: { target: 'client', id: 'first' } });
    channel.emit({ type: 'session.input_transcript.delta', delta: ' and send them' });
    channel.emit({ type: 'session.delegation.created', delegation: { target: 'client', id: 'second' } });
    expect(options.onDelegate).toHaveBeenCalledTimes(1);
    expect(channel.send.mock.calls.at(-1)[0]).toContain('No second task was started');
  });

  it('aborts tool work as soon as hangup starts and suppresses a late result', async () => {
    let finish: (value: string) => void = () => {};
    options.onDelegate = jest.fn(() => new Promise<string>(resolve => { finish = resolve; }));
    await session.start();
    const channel = FakePeer.latest.channel;
    channel.emit({ type: 'session.started' });
    channel.emit({ type: 'session.input_transcript.delta', delta: 'Check the catalog' });
    channel.emit({ type: 'session.delegation.created', delegation: { target: 'client', id: 'task' } });
    const signal = options.onDelegate.mock.calls[0][1];
    session.close();
    expect(signal.aborted).toBe(true);
    finish('Late result');
    await flush();
    expect(channel.send.mock.calls.some(([data]: [string]) => JSON.parse(data).type === 'session.commentary.append')).toBe(false);
  });

  it('keeps the call alive on tool failure without exposing internal errors', async () => {
    options.onDelegate = jest.fn(async () => { throw new Error('private backend detail'); });
    await session.start();
    const channel = FakePeer.latest.channel;
    channel.emit({ type: 'session.started' });
    channel.emit({ type: 'session.input_transcript.delta', delta: 'Check files' });
    channel.emit({ type: 'session.delegation.created', delegation: { target: 'client', id: 'task' } });
    await flush();
    expect(channel.send.mock.calls.at(-1)[0]).toContain('did not finish successfully');
    expect(JSON.stringify(channel.send.mock.calls)).not.toContain('private backend detail');
    expect(options.onClosed).not.toHaveBeenCalled();
  });

  it('bounds Unicode result updates and keeps full details in the UI', async () => {
    options.onDelegate = jest.fn(async () => '🙂'.repeat(1000));
    await session.start();
    const channel = FakePeer.latest.channel;
    channel.emit({ type: 'session.started' });
    channel.emit({ type: 'session.input_transcript.delta', delta: 'Check files' });
    channel.emit({ type: 'session.delegation.created', delegation: { target: 'client', id: 'task' } });
    await flush();
    const events = channel.send.mock.calls.map(([data]: [string]) => JSON.parse(data))
      .filter(event => event.type === 'session.commentary.append');
    expect(events).toHaveLength(4);
    for (const event of events) expect(new Blob([event.content]).size).toBeLessThanOrEqual(400);
    expect(events[0].content).toBe('🙂'.repeat(100));
    expect(events.at(-1).content).toContain('full result');
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
