import { readFileSync } from 'fs';
import { resolve } from 'path';
import { runInNewContext } from 'vm';

const source = readFileSync(resolve(__dirname, '../../../public/audio/local-live-processor.js'), 'utf8');

function processor(frameSamples: number) {
  let Processor: any;
  class AudioWorkletProcessor {
    port = { onmessage: null as any, postMessage: jest.fn() };
  }
  runInNewContext(source, { AudioWorkletProcessor, Float32Array,
    registerProcessor: (name: string, value: any) => { Processor = value; } });
  return new Processor({ processorOptions: { frameSamples } });
}

it.each([320, 1920, 4800])('captures and plays exact %i-sample frames across browser render blocks', frameSamples => {
  const worklet = processor(frameSamples);
  const input = new Float32Array(128).fill(0.25);
  worklet.process([[input]], [[new Float32Array(128)]]);
  expect(worklet.port.postMessage).not.toHaveBeenCalled();
  worklet.port.onmessage({ data: { type: 'ready' } });
  worklet.port.onmessage({ data: { type: 'audio', pcm: new Float32Array(frameSamples).fill(0.5).buffer } });
  const output: number[] = [];
  for (let i = 0; i < Math.ceil(frameSamples * 2 / 128); i++) {
    const block = new Float32Array(128);
    worklet.process([[input]], [[block]]);
    output.push(...Array.from(block));
  }
  const audio = worklet.port.postMessage.mock.calls.filter(([event]: any[]) => event.type === 'audio');
  expect(audio).toHaveLength(2);
  for (const [event] of audio) {
    const samples = Array.from(new Float32Array(event.pcm));
    expect(samples).toHaveLength(frameSamples);
    expect(samples.every(value => value === 0.25)).toBe(true);
  }
  expect(output.slice(0, frameSamples).every(value => value === 0.5)).toBe(true);
  expect(output.slice(frameSamples).every(value => value === 0)).toBe(true);
});

it('reports overrun and discards delayed playback', () => {
  const worklet = processor(320);
  worklet.port.onmessage({ data: { type: 'ready' } });
  for (let i = 0; i < 11; i++) {
    worklet.port.onmessage({ data: { type: 'audio', pcm: new Float32Array(320).fill(0.5).buffer } });
  }
  expect(worklet.port.postMessage).toHaveBeenCalledWith({ type: 'overrun' });
  const output = new Float32Array(128);
  worklet.process([], [[output]]);
  expect(Array.from(output).every(value => value === 0)).toBe(true);
});
