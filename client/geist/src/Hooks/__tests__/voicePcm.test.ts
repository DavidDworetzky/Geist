import { createVoicePcmEncoder } from '../voicePcm';

it.each([16000, 44100, 48000])('encodes %i Hz capture as continuous 16 kHz PCM', rate => {
  const encode = createVoicePcmEncoder(rate);
  const input = new Float32Array(rate).fill(0.5);
  const output: number[] = [];
  for (let offset = 0; offset < input.length; offset += 1024) {
    output.push(...Array.from(encode(input.slice(offset, offset + 1024))));
  }
  expect(output).toHaveLength(16000);
  expect(output.every(sample => sample === 16383)).toBe(true);
});

it('clips PCM amplitudes and rejects unsupported capture rates', () => {
  expect(Array.from(createVoicePcmEncoder(16000)(new Float32Array([-2, 0, 2]))))
    .toEqual([-32768, 0, 32767]);
  expect(() => createVoicePcmEncoder(8000)).toThrow('Try another browser');
});
