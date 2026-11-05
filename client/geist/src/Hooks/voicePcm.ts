// Stateful area averaging preserves timing across capture blocks, including 44.1 kHz.
export function createVoicePcmEncoder(sampleRate: number) {
  if (!Number.isFinite(sampleRate) || sampleRate < 16000) {
    throw new Error('Voice requires browser audio capture at 16 kHz or higher. Try another browser or audio device.');
  }
  const ratio = sampleRate / 16000;
  let sum = 0;
  let weight = 0;
  return (input: Float32Array): Int16Array => {
    const samples: number[] = [];
    for (let index = 0; index < input.length; index++) {
      const value = input[index];
      const sample = Math.max(-1, Math.min(1, value));
      let remaining = 1;
      while (remaining > 1e-9) {
        const portion = Math.min(remaining, ratio - weight);
        sum += sample * portion;
        weight += portion;
        remaining -= portion;
        if (weight >= ratio - 1e-9) {
          const average = sum / ratio;
          samples.push(average * (average < 0 ? 32768 : 32767));
          sum = 0;
          weight = 0;
        }
      }
    }
    return Int16Array.from(samples);
  };
}
