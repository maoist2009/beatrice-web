// Runs the production TS pipeline with ONNX Runtime Web WASM, not a copied implementation.
import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';
import * as ort from 'onnxruntime-web/wasm';
import 'fake-indexeddb/auto';
import { unzipSync } from 'fflate';
import { BetaProcessor } from '../src/lib/beta/runtime';
import { WaveformSynth } from '../src/lib/beta/synth';
import { collectFiles, exportParaphernaliaZip, parseParaphernalia, f16ToF32 } from '../src/lib/paraphernalia';
import { PitchFeatureExtractor, samplePitch } from '../src/lib/dsp';
import { rfft, irfft } from '../src/lib/fft';
import { saveModel, loadModel, listModels, deleteModel } from '../src/lib/cache';
import type { EngineParams } from '../src/lib/engine';

ort.env.wasm.numThreads = 1;
const params: EngineParams = { pitchShift: 0, averageSourcePitch: 0, intonationIntensity: 1, pitchCorrection: 0, pitchCorrectionType: 0, formantShift: 0, speaker: 0, inputGain: 1, outputGain: 1, monitor: false, minMidi: 33, maxMidi: 100, vqNeighbors: 0, convert: true };
function read(path: string) { const b = readFileSync(path); return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer; }
const reports: Record<string, unknown>[] = [];
function compare(label: string, a: Float32Array | Float64Array, b: Float32Array | Float64Array, tolerance: number, absTolerance = Infinity) {
  assert.equal(a.length, b.length, label);
  let ds = 0, bs = 0, max = 0;
  for (let i = 0; i < a.length; i++) { assert(Number.isFinite(a[i]) && Number.isFinite(b[i]), `${label}: nonfinite ${i}`); const d = a[i] - b[i]; ds += d * d; bs += b[i] * b[i]; max = Math.max(max, Math.abs(d)); }
  const rel = Math.sqrt(ds / Math.max(1e-24, bs));
  const result = { label, maxAbs: max, rmsRelative: rel };
  console.log('[production runtime]', JSON.stringify(result)); reports.push(result);
  assert(rel <= tolerance && max <= absTolerance, `${label} exceeds declared tolerance`);
}

for (const n of [480, 512, 560, 768]) {
  const input = new Float64Array(n); input[0] = 1;
  const re = new Float64Array(n / 2 + 1), im = new Float64Array(n / 2 + 1), back = new Float64Array(n);
  rfft(input, n, re, im);
  for (let k = 0; k < re.length; k++) { assert(Math.abs(re[k] - 1) < 1e-10); assert(Math.abs(im[k]) < 1e-10); }
  irfft(re, im, n, back); compare(`FFT ${n} impulse`, back, input, 1e-10);
}
assert.deepEqual([...f16ToF32(new Uint16Array([0x3c00, 0xc000, 0]).buffer, 0, 2)], [1, -2]);

for (const name of ['shigure', 'kurage']) {
  const raw = read(`.research/${name}.zip`);
  const files = await collectFiles([new File([raw], 'model.zip')]);
  const model = parseParaphernalia(files);
  assert.equal(model.format, 'beatrice-beta2'); assert.equal(model.speakers.nSpeakers, 1);
  assert.throws(() => parseParaphernalia({ ...files, formant_shift_embeddings: null }), /formant_shift/);
  assert.throws(() => parseParaphernalia({ ...files, toml: files.toml.replace('2.0.0-beta.1', 'rvc-v2') }), /Unsupported/);
  files.extras = { 'LICENSE-test.txt': new Blob(['retained attribution']) };
  await saveModel(`test-${name}`, model.name, model.version, files);
  const metadata = (await listModels()).find(m => m.key === `test-${name}`)!;
  assert(metadata && !('files' in metadata), 'Cache listing must not clone model weights');
  const reloaded = await loadModel(metadata.key);
  assert(reloaded);
  assert.equal(await reloaded.extras!['LICENSE-test.txt'].text(), 'retained attribution');
  assert.deepEqual(new Uint8Array(reloaded.formant_shift_embeddings!), new Uint8Array(files.formant_shift_embeddings!));
  await deleteModel(metadata.key); assert.equal(await loadModel(metadata.key), null);
  const exported = unzipSync(new Uint8Array(await (await exportParaphernaliaZip(files)).arrayBuffer()));
  assert.deepEqual(exported['formant_shift_embeddings.bin'], new Uint8Array(files.formant_shift_embeddings!));
  for (const key of ['phone_extractor', 'pitch_estimator', 'waveform_generator', 'speaker_embeddings'] as const) assert.deepEqual(exported[`${key}.bin`], new Uint8Array(files[key]));
  console.log('[ZIP round trip]', name, 'all original weight bytes equal');
  const golden = (key: string) => new Float32Array(read(`.research/golden-${name}/${key}.f32`));
  const input = golden('input'), count = input.length / 160;
  const reader = (i: number) => i < 0 || i >= input.length ? 0 : input[i];
  const fx = new PitchFeatureExtractor(), frame = new Float32Array(560), inst = new Float32Array(192 * count), corr = new Float32Array(256 * count), energy = new Float32Array(count);
  for (let f = 0; f < count; f++) {
    for (let i = 0; i < 560; i++) frame[i] = reader(f * 160 - 200 + i);
    const d = fx.process(frame); energy[f] = d.energy;
    for (let c = 0; c < 192; c++) inst[c * count + f] = d.instfreq[c];
    for (let c = 0; c < 256; c++) corr[c * count + f] = d.corr[c];
  }
  compare(`${name} DSP instfreq`, inst, golden('instfreq'), 1e-4);
  compare(`${name} DSP correlation`, corr, golden('corr_diff'), 1e-4);
  compare(`${name} DSP energy`, energy, golden('energy'), 1e-5);
  const logits = golden('pitch'), pitchFrame = new Float32Array(384), bins = new Float32Array(count), features = new Float32Array(3 * count);
  for (let f = 0; f < count; f++) {
    for (let c = 0; c < 384; c++) pitchFrame[c] = logits[c * count + f];
    const p = samplePitch(pitchFrame, 1, 383, 48); bins[f] = p.bin;
    features[f] = p.unvoiced; features[count + f] = p.half; features[2 * count + f] = p.dbl;
  }
  compare(`${name} band-48 indices`, bins, golden('quantized'), 0);
  compare(`${name} pitch features`, features, golden('features'), 1e-5);

  let seed = 12345;
  const random = () => { seed = (Math.imul(1664525, seed) + 1013904223) >>> 0; return seed / 4294967296; };
  const synth = new WaveformSynth(golden('ir_window'), random), ir = golden('ir'), ap = golden('aperiodicity'), pf = golden('post_filter'), f0 = golden('f0');
  const irFrame = new Float32Array(512), apFrame = new Float32Array(240), pfFrame = new Float32Array(512), audio = new Float32Array(count * 240);
  let offset = 0;
  for (let f = 0; f < count; f++) {
    for (let c = 0; c < 512; c++) { irFrame[c] = ir[c * count + f]; pfFrame[c] = pf[c * count + f]; }
    for (let c = 0; c < 240; c++) apFrame[c] = ap[c * count + f];
    const part = synth.push(irFrame, apFrame, pfFrame, f0[f]);
    if (part) { audio.set(part, offset); offset += part.length; }
  }
  const tail = synth.finish(); audio.set(tail, offset); assert.equal(offset + tail.length, audio.length);
  compare(`${name} isolated synthesis`, audio, golden('audio'), 1e-4, 1e-3);
  for (const chunk of [1, 4, 20]) {
    seed = 12345;
    const processor = await BetaProcessor.create(model, ort, 'wasm', chunk, params, random);
    seed = 12345; processor.reset();
    const traced: Record<string, Float32Array> = {};
    processor.onTrace = ({ stage, frame: start, data, channels }) => {
      if (!traced[stage]) traced[stage] = new Float32Array(channels * count);
      const width = data.length / channels;
      for (let c = 0; c < channels; c++) for (let j = 0; j < width && start + j < count; j++) traced[stage][c * count + start + j] = data[c * width + j];
    };
    const converted = new Float32Array(count * 240);
    offset = 0;
    while (processor.outputFrame < count) {
      const part = await processor.next(reader, count);
      converted.set(part.audio, offset); offset += part.audio.length;
    }
    const finish = processor.finish(); converted.set(finish, offset); assert.equal(offset + finish.length, converted.length);
    for (const [stage, data] of Object.entries(traced)) compare(`${name} c${chunk} ${stage}`, data, golden(stage), 1e-4);
    compare(`${name} c${chunk} end-to-end waveform`, converted, golden('audio'), 1e-3, 1e-3);
    // Negative control: a collapsed/all-zero waveform must fail the same numeric criterion.
    let referenceEnergy = 0;
    for (const v of golden('audio')) referenceEnergy += v * v;
    assert(referenceEnergy > 1, 'The oracle must exercise non-silent synthesis');
    assert(converted.some(v => Math.abs(v) > 0.01), 'Silent output cannot pass');
    await processor.dispose();
  }
}
writeFileSync('.research/runtime-results.json', JSON.stringify(reports, null, 2));
console.log('PRODUCTION TS + ORT WEB WASM PARITY PASS');