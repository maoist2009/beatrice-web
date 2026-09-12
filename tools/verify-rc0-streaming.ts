// Numerical equivalence: incremental rc.0 graphs (states carried across chunks) vs. the previous
// full-window graphs run once over the whole clip. Same real weights (hecko rc.0), ORT Web WASM in Node.
import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';
import * as ort from 'onnxruntime-web/wasm';
import { collectFiles, parseParaphernalia, f16ToF32 } from '../src/lib/paraphernalia';
import { extractPitch, extractPhone, extractWaveformGenerator, WG } from '../src/lib/layouts';
import { buildPitchEstimatorOnnx, buildPhoneExtractorOnnx, buildVocoderOnnx } from '../src/lib/graphs';
import { buildRc0Pitch, buildRc0Phone, buildRc0Vocoder } from '../src/lib/rc0/graphs';
import { StatefulSession } from '../src/lib/session';
import { PitchFeatureExtractor } from '../src/lib/dsp';

ort.env.wasm.numThreads = 1;
const read = (p: string) => { const b = readFileSync(p); return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer; };
const model = parseParaphernalia(await collectFiles([new File([read('.research/hecko.zip')], 'model.zip')]));
assert.equal(model.format, 'beatrice-rc0');
const pw = extractPitch(f16ToF32(model.files.pitch_estimator));
const hw = extractPhone(f16ToF32(model.files.phone_extractor));
const wg = extractWaveformGenerator(f16ToF32(model.files.waveform_generator));

const raw = readFileSync('.research/jfk.wav');
const s16 = new Int16Array(raw.buffer, raw.byteOffset + 44, (raw.byteLength - 44) / 2);
const T = 240; // frames (2.4 s) — fits inside one attention window so the bounded cache is exact
const pcm = new Float32Array(T * 160); for (let i = 0; i < pcm.length; i++) pcm[i] = s16[6000 + i] / 32768;
const at = (i: number) => (i < 0 || i >= pcm.length ? 0 : pcm[i]);

const reports: Record<string, unknown>[] = [];
function compare(label: string, a: Float32Array, b: Float32Array, tol: number) {
  assert.equal(a.length, b.length, label);
  let ds = 0, bs = 0, max = 0;
  for (let i = 0; i < a.length; i++) { assert(Number.isFinite(a[i]) && Number.isFinite(b[i]), `${label} nonfinite`); const d = a[i] - b[i]; ds += d * d; bs += b[i] * b[i]; max = Math.max(max, Math.abs(d)); }
  const rel = Math.sqrt(ds / Math.max(1e-24, bs));
  console.log('[stream≡window]', JSON.stringify({ label, maxAbs: max, rmsRelative: rel })); reports.push({ label, maxAbs: max, rmsRelative: rel });
  assert(rel <= tol, `${label}: rmsRelative ${rel} > ${tol}`);
}
const so = { executionProviders: ['wasm' as const], graphOptimizationLevel: 'all' as const };
async function runReference(session: ort.InferenceSession, feeds: Record<string, ort.Tensor>) {
  const result = await session.run(feeds);
  try { return Object.fromEntries(Object.entries(result).map(([name, tensor]) => [name, (tensor.data as Float32Array).slice()])); }
  finally { for (const tensor of Object.values(result)) tensor.dispose(); for (const tensor of Object.values(feeds)) tensor.dispose(); await session.release(); }
}

// ---------------- pitch: features once, old graph over all T, new graph per chunk
const fx = new PitchFeatureExtractor(), frame = new Float32Array(560);
const inst = new Float32Array(192 * T), corr = new Float32Array(256 * T);
for (let f = 0; f < T; f++) { for (let i = 0; i < 560; i++) frame[i] = at(f * 160 - 200 + i); const d = fx.process(frame); for (let c = 0; c < 192; c++) inst[c * T + f] = d.instfreq[c]; for (let c = 0; c < 256; c++) corr[c * T + f] = d.corr[c]; }
const oldPitch = await ort.InferenceSession.create(buildPitchEstimatorOnnx(pw, T), so);
const refLogits = (await runReference(oldPitch, { instfreq: new ort.Tensor('float32', inst, [1, 192, T]), corr_diff: new ort.Tensor('float32', corr, [1, 256, T]) })).logits;

// ---------------- phone
const wavFull = new Float32Array(T * 160); wavFull.set(pcm);
const oldPhone = await ort.InferenceSession.create(buildPhoneExtractorOnnx(hw, T), so);
const refUnits = (await runReference(oldPhone, { wav: new ort.Tensor('float32', wavFull, [1, 1, T * 160]) })).units;

// ---------------- vocoder: deterministic pseudo-random x and speaker K/V shared by both graphs
let seed = 7; const rnd = () => { seed = (Math.imul(1664525, seed) + 1013904223) >>> 0; return seed / 4294967296 - 0.5; };
const x = new Float32Array(256 * T); for (let i = 0; i < x.length; i++) x[i] = rnd();
const hd = model.setter!.attentionChannels / WG.heads;
const kt = Array.from({ length: 4 }, () => Float32Array.from({ length: WG.heads * hd * WG.kvLen }, rnd));
const vv = Array.from({ length: 4 }, () => Float32Array.from({ length: WG.heads * WG.kvLen * hd }, rnd));
const oldVoc = await ort.InferenceSession.create(buildVocoderOnnx(wg, T), so);
const feedsOld: Record<string, ort.Tensor> = { x: new ort.Tensor('float32', x, [1, 256, T]) };
for (let b = 0; b < 4; b++) { feedsOld[`kt${b}`] = new ort.Tensor('float32', kt[b], [1, WG.heads, hd, WG.kvLen]); feedsOld[`v${b}`] = new ort.Tensor('float32', vv[b], [1, WG.heads, WG.kvLen, hd]); }
const ro = await runReference(oldVoc, feedsOld);
const refIr = ro.ir, refAp = ro.aperiodicity, refPf = ro.post_filter;

for (const chunk of [4, 20]) {
  const P = 64; // ≥ T/4 → cache never truncates within this clip
  const pitch = await StatefulSession.create(ort as any, buildRc0Pitch(pw, chunk), 'wasm', false);
  const phone = await StatefulSession.create(ort as any, buildRc0Phone(hw, chunk, P), 'wasm', false);
  const voc = await StatefulSession.create(ort as any, buildRc0Vocoder(wg, chunk), 'wasm', false);
  const logits = new Float32Array(448 * T), units = new Float32Array(128 * T), ir = new Float32Array(512 * T), ap = new Float32Array(240 * T), pf = new Float32Array(512 * T);
  const cut = (src: Float32Array, ch: number, start: number, n: number) => { const o = new Float32Array(ch * n); for (let c = 0; c < ch; c++) for (let j = 0; j < n; j++) { const f = start + j; if (f < T) o[c * n + j] = src[c * T + f]; } return o; };
  const paste = (dst: Float32Array, src: Float32Array, ch: number, start: number, n: number) => { for (let c = 0; c < ch; c++) for (let j = 0; j < n && start + j < T; j++) dst[c * T + start + j] = src[c * n + j]; };
  const valid = new Float32Array(chunk + 1);
  const wav = new Float32Array(chunk * 160 + 80);
  let first = true;
  for (let e = 0; e < T; e += chunk) {
    for (let i = 0; i <= chunk; i++) valid[i] = e + i < T ? 1 : 0;
    const lo = await pitch.run({ instfreq: { data: cut(inst, 192, e, chunk + 1), dims: [1, 192, chunk + 1] }, corr_diff: { data: cut(corr, 256, e, chunk + 1), dims: [1, 256, chunk + 1] }, valid: { data: valid, dims: [1, 1, chunk + 1] } });
    paste(logits, lo.logits, 448, e, chunk);
    for (let i = 0; i < wav.length; i++) wav[i] = at(e * 160 - 40 + i);
    const ph = await phone.run({ wav: { data: wav, dims: [1, 1, wav.length] } });
    paste(units, ph.units, 128, e, chunk);
    const feeds: Record<string, { data: Float32Array; dims: number[] }> = { x: { data: cut(x, 256, e, chunk + 2), dims: [1, 256, chunk + 2] } };
    if (first) { for (let b = 0; b < 4; b++) { feeds[`kt${b}`] = { data: kt[b], dims: [1, WG.heads, hd, WG.kvLen] }; feeds[`v${b}`] = { data: vv[b], dims: [1, WG.heads, WG.kvLen, hd] }; } first = false; }
    const vo = await voc.run(feeds);
    paste(ir, vo.ir, 512, e, chunk); paste(ap, vo.aperiodicity, 240, e, chunk); paste(pf, vo.post_filter, 512, e, chunk);
  }
  // the old graphs zero-pad the future, so compare only frames whose look-ahead was real data
  const trim = (a: Float32Array, ch: number, n: number) => { const o = new Float32Array(ch * n); for (let c = 0; c < ch; c++) o.set(a.subarray(c * T, c * T + n), c * n); return o; };
  compare(`chunk ${chunk} pitch logits`, trim(logits, 448, T - 2), trim(refLogits, 448, T - 2), 2e-4);
  compare(`chunk ${chunk} phone units (attention KV cache)`, trim(units, 128, T), trim(refUnits, 128, T), 2e-4);
  compare(`chunk ${chunk} vocoder ir`, trim(ir, 512, T - 3), trim(refIr, 512, T - 3), 2e-4);
  compare(`chunk ${chunk} vocoder aperiodicity`, trim(ap, 240, T - 3), trim(refAp, 240, T - 3), 2e-4);
  compare(`chunk ${chunk} vocoder post_filter`, trim(pf, 512, T - 3), trim(refPf, 512, T - 3), 2e-4);
  await pitch.dispose(); await phone.dispose(); await voc.dispose();
}
writeFileSync('.research/rc0-streaming-results.json', JSON.stringify(reports, null, 2));
console.log('RC0 STREAMING ≡ FULL-WINDOW PASS');
