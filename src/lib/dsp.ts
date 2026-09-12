/**
 * Host-side DSP of Beatrice, based on the trainer source:
 *  - PitchEstimator.extract_pitch_features (hop 160, win 560, 64 instfreq bins, YIN-style difference over
 *    max_corr_period 256 / corr_win_length 304, cosine-windowed log energy)
 *  - PitchEstimator.sample_pitch (softmax → bin0=-100 → width-4 box filter → banded argmax + features)
 *  - ProcessorCore2::Process1 pitch arithmetic (pitch shift / intonation / pitch correction / clamp)
 */
export const HOP = 160, WIN = 560, CUTOFF = 64, MAX_CORR = 256, CORR_WIN = 304, PAD = (WIN - HOP) / 2; // 200
export const BINS = 448, BAND = 4, BPO = 96, BINS_PER_SEMITONE = BPO / 12;

const cosT = new Float32Array(CUTOFF * WIN), sinT = new Float32Array(CUTOFF * WIN), wcos = new Float32Array(WIN);
for (let k = 0; k < CUTOFF; k++) for (let n = 0; n < WIN; n++) { const a = (-2 * Math.PI * k * n) / WIN; cosT[k * WIN + n] = Math.cos(a); sinT[k * WIN + n] = Math.sin(a); }
for (let n = 0; n < WIN; n++) wcos[n] = Math.sin((Math.PI * (n + 0.5)) / WIN);

export interface FrameFeatures { instfreq: Float32Array /*192*/; corr: Float32Array /*256*/; energy: number }

/** Sequential feature extractor (keeps the previous frame's spectrum for the instantaneous-frequency channels). */
export class PitchFeatureExtractor {
  private prevRe = new Float32Array(CUTOFF);
  private prevIm = new Float32Array(CUTOFF);
  private first = true;
  reset() { this.prevRe.fill(0); this.prevIm.fill(0); this.first = true; }
  /** `frame` = 560 samples: original[f*160-200 .. f*160+360) with zeros outside the signal. */
  process(frame: Float32Array): FrameFeatures {
    const inst = new Float32Array(3 * CUTOFF);
    for (let k = 0; k < CUTOFF; k++) {
      let re = 0, im = 0;
      const co = k * WIN;
      for (let n = 0; n < WIN; n++) { const v = frame[n]; re += v * cosT[co + n]; im += v * sinT[co + n]; }
      inst[k] = Math.log10(Math.hypot(re, im) + 1e-5);
      if (this.first) { inst[CUTOFF + k] = 0; inst[2 * CUTOFF + k] = 0; }
      else {
        const dre = re * this.prevRe[k] + im * this.prevIm[k];
        const dim = im * this.prevRe[k] - re * this.prevIm[k];
        const den = Math.hypot(dre, dim) + 1e-5;
        inst[CUTOFF + k] = dre / den; inst[2 * CUTOFF + k] = dim / den;
      }
      this.prevRe[k] = re; this.prevIm[k] = im;
    }
    this.first = false;
    const corr = new Float32Array(MAX_CORR);
    for (let t = 0; t < MAX_CORR; t++) {
      const lag = t + 1; let acc = 0;
      const a = MAX_CORR, b = MAX_CORR - lag;
      for (let j = 0; j < CORR_WIN; j++) { const d = frame[a + j] - frame[b + j]; acc += d * d; }
      acc *= 2 / CORR_WIN;
      corr[t] = Math.sqrt(acc < 0 ? 0 : acc);
    }
    let e = 0;
    for (let n = 0; n < WIN; n++) { const v = frame[n] * wcos[n]; e += v * v; }
    if (e < 1e-3) e = 1e-3;
    return { instfreq: inst, corr, energy: Math.log10(e) * 0.5 };
  }
}

export interface PitchSample { bin: number; unvoiced: number; half: number; dbl: number }

/** logits: 448 values of one frame. minBin/maxBin mirror Beatrice20rc0_SetMin/MaxQuantizedPitch (bins outside are excluded). */
export function samplePitch(logits: Float32Array, minBin = 1, maxBin = logits.length - 1, bandWidth = BAND): PitchSample {
  const bins = logits.length;
  if ((bins !== 384 && bins !== 448) || bandWidth < 1 || bandWidth >= bins) throw new Error('Invalid Beatrice pitch geometry');
  if (minBin > maxBin || maxBin - minBin + 1 < bandWidth) throw new Error(`Pitch range must cover at least ${bandWidth} bins`);
  const p = new Float64Array(bins);
  let mx = -Infinity;
  for (let k = 0; k < bins; k++) { if (!Number.isFinite(logits[k])) throw new Error('PitchEstimator returned non-finite logits'); if (logits[k] > mx) mx = logits[k]; }
  let sum = 0;
  for (let k = 0; k < bins; k++) { p[k] = Math.exp(logits[k] - mx); sum += p[k]; }
  for (let k = 0; k < bins; k++) p[k] /= sum;
  const unvoiced = p[0];
  p[0] = -100;
  for (let k = 1; k < bins; k++) if (k < minBin || k > maxBin) p[k] = -100;
  const nBand = bins - bandWidth + 1;
  const band = new Float64Array(nBand);
  let best = 0, bestV = -Infinity;
  for (let k = 0; k < nBand; k++) { let acc = 0; for (let j = 0; j < bandWidth; j++) acc += p[k + j]; band[k] = acc; if (acc > bestV) { bestV = acc; best = k; } }
  let arg = best, argV = -Infinity;
  for (let k = best; k < best + bandWidth && k < bins; k++) if (p[k] > argV) { argV = p[k]; arg = k; }
  const bandP = band[best];
  let half = 0, dbl = 0;
  if (best > BPO) { let idx = best - BPO; if (idx < 1) idx = 1; half = band[idx] / (bandP + 1e-6); }
  if (best <= bins - bandWidth - BPO) { let idx = best + BPO; if (idx > bins - bandWidth) idx = bins - bandWidth; dbl = band[idx] / (bandP + 1e-6); }
  return { bin: arg, unvoiced, half, dbl };
}

export const binToHz = (bin: number) => 55 * 2 ** (bin / BPO);
export const binToMidi = (bin: number) => 33 + bin / BINS_PER_SEMITONE;
export const midiToBin = (midi: number) => Math.min(BINS - 1, Math.max(1, Math.round((midi - 33) * BINS_PER_SEMITONE)));
export const midiToHz = (m: number) => 440 * 2 ** ((m - 69) / 12);

export interface PitchParams {
  pitchShift: number; // semitones, [-24, 24]
  averageSourcePitch: number; // MIDI-ish bin space in the official code: bins
  intonationIntensity: number;
  pitchCorrection: number; // [0,1]
  pitchCorrectionType: 0 | 1;
}

/** Exact port of the pitch arithmetic in ProcessorCore2::Process1. Returns the quantized pitch fed to the waveform generator. */
export function officialTargetPitch(quantized: number, p: PitchParams): number {
  const k = BINS_PER_SEMITONE;
  let tmp = p.averageSourcePitch + (quantized - p.averageSourcePitch) * p.intonationIntensity + k * p.pitchShift;
  if (p.pitchCorrection !== 0) {
    if (p.pitchCorrectionType === 0) {
      const nearest = (Math.floor(tmp / k) + 0.5) * k;
      const nd = (tmp - nearest) * (2 / k);
      tmp = Math.abs(nd) < 1e-4 ? nearest : nearest + nd * Math.pow(Math.abs(nd), -p.pitchCorrection) * (k / 2);
    } else {
      const nearest = Math.round(tmp / k) * k;
      const nd = (tmp - nearest) * (2 / k);
      if (p.pitchCorrection > 1 - 1e-4) tmp = nearest;
      else if (nd >= 0) tmp = nearest + Math.pow(nd, 1 / (1 - p.pitchCorrection)) * (k / 2);
      else tmp = nearest - Math.pow(-nd, 1 / (1 - p.pitchCorrection)) * (k / 2);
    }
  }
  return Math.min(BINS - 1, Math.max(1, Math.round(tmp)));
}

/** formant shift (semitones, [-2, 2] in 0.5 steps) → index into the 9 formant-shift embeddings (ProcessorCore2::SetFormantShift). */
export const formantIndex = (semis: number) => Math.round(Math.min(2, Math.max(-2, semis)) * 2 + 4);
