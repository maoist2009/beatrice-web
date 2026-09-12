// beta.2 source-filter tail. Filter a frame only after the following frame resolves its last pitch mark.
import { rfft, irfft } from '../fft';

const HOP = 240, IR = 512, NOISE = 480, POST = 768, RING = 240 * 12;
const f32 = Math.fround;

export class WaveformSynth {
  private periodic = new Float32Array(RING);
  private aperiodic = new Float32Array(RING);
  private filtered = new Float32Array(RING);
  private excitation = new Float32Array(NOISE);
  private phaseSum = 0;
  private prevPhase = 0;
  private frame = 0;
  private emitted = 0;
  private finished = false;
  private amp = new Float32Array(257);
  private phase = new Float32Array(257);
  private prevAmp = new Float32Array(257);
  private prevIrPhase = new Float32Array(257);
  private re512 = new Float64Array(257);
  private im512 = new Float64Array(257);
  private ir = new Float32Array(IR);
  private re480 = new Float64Array(241);
  private im480 = new Float64Array(241);
  private noise = new Float32Array(NOISE);
  private re768 = new Float64Array(385);
  private im768 = new Float64Array(385);
  private filterRe = new Float64Array(385);
  private filterIm = new Float64Array(385);
  private post = new Float32Array(POST);
  private signal = new Float32Array(HOP);
  private previousFilter = new Float32Array(IR);
  private filterBuffer = new Float32Array(IR);
  private hann = Float32Array.from({ length: NOISE }, (_, n) => 0.5 - 0.5 * Math.cos(2 * Math.PI * n / NOISE));

  constructor(private window: Float32Array, private random: () => number = Math.random) { this.reset(); }
  reset() {
    this.periodic.fill(0); this.aperiodic.fill(0); this.filtered.fill(0); this.previousFilter.fill(0);
    this.prevAmp.fill(0); this.prevIrPhase.fill(0);
    this.frame = this.emitted = 0; this.finished = false;
    this.phaseSum = f32(this.random()); this.prevPhase = this.phaseSum;
    // The reference starts with two random frames, not a zero-prefixed excitation.
    for (let i = 0; i < NOISE; i++) this.excitation[i] = f32(this.random()) - 0.5;
  }

  push(irRaw: Float32Array, aper: Float32Array, filter: Float32Array, f0: number): Float32Array | null {
    if (this.finished) throw new Error('Cannot append audio after synthesizer.finish()');
    if (irRaw.length !== IR || aper.length !== HOP || filter.length !== IR || !Number.isFinite(f0) || f0 <= 0 || f0 >= 24000) throw new Error('Invalid Beatrice synthesis inputs');
    for (const a of [irRaw, aper, filter]) for (const v of a) if (!Number.isFinite(v)) throw new Error('Vocoder produced NaN/Infinity');
    const f = this.frame, base = f * HOP;
    for (let k = 0; k <= 256; k++) {
      this.amp[k] = Math.exp(irRaw[k]);
      if (!Number.isFinite(this.amp[k])) throw new Error('Vocoder impulse-response amplitude overflow');
      this.phase[k] = (k === 0 || k === 256 ? 0 : irRaw[256 + k]) + (k % 2 ? Math.PI : 0);
    }
    const increment = f32(f32(f0) / 24000);
    for (let n = 0; n < HOP; n++) {
      if (f === 0 && n === 0) continue;
      this.phaseSum += increment;
      const current = f32(this.phaseSum - Math.floor(this.phaseSum)), previous = this.prevPhase;
      if (previous > current) {
        const numerator = f32(1 - previous), fraction = f32(numerator / f32(numerator + current));
        const amp = n === 0 ? this.prevAmp : this.amp, phase = n === 0 ? this.prevIrPhase : this.phase;
        for (let k = 0; k <= 256; k++) {
          const angle = f32(phase[k] + f32(f32(k * (-2 * Math.PI / IR)) * fraction));
          this.re512[k] = f32(amp[k] * Math.cos(angle)); this.im512[k] = f32(amp[k] * Math.sin(angle));
        }
        irfft(this.re512, this.im512, IR, this.ir);
        for (let i = 0; i < IR; i++) this.periodic[(base + n - 1 + i) % RING] += f32(this.ir[i] * this.window[i]);
      }
      this.prevPhase = current;
    }
    this.prevAmp.set(this.amp); this.prevIrPhase.set(this.phase);

    if (f > 0) {
      this.excitation.copyWithin(0, HOP);
      for (let i = HOP; i < NOISE; i++) this.excitation[i] = f32(this.random()) - 0.5;
    }
    rfft(this.excitation, NOISE, this.re480, this.im480);
    this.re480[0] = this.im480[0] = 0;
    for (let k = 1; k <= HOP; k++) { this.re480[k] = f32(f32(this.re480[k]) * aper[k - 1]); this.im480[k] = f32(f32(this.im480[k]) * aper[k - 1]); }
    irfft(this.re480, this.im480, NOISE, this.noise);
    for (let i = 0; i < NOISE; i++) this.aperiodic[(base + i) % RING] += f32(this.noise[i] * this.hann[i]);

    if (f > 0) this.applyFilter((f - 1) * HOP, this.previousFilter);
    this.previousFilter.set(filter);
    this.frame++;
    return f >= 2 ? this.emit() : null;
  }

  private applyFilter(base: number, coefficients: Float32Array) {
    this.filterBuffer.set(coefficients); this.filterBuffer[0] += 1;
    rfft(this.filterBuffer, POST, this.filterRe, this.filterIm);
    for (const source of [this.periodic, this.aperiodic]) {
      for (let i = 0; i < HOP; i++) { this.signal[i] = source[(base + i) % RING]; source[(base + i) % RING] = 0; }
      rfft(this.signal, POST, this.re768, this.im768);
      for (let k = 0; k <= POST / 2; k++) {
        const r = this.re768[k], i = this.im768[k];
        this.re768[k] = r * this.filterRe[k] - i * this.filterIm[k];
        this.im768[k] = r * this.filterIm[k] + i * this.filterRe[k];
      }
      irfft(this.re768, this.im768, POST, this.post);
      for (let i = 0; i < POST; i++) this.filtered[(base + i) % RING] += this.post[i];
    }
    if (base === 0) this.filtered.fill(0, 0, 120);
  }

  private emit() {
    const out = new Float32Array(HOP), base = this.emitted++ * HOP + 120;
    for (let i = 0; i < HOP; i++) {
      const index = (base + i) % RING, value = this.filtered[index];
      if (!Number.isFinite(value)) throw new Error('Non-finite synthesized audio');
      out[i] = value; this.filtered[index] = 0;
    }
    return out;
  }

  finish(): Float32Array {
    if (this.finished) return new Float32Array(0);
    this.finished = true;
    if (!this.frame) return new Float32Array(0);
    this.applyFilter((this.frame - 1) * HOP, this.previousFilter);
    const result = new Float32Array((this.frame - this.emitted) * HOP);
    for (let off = 0; this.emitted < this.frame; off += HOP) result.set(this.emit(), off);
    return result;
  }
}