/**
 * Streaming port of the Beatrice 2.0.0-rc.0 Vocoder tail (everything after the four ConvNeXt stacks):
 *   overlap_add()   — pitch-synchronous FIR impulse-response synthesis, 512-tap, float64 phase accumulator
 *   generate_noise()— rectangular-window analysis / Hann-window synthesis of a uniform excitation, n_fft 480
 *   post filter     — 512-tap FIR applied per 240-sample frame via a 768-point FFT, output offset by 120
 *
 * The reference does this over a whole utterance with fold(); here it is done frame by frame with
 * persistent state, which is mathematically identical (all operations are shift-invariant overlap-adds)
 * apart from the initial phase, which the reference randomises (`torch.rand`) — so do we.
 */
import { rfft, irfft } from "./fft";

const HOP = 240, IR = 512, NFFT_N = 480, NFFT_P = 768, SR = 24000;

export class WaveformSynth {
  // periodic / aperiodic accumulation, indexed by absolute sample; ring of 4 frames + IR tail
  private readonly RING = HOP * 8;
  private per = new Float32Array(this.RING);
  private ape = new Float32Array(this.RING);
  private fir = new Float32Array(this.RING);
  private exc = new Float32Array(2 * HOP); // last two frames of excitation (rect analysis window is 480)
  private phase = Math.random(); // float64 phase accumulator, random initial phase (as in overlap_add)
  private frame = 0;

  private ampCur = new Float64Array(IR / 2 + 1);
  private phaCur = new Float64Array(IR / 2 + 1);
  private ampPrev = new Float64Array(IR / 2 + 1);
  private phaPrev = new Float64Array(IR / 2 + 1);
  private sRe = new Float64Array(IR / 2 + 1);
  private sIm = new Float64Array(IR / 2 + 1);
  private irBuf = new Float64Array(IR);
  private nRe = new Float64Array(NFFT_N / 2 + 1);
  private nIm = new Float64Array(NFFT_N / 2 + 1);
  private nBuf = new Float64Array(NFFT_N);
  private pRe = new Float64Array(NFFT_P / 2 + 1);
  private pIm = new Float64Array(NFFT_P / 2 + 1);
  private fRe = new Float64Array(NFFT_P / 2 + 1);
  private fIm = new Float64Array(NFFT_P / 2 + 1);
  private tRe = new Float64Array(NFFT_P / 2 + 1);
  private tIm = new Float64Array(NFFT_P / 2 + 1);
  private pBuf = new Float64Array(NFFT_P);
  private frameBuf = new Float64Array(HOP);
  private filterBuf = new Float64Array(IR);
  private readonly hann = new Float64Array(NFFT_N);

  constructor(private irWindow: Float32Array) {
    // torch.hann_window(N) = 0.5 - 0.5*cos(2πn/N)  (periodic)
    for (let n = 0; n < NFFT_N; n++) this.hann[n] = 0.5 - 0.5 * Math.cos((2 * Math.PI * n) / NFFT_N);
  }

  reset() { this.per.fill(0); this.ape.fill(0); this.fir.fill(0); this.exc.fill(0); this.frame = 0; this.phase = Math.random(); }

  /**
   * Push one 10 ms frame of generator output. Returns the finished 240-sample @24 kHz output frame
   * (index frame-1, one frame of overlap-add latency) or null while filling up.
   *   irRaw: 512 raw channels (log-amplitude 0..256, phase 257..511)
   *   aper:  240 channels, postFilter: 512 channels, f0Hz: target pitch of this frame
   */
  push(irRaw: Float32Array, aper: Float32Array, postFilter: Float32Array, f0Hz: number): Float32Array | null {
    const f = this.frame;
    const base = f * HOP;
    const R = this.RING;

    // ---- clear the region this frame will first write into (rolling buffers)
    for (let i = 0; i < HOP; i++) {
      this.per[(base + IR + i) % R] = 0;
      this.ape[(base + NFFT_N + i) % R] = 0;
      this.fir[(base + NFFT_P + i) % R] = 0;
    }

    // ---- impulse response of this frame: amp = exp(ir[0..256]), phase = [0, ir[257..511], 0], odd bins += π
    const H = IR / 2; // 256
    const amp = this.ampCur, pha = this.phaCur;
    for (let k = 0; k <= H; k++) amp[k] = Math.exp(irRaw[k]);
    pha[0] = 0; pha[H] = 0;
    for (let k = 1; k < H; k++) pha[k] = irRaw[H + k]; // ir[257 + (k-1)] = ir[256+k]
    for (let k = 1; k <= H; k += 2) pha[k] += Math.PI;

    // ---- periodic: advance the phase over this frame's 240 samples and drop an impulse at each wrap.
    // The reference marks index t where phase[t] > phase[t+1] and takes the IR of frame t // 240, so a
    // wrap detected while stepping to sample n belongs to sample n-1 (i.e. the previous frame when n == 0).
    const inc = f0Hz / SR;
    for (let n = 0; n < HOP; n++) {
      const prev = this.phase;
      let cur = prev + inc;
      cur -= Math.floor(cur);
      this.phase = cur;
      if (prev > cur) {
        const numer = 1 - prev;
        const frac = numer / (numer + cur);
        const dp = (-2 * Math.PI * frac) / IR;
        const sa = n === 0 ? this.ampPrev : amp, sp = n === 0 ? this.phaPrev : pha;
        for (let k = 0; k <= H; k++) {
          const a = sa[k], th = sp[k] + dp * k;
          this.sRe[k] = a * Math.cos(th); this.sIm[k] = a * Math.sin(th);
        }
        irfft(this.sRe, this.sIm, IR, this.irBuf);
        const at = base + n - 1;
        for (let i = 0; i < IR; i++) if (at + i >= 0) this.per[(at + i) % R] += this.irBuf[i] * this.irWindow[i];
      }
    }
    this.ampPrev.set(amp); this.phaPrev.set(pha);

    // ---- aperiodic: new uniform excitation, rectangular analysis over the last 480 samples
    this.exc.copyWithin(0, HOP);
    for (let n = 0; n < HOP; n++) this.exc[HOP + n] = Math.random() - 0.5;
    rfft(this.exc, NFFT_N, this.nRe, this.nIm);
    this.nRe[0] = 0; this.nIm[0] = 0;
    for (let k = 1; k <= NFFT_N / 2; k++) { const g = aper[k - 1]; this.nRe[k] *= g; this.nIm[k] *= g; }
    irfft(this.nRe, this.nIm, NFFT_N, this.nBuf);
    for (let i = 0; i < NFFT_N; i++) this.ape[(base + i) % R] += this.nBuf[i] * this.hann[i];

    // ---- post filter: 512-tap FIR (DC tap + 1) applied to this frame's 240 periodic+aperiodic samples
    this.filterBuf.set(postFilter);
    this.filterBuf[0] += 1.0;
    rfft(this.filterBuf, NFFT_P, this.fRe, this.fIm);
    for (let src = 0; src < 2; src++) {
      const buf = src === 0 ? this.per : this.ape;
      for (let i = 0; i < HOP; i++) this.frameBuf[i] = buf[(base + i) % R];
      rfft(this.frameBuf, NFFT_P, this.pRe, this.pIm);
      for (let k = 0; k <= NFFT_P / 2; k++) {
        this.tRe[k] = this.pRe[k] * this.fRe[k] - this.pIm[k] * this.fIm[k];
        this.tIm[k] = this.pRe[k] * this.fIm[k] + this.pIm[k] * this.fRe[k];
      }
      irfft(this.tRe, this.tIm, NFFT_P, this.pBuf);
      for (let i = 0; i < NFFT_P; i++) this.fir[(base + i) % R] += this.pBuf[i];
    }

    this.frame++;
    if (f < 1) return null;
    // output frame e = f-1 lives at buffer offset e*240 + 120 and is complete now
    const e = f - 1, off = e * HOP + 120;
    const out = new Float32Array(HOP);
    for (let i = 0; i < HOP; i++) out[i] = this.fir[(off + i) % R];
    return out;
  }
}

