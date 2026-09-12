/**
 * AudioWorklet source (inlined as a string → Blob URL so the single-file build works offline).
 * Contains a 1:1 JavaScript port of beatrice-vst/src/common/resample.h:
 *   AnyFreqInOut = ConvertStreamFunctionFrequency(host ↔ 48 kHz, windowed-sinc, 32 taps)
 *                ∘ ConvertStreamFunctionBlockSize<480>
 *                ∘ ConvertStreamFunctionFrom2In3OutTo6InOut<80>   (48k→16k pick every 3rd, 24k→48k zero-stuff)
 *                ∘ Process1(160 @16k → 240 @24k)
 * The 160-sample 16 kHz frames are forwarded to the main thread (WebGPU inference); 240-sample 24 kHz frames
 * coming back are consumed here. Capture and playback queues are bounded independently of inference.
 */
export const WORKLET_SOURCE = String.raw`
function normalizedSinc(x) { if (Math.abs(x) < 1e-8) return 1.0; return Math.sin(x * Math.PI) / (x * Math.PI); }
function computeSimpleFraction(ratio) {
  let l = { numer: 0, denom: 1 }, r = { numer: 1, denom: 0 };
  for (;;) {
    const m = { numer: l.numer + r.numer, denom: l.denom + r.denom };
    if (ratio * m.denom < m.numer) { if (m.numer >= 1000 || m.denom >= 1000) return l; r = m; }
    else { if (m.numer >= 1000 || m.denom >= 1000) return r; l = m; }
  }
}
class RingBuf {
  constructor(siz) { this.setSize(siz); }
  setSize(siz) { this.siz = Math.max(1, siz); this.data = new Float32Array(this.siz); this.w = 0; }
  push(v) { this.data[this.w] = v; this.w = (this.w + 1) % this.siz; }
  at(idx) { /* idx in [-siz, -1] */ let i = (this.w + idx) % this.siz; if (i < 0) i += this.siz; return this.data[i]; }
}
class DownUpSampler {
  constructor(outer, inner, filterSize, cutIn, cutOut) { this.filterSize = filterSize; this.setSampleRates(outer, inner, cutIn, cutOut); }
  isReady() { return this.ready; }
  setSampleRates(outer, inner, cutIn, cutOut) {
    if (outer <= 0 || inner <= 0) { this.ready = false; return; }
    this.downFirst = outer >= inner;
    if (this.downFirst) { this.high = outer; this.low = inner; this.cutDown = cutIn; this.cutUp = cutOut; }
    else { this.high = inner; this.low = outer; this.cutDown = cutOut; this.cutUp = cutIn; }
    const f = computeSimpleFraction(this.high / this.low);
    if (f.numer === 0 || f.denom === 0) { this.ready = false; return; }
    this.ratioHigh = f.numer; this.ratioLow = f.denom;
    this.reset(); this.ready = true;
  }
  reset() {
    const L = this.filterSize * this.ratioHigh + 1, center = Math.floor(L / 2);
    this.coefDown = new Float32Array(L); this.coefUp = new Float32Array(L);
    for (let i = 0; i < L; i++) {
      const sd = normalizedSinc((i - center) / this.ratioHigh * this.cutDown);
      const su = normalizedSinc((i - center) / this.ratioHigh * this.cutUp);
      const w = 0.5 - 0.5 * Math.cos(Math.PI * 2.0 / (L - 1) * i);
      this.coefDown[i] = this.cutDown * sd * w; this.coefUp[i] = this.cutUp * su * w;
    }
    this.clockDown = this.ratioHigh - 1; this.clockUp = this.ratioHigh - 1;
    this.bufHigh = new RingBuf(Math.floor(this.filterSize * this.ratioHigh / this.ratioLow) + 1);
    this.bufLow = new RingBuf(this.filterSize + 1);
  }
  downsample(input) {
    const gain = this.ratioLow / this.ratioHigh;
    const out = new Float32Array(Math.floor((input.length * this.ratioLow + this.clockDown) / this.ratioHigh));
    let o = 0;
    for (let n = 0; n < input.length; n++) {
      this.bufHigh.push(input[n]);
      this.clockDown += this.ratioLow;
      if (this.clockDown >= this.ratioHigh) {
        this.clockDown -= this.ratioHigh;
        let acc = 0, ib = -1;
        for (let f = this.ratioLow - this.clockDown; f < this.coefDown.length - 1; f += this.ratioLow) acc += this.bufHigh.at(ib--) * this.coefDown[f];
        out[o++] = acc * gain;
      }
    }
    return out;
  }
  upsample(input) {
    let len;
    if (this.downFirst) len = Math.floor((input.length * this.ratioHigh + this.clockDown - this.clockUp) / this.ratioLow);
    else len = Math.floor(((input.length + 1) * this.ratioHigh - this.clockUp - 1) / this.ratioLow);
    const out = new Float32Array(len);
    let ii = 0;
    for (let o = 0; o < len; o++) {
      this.clockUp += this.ratioLow;
      if (this.clockUp >= this.ratioHigh) { this.clockUp -= this.ratioHigh; this.bufLow.push(ii < input.length ? input[ii++] : 0); }
      let acc = 0, ib = -1;
      for (let f = this.clockUp; f < this.coefUp.length - 1; f += this.ratioHigh) acc += this.bufLow.at(ib--) * this.coefUp[f];
      out[o] = acc;
    }
    return out;
  }
  resampleIn(x) { return this.downFirst ? this.downsample(x) : this.upsample(x); }
  resampleOut(x) { return this.downFirst ? this.upsample(x) : this.downsample(x); }
}
// ConvertStreamFunctionFrom2In3OutTo6InOut<80>  (480 @48k → 160 @16k → model → 240 @24k → 480 @48k)
function make6n(model) {
  const fin = new Float32Array(160), fout = new Float32Array(240);
  return function (input480, output480) {
    for (let i = 0; i < 160; i++) fin[i] = input480[(i + 1) * 3 - 1];
    model(fin, fout);
    output480.fill(0);
    for (let i = 0; i < 240; i++) output480[i * 2] = fout[i];
  };
}
// ConvertStreamFunctionBlockSize<480>
function makeBlock480(fn) {
  const buffer = new Float32Array(480), processed = new Float32Array(480);
  let idx = 0;
  return function (input, output) {
    const n = input.length;
    for (let io = 0; io < n;) {
      const m = Math.min(480 - idx, n - io);
      output.set(buffer.subarray(idx, idx + m), io);
      buffer.set(input.subarray(io, io + m), idx);
      idx += m; io += m;
      if (idx === 480) { idx = 0; fn(buffer, processed); buffer.set(processed); }
    }
  };
}
class AnyFreqInOut {
  constructor(sampleRate, model) {
    this.sampleRate = sampleRate;
    this.inner = makeBlock480(make6n(model));
    const clampIn = Math.min(48000, Math.max(16000, sampleRate)), clampOut = Math.min(48000, Math.max(24000, sampleRate));
    this.rs = new DownUpSampler(sampleRate, 48000, 32, 0.99 * 16000 / clampIn, 0.99 * 24000 / clampOut);
    this.work = new Float32Array(0);
  }
  isReady() { return this.rs.isReady(); }
  latencySamples() { return 32 * this.sampleRate / Math.min(this.sampleRate, 48000); }
  process(input, output) {
    const a = this.rs.resampleIn(input);
    if (this.work.length !== a.length) this.work = new Float32Array(a.length);
    this.inner(a, this.work);
    const b = this.rs.resampleOut(this.work);
    const m = Math.min(b.length, output.length);
    output.set(b.subarray(0, m));
    for (let i = m; i < output.length; i++) output[i] = 0;
  }
}

class BeatriceProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.inGain = 1; this.outGain = 1; this.inGainT = 1; this.outGainT = 1;
    this.monitor = false;
    this.outRing = new Float32Array(240 * 100);
    this.outRead = 0; this.outWrite = 0; this.outCount = 0; this.outputDrops = 0;
    this.lastStatsFrame = -1;
    this.batch = []; this.batchFrames = 4; this.underruns = 0; this.frames = 0;
    this.fio = new AnyFreqInOut(sampleRate, (in160, out240) => this.process1(in160, out240));
    this.port.onmessage = (e) => {
      const d = e.data;
      if (d.type === 'params') { if (d.inputGain !== undefined) this.inGainT = d.inputGain; if (d.outputGain !== undefined) this.outGainT = d.outputGain; if (d.monitor !== undefined) this.monitor = d.monitor; if (d.batchFrames !== undefined) this.batchFrames = Math.max(1, d.batchFrames | 0); }
      else if (d.type === 'out') {
        const x = d.frames;
        if (!(x instanceof Float32Array)) return;
        if (this.outCount + x.length > this.outRing.length) {
          this.outputDrops++;
          this.outCount = 0; this.outRead = this.outWrite;
        }
        for (let i = Math.max(0, x.length - this.outRing.length); i < x.length; i++) {
          this.outRing[this.outWrite] = x[i]; this.outWrite = (this.outWrite + 1) % this.outRing.length; this.outCount++;
        }
      }
      else if (d.type === 'reset') { this.outCount = 0; this.outRead = this.outWrite; }
    };
    this.tmpIn = new Float32Array(128); this.tmpOut = new Float32Array(128);
    this.port.postMessage({ type: 'ready', sampleRate, latencySamples: Math.round(0.0375 * sampleRate + this.fio.latencySamples()) });
  }
  process1(in160, out240) {
    this.batch.push(Float32Array.from(in160));
    this.frames++;
    if (this.batch.length >= this.batchFrames) {
      const all = new Float32Array(this.batch.length * 160);
      this.batch.forEach((f, i) => all.set(f, i * 160));
      this.port.postMessage({ type: 'in', samples: all, frameIndex: this.frames - this.batch.length }, [all.buffer]);
      this.batch = [];
    }
    if (this.outCount >= 240) {
      for (let i = 0; i < 240; i++) { out240[i] = this.outRing[this.outRead]; this.outRead = (this.outRead + 1) % this.outRing.length; }
      this.outCount -= 240;
    }
    else if (this.monitor) { for (let i = 0; i < 240; i++) { const p = i * 160 / 240; const j = Math.floor(p); const t = p - j; out240[i] = in160[j] * (1 - t) + (j + 1 < 160 ? in160[j + 1] : in160[j]) * t; } }
    else { out240.fill(0); if (this.frames > 25) this.underruns++; }
  }
  process(inputs, outputs) {
    const inp = inputs[0] && inputs[0][0];
    const out = outputs[0];
    if (!out || !out[0]) return true;
    const n = out[0].length;
    if (this.tmpIn.length !== n) { this.tmpIn = new Float32Array(n); this.tmpOut = new Float32Array(n); }
    for (let i = 0; i < n; i++) { this.inGain += (this.inGainT - this.inGain) * 0.002; this.tmpIn[i] = inp ? inp[i] * this.inGain : 0; }
    this.fio.process(this.tmpIn, this.tmpOut);
    for (let i = 0; i < n; i++) { this.outGain += (this.outGainT - this.outGain) * 0.002; this.tmpOut[i] *= this.outGain; }
    for (let ch = 0; ch < out.length; ch++) out[ch].set(this.tmpOut);
    if (this.frames && this.frames - this.lastStatsFrame >= 25) {
      this.lastStatsFrame = this.frames;
      this.port.postMessage({ type: 'stats', frames: this.frames, underruns: this.underruns, queue: Math.floor(this.outCount / 240), outputDrops: this.outputDrops });
    }
    return true;
  }
}
registerProcessor('beatrice-processor', BeatriceProcessor);
`;
