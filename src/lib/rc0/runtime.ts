/**
 * Incremental rc.0 processor — same frame schedule as the validated beta.2 processor (analysis runs two
 * frames ahead of synthesis; energy shifted −1, quantised pitch / pitch features −2, reflect-padded at
 * start), plus the rc.0 specifics: kNN-VC codebook quantisation + channel RMS norm, 448 pitch bins with
 * band width 4, and speaker key/value cross-attention in the vocoder prenet.
 */
import type * as ORT from 'onnxruntime-web';
import { buildRc0Phone, buildRc0Pitch, buildRc0Vocoder } from './graphs';
import { extractPitch, extractPhone, extractWaveformGenerator, pitchExpectedHalves, phoneExpectedHalves, wgExpectedHalves, WG, PHONE, PITCH } from '../layouts';
import { f16ToF32, C as PC, type Paraphernalia } from '../paraphernalia';
import { PitchFeatureExtractor, samplePitch, officialTargetPitch, binToHz, binToMidi, midiToBin, formantIndex } from '../dsp';
import { WaveformSynth } from '../beta/synth';
import type { Backend, EngineParams, FrameResult } from '../engine';
import { StatefulSession } from '../session';

type OrtRuntime = typeof ORT;
type PcmReader = (sample: number) => number;

export class Rc0Processor {
  private phone!: StatefulSession;
  private pitch!: StatefulSession;
  private vocoder!: StatefulSession;
  private wg!: { embedPhone: { w: Float32Array; b: Float32Array }; embedQPitch: Float32Array; embedPitchFeat: { w: Float32Array; b: Float32Array }; irWindow: Float32Array };
  private fx = new PitchFeatureExtractor();
  private analysis = 0; private synthesis = 0; private featureNext = 0;
  private capacity: number;
  private units: Float32Array; private bins: Int32Array; private feats: Float32Array; private energy: Float32Array;
  private inst: Float32Array; private corr: Float32Array;
  private wavInput: Float32Array; private instInput: Float32Array; private corrInput: Float32Array; private validInput: Float32Array; private xInput: Float32Array;
  private frameInput = new Float32Array(560);
  private pitchInput = new Float32Array(PITCH.bins);
  private phoneVec = new Float32Array(PHONE.phone);
  private vqQ = new Float32Array(PHONE.phone); private vqI = new Int32Array(8); private vqS = new Float32Array(8);
  private synth!: WaveformSynth;
  private irFrame = new Float32Array(WG.irLen); private apFrame = new Float32Array(WG.hop); private pfFrame = new Float32Array(WG.irLen);
  // speaker-dependent
  private kt: Float32Array[] = []; private vv: Float32Array[] = []; private kvDirty = true;
  private codebook = new Float32Array(0); private spkEmb = new Float32Array(WG.hidden); private fmtEmb = new Float32Array(WG.hidden);
  private preparedSpeaker = -1; private preparedFormant = NaN;
  graphBytes = 0;
  backend: Backend = 'wasm';
  captured = false;
  onTrace?: (t: { stage: string; frame: number; data: Float32Array; channels: number }) => void;

  private model: Pick<Paraphernalia, 'speakers' | 'setter'>;
  private constructor(model: Paraphernalia, readonly chunk: number, private params: EngineParams, private random: () => number) {
    this.model = { speakers: model.speakers, setter: model.setter };
    this.capacity = chunk * 8 + 64;
    const cap = this.capacity;
    this.units = new Float32Array(cap * PHONE.phone); this.bins = new Int32Array(cap); this.feats = new Float32Array(cap * 3); this.energy = new Float32Array(cap);
    this.inst = new Float32Array(cap * 192); this.corr = new Float32Array(cap * 256);
    this.wavInput = new Float32Array(chunk * 160 + 80);
    this.instInput = new Float32Array(192 * (chunk + 1)); this.corrInput = new Float32Array(256 * (chunk + 1)); this.validInput = new Float32Array(chunk + 1);
    this.xInput = new Float32Array(WG.hidden * (chunk + 2));
  }

  /** chunk is rounded UP to a multiple of 4 (strided attention). attnPositions = per-subsequence KV memory. */
  static async create(model: Paraphernalia, ort: OrtRuntime, ep: Backend, chunk: number, params: EngineParams, opts: { gpuBound: boolean; attnPositions: number; random?: () => number }) {
    if (model.format !== 'beatrice-rc0') throw new Error('Rc0Processor requires rc.0-format weights');
    if (!model.setter) throw new Error('embedding_setter.bin is required (speaker cross-attention K/V projections)');
    if (!Number.isInteger(chunk) || chunk < 1 || chunk > 40 || !Number.isFinite(opts.attnPositions)) throw new Error('Invalid rc.0 chunk or attention cache');
    chunk = Math.max(4, Math.ceil(chunk / 4) * 4);
    const P = Math.max(chunk / 4, Math.min(256, Math.round(opts.attnPositions)));
    const check = (name: string, buf: ArrayBuffer, expected: number) => { if (buf.byteLength / 2 !== expected) throw new Error(`${name}: ${buf.byteLength / 2} f16 values, rc.0 layout needs ${expected}`); };
    check('pitch_estimator.bin', model.files.pitch_estimator, pitchExpectedHalves());
    check('phone_extractor.bin', model.files.phone_extractor, phoneExpectedHalves());
    check('waveform_generator.bin', model.files.waveform_generator, wgExpectedHalves());
    const e = new Rc0Processor(model, chunk, { ...params }, opts.random ?? Math.random);
    e.captured = false;
    try {
      // one graph at a time: decoded f32 weights + serialized protobuf are released before the next
      {
        const pw = extractPitch(f16ToF32(model.files.pitch_estimator));
        const gph = buildRc0Pitch(pw, chunk); e.graphBytes += gph.bytes.length;
        e.pitch = await StatefulSession.create(ort, gph, ep, opts.gpuBound);
      }
      {
        const hw = extractPhone(f16ToF32(model.files.phone_extractor));
        const gph = buildRc0Phone(hw, chunk, P); e.graphBytes += gph.bytes.length;
        e.phone = await StatefulSession.create(ort, gph, ep, opts.gpuBound);
      }
      {
        const wg = extractWaveformGenerator(f16ToF32(model.files.waveform_generator));
        e.wg = { embedPhone: { w: wg.embedPhone.w.slice(), b: wg.embedPhone.b!.slice() }, embedQPitch: wg.embedQPitch.slice(), embedPitchFeat: { w: wg.embedPitchFeat.w.slice(), b: wg.embedPitchFeat.b!.slice() }, irWindow: wg.irWindow.slice() };
        const gph = buildRc0Vocoder(wg, chunk); e.graphBytes += gph.bytes.length;
        e.vocoder = await StatefulSession.create(ort, gph, ep, opts.gpuBound);
      }
      e.backend = ep; e.prepareSpeaker(); e.reset();
      return e;
    } catch (err) { await e.dispose(); throw err; }
  }

  setParams(p: EngineParams) { this.params = { ...p }; if (p.speaker !== this.preparedSpeaker || p.formantShift !== this.preparedFormant) this.prepareSpeaker(); }
  private prepareSpeaker() {
    const m = this.model, s = Math.min(Math.max(0, this.params.speaker | 0), m.speakers.nSpeakers - 1);
    const set = m.setter!, H = WG.heads, hd = set.attentionChannels / H, L = WG.kvLen, KC = WG.kvCh;
    const kv = m.speakers.keyValue.subarray(s * L * KC, (s + 1) * L * KC);
    if (this.kt.length !== WG.preBlocks) { this.kt = Array.from({ length: WG.preBlocks }, () => new Float32Array(H * hd * L)); this.vv = Array.from({ length: WG.preBlocks }, () => new Float32Array(H * L * hd)); }
    if (s !== this.preparedSpeaker) {
      for (let b = 0; b < WG.preBlocks; b++) {
        const blk = set.blocks[b], kt = this.kt[b], vv = this.vv[b];
        for (let h = 0; h < H; h++) for (let j = 0; j < L; j++) for (let d = 0; d < hd; d++) {
          let ak = blk.kB[h][d], av = blk.vB[h][d];
          const kw = blk.kW[h], vw = blk.vW[h];
          for (let c = 0; c < KC; c++) { const x = kv[j * KC + c]; ak += kw[d * KC + c] * x; av += vw[d * KC + c] * x; }
          kt[(h * hd + d) * L + j] = ak; vv[(h * L + j) * hd + d] = av;
        }
      }
      const cb = PC.CODEBOOK_SIZE * PC.PHONE_CHANNELS;
      this.codebook = m.speakers.codebook.slice(s * cb, (s + 1) * cb);
      this.spkEmb = m.speakers.additive.slice(s * WG.hidden, (s + 1) * WG.hidden);
      this.kvDirty = true;
    }
    const fi = formantIndex(this.params.formantShift);
    this.fmtEmb = m.speakers.formant.slice(fi * WG.hidden, (fi + 1) * WG.hidden);
    this.preparedSpeaker = s; this.preparedFormant = this.params.formantShift;
  }

  reset() {
    this.analysis = this.synthesis = this.featureNext = 0;
    this.phone.reset(); this.pitch.reset(); this.vocoder.reset(); this.fx.reset();
    this.units.fill(0); this.bins.fill(0); this.feats.fill(0); this.energy.fill(0); this.inst.fill(0); this.corr.fill(0);
    if (this.synth) this.synth.reset(); else this.synth = new WaveformSynth(this.wg.irWindow, this.random);
    this.kvDirty = true;
  }
  async dispose() { await this.phone?.dispose(); await this.pitch?.dispose(); await this.vocoder?.dispose(); }
  get outputFrame() { return this.synthesis; }
  get requiredSamples() { return Math.ceil((this.synthesis + this.chunk + 2) / this.chunk) * this.chunk * 160 + 360; }
  finish() { return this.synth.finish(); }

  private async analyze(read: PcmReader, validFrames: number) {
    const a = this.analysis, n = this.chunk, cap = this.capacity;
    for (let i = 0; i < this.wavInput.length; i++) this.wavInput[i] = read(a * 160 - 40 + i);
    const units = (await this.phone.run({ wav: { data: this.wavInput, dims: [1, 1, this.wavInput.length] } })).units;
    this.onTrace?.({ stage: 'phone', frame: a, data: units, channels: PHONE.phone });
    while (this.featureNext <= a + n) {
      const f = this.featureNext++, slot = f % cap;
      for (let i = 0; i < 560; i++) this.frameInput[i] = read(f * 160 - 200 + i);
      const ft = this.fx.process(this.frameInput);
      this.inst.set(ft.instfreq, slot * 192); this.corr.set(ft.corr, slot * 256); this.energy[slot] = ft.energy;
    }
    for (let i = 0; i <= n; i++) {
      const slot = (a + i) % cap;
      for (let c = 0; c < 192; c++) this.instInput[c * (n + 1) + i] = this.inst[slot * 192 + c];
      for (let c = 0; c < 256; c++) this.corrInput[c * (n + 1) + i] = this.corr[slot * 256 + c];
      this.validInput[i] = a + i < validFrames ? 1 : 0;
    }
    const logits = (await this.pitch.run({ instfreq: { data: this.instInput, dims: [1, 192, n + 1] }, corr_diff: { data: this.corrInput, dims: [1, 256, n + 1] }, valid: { data: this.validInput, dims: [1, 1, n + 1] } })).logits;
    this.onTrace?.({ stage: 'pitch', frame: a, data: logits, channels: PITCH.bins });
    const minBin = midiToBin(this.params.minMidi), maxBin = Math.max(minBin + 3, midiToBin(this.params.maxMidi));
    for (let i = 0; i < n; i++) {
      const slot = (a + i) % cap;
      for (let c = 0; c < PHONE.phone; c++) this.units[slot * PHONE.phone + c] = units[c * n + i];
      for (let c = 0; c < PITCH.bins; c++) this.pitchInput[c] = logits[c * n + i];
      const p = samplePitch(this.pitchInput, minBin, maxBin, 4);
      this.bins[slot] = p.bin; this.feats.set([p.unvoiced, p.half, p.dbl], slot * 3);
    }
    this.analysis += n;
  }

  /** VectorQuantizer.forward (top-k cosine mean over the speaker codebook) + ConverterNetwork RMS norm. */
  private quantizeAndNorm(phone: Float32Array) {
    const N = PHONE.phone, K = Math.min(8, Math.max(0, this.params.vqNeighbors | 0));
    if (K > 0 && this.codebook.length) {
      let nrm = 0; for (let c = 0; c < N; c++) nrm += phone[c] * phone[c];
      nrm = 1 / Math.max(Math.sqrt(nrm), 1e-6);
      const q = this.vqQ; for (let c = 0; c < N; c++) q[c] = phone[c] * nrm;
      const bi = this.vqI, bs = this.vqS; bi.fill(-1, 0, K); bs.fill(-Infinity, 0, K);
      for (let k = 0; k < PC.CODEBOOK_SIZE; k++) {
        let s = 0; const off = k * N; for (let c = 0; c < N; c++) s += q[c] * this.codebook[off + c];
        if (s > bs[K - 1]) { let p = K - 1; while (p > 0 && bs[p - 1] < s) { bs[p] = bs[p - 1]; bi[p] = bi[p - 1]; p--; } bs[p] = s; bi[p] = k; }
      }
      phone.fill(0);
      for (let j = 0; j < K; j++) { const off = bi[j] * N; for (let c = 0; c < N; c++) phone[c] += this.codebook[off + c]; }
      for (let c = 0; c < N; c++) phone[c] /= K;
    }
    let ms = 0; for (let c = 0; c < N; c++) ms += phone[c] * phone[c];
    const g = 1 / Math.sqrt(ms / N + 1.1920929e-7);
    for (let c = 0; c < N; c++) phone[c] *= g;
  }

  async next(read: PcmReader, validFrames = Infinity) {
    const t0 = performance.now(), e = this.synthesis, n = this.chunk, cap = this.capacity;
    while (this.analysis < e + n + 2) await this.analyze(read, validFrames);
    const w = this.wg, p = this.params, C = WG.hidden, T = n + 2, NP = PHONE.phone;
    this.xInput.fill(0);
    for (let i = 0; i < T; i++) {
      const f = e + i;
      if (f >= validFrames) continue;
      const pitchSlot = Math.abs(f - 2) % cap, energySlot = Math.abs(f - 1) % cap, phoneSlot = f % cap;
      const ph = this.phoneVec; ph.set(this.units.subarray(phoneSlot * NP, (phoneSlot + 1) * NP)); this.quantizeAndNorm(ph);
      const target = officialTargetPitch(this.bins[pitchSlot], p);
      const energy = this.energy[energySlot], f1 = this.feats[pitchSlot * 3], f2 = this.feats[pitchSlot * 3 + 1], f3 = this.feats[pitchSlot * 3 + 2];
      for (let c = 0; c < C; c++) {
        let v = w.embedPhone.b[c] + w.embedQPitch[target * C + c] + w.embedPitchFeat.b[c] + this.spkEmb[c] + this.fmtEmb[c];
        const pw = w.embedPhone.w, off = c * NP; for (let k = 0; k < NP; k++) v += pw[off + k] * ph[k];
        const fw = w.embedPitchFeat.w, fo = c * 4; v += fw[fo] * energy + fw[fo + 1] * f1 + fw[fo + 2] * f2 + fw[fo + 3] * f3;
        this.xInput[c * T + i] = v / (1 + Math.exp(-v));
      }
    }
    this.onTrace?.({ stage: 'embedding', frame: e, data: this.xInput.slice(), channels: C });
    const feeds: Record<string, { data: Float32Array; dims: number[] }> = { x: { data: this.xInput, dims: [1, C, T] } };
    if (this.kvDirty) {
      const hd = this.model.setter!.attentionChannels / WG.heads;
      for (let b = 0; b < WG.preBlocks; b++) { feeds[`kt${b}`] = { data: this.kt[b], dims: [1, WG.heads, hd, WG.kvLen] }; feeds[`v${b}`] = { data: this.vv[b], dims: [1, WG.heads, WG.kvLen, hd] }; }
      this.kvDirty = false;
    }
    const v = await this.vocoder.run(feeds);
    for (const [stage, data] of Object.entries(v)) this.onTrace?.({ stage, frame: e, data, channels: stage === 'aperiodicity' ? WG.hop : WG.irLen });
    const inferMs = performance.now() - t0, t1 = performance.now();
    const out: Float32Array[] = [], frames: FrameResult[] = [];
    for (let i = 0; i < n && e + i < validFrames; i++) {
      const f = e + i, slot = f % cap, bin = this.bins[slot];
      const target = officialTargetPitch(bin, p); // Synthesis uses unshifted-in-time F0 in the trainer.
      for (let c = 0; c < WG.irLen; c++) { this.irFrame[c] = v.ir[c * n + i]; this.pfFrame[c] = v.post_filter[c * n + i]; }
      for (let c = 0; c < WG.hop; c++) this.apFrame[c] = v.aperiodicity[c * n + i];
      const audio = this.synth.push(this.irFrame, this.apFrame, this.pfFrame, binToHz(target));
      if (audio) out.push(audio);
      frames.push({ frame: f, bin, hz: binToHz(bin), midi: binToMidi(bin), energy: this.energy[slot], unvoiced: this.feats[slot * 3], half: this.feats[slot * 3 + 1], dbl: this.feats[slot * 3 + 2], targetBin: target, targetHz: binToHz(target), phone: this.units.slice(slot * NP, (slot + 1) * NP) });
    }
    this.synthesis += n;
    const audio = new Float32Array(out.length * WG.hop);
    out.forEach((part, i) => audio.set(part, i * WG.hop));
    return { audio, frames, inferMs, synthMs: performance.now() - t1 };
  }
}
