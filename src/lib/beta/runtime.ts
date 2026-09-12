import type * as ORT from 'onnxruntime-web';
import { buildBetaPhone, buildBetaPitch, buildBetaVocoder, type GruMode } from './graphs';
import { readBetaWeights, type BetaWave } from './layouts';
import type { Paraphernalia } from '../paraphernalia';
import { PitchFeatureExtractor, samplePitch, officialTargetPitch, binToHz, binToMidi, formantIndex } from '../dsp';
import { WaveformSynth } from './synth';
import type { Backend, EngineParams, FrameResult } from '../engine';
import { StatefulSession } from '../session';

export type OrtRuntime = typeof ORT;
export type PcmReader = (sample: number) => number;
export interface BetaTrace { stage: string; frame: number; data: Float32Array; channels: number }

/** Shared by file and microphone modes. Its convolution/GRU states never restart at chunk boundaries. */
export class BetaProcessor {
  private phone!: StatefulSession;
  private pitch!: StatefulSession;
  private vocoder!: StatefulSession;
  private wave!: Pick<BetaWave, 'phone' | 'pitch' | 'features' | 'window'>;
  private fx = new PitchFeatureExtractor();
  private analysis = 0;
  private synthesis = 0;
  private featureNext = 0;
  private capacity: number;
  private phones: Float32Array;
  private bins: Int32Array;
  private feats: Float32Array;
  private energy: Float32Array;
  private inst: Float32Array;
  private corr: Float32Array;
  private wavInput: Float32Array;
  private instInput: Float32Array;
  private corrInput: Float32Array;
  private validInput: Float32Array;
  private xInput: Float32Array;
  private frameInput = new Float32Array(560);
  private pitchInput = new Float32Array(384);
  private synth!: WaveformSynth;
  private irFrame = new Float32Array(512);
  private apFrame = new Float32Array(240);
  private pfFrame = new Float32Array(512);
  graphBytes = 0;
  backend: Backend = 'wasm';
  captured = false;
  onTrace?: (trace: BetaTrace) => void;

  private model: Pick<Paraphernalia, 'speakers'>;
  private constructor(model: Paraphernalia, readonly chunk: number, private params: EngineParams, private random: () => number) {
    this.model = { speakers: model.speakers };
    this.capacity = chunk * 8 + 64;
    this.phones = new Float32Array(this.capacity * 256); this.bins = new Int32Array(this.capacity);
    this.feats = new Float32Array(this.capacity * 3); this.energy = new Float32Array(this.capacity);
    this.inst = new Float32Array(this.capacity * 192); this.corr = new Float32Array(this.capacity * 256);
    this.wavInput = new Float32Array(chunk * 160 + 80);
    this.instInput = new Float32Array(192 * (chunk + 1)); this.corrInput = new Float32Array(256 * (chunk + 1));
    this.validInput = new Float32Array(chunk + 1); this.xInput = new Float32Array(256 * (chunk + 2));
  }

  static async create(model: Paraphernalia, runtime: OrtRuntime, ep: Backend, chunk: number, params: EngineParams, random = Math.random, gruMode?: GruMode, gpuBound = false) {
    if (model.format !== 'beatrice-beta2') throw new Error('BetaProcessor requires beta.2-format weights');
    if (!Number.isInteger(chunk) || chunk < 1 || chunk > 40) throw new Error('Invalid chunk size');
    // WebGPU has no GRU kernel: use the primitive-op unroll. Capture is deliberately disabled.
    gruMode ??= ep === 'webgpu' ? 'primitives' : 'onnx-gru';
    const e = new BetaProcessor(model, chunk, { ...params }, random);
    e.captured = false;
    const weights = readBetaWeights(model.files);
    // Keep only the small CPU conditioning tables after the graphs own their NN weights.
    // Copies are intentional: a subarray would retain the whole ~19 MB decoded generator buffer.
    e.wave = {
      phone: { w: weights.wave.phone.w.slice(), b: weights.wave.phone.b!.slice() },
      pitch: weights.wave.pitch.slice(),
      features: { w: weights.wave.features.w.slice(), b: weights.wave.features.b!.slice() },
      window: weights.wave.window.slice(),
    };
    try {
      // Build and initialize one graph at a time to avoid retaining three serialized model copies.
      let graph = buildBetaPhone(weights.phone, chunk, gruMode);
      e.graphBytes += graph.bytes.length; e.phone = await StatefulSession.create(runtime, graph, ep, gpuBound);
      graph = buildBetaPitch(weights.pitch, chunk);
      e.graphBytes += graph.bytes.length; e.pitch = await StatefulSession.create(runtime, graph, ep, gpuBound);
      graph = buildBetaVocoder(weights.wave, chunk);
      e.graphBytes += graph.bytes.length; e.vocoder = await StatefulSession.create(runtime, graph, ep, gpuBound);
      e.backend = ep; e.reset();
      return e;
    } catch (error) { await e.dispose(); throw error; }
  }

  setParams(params: EngineParams) { this.params = { ...params }; }
  reset() {
    this.analysis = this.synthesis = this.featureNext = 0;
    this.phone.reset(); this.pitch.reset(); this.vocoder.reset(); this.fx.reset();
    this.phones.fill(0); this.bins.fill(0); this.feats.fill(0); this.energy.fill(0); this.inst.fill(0); this.corr.fill(0);
    if (this.synth) this.synth.reset(); else this.synth = new WaveformSynth(this.wave.window, this.random);
  }
  async dispose() { await this.phone?.dispose(); await this.pitch?.dispose(); await this.vocoder?.dispose(); }
  get outputFrame() { return this.synthesis; }
  get requiredSamples() { return Math.ceil((this.synthesis + this.chunk + 2) / this.chunk) * this.chunk * 160 + 360; }
  finish() { return this.synth.finish(); }

  private async analyze(read: PcmReader, validFrames: number) {
    const a = this.analysis, n = this.chunk, cap = this.capacity;
    for (let i = 0; i < this.wavInput.length; i++) this.wavInput[i] = read(a * 160 - 40 + i);
    const units = (await this.phone.run({ wav: { data: this.wavInput, dims: [1, 1, this.wavInput.length] } })).units;
    this.onTrace?.({ stage: 'phone', frame: a, data: units, channels: 256 });
    while (this.featureNext <= a + n) {
      const f = this.featureNext++, slot = f % cap;
      for (let i = 0; i < 560; i++) this.frameInput[i] = read(f * 160 - 200 + i);
      const feature = this.fx.process(this.frameInput);
      this.inst.set(feature.instfreq, slot * 192); this.corr.set(feature.corr, slot * 256); this.energy[slot] = feature.energy;
    }
    for (let i = 0; i <= n; i++) {
      const slot = (a + i) % cap;
      for (let c = 0; c < 192; c++) this.instInput[c * (n + 1) + i] = this.inst[slot * 192 + c];
      for (let c = 0; c < 256; c++) this.corrInput[c * (n + 1) + i] = this.corr[slot * 256 + c];
      this.validInput[i] = a + i < validFrames ? 1 : 0;
    }
    const logits = (await this.pitch.run({ instfreq: { data: this.instInput, dims: [1, 192, n + 1] }, corr_diff: { data: this.corrInput, dims: [1, 256, n + 1] }, valid: { data: this.validInput, dims: [1, 1, n + 1] } })).logits;
    this.onTrace?.({ stage: 'pitch', frame: a, data: logits, channels: 384 });
    const min = Math.max(1, Math.min(336, Math.round((this.params.minMidi - 33) * 8)));
    const max = Math.max(min + 47, Math.min(383, Math.round((this.params.maxMidi - 33) * 8)));
    for (let i = 0; i < n; i++) {
      const slot = (a + i) % cap;
      for (let c = 0; c < 256; c++) this.phones[slot * 256 + c] = units[c * n + i];
      for (let c = 0; c < 384; c++) this.pitchInput[c] = logits[c * n + i];
      const p = samplePitch(this.pitchInput, min, max, 48);
      this.bins[slot] = p.bin;
      this.feats.set([p.unvoiced, p.half, p.dbl], slot * 3);
    }
    this.analysis += n;
  }

  async next(read: PcmReader, validFrames = Infinity) {
    const t0 = performance.now(), e = this.synthesis, n = this.chunk, cap = this.capacity;
    while (this.analysis < e + n + 2) await this.analyze(read, validFrames);
    const w = this.wave, p = this.params, speaker = p.speaker * 256, formant = formantIndex(p.formantShift) * 256;
    const T = n + 2;
    this.xInput.fill(0);
    for (let i = 0; i < T; i++) {
      const f = e + i;
      if (f >= validFrames) continue;
      // F.pad(..., mode="reflect"): f=0 uses pitch[2]/energy[1], f=1 uses pitch[1].
      const pitchSlot = Math.abs(f - 2) % cap, energySlot = Math.abs(f - 1) % cap, phoneSlot = f % cap;
      const target = Math.min(383, officialTargetPitch(this.bins[pitchSlot], p));
      for (let c = 0; c < 256; c++) {
        let v = w.phone.b![c];
        for (let k = 0; k < 256; k++) v += w.phone.w[c * 256 + k] * this.phones[phoneSlot * 256 + k];
        v += w.pitch[target * 256 + c];
        let feature = w.features.b![c] + w.features.w[c * 4] * this.energy[energySlot];
        for (let k = 0; k < 3; k++) feature += w.features.w[c * 4 + k + 1] * this.feats[pitchSlot * 3 + k];
        v += feature + this.model.speakers.additive[speaker + c] + this.model.speakers.formant[formant + c];
        this.xInput[c * T + i] = v / (1 + Math.exp(-v));
      }
    }
    this.onTrace?.({ stage: 'embedding', frame: e, data: this.xInput.slice(), channels: 256 });
    const v = await this.vocoder.run({ x: { data: this.xInput, dims: [1, 256, T] } });
    for (const [stage, data] of Object.entries(v)) this.onTrace?.({ stage, frame: e, data, channels: stage === 'aperiodicity' ? 240 : 512 });
    const inferMs = performance.now() - t0, t1 = performance.now();
    const out: Float32Array[] = [], frames: FrameResult[] = [];
    for (let i = 0; i < n && e + i < validFrames; i++) {
      const f = e + i, slot = f % cap, bin = this.bins[slot];
      const target = Math.min(383, officialTargetPitch(bin, p));
      for (let c = 0; c < 512; c++) { this.irFrame[c] = v.ir[c * n + i]; this.pfFrame[c] = v.post_filter[c * n + i]; }
      for (let c = 0; c < 240; c++) this.apFrame[c] = v.aperiodicity[c * n + i];
      const audio = this.synth.push(this.irFrame, this.apFrame, this.pfFrame, binToHz(target));
      if (audio) out.push(audio);
      frames.push({ frame: f, bin, hz: binToHz(bin), midi: binToMidi(bin), energy: this.energy[slot], unvoiced: this.feats[slot * 3], half: this.feats[slot * 3 + 1], dbl: this.feats[slot * 3 + 2], targetBin: target, targetHz: binToHz(target), phone: this.phones.slice(slot * 256, (slot + 1) * 256) });
    }
    this.synthesis += n;
    const audio = new Float32Array(out.length * 240);
    out.forEach((part, i) => audio.set(part, i * 240));
    return { audio, frames, inferMs, synthMs: performance.now() - t1 };
  }
}