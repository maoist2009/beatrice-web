// Compute-only engine. No DOM, microphone, audio decoding, or unscheduled Promise recursion.
import type { Paraphernalia } from './paraphernalia';
import type { ModelFormat } from './formats';
import { loadOrt, prepareWebGpu, webgpuAdapterInfo } from './ort-runtime';
import { CooperativePump } from './cooperative-pump';
import { sessionResources, type OrtRuntime, type Backend } from './session';
import { DEFAULT_PARAMS, initialStats, type EngineOptions, type EngineParams, type EngineStats, type FrameResult, type ComponentReport } from './runtime-types';

export interface StreamProcessor {
  readonly chunk: number; graphBytes: number; backend: Backend; captured: boolean;
  setParams(params: EngineParams): void; reset(): void; dispose(): Promise<void>;
  readonly outputFrame: number; readonly requiredSamples: number;
  finish(): Float32Array;
  next(read: (sample: number) => number, validFrames?: number): Promise<{ audio: Float32Array; frames: FrameResult[]; inferMs: number; synthMs: number }>;
}
export type ProcessorFactory = (model: Paraphernalia, ort: OrtRuntime, ep: Backend, chunk: number, params: EngineParams, gpuBound: boolean, capture: boolean) => Promise<StreamProcessor>;
const yieldTask = () => new Promise<void>(resolve => setTimeout(resolve, 0));

export class StreamEngine {
  backend: Backend = 'wasm'; graphBytes = 0; captured = false; gpuBound = false;
  reports: ComponentReport[] = []; stats: EngineStats = initialStats();
  params = { ...DEFAULT_PARAMS };
  overruns = 0;
  onFrames: ((frames: FrameResult[]) => void) | null = null;
  onStats: ((stats: EngineStats) => void) | null = null;
  onAudio: ((audio: Float32Array) => void) | null = null;
  onFailure: ((message: string) => void) | null = null;
  /** Overrun safe-pause: mic input stops, but the engine is retained and restartable. */
  onOverrun: ((count: number, reason: string) => void) | null = null;
  private processor!: StreamProcessor;
  private ring = new Float32Array(16000 * 2);
  private snapshot = new Float32Array(160 * 86);
  private received = 0;
  private epoch = 0;
  private mode: 'idle' | 'live' | 'file' | 'disposed' = 'idle';
  private pump: CooperativePump | null = null;
  private activeFile: Promise<{ audio: Float32Array; sampleRate: number; frames: FrameResult[] }> | null = null;
  private disposing: Promise<void> | null = null;
  private lastDisplay = 0;
  private constructor(readonly format: ModelFormat, private speakers: number, private options: EngineOptions) {}

  static async create(model: Paraphernalia, options: EngineOptions, factory: ProcessorFactory, format: ModelFormat) {
    const e = new StreamEngine(format, model.speakers.nSpeakers, options);
    const ort = await loadOrt();
    if (options.backend === 'webgpu' || options.backend === 'auto') {
      await prepareWebGpu(ort, { forceFp32: options.gpuPrecision === 'fp32', onLog: options.onLog });
    }
    const warmup = async (c: StreamProcessor) => {
      let ms = 0;
      for (let n = 0; n < 3; n++) { await yieldTask(); const m = await c.next(() => 0, c.chunk * 4); ms = m.inferMs; }
      return ms;
    };
    const measure = async (c: StreamProcessor) => {
      let ms = 0;
      for (let n = 0; n < 5; n++) { await yieldTask(); const m = await c.next(() => 0, c.chunk * 4); ms += m.inferMs + m.synthMs; }
      return ms / 5;
    };
    const adopt = (candidate: StreamProcessor, ep: Backend, bound: boolean, warmMs: number) => {
      candidate.reset(); e.processor = candidate; e.backend = ep; e.gpuBound = bound;
      e.graphBytes = candidate.graphBytes; e.captured = candidate.captured;
      e.stats = { ...initialStats(), backend: ep, chunkFrames: candidate.chunk, graphBytes: e.graphBytes, ioMode: bound ? 'gpu-bound' : 'cpu-tensors', capture: candidate.captured, resources: sessionResources(), overruns: e.overruns };
      if (ep === 'webgpu') {
        const info = webgpuAdapterInfo() ?? (ort.env.webgpu as unknown as { adapter?: { info?: { vendor?: string; architecture?: string; description?: string } } }).adapter?.info;
        if (info) e.stats.adapter = { vendor: info.vendor ?? '', architecture: info.architecture ?? '', description: info.description ?? '' };
        const dev = (ort.env.webgpu as unknown as { device?: { features?: { has(f: string): boolean } } }).device;
        const f16 = dev?.features?.has?.('shader-f16');
        options.onLog?.(`WebGPU kernel precision: ${f16 === undefined ? 'device not exposed by ORT' : f16 ? 'fp16-capable device (ORT may use f16 kernels)' : 'fp32 device (no shader-f16)'}; graph capture ${candidate.captured ? 'ON (replaying recorded commands)' : 'OFF (per-op dispatch)'}`);
      }
      options.onLog?.(`Ready in isolated worker: ${format}, ${candidate.chunk * 10} ms chunk; warmup networks ${warmMs.toFixed(1)} ms. Warmup is not an audio-latency measurement.`);
    };
    e.reports = Object.entries(model.sizes).filter(([, bytes]) => bytes > 0).map(([name, bytes]) => ({ name: `${name}.bin`, bytes, status: 'verified', detail: `${format} export layout. Not a physical-device latency/parity guarantee.` }));

    if (options.backend === 'auto') {
      // Measure BOTH backends on this device and keep the faster one. On Adreno 6xx this
      // typically picks WASM (GPU dispatch overhead > GPU compute for this network); on desktop
      // GPUs it typically picks WebGPU. No more guessing from the user.
      const gpuBound = options.gpuMode !== 'compatible';
      const cap = !!options.graphCapture;
      const bench = async (ep: Backend, bound: boolean, capture: boolean): Promise<{ candidate: StreamProcessor; ms: number } | null> => {
        let candidate: StreamProcessor | null = null;
        try {
          options.onLog?.(`Benchmarking ${ep}${bound ? ' (bounded GPU I/O)' : ''}${capture ? ' (capture ON)' : ''}…`);
          await yieldTask();
          candidate = await factory(model, ort, ep, options.chunkFrames, e.params, bound, capture);
          const warm = await warmup(candidate);
          const ms = await measure(candidate);
          options.onLog?.(`${ep}: ${ms.toFixed(1)} ms/chunk after warmup (${warm.toFixed(1)} ms warmup)`);
          return { candidate, ms };
        } catch (error) {
          try { await candidate?.dispose(); } catch (cleanup) { options.onLog?.(`Cleanup: ${String(cleanup)}`); }
          options.onLog?.(`${ep} benchmark unavailable: ${String(error)}`);
          return null;
        }
      };
      const gpu = await bench('webgpu', gpuBound, cap);
      const cpu = await bench('wasm', false, false);
      const useGpu = !!gpu && (!cpu || gpu.ms <= cpu.ms);
      options.onLog?.(`auto-backend measured: WebGPU ${gpu ? `${gpu.ms.toFixed(1)} ms/chunk` : 'n/a'} vs WASM ${cpu ? `${cpu.ms.toFixed(1)} ms/chunk` : 'n/a'} → ${useGpu ? 'WebGPU' : 'WASM'}`);
      const loser = useGpu ? cpu : gpu;
      try { await loser?.candidate.dispose(); } catch { /* bench cleanup best-effort */ }
      const winner = useGpu ? gpu : cpu;
      if (!winner) throw new Error('No backend passed the auto benchmark');
      adopt(winner.candidate, winner.candidate.backend, useGpu && gpuBound, winner.ms);
      return e;
    }

    const captureWanted = !!options.graphCapture;
    const attempts: { ep: Backend; bound: boolean; capture: boolean }[] = options.backend === 'webgpu'
      ? [...(options.gpuMode === 'compatible' ? [] : captureWanted ? [{ ep: 'webgpu' as const, bound: true, capture: true }, { ep: 'webgpu' as const, bound: true, capture: false }] : [{ ep: 'webgpu' as const, bound: true, capture: false }]), { ep: 'webgpu', bound: false, capture: false }, { ep: 'wasm', bound: false, capture: false }]
      : [{ ep: 'wasm', bound: false, capture: false }];
    let last: unknown;
    for (const { ep, bound, capture: attemptCapture } of attempts) {
      let candidate: StreamProcessor | null = null;
      try {
        options.onLog?.(`Initializing ${ep}, ${bound ? 'bounded GPU I/O' : 'CPU I/O'}, capture ${attemptCapture ? 'ON' : 'OFF'}`);
        await yieldTask();
        candidate = await factory(model, ort, ep, options.chunkFrames, e.params, bound, attemptCapture);
        const warmMs = await warmup(candidate);
        adopt(candidate, ep, bound, warmMs);
        break;
      } catch (error) {
        last = error;
        try { await candidate?.dispose(); } catch (cleanup) { options.onLog?.(`Cleanup: ${String(cleanup)}`); }
        options.onLog?.(`${ep}/${bound ? 'bound' : 'standard'} unavailable: ${String(error)}`);
        await yieldTask();
      }
    }
    if (!e.processor) throw new Error(`No backend passed initialization/warmup: ${String(last)}`);
    return e;
  }
  setParams(update: Partial<EngineParams>) {
    if (this.mode === 'file' || this.mode === 'disposed') return;
    const p = { ...this.params, ...update };
    if (!Number.isInteger(p.speaker) || p.speaker < 0 || p.speaker >= this.speakers) throw new Error('Invalid speaker');
    for (const [key, value] of Object.entries(p)) if (typeof value === 'number' && !Number.isFinite(value)) throw new Error(`Invalid parameter ${key}`);
    if (p.inputGain < 0 || p.outputGain < 0 || p.vqNeighbors < 0 || p.vqNeighbors > 8 || p.minMidi >= p.maxMidi) throw new Error('Invalid gain, VQ, or pitch range');
    this.params = p; // Applied only immediately before the next chunk, never midway through a run.
  }
  private update(part: { inferMs: number; synthMs: number }, force = false) {
    Object.assign(this.stats, { inferMs: part.inferMs, synthMs: part.synthMs, rtf: (part.inferMs + part.synthMs) / (this.processor.chunk * 10), frames: this.processor.outputFrame, measured: true, overruns: this.overruns, resources: sessionResources() });
    const now = performance.now();
    if (force || now - this.lastDisplay > 150) { this.lastDisplay = now; this.onStats?.({ ...this.stats }); }
  }
  /** Overrun: fall behind real time → pause safely, keep the engine (no crash, no rebasing). */
  private softPause(reason: string) {
    if (this.mode !== 'live') return;
    this.overruns++;
    this.mode = 'idle'; // CooperativePump.ready() becomes false → pump sleeps; engine retained
    this.stats.overruns = this.overruns;
    this.options.onLog?.(`Overrun #${this.overruns}: ${reason}; inference paused safely (engine retained, resume with Live mic)`);
    this.onStats?.({ ...this.stats });
    this.onOverrun?.(this.overruns, reason);
  }
  async start() {
    await this.stop();
    if (this.mode === 'disposed') throw new Error('Engine is disposed');
    this.processor.reset(); this.processor.setParams(this.params);
    this.ring.fill(0); this.received = 0; this.mode = 'live'; this.stats.measured = false;
    const epoch = this.epoch;
    this.pump = new CooperativePump(
      () => this.mode === 'live' && epoch === this.epoch && this.received >= this.processor.requiredSamples,
      async () => {
        // Single monotonic input timeline. Never add a ringBase to only half the comparisons.
        const behind = this.received - this.processor.outputFrame * 160;
        if (behind > Math.max(16000, this.processor.chunk * 160 * 6)) { this.softPause('inference >1 s behind real time'); return; }
        // Capture the small window BEFORE awaiting GPU work. Otherwise the live ring can wrap while a
        // slow device is executing phone layers and the subsequent pitch DSP reads overwritten PCM.
        const start = Math.max(0, this.processor.outputFrame * 160 - 560);
        const end = this.processor.requiredSamples;
        if (start < this.received - this.ring.length || end > this.received || end - start > this.snapshot.length) { this.softPause('input backlog exceeded the snapshot budget'); return; }
        for (let i = start; i < end; i++) this.snapshot[i - start] = this.ring[i % this.ring.length];
        this.processor.setParams(this.params);
        const part = await this.processor.next(i => {
          if (i < 0) return 0;
          if (i < start || i >= end) throw new Error('PCM request outside the captured inference window');
          return this.snapshot[i - start];
        });
        if (epoch !== this.epoch || this.mode !== 'live') return;
        this.update(part);
        this.onFrames?.(part.frames);
        if (this.params.convert && part.audio.length) this.onAudio?.(part.audio);
      },
      error => { this.epoch++; this.mode = 'idle'; this.options.onLog?.(`ERROR: ${String(error)}`); this.onFailure?.(String(error)); },
    );
  }
  push(samples: Float32Array, startSample: number) {
    if (this.mode !== 'live') return;
    if (startSample !== this.received || samples.length > 160 * 40) throw new Error('Live PCM sequence lost; restart microphone to reset model state');
    for (let i = 0; i < samples.length; i++) this.ring[(this.received + i) % this.ring.length] = samples[i];
    this.received += samples.length;
    this.pump!.wake(); // No wake until data is actually available; no immediate recursive finally().
  }
  convert16(pcm: Float32Array, progress?: (fraction: number, message: string) => void) {
    if (this.mode !== 'idle' || this.activeFile || this.disposing) return Promise.reject(new Error('Engine is busy'));
    const epoch = ++this.epoch, p = { ...this.params };
    this.mode = 'file';
    const run = async () => {
      const count = Math.ceil(pcm.length / 160);
      if (count < 3 || count > 6000) throw new Error('Use clips between 30 ms and 60 seconds');
      for (const v of pcm) if (!Number.isFinite(v)) throw new Error('Invalid input PCM');
      this.processor.reset(); this.processor.setParams(p);
      const result = new Float32Array(count * 240), frames: FrameResult[] = [];
      let offset = 0;
      while (this.processor.outputFrame < count) {
        await yieldTask(); // Let cancel, worker heartbeat and pending control messages run every chunk.
        if (epoch !== this.epoch) throw new DOMException('Conversion cancelled', 'AbortError');
        const part = await this.processor.next(i => i < 0 || i >= pcm.length ? 0 : Math.fround(pcm[i] * p.inputGain), count);
        if (epoch !== this.epoch) throw new DOMException('Conversion cancelled', 'AbortError');
        if (offset + part.audio.length > result.length) throw new Error('Synthesis output exceeds preallocated audio');
        result.set(part.audio, offset); offset += part.audio.length;
        frames.push(...part.frames); if (frames.length > 600) frames.splice(0, frames.length - 600);
        this.update(part);
        progress?.(Math.min(0.99, this.processor.outputFrame / count), `Converting ${Math.min(count, this.processor.outputFrame)}/${count} frames`);
      }
      const tail = this.processor.finish();
      if (offset + tail.length !== result.length) throw new Error('Unexpected synthesis length');
      result.set(tail, offset);
      for (let i = 0; i < result.length; i++) result[i] *= p.outputGain;
      this.update({ inferMs: this.stats.inferMs, synthMs: this.stats.synthMs }, true);
      const audio = result.subarray(0, Math.round(pcm.length * 1.5));
      progress?.(1, 'Synthesis complete');
      return { audio, sampleRate: 24000, frames };
    };
    this.activeFile = run().finally(() => { this.mode = 'idle'; this.activeFile = null; });
    return this.activeFile;
  }
  async stop() {
    if (this.mode === 'disposed') return;
    this.epoch++;
    if (this.mode === 'live') this.mode = 'idle';
    const pump = this.pump; this.pump = null;
    await pump?.stop();
    try { await this.activeFile; } catch { /* Cancellation is reported to the caller. */ }
    this.mode = 'idle';
  }
  dispose(): Promise<void> {
    if (this.disposing) return this.disposing;
    this.disposing = (async () => {
      await this.stop(); this.mode = 'disposed';
      await this.processor.dispose(); this.ring = new Float32Array(0); this.snapshot = new Float32Array(0);
      this.onAudio = this.onFrames = this.onStats = this.onFailure = this.onOverrun = null;
    })();
    return this.disposing;
  }
}
