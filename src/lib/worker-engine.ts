import InferenceWorker from './inference.worker?worker&inline';
import type { Paraphernalia } from './paraphernalia';
import type { ModelFormat } from './formats';
import { DEFAULT_PARAMS, initialStats, type Backend, type ComponentReport, type EngineOptions, type EngineParams, type EngineStats, type FrameResult, type VoiceEngine } from './runtime-types';
import type { WorkerAudio, WorkerEvent, WorkerReady, WorkerRequest } from './worker-protocol';
import { offlineResample } from './wav';
import { WORKLET_SOURCE } from './worklet-source';
import { rememberRun } from './last-run';

type Request = WorkerRequest extends infer R ? R extends { id: number } ? Omit<R, 'id'> : never : never;
interface Pending { resolve: (value: unknown) => void; reject: (error: Error) => void; type: WorkerRequest['type'] }

/** Main thread only owns audio I/O and bounded messages. A stuck worker can be terminated without UI loss. */
export class WorkerEngine implements VoiceEngine {
  backend: Backend = 'wasm'; graphBytes = 0; captured = false;
  reports: ComponentReport[] = []; stats: EngineStats = initialStats();
  onFrames: ((frames: FrameResult[]) => void) | null = null;
  onStats: ((stats: EngineStats) => void) | null = null;
  onDiag: VoiceEngine['onDiag'] = null;
  onState: VoiceEngine['onState'] = null;
  private worker: Worker;
  private pending = new Map<number, Pending>();
  private seq = 0;
  private dead = false;
  private mode: 'initializing' | 'idle' | 'starting' | 'live' | 'file' = 'initializing';
  private generation = 0;
  private params = { ...DEFAULT_PARAMS };
  private watchdog: ReturnType<typeof setInterval>;
  private lastCompute = performance.now();
  private lastStage = 'Starting inference worker';
  private progress: ((fraction: number, text: string) => void) | null = null;
  private context: AudioContext | null = null;
  private stream: MediaStream | null = null;
  private source: MediaStreamAudioSourceNode | null = null;
  private node: AudioWorkletNode | null = null;
  private queuedPcm = 0;
  private disposing: Promise<void> | null = null;
  private paramsInFlight = false;
  private paramsDirty = false;
  private lastSaved = 0;

  private constructor(readonly format: ModelFormat, private options: EngineOptions) {
    this.breadcrumb(true);
    this.worker = new InferenceWorker();
    this.worker.onmessage = (event: MessageEvent<WorkerEvent>) => this.handle(event.data);
    this.worker.onerror = event => { event.preventDefault(); this.fail(`Inference worker failed: ${event.message}`); };
    this.worker.onmessageerror = () => this.fail('Unable to receive the inference result');
    this.watchdog = setInterval(() => {
      const timeout = this.mode === 'initializing' ? 120000 : 30000;
      if (this.mode !== 'idle' && performance.now() - this.lastCompute > timeout) this.fail(`No inference progress for ${timeout / 1000}s at: ${this.lastStage}. Worker terminated. Try compatibility mode or WASM.`);
    }, 1000);
  }

  static async create(model: Paraphernalia, options: EngineOptions) {
    if (typeof Worker === 'undefined') throw new Error('This app requires dedicated Workers for safe inference');
    const e = new WorkerEngine(model.format, options);
    const abort = () => e.kill('Engine initialization cancelled');
    options.signal?.addEventListener('abort', abort, { once: true });
    try {
      options.signal?.throwIfAborted();
      const ready = await e.rpc<WorkerReady>({ type: 'init', options: { backend: options.backend, ctxFrames: options.ctxFrames, chunkFrames: options.chunkFrames, gpuMode: options.gpuMode }, files: { ...model.files, images: {}, extras: {} } });
      e.stats = ready.stats; e.reports = ready.reports; e.backend = ready.stats.backend; e.graphBytes = ready.stats.graphBytes;
      e.mode = 'idle'; e.breadcrumb(false); return e;
    } catch (error) { e.kill(); throw error; }
    finally { options.signal?.removeEventListener('abort', abort); }
  }

  private rpc<T = void>(request: Request, transfer: Transferable[] = []): Promise<T> {
    if (this.dead) return Promise.reject(new Error('Worker is stopped; rebuild the engine'));
    const id = ++this.seq;
    return new Promise<T>((resolve, reject) => {
      this.pending.set(id, { resolve: resolve as (value: unknown) => void, reject, type: request.type });
      try { this.worker.postMessage({ ...request, id }, transfer); }
      catch (error) { this.pending.delete(id); reject(error); }
    });
  }

  private handle(event: WorkerEvent) {
    if (this.dead) return;
    if (event.type === 'reply') {
      const p = this.pending.get(event.id); if (!p) return;
      this.pending.delete(event.id);
      if (p.type !== 'pcm' && p.type !== 'params') this.lastCompute = performance.now();
      if (event.error) { const error = new Error(event.error); error.name = event.name ?? 'Error'; p.reject(error); }
      else p.resolve(event.value);
    } else if (event.type === 'stage') {
      this.lastStage = event.text; this.lastCompute = performance.now(); this.stats.stage = event.text;
      this.options.onStage?.(event.text);
      if (performance.now() - this.lastSaved > 1000) this.breadcrumb(true);
    } else if (event.type === 'stats') {
      this.lastCompute = performance.now();
      this.stats = { ...event.stats, sampleRate: this.context?.sampleRate ?? this.stats.sampleRate, underruns: this.stats.underruns, queue: this.stats.queue, stage: this.lastStage };
      this.onStats?.({ ...this.stats });
    } else if (event.type === 'frames') {
      if (this.mode === 'live') this.onFrames?.(event.frames);
    } else if (event.type === 'audio') {
      this.lastCompute = performance.now();
      if (this.mode === 'live') this.node?.port.postMessage({ type: 'out', frames: event.audio }, [event.audio.buffer]);
    } else if (event.type === 'progress') {
      this.lastCompute = performance.now(); this.progress?.(0.05 + event.fraction * 0.9, event.message);
    } else if (event.type === 'log') this.options.onLog?.(event.text);
    else if (event.type === 'fatal') this.fail(event.text);
  }

  setParams(update: Partial<EngineParams>) {
    if (this.mode === 'file' || this.dead) return;
    this.params = { ...this.params, ...update };
    this.paramsDirty = true; this.flushParams();
    this.node?.port.postMessage({ type: 'params', inputGain: this.params.inputGain, outputGain: this.params.outputGain, monitor: this.params.monitor && !this.params.convert, batchFrames: Math.min(4, this.stats.chunkFrames ?? 4) });
  }
  private flushParams() {
    if (this.paramsInFlight || !this.paramsDirty || this.dead) return;
    this.paramsInFlight = true; this.paramsDirty = false;
    void this.rpc({ type: 'params', params: this.params }).catch(e => this.fail(String(e))).finally(() => { this.paramsInFlight = false; if (this.paramsDirty) this.flushParams(); });
  }

  async convertBuffer(pcm: Float32Array, rate: number, progress?: (fraction: number, text: string) => void) {
    if (this.mode !== 'idle' || this.dead) throw new Error('Stop the current task before converting');
    const generation = ++this.generation;
    this.mode = 'file'; this.onState?.('file'); this.lastCompute = performance.now();
    this.breadcrumb(true);
    this.progress = progress ?? null;
    try {
      progress?.(0, 'Resampling input in the browser');
      const input = await offlineResample(pcm, rate, 16000);
      if (generation !== this.generation) throw new DOMException('Cancelled', 'AbortError');
      await this.rpc({ type: 'params', params: this.params });
      const converted = await this.rpc<WorkerAudio>({ type: 'file', pcm: input }, [input.buffer]);
      if (generation !== this.generation) throw new DOMException('Cancelled', 'AbortError');
      progress?.(0.96, 'Rendering output');
      const audio = await offlineResample(converted.audio, converted.sampleRate, rate);
      if (generation !== this.generation) throw new DOMException('Cancelled', 'AbortError');
      progress?.(1, 'Ready');
      return { audio, sampleRate: rate, frames: converted.frames };
    } finally {
      this.progress = null;
      if (!this.dead) { this.mode = 'idle'; this.breadcrumb(false); this.onState?.('idle'); }
    }
  }

  async start(deviceId?: string) {
    if (this.mode !== 'idle' || this.dead) throw new Error('Engine is busy or stopped');
    this.mode = 'starting'; this.lastCompute = performance.now();
    this.breadcrumb(true);
    this.stats = { ...this.stats, measured: false, frames: 0, rtf: 0, underruns: 0, queue: 0 };
    this.onStats?.({ ...this.stats });
    const generation = ++this.generation;
    let context: AudioContext | null = null, stream: MediaStream | null = null;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: { deviceId: deviceId ? { exact: deviceId } : undefined, channelCount: 1, echoCancellation: false, autoGainControl: false, noiseSuppression: false } });
      if (generation !== this.generation || this.dead) throw new DOMException('Cancelled', 'AbortError');
      this.stream = stream;
      await this.rpc({ type: 'params', params: this.params }); await this.rpc({ type: 'start' });
      context = new AudioContext({ latencyHint: 'interactive' }); this.context = context;
      const url = URL.createObjectURL(new Blob([WORKLET_SOURCE], { type: 'application/javascript' }));
      try { await context.audioWorklet.addModule(url); } finally { URL.revokeObjectURL(url); }
      if (generation !== this.generation || this.dead) throw new DOMException('Cancelled', 'AbortError');
      const node = new AudioWorkletNode(context, 'beatrice-processor', { numberOfInputs: 1, numberOfOutputs: 1, outputChannelCount: [1] });
      this.node = node; this.queuedPcm = 0;
      node.port.onmessage = ({ data }) => {
        if (generation !== this.generation || this.dead) return;
        if (data.type === 'in') {
          if (this.queuedPcm >= 12) { this.fail('Input message backlog exceeded 12 packets; paused safely. Try a larger chunk or file mode.'); return; }
          this.queuedPcm++;
          const pcm = data.samples as Float32Array;
          void this.rpc({ type: 'pcm', pcm, startSample: data.frameIndex * 160 }, [pcm.buffer]).catch(e => { if (!this.dead) this.fail(String(e)); }).finally(() => { this.queuedPcm = Math.max(0, this.queuedPcm - 1); });
        } else if (data.type === 'stats') {
          this.stats.underruns = data.underruns; this.stats.queue = data.queue; this.onStats?.({ ...this.stats });
        }
      };
      node.onprocessorerror = () => this.fail('AudioWorklet failed; microphone stopped');
      this.source = context.createMediaStreamSource(stream); this.source.connect(node); node.connect(context.destination);
      this.stats.sampleRate = context.sampleRate; this.setParams(this.params);
      await context.resume();
      this.mode = 'live'; this.lastCompute = performance.now(); this.onState?.('live');
      this.breadcrumb(true);
    } catch (error) {
      stream?.getTracks().forEach(t => t.stop()); if (context && context.state !== 'closed') void context.close();
      this.stop(); throw error;
    }
  }

  private closeAudio() {
    this.node?.port.close(); this.node?.disconnect(); this.node = null;
    this.source?.disconnect(); this.source = null;
    this.stream?.getTracks().forEach(t => t.stop()); this.stream = null;
    const ctx = this.context; this.context = null;
    if (ctx && ctx.state !== 'closed') void ctx.close();
  }
  stop() {
    this.generation++; this.closeAudio(); this.mode = 'idle';
    if (!this.dead) {
      const timeout = setTimeout(() => this.fail('Worker did not acknowledge stop; terminated. Rebuild to continue.'), 3500);
      void this.rpc({ type: 'stop' }).catch(() => {}).finally(() => clearTimeout(timeout));
    }
    this.onState?.('idle');
    this.breadcrumb(false);
  }
  private breadcrumb(pending: boolean) {
    this.lastSaved = performance.now();
    rememberRun({ pending, mode: this.mode, stage: this.lastStage, backend: this.options.backend });
  }
  private fail(message: string) {
    if (this.dead) return;
    this.options.onLog?.(`ERROR: ${message} [last stage: ${this.lastStage}]`);
    this.kill(message); this.onState?.('error');
  }
  private kill(message = 'Inference worker terminated') {
    if (this.dead) return;
    this.dead = true; this.generation++; clearInterval(this.watchdog); this.closeAudio();
    this.worker.terminate(); this.worker.onmessage = null; this.worker.onerror = null;
    for (const p of this.pending.values()) p.reject(new Error(message));
    this.pending.clear(); this.mode = 'idle'; this.queuedPcm = 0;
    this.breadcrumb(false);
  }
  dispose() {
    if (this.disposing) return this.disposing;
    this.disposing = (async () => {
      if (this.dead) return;
      this.generation++; this.closeAudio();
      let timer: ReturnType<typeof setTimeout> | undefined;
      try { await Promise.race([this.rpc({ type: 'dispose' }), new Promise<void>(resolve => { timer = setTimeout(resolve, 1500); })]); }
      catch { /* Worker termination is the final safety boundary. */ }
      finally { clearTimeout(timer); this.kill(); this.onFrames = this.onStats = this.onDiag = this.onState = null; }
    })();
    return this.disposing;
  }
}