import type * as ORT from 'onnxruntime-web';
import type { ResourceStats } from './runtime-types';
import { injectedWebGpuDevice } from './ort-runtime';

export type OrtRuntime = typeof ORT;
export type Backend = 'webgpu' | 'wasm';
export interface IoSpec { name: string; dims: number[] }
export interface StateSpec { input: string; output: string; dims: number[]; init?: number }
export interface StreamingGraph { bytes: Uint8Array; states: StateSpec[]; inputs: IoSpec[]; outputs: IoSpec[] }
export interface RunInput { data: Float32Array; dims: number[] }

interface GBuffer { destroy(): void; mapAsync(mode: number): Promise<void>; getMappedRange(): ArrayBuffer; unmap(): void }
interface GDevice {
  limits: { maxBufferSize: number; maxStorageBufferBindingSize: number };
  lost: Promise<{ message: string }>;
  createBuffer(descriptor: { size: number; usage: number; label?: string }): GBuffer;
  pushErrorScope(filter: 'validation' | 'out-of-memory'): void;
  popErrorScope(): Promise<{ message: string } | null>;
  queue: { writeBuffer(buffer: GBuffer, offset: number, data: ArrayBufferLike, dataOffset: number, bytes: number): void; submit(commands: unknown[]): void; onSubmittedWorkDone(): Promise<void> };
  createCommandEncoder(): { clearBuffer(buffer: GBuffer): void; copyBufferToBuffer(a: GBuffer, ao: number, b: GBuffer, bo: number, size: number): void; finish(): unknown };
}
const align = (n: number) => Math.ceil(n / 16) * 16;
const elements = (dims: number[]) => {
  if (dims.some(n => !Number.isInteger(n) || n <= 0)) throw new Error('Graph I/O must have positive static dimensions');
  return dims.reduce((a, b) => a * b, 1);
};
const resources: ResourceStats = { sessions: 0, gpuBuffers: 0, gpuBytes: 0, cpuIoBytes: 0, retainedGraphBytes: 0, inFlight: 0, runs: 0 };
export const sessionResources = (): ResourceStats => ({ ...resources });
let reportStage: (stage: string) => void = () => {};
export function observeSessionStages(callback: (stage: string) => void) { reportStage = callback; }
const MAX_OWNED_GPU = 32 * 1024 * 1024;

/** Owns exactly one session and its fixed I/O. Graph bytes are NOT retained. Graph capture is disabled. */
export class StatefulSession {
  readonly gpuBound: boolean;
  private readonly specs: Omit<StreamingGraph, 'bytes'>;
  private session: ORT.InferenceSession | null;
  private device: GDevice | null = null;
  private buffers = new Map<string, { buffer: GBuffer; bytes: number; allocated: number }>();
  private staging: GBuffer | null = null;
  private stagingBytes = 0;
  private feeds: Record<string, ORT.Tensor> = {};
  private fetches: Record<string, ORT.Tensor> = {};
  private cpu = new Map<string, Float32Array>();
  private outputs: Record<string, Float32Array> = {};
  private outputOffsets: { name: string; bytes: number; offset: number }[] = [];
  private supplied = new Set<string>();
  private stateOutputs: Set<string>;
  private task: Promise<Record<string, Float32Array>> | null = null;
  private disposing: Promise<void> | null = null;
  private fault: Error | null = null;
  private localCpuBytes = 0;
  private loss = { active: true, message: '' };
  private constructor(private ort: OrtRuntime, session: ORT.InferenceSession, graph: StreamingGraph, gpuBound: boolean, readonly captured: boolean) {
    this.session = session; this.gpuBound = gpuBound;
    this.specs = { states: graph.states, inputs: graph.inputs, outputs: graph.outputs };
    this.stateOutputs = new Set(graph.states.map(s => s.output));
    resources.sessions++;
  }

  static async create(ort: OrtRuntime, graph: StreamingGraph, ep: Backend, gpuBound = false, capture = false): Promise<StatefulSession> {
    if (graph.bytes.byteLength > 64 * 1024 * 1024) throw new Error('Individual graph exceeds the 64 MiB safety budget');
    const label = graph.outputs.find(o => !graph.states.some(s => s.output === o.name))?.name ?? 'model';
    // Capture needs static shapes + fully GPU-bound I/O; only attempted in bound WebGPU mode.
    const wantCapture = capture && ep === 'webgpu' && gpuBound;
    const injected = ep === 'webgpu' ? injectedWebGpuDevice() : null;
    const open = (cap: boolean) => ort.InferenceSession.create(graph.bytes, {
      // The app-owned device (FP32-forced, no shader-f16) is passed through the EP option;
      // ort.env.webgpu.device assignment alone is ignored by some ORT builds (#26107).
      executionProviders: [injected ? { name: 'webgpu' as const, device: injected } : ep],
      graphOptimizationLevel: 'all',
      // Records the command sequence on the first run and replays it afterwards — the
      // mechanism that removes per-chunk dispatch overhead on WebGPU. Drivers that cannot
      // capture throw here; we retry once with capture OFF instead of failing the build.
      enableGraphCapture: cap,
      // Android Chrome reuses one renderer across same-tab reloads. The default WASM arena
      // retains freed linear memory, so every rebuild used to stack another arena → OOM.
      ...(ep === 'wasm' ? { enableCpuMemArena: false, enableMemPattern: false } : {}),
      ...(ep === 'webgpu' && gpuBound ? { preferredOutputLocation: 'gpu-buffer' as const } : {}),
    } as ORT.InferenceSession.SessionOptions);
    let session: ORT.InferenceSession; let captured = false;
    reportStage(`Initialize ${label} (${ep}, capture ${wantCapture ? 'ON (experimental)' : 'OFF'})`);
    if (wantCapture) {
      try { session = await open(true); captured = true; }
      catch (error) {
        reportStage(`Graph capture rejected (${String(error).slice(0, 90)}); retrying with capture OFF`);
        session = await open(false);
      }
    } else session = await open(false);
    const value = new StatefulSession(ort, session, graph, ep === 'webgpu' && gpuBound, captured);
    try {
      if (value.gpuBound) await value.bindGpu(); else value.bindCpu();
      value.reset();
      return value;
    } catch (error) { await value.dispose(); throw error; }
  }

  private array(n: number) { const a = new Float32Array(n); this.localCpuBytes += a.byteLength; resources.cpuIoBytes += a.byteLength; return a; }
  private bindCpu() {
    for (const input of this.specs.inputs) {
      const data = this.array(elements(input.dims)); this.cpu.set(input.name, data);
      this.feeds[input.name] = new this.ort.Tensor('float32', data, input.dims);
    }
    for (const output of this.specs.outputs) if (!this.stateOutputs.has(output.name)) this.outputs[output.name] = this.array(elements(output.dims));
  }

  private async bindGpu() {
    const device = ((this.ort.env.webgpu as unknown as { device: GDevice | undefined }).device ?? injectedWebGpuDevice()) as GDevice | null;
    if (!device) throw new Error('ORT WebGPU device is unavailable');
    this.device = device;
    const bindings = [...this.specs.inputs, ...this.specs.outputs];
    if (new Set(bindings.map(s => s.name)).size !== bindings.length) throw new Error('Graph has duplicate I/O buffer names');
    const bytes = bindings.reduce((sum, s) => sum + align(elements(s.dims) * 4), 0);
    const stagingBytes = this.specs.outputs.filter(o => !this.stateOutputs.has(o.name)).reduce((sum, o) => sum + align(elements(o.dims) * 4), 0);
    if (resources.gpuBytes + bytes + stagingBytes > MAX_OWNED_GPU) throw new Error('Persistent GPU I/O exceeds the 32 MiB budget; lower attention memory or use compatibility mode');
    for (const s of bindings) if (elements(s.dims) * 4 > Math.min(device.limits.maxBufferSize, device.limits.maxStorageBufferBindingSize)) throw new Error(`GPU binding limit exceeded: ${s.name}`);
    const loss = this.loss;
    void device.lost.then(info => { if (loss.active) loss.message = info.message || 'GPU device lost'; });
    device.pushErrorScope('out-of-memory'); device.pushErrorScope('validation');
    let failure: unknown;
    try {
      const bind = (s: IoSpec) => {
        const bytes = elements(s.dims) * 4, allocated = align(bytes);
        const buffer = device.createBuffer({ size: allocated, usage: 0x80 | 0x04 | 0x08, label: s.name });
        // Track immediately so partial allocation/fromGpuBuffer failures roll back too.
        this.buffers.set(s.name, { buffer, bytes, allocated }); resources.gpuBytes += allocated; resources.gpuBuffers++;
        return this.ort.Tensor.fromGpuBuffer(buffer as never, { dataType: 'float32', dims: s.dims });
      };
      for (const s of this.specs.inputs) this.feeds[s.name] = bind(s);
      let offset = 0;
      for (const s of this.specs.outputs) {
        this.fetches[s.name] = bind(s);
        if (this.stateOutputs.has(s.name)) continue;
        const bytes = elements(s.dims) * 4;
        this.outputs[s.name] = this.array(bytes / 4); this.outputOffsets.push({ name: s.name, bytes, offset }); offset += align(bytes);
      }
      this.stagingBytes = Math.max(16, offset);
      this.staging = device.createBuffer({ size: this.stagingBytes, usage: 0x01 | 0x08, label: 'beatrice-readback' });
      resources.gpuBytes += this.stagingBytes; resources.gpuBuffers++;
    } catch (error) { failure = error; }
    const validation = await device.popErrorScope(), memory = await device.popErrorScope();
    if (failure || validation || memory) throw failure ?? new Error((memory ?? validation)!.message);
  }

  reset() {
    this.assertUsable();
    if (this.task) throw new Error('Cannot reset an in-flight session');
    this.supplied.clear();
    if (this.device) {
      const command = this.device.createCommandEncoder();
      for (const s of this.specs.states) {
        const binding = this.buffers.get(s.input)!;
        if (s.init) {
          const initial = new Float32Array(binding.bytes / 4).fill(s.init);
          this.device.queue.writeBuffer(binding.buffer, 0, initial.buffer, 0, binding.bytes);
        } else command.clearBuffer(binding.buffer);
        this.supplied.add(s.input);
      }
      this.device.queue.submit([command.finish()]);
    } else for (const s of this.specs.states) { this.cpu.get(s.input)!.fill(s.init ?? 0); this.supplied.add(s.input); }
  }

  run(input: Record<string, RunInput>): Promise<Record<string, Float32Array>> {
    this.assertUsable();
    if (this.task) return Promise.reject(new Error('Concurrent session.run() is not allowed'));
    this.task = this.execute(input).catch(error => { this.fault = error instanceof Error ? error : new Error(String(error)); throw error; }).finally(() => { this.task = null; });
    return this.task;
  }
  private assertUsable() {
    if (this.disposing || !this.session) throw new Error('Session is disposed');
    if (this.loss.message) throw new Error(this.loss.message);
    if (this.fault) throw this.fault;
  }
  private async execute(input: Record<string, RunInput>) {
    resources.inFlight++; resources.runs++;
    const label = Object.keys(this.outputs).join('/');
    reportStage(`Infer ${label} (${this.device ? 'GPU-bound, capture OFF' : 'CPU I/O'})`);
    try {
      for (const [name, v] of Object.entries(input)) {
        const spec = this.specs.inputs.find(s => s.name === name);
        if (!spec || v.data.length !== elements(spec.dims) || v.dims.join(',') !== spec.dims.join(',')) throw new Error(`Invalid graph input: ${name}`);
        if (this.device) this.device.queue.writeBuffer(this.buffers.get(name)!.buffer, 0, v.data.buffer, v.data.byteOffset, v.data.byteLength);
        else this.cpu.get(name)!.set(v.data);
        this.supplied.add(name);
      }
      if (this.specs.inputs.some(s => !this.supplied.has(s.name))) throw new Error('Missing initial graph input');
      if (this.device) {
        const result = await this.session!.run(this.feeds, this.fetches);
        for (const [name, tensor] of Object.entries(result)) if (tensor !== this.fetches[name]) tensor.dispose();
        const command = this.device.createCommandEncoder();
        for (const s of this.specs.states) command.copyBufferToBuffer(this.buffers.get(s.output)!.buffer, 0, this.buffers.get(s.input)!.buffer, 0, elements(s.dims) * 4);
        for (const o of this.outputOffsets) command.copyBufferToBuffer(this.buffers.get(o.name)!.buffer, 0, this.staging!, o.offset, o.bytes);
        this.device.queue.submit([command.finish()]);
        reportStage(`Read back ${label}`);
        await this.staging!.mapAsync(1);
        try {
          const memory = this.staging!.getMappedRange();
          for (const o of this.outputOffsets) this.outputs[o.name].set(new Float32Array(memory, o.offset, o.bytes / 4));
        } finally { this.staging!.unmap(); }
      } else {
        const result = await this.session!.run(this.feeds);
        try {
          for (const [name, data] of Object.entries(this.outputs)) data.set(result[name].data as Float32Array);
          for (const s of this.specs.states) this.cpu.get(s.input)!.set(result[s.output].data as Float32Array);
        } finally { for (const tensor of Object.values(result)) tensor.dispose(); }
      }
      for (const [name, data] of Object.entries(this.outputs)) for (const value of data) if (!Number.isFinite(value)) throw new Error(`${name}: NaN/Infinity`);
      return this.outputs; // Persistent views, valid until this session's next run.
    } finally { resources.inFlight--; }
  }

  dispose(): Promise<void> {
    if (this.disposing) return this.disposing;
    this.disposing = (async () => {
      try { await this.task; } catch { /* Release still required after a failed run. */ }
      try { await this.session?.release(); }
      finally {
        if (this.session) resources.sessions--; this.session = null; this.loss.active = false;
        for (const b of this.buffers.values()) { b.buffer.destroy(); resources.gpuBytes -= b.allocated; resources.gpuBuffers--; }
        this.buffers.clear();
        if (this.staging) { this.staging.destroy(); resources.gpuBytes -= this.stagingBytes; resources.gpuBuffers--; this.staging = null; }
        // User-created GPU tensors do not own their buffers; never dispose them via ORT.
        if (!this.device) for (const tensor of Object.values(this.feeds)) tensor.dispose();
        this.feeds = {}; this.fetches = {}; this.cpu.clear(); this.outputs = {}; this.supplied.clear();
        resources.cpuIoBytes -= this.localCpuBytes; this.localCpuBytes = 0; this.device = null;
      }
    })();
    return this.disposing;
  }
}