import type { ModelFormat } from './formats';
import type { PitchParams } from './dsp';

export type Backend = 'webgpu' | 'wasm';
/** 'auto' benchmarks WebGPU vs WASM during the build and keeps the faster one (measured RTF). */
export type BackendChoice = Backend | 'auto';
/** 'fp32' forces an FP32-only WebGPU device (shader-f16 not requested) — the Adreno 6xx garble workaround. */
export type GpuPrecision = 'auto' | 'fp32';
export type GpuMode = 'bound' | 'compatible';
export interface ComponentReport { name: string; status: 'verified' | 'mismatch'; bytes: number; detail: string }
export interface FrameResult {
  frame: number; bin: number; hz: number; midi: number; unvoiced: number; half: number; dbl: number;
  energy: number; targetBin: number; targetHz: number; phone: Float32Array;
}
export interface EngineParams extends PitchParams {
  formantShift: number; speaker: number; inputGain: number; outputGain: number; monitor: boolean;
  minMidi: number; maxMidi: number; vqNeighbors: number; convert: boolean;
}
export interface ResourceStats {
  sessions: number; gpuBuffers: number; gpuBytes: number; cpuIoBytes: number; retainedGraphBytes: number;
  inFlight: number; runs: number;
}
export interface EngineStats {
  backend: Backend; latencyMs: number; inferMs: number; synthMs: number; rtf: number; frames: number;
  underruns: number; queue: number; sampleRate: number; graphBytes: number;
  measured?: boolean; chunkFrames?: number; dropped?: number; capture?: boolean;
  /** Overrun safe-pauses: inference fell behind real time; mic paused, engine retained. */
  overruns?: number;
  ioMode?: 'gpu-bound' | 'cpu-tensors'; stage?: string; resources?: ResourceStats;
  adapter?: { vendor: string; architecture: string; description: string };
}
export interface EngineOptions {
  backend: BackendChoice; ctxFrames: number; chunkFrames: number; gpuMode?: GpuMode; gpuPrecision?: GpuPrecision;
  /** Experimental: record GPU commands on first run, replay afterwards (kills per-chunk dispatch overhead). */
  graphCapture?: boolean;
  signal?: AbortSignal;
  onLog?: (message: string) => void; onStage?: (stage: string) => void;
  gruMode?: 'onnx-gru' | 'primitives';
}
export interface VoiceEngine {
  readonly format: ModelFormat;
  backend: Backend; graphBytes: number; reports: ComponentReport[]; stats: EngineStats;
  onFrames: ((frames: FrameResult[]) => void) | null;
  onStats: ((stats: EngineStats) => void) | null;
  onDiag: ((d: { ir: any; ap: any; pf: any; x: any; f0: number; frame: number }) => void) | null;
  onState: ((state: 'idle' | 'live' | 'file' | 'error') => void) | null;
  setParams(params: Partial<EngineParams>): void;
  start(deviceId?: string): Promise<void>;
  stop(): void;
  dispose(): Promise<void>;
  convertBuffer(pcm: Float32Array, rate: number, progress?: (fraction: number, text: string) => void): Promise<{ audio: Float32Array; sampleRate: number; frames: FrameResult[] }>;
}
export const DEFAULT_PARAMS: EngineParams = {
  // These are the official ProcessorCore1/2 defaults, not the trainer's vq_topk.
  // averageSourcePitch is the official core parameter unit (default 52.0).
  pitchShift: 0, averageSourcePitch: 52, intonationIntensity: 1, pitchCorrection: 0, pitchCorrectionType: 0,
  formantShift: 0, speaker: 0, inputGain: 1, outputGain: 1, monitor: false, minMidi: 33.125, maxMidi: 80.875,
  vqNeighbors: 0, convert: true,
};
export const initialStats = (): EngineStats => ({ backend: 'wasm', latencyMs: 0, inferMs: 0, synthMs: 0, rtf: 0, frames: 0, underruns: 0, queue: 0, sampleRate: 0, graphBytes: 0, measured: false, overruns: 0 });
