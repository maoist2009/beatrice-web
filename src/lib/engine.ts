// Main-thread API only. Neural model building and inference are isolated in a disposable Worker.
import type { Paraphernalia } from './paraphernalia';
import type { EngineOptions, VoiceEngine } from './runtime-types';
export type { Backend, BackendChoice, GpuPrecision, GpuMode, ComponentReport, EngineOptions, EngineParams, EngineStats, FrameResult, VoiceEngine, ResourceStats } from './runtime-types';
export { loadOrt, ORT_VERSION } from './ort-runtime';
export { hardTerminateWorker, sharedWorkerAlive } from './shared-worker';
export async function createVoiceEngine(model: Paraphernalia, options: EngineOptions): Promise<VoiceEngine> {
  const { WorkerEngine } = await import('./worker-engine');
  return WorkerEngine.create(model, options);
}
export async function webgpuAvailable(): Promise<boolean> {
  const gpu = (navigator as Navigator & { gpu?: { requestAdapter(): Promise<unknown> } }).gpu;
  if (!gpu) return false;
  try { return !!await gpu.requestAdapter(); } catch { return false; }
}