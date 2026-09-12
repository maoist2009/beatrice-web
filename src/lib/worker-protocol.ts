import type { ParaphernaliaFiles } from './paraphernalia';
import type { EngineOptions, EngineParams, EngineStats, FrameResult, ComponentReport } from './runtime-types';
export type WorkerRequest =
  | { id: number; type: 'init'; files: ParaphernaliaFiles; options: Omit<EngineOptions, 'onLog' | 'onStage' | 'signal'> }
  | { id: number; type: 'file'; pcm: Float32Array }
  | { id: number; type: 'start' | 'stop' | 'dispose' }
  | { id: number; type: 'params'; params: Partial<EngineParams> }
  | { id: number; type: 'pcm'; pcm: Float32Array; startSample: number };
export interface WorkerReady { stats: EngineStats; reports: ComponentReport[] }
export interface WorkerAudio { audio: Float32Array; sampleRate: number; frames: FrameResult[] }
export type WorkerEvent =
  | { type: 'reply'; id: number; value?: WorkerReady | WorkerAudio; error?: string; name?: string }
  | { type: 'log' | 'stage' | 'fatal'; text: string }
  | { type: 'stats'; stats: EngineStats }
  | { type: 'frames'; frames: FrameResult[] }
  | { type: 'audio'; audio: Float32Array }
  | { type: 'progress'; fraction: number; message: string };