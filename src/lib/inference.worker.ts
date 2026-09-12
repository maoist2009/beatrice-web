import { parseParaphernalia } from './paraphernalia';
import { StreamEngine } from './stream-engine';
import { Rc0Processor } from './rc0/runtime';
import { BetaProcessor } from './beta/runtime';
import { observeSessionStages } from './session';
import type { WorkerEvent, WorkerRequest, WorkerReady, WorkerAudio } from './worker-protocol';

const worker = self as unknown as { onmessage: ((event: MessageEvent<WorkerRequest>) => void) | null; postMessage(message: WorkerEvent, transfer?: Transferable[]): void };
let engine: StreamEngine | null = null;
let initializing = false;
let lastStage = 0;
let pendingFrames: import('./runtime-types').FrameResult[] = [];
let lastFrames = 0;
const send = (event: WorkerEvent, transfer: Transferable[] = []) => worker.postMessage(event, transfer);
observeSessionStages(stage => {
  const now = performance.now();
  if (initializing || now - lastStage > 250) { lastStage = now; send({ type: 'stage', text: stage }); }
});

worker.onmessage = ({ data }) => { void handle(data); };
async function handle(request: WorkerRequest) {
  try {
    let value: WorkerReady | WorkerAudio | undefined;
    let transfer: Transferable[] = [];
    if (request.type === 'init') {
      if (engine || initializing) throw new Error('Only one model per inference worker');
      initializing = true;
      const model = parseParaphernalia(request.files);
      const options = { ...request.options, onLog: (text: string) => send({ type: 'log', text }) };
      const isBeta = model.format === 'beatrice-beta2';
      engine = await StreamEngine.create(model, options, (m, ort, ep, chunk, params, bound) => isBeta
        ? BetaProcessor.create(m, ort, ep, chunk, params, Math.random, ep === 'webgpu' ? 'primitives' : 'onnx-gru', bound)
        : Rc0Processor.create(m, ort, ep, chunk, params, { gpuBound: bound, attnPositions: Math.max(8, Math.round(options.ctxFrames / 4)) }), model.format);
      engine.onStats = stats => send({ type: 'stats', stats });
      engine.onFailure = text => send({ type: 'fatal', text });
      engine.onAudio = audio => send({ type: 'audio', audio }, [audio.buffer as ArrayBuffer]);
      engine.onFrames = frames => {
        pendingFrames.push(...frames); if (pendingFrames.length > 100) pendingFrames.splice(0, pendingFrames.length - 100);
        if (performance.now() - lastFrames < 150) return;
        const batch = pendingFrames; pendingFrames = []; lastFrames = performance.now();
        send({ type: 'frames', frames: batch }, batch.map(f => f.phone.buffer as ArrayBuffer));
      };
      value = { stats: engine.stats, reports: engine.reports };
      initializing = false;
    } else {
      if (!engine) throw new Error('Worker has no initialized model');
      if (request.type === 'params') engine.setParams(request.params);
      if (request.type === 'file') {
        value = await engine.convert16(request.pcm, (fraction, message) => send({ type: 'progress', fraction, message }));
        transfer = [value.audio.buffer, ...value.frames.map(f => f.phone.buffer)] as ArrayBuffer[];
      }
      if (request.type === 'start') { pendingFrames = []; await engine.start(); }
      if (request.type === 'pcm') engine.push(request.pcm, request.startSample);
      if (request.type === 'stop') { await engine.stop(); pendingFrames = []; }
      if (request.type === 'dispose') { await engine.dispose(); engine = null; pendingFrames = []; }
    }
    send({ type: 'reply', id: request.id, value }, transfer);
  } catch (error) {
    initializing = false;
    const e = error instanceof Error ? error : new Error(String(error));
    if (request.type === 'pcm') { void engine?.stop(); send({ type: 'fatal', text: e.message }); }
    send({ type: 'reply', id: request.id, error: e.message, name: e.name });
  }
}