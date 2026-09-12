// Tests production scheduling/resource code, with fake ORT/GPU lifetimes (not fake voice inference).
import assert from 'node:assert/strict';
import { writeFileSync, mkdirSync } from 'node:fs';
import { CooperativePump } from '../src/lib/cooperative-pump';
import { StatefulSession, sessionResources, type StreamingGraph } from '../src/lib/session';
import { StreamEngine } from '../src/lib/stream-engine';
import { parseParaphernalia } from '../src/lib/paraphernalia';

const sleep = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));
const evidence: Record<string, unknown>[] = [];
// Many-speaker metadata preview: no eager FP32 expansion, one object URL for a shared portrait.
const oneImage = new Blob([new Uint8Array([1, 2, 3])]);
const many = parseParaphernalia({
  tomlName: 'model.toml',
  toml: '[model]\nversion="2.0.0-rc.0"\n' + Array.from({ length: 100 }, (_, n) => `[voice.${n}]\nname="voice${n}"\n[voice.${n}.portrait]\npath="shared.png"\n`).join(''),
  phone_extractor: new ArrayBuffer(0), pitch_estimator: new ArrayBuffer(0), waveform_generator: new ArrayBuffer(0),
  speaker_embeddings: new ArrayBuffer(22993408), embedding_setter: null, images: { 'shared.png': oneImage },
}, { metadataOnly: true });
assert.equal(many.speakers.nSpeakers, 100); assert.equal(many.speakers.keyValue.byteLength, 0);
const portraits = new Set(many.voices.map(v => v.portraitUrl)); assert.equal(portraits.size, 1);
for (const url of portraits) if (url) URL.revokeObjectURL(url);
evidence.push({ test: '100-speaker UI preview does not decode weights or duplicate shared portrait URLs', passed: true });

let available = 0, calls = 0, failures = 0, ticks = 0;
const timer = setInterval(() => ticks++, 2);
const pump = new CooperativePump(() => available > 0, async () => { calls++; available--; await sleep(1); }, () => failures++);
for (let i = 0; i < 500; i++) pump.wake();
await sleep(30);
assert.equal(calls, 0); assert.equal(pump.pendingTasks, 0); assert(ticks > 2);
available = 3;
for (let i = 0; i < 1000; i++) pump.wake();
await sleep(50);
assert.equal(calls, 3); assert.equal(pump.pendingTasks, 0); await pump.stop();
clearInterval(timer);
evidence.push({ test: 'idle pump yields to timers, duplicate wakes are coalesced', calls, ticks, failures });

const broken = new CooperativePump(() => true, async () => { throw new Error('injected failure'); }, () => failures++);
broken.wake(); await sleep(30);
assert.equal(failures, 1); assert.equal(broken.pendingTasks, 0); await broken.stop();
evidence.push({ test: 'inference failure stops, never spins or retries indefinitely', passed: true });

const graph: StreamingGraph = { bytes: new Uint8Array([1, 2, 3]),
  inputs: [{ name: 'input', dims: [1, 4] }, { name: 'state', dims: [1, 4] }],
  outputs: [{ name: 'audio', dims: [1, 4] }, { name: 'state_next', dims: [1, 4] }],
  states: [{ input: 'state', output: 'state_next', dims: [1, 4] }],
};
let tensorLive = 0, gpuLive = 0, releases = 0, failAllocation = 0, allocations = 0;
class Buffer {
  data: ArrayBuffer; destroyed = false;
  constructor(bytes: number) { this.data = new ArrayBuffer(bytes); gpuLive++; }
  destroy() { assert(!this.destroyed); this.destroyed = true; gpuLive--; }
  async mapAsync() { assert(!this.destroyed); }
  getMappedRange() { assert(!this.destroyed); return this.data; }
  unmap() {}
}
class Tensor {
  data: Float32Array; buffer?: Buffer; released = false;
  constructor(_type: string, data: Float32Array, readonly dims: number[]) { this.data = data; tensorLive++; }
  static fromGpuBuffer(buffer: Buffer, options: { dims: number[] }) { return { buffer, dims: options.dims, dispose() { throw new Error('Do not dispose user-owned GPU tensors'); } }; }
  dispose() { assert(!this.released, 'double tensor disposal'); this.released = true; tensorLive--; }
}
const device = {
  limits: { maxBufferSize: 1024 * 1024, maxStorageBufferBindingSize: 1024 * 1024 },
  lost: new Promise(() => {}),
  pushErrorScope() {}, async popErrorScope() { return null; },
  createBuffer({ size }: { size: number }) { if (++allocations === failAllocation) throw new Error('injected allocation failure'); return new Buffer(size); },
  queue: {
    writeBuffer(buffer: Buffer, off: number, data: ArrayBuffer, start: number, length: number) { new Uint8Array(buffer.data).set(new Uint8Array(data, start, length), off); },
    submit() {}, async onSubmittedWorkDone() {},
  },
  createCommandEncoder() { return {
    clearBuffer(buffer: Buffer) { new Uint8Array(buffer.data).fill(0); },
    copyBufferToBuffer(a: Buffer, ao: number, b: Buffer, bo: number, size: number) { new Uint8Array(b.data).set(new Uint8Array(a.data, ao, size), bo); }, finish() { return {}; },
  }; },
};
let failRun = false;
const fakeOrt = {
  Tensor, env: { webgpu: { device } },
  InferenceSession: { async create(_bytes: Uint8Array, options: { enableGraphCapture: boolean }) {
    assert.equal(options.enableGraphCapture, false);
    return {
      async run(feeds: Record<string, Tensor>, fetches?: Record<string, { buffer: Buffer }>) {
        if (failRun) throw new Error('injected inference error');
        if (fetches) { for (const t of Object.values(fetches)) new Float32Array(t.buffer.data).fill(0.25); return fetches; }
        return { audio: new Tensor('float32', feeds.input.data.slice(), [1, 4]), state_next: new Tensor('float32', new Float32Array(4).fill(0.1), [1, 4]) };
      },
      async release() { releases++; },
    };
  } },
};

for (const bound of [false, true]) {
  allocations = 0;
  const session = await StatefulSession.create(fakeOrt as never, graph, bound ? 'webgpu' : 'wasm', bound);
  const stable = sessionResources();
  assert(!('bytes' in (session as any).specs), 'Serialized weights must not be retained by a session');
  const input = new Float32Array(4).fill(0.25);
  let first: Float32Array | undefined;
  for (let i = 0; i < 2000; i++) {
    const result = await session.run({ input: { data: input, dims: [1, 4] } });
    if (first) assert.equal(result.audio, first, 'Output workspace should be reused'); else first = result.audio;
    assert.equal(sessionResources().gpuBytes, stable.gpuBytes);
    assert.equal(sessionResources().cpuIoBytes, stable.cpuIoBytes);
  }
  const a = session.run({ input: { data: input, dims: [1, 4] } });
  await assert.rejects(session.run({ input: { data: input, dims: [1, 4] } }), /Concurrent/);
  await a;
  session.reset();
  failRun = true; await assert.rejects(session.run({ input: { data: input, dims: [1, 4] } })); failRun = false;
  await session.dispose(); await session.dispose();
  const after = sessionResources();
  assert.equal(after.sessions, 0); assert.equal(after.gpuBytes, 0); assert.equal(after.cpuIoBytes, 0); assert.equal(after.inFlight, 0);
  assert.equal(tensorLive, 0); assert.equal(gpuLive, 0);
  evidence.push({ test: `2000 ${bound ? 'bound GPU' : 'CPU'} runs, fault + double dispose`, stable, after });
}
allocations = 0; failAllocation = 3;
await assert.rejects(StatefulSession.create(fakeOrt as never, graph, 'webgpu', true), /allocation failure/);
assert.equal(gpuLive, 0); assert.equal(sessionResources().gpuBytes, 0); assert.equal(sessionResources().sessions, 0);
evidence.push({ test: 'partial GPU allocation rolls back', passed: true, releases });
failAllocation = 0;

// Monotonic PCM timeline with a deliberately slow processor; overrun must stop, never reset/rebase-loop.
const core = new (StreamEngine as any)('beatrice-rc0', 1, {});
let pos = 0, steps = 0, nextFailures = 0;
core.processor = {
  chunk: 8, graphBytes: 0, backend: 'wasm', captured: false,
  reset() { pos = 0; }, setParams() {}, async dispose() {},
  get outputFrame() { return pos; }, get requiredSamples() { return (pos + 8) * 160; },
  finish() { return new Float32Array(0); },
  async next(read: (i: number) => number) { read(pos * 160); await sleep(3); steps++; pos += 8; return { audio: new Float32Array(1920), frames: [], inferMs: 3, synthMs: 0 }; },
};
core.onFailure = () => nextFailures++;
await core.start();
for (let i = 0; i < 30; i++) core.push(new Float32Array(640), i * 640);
await sleep(40);
assert.equal(nextFailures, 1); assert.equal(steps, 0); assert.equal(core.mode, 'idle');
await core.start(); core.push(new Float32Array(640), 0); await sleep(15); assert.equal(steps, 0);
core.push(new Float32Array(640), 640); await sleep(30); assert.equal(steps, 1);
await core.stop(); await sleep(15); assert.equal(steps, 1);
const cancelled = core.convert16(new Float32Array(16000)).then(() => false, (e: Error) => e.name === 'AbortError');
await sleep(5); await core.stop(); assert.equal(await cancelled, true);
let snapshotChecked = false;
core.processor.next = async (read: (sample: number) => number) => {
  const first = read(0);
  core.ring.fill(9); // Deliberate overwrite while GPU work would be awaiting completion.
  await sleep(3);
  assert.equal(read(0), first); snapshotChecked = true; pos += 8;
  return { audio: new Float32Array(1920), frames: [], inferMs: 3, synthMs: 0 };
};
await core.start(); core.push(new Float32Array(1280).fill(1), 0);
await sleep(30); assert(snapshotChecked); await core.stop();
await core.dispose();
evidence.push({ test: 'overrun pauses and normal input advances once, no rebase', passed: true });
evidence.push({ test: 'file cancellation waits for the active chunk and returns to idle', passed: true });
evidence.push({ test: 'live PCM is snapshotted before asynchronous inference, immune to ring overwrite', passed: true });
mkdirSync('docs', { recursive: true }); writeFileSync('docs/stability-unit-results.json', JSON.stringify(evidence, null, 2));
console.log('STABILITY UNIT PASS', JSON.stringify(evidence));