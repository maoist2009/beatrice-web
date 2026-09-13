import InferenceWorker from './inference.worker?worker&inline';

/**
 * ONE inference worker per page, shared across engine rebuilds.
 *
 * Every rebuild used to spawn a fresh Worker, each holding its own ORT WASM linear
 * memory. Android Chrome reuses the renderer process across same-tab reloads, so those
 * big heaps stacked up until the OS killed the tab (each reload shorter time-to-crash).
 *
 * Now: rebuilds only release the model (rpc 'dispose') — the worker and its heap survive.
 * The worker is terminated only when it is poisoned (fatal error / init failure; the next
 * build respawns a fresh one) or when the page goes away (pagehide/beforeunload), which
 * frees the WASM heap and GPU buffers for good.
 */
let shared: Worker | null = null;
let alive = false;

export function acquireSharedWorker(): Worker {
  if (shared && alive) return shared;
  shared = new InferenceWorker();
  alive = true;
  return shared;
}

export function sharedWorkerAlive(): boolean {
  return alive && !!shared;
}

/** Fatal path: the current worker is poisoned; the next acquire spawns a fresh one. */
export function killSharedWorker(): void {
  if (shared) { try { shared.terminate(); } catch { /* already gone */ } }
  shared = null;
  alive = false;
}

/** pagehide / beforeunload: drop the WASM heap and GPU buffers immediately. */
export function hardTerminateWorker(): void {
  killSharedWorker();
}
