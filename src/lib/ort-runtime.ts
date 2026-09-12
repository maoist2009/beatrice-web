import type * as ORT from 'onnxruntime-web';
export const ORT_VERSION = '1.29.0';
let pending: Promise<typeof ORT> | null = null;
export function loadOrt(): Promise<typeof ORT> {
  if (!pending) {
    const base = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
    pending = import(/* @vite-ignore */ `${base}ort.webgpu.min.mjs`).then(module => {
      const ort = (module.InferenceSession ? module : module.default) as typeof ORT;
      ort.env.wasm.wasmPaths = base;
      ort.env.wasm.numThreads = 1;
      ort.env.wasm.proxy = false; // We own the worker, including its cancellation/termination.
      return ort;
    }).catch(error => { pending = null; throw error; });
  }
  return pending;
}