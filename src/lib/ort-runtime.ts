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
      // Hint for the ORT-managed adapter path (ignored on ORT builds without this flag).
      // Battery savers clock mobile GPUs (Adreno!) to the floor without it.
      try { (ort.env.webgpu as unknown as Record<string, unknown>).powerPreference = 'high-performance'; } catch { /* flag unsupported */ }
      return ort;
    }).catch(error => { pending = null; throw error; });
  }
  return pending;
}

export interface AdapterInfo { vendor: string; architecture: string; description: string }
/** Minimal structural GPUAdapter — WebGPU types are not in the configured DOM lib. */
interface GAdapter {
  info?: Partial<AdapterInfo>;
  features?: { has(feature: string): boolean };
  requestDevice(descriptor?: { label?: string; requiredFeatures?: string[] }): Promise<unknown>;
}
interface GNavigator { gpu?: { requestAdapter(options?: { powerPreference?: 'low-power' | 'high-performance' }): Promise<GAdapter | null> } }
let injectedDevice: unknown = null;
let injectedAdapterInfo: AdapterInfo | null = null;
let preparing: Promise<void> | null = null;
let planKey: string | null = null;

/** App-owned device (set only when forcing FP32); consumed by session.ts at session creation. */
export const injectedWebGpuDevice = (): unknown => injectedDevice;
export const webgpuAdapterInfo = (): AdapterInfo | null => injectedAdapterInfo;

interface WebGpuPlan { forceFp32: boolean; onLog?: (text: string) => void }

/**
 * Provision WebGPU BEFORE any session is created:
 * - Always requests a high-performance adapter.
 * - forceFp32: requests the device WITHOUT 'shader-f16', so ORT compiles FP32-only WGSL.
 *   Adreno 6xx fp16/mediump paths are known to corrupt this network's output (garbled audio);
 *   on those GPUs CPU is often also faster, so pair this with backend 'auto'.
 * - Otherwise device creation is left to ORT (shader-f16 enabled where the driver is trustworthy).
 */
export function prepareWebGpu(ort: typeof ORT, plan: WebGpuPlan): Promise<void> {
  // The shared worker outlives rebuilds: if the precision plan changed between builds,
  // the previously injected device must be dropped (and destroyed) before re-provisioning.
  const key = plan.forceFp32 ? 'fp32' : 'auto';
  if (preparing && planKey === key) return preparing;
  planKey = key;
  if (injectedDevice) {
    try { (injectedDevice as { destroy?: () => void }).destroy?.(); } catch { /* already destroyed */ }
    injectedDevice = null;
  }
  preparing = (async () => {
    void ort;
    const gpu = (navigator as Navigator & GNavigator).gpu;
    if (!gpu) { plan.onLog?.('WebGPU: navigator.gpu unavailable; will fall back to WASM'); return; }
    try {
      const adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' });
      if (!adapter) { plan.onLog?.('WebGPU: no adapter granted; will fall back to WASM'); return; }
      const info = adapter.info;
      injectedAdapterInfo = { vendor: info?.vendor ?? '', architecture: info?.architecture ?? '', description: info?.description ?? '' };
      // fp16 is NOT a degradation of old GPUs — it is an opt-in acceleration. ORT compiles
      // fp16 WGSL kernels only when the requested device actually carries 'shader-f16';
      // everything else stays fp32. Log the adapter capability so each device answers for itself.
      const adapterF16 = adapter.features?.has?.('shader-f16') ?? false;
      plan.onLog?.(`WebGPU adapter: ${injectedAdapterInfo.description || injectedAdapterInfo.vendor || 'no description'}; shader-f16 ${adapterF16 ? 'EXPOSED (fp16 possible)' : 'NOT EXPOSED (fp32 only)'}`);
      if (plan.forceFp32) {
        injectedDevice = await adapter.requestDevice({ label: 'beatrice-fp32', requiredFeatures: [] });
        plan.onLog?.(`WebGPU: app-owned device created WITHOUT shader-f16 → ORT must compile fp32 kernels${adapterF16 ? ' (adapter had it; deliberately not used)' : ''}`);
      } else {
        plan.onLog?.(`WebGPU: ORT-managed device → ${adapterF16 ? 'fp16 kernels where ORT chooses them' : 'fp32 kernels'}`);
      }
    } catch (error) {
      plan.onLog?.(`WebGPU adapter init failed: ${String(error)} (WASM fallback remains)`);
    }
  })();
  return preparing;
}
