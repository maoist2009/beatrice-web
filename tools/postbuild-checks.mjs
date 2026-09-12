// Verifies the shipped bundle. A build-time PostCSS hook runs before Vite emits dist/, so this builds an
// identical bundle itself via Vite's programmatic API with the hook disabled (no recursion, same plugins).
import { execFileSync } from 'node:child_process';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
const { build } = await import('vite');
const outDir = '.research/dist-check';
await build({ configFile: 'vite.config.ts', build: { outDir, emptyOutDir: true }, css: { postcss: { plugins: [] } }, logLevel: 'warn' });
const env = { ...process.env, DIST_DIR: outDir };
for (const args of [['tools/browser-smoke.mjs'], ['tools/audit-dist.mjs']]) execFileSync(process.execPath, args, { env, stdio: 'inherit', timeout: 260000 });
mkdirSync('docs', { recursive: true });
const report = {
  scope: 'Beatrice 2.0.0-beta.2 (stateful GRU) and 2.0.0-rc.0 (windowed). Fused FP16 export semantics, not native beatrice.lib parity. rc.1/rc.2/rc.3 are VST releases over the same rc.0 paraphernalia format.',
  provenance: JSON.parse(readFileSync('.research/provenance.json', 'utf8')),
  torchOnnx: JSON.parse(readFileSync('.research/torch-onnx-results.json', 'utf8')),
  productionRuntime: JSON.parse(readFileSync('.research/runtime-results.json', 'utf8')),
  browser: JSON.parse(readFileSync('.research/browser-results.json', 'utf8')),
  gruOnGpu: 'The primitives GRU (MatMul/Sigmoid/Tanh) matches the native GRU node against PyTorch; device-level GPU speed is not measured here.',
  unverified: ['Adreno 830 hardware performance', 'physical microphone-to-speaker latency', 'official native beatrice.lib parity', 'WebGPU vs WASM speed on any device'],
  notImplemented: ['RVC v2 inference', 'Chinese-language training data (the frozen extractors are Japanese/English/singing only)'],
};
writeFileSync('docs/beta2-validation.json', JSON.stringify(report, null, 2));
console.log('[checks] Evidence saved outside dist: docs/beta2-validation.json');
