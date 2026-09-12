import { execFileSync } from 'node:child_process';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
// Browser and dist checks run in a separate invocation AFTER the build emits dist/index.html.
for (const args of [
  ['node_modules/typescript/bin/tsc', '--noEmit'],
  ['tools/run-beta-tests.mjs'],
]) execFileSync(process.execPath, args, { stdio: 'inherit', timeout: 260000 });
// The combined report is written by tools/postbuild-checks.mjs once the browser run has produced results.
void mkdirSync;
const report = {
  scope: 'Beatrice 2.0.0-beta.2, not RVC v2. Fused FP16 export semantics, not native beatrice.lib parity.',
  provenance: JSON.parse(readFileSync('.research/provenance.json', 'utf8')),
  torchOnnx: JSON.parse(readFileSync('.research/torch-onnx-results.json', 'utf8')),
  productionRuntime: JSON.parse(readFileSync('.research/runtime-results.json', 'utf8')),
  browser: JSON.parse(readFileSync('.research/browser-results.json', 'utf8')),
  unverified: ['Adreno 830 hardware performance', 'physical microphone-to-speaker latency', 'official native beatrice.lib parity'],
  notImplemented: ['RVC v2 inference'],
};
writeFileSync('docs/beta2-validation.json', JSON.stringify(report, null, 2));
console.log('[checks] Evidence saved outside dist: docs/beta2-validation.json');