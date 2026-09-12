// Explicit development-only validation summary / cleanup. Not imported by the application.
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { readFileSync, writeFileSync, existsSync, rmSync } from 'node:fs';
import { createHash } from 'node:crypto';
const read = path => JSON.parse(readFileSync(path, 'utf8'));
for (const command of [['node_modules/typescript/bin/tsc', '--noEmit'], ['tools/audit-dist.mjs']]) execFileSync(process.execPath, command, { stdio: 'inherit', timeout: 30000 });
const report = {
  build: 'worker-isolation-no-capture-v2',
  reproduced: read('docs/freeze-reproduction.json'),
  lifetimeTests: read('docs/stability-unit-results.json'),
  wasmBrowser: read('docs/browser-stability-wasm.json'),
  softwareWebgpuBrowser: read('docs/browser-stability-webgpu.json'),
  rc0NumericRegression: read('.research/rc0-streaming-results.json'),
  notes: [
    'Device-specific 13% renderer crash is not conclusively attributed to OOM without device crash/memory logs.',
    'Tracked GPU buffers are application I/O only, not all ORT, process, driver or GPU allocations.',
    'A software WebGPU device verifies execution and bounded resources, not mobile GPU performance.',
    'WASM sustained fake-microphone UI test is 10 seconds per model; this is not a multi-hour soak test.',
    'Adreno 6xx/7xx/830 have not been physically tested in this environment.',
    'All production engine paths explicitly disable enableGraphCapture.',
  ],
};
assert.equal(report.softwareWebgpuBrowser.pageErrors.length, 0);
writeFileSync('docs/stability-validation.json', JSON.stringify(report, null, 2));
const html = readFileSync('dist/index.html');
writeFileSync('docs/stability-build.json', JSON.stringify({ bytes: html.length, sha256: createHash('sha256').update(html).digest('hex'), assets: ['index.html'] }, null, 2));
// All files under this ignored directory were created by the reproducible research/test tools.
if (existsSync('.research')) {
  assert(existsSync('docs/stability-validation.json'));
  rmSync('.research', { recursive: true, force: true });
}
console.log('[final stability checks] Reports retained in docs/. Development models/browser/runtime downloads removed.');