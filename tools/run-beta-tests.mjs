// Explicit development command: node tools/run-beta-tests.mjs. Never imported by the browser or the final build.
import { build } from 'esbuild';
import { execFileSync } from 'node:child_process';
import { existsSync } from 'node:fs';

const env = { ...process.env, PYTHONPATH: `${process.cwd()}/.research/python`, OMP_NUM_THREADS: '1', OPENBLAS_NUM_THREADS: '1', MKL_NUM_THREADS: '1' };
function run(cmd, args) {
  execFileSync(cmd, args, { env, stdio: 'inherit', timeout: 200000 });
}
await build({ entryPoints: ['tools/build-beta-fixtures.ts'], outfile: '.research/build-fixtures.mjs', bundle: true, platform: 'node', format: 'esm', target: 'es2022' });
run(process.execPath, ['--max-old-space-size=384', '.research/build-fixtures.mjs']);
if (!existsSync('.research/jfk.wav')) run('curl', ['-fLsS', '--max-time', '60', 'https://raw.githubusercontent.com/ggml-org/whisper.cpp/master/samples/jfk.wav', '-o', '.research/jfk.wav']);
run('python3', ['tools/reference-beta.py']);
if (existsSync('tools/verify-beta-runtime.ts')) {
  await build({ entryPoints: ['tools/verify-beta-runtime.ts'], outfile: '.research/verify-runtime.mjs', bundle: true, platform: 'node', format: 'esm', target: 'es2022', packages: 'external' });
  run(process.execPath, ['--max-old-space-size=384', '.research/verify-runtime.mjs']);
}