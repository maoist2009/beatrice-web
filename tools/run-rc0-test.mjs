// Explicit dev command: node tools/run-rc0-test.mjs (fetches the small public rc.0 model + test audio via curl).
import { build } from 'esbuild';
import { execFileSync } from 'node:child_process';
import { existsSync, mkdirSync } from 'node:fs';
mkdirSync('.research', { recursive: true });
const curl = (url, out) => { if (!existsSync(out)) execFileSync('curl', ['-fLsS', '--max-time', '180', url, '-o', out], { stdio: 'inherit', timeout: 200000 }); };
curl('https://huggingface.co/hecko/beatrice-old-tts/resolve/74fafd8f8840b51005700bebe244407885b262b8/old%20tts.zip', '.research/hecko.zip');
curl('https://raw.githubusercontent.com/ggml-org/whisper.cpp/master/samples/jfk.wav', '.research/jfk.wav');
await build({ entryPoints: ['tools/verify-rc0-streaming.ts'], outfile: '.research/verify-rc0.mjs', bundle: true, platform: 'node', format: 'esm', target: 'es2022', packages: 'external' });
execFileSync(process.execPath, ['--max-old-space-size=384', '.research/verify-rc0.mjs'], { stdio: 'inherit', timeout: 250000 });
execFileSync(process.execPath, ['node_modules/typescript/bin/tsc', '--noEmit'], { stdio: 'inherit', timeout: 120000 });
