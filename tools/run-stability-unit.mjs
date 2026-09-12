import { build } from 'esbuild';
import { execFileSync } from 'node:child_process';
import { mkdirSync } from 'node:fs';
mkdirSync('.research', { recursive: true });
await build({ entryPoints: ['tools/stability-unit.ts'], outfile: '.research/stability-unit.mjs', bundle: true, platform: 'node', format: 'esm', packages: 'external' });
execFileSync(process.execPath, ['--max-old-space-size=256', '.research/stability-unit.mjs'], { stdio: 'inherit', timeout: 30000 });