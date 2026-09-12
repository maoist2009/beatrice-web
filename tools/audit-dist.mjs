// Explicit post-build check. Does not mutate the production bundle.
import assert from 'node:assert/strict';
import { readdirSync, readFileSync, statSync } from 'node:fs';
import { gzipSync } from 'node:zlib';
import { createHash } from 'node:crypto';
const dist = process.env.DIST_DIR ?? 'dist';
const files = readdirSync(dist, { recursive: true }).filter(p => statSync(`${dist}/${p}`).isFile());
assert.deepEqual(files, ['index.html'], 'Unexpected production assets (models/tests must not be copied into dist)');
const html = readFileSync(`${dist}/index.html`);
assert(html.length < 512 * 1024, `dist exceeds the 512 KiB uncompressed budget: ${html.length}`);
const text = html.toString('utf8');
for (const marker of ['REFERENCE PARITY PASS', 'torch-onnx-results.json', 'golden-shigure', 'browser-smoke.mjs', '.research/python', 'playwright.chromium', 'indexedDB.cmp =']) assert(!text.includes(marker), `Development code leaked into dist: ${marker}`);
console.log('[dist audit]', JSON.stringify({ files, bytes: html.length, gzipBytes: gzipSync(html).length, sha256: createHash('sha256').update(html).digest('hex') }));