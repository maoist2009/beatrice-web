// Production-bundle regression test: one-second real speech, repetition, worker release and live-mode UI.
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { appendFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';

mkdirSync('.research', { recursive: true }); mkdirSync('docs', { recursive: true });
if (!existsSync('.research/jfk.wav')) execFileSync('curl', ['-fLsS', '--max-time', '60', 'https://raw.githubusercontent.com/ggml-org/whisper.cpp/master/samples/jfk.wav', '-o', '.research/jfk.wav']);
const original = readFileSync('.research/jfk.wav');
let offset = 12, dataOffset = 0;
while (offset + 8 <= original.length) {
  const size = original.readUInt32LE(offset + 4);
  if (original.toString('ascii', offset, offset + 4) === 'data') { dataOffset = offset + 8; break; }
  offset += 8 + size + (size % 2);
}
assert(dataOffset > 0);
const wav = Buffer.alloc(44 + 16000 * 2);
wav.write('RIFF'); wav.writeUInt32LE(wav.length - 8, 4); wav.write('WAVE', 8); wav.write('fmt ', 12); wav.writeUInt32LE(16, 16);
wav.writeUInt16LE(1, 20); wav.writeUInt16LE(1, 22); wav.writeUInt32LE(16000, 24); wav.writeUInt32LE(32000, 28); wav.writeUInt16LE(2, 32); wav.writeUInt16LE(16, 34);
wav.write('data', 36); wav.writeUInt32LE(32000, 40); original.copy(wav, 44, dataOffset + 12000, dataOffset + 44000);
writeFileSync('.research/one-second.wav', wav);

process.env.PLAYWRIGHT_BROWSERS_PATH = `${process.cwd()}/.research/browsers`;
process.env.LD_LIBRARY_PATH = `${process.cwd()}/.research/sysroot/usr/lib/x86_64-linux-gnu:${process.cwd()}/.research/sysroot/lib/x86_64-linux-gnu`;
const { chromium } = await import('playwright');
const useGpu = process.env.TEST_GPU === '1';
if (useGpu) {
  const log = console.log;
  console.log = (...items) => { appendFileSync('.research/gpu-progress.log', `${new Date().toISOString()} ${items.map(i => typeof i === 'string' ? i : JSON.stringify(i)).join(' ')}\n`); log(...items); };
  if (!existsSync('.research/hecko.zip')) execFileSync('curl', ['-fLsS', '--max-time', '60', 'https://huggingface.co/hecko/beatrice-old-tts/resolve/74fafd8f8840b51005700bebe244407885b262b8/old%20tts.zip', '-o', '.research/hecko.zip']);
}
const browser = await chromium.launch({ headless: true, args: [
  '--no-sandbox', '--disable-dev-shm-usage', '--enable-precise-memory-info',
  '--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream', `--use-file-for-fake-audio-capture=${process.cwd()}/.research/one-second.wav`,
  ...(useGpu ? ['--enable-unsafe-webgpu', '--use-angle=swiftshader'] : []),
] });
const context = await browser.newContext({ viewport: { width: 1440, height: 1000 }, acceptDownloads: true, permissions: ['microphone'] });
await context.route('http://127.0.0.1:4178/**', route => route.fulfill({ contentType: 'text/html', body: readFileSync('dist/index.html') }));
const page = await context.newPage(); page.setDefaultTimeout(30000);
const errors = [], output = [], snapshots = [];
page.on('pageerror', error => errors.push(error.message));
page.on('crash', () => errors.push('PAGE_CRASH'));
page.on('console', message => { if (message.type() === 'error') console.log('[console]', message.text().slice(0, 400)); });
const cdp = await context.newCDPSession(page);
await cdp.send('HeapProfiler.enable');
const hardTimeout = setTimeout(() => { console.log('[timeout] browser test exceeded 150s'); void browser.close(); setTimeout(() => process.exit(1), 1000).unref(); }, 150000);
let inspectTimer;
if (useGpu) inspectTimer = setInterval(() => { void page.getByRole('status').allTextContents().then(s => console.log('[last stage]', s)).catch(() => {}); }, 5000);

async function diagnostic() {
  const downloaded = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Save diagnostics', exact: true }).click();
  const download = await downloaded, path = await download.path();
  return JSON.parse(readFileSync(path, 'utf8'));
}
async function memory() { await cdp.send('HeapProfiler.collectGarbage'); return await cdp.send('Runtime.getHeapUsage'); }
async function pulseStart() { await page.evaluate(() => {
  clearInterval(window.__pulseTimer); window.__pulse = { ticks: 0, maxGap: 0 }; let previous = performance.now();
  window.__pulseTimer = setInterval(() => { const now = performance.now(); window.__pulse.ticks++; window.__pulse.maxGap = Math.max(window.__pulse.maxGap, now - previous); previous = now; }, 20);
}); }
async function convert() {
  await page.getByRole('button', { name: 'Convert', exact: true }).click();
  await page.waitForFunction(() => [...document.querySelectorAll('button')].some(b => b.textContent.trim() === 'Convert' && !b.disabled), undefined, { timeout: 160000 });
  const alert = page.getByRole('alert');
  if (await alert.count()) throw new Error(await alert.textContent());
  await page.getByRole('link', { name: 'download WAV', exact: true }).waitFor();
  const values = await page.evaluate(async () => {
    const audio = document.querySelectorAll('audio')[1], blob = await (await fetch(audio.src)).arrayBuffer();
    const decoded = await new OfflineAudioContext(1, 1, 44100).decodeAudioData(blob), x = decoded.getChannelData(0);
    let sum = 0, peak = 0; for (const sample of x) { sum += sample * sample; peak = Math.max(peak, Math.abs(sample)); }
    return { seconds: decoded.duration, rms: Math.sqrt(sum / x.length), peak, pulse: window.__pulse };
  });
  assert(Math.abs(values.seconds - 1) < 0.01); assert(values.rms > 1e-4); assert(values.peak <= 1);
  output.push(values); console.log('[1s conversion]', JSON.stringify(values));
}

try {
  await page.goto('http://127.0.0.1:4178/', { waitUntil: 'networkidle' });
  const gpu = await page.evaluate(async () => {
    const adapter = await navigator.gpu?.requestAdapter();
    return adapter ? { available: true, info: { vendor: adapter.info?.vendor, architecture: adapter.info?.architecture, description: adapter.info?.description }, maxBuffer: adapter.limits.maxBufferSize } : { available: false };
  });
  console.log('[browser GPU]', gpu);
  if (useGpu) assert(gpu.available, 'Software WebGPU is required for this test');
  const models = ['Old TTS voices (8 speakers)', ...(useGpu ? [] : ['刻鳴時雨 (CV: 丸ころ)'])];
  for (const title of models) {
    if (useGpu) await page.locator('input[type=file][accept^=".zip"]').setInputFiles('.research/hecko.zip');
    else await page.getByRole('listitem').filter({ hasText: title }).first().getByRole('button', { name: 'get', exact: true }).click();
    await page.waitForFunction(() => [...document.querySelectorAll('button')].some(b => b.textContent.includes('Build & compile') && !b.disabled), undefined, { timeout: 120000 });
    await page.locator('select').nth(0).selectOption(useGpu ? 'webgpu' : 'wasm');
    await page.getByRole('button', { name: 'Build & compile', exact: true }).click();
    await page.waitForFunction(() => [...document.querySelectorAll('button')].some(b => b.textContent.includes('Live mic') && !b.disabled), undefined, { timeout: 180000 });
    if (await page.getByRole('alert').count()) throw new Error(await page.getByRole('alert').textContent());
    assert.equal(page.workers().length, 1);
    await page.locator('input[type=file][accept*=".wav"]').setInputFiles('.research/one-second.wav');
    await page.getByText('one-second.wav', { exact: true }).waitFor();
    await pulseStart();
    const count = useGpu ? 2 : 6;
    for (let i = 0; i < count; i++) {
      await convert();
      const d = await diagnostic(); const heap = await memory();
      assert.equal(d.stats.resources.sessions, 3); assert.equal(d.stats.resources.inFlight, 0); assert.equal(d.stats.capture, false);
      assert.equal(d.stats.resources.retainedGraphBytes, 0);
      if (useGpu) { assert.equal(d.stats.backend, 'webgpu'); assert.equal(d.stats.ioMode, 'gpu-bound'); }
      snapshots.push({ title, iteration: i, resources: d.stats.resources, heap: heap.usedSize, backend: d.stats.backend, ioMode: d.stats.ioMode, rtf: d.stats.rtf });
      if (i >= 1) {
        const first = snapshots[snapshots.length - i - 1];
        assert.equal(d.stats.resources.gpuBytes, first.resources.gpuBytes);
        assert.equal(d.stats.resources.cpuIoBytes, first.resources.cpuIoBytes);
      }
    }
    console.log('[repeat snapshots]', JSON.stringify(snapshots.filter(s => s.title === title)));
    // Live mode with fake microphone data. UI heartbeat must continue before AND after enough PCM exists.
    await pulseStart();
    await page.getByRole('button', { name: 'Live mic', exact: true }).click();
    await page.getByRole('button', { name: 'Stop', exact: true }).waitFor();
    await page.waitForTimeout(useGpu ? 8000 : 10000);
    const livePulse = await page.evaluate(() => window.__pulse);
    assert(livePulse.ticks > 30, 'Realtime mode starved the main thread');
    const live = await diagnostic();
    if (!useGpu) assert(live.stats.measured && live.stats.frames > 0, 'Realtime RTF/frame count remained zero');
    console.log('[live]', JSON.stringify({ frames: live.stats.frames, rtf: live.stats.rtf, pulse: livePulse, error: live.logs.filter(s => s.includes('ERROR')) }));
    const stop = page.getByRole('button', { name: 'Stop', exact: true });
    if (await stop.isVisible()) await stop.click();
    else assert(live.logs.some(s => s.includes('backlog') || s.includes('cannot keep up')), 'Expected controlled overrun pause, not an unexplained stop');
    await page.getByRole('button', { name: 'Stop / release Worker', exact: true }).click();
    await page.waitForTimeout(1800); assert.equal(page.workers().length, 0);
    console.log('[worker release] no live workers');
  }
  assert.equal(errors.length, 0, errors.join('\n'));
  const report = { browser: browser.version(), gpu, deviceNote: 'Desktop sandbox / software GPU, not Adreno hardware', output, snapshots, pageErrors: errors, checks: ['1-second real speech', 'repeated conversion', 'stable tracked I/O allocations', 'realtime timer responsiveness', ...(useGpu ? ['cancel live GPU work without freezing UI'] : ['nonzero realtime RTF/frame count']), 'worker termination'], excludedClaims: ['total process/GPU memory', 'Adreno 6/7/830 stability', 'device OOM root cause'] };
  writeFileSync(`docs/browser-stability-${useGpu ? 'webgpu' : 'wasm'}.json`, JSON.stringify(report, null, 2));
  console.log('BROWSER STABILITY PASS', useGpu ? 'WebGPU' : 'WASM');
} finally { clearTimeout(hardTimeout); clearInterval(inspectTimer); await context.close(); await browser.close(); }