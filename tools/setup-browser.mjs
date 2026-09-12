import { execFileSync } from 'node:child_process';
import { existsSync, mkdirSync, readdirSync, writeFileSync } from 'node:fs';
process.env.PLAYWRIGHT_BROWSERS_PATH = `${process.cwd()}/.research/browsers`;
const { chromium } = await import('playwright');
if (!existsSync(chromium.executablePath())) {
  execFileSync(process.execPath, ['node_modules/playwright/cli.js', 'install', 'chromium', '--only-shell'], { env: process.env, stdio: 'inherit', timeout: 200000 });
}
if (!existsSync('LICENSE')) execFileSync('curl', ['-fLsS', '--max-time', '30', 'https://www.gnu.org/licenses/gpl-3.0.txt', '-o', 'LICENSE']);
// Minimal sandbox images lack Chromium's system libraries. Unpack test-only Debian packages locally;
// this neither installs OS services nor adds browser binaries to the application/public directory.
const libs = `${process.cwd()}/.research/sysroot`;
mkdirSync(`${libs}/packages`, { recursive: true });
mkdirSync(`${libs}/apt-lists/partial`, { recursive: true });
mkdirSync(`${libs}/apt-cache/archives/partial`, { recursive: true });
const aptOptions = ['-o', `Dir::State::lists=${libs}/apt-lists`, '-o', `Dir::Cache=${libs}/apt-cache`, '-o', 'Acquire::Languages=none'];
process.env.LD_LIBRARY_PATH = `${libs}/usr/lib/x86_64-linux-gnu:${libs}/lib/x86_64-linux-gnu`;
const shellDir = readdirSync(process.env.PLAYWRIGHT_BROWSERS_PATH).find(p => p.startsWith('chromium_headless_shell-'));
const shell = `${process.env.PLAYWRIGHT_BROWSERS_PATH}/${shellDir}/chrome-headless-shell-linux64/chrome-headless-shell`;
const packages = {
  'libnspr4.so': 'libnspr4', 'libnss3.so': 'libnss3', 'libnssutil3.so': 'libnss3', 'libsmime3.so': 'libnss3',
  'libatk-1.0.so.0': 'libatk1.0-0', 'libatk-bridge-2.0.so.0': 'libatk-bridge2.0-0', 'libatspi.so.0': 'libatspi2.0-0',
  'libcups.so.2': 'libcups2', 'libdrm.so.2': 'libdrm2', 'libXcomposite.so.1': 'libxcomposite1', 'libXdamage.so.1': 'libxdamage1',
  'libXfixes.so.3': 'libxfixes3', 'libXrandr.so.2': 'libxrandr2', 'libgbm.so.1': 'libgbm1', 'libxkbcommon.so.0': 'libxkbcommon0',
  'libasound.so.2': 'libasound2', 'libpango-1.0.so.0': 'libpango-1.0-0', 'libcairo.so.2': 'libcairo2', 'libdbus-1.so.3': 'libdbus-1-3',
  'libX11.so.6': 'libx11-6', 'libxcb.so.1': 'libxcb1', 'libXext.so.6': 'libxext6', 'libXrender.so.1': 'libxrender1',
  'libXi.so.6': 'libxi6',
  'libglib-2.0.so.0': 'libglib2.0-0', 'libgobject-2.0.so.0': 'libglib2.0-0', 'libgio-2.0.so.0': 'libglib2.0-0',
  'libharfbuzz.so.0': 'libharfbuzz0b', 'libthai.so.0': 'libthai0', 'libdatrie.so.1': 'libdatrie1', 'libfribidi.so.0': 'libfribidi0',
  'libwayland-server.so.0': 'libwayland-server0', 'libgraphite2.so.3': 'libgraphite2-3', 'libpixman-1.so.0': 'libpixman-1-0',
};
for (let attempt = 0; attempt < 5; attempt++) {
  const needed = execFileSync('ldd', [shell], { env: process.env, encoding: 'utf8' }).split('\n').filter(s => s.includes('not found')).map(s => s.trim().split(' ')[0]);
  console.log('[browser missing libraries]', needed);
  if (!needed.length) break;
  if (!existsSync(`${libs}/apt-ready`)) {
    execFileSync('apt-get', [...aptOptions, 'update'], { stdio: 'inherit', timeout: 90000 });
    writeFileSync(`${libs}/apt-ready`, 'local test dependency index');
  }
  const names = [...new Set(needed.map(lib => { if (!packages[lib]) throw new Error(`Unmapped browser dependency: ${lib}`); return packages[lib]; }))];
  execFileSync('apt-get', [...aptOptions, 'download', ...names], { cwd: `${libs}/packages`, stdio: 'inherit', timeout: 60000 });
  for (const file of readdirSync(`${libs}/packages`).filter(p => p.endsWith('.deb'))) execFileSync('dpkg-deb', ['-x', `${libs}/packages/${file}`, libs]);
}
const browser = await chromium.launch({ headless: true, args: ['--no-sandbox', '--disable-dev-shm-usage'] });
console.log('[browser available]', browser.version());
await browser.close();