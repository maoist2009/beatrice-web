/* eslint-disable */
/**
 * repo-cloner
 * 目标：把 https://github.com/maoist2009/beatrice-web.git 克隆到 <项目根>/vendor/beatrice-web
 * 策略（按顺序）：
 *   1) git clone --depth 1 （真正的 git）
 *   2) 查询 GitHub API 拿 default_branch，下载 codeload tar.gz，纯 Node zlib + 手写 tar 解包
 *   3) 扫描 LFS 指针文件并通过 LFS batch API 下载真实二进制
 */
const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');
const https = require('https');
const zlib = require('zlib');

const OWNER = 'maoist2009';
const REPO = 'beatrice-web';
const REPO_HTTPS = `https://github.com/${OWNER}/${REPO}.git`;

const log = (...a) => console.log('[repo-cloner]', ...a);

function findRoot() {
  if (process.env.INIT_CWD && fs.existsSync(process.env.INIT_CWD)) return process.env.INIT_CWD;
  let dir = __dirname;
  for (let i = 0; i < 8; i++) {
    if (
      fs.existsSync(path.join(dir, 'vite.config.ts')) ||
      fs.existsSync(path.join(dir, 'vite.config.js')) ||
      fs.existsSync(path.join(dir, 'index.html'))
    ) {
      return dir;
    }
    dir = path.dirname(dir);
  }
  return process.cwd();
}

const ROOT = findRoot();
const DEST = path.join(ROOT, 'vendor', 'beatrice-web');

function httpsRequest(url, opts = {}) {
  return new Promise((resolve, reject) => {
    const u = new URL(url);
    const req = https.request(
      {
        hostname: u.hostname,
        path: u.pathname + u.search,
        method: opts.method || 'GET',
        headers: Object.assign({ 'User-Agent': 'node-repo-cloner' }, opts.headers || {}),
      },
      (res) => {
        const chunks = [];
        res.on('data', (c) => chunks.push(c));
        res.on('end', () => resolve({ status: res.statusCode, headers: res.headers, body: Buffer.concat(chunks) }));
        res.on('error', reject);
      }
    );
    req.on('error', reject);
    req.setTimeout(120000, () => req.destroy(new Error('timeout')));
    if (opts.body) req.write(opts.body);
    req.end();
  });
}

async function httpsGetBuffer(url, depth = 0) {
  const res = await httpsRequest(url);
  if ([301, 302, 307, 308].includes(res.status)) {
    if (depth > 5) throw new Error('too many redirects');
    return httpsGetBuffer(new URL(res.headers.location, url).toString(), depth + 1);
  }
  if (res.status !== 200) throw new Error(`HTTP ${res.status} for ${url}`);
  return res.body;
}

function tryGit() {
  try {
    execSync('git --version', { stdio: 'ignore' });
  } catch (e) {
    log('git binary not available:', e.message.split('\n')[0]);
    return false;
  }
  try {
    log('running: git clone --depth 1', REPO_HTTPS);
    execSync(`git clone --depth 1 ${REPO_HTTPS} "${DEST}"`, {
      stdio: 'inherit',
      timeout: 240000,
      env: Object.assign({}, process.env, { GIT_TERMINAL_PROMPT: '0' }),
    });
    return fs.existsSync(path.join(DEST, 'package.json'));
  } catch (e) {
    log('git clone failed:', e.message.split('\n')[0]);
    return false;
  }
}

async function getDefaultBranch() {
  try {
    const buf = await httpsGetBuffer(`https://api.github.com/repos/${OWNER}/${REPO}`);
    const j = JSON.parse(buf.toString('utf8'));
    if (j && j.default_branch) return j.default_branch;
  } catch (e) {
    log('default branch lookup failed:', e.message);
  }
  return null;
}

function parseTar(raw) {
  const entries = [];
  let off = 0;
  while (off + 512 <= raw.length) {
    const header = raw.subarray(off, off + 512);
    if (header.every((b) => b === 0)) break;
    let name = header.subarray(0, 100).toString('utf8').replace(/\0.*$/, '');
    const magic = header.subarray(257, 262).toString('utf8');
    if (magic === 'ustar') {
      const prefix = header.subarray(345, 500).toString('utf8').replace(/\0.*$/, '');
      if (prefix) name = prefix + '/' + name;
    }
    const sizeStr = header.subarray(124, 136).toString('utf8').replace(/\0.*$/, '').trim();
    const size = sizeStr ? parseInt(sizeStr, 8) : 0;
    const typeFlag = String.fromCharCode(header[156]);
    off += 512;
    const data = raw.subarray(off, off + size);
    off += Math.ceil(size / 512) * 512;
    entries.push({ name, typeFlag, data });
  }
  return entries;
}

async function tarballFallback() {
  let branch = await getDefaultBranch();
  const candidates = [];
  if (branch) candidates.push(branch);
  candidates.push('main', 'master');
  let buf = null;
  for (const b of [...new Set(candidates)]) {
    const url = `https://codeload.github.com/${OWNER}/${REPO}/tar.gz/refs/heads/${b}`;
    try {
      log('downloading tarball:', url);
      buf = await httpsGetBuffer(url);
      log('tarball size:', (buf.length / 1024 / 1024).toFixed(2), 'MB');
      break;
    } catch (e) {
      log('tarball failed for', b, '->', e.message);
    }
  }
  if (!buf) throw new Error('git and tarball both failed');
  const raw = zlib.gunzipSync(buf);
  const entries = parseTar(raw);
  log('tar entries:', entries.length);
  fs.mkdirSync(DEST, { recursive: true });
  for (const e of entries) {
    const rel = e.name.split('/').slice(1).join('/'); // strip "beatrice-web-<branch>/"
    if (!rel) continue;
    const out = path.join(DEST, rel);
    if (!out.startsWith(DEST)) continue;
    if (e.typeFlag === '5' || e.name.endsWith('/')) {
      fs.mkdirSync(out, { recursive: true });
      continue;
    }
    if (e.typeFlag === '0' || e.typeFlag === '\0' || e.typeFlag === '') {
      fs.mkdirSync(path.dirname(out), { recursive: true });
      fs.writeFileSync(out, e.data);
    }
  }
}

function walk(dir, acc = []) {
  for (const name of fs.readdirSync(dir)) {
    if (name === '.git' || name === 'node_modules') continue;
    const p = path.join(dir, name);
    const st = fs.statSync(p);
    if (st.isDirectory()) walk(p, acc);
    else acc.push(p);
  }
  return acc;
}

async function resolveLfs(files) {
  const pointers = [];
  for (const f of files) {
    try {
      const st = fs.statSync(f);
      if (st.size > 4096) continue;
      const head = fs.readFileSync(f).subarray(0, 300).toString('utf8');
      if (!head.startsWith('version https://git-lfs.github.com/spec/v1')) continue;
      const oid = (head.match(/oid sha256:([0-9a-f]{64})/) || [])[1];
      const size = parseInt((head.match(/size (\d+)/) || [])[1] || '0', 10);
      if (oid) pointers.push({ file: f, oid, size });
    } catch (e) {
      /* ignore */
    }
  }
  if (!pointers.length) return;
  log('LFS pointers found:', pointers.map((p) => path.relative(DEST, p.file)).join(', '));
  const body = JSON.stringify({
    operation: 'download',
    transfers: ['basic'],
    objects: pointers.map((p) => ({ oid: p.oid, size: p.size })),
  });
  const res = await httpsRequest(`https://github.com/${OWNER}/${REPO}.git/info/lfs/objects/batch`, {
    method: 'POST',
    headers: {
      Accept: 'application/vnd.git-lfs+json',
      'Content-Type': 'application/vnd.git-lfs+json',
      'Content-Length': Buffer.byteLength(body),
    },
    body,
  });
  if (res.status !== 200) throw new Error('LFS batch HTTP ' + res.status);
  const j = JSON.parse(res.body.toString('utf8'));
  for (const obj of j.objects || []) {
    const p = pointers.find((x) => x.oid === obj.oid);
    if (!p || !obj.actions || !obj.actions.download) continue;
    log('LFS downloading', path.relative(DEST, p.file), (obj.size / 1024 / 1024).toFixed(2), 'MB');
    const data = await httpsGetBuffer(obj.actions.download.href);
    fs.writeFileSync(p.file, data);
  }
}

(async () => {
  log('project root:', ROOT);
  if (fs.existsSync(path.join(DEST, 'package.json'))) {
    log('vendor/beatrice-web already present -> skip');
    return;
  }
  fs.rmSync(DEST, { recursive: true, force: true });
  fs.mkdirSync(path.join(ROOT, 'vendor'), { recursive: true });

  let ok = tryGit();
  if (!ok) {
    log('falling back to codeload tarball + node tar parser');
    await tarballFallback();
  }
  if (!fs.existsSync(path.join(DEST, 'package.json'))) {
    throw new Error('clone produced no package.json at ' + DEST);
  }

  const files = walk(DEST);
  await resolveLfs(files);

  log('CLONE OK —', files.length, 'files');
  const sizes = files
    .map((f) => ({ f, s: fs.statSync(f).size }))
    .sort((a, b) => b.s - a.s)
    .slice(0, 12);
  for (const x of sizes) log('  ', (x.s / 1024).toFixed(1).padStart(10), 'KB', path.relative(DEST, x.f));
  log('--- package.json ---');
  log(fs.readFileSync(path.join(DEST, 'package.json'), 'utf8'));
})().catch((e) => {
  console.error('[repo-cloner] FATAL:', e);
  process.exit(1);
});
