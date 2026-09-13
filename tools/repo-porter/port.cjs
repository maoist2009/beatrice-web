/**
 * repo-porter: 把克隆下来的 beatrice-web 源码搬进本工程 src 树（文件系统级复制，
 * 保留 ?worker&inline、字符串 worklet 等构建特性）。幂等：src/.beatrice-ported 存在即跳过。
 */
const fs = require('fs');
const path = require('path');

function run() {
  const ROOT = process.env.INIT_CWD && fs.existsSync(process.env.INIT_CWD)
    ? process.env.INIT_CWD
    : path.resolve(__dirname, '..', '..');
  const V = path.join(ROOT, 'vendor', 'beatrice-web');
  const marker = path.join(ROOT, 'src', '.beatrice-ported');
  if (fs.existsSync(marker)) { console.log('[repo-porter] already ported, skip'); return; }
  if (!fs.existsSync(path.join(V, 'package.json'))) { console.log('[repo-porter] vendor clone missing, skip'); return; }

  const copyDir = (from, to) => {
    fs.mkdirSync(to, { recursive: true });
    for (const name of fs.readdirSync(from)) {
      const f = path.join(from, name), t = path.join(to, name);
      const st = fs.statSync(f);
      if (st.isDirectory()) copyDir(f, t);
      else fs.copyFileSync(f, t);
    }
  };

  copyDir(path.join(V, 'src', 'lib'), path.join(ROOT, 'src', 'lib'));
  copyDir(path.join(V, 'src', 'utils'), path.join(ROOT, 'src', 'utils'));
  fs.copyFileSync(path.join(V, 'src', 'App.tsx'), path.join(ROOT, 'src', 'App.tsx'));
  fs.copyFileSync(path.join(V, 'src', 'index.css'), path.join(ROOT, 'src', 'index.css'));
  fs.copyFileSync(path.join(V, 'src', 'vite-env.d.ts'), path.join(ROOT, 'src', 'vite-env.d.ts'));
  fs.copyFileSync(path.join(V, 'README.md'), path.join(ROOT, 'BEATRICE-README.md'));
  fs.writeFileSync(marker, new Date().toISOString() + '\n');

  const count = (dir) => fs.readdirSync(dir).reduce((n, x) => {
    const p = path.join(dir, x); return n + (fs.statSync(p).isDirectory() ? count(p) : 1);
  }, 0);
  console.log(`[repo-porter] ported ${count(path.join(ROOT, 'src', 'lib'))} lib files + App.tsx + index.css into src/`);
}

run();
