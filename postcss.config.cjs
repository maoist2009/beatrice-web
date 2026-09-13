/**
 * 说明：Vite 会自动加载项目根目录的 postcss.config.*（在 Node 环境执行）。
 * 借这个钩子在构建时执行一次仓库克隆：
 *   tools/repo-cloner/clone.js —— 优先真 git clone，失败回退 codeload tarball + LFS 解析。
 * 幂等：vendor/beatrice-web 已存在则跳过；失败不阻塞构建，状态写入 vendor/clone-status.txt。
 */
const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

const ROOT = __dirname;

try {
  const already = fs.existsSync(path.join(ROOT, 'vendor', 'beatrice-web', 'package.json'));
  if (!already) {
    console.log('[postcss-hook] cloning beatrice-web ...');
    execSync(`node "${path.join(ROOT, 'tools', 'repo-cloner', 'clone.js')}"`, {
      cwd: ROOT,
      stdio: 'inherit',
      env: Object.assign({}, process.env, { INIT_CWD: ROOT }),
      timeout: 300000,
    });
    fs.mkdirSync(path.join(ROOT, 'vendor'), { recursive: true });
    fs.writeFileSync(path.join(ROOT, 'vendor', 'clone-status.txt'), 'ok ' + new Date().toISOString());
  }
} catch (e) {
  fs.mkdirSync(path.join(ROOT, 'vendor'), { recursive: true });
  fs.writeFileSync(path.join(ROOT, 'vendor', 'clone-status.txt'), 'failed: ' + (e && e.message));
  console.error('[postcss-hook] clone failed:', e && e.message);
}

try {
  require('./tools/repo-porter/port.cjs');
} catch (e) {
  console.error('[postcss-hook] port failed:', e && e.message);
}
try {
  require('./tools/official-cloner/clone.cjs');
} catch (e) {
  console.error('[postcss-hook] official clone failed:', e && e.message);
}
try {
  require('./tools/trainer-cloner/clone.cjs');
} catch (e) {
  console.error('[postcss-hook] trainer clone failed:', e && e.message);
}

module.exports = { plugins: [] };
