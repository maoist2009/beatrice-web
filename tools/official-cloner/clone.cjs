const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

const root = process.env.INIT_CWD && fs.existsSync(process.env.INIT_CWD) ? process.env.INIT_CWD : path.resolve(__dirname, '..', '..');
const dest = path.join(root, 'vendor', 'prj-beatrice-vst');
if (fs.existsSync(path.join(dest, '.git')) || fs.existsSync(path.join(dest, 'CMakeLists.txt'))) {
  console.log('[official-cloner] prj-beatrice/beatrice-vst already present');
} else {
  fs.mkdirSync(path.dirname(dest), { recursive: true });
  console.log('[official-cloner] git clone --depth 1 https://github.com/prj-beatrice/beatrice-vst.git');
  execSync(`git clone --depth 1 https://github.com/prj-beatrice/beatrice-vst.git "${dest}"`, {
    cwd: root,
    stdio: 'inherit',
    timeout: 300000,
    env: { ...process.env, GIT_TERMINAL_PROMPT: '0' },
  });
}
if (!fs.existsSync(path.join(dest, 'CMakeLists.txt'))) throw new Error('official VST clone has no CMakeLists.txt');
console.log('[official-cloner] commit:', execSync('git rev-parse HEAD', { cwd: dest }).toString().trim());