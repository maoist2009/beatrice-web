const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

const root = process.env.INIT_CWD && fs.existsSync(process.env.INIT_CWD) ? process.env.INIT_CWD : path.resolve(__dirname, '..', '..');
const dest = path.join(root, 'vendor', 'fierce-cats-beatrice-trainer');
if (fs.existsSync(path.join(dest, '.git')) || fs.existsSync(path.join(dest, 'pyproject.toml'))) {
  console.log('[trainer-cloner] fierce-cats/beatrice-trainer already present');
} else {
  fs.mkdirSync(path.dirname(dest), { recursive: true });
  console.log('[trainer-cloner] git clone --depth 1 https://huggingface.co/fierce-cats/beatrice-trainer');
  execSync(`git clone --depth 1 https://huggingface.co/fierce-cats/beatrice-trainer "${dest}"`, {
    cwd: root,
    stdio: 'inherit',
    timeout: 300000,
    env: { ...process.env, GIT_TERMINAL_PROMPT: '0' },
  });
}
console.log('[trainer-cloner] HEAD:', execSync('git rev-parse HEAD', { cwd: dest }).toString().trim());