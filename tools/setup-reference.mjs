// Optional development environment, isolated from npm dependencies and the deployed app.
import { execFileSync } from 'node:child_process';
import { existsSync, mkdirSync, readFileSync } from 'node:fs';
const dir = '.research';
mkdirSync(dir, { recursive: true });
for (const p of ['/sys/fs/cgroup/memory.max', '/sys/fs/cgroup/memory/memory.limit_in_bytes']) {
  if (existsSync(p)) console.log('[memory limit]', readFileSync(p, 'utf8').trim());
}
const env = { ...process.env, PYTHONPATH: `${process.cwd()}/${dir}/pip:${process.cwd()}/${dir}/python`, OMP_NUM_THREADS: '1', OPENBLAS_NUM_THREADS: '1', MKL_NUM_THREADS: '1' };
function run(args) { return execFileSync('python3', args, { env, encoding: 'utf8', timeout: 260000, maxBuffer: 4 * 1024 * 1024 }); }
if (!existsSync(`${dir}/pip/pip`)) {
  execFileSync('curl', ['-fLsS', '--max-time', '60', 'https://bootstrap.pypa.io/get-pip.py', '-o', `${dir}/get-pip.py`]);
  console.log(run([`${dir}/get-pip.py`, '--target', `${dir}/pip`, '--no-cache-dir', '--no-warn-script-location']));
}
if (!existsSync(`${dir}/python/torch`)) {
  console.log(run(['-m', 'pip', 'install', '--target', `${dir}/python`, '--no-cache-dir', '--no-deps', '--index-url', 'https://download.pytorch.org/whl/cpu', 'torch==2.5.1+cpu']));
}
if (!existsSync(`${dir}/python/onnxruntime`)) {
  console.log(run(['-m', 'pip', 'install', '--target', `${dir}/python`, '--no-cache-dir', 'numpy==1.26.4', 'onnx==1.17.0', 'onnxruntime==1.20.1', 'filelock', 'typing_extensions', 'sympy==1.13.1', 'networkx', 'jinja2', 'fsspec']));
}
console.log(run(['-c', 'import torch,numpy,onnx,onnxruntime; print("REFERENCE READY",torch.__version__,numpy.__version__,onnx.__version__,onnxruntime.__version__)']));