// Development-only: curl retrieves upstream sources/weights; nothing here is imported by src/.
import { execFileSync } from 'node:child_process';
import { mkdirSync, readFileSync, writeFileSync, existsSync, statSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { unzipSync } from 'fflate';

const root = '.research';
mkdirSync(root, { recursive: true });
function curl(url, file) {
  if (!existsSync(file)) {
    execFileSync('curl', ['-fL', '--retry', '2', '--connect-timeout', '15', '--max-time', '100', '-sS', url, '-o', file], { timeout: 220000, maxBuffer: 1024 * 1024 });
  }
  return file;
}
let env = '';
for (const [cmd, args] of [['curl', ['--version']], ['python3', ['-c', 'import sys,importlib.util; print(sys.version); print({x:bool(importlib.util.find_spec(x)) for x in ["torch","numpy","onnx","onnxruntime","torchaudio"]})']]]) {
  try { env += execFileSync(cmd, args, { env: { ...process.env, PYTHONPATH: `${process.cwd()}/.research/python` }, encoding: 'utf8', timeout: 15000 }); }
  catch (e) { env += String(e.message); }
}
writeFileSync(`${root}/environment.txt`, env);
console.log('[research environment]', env.slice(0, 1800));

const revision = '5ddb63ea854832a914a846a22830feb9ff7e691b';
curl(`https://huggingface.co/fierce-cats/beatrice-trainer/resolve/${revision}/beatrice_trainer/__main__.py`, `${root}/beta2.py`);
if (createHash('sha256').update(readFileSync(`${root}/beta2.py`)).digest('hex') !== '87f56c44a744ba012f14c036c34dcc8d6789a141a8c36fe00038c947561934ad') throw new Error('Pinned trainer source checksum mismatch');
curl(`https://huggingface.co/fierce-cats/beatrice-trainer/resolve/${revision}/LICENSE`, `${root}/LICENSE-trainer.txt`);
const manifest = { trainer: { repo: 'fierce-cats/beatrice-trainer', revision, sha256: createHash('sha256').update(readFileSync(`${root}/beta2.py`)).digest('hex') }, models: [] };

for (const [name, repo, zip, modelRevision, expected] of [
  ['shigure', 'yasyune/Shigure_Tokina_Beatrice_2.0.0-beta.2', 'paraphernalia_shigure_00005000.zip', '43c4e3aaf27aceb2b6fe74842331fd4bf013ac01', 'ccb2f9eafc3039547272f3183d7d7e8e82d025f3219c0632e8c2edaff5f9e116'],
  ['kurage', 'yasyune/Kurage_Kikoto_Beatrice_2.0.0-beta.2', 'paraphernalia_kurage_00005000.zip', '69c6155d7e716b31b4cb2fd28b9fd62f5b7395c5', '62a9338c904bcb07b36336d604e068dc9ff68b88ed66348ea00b9f38b4582f10'],
]) {
  const url = `https://huggingface.co/${repo}/resolve/${modelRevision}/${zip}`;
  const archive = curl(url, `${root}/${name}.zip`);
  const hash = createHash('sha256').update(readFileSync(archive)).digest('hex');
  if (hash !== expected) throw new Error(`${name}: SHA-256 mismatch`);
  const folder = `${root}/${name}`;
  mkdirSync(folder, { recursive: true });
  const entries = unzipSync(new Uint8Array(readFileSync(archive)));
  const files = [];
  for (const [path, content] of Object.entries(entries)) {
    if (path.endsWith('/')) continue;
    const file = path.split('/').pop();
    writeFileSync(`${folder}/${file}`, content);
    files.push({ path, bytes: content.byteLength, sha256: createHash('sha256').update(content).digest('hex') });
  }
  curl(`https://huggingface.co/${repo}/resolve/${modelRevision}/README.md`, `${folder}/MODEL-CARD.md`);
  manifest.models.push({ name, repo, revision: modelRevision, url, bytes: statSync(archive).size, sha256: hash, files });
  console.log(`[research ${name}]`, files.map(f => `${f.path}: ${f.bytes}`).join('\n'));
}
writeFileSync(`${root}/provenance.json`, JSON.stringify(manifest, null, 2));
console.log('[research] official beta.2 source:', revision);