// Development-only: probes HuggingFace for downloadable rc.0/beta.2 paraphernalia archives,
// validates revision/sha via curl, and prints a catalog fragment. Not bundled into dist.
import { execFileSync } from 'node:child_process';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { createHash } from 'node:crypto';

mkdirSync('.research/discover', { recursive: true });
function curl(args) {
  return execFileSync('curl', ['-fLsS', '--connect-timeout', '10', '--max-time', '25', ...args], { encoding: 'utf8', timeout: 40000, maxBuffer: 8 * 1024 * 1024 });
}
function api(url) { return JSON.parse(curl([url])); }

const candidates = new Map();
// Anchor searches that previously surfaced real rc.0/beta packages.
for (const q of ['beatrice', 'Beatrice_2.0.0', 'paraphernalia', 'voice-conversion beatrice', 'Tsukuyomi つくよみちゃん', 'OLUNE Beatrice']) {
  let results = [];
  try { results = api(`https://huggingface.co/api/models?search=${encodeURIComponent(q)}&limit=50&full=true`); }
  catch (e) { console.warn('[search failed]', q, String(e).slice(0, 200)); }
  for (const r of results) candidates.set(r.modelId ?? r.id, r);
}

const accepted = [];
for (const [id, meta] of candidates) {
  let detail;
  try { detail = api(`https://huggingface.co/api/models/${id}?blobs=true`); } catch { continue; }
  const rev = detail.sha;
  for (const f of detail.siblings ?? []) {
    const name = f.rfilename;
    if (!/\.zip$/i.test(name)) continue;
    if (/(checkpoint|events|log)/i.test(name)) continue;
    const url = `https://huggingface.co/${id}/resolve/${rev}/${encodeURI(name)}?download=true`;
    // HEAD via range-free curl; skip huge archives without downloading them.
    let head = '';
    try { head = curl(['-I', url]); } catch { continue; }
    const size = Number(/content-length:\s*(\d+)/i.exec(head)?.[1] ?? f.lfs?.size ?? 0);
    if (size <= 0 || size > 160 * 1024 * 1024) continue;
    const marker = `${name} ${JSON.stringify(detail.cardData ?? {})}`.toLowerCase();
    let format = null;
    if (/rc\.0|rc-0|rc0/.test(marker) && /2\.0/.test(marker)) format = 'beatrice-rc0';
    else if (/beta\.1|beta\.2|beta1|beta2/.test(marker)) format = 'beatrice-beta2';
    if (!format) {
      // Peek at the TOML inside the zip only when affordable is not possible server-side; accept
      // nothing by guessing. Names that don't say rc.0/beta.* stay out of the catalog.
      continue;
    }
    accepted.push({ id, name, format, url, bytes: size, sha256: f.lfs?.sha256 ?? null, title: detail.cardData?.language?.join(',') ?? '', note: `${detail.cardData?.tags?.join(', ') ?? ''}` });
  }
}
const out = accepted.sort((a, b) => a.format.localeCompare(b.format) || a.id.localeCompare(b.id));
writeFileSync('.research/discover/catalog.json', JSON.stringify(out, null, 2));

// Authoritative probe of the rc.0 archive surfaced by search: verify the TOML + all bin layouts with curl.
import { unzipSync } from 'fflate';
const probe = 'https://huggingface.co/wok000/vcclient_model/resolve/main/beatrice_v2_rc0/beatrice_2.0.0-rc.0_20250824.zip?download=true';
const zbin = execFileSync('curl', ['-fLsS', '--max-time', '180', probe, '-o', '.research/discover/official-rc0.zip'], { stdio: 'inherit', timeout: 200000 });
void zbin;
const bytes = readFileSync('.research/discover/official-rc0.zip');
const sha = createHash('sha256').update(bytes).digest('hex');
const z = unzipSync(new Uint8Array(bytes));
const names = Object.keys(z).filter(p => !p.endsWith('/'));
const tomlName = names.find(p => /\.toml$/i.test(p));
const sizes = Object.fromEntries(names.map(p => [p, z[p].length]));
writeFileSync('.research/discover/official-rc0.json', JSON.stringify({ url: probe, bytes: bytes.length, sha256: sha, toml: tomlName, tomlBody: tomlName ? new TextDecoder().decode(z[tomlName]) : null, files: sizes }, null, 2));
console.log('[official rc.0 probe]\nsha256', sha, '\nfiles', names.map(n => `${n}:${sizes[n]}`).join('\n'));
