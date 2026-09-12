/**
 * Built-in Beatrice 2.0.0-rc.0 models fetched directly from HuggingFace.
 * HF `resolve/` URLs send `Access-Control-Allow-Origin: *`, so the browser can download them cross-origin
 * with no proxy. Files are handed to collectFiles() (zip-aware) and then cached in IndexedDB.
 */
import { collectFiles, type ParaphernaliaFiles } from "./paraphernalia";

export interface CatalogEntry {
  id: string;
  name: string;
  author: string;
  license: string;
  bytes: number;
  url: string;
  zip: boolean;
  note: string;
  version?: string;
  sha256?: string;
  modelCard?: string;
  termsUrl?: string;
}

export interface ExternalModelEntry {
  name: string;
  version: string;
  source: string;
  url: string;
  note: string;
}

export const CATALOG: CatalogEntry[] = [
  {
    id: "wok000-jvs-rc0",
    name: "JVS rc.0 (100 Japanese corpus voices)",
    author: "wok000 / Beatrice JVS corpus package",
    license: "Beatrice paraphernalia terms (JVS corpus models)",
    bytes: 40540637,
    version: "2.0.0-rc.0",
    url: "https://huggingface.co/wok000/vcclient_model/resolve/main/beatrice_v2_rc0/beatrice_2.0.0-rc.0_20250824.zip?download=true",
    sha256: "48dab9c4de25c66fc21d8b54b6adec784b1f521800ac26ca45d0fe6baa6d26a8",
    zip: true,
    modelCard: "https://huggingface.co/wok000/vcclient_model/resolve/main/beatrice_v2_rc0/readme.md",
    termsUrl: "https://prj-beatrice.com/",
    note: "JVS corpus package with 100 speaker embeddings. This is NOT the separate Official Model 1 containing Tsukuyomi-chan, Tokina Shigure and OLUNE. It is not a Mandarin-trained model.",
  },
  {
    id: "hecko-old-tts",
    name: "Old TTS voices (8 speakers)",
    author: "hecko",
    license: "Author disclaims model copyright; see source/voice terms",
    bytes: 19172313,
    url: "https://huggingface.co/hecko/beatrice-old-tts/resolve/74fafd8f8840b51005700bebe244407885b262b8/old%20tts.zip?download=true",
    zip: true,
    version: '2.0.0-rc.0',
    sha256: '0229766f5f4fd60c3ba6883eb49a8c78caac48548310e8db4b30cb19aef30423',
    modelCard: 'https://huggingface.co/hecko/beatrice-old-tts/resolve/74fafd8f8840b51005700bebe244407885b262b8/README.md',
    termsUrl: 'https://huggingface.co/hecko/beatrice-old-tts',
    note: "Classic TTS voices (SAM, DECtalk/Paul, eSpeak, Speak & Spell, XP Mike/Mary…) trained for the 2.0.0-rc.0 architecture. Verified to contain the full rc.0 paraphernalia (phone/pitch/waveform/speaker/embedding-setter).",
  },
  {
    id: 'yasyune-shigure-beta2', name: '刻鳴時雨 (CV: 丸ころ)', author: 'yasyune / 管理者: 瓶詰め',
    version: '2.0.0-beta.2', license: 'Character/corpus terms; attribution required', bytes: 22504613, zip: true,
    url: 'https://huggingface.co/yasyune/Shigure_Tokina_Beatrice_2.0.0-beta.2/resolve/43c4e3aaf27aceb2b6fe74842331fd4bf013ac01/paraphernalia_shigure_00005000.zip?download=true',
    sha256: 'ccb2f9eafc3039547272f3183d7d7e8e82d025f3219c0632e8c2edaff5f9e116',
    modelCard: 'https://huggingface.co/yasyune/Shigure_Tokina_Beatrice_2.0.0-beta.2/resolve/43c4e3aaf27aceb2b6fe74842331fd4bf013ac01/README.md',
    termsUrl: 'https://bindume-chan.booth.pm/items/3640133',
    note: 'Japanese ITA-corpus voice, GRU-based beta.2 architecture. Credit 刻鳴時雨 (CV: 丸ころ), 管理者: 瓶詰め.',
  },
  {
    id: 'yasyune-kurage-beta2', name: '黄琴海月 / Kikoto Kurage', author: 'yasyune',
    version: '2.0.0-beta.2', license: 'Kikoto Kurage terms of use', bytes: 22505195, zip: true,
    url: 'https://huggingface.co/yasyune/Kurage_Kikoto_Beatrice_2.0.0-beta.2/resolve/69c6155d7e716b31b4cb2fd28b9fd62f5b7395c5/paraphernalia_kurage_00005000.zip?download=true',
    sha256: '62a9338c904bcb07b36336d604e068dc9ff68b88ed66348ea00b9f38b4582f10',
    modelCard: 'https://huggingface.co/yasyune/Kurage_Kikoto_Beatrice_2.0.0-beta.2/resolve/69c6155d7e716b31b4cb2fd28b9fd62f5b7395c5/README.md',
    termsUrl: 'https://kikyohiroto1227.wixsite.com/kikoto-utau/terms-of-service',
    note: 'Japanese ITA/MANA-corpus voice, GRU-based beta.2 architecture, separate formant embeddings.',
  },
];

/** Discovery links only. Paid models are not mirrored because their licences forbid redistribution. */
export const EXTERNAL_MODELS: ExternalModelEntry[] = [
  { name: "Cherry", version: "2.0.0-rc.0", source: "BOOTH · hananovoice01", url: "https://booth.pm/ja/items/7002889", note: "Paid; whisper/singing examples; redistribution prohibited." },
  { name: "Mallw (+ related voice collection)", version: "2.0.0-rc.0", source: "BOOTH · hananovoice01", url: "https://booth.pm/ja/items/7010858", note: "Paid; page links to Lunaria, Heath, Anemone, Erica, Gaura and others; redistribution prohibited." },
  { name: "あこちゃん", version: "2.0.0-rc.0", source: "BOOTH · sippoppoo", url: "https://booth.pm/ja/items/6593875", note: "Paid Japanese voice; commercial/singing use listed by the seller." },
];

export async function fetchCatalogModel(
  entry: CatalogEntry,
  onProgress?: (loaded: number, total: number) => void,
  signal?: AbortSignal,
): Promise<ParaphernaliaFiles> {
  const res = await fetch(entry.url, { redirect: "follow", credentials: 'omit', signal });
  if (!res.ok) throw new Error(`HuggingFace returned ${res.status} for ${entry.name}`);
  const total = Number(res.headers.get("content-length")) || entry.bytes;
  const maxBytes = 160 * 1024 * 1024;
  if (total > maxBytes) throw new Error('Archive exceeds the 160 MiB download limit');
  let buf: Uint8Array;
  if (res.body && onProgress) {
    const reader = res.body.getReader();
    const chunks: Uint8Array[] = [];
    let loaded = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      loaded += value.length;
      if (loaded > maxBytes) { await reader.cancel(); throw new Error('Archive exceeds the download memory limit'); }
      onProgress(loaded, total);
    }
    buf = new Uint8Array(loaded);
    let o = 0;
    for (const c of chunks) { buf.set(c, o); o += c.length; }
  } else {
    buf = new Uint8Array(await res.arrayBuffer());
    onProgress?.(total, total);
  }
  if (entry.sha256) {
    const digest = await crypto.subtle.digest('SHA-256', buf.buffer as ArrayBuffer);
    const hash = Array.from(new Uint8Array(digest), n => n.toString(16).padStart(2, '0')).join('');
    if (hash !== entry.sha256) throw new Error('Model SHA-256 mismatch; download was not accepted');
  }
  signal?.throwIfAborted();
  const files = await collectFiles([new File([buf.buffer as ArrayBuffer], 'model.zip', { type: 'application/zip' })]);
  files.extras ??= {};
  files.extras['DOWNLOAD-SOURCE.txt'] = new Blob([`${entry.name}\nAuthor: ${entry.author}\nSource: ${entry.url}\nSHA-256: ${entry.sha256 ?? 'not supplied'}\nTerms: ${entry.termsUrl ?? entry.license}\n`]);
  if (entry.modelCard) {
    const card = await fetch(entry.modelCard, { signal, credentials: 'omit' });
    if (!card.ok) throw new Error('Unable to download the model card. Retry to preserve the model attribution.');
    files.extras['MODEL-CARD.md'] = await card.blob();
  }
  return files;
}

export function customHuggingFaceEntry(value: string): CatalogEntry {
  const url = new URL(value.trim());
  if (url.protocol !== 'https:' || url.hostname !== 'huggingface.co' || url.username || url.password) throw new Error('Use an HTTPS huggingface.co ZIP URL');
  url.pathname = url.pathname.replace('/blob/', '/resolve/');
  if (!url.pathname.includes('/resolve/') || !/\.zip$/i.test(url.pathname)) throw new Error('Select a public paraphernalia .zip under /resolve/');
  url.searchParams.set('download', 'true');
  return { id: 'custom-hf', name: decodeURIComponent(url.pathname.split('/').pop()!), author: 'HuggingFace author', license: 'See repository terms', bytes: 0, zip: true, url: url.href, note: 'Custom download: format and file sizes are validated after extraction; no pinned upstream checksum.' };
}
