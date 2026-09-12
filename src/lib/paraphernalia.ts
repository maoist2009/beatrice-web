/**
 * Beatrice 2.0.0-rc.0 "paraphernalia" loader.
 *
 * Facts used here are taken from the official sources:
 *  - beatrice_trainer/__main__.py (export code): every *.bin is written by `dump_params`
 *    AFTER `.half()`, i.e. raw little-endian float16 in `dump()` order, no header.
 *  - beatricelib/beatrice.h (20rc0 section): constants and the content of speaker_embeddings.bin
 *    (codebook / additive speaker embedding / formant-shift embedding / key-value speaker embedding).
 *  - CrossAttention.dump_kv (trainer): layout of embedding_setter.bin (per block, per head K/V weights, then biases).
 */
import { strToU8, unzipSync, zipSync } from "fflate";
import { parseToml, TomlTable } from "./toml";
import { resolveModelFormat, type ModelFormat } from './formats';

export const C = {
  IN_HOP: 160,
  OUT_HOP: 240,
  PITCH_BINS_PER_OCTAVE: 96,
  WG_HIDDEN: 256,
  IN_SR: 16000,
  OUT_SR: 24000,
  PHONE_CHANNELS: 128,
  PITCH_BINS: 448,
  CODEBOOK_SIZE: 512,
  KV_LENGTH: 384,
  KV_SPK_CH: 128,
  N_BLOCKS: 4,
  N_FORMANT: 9,
} as const;

export interface Voice {
  id: number;
  name: string;
  description: string;
  averagePitch: number | null; // MIDI note (toml `average_pitch`)
  portraitPath: string | null;
  portraitUrl: string | null;
}

export interface ParaphernaliaFiles {
  toml: string;
  tomlName: string;
  phone_extractor: ArrayBuffer;
  pitch_estimator: ArrayBuffer;
  waveform_generator: ArrayBuffer;
  speaker_embeddings: ArrayBuffer;
  embedding_setter: ArrayBuffer | null;
  formant_shift_embeddings?: ArrayBuffer | null;
  extras?: Record<string, Blob>;
  images: Record<string, Blob>;
}

export interface SpeakerEmbeddings {
  nSpeakers: number;
  codebook: Float32Array; // n * 512 * 128
  additive: Float32Array; // n * 256
  formant: Float32Array; // 9 * 256
  keyValue: Float32Array; // n * 384 * 128
  orderNote: string;
}

export interface EmbeddingSetter {
  attentionChannels: number; // qk_channels == vo_channels of the prenet cross-attention
  blocks: { kW: Float32Array[]; vW: Float32Array[]; kB: Float32Array[]; vB: Float32Array[] }[];
}

export interface Paraphernalia {
  name: string;
  version: string;
  format: ModelFormat;
  description: string;
  voices: Voice[];
  files: ParaphernaliaFiles;
  speakers: SpeakerEmbeddings;
  setter: EmbeddingSetter | null;
  sizes: Record<string, number>;
}

/** Export the exact files currently held by the browser as a VST/VCClient-compatible paraphernalia ZIP. */
export async function exportParaphernaliaZip(files: ParaphernaliaFiles): Promise<Blob> {
  const entries: Record<string, Uint8Array> = {
    [files.tomlName]: strToU8(files.toml),
    "phone_extractor.bin": new Uint8Array(files.phone_extractor),
    "pitch_estimator.bin": new Uint8Array(files.pitch_estimator),
    "waveform_generator.bin": new Uint8Array(files.waveform_generator),
    "speaker_embeddings.bin": new Uint8Array(files.speaker_embeddings),
  };
  if (files.embedding_setter) entries["embedding_setter.bin"] = new Uint8Array(files.embedding_setter);
  if (files.formant_shift_embeddings) entries['formant_shift_embeddings.bin'] = new Uint8Array(files.formant_shift_embeddings);
  for (const [name, blob] of Object.entries(files.images)) entries[name] = new Uint8Array(await blob.arrayBuffer());
  for (const [name, blob] of Object.entries(files.extras ?? {})) if (!entries[name]) entries[name] = new Uint8Array(await blob.arrayBuffer());
  // Stored ZIP avoids expensive recompression on phones; original model bytes are unchanged.
  const zipped = zipSync(entries, { level: 0 });
  return new Blob([zipped.slice().buffer], { type: "application/zip" });
}

// ---------- float16 ----------
export function f16ToF32(buf: ArrayBuffer, byteOffset = 0, count?: number): Float32Array {
  const n = count ?? ((buf.byteLength - byteOffset) / 2);
  if (!Number.isInteger(n) || byteOffset < 0 || byteOffset % 2 || n < 0 || byteOffset + n * 2 > buf.byteLength) throw new Error('Invalid FP16 buffer range');
  const u16 = new Uint16Array(buf, byteOffset, n);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const h = u16[i];
    const s = (h & 0x8000) ? -1 : 1;
    const e = (h >> 10) & 0x1f;
    const f = h & 0x3ff;
    if (e === 0) out[i] = s * f * 2 ** -24;
    else if (e === 31) out[i] = f ? NaN : s * Infinity;
    else out[i] = s * (1 + f / 1024) * 2 ** (e - 15);
  }
  return out;
}

// ---------- file collection (folder / multi-select / zip) ----------
export async function collectFiles(files: File[]): Promise<ParaphernaliaFiles> {
  const entries: { name: string; data: () => Promise<ArrayBuffer> }[] = [];
  for (const f of files) {
    if (/\.zip$/i.test(f.name)) {
      if (f.size > 160 * 1024 * 1024) throw new Error('Model archive exceeds the 160 MiB safety limit. Import its folder instead.');
      let total = 0;
      const z = unzipSync(new Uint8Array(await f.arrayBuffer()), { filter: (entry) => {
        total += entry.originalSize;
        if (entry.originalSize > 64 * 1024 * 1024 || total > 192 * 1024 * 1024) throw new Error('Expanded model exceeds the memory safety limit');
        return !entry.name.endsWith('/');
      } });
      for (const [path, bytes] of Object.entries(z)) {
        if (path.endsWith("/")) continue;
        const b = bytes;
        entries.push({ name: path.split("/").pop()!, data: async () => b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer });
      }
    } else {
      entries.push({ name: f.name, data: () => f.arrayBuffer() });
    }
  }
  const find = (n: string) => entries.find((e) => e.name.toLowerCase() === n);
  const need = async (n: string) => {
    const e = find(n);
    if (!e) throw new Error(`Missing ${n} — select the whole paraphernalia folder (or its .zip).`);
    return e.data();
  };
  const tomlEntry = entries.find((e) => /\.toml$/i.test(e.name));
  if (!tomlEntry) throw new Error("No beatrice_paraphernalia_*.toml found.");
  if (entries.filter(e => /\.toml$/i.test(e.name)).length !== 1) throw new Error('The archive contains multiple models. Import one paraphernalia folder at a time.');
  const images: Record<string, Blob> = {};
  for (const e of entries) {
    if (/\.(png|jpe?g|webp)$/i.test(e.name)) images[e.name] = new Blob([await e.data()]);
  }
  const setter = find("embedding_setter.bin");
  const formant = find('formant_shift_embeddings.bin');
  const extras: Record<string, Blob> = {};
  for (const entry of entries) {
    if (/\.(txt|md|json)$/i.test(entry.name) || /^licen[cs]e/i.test(entry.name)) extras[entry.name] = new Blob([await entry.data()]);
  }
  return {
    toml: new TextDecoder().decode(await tomlEntry.data()),
    tomlName: tomlEntry.name,
    phone_extractor: await need("phone_extractor.bin"),
    pitch_estimator: await need("pitch_estimator.bin"),
    waveform_generator: await need("waveform_generator.bin"),
    speaker_embeddings: await need("speaker_embeddings.bin"),
    embedding_setter: setter ? await setter.data() : null,
    // The official rc.0 archive uses embedding_setter.bin; a split variant puts the formant table in its
    // own formant_shift_embeddings.bin. Keep either; never synthesize one from the other.
    formant_shift_embeddings: formant ? await formant.data() : null,
    extras,
    images,
  };
}

// ---------- parsing ----------
export function parseParaphernalia(files: ParaphernaliaFiles, options: { metadataOnly?: boolean } = {}): Paraphernalia {
  const t = parseToml(files.toml);
  const model = (t.model as TomlTable) ?? {};
  const version = String(model.version ?? "unknown");
  const format = resolveModelFormat(version);
  const speakers = options.metadataOnly ? speakerMetadata(files, format) : format === 'beatrice-beta2' ? parseBetaSpeakers(files) : parseSpeakerEmbeddings(files.speaker_embeddings, 'beatrice-rc0', files);
  const setter = !options.metadataOnly && format === 'beatrice-rc0' && files.embedding_setter ? parseEmbeddingSetter(files.embedding_setter) : null;
  const portraitUrls = new Map<Blob, string>();
  const voicesT = (t.voice as TomlTable) ?? {};
  const voices: Voice[] = Object.keys(voicesT)
    .map(Number)
    .filter((k) => !Number.isNaN(k))
    .sort((a, b) => a - b)
    .map((id) => {
      const v = voicesT[String(id)] as TomlTable;
      const portrait = (v.portrait as TomlTable) ?? {};
      const path = typeof portrait.path === "string" ? portrait.path : null;
      const blob = path ? files.images[path.split("/").pop()!] : undefined;
      if (blob && !portraitUrls.has(blob)) portraitUrls.set(blob, URL.createObjectURL(blob));
      return {
        id,
        name: String(v.name ?? `voice ${id}`),
        description: String(v.description ?? ""),
        averagePitch: typeof v.average_pitch === "number" ? v.average_pitch : null,
        portraitPath: path,
        portraitUrl: blob ? portraitUrls.get(blob)! : null,
      };
    });

  // UI previews do not need ~46 MB of decoded JVS speaker embeddings. Decode those only in the worker.
  if (voices.some(v => !Number.isInteger(v.id) || v.id < 0 || v.id >= speakers.nSpeakers)) {
    for (const url of portraitUrls.values()) URL.revokeObjectURL(url);
    throw new Error('TOML voice ID exceeds speaker embedding count');
  }
  return {
    name: String(model.name ?? files.tomlName),
    version,
    format,
    description: String(model.description ?? ""),
    voices,
    files,
    speakers,
    setter,
    sizes: {
      phone_extractor: files.phone_extractor.byteLength,
      pitch_estimator: files.pitch_estimator.byteLength,
      waveform_generator: files.waveform_generator.byteLength,
      speaker_embeddings: files.speaker_embeddings.byteLength,
      embedding_setter: files.embedding_setter?.byteLength ?? 0,
      formant_shift_embeddings: files.formant_shift_embeddings?.byteLength ?? 0,
    },
  };
}

function speakerMetadata(files: ParaphernaliaFiles, format: ModelFormat): SpeakerEmbeddings {
  const perSpeaker = 512 * 128 + 256 + 384 * 128;
  const n = format === 'beatrice-beta2' ? files.speaker_embeddings.byteLength / 512
    : (files.speaker_embeddings.byteLength / 2 - (files.formant_shift_embeddings ? 0 : 9 * 256)) / perSpeaker;
  if (!Number.isInteger(n) || n < 1 || n > 512) throw new Error('Invalid speaker embedding dimensions');
  if (format === 'beatrice-beta2' && files.formant_shift_embeddings?.byteLength !== 4608) throw new Error('Missing beta.2 formant_shift_embeddings.bin');
  const empty = new Float32Array(0);
  return { nSpeakers: n, codebook: empty, additive: empty, formant: empty, keyValue: empty, orderNote: 'Metadata only; inference worker decodes actual parameters.' };
}

/** rc.0 speaker_embeddings sometimes ships split files (formant in its own .bin). */
function parseRc0Speakers(files: ParaphernaliaFiles, fixed: number, perSpeaker: number): SpeakerEmbeddings {
  const halves = files.speaker_embeddings.byteLength / 2;
  // Case A: legacy combined = 9*256 + n*perSpeaker.  Case B: combined sans formant = n*perSpeaker, with a
  // separate formant_shift_embeddings.bin (9*256). A distractor is formant inside + separate file, which
  // we refuse rather than double-apply.
  const hasFormantFile = files.formant_shift_embeddings != null;
  let n: number;
  let formant: Float32Array;
  if ((halves - fixed) % perSpeaker === 0 && !hasFormantFile) {
    n = (halves - fixed) / perSpeaker;
    const all = f16ToF32(files.speaker_embeddings);
    let o = 0;
    const take = (l: number) => { const s = all.subarray(o, o + l); o += l; return s; };
    const codebook = take(n * C.CODEBOOK_SIZE * C.PHONE_CHANNELS);
    const additive = take(n * C.WG_HIDDEN);
    formant = take(C.N_FORMANT * C.WG_HIDDEN);
    const keyValue = take(n * C.KV_LENGTH * C.KV_SPK_CH);
    return { nSpeakers: n, codebook, additive, formant, keyValue, orderNote: 'rc.0 combined: codebook, additive, formant, key/value (official export order)' };
  }
  if (halves % perSpeaker === 0 && hasFormantFile && files.formant_shift_embeddings!.byteLength === fixed * 2) {
    n = halves / perSpeaker;
    const all = f16ToF32(files.speaker_embeddings);
    let o = 0;
    const take = (l: number) => { const s = all.subarray(o, o + l); o += l; return s; };
    const codebook = take(n * C.CODEBOOK_SIZE * C.PHONE_CHANNELS);
    const additive = take(n * C.WG_HIDDEN);
    const keyValue = take(n * C.KV_LENGTH * C.KV_SPK_CH);
    return { nSpeakers: n, codebook, additive, formant: f16ToF32(files.formant_shift_embeddings!), keyValue, orderNote: 'rc.0 split: codebook+additive+kv, with separate formant_shift_embeddings.bin' };
  }
  throw new Error(`speaker_embeddings.bin size ${files.speaker_embeddings.byteLength} B is not (9*256 + n*(512*128+256+384*128)) f16 — not a 2.0.0-rc.0 file?`);
}

function parseBetaSpeakers(files: ParaphernaliaFiles): SpeakerEmbeddings {
  const n = files.speaker_embeddings.byteLength / (256 * 2);
  if (!Number.isInteger(n) || n < 1) throw new Error('Invalid beta.2 speaker_embeddings.bin');
  if (files.formant_shift_embeddings?.byteLength !== 9 * 256 * 2) throw new Error('beta.2 requires formant_shift_embeddings.bin (4608 bytes). It must not be replaced by rc.0 embedding_setter.bin.');
  return {
    nSpeakers: n, additive: f16ToF32(files.speaker_embeddings), formant: f16ToF32(files.formant_shift_embeddings),
    codebook: new Float32Array(0), keyValue: new Float32Array(0),
    orderNote: 'beta.2: separate additive-speaker and formant files; no VQ or cross-attention embeddings.',
  };
}

/** speaker_embeddings.bin (rc0). Supports the official combined layout (codebook/additive/formant/kv) and
 *  the split variant with a separate formant_shift_embeddings.bin. n_speakers is solved by integrality. */
export function parseSpeakerEmbeddings(buf: ArrayBuffer, format?: 'beatrice-rc0' | 'beatrice-beta2', files?: ParaphernaliaFiles): SpeakerEmbeddings {
  if (format === 'beatrice-rc0' && files) return parseRc0Speakers(files, C.N_FORMANT * C.WG_HIDDEN, C.CODEBOOK_SIZE * C.PHONE_CHANNELS + C.WG_HIDDEN + C.KV_LENGTH * C.KV_SPK_CH);
  const halves = buf.byteLength / 2;
  const perSpeaker = C.CODEBOOK_SIZE * C.PHONE_CHANNELS + C.WG_HIDDEN + C.KV_LENGTH * C.KV_SPK_CH;
  const fixed = C.N_FORMANT * C.WG_HIDDEN;
  const n = (halves - fixed) / perSpeaker;
  if (!Number.isInteger(n) || n <= 0) {
    throw new Error(
      `speaker_embeddings.bin size ${buf.byteLength} B is not (9*256 + n*(512*128 + 256 + 384*128)) float16 — not a 2.0.0-rc.0 file?`,
    );
  }
  const all = f16ToF32(buf);
  let o = 0;
  const take = (len: number) => { const s = all.subarray(o, o + len); o += len; return s; };
  const codebook = take(n * C.CODEBOOK_SIZE * C.PHONE_CHANNELS);
  const additive = take(n * C.WG_HIDDEN);
  const formant = take(C.N_FORMANT * C.WG_HIDDEN);
  const keyValue = take(n * C.KV_LENGTH * C.KV_SPK_CH);
  return { nSpeakers: n, codebook, additive, formant, keyValue, orderNote: 'rc.0 combined: codebook, additive, formant, key/value (official export order)' };
}

/** embedding_setter.bin = 4 × CrossAttention.dump_kv (per head: K W, V W ; then per head: K b, V b). */
export function parseEmbeddingSetter(buf: ArrayBuffer): EmbeddingSetter {
  const halves = buf.byteLength / 2;
  // per block: heads*(hqk*128 + hvo*128) + heads*(hqk + hvo) = A*128*2 + 2A = 258 A  (A = qk = vo channels)
  const A = halves / (C.N_BLOCKS * 258);
  if (!Number.isInteger(A)) throw new Error(`embedding_setter.bin size ${buf.byteLength} B does not match 4 × dump_kv layout`);
  const heads = 4; // ConvNeXtBlock default num_heads
  const hd = A / heads;
  const all = f16ToF32(buf);
  let o = 0;
  const take = (len: number) => { const s = all.subarray(o, o + len); o += len; return s; };
  const blocks = [];
  for (let b = 0; b < C.N_BLOCKS; b++) {
    const kW: Float32Array[] = [], vW: Float32Array[] = [], kB: Float32Array[] = [], vB: Float32Array[] = [];
    for (let h = 0; h < heads; h++) { kW.push(take(hd * C.KV_SPK_CH)); vW.push(take(hd * C.KV_SPK_CH)); }
    for (let h = 0; h < heads; h++) { kB.push(take(hd)); vB.push(take(hd)); }
    blocks.push({ kW, vW, kB, vB });
  }
  return { attentionChannels: A, blocks };
}
