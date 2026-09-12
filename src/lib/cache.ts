/** IndexedDB cache for loaded paraphernalia (raw files, so re-loading needs no folder picker). */
import type { ParaphernaliaFiles } from "./paraphernalia";

const DB = "beatrice-web-cache";
const STORE = "models";
const META = 'metadata';

export interface CachedModelMeta { key: string; name: string; version: string; bytes: number; savedAt: number }

function open(): Promise<IDBDatabase> {
  return new Promise((res, rej) => {
    const r = indexedDB.open(DB, 2);
    r.onupgradeneeded = () => {
      const db = r.result;
      if (!db.objectStoreNames.contains(STORE)) db.createObjectStore(STORE, { keyPath: 'key' });
      if (!db.objectStoreNames.contains(META)) {
        const meta = db.createObjectStore(META, { keyPath: 'key' });
        const cursor = r.transaction!.objectStore(STORE).openCursor();
        cursor.onsuccess = () => {
          const c = cursor.result;
          if (!c) return;
          const { key, name, version, bytes, savedAt } = c.value;
          meta.put({ key, name, version, bytes, savedAt }); c.continue();
        };
      }
    };
    r.onsuccess = () => res(r.result);
    r.onerror = () => rej(r.error);
  });
}

function tx<T>(mode: IDBTransactionMode, fn: (s: IDBObjectStore, meta: IDBObjectStore) => IDBRequest<T>): Promise<T> {
  return open().then((db) => new Promise<T>((res, rej) => {
    const t = db.transaction([STORE, META], mode);
    const req = fn(t.objectStore(STORE), t.objectStore(META));
    req.onerror = () => rej(req.error);
    t.oncomplete = () => { db.close(); res(req.result); };
    t.onabort = () => { db.close(); rej(t.error ?? new Error('Model cache transaction aborted')); };
  }));
}

export async function saveModel(key: string, name: string, version: string, files: ParaphernaliaFiles): Promise<void> {
  const images: Record<string, ArrayBuffer> = {};
  for (const [k, b] of Object.entries(files.images)) images[k] = await b.arrayBuffer();
  const bytes = files.phone_extractor.byteLength + files.pitch_estimator.byteLength + files.waveform_generator.byteLength + files.speaker_embeddings.byteLength + (files.embedding_setter?.byteLength ?? 0);
  const metadata = { key, name, version, bytes: bytes + (files.formant_shift_embeddings?.byteLength ?? 0), savedAt: Date.now() };
  await tx("readwrite", (s, meta) => { meta.put(metadata); return s.put({ ...metadata, files: { ...files, images } }); });
}

export async function listModels(): Promise<CachedModelMeta[]> {
  const all = await tx<CachedModelMeta[]>('readonly', (_s, meta) => meta.getAll());
  return all.sort((a, b) => b.savedAt - a.savedAt);
}

export async function loadModel(key: string): Promise<ParaphernaliaFiles | null> {
  const r = await tx<any>("readonly", (s) => s.get(key));
  if (!r) return null;
  const images: Record<string, Blob> = {};
  for (const [k, ab] of Object.entries(r.files.images as Record<string, ArrayBuffer>)) images[k] = new Blob([ab]);
  return { ...r.files, images } as ParaphernaliaFiles;
}

export async function deleteModel(key: string): Promise<void> {
  await tx("readwrite", (s, meta) => { meta.delete(key); return s.delete(key); });
}

export async function storageEstimate(): Promise<{ usage: number; quota: number } | null> {
  if (!navigator.storage?.estimate) return null;
  const e = await navigator.storage.estimate();
  return { usage: e.usage ?? 0, quota: e.quota ?? 0 };
}
