/** 16-bit PCM RIFF/WAVE encoder (mono, float32 [-1,1] → s16le). */
export function encodeWav(samples: Float32Array, sampleRate: number): Blob {
  if (!Number.isInteger(sampleRate) || sampleRate <= 0 || !samples.length) throw new Error('Invalid WAV dimensions');
  const n = samples.length;
  const buf = new ArrayBuffer(44 + n * 2);
  const v = new DataView(buf);
  const u = new Uint8Array(buf);
  const wstr = (off: number, s: string) => { for (let i = 0; i < s.length; i++) u[off + i] = s.charCodeAt(i); };
  wstr(0, "RIFF");
  v.setUint32(4, 36 + n * 2, true);
  wstr(8, "WAVE");
  wstr(12, "fmt ");
  v.setUint32(16, 16, true);
  v.setUint16(20, 1, true);           // PCM
  v.setUint16(22, 1, true);           // mono
  v.setUint32(24, sampleRate, true);
  v.setUint32(28, sampleRate * 2, true);
  v.setUint16(32, 2, true);
  v.setUint16(34, 16, true);
  wstr(36, "data");
  v.setUint32(40, n * 2, true);
  for (let i = 0; i < n; i++) {
    if (!Number.isFinite(samples[i])) throw new Error(`Non-finite output sample ${i}; refusing to encode it as silence`);
    const s = Math.max(-1, Math.min(1, samples[i]));
    v.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Blob([buf], { type: "audio/wav" });
}

/** Decode an audio file (wav/mp3/ogg/opus/…) to mono float32 via the browser's decoder. */
export async function decodeAudioFile(data: ArrayBuffer): Promise<{ pcm: Float32Array; sampleRate: number; duration: number; channels: number }> {
  if (data.byteLength > 48 * 1024 * 1024) throw new Error('Please use audio files smaller than 48 MiB for browser testing');
  const ctx = new OfflineAudioContext(1, 1, 44100);
  const buf = await ctx.decodeAudioData(data);
  if (buf.duration > 60) throw new Error('Please use audio clips of 60 seconds or less while testing');
  const n = buf.length;
  const pcm = new Float32Array(n);
  for (let c = 0; c < buf.numberOfChannels; c++) {
    const ch = buf.getChannelData(c);
    for (let i = 0; i < n; i++) pcm[i] += ch[i];
  }
  const inv = 1 / buf.numberOfChannels;
  for (let i = 0; i < n; i++) pcm[i] *= inv;
  return { pcm, sampleRate: buf.sampleRate, duration: buf.duration, channels: buf.numberOfChannels };
}

/** Browser resampling, not the native VST resampler. The 16 kHz model boundary is tested separately. */
export async function offlineResample(pcm: Float32Array, from: number, to: number): Promise<Float32Array> {
  if (from === to) return pcm.slice();
  const len = Math.max(16, Math.ceil((pcm.length * to) / from));
  const ctx = new OfflineAudioContext(1, len, to);
  const b = ctx.createBuffer(1, pcm.length, from);
  b.getChannelData(0).set(pcm);
  const s = ctx.createBufferSource();
  s.buffer = b; s.connect(ctx.destination); s.start(0);
  const out = await ctx.startRendering();
  return out.getChannelData(0);
}
