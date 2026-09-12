import { readFileSync, writeFileSync, readdirSync, mkdirSync } from 'node:fs';
import { readBetaPhone, readBetaPitch, readBetaWave } from '../src/lib/beta/layouts';
import { buildBetaPhone, buildBetaPitch, buildBetaVocoder } from '../src/lib/beta/graphs';
import { parseParaphernalia } from '../src/lib/paraphernalia';

const output = '.research/graphs';
mkdirSync(output, { recursive: true });
function read(path: string) { const b = readFileSync(path); return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer; }
for (const model of ['shigure', 'kurage']) {
  const dir = `.research/${model}`, toml = readdirSync(dir).find(p => p.endsWith('.toml'))!;
  const files = { toml: readFileSync(`${dir}/${toml}`, 'utf8'), tomlName: toml, phone_extractor: read(`${dir}/phone_extractor.bin`), pitch_estimator: read(`${dir}/pitch_estimator.bin`), waveform_generator: read(`${dir}/waveform_generator.bin`), speaker_embeddings: read(`${dir}/speaker_embeddings.bin`), formant_shift_embeddings: read(`${dir}/formant_shift_embeddings.bin`), embedding_setter: null, images: {} };
  const parsed = parseParaphernalia(files);
  console.log('[actual TS loader]', model, parsed.format, parsed.version, parsed.speakers.nSpeakers);
  const phone = readBetaPhone(files.phone_extractor), pitch = readBetaPitch(files.pitch_estimator), wave = readBetaWave(files.waveform_generator);
  for (const chunk of [1, 4, 20]) {
    for (const [name, build] of [
      ['phone', () => buildBetaPhone(phone, chunk, 'onnx-gru')],
      ['phone-primitives', () => buildBetaPhone(phone, chunk, 'primitives')],
      ['pitch', () => buildBetaPitch(pitch, chunk)],
      ['vocoder', () => buildBetaVocoder(wave, chunk)],
    ] as const) {
      const graph = build(), prefix = `${output}/${model}-${name}-${chunk}`;
      writeFileSync(`${prefix}.onnx`, graph.bytes);
      writeFileSync(`${prefix}.json`, JSON.stringify(graph.states));
      console.log('[graph]', model, name, chunk, graph.bytes.length);
    }
  }
}