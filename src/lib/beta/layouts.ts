// Source: fierce-cats/beatrice-trainer @ 5ddb63ea854832a914a846a22830feb9ff7e691b, MIT.
// This reads the fused FP16 export, not a PyTorch checkpoint or an rc.0 layout.
import { f16ToF32, type ParaphernaliaFiles } from '../paraphernalia';
import type { Lin } from '../layouts';

export const BETA = { phone: 256, hidden: 256, pitch: 384, band: 48, pitchHidden: 192, hop: 160, outHop: 240 } as const;
export interface Block { dw: Lin; pw1: Lin; pw2: Lin }
export interface Stack { embed: Lin; norm: Lin | null; blocks: Block[]; final: Lin | null; channels: number; inter: number; kernel: number; delay: number; dwKernel: number }
export interface GruLayer { wi: Float32Array; bi: Float32Array; wh: Float32Array; bh: Float32Array }
export interface BetaPhone { fe: Float32Array[]; projection: Lin; gru: GruLayer[]; backbone: Stack; head: Lin }
export interface BetaPitch { if0: Lin; if1: Lin; cr0: Lin; cr1: Lin; backbone: Stack; head: Lin }
export interface BetaWave {
  phone: Lin; pitch: Float32Array; features: Lin; pre: Stack;
  ir: Stack; irPost: Lin; window: Float32Array; ap: Stack; apPost: Lin; pf: Stack; pfPost: Lin;
}

class Reader {
  private a: Float32Array;
  private offset = 0;
  constructor(buf: ArrayBuffer, private name: string, expected: number) {
    if (buf.byteLength !== expected) throw new Error(`${name}: beta.2 needs ${expected} bytes, received ${buf.byteLength}`);
    this.a = f16ToF32(buf);
    for (const value of this.a) if (!Number.isFinite(value)) throw new Error(`${name}: non-finite FP16 weight`);
  }
  take(n: number) {
    const o = this.offset;
    if (o + n > this.a.length) throw new Error(`${this.name}: weight layout overrun at ${o}`);
    this.offset += n;
    return this.a.subarray(o, o + n);
  }
  lin(out: number, input: number, bias = true): Lin { return { w: this.take(out * input), b: bias ? this.take(out) : null }; }
  stack(c: number, inter: number, n: number, kernel: number, delay: number, dwKernel: number, ws = false): Stack {
    const embed = this.lin(c, c * kernel), norm = ws ? null : this.lin(c, 1);
    const blocks = Array.from({ length: n }, () => ({ dw: this.lin(c, dwKernel), pw1: this.lin(inter, c), pw2: this.lin(c, inter) }));
    return { embed, norm, blocks, final: ws ? null : this.lin(c, 1), channels: c, inter, kernel, delay, dwKernel };
  }
  end() { if (this.offset !== this.a.length) throw new Error(`${this.name}: ${this.a.length - this.offset} unread weights`); }
}

export function readBetaPhone(buf: ArrayBuffer): BetaPhone {
  const r = new Reader(buf, 'phone_extractor.bin', 10847360);
  const ch = [1, 32, 64, 128, 256, 256, 256], k = [10, 3, 3, 3, 3, 2];
  const fe = k.map((kernel, i) => r.take(ch[i] * ch[i + 1] * kernel));
  const projection = r.lin(256, 256);
  const gru = Array.from({ length: 3 }, () => ({ wi: r.take(3 * 256 * 256), bi: r.take(3 * 256), wh: r.take(3 * 256 * 256), bh: r.take(3 * 256) }));
  const backbone = r.stack(256, 768, 8, 7, 0, 17);
  const head = r.lin(256, 256);
  r.end();
  return { fe, projection, gru, backbone, head };
}

export function readBetaPitch(buf: ArrayBuffer): BetaPitch {
  const r = new Reader(buf, 'pitch_estimator.bin', 3434112);
  const if0 = r.lin(192, 192), if1 = r.lin(192, 192), cr0 = r.lin(192, 256), cr1 = r.lin(192, 192);
  const backbone = r.stack(192, 576, 6, 3, 1, 33);
  const head = r.lin(384, 192);
  r.end();
  return { if0, if1, cr0, cr1, backbone, head };
}

export function readBetaWave(buf: ArrayBuffer): BetaWave {
  const r = new Reader(buf, 'waveform_generator.bin', 9528320);
  const phone = r.lin(256, 256), pitch = r.take(384 * 256), features = r.lin(256, 4);
  const pre = r.stack(256, 768, 4, 7, 2, 33);
  const ir = r.stack(256, 768, 2, 3, 0, 33, true), irPost = r.lin(512, 256), window = r.take(512);
  const ap = r.stack(256, 768, 1, 3, 0, 33, true), apPost = r.lin(240, 256, false);
  const pf = r.stack(256, 768, 1, 3, 0, 33, true), pfPost = r.lin(512, 256, false);
  r.end();
  return { phone, pitch, features, pre, ir, irPost, window, ap, apPost, pf, pfPost };
}

export function readBetaWeights(files: ParaphernaliaFiles) {
  return { phone: readBetaPhone(files.phone_extractor), pitch: readBetaPitch(files.pitch_estimator), wave: readBetaWave(files.waveform_generator) };
}