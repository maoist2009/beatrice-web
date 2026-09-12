/**
 * Byte layouts of the Beatrice 2.0.0-rc.0 paraphernalia `.bin` files.
 *
 * ALL LAYOUTS BELOW ARE READ DIRECTLY FROM THE OFFICIAL EXPORTER
 * (beatrice_trainer/__main__.py, MIT). The files are headerless little-endian float16
 * (the exporter calls merge_weights(); half(); dump()).
 *
 * dump_layer():  Linear/Conv1d/LayerNorm → weight, bias
 *                MultiheadAttention      → in_proj_weight reordered [heads,3,head_dim,C] with q,k
 *                                          pre-multiplied by 1/sqrt(sqrt(head_dim)); in_proj_bias
 *                                          reordered [heads,3,head_dim]; out_proj.weight; out_proj.bias
 *                Embedding/Parameter     → weight
 * ConvNeXtBlock.dump():  [mha] dwconv pwconv1 pwconv2     (attn_norm / norm / gamma / scales are
 *                                                          folded away by merge_weights and NOT written)
 * ConvNeXtStack.dump():  embed [norm] blocks [final_layer_norm]   ([] only when not weight-standardized)
 * CrossAttention.dump(): q_projection.weight×1/sqrt(sqrt(hd)), q_projection.bias×same, out_proj w, b
 * CrossAttention.dump_kv(): per head (K w, V w), then per head (K b, V b)
 * PhoneExtractor.dump(): feature_extractor(6 convs, bias=False) backbone head
 * PitchEstimator.dump(): instfreq_embed_0/1 corr_embed_0/1 backbone head
 * ConverterNetwork.dump(): embed_phone embed_quantized_pitch embed_pitch_features vocoder
 * Vocoder.dump(): prenet ir_generator ir_generator_post ir_window
 *                 aperiodicity_generator aperiodicity_generator_post
 *                 post_filter_generator post_filter_generator_post
 * ConverterNetwork.dump_speaker_embeddings(): vq.codebooks embed_speaker embed_formant_shift
 *                                             key_value_speaker_embedding
 */

export interface Lin { w: Float32Array; b: Float32Array | null }

class Reader {
  o = 0;
  constructor(private a: Float32Array, private what: string) {}
  take(n: number): Float32Array {
    if (this.o + n > this.a.length) throw new Error(`${this.what}: layout overrun at ${this.o}+${n} > ${this.a.length}`);
    const s = this.a.subarray(this.o, this.o + n); this.o += n; return s;
  }
  lin(cout: number, cin: number, bias = true): Lin { return { w: this.take(cout * cin), b: bias ? this.take(cout) : null }; }
  end() { if (this.o !== this.a.length) throw new Error(`${this.what}: ${this.a.length - this.o} float16 values left over (read ${this.o})`); }
}

// ============================ PitchEstimator ============================
// channels 192, intermediate 384, 9 blocks, embed CausalConv k=3 delay=1, dw k=33 delay=0,
// enable_scaling=True (all scales folded), bins 448.
export const PITCH = { ch: 192, inter: 384, bins: 448, nBlocks: 9, dwK: 33, embedK: 3, ifIn: 192, crIn: 256 } as const;

export interface PitchWeights {
  if0: Lin; if1: Lin; cr0: Lin; cr1: Lin;
  embed: Lin; norm: Lin; blocks: { dw: Lin; pw1: Lin; pw2: Lin }[]; finalNorm: Lin; head: Lin;
}

export function pitchExpectedHalves(): number {
  const { ch: c, inter, bins, nBlocks, dwK, embedK, ifIn, crIn } = PITCH;
  return (ifIn * c + c) + (c * c + c) + (crIn * c + c) + (c * c + c)
    + (c * c * embedK + c) + 2 * c
    + nBlocks * ((c * dwK + c) + (inter * c + inter) + (c * inter + c))
    + 2 * c + (bins * c + bins);
}

export function extractPitch(f32: Float32Array): PitchWeights {
  const { ch: c, inter, bins, nBlocks, dwK, embedK, ifIn, crIn } = PITCH;
  const r = new Reader(f32, "pitch_estimator.bin");
  const if0 = r.lin(c, ifIn), if1 = r.lin(c, c), cr0 = r.lin(c, crIn), cr1 = r.lin(c, c);
  const embed = r.lin(c, c * embedK);
  const norm = r.lin(c, 1);
  const blocks = [];
  for (let i = 0; i < nBlocks; i++) blocks.push({ dw: r.lin(c, dwK), pw1: r.lin(inter, c), pw2: r.lin(c, inter) });
  const finalNorm = r.lin(c, 1);
  const head = r.lin(bins, c);
  r.end();
  return { if0, if1, cr0, cr1, embed, norm, blocks, finalNorm, head };
}

// ============================ PhoneExtractor ============================
// FeatureExtractor(hidden=128): conv0 1→16 k10 s5, conv1 16→32 k3 s2, conv2 32→64 k3 s2,
// conv3 64→128 k3 s2, conv4 128→128 k3 s2, conv5 128→128 k2 s2 — all bias=False.
// backbone: ConvNeXtStack(128, 128, 384, 20 blocks, delay 0, embed k=9, dw k=17, use_mha=True), head 128→128.
export const PHONE = {
  hidden: 128, phone: 128, inter: 384, nBlocks: 20, heads: 4, embedK: 9, dwK: 17,
  feCh: [16, 32, 64, 128, 128, 128], feK: [10, 3, 3, 3, 3, 2], feS: [5, 2, 2, 2, 2, 2],
} as const;

export interface PhoneBlock {
  wq: Float32Array; wk: Float32Array; wv: Float32Array; bq: Float32Array; bk: Float32Array; bv: Float32Array;
  out: Lin; dw: Lin; pw1: Lin; pw2: Lin;
}
export interface PhoneWeights { fe: Float32Array[]; embed: Lin; norm: Lin; blocks: PhoneBlock[]; finalNorm: Lin; head: Lin }

export function phoneExpectedHalves(): number {
  const { hidden: h, phone, inter, nBlocks, embedK, dwK, feCh, feK } = PHONE;
  let n = 0, cin = 1;
  for (let i = 0; i < 6; i++) { n += feCh[i] * cin * feK[i]; cin = feCh[i]; }
  n += h * h * embedK + h + 2 * h;
  n += nBlocks * (3 * h * h + 3 * h + h * h + h + (h * dwK + h) + (inter * h + inter) + (h * inter + h));
  return n + 2 * h + (phone * h + phone);
}

export function extractPhone(f32: Float32Array): PhoneWeights {
  const { hidden: h, phone, inter, nBlocks, heads, embedK, dwK, feCh, feK } = PHONE;
  const hd = h / heads;
  const r = new Reader(f32, "phone_extractor.bin");
  const fe: Float32Array[] = [];
  let cin = 1;
  for (let i = 0; i < 6; i++) { fe.push(r.take(feCh[i] * cin * feK[i])); cin = feCh[i]; }
  const embed = r.lin(h, h * embedK);
  const norm = r.lin(h, 1);
  const blocks: PhoneBlock[] = [];
  for (let i = 0; i < nBlocks; i++) {
    const inW = r.take(3 * h * h);   // [heads, 3, hd, C]
    const inB = r.take(3 * h);       // [heads, 3, hd]
    const wq = new Float32Array(h * h), wk = new Float32Array(h * h), wv = new Float32Array(h * h);
    const bq = new Float32Array(h), bk = new Float32Array(h), bv = new Float32Array(h);
    const W = [wq, wk, wv], B = [bq, bk, bv];
    for (let hh = 0; hh < heads; hh++)
      for (let p = 0; p < 3; p++)
        for (let d = 0; d < hd; d++) {
          const row = hh * hd + d, src = ((hh * 3 + p) * hd + d) * h;
          W[p].set(inW.subarray(src, src + h), row * h);
          B[p][row] = inB[(hh * 3 + p) * hd + d];
        }
    blocks.push({ wq, wk, wv, bq, bk, bv, out: r.lin(h, h), dw: r.lin(h, dwK), pw1: r.lin(inter, h), pw2: r.lin(h, inter) });
  }
  const finalNorm = r.lin(h, 1);
  const head = r.lin(phone, h);
  r.end();
  return { fe, embed, norm, blocks, finalNorm, head };
}

// ============================ WaveformGenerator (ConverterNetwork tail) ============================
export const WG = {
  hidden: 256, phone: 128, bins: 448, hop: 240, irLen: 512, kvLen: 384, kvCh: 128,
  attnCh: 128, heads: 4, preBlocks: 4, irBlocks: 2, apBlocks: 1, pfBlocks: 1,
  preEmbedK: 7, preDelay: 2, genEmbedK: 3, dwK: 33,
} as const;

export interface WSBlock { dw: Lin; pw1: Lin; pw2: Lin }
export interface PreBlock { qW: Float32Array; qB: Float32Array; outW: Float32Array; outB: Float32Array; dw: Lin; pw1: Lin; pw2: Lin }
export interface WSStack { embed: Lin; blocks: WSBlock[] }
export interface WGWeights {
  embedPhone: Lin; embedQPitch: Float32Array; embedPitchFeat: Lin;
  pre: { embed: Lin; norm: Lin; blocks: PreBlock[]; finalNorm: Lin };
  irGen: WSStack; irPost: Lin; irWindow: Float32Array;
  apGen: WSStack; apPost: Lin;
  pfGen: WSStack; pfPost: Lin;
}

function wsStackHalves(nBlocks: number, cin: number): number {
  const { hidden: h, genEmbedK, dwK } = WG;
  const inter = h * 2;
  return (h * cin * genEmbedK + h) + nBlocks * ((h * dwK + h) + (inter * h + inter) + (h * inter + h));
}

export function wgExpectedHalves(): number {
  const { hidden: h, phone, bins, hop, irLen, attnCh, preBlocks, preEmbedK, dwK, irBlocks, apBlocks, pfBlocks } = WG;
  const inter = h * 2;
  let n = (h * phone + h) + (bins * h) + (h * 4 + h);
  // prenet (not weight-standardised → norm + final_layer_norm are dumped)
  n += (h * h * preEmbedK + h) + 2 * h;
  n += preBlocks * ((attnCh * h + attnCh) + (h * attnCh + h) + (h * dwK + h) + (inter * h + inter) + (h * inter + h));
  n += 2 * h;
  n += wsStackHalves(irBlocks, h) + (irLen * h + irLen) + irLen;
  n += wsStackHalves(apBlocks, h) + hop * h;
  n += wsStackHalves(pfBlocks, h) + irLen * h;
  return n;
}

export function extractWaveformGenerator(f32: Float32Array): WGWeights {
  const { hidden: h, phone, bins, hop, irLen, attnCh, preBlocks, preEmbedK, genEmbedK, dwK, irBlocks, apBlocks, pfBlocks } = WG;
  const inter = h * 2;
  const r = new Reader(f32, "waveform_generator.bin");
  const embedPhone = r.lin(h, phone);
  const embedQPitch = r.take(bins * h);
  const embedPitchFeat = r.lin(h, 4);
  // --- prenet
  const preEmbed = r.lin(h, h * preEmbedK);
  const preNorm = r.lin(h, 1);
  const preB: PreBlock[] = [];
  for (let i = 0; i < preBlocks; i++) {
    const qW = r.take(attnCh * h), qB = r.take(attnCh);
    const outW = r.take(h * attnCh), outB = r.take(h);
    preB.push({ qW, qB, outW, outB, dw: r.lin(h, dwK), pw1: r.lin(inter, h), pw2: r.lin(h, inter) });
  }
  const preFinal = r.lin(h, 1);
  const ws = (nBlocks: number, cin: number): WSStack => {
    const embed = r.lin(h, cin * genEmbedK);
    const blocks: WSBlock[] = [];
    for (let i = 0; i < nBlocks; i++) blocks.push({ dw: r.lin(h, dwK), pw1: r.lin(inter, h), pw2: r.lin(h, inter) });
    return { embed, blocks };
  };
  const irGen = ws(irBlocks, h);
  const irPost = r.lin(irLen, h);
  const irWindow = r.take(irLen);
  const apGen = ws(apBlocks, h);
  const apPost = r.lin(hop, h, false);
  const pfGen = ws(pfBlocks, h);
  const pfPost = r.lin(irLen, h, false);
  r.end();
  return { embedPhone, embedQPitch, embedPitchFeat, pre: { embed: preEmbed, norm: preNorm, blocks: preB, finalNorm: preFinal }, irGen, irPost, irWindow, apGen, apPost, pfGen, pfPost };
}
