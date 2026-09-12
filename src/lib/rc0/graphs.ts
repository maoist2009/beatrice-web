/**
 * Incremental (stateful) Beatrice 2.0.0-rc.0 graphs.
 *
 * Every frame passes through every layer exactly once — like the official inference library and unlike
 * the earlier sliding-window port, which recomputed ctx+chunk frames per chunk (up to 14× wasted work).
 *
 *  - Causal convolutions carry (k-1-delay) frames of history as explicit state (shared helpers).
 *  - PhoneExtractor self-attention (4 strided subsequences, causal) carries a bounded K/V cache of P
 *    positions per subsequence and block plus an additive validity mask, so cold cache slots contribute
 *    exactly zero probability. P is this port's selectable bound; native library state parity is unverified.
 *  - Vocoder prenet cross-attention reads a fixed speaker K/V, so it needs no temporal state.
 *
 * The chunk must be a multiple of 4 (frame t → subsequence t%4). Attention scale is already baked into the
 * dumped q/k weights (1/√√hd each), so no extra scaling is applied.
 */
import { GraphBuilder, I } from '../onnx';
import { conv, stack, remember, finish } from '../beta/graphs';
import type { Stack } from '../beta/layouts';
import type { StateSpec, StreamingGraph } from '../session';
import { PITCH, PHONE, WG, type PitchWeights, type PhoneWeights, type PhoneBlock, type WGWeights, type Lin } from '../layouts';

const MASK = -1e9;

const asStack = (embed: Lin, norm: Lin | null, blocks: { dw: Lin; pw1: Lin; pw2: Lin }[], final: Lin | null,
  channels: number, inter: number, kernel: number, delay: number, dwKernel: number): Stack =>
  ({ embed, norm, blocks, final, channels, inter, kernel, delay, dwKernel });

/** ConvNeXt conv branch on channels-last [1,C,ch] with explicit dw-conv history. */
function convBranch(g: GraphBuilder, states: StateSpec[], xt: string, b: { dw: Lin; pw1: Lin; pw2: Lin }, ch: number, inter: number, k: number, chunk: number, name: string, norm: boolean) {
  let y = conv(g, states, g.transpose(xt, [0, 2, 1]), b.dw, ch, k, chunk, name, 0, true);
  y = g.transpose(y, [0, 2, 1]);
  if (norm) y = g.layerNorm(y, null, null, ch);
  y = g.geluTanh(g.linear(y, b.pw1.w, b.pw1.b, inter, ch));
  y = g.linear(y, b.pw2.w, b.pw2.b, ch, inter);
  return g.add(xt, y);
}

// ================================================================ PitchEstimator
export function buildRc0Pitch(w: PitchWeights, chunk: number): StreamingGraph {
  const g = new GraphBuilder('beatrice_rc0_pitch'), states: StateSpec[] = [];
  const c = PITCH.ch;
  const inst = g.input('instfreq', [1, PITCH.ifIn, chunk + 1]);
  const corr = g.input('corr_diff', [1, PITCH.crIn, chunk + 1]);
  const valid = g.input('valid', [1, 1, chunk + 1]);
  const branch = (x: string, a: Lin, b: Lin, cin: number) => g.conv1d(g.geluTanh(g.conv1d(x, a.w, a.b, c, cin, 1)), b.w, b.b, c, c, 1);
  // rc.0 fixed the missing activation: GELU after the sum.
  const x = g.mul(g.geluTanh(g.add(branch(inst, w.if0, w.if1, PITCH.ifIn), branch(corr, w.cr0, w.cr1, PITCH.crIn))), valid);
  const y = stack(g, states, x, asStack(w.embed, w.norm, w.blocks, w.finalNorm, c, PITCH.inter, PITCH.embedK, 1, PITCH.dwK), chunk, 'pitch');
  const logits = g.conv1d(y, w.head.w, w.head.b, PITCH.bins, c, 1);
  g.named('Identity', [logits], 'logits'); g.output('logits', [1, PITCH.bins, chunk]);
  return finish(g, states);
}

// ================================================================ PhoneExtractor
function streamingMha(g: GraphBuilder, states: StateSpec[], xt: string, b: PhoneBlock, i: number, chunk: number, P: number) {
  const H = PHONE.heads, Ch = PHONE.hidden, hd = Ch / H, Cs = chunk / 4;
  // frame 4r+s → subsequence s, position r   ([1,C,Ch] → [Cs,4,Ch] → [4,Cs,Ch])
  const sub = g.transpose(g.reshape(xt, [Cs, 4, Ch]), [1, 0, 2]);
  const n = g.layerNorm(sub, null, null, Ch); // attn_norm affine folded into in_proj
  const proj = (wt: Float32Array, bs: Float32Array) => g.transpose(g.reshape(g.linear(n, wt, bs, Ch, Ch), [4, Cs, H, hd]), [0, 2, 1, 3]); // [4,H,Cs,hd]
  const q = proj(b.wq, b.bq), k = proj(b.wk, b.bk), v = proj(b.wv, b.bv);

  const kc = g.input(`p${i}_k`, [4, H, P, hd]);
  const vc = g.input(`p${i}_v`, [4, H, P, hd]);
  const mc = g.input(`p${i}_m`, [4, 1, 1, P]);
  const kf = g.node('Concat', [kc, k], { axis: I(2) })[0];               // [4,H,P+Cs,hd]
  const vf = g.node('Concat', [vc, v], { axis: I(2) })[0];
  const mf = g.node('Concat', [mc, g.f32(g.uniq('mzero'), new Float32Array(4 * Cs), [4, 1, 1, Cs])], { axis: I(3) })[0]; // [4,1,1,P+Cs]
  remember(g, states, `p${i}_k`, g.slice(kf, [Cs], [P + Cs], [2]), [4, H, P, hd]);
  remember(g, states, `p${i}_v`, g.slice(vf, [Cs], [P + Cs], [2]), [4, H, P, hd]);
  remember(g, states, `p${i}_m`, g.slice(mf, [Cs], [P + Cs], [3]), [4, 1, 1, P], MASK);

  // causal part over the new keys: query r may see new key r' iff r' <= r
  const causal = new Float32Array(Cs * (P + Cs));
  for (let r = 0; r < Cs; r++) for (let j = P; j < P + Cs; j++) if (j - P > r) causal[r * (P + Cs) + j] = MASK;
  const causalC = g.f32(g.uniq('causal'), causal, [1, 1, Cs, P + Cs]);

  let s = g.add(g.add(g.matmul(q, g.transpose(kf, [0, 1, 3, 2])), mf), causalC); // [4,H,Cs,P+Cs]
  s = g.softmax(s, -1);
  let o = g.matmul(s, vf);                                                     // [4,H,Cs,hd]
  o = g.reshape(g.transpose(o, [0, 2, 1, 3]), [4, Cs, Ch]);
  o = g.linear(o, b.out.w, b.out.b, Ch, Ch);
  const back = g.reshape(g.transpose(o, [1, 0, 2]), [1, chunk, Ch]);
  return g.add(xt, back);
}

export function buildRc0Phone(w: PhoneWeights, chunk: number, attnPositions: number): StreamingGraph {
  if (chunk % 4 !== 0) throw new Error('rc.0 phone chunk must be a multiple of 4');
  if (attnPositions < chunk / 4) throw new Error('attention memory must cover at least one chunk');
  const g = new GraphBuilder('beatrice_rc0_phone'), states: StateSpec[] = [];
  const C = PHONE.hidden;
  let x = g.input('wav', [1, 1, chunk * 160 + 80]); // caller supplies [e*160-40, e*160+chunk*160+40)
  let cin = 1;
  for (let i = 0; i < 6; i++) {
    x = g.geluTanh(g.conv1d(x, w.fe[i], null, PHONE.feCh[i], cin, PHONE.feK[i], { stride: PHONE.feS[i] }));
    cin = PHONE.feCh[i];
  }
  // FeatureProjection norm: affine folded into backbone.embed → normalise only
  x = g.transpose(g.layerNorm(g.transpose(x, [0, 2, 1]), null, null, C), [0, 2, 1]);
  x = conv(g, states, x, w.embed, C, PHONE.embedK, chunk, 'phone_embed', 0);
  let xt = g.layerNorm(g.transpose(x, [0, 2, 1]), w.norm.w, w.norm.b, C);
  w.blocks.forEach((b, i) => {
    xt = streamingMha(g, states, xt, b, i, chunk, attnPositions);
    xt = convBranch(g, states, xt, b, C, PHONE.inter, PHONE.dwK, chunk, `phone_dw${i}`, true);
  });
  xt = g.geluTanh(g.layerNorm(xt, w.finalNorm.w, w.finalNorm.b, C)); // head after GELU
  const units = g.conv1d(g.transpose(xt, [0, 2, 1]), w.head.w, w.head.b, PHONE.phone, C, 1);
  g.named('Identity', [units], 'units'); g.output('units', [1, PHONE.phone, chunk]);
  return finish(g, states);
}

// ================================================================ Vocoder
/**
 * inputs : x [1,256,chunk+2]  (SiLU'd embedding sum, 2 frames of prenet look-ahead)
 *          kt{i} [1,H,hd,384], v{i} [1,H,384,hd]  fixed speaker key/value per prenet block
 * outputs: ir [1,512,chunk], aperiodicity [1,240,chunk], post_filter [1,512,chunk]
 */
export function buildRc0Vocoder(w: WGWeights, chunk: number): StreamingGraph {
  const g = new GraphBuilder('beatrice_rc0_vocoder'), states: StateSpec[] = [];
  const C = WG.hidden, inter = C * 2, A = WG.attnCh, H = WG.heads, hd = A / H;
  const xIn = g.input('x', [1, C, chunk + 2]);
  let x = conv(g, states, xIn, w.pre.embed, C, WG.preEmbedK, chunk, 'pre_embed', WG.preDelay);
  let xt = g.layerNorm(g.transpose(x, [0, 2, 1]), w.pre.norm.w, w.pre.norm.b, C);
  w.pre.blocks.forEach((b, i) => {
    const kt = g.input(`kt${i}`, [1, H, hd, WG.kvLen]);
    const vv = g.input(`v${i}`, [1, H, WG.kvLen, hd]);
    const n = g.layerNorm(xt, null, null, C);
    const q = g.transpose(g.reshape(g.linear(n, b.qW, b.qB, A, C), [1, chunk, H, hd]), [0, 2, 1, 3]);
    let o = g.matmul(g.softmax(g.matmul(q, kt), -1), vv);           // [1,H,chunk,hd]
    o = g.reshape(g.transpose(o, [0, 2, 1, 3]), [1, chunk, A]);
    xt = g.add(xt, g.linear(o, b.outW, b.outB, C, A));
    xt = convBranch(g, states, xt, b, C, inter, WG.dwK, chunk, `pre_dw${i}`, true);
  });
  xt = g.layerNorm(xt, w.pre.finalNorm.w, w.pre.finalNorm.b, C);
  const pre = g.transpose(xt, [0, 2, 1]);
  const gen = (s: { embed: Lin; blocks: { dw: Lin; pw1: Lin; pw2: Lin }[] }, post: Lin, cout: number, name: string) => {
    const y = stack(g, states, pre, asStack(s.embed, null, s.blocks, null, C, inter, WG.genEmbedK, 0, WG.dwK), chunk, name);
    const value = g.conv1d(g.silu(y), post.w, post.b, cout, C, 1);
    g.named('Identity', [value], name); g.output(name, [1, cout, chunk]);
  };
  gen(w.irGen, w.irPost, WG.irLen, 'ir');
  gen(w.apGen, w.apPost, WG.hop, 'aperiodicity');
  gen(w.pfGen, w.pfPost, WG.irLen, 'post_filter');
  return finish(g, states);
}
