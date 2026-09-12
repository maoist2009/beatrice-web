/**
 * ONNX graph construction for the three Beatrice 2.0.0-rc.0 networks, ported from the official
 * torch source. Graphs use a FIXED time length so WebGPU shaders compile once.
 *
 * CausalConv1d(k, delay): torch uses padding = (k-1)*d - delay on BOTH sides and then trims
 * (k-1)*d - 2*delay from the tail → equivalent to asymmetric pads [ (k-1)*d - delay , delay ].
 */
import { GraphBuilder, IS } from "./onnx";
import { PITCH, PHONE, WG, type PitchWeights, type PhoneWeights, type WGWeights, type Lin } from "./layouts";

const causalPads = (k: number, delay: number): [number, number] => [k - 1 - delay, delay];

export function buildPitchEstimatorOnnx(w: PitchWeights, T: number): Uint8Array {
  const g = new GraphBuilder("beatrice_pitch_estimator");
  const c = PITCH.ch;
  const instfreq = g.input("instfreq", [1, PITCH.ifIn, T]);
  const corr = g.input("corr_diff", [1, PITCH.crIn, T]);
  g.output("logits", [1, PITCH.bins, T]);

  let xi = g.geluTanh(g.conv1d(instfreq, w.if0.w, w.if0.b, c, PITCH.ifIn, 1));
  xi = g.conv1d(xi, w.if1.w, w.if1.b, c, c, 1);
  let xc = g.geluTanh(g.conv1d(corr, w.cr0.w, w.cr0.b, c, PITCH.crIn, 1));
  xc = g.conv1d(xc, w.cr1.w, w.cr1.b, c, c, 1);
  let x = g.geluTanh(g.add(xi, xc));
  x = g.conv1d(x, w.embed.w, w.embed.b, c, c, PITCH.embedK, { pads: causalPads(PITCH.embedK, 1) });
  let xt = g.layerNorm(g.transpose(x, [0, 2, 1]), w.norm.w, w.norm.b, c);
  for (const b of w.blocks) {
    let h = g.transpose(xt, [0, 2, 1]);
    h = g.conv1d(h, b.dw.w, b.dw.b, c, 1, PITCH.dwK, { pads: causalPads(PITCH.dwK, 0), groups: c });
    h = g.layerNorm(g.transpose(h, [0, 2, 1]), null, null, c); // affine folded into pwconv1
    h = g.geluTanh(g.linear(h, b.pw1.w, b.pw1.b, PITCH.inter, c));
    h = g.linear(h, b.pw2.w, b.pw2.b, c, PITCH.inter);         // gamma & scales folded in
    xt = g.add(xt, h);
  }
  xt = g.layerNorm(xt, w.finalNorm.w, w.finalNorm.b, c);
  pointwiseOut(g, g.transpose(xt, [0, 2, 1]), w.head, PITCH.bins, c, "logits");
  return g.build();
}

export function buildPhoneExtractorOnnx(w: PhoneWeights, T: number): Uint8Array {
  if (T % 4 !== 0) throw new Error("phone window must be a multiple of 4");
  const g = new GraphBuilder("beatrice_phone_extractor");
  const C = PHONE.hidden, H = PHONE.heads, hd = C / H, T4 = T / 4;
  const wav = g.input("wav", [1, 1, T * 160]);
  g.output("units", [1, PHONE.phone, T]);

  let x = g.pad(wav, [0, 0, 40, 0, 0, 40]); // F.pad(x, (40, 40))
  let cin = 1;
  for (let i = 0; i < 6; i++) {
    x = g.geluTanh(g.conv1d(x, w.fe[i], null, PHONE.feCh[i], cin, PHONE.feK[i], { stride: PHONE.feS[i] }));
    cin = PHONE.feCh[i];
  }
  // FeatureProjection.norm: affine folded into backbone.embed by merge_weights → normalise only
  x = g.transpose(g.layerNorm(g.transpose(x, [0, 2, 1]), null, null, C), [0, 2, 1]);
  x = g.conv1d(x, w.embed.w, w.embed.b, C, C, PHONE.embedK, { pads: causalPads(PHONE.embedK, 0) });
  let xt = g.layerNorm(g.transpose(x, [0, 2, 1]), w.norm.w, w.norm.b, C);

  const mask = new Float32Array(T4 * T4);
  for (let q = 0; q < T4; q++) for (let k = q + 1; k < T4; k++) mask[q * T4 + k] = -1e9;
  const maskC = g.f32("causal_mask", mask, [T4, T4]);

  for (const b of w.blocks) {
    // strided split: frame t → subsequence t%4 at position t//4  (x.view(B,C,T/4,4).permute(0,3,2,1))
    const sub = g.transpose(g.reshape(xt, [T4, 4, C]), [1, 0, 2]); // [4, T4, C]
    let h = g.layerNorm(sub, null, null, C);                       // attn_norm folded into in_proj
    // q/k weights carry 1/sqrt(sqrt(hd)) each from dump_layer → the 1/sqrt(hd) scale is already applied
    const q = g.transpose(g.reshape(g.linear(h, b.wq, b.bq, C, C), [4, T4, H, hd]), [0, 2, 1, 3]);
    const k = g.transpose(g.reshape(g.linear(h, b.wk, b.bk, C, C), [4, T4, H, hd]), [0, 2, 3, 1]);
    const v = g.transpose(g.reshape(g.linear(h, b.wv, b.bv, C, C), [4, T4, H, hd]), [0, 2, 1, 3]);
    let o = g.matmul(g.softmax(g.add(g.matmul(q, k), maskC), -1), v);
    o = g.reshape(g.transpose(o, [0, 2, 1, 3]), [4, T4, C]);
    o = g.linear(o, b.out.w, b.out.b, C, C);
    xt = g.add(xt, g.reshape(g.transpose(o, [1, 0, 2]), [1, T, C]));

    h = g.transpose(xt, [0, 2, 1]);
    h = g.conv1d(h, b.dw.w, b.dw.b, C, 1, PHONE.dwK, { pads: causalPads(PHONE.dwK, 0), groups: C });
    h = g.layerNorm(g.transpose(h, [0, 2, 1]), null, null, C);
    h = g.geluTanh(g.linear(h, b.pw1.w, b.pw1.b, PHONE.inter, C));
    h = g.linear(h, b.pw2.w, b.pw2.b, C, PHONE.inter);
    xt = g.add(xt, h);
  }
  xt = g.geluTanh(g.layerNorm(xt, w.finalNorm.w, w.finalNorm.b, C)); // head is applied after a GELU
  pointwiseOut(g, g.transpose(xt, [0, 2, 1]), w.head, PHONE.phone, C, "units");
  return g.build();
}

/**
 * Vocoder graph. `x` (the summed + SiLU'd embeddings) and the speaker-dependent cross-attention
 * K/V are computed on the CPU and fed in, so the graph never has to be rebuilt when the speaker changes.
 *   inputs : x [1,256,T], kt{i} [1,4,32,384], v{i} [1,4,384,32]  for i in 0..3
 *   outputs: ir [1,512,T], aperiodicity [1,240,T], post_filter [1,512,T]
 */
export function buildVocoderOnnx(w: WGWeights, T: number): Uint8Array {
  const g = new GraphBuilder("beatrice_vocoder");
  const C = WG.hidden, inter = C * 2, A = WG.attnCh, H = WG.heads, hd = A / H;
  const xIn = g.input("x", [1, C, T]);
  g.output("ir", [1, WG.irLen, T]);
  g.output("aperiodicity", [1, WG.hop, T]);
  g.output("post_filter", [1, WG.irLen, T]);

  // ---- prenet (cross-attention to the speaker key/value embedding)
  let x = g.conv1d(xIn, w.pre.embed.w, w.pre.embed.b, C, C, WG.preEmbedK, { pads: causalPads(WG.preEmbedK, WG.preDelay) });
  let xt = g.layerNorm(g.transpose(x, [0, 2, 1]), w.pre.norm.w, w.pre.norm.b, C);
  w.pre.blocks.forEach((b, i) => {
    const kt = g.input(`kt${i}`, [1, H, hd, WG.kvLen]);
    const vv = g.input(`v${i}`, [1, H, WG.kvLen, hd]);
    let h = g.layerNorm(xt, null, null, C); // attn_norm folded into q_projection
    // q_projection carries 1/sqrt(sqrt(hd)) and the K side carries the other half → no extra scale
    const q = g.transpose(g.reshape(g.linear(h, b.qW, b.qB, A, C), [1, T, H, hd]), [0, 2, 1, 3]);
    let o = g.matmul(g.softmax(g.matmul(q, kt), -1), vv);           // [1,H,T,hd]
    o = g.reshape(g.transpose(o, [0, 2, 1, 3]), [1, T, A]);
    xt = g.add(xt, g.linear(o, b.outW, b.outB, C, A));
    h = g.transpose(xt, [0, 2, 1]);
    h = g.conv1d(h, b.dw.w, b.dw.b, C, 1, WG.dwK, { pads: causalPads(WG.dwK, 0), groups: C });
    h = g.layerNorm(g.transpose(h, [0, 2, 1]), null, null, C);
    h = g.geluTanh(g.linear(h, b.pw1.w, b.pw1.b, inter, C));
    h = g.linear(h, b.pw2.w, b.pw2.b, C, inter);
    xt = g.add(xt, h);
  });
  xt = g.layerNorm(xt, w.pre.finalNorm.w, w.pre.finalNorm.b, C);
  const pre = g.transpose(xt, [0, 2, 1]); // [1, C, T]

  // ---- weight-standardised generator stacks: embed → blocks (NO LayerNorm anywhere: norm = Identity)
  const wsStack = (stack: typeof w.irGen, input: string): string => {
    let y = g.conv1d(input, stack.embed.w, stack.embed.b, C, C, WG.genEmbedK, { pads: causalPads(WG.genEmbedK, 0) });
    for (const b of stack.blocks) {
      const id = y;
      let h = g.conv1d(y, b.dw.w, b.dw.b, C, 1, WG.dwK, { pads: causalPads(WG.dwK, 0), groups: C });
      let ht = g.transpose(h, [0, 2, 1]);
      ht = g.geluTanh(g.linear(ht, b.pw1.w, b.pw1.b, inter, C));
      ht = g.linear(ht, b.pw2.w, b.pw2.b, C, inter);
      y = g.add(id, g.transpose(ht, [0, 2, 1]));
    }
    return y;
  };
  const post = (y: string, lin: Lin, cout: number, name: string) => {
    const W = g.f32(g.uniq("pw"), lin.w, [cout, C, 1]);
    const ins = [g.silu(y), W];
    if (lin.b) ins.push(g.f32(g.uniq("pb"), lin.b, [cout]));
    g.named("Conv", ins, name, { kernel_shape: IS([1]), pads: IS([0, 0]) });
  };
  post(wsStack(w.irGen, pre), w.irPost, WG.irLen, "ir");                  // ir_scale folded in
  post(wsStack(w.apGen, pre), w.apPost, WG.hop, "aperiodicity");          // aperiodicity_scale folded in
  post(wsStack(w.pfGen, pre), w.pfPost, WG.irLen, "post_filter");         // post_filter_scale folded in
  return g.build();
}

function pointwiseOut(g: GraphBuilder, x: string, head: Lin, cout: number, cin: number, name: string) {
  const W = g.f32(g.uniq("head_w"), head.w, [cout, cin, 1]);
  const ins = [x, W];
  if (head.b) ins.push(g.f32(g.uniq("head_b"), head.b, [cout]));
  g.named("Conv", ins, name, { kernel_shape: IS([1]), pads: IS([0, 0]) });
}
