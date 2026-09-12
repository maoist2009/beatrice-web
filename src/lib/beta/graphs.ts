// Incremental beta.2 graphs. All temporal convolution and GRU states cross the run boundary explicitly.
import { GraphBuilder, I } from '../onnx';
import type { Lin } from '../layouts';
import type { BetaPhone, BetaPitch, BetaWave, GruLayer, Stack } from './layouts';
import type { StateSpec, StreamingGraph } from '../session';
export type { StateSpec, StreamingGraph } from '../session';

export function finish(g: GraphBuilder, states: StateSpec[]): StreamingGraph {
  return { bytes: g.build(), states, inputs: g.inputSpecs, outputs: g.outputSpecs };
}

export function remember(g: GraphBuilder, states: StateSpec[], input: string, value: string, dims: number[], init = 0) {
  const output = `${input}_next`;
  g.named('Identity', [value], output);
  g.output(output, dims);
  states.push({ input, output, dims, init });
}

// Only consumed frames enter the cache. The right-hand lookahead is sent again on the next invocation.
export function conv(g: GraphBuilder, states: StateSpec[], x: string, w: Lin, c: number, k: number, chunk: number, name: string, delay = 0, depthwise = false) {
  const left = k - 1 - delay;
  const history = g.input(name, [1, c, left]);
  const joined = g.node('Concat', [history, x], { axis: I(2) })[0];
  remember(g, states, name, g.slice(joined, [chunk], [chunk + left], [2]), [1, c, left]);
  return g.conv1d(joined, w.w, w.b, c, depthwise ? 1 : c, k, { groups: depthwise ? c : 1 });
}

export function stack(g: GraphBuilder, states: StateSpec[], input: string, w: Stack, chunk: number, prefix: string) {
  const c = w.channels;
  let x = conv(g, states, input, w.embed, c, w.kernel, chunk, `${prefix}_embed`, w.delay);
  if (w.norm) x = g.transpose(g.layerNorm(g.transpose(x, [0, 2, 1]), w.norm.w, w.norm.b, c), [0, 2, 1]);
  w.blocks.forEach((b, i) => {
    const identity = x;
    let y = conv(g, states, x, b.dw, c, w.dwKernel, chunk, `${prefix}_dw${i}`, 0, true);
    y = g.transpose(y, [0, 2, 1]);
    if (w.norm) y = g.layerNorm(y, null, null, c);
    y = g.geluTanh(g.linear(y, b.pw1.w, b.pw1.b, w.inter, c));
    y = g.linear(y, b.pw2.w, b.pw2.b, c, w.inter);
    x = g.add(identity, g.transpose(y, [0, 2, 1]));
  });
  if (w.final) x = g.transpose(g.layerNorm(g.transpose(x, [0, 2, 1]), w.final.w, w.final.b, c), [0, 2, 1]);
  return x;
}

// PyTorch gate rows are [reset, update, new]; ONNX requires [update, reset, hidden].
export function onnxGruWeights(w: GruLayer, c = 256) {
  function reorder(src: Float32Array, width: number) {
    const out = new Float32Array(src.length);
    [1, 0, 2].forEach((gate, to) => out.set(src.subarray(gate * c * width, (gate + 1) * c * width), to * c * width));
    return out;
  }
  const bias = new Float32Array(6 * c);
  bias.set(reorder(w.bi, 1)); bias.set(reorder(w.bh, 1), 3 * c);
  return { wi: reorder(w.wi, c), wh: reorder(w.wh, c), bias };
}

export type GruMode = 'onnx-gru' | 'primitives';

/**
 * GRU expressed with MatMul/Sigmoid/Tanh/Mul/Add, which DO have WebGPU kernels (the ORT Web WebGPU
 * operator table ships no GRU/RNN kernel, so a native GRU node forces a WASM fallback for the whole
 * graph). Semantics match PyTorch's CPU GRU cell, i.e. ONNX linear_before_reset=1:
 *   n = tanh(Win·x + bin + r·(Whn·h + bhn))
 * which is what the shipped beta.2 weights were exported from.
 */
function gruPrimitives(g: GraphBuilder, states: StateSpec[], sequence: string, layer: GruLayer, chunk: number, index: number) {
  const c = 256;
  // PyTorch stores [3c, c] (out × in); MatMul needs x·Wᵀ, so the data must be transposed to [c, 3c].
  const t = (m: Float32Array) => {
    const out = new Float32Array(c * 3 * c);
    for (let o = 0; o < 3 * c; o++) for (let i = 0; i < c; i++) out[i * 3 * c + o] = m[o * c + i];
    return out;
  };
  const weights = onnxGruWeights(layer, c);
  const giAll = g.add(g.matmul(sequence, g.f32(`grup${index}_wi`, t(weights.wi), [c, 3 * c])), g.f32(`grup${index}_bi`, weights.bias.subarray(0, 3 * c), [3 * c]));
  let h = g.input(`gru${index}`, [1, c]);
  const wh = g.f32(`grup${index}_wh`, t(weights.wh), [c, 3 * c]);
  const bh = g.f32(`grup${index}_bh`, weights.bias.subarray(3 * c), [3 * c]);
  const outs: string[] = [];
  // gi and gh are [1, 3c]; gate k occupies the last-axis range [k*c, (k+1)*c) in z,r,n order.
  const chan = (x: string, k: number) => g.slice(x, [k * c], [(k + 1) * c], [-1]);
  for (let t = 0; t < chunk; t++) {
    const gi = g.reshape(g.slice(giAll, [t], [t + 1], [0]), [1, 3 * c]);
    const gh = g.add(g.matmul(h, wh), bh);
    const z = g.node('Sigmoid', [g.add(chan(gi, 0), chan(gh, 0))])[0];
    const r = g.node('Sigmoid', [g.add(chan(gi, 1), chan(gh, 1))])[0];
    const n = g.node('Tanh', [g.add(chan(gi, 2), g.mul(r, chan(gh, 2)))])[0];
    const one = g.scalar(1);
    h = g.add(g.mul(g.node('Sub', [one, z])[0], n), g.mul(z, h));
    outs.push(g.reshape(h, [1, 1, c]));
  }
  remember(g, states, `gru${index}`, h, [1, c]);
  return g.node('Concat', outs, { axis: I(0) })[0];
}

export function buildBetaPhone(w: BetaPhone, chunk: number, gruMode: GruMode = 'onnx-gru'): StreamingGraph {
  const g = new GraphBuilder(`beatrice_beta2_phone_${gruMode}`), states: StateSpec[] = [];
  let x = g.input('wav', [1, 1, chunk * 160 + 80]);
  const ch = [1, 32, 64, 128, 256, 256, 256], kernel = [10, 3, 3, 3, 3, 2], stride = [5, 2, 2, 2, 2, 2];
  for (let i = 0; i < 6; i++) x = g.geluTanh(g.conv1d(x, w.fe[i], null, ch[i + 1], ch[i], kernel[i], { stride: stride[i] }));
  x = g.transpose(g.layerNorm(g.transpose(x, [0, 2, 1]), null, null, 256), [0, 2, 1]);
  x = g.conv1d(x, w.projection.w, w.projection.b, 256, 256, 1);
  const projected = x;
  let sequence = g.transpose(x, [2, 0, 1]);
  for (let i = 0; i < w.gru.length; i++) {
    if (gruMode === 'primitives') {
      sequence = gruPrimitives(g, states, sequence, w.gru[i], chunk, i);
      continue;
    }
    const weights = onnxGruWeights(w.gru[i]);
    const h = g.input(`gru${i}`, [1, 1, 256]);
    const [y, next] = g.node('GRU', [
      sequence,
      g.f32(`gru${i}_wi`, weights.wi, [1, 768, 256]),
      g.f32(`gru${i}_wh`, weights.wh, [1, 768, 256]),
      g.f32(`gru${i}_bias`, weights.bias, [1, 1536]),
      '', h,
    ], { hidden_size: I(256), linear_before_reset: I(1) }, 2);
    remember(g, states, `gru${i}`, next, [1, 1, 256]);
    sequence = g.reshape(y, [chunk, 1, 256]);
  }
  x = g.add(projected, g.transpose(sequence, [1, 2, 0]));
  x = g.geluTanh(stack(g, states, x, w.backbone, chunk, 'phone'));
  x = g.conv1d(x, w.head.w, w.head.b, 256, 256, 1);
  g.named('Identity', [x], 'units'); g.output('units', [1, 256, chunk]);
  return finish(g, states);
}

export function buildBetaPitch(w: BetaPitch, chunk: number): StreamingGraph {
  const g = new GraphBuilder('beatrice_beta2_pitch'), states: StateSpec[] = [];
  const inst = g.input('instfreq', [1, 192, chunk + 1]);
  const corr = g.input('corr_diff', [1, 256, chunk + 1]);
  const valid = g.input('valid', [1, 1, chunk + 1]);
  const branch = (x: string, a: Lin, b: Lin, cin: number) => g.conv1d(g.geluTanh(g.conv1d(x, a.w, a.b, 192, cin, 1)), b.w, b.b, 192, 192, 1);
  // The missing post-sum GELU in beta.2 is intentional here: adding it changes the trained network.
  const x = stack(g, states, g.mul(g.add(branch(inst, w.if0, w.if1, 192), branch(corr, w.cr0, w.cr1, 256)), valid), w.backbone, chunk, 'pitch');
  const logits = g.conv1d(x, w.head.w, w.head.b, 384, 192, 1);
  g.named('Identity', [logits], 'logits'); g.output('logits', [1, 384, chunk]);
  return finish(g, states);
}

export function buildBetaVocoder(w: BetaWave, chunk: number): StreamingGraph {
  const g = new GraphBuilder('beatrice_beta2_vocoder'), states: StateSpec[] = [];
  const input = g.input('x', [1, 256, chunk + 2]);
  const pre = stack(g, states, input, w.pre, chunk, 'pre');
  const out = (s: Stack, post: Lin, c: number, name: string) => {
    const value = g.conv1d(g.silu(stack(g, states, pre, s, chunk, name)), post.w, post.b, c, 256, 1);
    g.named('Identity', [value], name); g.output(name, [1, c, chunk]);
  };
  out(w.ir, w.irPost, 512, 'ir'); out(w.ap, w.apPost, 240, 'aperiodicity'); out(w.pf, w.pfPost, 512, 'post_filter');
  return finish(g, states);
}