/**
 * Tiny in-browser ONNX (protobuf) writer. We build the Beatrice graphs at runtime from the
 * float16 paraphernalia weights and hand the serialized ModelProto to onnxruntime-web (WebGPU / WASM).
 */

type Attr =
  | { t: "i"; v: number }
  | { t: "f"; v: number }
  | { t: "is"; v: number[] }
  | { t: "fs"; v: number[] }
  | { t: "s"; v: string };

export const I = (v: number): Attr => ({ t: "i", v });
export const F = (v: number): Attr => ({ t: "f", v });
export const IS = (v: number[]): Attr => ({ t: "is", v });
export const FS = (v: number[]): Attr => ({ t: "fs", v });
export const S = (v: string): Attr => ({ t: "s", v });

const enc = new TextEncoder();

class Writer {
  private chunks: Uint8Array[] = [];
  private len = 0;
  push(b: Uint8Array) { this.chunks.push(b); this.len += b.length; }
  varint(n: number | bigint) {
    let v = BigInt(n);
    if (v < 0n) v = (1n << 64n) + v;
    const out: number[] = [];
    do { let b = Number(v & 0x7fn); v >>= 7n; if (v !== 0n) b |= 0x80; out.push(b); } while (v !== 0n);
    this.push(Uint8Array.from(out));
  }
  tag(field: number, wire: number) { this.varint((field << 3) | wire); }
  int(field: number, v: number | bigint) { this.tag(field, 0); this.varint(v); }
  float(field: number, v: number) { this.tag(field, 5); const b = new Uint8Array(4); new DataView(b.buffer).setFloat32(0, v, true); this.push(b); }
  bytes(field: number, b: Uint8Array) { this.tag(field, 2); this.varint(b.length); this.push(b); }
  str(field: number, s: string) { this.bytes(field, enc.encode(s)); }
  // Splice protobuf chunks instead of materializing tensor -> graph -> model copies.
  // The final ModelProto is flattened once in finish().
  msg(field: number, w: Writer) {
    this.tag(field, 2); this.varint(w.len);
    for (const chunk of w.chunks) this.push(chunk);
  }
  finish(): Uint8Array {
    const out = new Uint8Array(this.len);
    let o = 0;
    for (const c of this.chunks) { out.set(c, o); o += c.length; }
    return out;
  }
}

export type Dim = number | string;

function tensorProto(name: string, dataType: number, dims: number[], raw: Uint8Array): Writer {
  const w = new Writer();
  for (const d of dims) w.int(1, d);
  w.int(2, dataType);
  w.str(8, name);
  w.bytes(9, raw);
  return w;
}

function valueInfo(name: string, elemType: number, dims: Dim[]): Writer {
  const shape = new Writer();
  for (const d of dims) {
    const dim = new Writer();
    if (typeof d === "number") dim.int(1, d); else dim.str(2, d);
    shape.msg(1, dim);
  }
  const tt = new Writer();
  tt.int(1, elemType);
  tt.msg(2, shape);
  const tp = new Writer();
  tp.msg(1, tt);
  const vi = new Writer();
  vi.str(1, name);
  vi.msg(2, tp);
  return vi;
}

export class GraphBuilder {
  private nodes: Writer[] = [];
  private inits: Writer[] = [];
  private inputs: Writer[] = [];
  private outputs: Writer[] = [];
  private counter = 0;
  public bytesOfWeights = 0;
  /** Static I/O specs (only numeric dims) — used for GPU buffer pre-allocation / graph capture. */
  readonly inputSpecs: { name: string; dims: number[] }[] = [];
  readonly outputSpecs: { name: string; dims: number[] }[] = [];
  constructor(private graphName: string) {}

  uniq(prefix: string) { return `${prefix}_${this.counter++}`; }

  input(name: string, dims: Dim[]) {
    this.inputs.push(valueInfo(name, 1, dims));
    if (dims.every((d) => typeof d === "number")) this.inputSpecs.push({ name, dims: dims as number[] });
    return name;
  }
  output(name: string, dims: Dim[]) {
    this.outputs.push(valueInfo(name, 1, dims));
    if (dims.every((d) => typeof d === "number")) this.outputSpecs.push({ name, dims: dims as number[] });
    return name;
  }

  f32(name: string, data: Float32Array | number[], dims: number[]): string {
    const arr = data instanceof Float32Array ? data : Float32Array.from(data);
    const n = dims.reduce((a, b) => a * b, 1);
    if (arr.length !== n) throw new Error(`const ${name}: ${arr.length} values for dims [${dims}]`);
    // No defensive copy here: build() concatenates exactly once. Copying again doubled peak memory
    // during session creation (the 40 MB models tripled to ~140 MB transient on phones).
    const raw = new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength);
    this.bytesOfWeights += raw.length;
    this.inits.push(tensorProto(name, 1, dims, raw));
    return name;
  }
  i64(name: string, data: number[], dims?: number[]): string {
    const b = new BigInt64Array(data.map((x) => BigInt(x)));
    this.inits.push(tensorProto(name, 7, dims ?? [data.length], new Uint8Array(b.buffer)));
    return name;
  }

  node(op: string, inputs: string[], attrs: Record<string, Attr> = {}, nOut = 1, domain?: string): string[] {
    const outs: string[] = [];
    for (let i = 0; i < nOut; i++) outs.push(this.uniq(op.toLowerCase()));
    const w = new Writer();
    for (const i of inputs) w.str(1, i);
    for (const o of outs) w.str(2, o);
    w.str(3, this.uniq("n_" + op));
    w.str(4, op);
    for (const [k, a] of Object.entries(attrs)) {
      const aw = new Writer();
      aw.str(1, k);
      switch (a.t) {
        case "f": aw.float(2, a.v); aw.int(20, 1); break;
        case "i": aw.int(3, a.v); aw.int(20, 2); break;
        case "s": aw.str(4, a.v); aw.int(20, 3); break;
        case "fs": for (const v of a.v) aw.float(7, v); aw.int(20, 6); break;
        case "is": for (const v of a.v) aw.int(8, v); aw.int(20, 7); break;
      }
      w.msg(5, aw);
    }
    if (domain) w.str(7, domain);
    this.nodes.push(w);
    return outs;
  }
  /** node with an explicitly named single output (for graph outputs) */
  named(op: string, inputs: string[], outName: string, attrs: Record<string, Attr> = {}): string {
    const w = new Writer();
    for (const i of inputs) w.str(1, i);
    w.str(2, outName);
    w.str(3, this.uniq("n_" + op));
    w.str(4, op);
    for (const [k, a] of Object.entries(attrs)) {
      const aw = new Writer();
      aw.str(1, k);
      switch (a.t) {
        case "f": aw.float(2, a.v); aw.int(20, 1); break;
        case "i": aw.int(3, a.v); aw.int(20, 2); break;
        case "s": aw.str(4, a.v); aw.int(20, 3); break;
        case "fs": for (const v of a.v) aw.float(7, v); aw.int(20, 6); break;
        case "is": for (const v of a.v) aw.int(8, v); aw.int(20, 7); break;
      }
      w.msg(5, aw);
    }
    this.nodes.push(w);
    return outName;
  }

  // ---- convenience ops ----
  add(a: string, b: string) { return this.node("Add", [a, b])[0]; }
  mul(a: string, b: string) { return this.node("Mul", [a, b])[0]; }
  matmul(a: string, b: string) { return this.node("MatMul", [a, b])[0]; }
  transpose(x: string, perm: number[]) { return this.node("Transpose", [x], { perm: IS(perm) })[0]; }
  reshape(x: string, shape: number[]) { return this.node("Reshape", [x, this.i64(this.uniq("shape"), shape)])[0]; }
  softmax(x: string, axis = -1) { return this.node("Softmax", [x], { axis: I(axis) })[0]; }
  /** F.gelu(approximate="tanh") as the single fused opset-20 Gelu op (one dispatch instead of seven). */
  geluTanh(x: string): string {
    return this.node("Gelu", [x], { approximate: S("tanh") })[0];
  }
  silu(x: string): string { return this.mul(x, this.node("Sigmoid", [x])[0]); }
  scalar(v: number): string { return this.f32(this.uniq("c"), [v], []); }
  /** LayerNormalization over the last axis (opset 17). scale/bias may be null → identity affine. */
  layerNorm(x: string, scale: Float32Array | null, bias: Float32Array | null, channels: number, eps = 1e-5): string {
    const s = this.f32(this.uniq("ln_w"), scale ?? new Float32Array(channels).fill(1), [channels]);
    const b = this.f32(this.uniq("ln_b"), bias ?? new Float32Array(channels), [channels]);
    return this.node("LayerNormalization", [x, s, b], { axis: I(-1), epsilon: F(eps) })[0];
  }
  /** Conv1d on [N, Cin, T]. weight [Cout, Cin/groups, K]. pads = [left, right]. */
  conv1d(x: string, w: Float32Array, b: Float32Array | null, cout: number, cinPerGroup: number, k: number, opts: { stride?: number; pads?: [number, number]; groups?: number } = {}): string {
    const W = this.f32(this.uniq("conv_w"), w, [cout, cinPerGroup, k]);
    const ins = [x, W];
    if (b) ins.push(this.f32(this.uniq("conv_b"), b, [cout]));
    return this.node("Conv", ins, {
      kernel_shape: IS([k]),
      strides: IS([opts.stride ?? 1]),
      pads: IS(opts.pads ?? [0, 0]),
      group: I(opts.groups ?? 1),
    })[0];
  }
  /** x [.., Cin] · W^T + b with torch Linear weight [Cout, Cin]. */
  linear(x: string, w: Float32Array, b: Float32Array | null, cout: number, cin: number): string {
    const wt = new Float32Array(cin * cout);
    for (let o = 0; o < cout; o++) for (let i = 0; i < cin; i++) wt[i * cout + o] = w[o * cin + i];
    let y = this.matmul(x, this.f32(this.uniq("lin_w"), wt, [cin, cout]));
    if (b) y = this.add(y, this.f32(this.uniq("lin_b"), b, [cout]));
    return y;
  }
  slice(x: string, starts: number[], ends: number[], axes: number[]): string {
    return this.node("Slice", [x, this.i64(this.uniq("st"), starts), this.i64(this.uniq("en"), ends), this.i64(this.uniq("ax"), axes)])[0];
  }
  pad(x: string, pads: number[]): string {
    return this.node("Pad", [x, this.i64(this.uniq("pads"), pads)], { mode: S("constant") })[0];
  }

  build(opset = 20): Uint8Array {
    const g = new Writer();
    for (const n of this.nodes) g.msg(1, n);
    g.str(2, this.graphName);
    for (const t of this.inits) g.msg(5, t);
    for (const i of this.inputs) g.msg(11, i);
    for (const o of this.outputs) g.msg(12, o);
    const m = new Writer();
    m.int(1, 9); // ir_version 9 (opset 20: fused Gelu, LayerNormalization, Split with split input)
    m.str(2, "beatrice-web");
    m.msg(7, g);
    const op = new Writer();
    op.str(1, "");
    op.int(2, opset);
    m.msg(8, op);
    return m.finish();
  }
}
