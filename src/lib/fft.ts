/** Complex FFT: radix-2 (iterative) for powers of two, Bluestein (chirp-z) otherwise.
 *  Sizes required by the Beatrice vocoder: 480, 512, 768 — all correct even when not a power of two. */

class Fft {
  readonly n: number;
  private cos: Float64Array;
  private sin: Float64Array;
  private rev: Int32Array;
  private bl: { m: number; inner: Fft; wr: Float64Array; wi: Float64Array; ar: Float64Array; ai: Float64Array; br: Float64Array; bi: Float64Array } | null = null;

  constructor(n: number) {
    this.n = n;
    if ((n & (n - 1)) === 0) {
      const half = n >> 1;
      this.cos = new Float64Array(half); this.sin = new Float64Array(half);
      for (let i = 0; i < half; i++) { const a = (-2 * Math.PI * i) / n; this.cos[i] = Math.cos(a); this.sin[i] = Math.sin(a); }
      this.rev = new Int32Array(n);
      let bits = 0; while (1 << bits < n) bits++;
      for (let i = 0; i < n; i++) { let r = 0; for (let b = 0; b < bits; b++) if (i & (1 << b)) r |= 1 << (bits - 1 - b); this.rev[i] = r; }
    } else {
      // Bluestein: chirp factors w_k = exp(-i π k² / n)
      let m = 1; while (m < 2 * n - 1) m <<= 1;
      const wr = new Float64Array(n), wi = new Float64Array(n);
      for (let k = 0; k < n; k++) { const a = (-Math.PI * ((k * k) % (2 * n))) / n; wr[k] = Math.cos(a); wi[k] = Math.sin(a); }
      const inner = new Fft(m);
      const ar = new Float64Array(m), ai = new Float64Array(m);
      ar[0] = 1; ai[0] = 0;
      for (let k = 1; k < n; k++) { ar[k] = ar[m - k] = wr[k]; ai[k] = ai[m - k] = -wi[k]; }
      // pre-transform the chirp filter once
      inner.transform(ar, ai, false);
      this.cos = new Float64Array(1); this.sin = new Float64Array(1); this.rev = new Int32Array(1);
      this.bl = { m, inner, wr, wi, ar, ai, br: new Float64Array(m), bi: new Float64Array(m) };
    }
  }

  /** in-place forward DFT (inverse = false) or inverse (true, WITHOUT 1/n scaling). */
  transform(re: Float64Array, im: Float64Array, inverse = false) {
    if (this.bl) return this.bluestein(re, im, inverse);
    const n = this.n, rev = this.rev, cs = this.cos, sn = this.sin;
    for (let i = 0; i < n; i++) { const j = rev[i]; if (j > i) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; } }
    for (let size = 2; size <= n; size <<= 1) {
      const half = size >> 1, step = n / size;
      for (let i = 0; i < n; i += size) {
        for (let j = i, k = 0; j < i + half; j++, k += step) {
          const c = cs[k], s = inverse ? -sn[k] : sn[k];
          const l = j + half;
          const tr = re[l] * c - im[l] * s, ti = re[l] * s + im[l] * c;
          re[l] = re[j] - tr; im[l] = im[j] - ti;
          re[j] += tr; im[j] += ti;
        }
      }
    }
  }

  private bluestein(re: Float64Array, im: Float64Array, inverse: boolean) {
    const { m, inner, wr, wi, ar, ai, br, bi } = this.bl!;
    const n = this.n;
    br.fill(0); bi.fill(0);
    for (let k = 0; k < n; k++) {
      const c = wr[k], s = inverse ? -wi[k] : wi[k];
      br[k] = re[k] * c - im[k] * s;
      bi[k] = re[k] * s + im[k] * c;
    }
    inner.transform(br, bi, false);
    for (let k = 0; k < m; k++) {
      const cr = ar[k], ci = inverse ? -ai[k] : ai[k];
      const tr = br[k] * cr - bi[k] * ci, ti = br[k] * ci + bi[k] * cr;
      br[k] = tr; bi[k] = ti;
    }
    inner.transform(br, bi, true);
    const sc = 1 / m;
    for (let k = 0; k < m; k++) { br[k] *= sc; bi[k] *= sc; }
    for (let k = 0; k < n; k++) {
      const c = wr[k], s = inverse ? -wi[k] : wi[k];
      re[k] = br[k] * c - bi[k] * s;
      im[k] = br[k] * s + bi[k] * c;
    }
  }
}

const cache = new Map<number, Fft>();
const getFft = (n: number): Fft => { let f = cache.get(n); if (!f) { f = new Fft(n); cache.set(n, f); } return f; };
const scratch = new Map<number, { re: Float64Array; im: Float64Array }>();
const getScratch = (n: number) => {
  let s = scratch.get(n);
  if (!s) { s = { re: new Float64Array(n), im: new Float64Array(n) }; scratch.set(n, s); }
  s.re.fill(0); s.im.fill(0);
  return s;
};

/** real → half spectrum (n/2+1 bins), matching torch.fft.rfft. `x` may be shorter than n (zero-padded). */
export function rfft(x: Float32Array | Float64Array, n: number, outRe: Float64Array, outIm: Float64Array) {
  const { re, im } = getScratch(n);
  const L = Math.min(x.length, n);
  for (let i = 0; i < L; i++) re[i] = x[i];
  getFft(n).transform(re, im, false);
  const h = n >> 1;
  for (let k = 0; k <= h; k++) { outRe[k] = re[k]; outIm[k] = im[k]; }
}

/** half spectrum (n/2+1 bins) → real signal of length n, scaled by 1/n (matches torch.fft.irfft). */
export function irfft(inRe: Float64Array, inIm: Float64Array, n: number, out: Float32Array | Float64Array) {
  const { re, im } = getScratch(n);
  const h = n >> 1;
  for (let k = 0; k <= h; k++) { re[k] = inRe[k]; im[k] = inIm[k]; }
  for (let k = h + 1; k < n; k++) { re[k] = inRe[n - k]; im[k] = -inIm[n - k]; }
  getFft(n).transform(re, im, true);
  const s = 1 / n;
  for (let i = 0; i < n; i++) out[i] = re[i] * s;
}
