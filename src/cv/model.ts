import { bytesOf, floatsOf, MatScope, scoped, type CV, type Mat } from './opencv';

/**
 * Capture model: how should the REFERENCE look through THIS capture?
 *
 * Measured on a real photo: after excellent alignment (correlation 0.82) the capture still had 15% of
 * the panel area as copper the design does not have, against 0.3% the other way - traces
 * systematically ~0.2 mm/side wider than designed. Left alone that reads as ~1,400 local "defects"
 * (every trace edge), and no edge tolerance small enough to still catch real defects can absorb it.
 *
 * Two effects are fitted and separated from local defects:
 *   - width bias  delta: copper wider (+) / narrower (-) than designed, per side. Real under/over-etch
 *                 and optical bloom look identical in a photo, so it is REPORTED and blocks a clean
 *                 "pass" when large.
 *   - blur        sigma: optical softness. Gaps narrower than ~2 sigma fuse and become un-judgeable,
 *                 which sets the honest detection limit.
 *
 * They are fitted PER TILE, not globally. A global fit was tried first: on the same photo the upper
 * half was visibly defocused (the panel was bowed) and the lower half sharp, so one blur value was too
 * soft for the sharp half (the reference's dense fan-out fused while the capture kept it separate ->
 * a cluster of false "open"s) and too crisp for the blurry half. Per-tile parameters are blended with
 * bilinear weights so there are no seams.
 *
 * Local defects are judged against E = blur(dilate(reference, delta), sigma) > 0.5.
 */

export interface CaptureModel {
  /** Median width offset per side, working px (>0: capture copper is wider than the reference). */
  biasPx: number;
  /** Median optical blur (Gaussian sigma), working px. */
  sigmaPx: number;
  /** Range of per-tile values, for reporting how uneven the capture is. */
  biasRangePx: [number, number];
  sigmaRangePx: [number, number];
  grid: { cols: number; rows: number };
  /** The reference as it should look through this capture (CV_8UC1, copper = 255). */
  expected: Mat;
  /** Mismatch (fraction of valid area) against the raw reference, and against the fitted expectation. */
  mismatchBefore: number;
  mismatchAfter: number;
}

const TILE_MM = 30;
const DELTAS = [-3, -2, -1, 0, 1, 2, 3, 4, 5]; // search-scale px
const SIGMAS = [0, 1.5, 3, 4.5]; // search-scale px
const MIN_TILE_VALID = 0.5; // a tile needs this fraction of valid pixels to be fitted on its own

/** Signed distance to the copper boundary, sub-pixel: negative inside copper, positive outside. */
function signedDistance(cv: CV, bin: Mat): Float32Array {
  return scoped((s) => {
    const inv = s.add(new cv.Mat());
    cv.bitwise_not(bin, inv);
    const dIn = s.add(new cv.Mat());
    const dOut = s.add(new cv.Mat());
    cv.distanceTransform(bin, dIn, cv.DIST_L2, cv.DIST_MASK_5); // copper pixel -> distance to nearest non-copper
    cv.distanceTransform(inv, dOut, cv.DIST_L2, cv.DIST_MASK_5); // background pixel -> distance to nearest copper
    const a = floatsOf(dIn);
    const b = floatsOf(dOut);
    const sd = new Float32Array(a.length);
    // a pixel centre is half a pixel from the boundary of its own class, hence the 0.5
    for (let i = 0; i < sd.length; i++) sd[i] = b[i]! > 0 ? b[i]! - 0.5 : -(a[i]! - 0.5);
    return sd;
  });
}

/**
 * Gaussian blur of a float image. For a large sigma the image is shrunk first (a wide blur has no
 * detail to lose), blurred, and enlarged again: a 12 MP image with a ~80-tap kernel otherwise takes
 * seconds per candidate in WASM.
 */
function blurFloat(cv: CV, data: Float32Array, w: number, h: number, sigma: number): Float32Array {
  return scoped((s) => {
    const k = sigma >= 6 ? Math.floor(sigma / 3) : 1;
    const src = s.add(new cv.Mat(h, w, cv.CV_32FC1));
    src.data32F.set(data);
    if (k === 1) {
      const dst = s.add(new cv.Mat());
      cv.GaussianBlur(src, dst, new cv.Size(0, 0), sigma, sigma, cv.BORDER_REPLICATE);
      return floatsOf(dst);
    }
    const sw = Math.max(2, Math.ceil(w / k));
    const sh = Math.max(2, Math.ceil(h / k));
    const small = s.add(new cv.Mat());
    cv.resize(src, small, new cv.Size(sw, sh), 0, 0, cv.INTER_AREA);
    s.free(src);
    const blurred = s.add(new cv.Mat());
    cv.GaussianBlur(small, blurred, new cv.Size(0, 0), sigma / k, sigma / k, cv.BORDER_REPLICATE);
    const up = s.add(new cv.Mat());
    cv.resize(blurred, up, new cv.Size(w, h), 0, 0, cv.INTER_LINEAR);
    return floatsOf(up);
  });
}

const median = (v: number[]): number => {
  const s = [...v].sort((a, b) => a - b);
  return s[s.length >> 1]!;
};

const snap = (v: number, allowed: number[]): number => allowed.reduce((best, a) => (Math.abs(a - v) < Math.abs(best - v) ? a : best), allowed[0]!);

export function fitCaptureModel(cv: CV, out: MatScope, ref: Mat, cap: Mat, valid: Mat, workPxPerMm: number): CaptureModel {
  return scoped((tmp) => fitInner(cv, tmp, out, ref, cap, valid, workPxPerMm));
}

function fitInner(cv: CV, scope: MatScope, out: MatScope, ref: Mat, cap: Mat, valid: Mat, workPxPerMm: number): CaptureModel {
  const w = ref.cols;
  const h = ref.rows;

  // ---- search on a reduced copy: dozens of blurs at full size would be needlessly slow -------
  const f = Math.max(1, Math.round(Math.max(w, h) / 1400));
  const sw = Math.max(8, Math.round(w / f));
  const sh = Math.max(8, Math.round(h / f));
  const shrink = (m: Mat, interp: number, binarize: boolean): Mat => {
    const o = scope.add(new cv.Mat());
    cv.resize(m, o, new cv.Size(sw, sh), 0, 0, interp);
    if (!binarize) return o;
    const t = scope.add(new cv.Mat());
    cv.threshold(o, t, 127, 255, cv.THRESH_BINARY);
    return t;
  };
  const refS = f === 1 ? ref : shrink(ref, cv.INTER_AREA, true);
  const capBytes = bytesOf(f === 1 ? cap : shrink(cap, cv.INTER_AREA, true));
  const validBytes = bytesOf(f === 1 ? valid : shrink(valid, cv.INTER_NEAREST, false));
  const sd = signedDistance(cv, refS);
  const n = sw * sh;

  // ---- tile grid ----------------------------------------------------------------------------
  const apm = workPxPerMm / f;
  const cols = Math.max(1, Math.round(sw / (TILE_MM * apm)));
  const rows = Math.max(1, Math.round(sh / (TILE_MM * apm)));
  const nTiles = cols * rows;
  const tileOf = new Uint16Array(n);
  for (let y = 0; y < sh; y++) {
    const r = Math.min(rows - 1, Math.floor((y * rows) / sh));
    for (let x = 0; x < sw; x++) tileOf[y * sw + x] = r * cols + Math.min(cols - 1, Math.floor((x * cols) / sw));
  }
  const tileValid = new Float64Array(nTiles);
  const tileArea = new Float64Array(nTiles);
  for (let i = 0; i < n; i++) {
    tileArea[tileOf[i]!]!++;
    if (validBytes[i]) tileValid[tileOf[i]!]!++;
  }
  let totalValid = 0;
  for (let t = 0; t < nTiles; t++) totalValid += tileValid[t]!;
  if (totalValid === 0) throw new Error('capture model: no valid pixels');

  // ---- evaluate every candidate; keep the per-tile mismatch of each --------------------------
  const src = scope.add(new cv.Mat(sh, sw, cv.CV_32FC1));
  const dst = scope.add(new cv.Mat());
  const mismatchPerTile = (delta: number, sigma: number): Float64Array => {
    const m = new Float32Array(n);
    for (let i = 0; i < n; i++) m[i] = sd[i]! < delta ? 1 : 0;
    let e: Float32Array = m;
    const blurred = sigma > 0.25;
    if (blurred) {
      src.data32F.set(m);
      cv.GaussianBlur(src, dst, new cv.Size(0, 0), sigma, sigma, cv.BORDER_REPLICATE);
      e = dst.data32F;
    }
    const bad = new Float64Array(nTiles);
    for (let i = 0; i < n; i++) {
      if (!validBytes[i]) continue;
      const exp = blurred ? e[i]! > 0.5 : e[i]! > 0;
      if (exp !== capBytes[i]! > 0) bad[tileOf[i]!]!++;
    }
    return bad;
  };

  const cands: { delta: number; sigma: number; bad: Float64Array; total: number }[] = [];
  for (const sigma of SIGMAS) {
    for (const delta of DELTAS) {
      const bad = mismatchPerTile(delta, sigma);
      cands.push({ delta, sigma, bad, total: bad.reduce((a, v) => a + v, 0) });
    }
  }
  const raw = cands.find((c) => c.delta === 0 && c.sigma === 0)!;
  const globalBest = cands.reduce((b, c) => (c.total < b.total ? c : b));

  // ---- per-tile optimum, falling back to the global optimum for tiles that cannot be fitted ---
  const tDelta = new Array<number>(nTiles);
  const tSigma = new Array<number>(nTiles);
  for (let t = 0; t < nTiles; t++) {
    if (tileValid[t]! / Math.max(1, tileArea[t]!) < MIN_TILE_VALID) {
      tDelta[t] = globalBest.delta;
      tSigma[t] = globalBest.sigma;
      continue;
    }
    let bestC = cands[0]!;
    for (const c of cands) if (c.bad[t]! < bestC.bad[t]!) bestC = c;
    tDelta[t] = bestC.delta;
    tSigma[t] = bestC.sigma;
  }
  // robustness: a 3x3 median across neighbouring tiles rejects a tile whose fit was pulled by a real defect
  const med3 = (a: number[]): number[] =>
    a.map((_, t) => {
      const r = Math.floor(t / cols);
      const c = t % cols;
      const vals: number[] = [];
      for (let rr = Math.max(0, r - 1); rr <= Math.min(rows - 1, r + 1); rr++)
        for (let cc = Math.max(0, c - 1); cc <= Math.min(cols - 1, c + 1); cc++) vals.push(a[rr * cols + cc]!);
      return median(vals);
    });
  const fDelta = med3(tDelta).map((v) => snap(v, DELTAS));
  const fSigma = med3(tSigma).map((v) => snap(v, SIGMAS));

  let badAfter = 0;
  for (let t = 0; t < nTiles; t++) {
    const c = cands.find((x) => x.delta === fDelta[t] && x.sigma === fSigma[t])!;
    badAfter += c.bad[t]!;
  }

  // ---- materialise at full working resolution, blending tiles bilinearly ---------------------
  const sdFull = f === 1 ? sd : signedDistance(cv, ref);
  const nFull = w * h;
  const acc = new Float32Array(nFull);
  const pairs = new Map<string, { delta: number; sigma: number }>();
  for (let t = 0; t < nTiles; t++) pairs.set(`${fDelta[t]}|${fSigma[t]}`, { delta: fDelta[t]!, sigma: fSigma[t]! });

  // Separable bilinear weights between tile centres: per-column / per-row neighbours and fractions.
  const colIdx0 = new Int32Array(w);
  const colIdx1 = new Int32Array(w);
  const colFrac = new Float32Array(w);
  for (let x = 0; x < w; x++) {
    const u = ((x + 0.5) * cols) / w - 0.5;
    const i0 = Math.max(0, Math.min(cols - 1, Math.floor(u)));
    colIdx0[x] = i0;
    colIdx1[x] = Math.min(cols - 1, i0 + 1);
    colFrac[x] = Math.max(0, Math.min(1, u - i0));
  }
  const rowIdx0 = new Int32Array(h);
  const rowIdx1 = new Int32Array(h);
  const rowFrac = new Float32Array(h);
  for (let y = 0; y < h; y++) {
    const v = ((y + 0.5) * rows) / h - 0.5;
    const i0 = Math.max(0, Math.min(rows - 1, Math.floor(v)));
    rowIdx0[y] = i0;
    rowIdx1[y] = Math.min(rows - 1, i0 + 1);
    rowFrac[y] = Math.max(0, Math.min(1, v - i0));
  }

  for (const [key, p] of pairs) {
    const dW = p.delta * f;
    const sW = p.sigma * f;
    const bin = new Float32Array(nFull);
    for (let i = 0; i < nFull; i++) bin[i] = sdFull[i]! < dW ? 1 : 0;
    const e = sW > 0.25 ? blurFloat(cv, bin, w, h, sW) : bin;
    const ind = new Float32Array(nTiles);
    for (let t = 0; t < nTiles; t++) ind[t] = `${fDelta[t]}|${fSigma[t]}` === key ? 1 : 0;
    const single = pairs.size === 1;
    for (let y = 0; y < h; y++) {
      const r0 = rowIdx0[y]! * cols;
      const r1 = rowIdx1[y]! * cols;
      const fy = rowFrac[y]!;
      for (let x = 0; x < w; x++) {
        const i = y * w + x;
        if (single) {
          acc[i] = e[i]!;
          continue;
        }
        const fx = colFrac[x]!;
        const c0 = colIdx0[x]!;
        const c1 = colIdx1[x]!;
        const wgt = ind[r0 + c0]! * (1 - fx) * (1 - fy) + ind[r0 + c1]! * fx * (1 - fy) + ind[r1 + c0]! * (1 - fx) * fy + ind[r1 + c1]! * fx * fy;
        acc[i]! += wgt * e[i]!;
      }
    }
  }
  const outBytes = new Uint8Array(nFull);
  for (let i = 0; i < nFull; i++) outBytes[i] = acc[i]! > 0.5 ? 255 : 0;
  const expected = out.add(new cv.Mat(h, w, cv.CV_8UC1));
  expected.data.set(outBytes);

  const range = (v: number[]): [number, number] => [Math.min(...v) * f, Math.max(...v) * f];
  return {
    biasPx: median(fDelta) * f,
    sigmaPx: median(fSigma) * f,
    biasRangePx: range(fDelta),
    sigmaRangePx: range(fSigma),
    grid: { cols, rows },
    expected,
    mismatchBefore: raw.total / totalValid,
    mismatchAfter: badAfter / totalValid,
  };
}
