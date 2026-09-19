import { cvConfig, type BinarizeChannel } from '../config';
import { PipelineError } from '../errors';
import { bytesOf, MatScope, type CV, type Mat } from './opencv';

/**
 * Copper / substrate binarisation. This ONE function is used by golden-reference registration AND by
 * inspection (plan §5.3) - two independently tuned binarisers would disagree systematically and
 * show up as false defects everywhere, not just at real mismatches.
 *
 * Why not a fixed threshold: lighting varies across a board and between shots (in a real test photo
 * the dimmer half of the panel dropped out under a fixed colour threshold).
 * Why not a small adaptive window (the old code used block size 11 at ~40 px/mm = 0.27 mm): a window
 * smaller than the copper features turns the interior of every wide trace/pour into "background".
 *
 * Instead: take a first-pass classification, estimate the local copper level L(x,y) and substrate
 * level D(x,y) SEPARATELY (masked, heavily smoothed - so a missing trace barely moves them), and
 * threshold each pixel at (L+D)/2. Dense and sparse regions both work because each level is
 * estimated from its own pixels.
 */

export interface BinarizeOptions {
  pxPerMm: number;
  channel?: BinarizeChannel;
  /** 255 where the pixel carries real image data. Warp borders are excluded from every statistic. */
  valid?: Mat;
  /** true (default): copper is brighter than substrate in the chosen channel. */
  copperIsBright?: boolean;
}

export interface BinarizeResult {
  /** CV_8UC1, copper = 255. Owned by the caller's scope. */
  mask: Mat;
  /** copper pixels / valid pixels */
  coverage: number;
  /** mean copper level - mean substrate level, 0-255 */
  contrast: number;
  /** the first-pass global Otsu level (diagnostic) */
  otsu: number;
}

function scorePlane(rgb: Uint8Array, n: number, channel: BinarizeChannel): Uint8Array {
  const out = new Uint8Array(n);
  switch (channel) {
    case 'red':
      for (let i = 0; i < n; i++) out[i] = rgb[i * 3]!;
      break;
    case 'green':
      for (let i = 0; i < n; i++) out[i] = rgb[i * 3 + 1]!;
      break;
    case 'value':
      for (let i = 0; i < n; i++) out[i] = Math.max(rgb[i * 3]!, rgb[i * 3 + 1]!, rgb[i * 3 + 2]!);
      break;
    default:
      for (let i = 0; i < n; i++) out[i] = (77 * rgb[i * 3]! + 150 * rgb[i * 3 + 1]! + 29 * rgb[i * 3 + 2]!) >> 8;
  }
  return out;
}

/** Otsu's level: pixels strictly above it are foreground. */
export function otsuLevel(hist: Uint32Array, total: number): number {
  let sum = 0;
  for (let i = 0; i < 256; i++) sum += i * hist[i]!;
  let sumB = 0;
  let wB = 0;
  let best = -1;
  let level = 127;
  for (let t = 0; t < 256; t++) {
    wB += hist[t]!;
    if (wB === 0) continue;
    const wF = total - wB;
    if (wF === 0) break;
    sumB += t * hist[t]!;
    const mB = sumB / wB;
    const mF = (sum - sumB) / wF;
    const between = wB * wF * (mB - mF) * (mB - mF);
    if (between > best) {
      best = between;
      level = t;
    }
  }
  return level;
}

/**
 * An opening removes speckle but also erases any feature narrower than its kernel, and the
 * narrowest copper on the reference board is 0.25 mm (plan §6). At 10 px/mm that is 2.5 px - a 3x3
 * opening would delete real traces - so the kernel is capped to what the resolution can afford.
 */
export function speckleKernel(pxPerMm: number): number {
  const affordable = Math.floor(0.25 * pxPerMm * 0.8);
  let k = Math.min(cvConfig.morphKernel, affordable);
  if (k % 2 === 0) k -= 1;
  return Math.max(1, k);
}

/** Temporaries live in an internal scope freed on return; only the final mask joins `out`. */
export function binarizeCopper(cv: CV, out: MatScope, rgb: Mat, opts: BinarizeOptions): BinarizeResult {
  const tmp = new MatScope();
  try {
    return binarizeInner(cv, tmp, out, rgb, opts);
  } finally {
    tmp.dispose();
  }
}

function binarizeInner(cv: CV, scope: MatScope, out: MatScope, rgb: Mat, opts: BinarizeOptions): BinarizeResult {
  if (rgb.type() !== cv.CV_8UC3) throw new Error('binarizeCopper expects an 8-bit 3-channel RGB Mat');
  const w = rgb.cols;
  const h = rgb.rows;
  const n = w * h;
  const channel = opts.channel ?? cvConfig.channel;
  if (opts.valid && (opts.valid.cols !== w || opts.valid.rows !== h)) throw new Error('valid mask size mismatch');
  // Copy: this function allocates Mats below, which can detach any live view into the WASM heap.
  const valid = opts.valid ? bytesOf(opts.valid) : null;

  const score = scorePlane(rgb.data, n, channel);
  if (opts.copperIsBright === false) for (let i = 0; i < n; i++) score[i] = 255 - score[i]!;

  // ---- first pass: global Otsu over valid pixels -------------------------------------------
  const hist = new Uint32Array(256);
  let validCount = 0;
  for (let i = 0; i < n; i++) {
    if (valid && !valid[i]) continue;
    hist[score[i]!]!++;
    validCount++;
  }
  if (validCount < 1000) throw new PipelineError('BINARIZATION_IMPLAUSIBLE', 'Too little usable image area to binarise');
  const otsu = otsuLevel(hist, validCount);

  const cls = new Uint8Array(n); // 1 = copper
  for (let i = 0; i < n; i++) cls[i] = valid && !valid[i] ? 0 : score[i]! > otsu ? 1 : 0;

  // ---- illumination-adaptive refinement, on a coarse grid ----------------------------------
  const f = Math.max(1, Math.round(Math.min(w, h) / 200));
  const lw = Math.ceil(w / f);
  const lh = Math.ceil(h / f);
  const sigma = Math.max(2, (cvConfig.illuminationSigmaMm * opts.pxPerMm) / f);
  const cell = f * f;

  for (let iter = 0; iter < 2; iter++) {
    const sumC = new Float32Array(lw * lh);
    const cntC = new Float32Array(lw * lh);
    const sumS = new Float32Array(lw * lh);
    const cntS = new Float32Array(lw * lh);
    let gSumC = 0;
    let gCntC = 0;
    let gSumS = 0;
    let gCntS = 0;
    for (let y = 0; y < h; y++) {
      const row = ((y / f) | 0) * lw;
      for (let x = 0; x < w; x++) {
        const i = y * w + x;
        if (valid && !valid[i]) continue;
        const li = row + ((x / f) | 0);
        const v = score[i]!;
        if (cls[i]) {
          sumC[li] += v;
          cntC[li] += 1;
          gSumC += v;
          gCntC++;
        } else {
          sumS[li] += v;
          cntS[li] += 1;
          gSumS += v;
          gCntS++;
        }
      }
    }
    if (gCntC === 0 || gCntS === 0) {
      throw new PipelineError(
        'BINARIZATION_IMPLAUSIBLE',
        'The image contains only one material (all copper or all substrate) - wrong subject, blown-out, or blank capture',
      );
    }
    const Lg = gSumC / gCntC;
    const Dg = gSumS / gCntS;

    const blur = (arr: Float32Array): Float32Array => {
      const src = scope.add(new cv.Mat(lh, lw, cv.CV_32FC1));
      src.data32F.set(arr);
      const dst = scope.add(new cv.Mat());
      cv.GaussianBlur(src, dst, new cv.Size(0, 0), sigma, sigma, cv.BORDER_CONSTANT);
      return Float32Array.from(dst.data32F);
    };
    const bSumC = blur(sumC);
    const bCntC = blur(cntC);
    const bSumS = blur(sumS);
    const bCntS = blur(cntS);

    const thr = new Float32Array(lw * lh);
    for (let li = 0; li < thr.length; li++) {
      // Fall back to the global level where a neighbourhood has (almost) none of that material.
      const wC = Math.min(1, bCntC[li]! / cell / 0.01);
      const wS = Math.min(1, bCntS[li]! / cell / 0.01);
      const Lloc = bCntC[li]! > 1e-6 ? bSumC[li]! / bCntC[li]! : Lg;
      const Dloc = bCntS[li]! > 1e-6 ? bSumS[li]! / bCntS[li]! : Dg;
      thr[li] = (wC * Lloc + (1 - wC) * Lg + (wS * Dloc + (1 - wS) * Dg)) / 2;
    }
    const thrLow = scope.add(new cv.Mat(lh, lw, cv.CV_32FC1));
    thrLow.data32F.set(thr);
    const thrFull = scope.add(new cv.Mat());
    cv.resize(thrLow, thrFull, new cv.Size(w, h), 0, 0, cv.INTER_LINEAR);
    const t = thrFull.data32F;
    for (let i = 0; i < n; i++) cls[i] = valid && !valid[i] ? 0 : score[i]! > t[i]! ? 1 : 0;
  }

  // ---- speckle cleanup (resolution-aware) --------------------------------------------------
  let mask = scope.add(new cv.Mat(h, w, cv.CV_8UC1));
  {
    const out = mask.data;
    for (let i = 0; i < n; i++) out[i] = cls[i] ? 255 : 0;
  }
  const k = speckleKernel(opts.pxPerMm);
  if (k > 1) {
    const opened = scope.add(new cv.Mat());
    const kernel = scope.add(cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(k, k)));
    cv.morphologyEx(mask, opened, cv.MORPH_OPEN, kernel);
    mask = opened;
  }

  // ---- validate the ARTIFACT, not the steps (plan §4.6) ------------------------------------
  const md = bytesOf(mask);
  let copper = 0;
  let sumCu = 0;
  let sumSu = 0;
  let cntSu = 0;
  for (let i = 0; i < n; i++) {
    if (valid && !valid[i]) continue;
    if (md[i]) {
      copper++;
      sumCu += score[i]!;
    } else {
      cntSu++;
      sumSu += score[i]!;
    }
  }
  const coverage = copper / validCount;
  const contrast = copper > 0 && cntSu > 0 ? sumCu / copper - sumSu / cntSu : 0;

  if (coverage < cvConfig.coverageMin || coverage > cvConfig.coverageMax) {
    throw new PipelineError(
      'BINARIZATION_IMPLAUSIBLE',
      `Implausible copper coverage ${(coverage * 100).toFixed(1)}% (accepted ${cvConfig.coverageMin * 100}-${cvConfig.coverageMax * 100}%). ` +
        'The subject is not the board, or lighting/focus is too poor to separate copper from substrate.',
      { coverage },
    );
  }
  if (contrast < cvConfig.minContrast) {
    throw new PipelineError(
      'BINARIZATION_IMPLAUSIBLE',
      `Copper/substrate contrast is only ${contrast.toFixed(0)}/255 (need >= ${cvConfig.minContrast}) - capture is too flat or washed out to trust.`,
      { contrast },
    );
  }

  const outMask = out.add(new cv.Mat(h, w, cv.CV_8UC1));
  outMask.data.set(md);
  return { mask: outMask, coverage, contrast, otsu };
}
