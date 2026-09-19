import { cvConfig } from '../config';
import { PipelineError } from '../errors';
import { binarizeCopper, type BinarizeResult } from './binarize';
import { decodeGrayRegion, decodeRgb, type Rect } from './image';
import { estimateLocalField, type LocalField } from './localAlign';
import { bytesOf, matFromRaw, scoped, type CV, type Mat, type MatScope } from './opencv';

/**
 * Registers a photograph of a board onto the reference raster (plan §5.2).
 *
 * The plan aligns on the panel's three fiducials. That cannot work as written: (a) a homography needs
 * >= 4 point correspondences, three give only an affine, and (b) in the shipped aoi.json every
 * fiducial and tooling/reference hole lies in a panel rail OUTSIDE every board placement, so a
 * single-board frame (which plan §6 requires for resolution) contains none of them. So instead:
 *
 *   1. coarse:   board quad - operator-tapped corners, else the copper-orange outline of the panel
 *   2. orient:   try the four 90-degree rotations, scored by correlation against the reference under
 *                both copper polarities (Android can hand us a sideways image with no EXIF tag)
 *   3. refine:   ECC homography between the reference and the binarised capture
 *   4. local:    per-tile translation field for board bow / lens distortion (see localAlign.ts)
 *   5. gate:     refuse (ALIGNMENT_FAILED) if the aligned correlation is still poor
 */

export type Pt = [number, number];
/** Clockwise (in image coordinates, y down): top-left, top-right, bottom-right, bottom-left. */
export type Quad = [Pt, Pt, Pt, Pt];
type Mat3 = number[]; // row-major 3x3

export interface AlignInput {
  capturePath: string;
  reference: { maskPath: string; copperValue: 0 | 1; pxPerMm: number };
  /** The reference region the capture covers (a placement, or the whole raster), in reference pixels. */
  target: Rect;
  /** Operator-tapped corners as fractions (0-1) of the oriented capture: the board's four corners in ANY order
   *  (they are sorted clockwise here and the orientation is resolved by correlation, exactly as for an
   *  auto-detected outline - a phone photo of a portrait board is often stored sideways, so asking the
   *  operator for "reference orientation" would be unusable). */
  corners?: Quad | undefined;
}

export interface AlignmentResult {
  method: 'manual-corners' | 'auto-outline';
  orientation: number;
  /** The reference raster's actual copper polarity is the opposite of its declared copperValue. */
  referenceInverted: boolean;
  /** Correlation between the aligned capture's copper mask and the reference (1 = identical). */
  score: number;
  eccApplied: boolean;
  /** Non-rigid correction applied on top of the homography (board bow / lens distortion). */
  localCorrection: { maxMm: number; meanMm: number; tilesUsed: number; tilesTotal: number } | null;
  workPxPerMm: number;
  /** Measured from the capture's own geometry - never copied from the reference. */
  effectivePxPerMm: number;
  work: { width: number; height: number };
  /** capture (decoded) px -> working px */
  homography: Mat3;
  warnings: string[];
  warpedRgb: Mat;
  valid: Mat;
  captureMask: Mat;
  captureCoverage: number;
  /** Reference in the working frame, copper = 255, polarity-corrected. */
  golden: Uint8Array;
}

// ---------------------------------------------------------------------------------------------
// small linear-algebra helpers (3x3 row-major)
// ---------------------------------------------------------------------------------------------
const mul3 = (a: Mat3, b: Mat3): Mat3 => {
  const o = new Array<number>(9).fill(0);
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) for (let k = 0; k < 3; k++) o[i * 3 + j]! += a[i * 3 + k]! * b[k * 3 + j]!;
  return o;
};
const inv3 = (m: Mat3): Mat3 => {
  const [a, b, c, d, e, f, g, h, i] = m as [number, number, number, number, number, number, number, number, number];
  const A = e * i - f * h;
  const B = -(d * i - f * g);
  const C = d * h - e * g;
  const det = a * A + b * B + c * C;
  if (Math.abs(det) < 1e-14) throw new PipelineError('ALIGNMENT_FAILED', 'Degenerate alignment transform');
  const s = 1 / det;
  return [A * s, -(b * i - c * h) * s, (b * f - c * e) * s, B * s, (a * i - c * g) * s, -(a * f - c * d) * s, C * s, -(a * h - b * g) * s, (a * e - b * d) * s];
};
const scale3 = (s: number): Mat3 => [s, 0, 0, 0, s, 0, 0, 0, 1];
const apply = (m: Mat3, x: number, y: number): Pt => {
  const w = m[6]! * x + m[7]! * y + m[8]!;
  return [(m[0]! * x + m[1]! * y + m[2]!) / w, (m[3]! * x + m[4]! * y + m[5]!) / w];
};
export const polyArea = (p: Pt[]): number => {
  let a = 0;
  for (let i = 0; i < p.length; i++) {
    const [x1, y1] = p[i]!;
    const [x2, y2] = p[(i + 1) % p.length]!;
    a += x1 * y2 - x2 * y1;
  }
  return Math.abs(a) / 2;
};
export const rotateQuad = (q: Quad, k: number): Quad => [0, 1, 2, 3].map((i) => q[(i + k) % 4]!) as Quad;

/** Cyclic clockwise order (image coords), starting at the point nearest the top-left. */
export function orderClockwise(pts: Pt[]): Quad {
  const cx = pts.reduce((s, p) => s + p[0], 0) / pts.length;
  const cy = pts.reduce((s, p) => s + p[1], 0) / pts.length;
  const sorted = [...pts].sort((a, b) => Math.atan2(a[1] - cy, a[0] - cx) - Math.atan2(b[1] - cy, b[0] - cx));
  let start = 0;
  sorted.forEach((p, i) => {
    if (p[0] + p[1] < sorted[start]![0] + sorted[start]![1]) start = i;
  });
  return [0, 1, 2, 3].map((i) => sorted[(start + i) % 4]!) as Quad;
}

// ---------------------------------------------------------------------------------------------
// OpenCV glue. Every helper frees its own temporaries; only results join `out`.
// ---------------------------------------------------------------------------------------------
export function homographyFromQuads(cv: CV, src: Quad, dst: Quad): Mat3 {
  return scoped((s) => {
    const a = s.add(cv.matFromArray(4, 1, cv.CV_32FC2, src.flat()));
    const b = s.add(cv.matFromArray(4, 1, cv.CV_32FC2, dst.flat()));
    return Array.from(s.add(cv.getPerspectiveTransform(a, b)).data64F);
  });
}

export function warpRgb(cv: CV, out: MatScope, src: Mat, h: Mat3, width: number, height: number): { rgb: Mat; valid: Mat } {
  return scoped((tmp) => {
    const hm = tmp.add(cv.matFromArray(3, 3, cv.CV_64FC1, h));
    const size = new cv.Size(width, height);
    const rgb = out.add(new cv.Mat());
    cv.warpPerspective(src, rgb, hm, size, cv.INTER_LINEAR, cv.BORDER_CONSTANT, new cv.Scalar(0, 0, 0, 0));
    const ones = tmp.add(new cv.Mat(src.rows, src.cols, cv.CV_8UC1, new cv.Scalar(255)));
    const valid = out.add(new cv.Mat());
    cv.warpPerspective(ones, valid, hm, size, cv.INTER_NEAREST, cv.BORDER_CONSTANT, new cv.Scalar(0));
    return { rgb, valid };
  });
}

/** Warp with the homography plus a per-pixel displacement field: aligned(x) = capture(back(x + d(x))). */
function remapWithField(cv: CV, out: MatScope, src: Mat, back: Mat3, field: LocalField, width: number, height: number): { rgb: Mat; valid: Mat } {
  const n = width * height;
  const mapx = new Float32Array(n);
  const mapy = new Float32Array(n);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = y * width + x;
      const X = x + field.dx[i]!;
      const Y = y + field.dy[i]!;
      const wgt = back[6]! * X + back[7]! * Y + back[8]!;
      mapx[i] = (back[0]! * X + back[1]! * Y + back[2]!) / wgt;
      mapy[i] = (back[3]! * X + back[4]! * Y + back[5]!) / wgt;
    }
  }
  return scoped((tmp) => {
    const mx = tmp.add(new cv.Mat(height, width, cv.CV_32FC1));
    mx.data32F.set(mapx);
    const my = tmp.add(new cv.Mat(height, width, cv.CV_32FC1));
    my.data32F.set(mapy);
    const rgb = out.add(new cv.Mat());
    cv.remap(src, rgb, mx, my, cv.INTER_LINEAR, cv.BORDER_CONSTANT, new cv.Scalar(0, 0, 0, 0));
    const ones = tmp.add(new cv.Mat(src.rows, src.cols, cv.CV_8UC1, new cv.Scalar(255)));
    const valid = out.add(new cv.Mat());
    cv.remap(ones, valid, mx, my, cv.INTER_NEAREST, cv.BORDER_CONSTANT, new cv.Scalar(0));
    return { rgb, valid };
  });
}

/** Pearson correlation of two same-size binary masks after blurring, over valid pixels. */
function maskCorrelation(cv: CV, a: Uint8Array, b: Uint8Array, valid: Uint8Array | null, w: number, h: number, sigma: number): number {
  const blurred = (data: Uint8Array): Float32Array =>
    scoped((s) => {
      const src = s.add(new cv.Mat(h, w, cv.CV_8UC1));
      src.data.set(data);
      const f = s.add(new cv.Mat());
      src.convertTo(f, cv.CV_32F, 1 / 255);
      const out = s.add(new cv.Mat());
      cv.GaussianBlur(f, out, new cv.Size(0, 0), sigma, sigma, cv.BORDER_REPLICATE);
      return Float32Array.from(out.data32F);
    });
  const A = blurred(a);
  const B = blurred(b);
  let n = 0;
  let sa = 0;
  let sb = 0;
  for (let i = 0; i < A.length; i++) {
    if (valid && !valid[i]) continue;
    n++;
    sa += A[i]!;
    sb += B[i]!;
  }
  if (n === 0) return 0;
  const ma = sa / n;
  const mb = sb / n;
  let saa = 0;
  let sbb = 0;
  let sab = 0;
  for (let i = 0; i < A.length; i++) {
    if (valid && !valid[i]) continue;
    const da = A[i]! - ma;
    const db = B[i]! - mb;
    saa += da * da;
    sbb += db * db;
    sab += da * db;
  }
  const denom = Math.sqrt(saa * sbb);
  return denom < 1e-9 ? 0 : sab / denom;
}

// ---------------------------------------------------------------------------------------------
// coarse board outline
// ---------------------------------------------------------------------------------------------
/**
 * Bare copper is strongly saturated orange (measured on a real photo: H~18, S~150, V~235) while the
 * wood desk it lay on shares the hue but is far less saturated (S~55-70), jeans are blue and the
 * floor is grey. Opening removes thin wood-grain streaks that survive the colour gate; closing joins
 * the trace network and rails into one blob; the largest blob is the panel.
 *
 * NOTE this is specific to bare copper. For a solder-masked board the outline is not orange - use
 * operator-tapped corners.
 */
export function detectOutlineQuad(cv: CV, scope: MatScope, rgb: Mat): Quad {
  const hsv = scope.add(new cv.Mat());
  cv.cvtColor(rgb, hsv, cv.COLOR_RGB2HSV);
  const lo = scope.add(new cv.Mat(hsv.rows, hsv.cols, hsv.type(), new cv.Scalar(4, 100, 110, 0)));
  const hi = scope.add(new cv.Mat(hsv.rows, hsv.cols, hsv.type(), new cv.Scalar(32, 255, 255, 255)));
  const mask = scope.add(new cv.Mat());
  cv.inRange(hsv, lo, hi, mask);

  const longSide = Math.max(rgb.cols, rgb.rows);
  // CLOSE first, then open. Opening first (the original order) erased every thin trace, so a board whose
  // copper is thin traces plus a narrow border - rather than broad copper margins - collapsed to a
  // ragged handful of rails and was rejected (solidity 0.09). Closing joins the trace network into one
  // region; the larger opening afterwards then strips thin wood-grain streaks that merged into it.
  const kClose = Math.max(9, Math.round(longSide / 60) | 1);
  const kOpen = Math.max(5, Math.round(longSide / 100) | 1);
  const tmp = scope.add(new cv.Mat());
  cv.morphologyEx(mask, tmp, cv.MORPH_CLOSE, scope.add(cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(kClose, kClose))));
  cv.morphologyEx(tmp, mask, cv.MORPH_OPEN, scope.add(cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(kOpen, kOpen))));

  const contours = scope.add(new cv.MatVector());
  const hierarchy = scope.add(new cv.Mat());
  cv.findContours(mask, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
  let best = -1;
  let bestArea = 0;
  for (let i = 0; i < contours.size(); i++) {
    const c = contours.get(i);
    const a = cv.contourArea(c);
    c.delete();
    if (a > bestArea) {
      bestArea = a;
      best = i;
    }
  }
  const frameArea = rgb.cols * rgb.rows;
  if (best < 0 || bestArea < 0.03 * frameArea) {
    throw new PipelineError(
      'ALIGNMENT_FAILED',
      'Could not find the board in the photo (no large copper-coloured region). Fill the frame with the board on a plain surface, or tap its four corners.',
    );
  }

  const contour = scope.add(contours.get(best));
  const hull = scope.add(new cv.Mat());
  cv.convexHull(contour, hull, false, true);
  const hullArea = cv.contourArea(hull);
  const solidity = bestArea / Math.max(1, hullArea);
  if (solidity < 0.8) {
    throw new PipelineError('ALIGNMENT_FAILED', `The detected board outline is ragged (solidity ${solidity.toFixed(2)}) - likely merged with background. Tap the four corners instead.`);
  }

  const perim = cv.arcLength(hull, true);
  let quad: Pt[] | null = null;
  for (let eps = 0.01; eps <= 0.09 && !quad; eps += 0.005) {
    const approx = scope.add(new cv.Mat());
    cv.approxPolyDP(hull, approx, eps * perim, true);
    if (approx.rows === 4) {
      const d = approx.data32S;
      quad = [0, 1, 2, 3].map((i) => [d[i * 2]!, d[i * 2 + 1]!] as Pt);
    }
  }
  if (!quad) {
    const rect = cv.minAreaRect(hull);
    quad = (cv as unknown as { RotatedRect: { points(r: unknown): { x: number; y: number }[] } }).RotatedRect.points(rect).map((p) => [p.x, p.y] as Pt);
  }
  return orderClockwise(quad);
}

interface Evaluated {
  rgb: Mat;
  valid: Mat;
  bin: BinarizeResult;
  score: number;
}

// ---------------------------------------------------------------------------------------------
// main entry
// ---------------------------------------------------------------------------------------------
export async function alignCapture(cv: CV, scope: MatScope, input: AlignInput): Promise<AlignmentResult> {
  const warnings: string[] = [];
  const G = input.reference.pxPerMm;
  const T = input.target;

  // ---- 1. coarse quad on a preview ----------------------------------------------------------
  const preview = await decodeRgb(input.capturePath, 1400);
  const srcW = preview.sourceWidth;
  const srcH = preview.sourceHeight;
  let quadSrc: Quad; // in oriented-source pixels
  let method: AlignmentResult['method'];
  if (input.corners) {
    quadSrc = orderClockwise(input.corners.map(([fx, fy]) => [fx * srcW, fy * srcH] as Pt));
    method = 'manual-corners';
  } else {
    const q = scoped((s) => detectOutlineQuad(cv, s, matFromRaw(cv, s, preview.data, preview.width, preview.height, 3)));
    quadSrc = q.map(([x, y]) => [x / preview.scale, y / preview.scale] as Pt) as Quad;
    method = 'auto-outline';
  }

  // ---- 2. resolution: measured from the quad, never assumed --------------------------------
  const areaMm2 = (T.w / G) * (T.h / G);
  const effNative = Math.sqrt(polyArea(quadSrc) / areaMm2);
  if (effNative < cvConfig.refusePxPerMm) {
    throw new PipelineError(
      'RESOLUTION_TOO_LOW',
      `The board is only ${effNative.toFixed(1)} px/mm in this photo (minimum ${cvConfig.refusePxPerMm}). Move the camera closer / use the highest-resolution photo mode.`,
      { effectivePxPerMm: effNative },
    );
  }
  if (effNative < cvConfig.minPxPerMm) {
    warnings.push(
      `Low resolution: ${effNative.toFixed(1)} px/mm (< ${cvConfig.minPxPerMm}). Features below ~${(2 / effNative).toFixed(2)} mm cannot be resolved, so a pass here is weak evidence.`,
    );
  }
  const workPxPerMm = Math.min(effNative, G, cvConfig.maxWorkPxPerMm);
  const ws = workPxPerMm / G;
  const work = { width: Math.max(8, Math.round(T.w * ws)), height: Math.max(8, Math.round(T.h * ws)) };

  // decode the capture only as large as it needs to be (1.5x oversampled vs. the working frame)
  const capScale = Math.min(1, (workPxPerMm * 1.5) / effNative);
  const cap = await decodeRgb(input.capturePath, Math.round(Math.max(srcW, srcH) * capScale));
  const capMat = matFromRaw(cv, scope, cap.data, cap.width, cap.height, 3);
  const quadCap = quadSrc.map(([x, y]) => [x * cap.scale, y * cap.scale] as Pt) as Quad;
  const dstQuad = (s: number): Quad => [
    [0, 0],
    [work.width * s, 0],
    [work.width * s, work.height * s],
    [0, work.height * s],
  ];

  // ---- 3. orientation + polarity, scored at low resolution ---------------------------------
  const lowPxPerMm = Math.min(workPxPerMm, 5);
  const ls = lowPxPerMm / workPxPerMm;
  const low = { width: Math.max(8, Math.round(work.width * ls)), height: Math.max(8, Math.round(work.height * ls)) };
  const declared = input.reference;
  const goldLow = await decodeGrayRegion(declared.maskPath, { rect: T, outWidth: low.width, outHeight: low.height, copperValue: declared.copperValue });
  // pre-shrink the capture so the low-resolution warp does not alias
  const r = Math.min(1, (lowPxPerMm * 1.5) / (effNative * cap.scale));
  const capLow = scope.add(new cv.Mat());
  cv.resize(capMat, capLow, new cv.Size(Math.max(8, Math.round(cap.width * r)), Math.max(8, Math.round(cap.height * r))), 0, 0, cv.INTER_AREA);
  const rActual = capLow.cols / cap.width;

  const candidates: { k: number; corr: number }[] = [];
  const ks = [0, 1, 2, 3];
  for (const k of ks) {
    const corr = scoped((tmp) => {
      try {
        const h = homographyFromQuads(cv, rotateQuad(quadCap, k).map(([x, y]) => [x * rActual, y * rActual] as Pt) as Quad, dstQuad(ls));
        const { rgb, valid } = warpRgb(cv, tmp, capLow, h, low.width, low.height);
        const bin = binarizeCopper(cv, tmp, rgb, { pxPerMm: lowPxPerMm, valid });
        return maskCorrelation(cv, goldLow.data, bytesOf(bin.mask), bytesOf(valid), low.width, low.height, 1.5);
      } catch (e) {
        if (!(e instanceof PipelineError)) throw e;
        return 0; // e.g. a wrong-orientation warp that binarises to nonsense
      }
    });
    candidates.push({ k, corr });
  }
  scope.free(capLow);
  candidates.sort((a, b) => Math.abs(b.corr) - Math.abs(a.corr));
  const bestCand = candidates[0]!;
  if (Math.abs(bestCand.corr) < 0.12) {
    throw new PipelineError(
      'ALIGNMENT_FAILED',
      `The photo does not match the reference in any orientation (best correlation ${bestCand.corr.toFixed(2)}). Wrong board/side, badly out of frame, or too blurry.`,
      { correlations: candidates },
    );
  }
  const orientation = bestCand.k;
  const referenceInverted = bestCand.corr < 0;
  if (referenceInverted) {
    warnings.push(
      `The reference raster's copper polarity is the OPPOSITE of its declared copperValue=${declared.copperValue} ` +
        `(correlation ${bestCand.corr.toFixed(2)} against the photographed copper). Using the measured polarity. ` +
        "If this is a Gerber-derived reference, the AOI export's copperValue disagrees with its own raster.",
    );
  }

  // ---- 4. full working-resolution warp + ECC refinement ------------------------------------
  const goldWork = await decodeGrayRegion(declared.maskPath, {
    rect: T,
    outWidth: work.width,
    outHeight: work.height,
    copperValue: referenceInverted ? ((1 - declared.copperValue) as 0 | 1) : declared.copperValue,
  });
  const golden = new Uint8Array(goldWork.data);

  const evaluate = (rgb: Mat, valid: Mat): Evaluated => {
    const bin = binarizeCopper(cv, scope, rgb, { pxPerMm: workPxPerMm, valid });
    const score = maskCorrelation(cv, golden, bytesOf(bin.mask), bytesOf(valid), work.width, work.height, 1.0);
    return { rgb, valid, bin, score };
  };
  const finalize = (h: Mat3): Evaluated => {
    const { rgb, valid } = warpRgb(cv, scope, capMat, h, work.width, work.height);
    return evaluate(rgb, valid);
  };
  const discard = (e: Evaluated) => {
    scope.free(e.rgb);
    scope.free(e.valid);
    scope.free(e.bin.mask);
  };

  let H = homographyFromQuads(cv, rotateQuad(quadCap, orientation), dstQuad(1));
  let cur = finalize(H);
  let eccApplied = false;

  try {
    const Hnew = scoped((tmp) => {
      const se = Math.min(1, 1000 / Math.max(work.width, work.height));
      const ew = Math.max(8, Math.round(work.width * se));
      const eh = Math.max(8, Math.round(work.height * se));
      const toF = (data: Uint8Array): Mat => {
        const m8 = tmp.add(new cv.Mat(work.height, work.width, cv.CV_8UC1));
        m8.data.set(data);
        const small = tmp.add(new cv.Mat());
        cv.resize(m8, small, new cv.Size(ew, eh), 0, 0, cv.INTER_AREA);
        const f = tmp.add(new cv.Mat());
        small.convertTo(f, cv.CV_32F, 1 / 255);
        const b = tmp.add(new cv.Mat());
        cv.GaussianBlur(f, b, new cv.Size(0, 0), 1.5, 1.5, cv.BORDER_REPLICATE);
        return b;
      };
      const template = toF(golden);
      const input2 = toF(bytesOf(cur.bin.mask));
      const validSmall = tmp.add(new cv.Mat());
      cv.resize(cur.valid, validSmall, new cv.Size(ew, eh), 0, 0, cv.INTER_NEAREST);
      const warp = tmp.add(cv.Mat.eye(3, 3, cv.CV_32F));
      const criteria = new cv.TermCriteria(cv.TermCriteria_COUNT + cv.TermCriteria_EPS, 120, 1e-5);
      cv.findTransformECC(template, input2, warp, cv.MOTION_HOMOGRAPHY, criteria, validSmall, 5);
      const wSmall: Mat3 = Array.from(warp.data32F);
      // ECC coordinates are at scale `se`; bring the warp back to working-frame coordinates.
      const wWork = mul3(scale3(1 / se), mul3(wSmall, scale3(se)));
      // aligned(x) = warped(W x)  =>  H_total = W^-1 * H
      return mul3(inv3(wWork), H);
    });
    const refined = finalize(Hnew);
    if (refined.score >= cur.score) {
      discard(cur);
      H = Hnew;
      cur = refined;
      eccApplied = true;
    } else {
      warnings.push(`ECC refinement did not improve the match (${cur.score.toFixed(3)} -> ${refined.score.toFixed(3)}); kept the coarse alignment.`);
      discard(refined);
    }
  } catch (e) {
    if (e instanceof PipelineError) throw e;
    warnings.push('ECC refinement did not converge; using the coarse outline alignment.');
  }

  // ---- 4b. non-rigid refinement (board bow / lens distortion) --------------------------------
  let localCorrection: AlignmentResult['localCorrection'] = null;
  try {
    const field = estimateLocalField(cv, golden, bytesOf(cur.bin.mask), bytesOf(cur.valid), work.width, work.height, workPxPerMm);
    if (field) {
      const mapped = remapWithField(cv, scope, capMat, inv3(H), field, work.width, work.height);
      const refined = evaluate(mapped.rgb, mapped.valid);
      if (refined.score >= cur.score - 0.002) {
        discard(cur);
        cur = refined;
        localCorrection = {
          maxMm: field.maxPx / workPxPerMm,
          meanMm: field.meanPx / workPxPerMm,
          tilesUsed: field.validTiles,
          tilesTotal: field.totalTiles,
        };
        if (localCorrection.maxMm > 0.6) {
          warnings.push(
            `The board is not flat or the lens distorts: local corrections of up to ${localCorrection.maxMm.toFixed(2)} mm were needed. ` +
              'Detections in the most-corrected areas are less reliable - lay the board flat on a rigid surface.',
          );
        }
      } else {
        warnings.push(`Local refinement did not improve the match (${cur.score.toFixed(3)} -> ${refined.score.toFixed(3)}); kept the global alignment.`);
        discard(refined);
      }
    }
  } catch (e) {
    if (e instanceof PipelineError) throw e;
    warnings.push('Local refinement failed; using the global alignment only.');
  }

  if (cur.score < cvConfig.minAlignmentScore) {
    throw new PipelineError(
      'ALIGNMENT_FAILED',
      `Alignment quality is too low to trust (correlation ${cur.score.toFixed(2)}, need >= ${cvConfig.minAlignmentScore}). Retake the photo flat, in focus and fully in frame, or tap the four corners.`,
      { score: cur.score },
    );
  }
  scope.free(capMat); // the decoded capture is no longer needed once the aligned image exists

  // ---- 5. measured resolution: local scale of the final mapping at the frame centre ---------
  const back = inv3(H); // working -> capture(decoded)
  const [cx, cy] = [work.width / 2, work.height / 2];
  const p0 = apply(back, cx, cy);
  const px = apply(back, cx + 1, cy);
  const py = apply(back, cx, cy + 1);
  const jdet = Math.abs((px[0] - p0[0]) * (py[1] - p0[1]) - (px[1] - p0[1]) * (py[0] - p0[0]));
  const capPxPerWorkPx = Math.sqrt(jdet);
  const effectivePxPerMm = (capPxPerWorkPx * workPxPerMm) / cap.scale;

  return {
    method,
    orientation,
    referenceInverted,
    score: cur.score,
    eccApplied,
    localCorrection,
    workPxPerMm,
    effectivePxPerMm,
    work,
    homography: H,
    warnings,
    warpedRgb: cur.rgb,
    valid: cur.valid,
    captureMask: cur.bin.mask,
    captureCoverage: cur.bin.coverage,
    golden,
  };
}
