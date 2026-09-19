import { MatScope, type CV, type Mat } from './opencv';

/**
 * Non-rigid refinement on top of the global homography.
 *
 * Measured on a real photo: after the global homography + ECC the lower half of the panel was
 * within 1 px but the upper half was off by 2-10 px (0.2-1 mm) and varied smoothly - the panel was
 * bowed (it hung over the edge of the desk). A homography cannot express that, and a tolerance of a
 * few tenths of a millimetre cannot absorb it: every trace edge in the affected area became a
 * "defect" (1,400 of them). Real boards are never perfectly flat and lenses are never perfect, so this
 * stage is needed on real captures, not just on that one.
 *
 * Method: a translation per tile against the reference (ECC), then a robust, smooth displacement
 * field. Translation-only per tile keeps the degrees of freedom far too low to "align away" a real
 * defect (a 1 mm^2 defect inside a ~400 mm^2 tile barely changes the correlation), and the field is
 * clamped, median-filtered and smoothed so a single bad tile cannot tear the image.
 */

export interface LocalField {
  /** Displacement in WORKING pixels: aligned(x) = warped(x + d(x)). Full working-frame size. */
  dx: Float32Array;
  dy: Float32Array;
  width: number;
  height: number;
  maxPx: number;
  meanPx: number;
  validTiles: number;
  totalTiles: number;
}

const ANALYSIS_PX_PER_MM = 10;
const TILE_MM = 20;
const MAX_SHIFT_MM = 0.8;
const MIN_TILE_CC = 0.4;
const MIN_VALID_FRACTION = 0.2;

/** Temporaries are freed on return; only plain arrays escape. */
export function estimateLocalField(
  cv: CV,
  golden: Uint8Array,
  capture: Uint8Array,
  valid: Uint8Array,
  w: number,
  h: number,
  workPxPerMm: number,
): LocalField | null {
  const scope = new MatScope();
  try {
    return estimateInner(cv, scope, golden, capture, valid, w, h, workPxPerMm);
  } finally {
    scope.dispose();
  }
}

function estimateInner(
  cv: CV,
  scope: MatScope,
  golden: Uint8Array,
  capture: Uint8Array,
  valid: Uint8Array,
  w: number,
  h: number,
  workPxPerMm: number,
): LocalField | null {
  const ds = Math.max(1, workPxPerMm / ANALYSIS_PX_PER_MM);
  const aw = Math.max(8, Math.round(w / ds));
  const ah = Math.max(8, Math.round(h / ds));
  const apm = workPxPerMm / ds; // analysis px per mm

  const toAnalysis = (data: Uint8Array, blur: boolean): Mat => {
    const m8 = scope.add(new cv.Mat(h, w, cv.CV_8UC1));
    m8.data.set(data);
    const small = scope.add(new cv.Mat());
    cv.resize(m8, small, new cv.Size(aw, ah), 0, 0, cv.INTER_AREA);
    if (!blur) return small;
    const f = scope.add(new cv.Mat());
    small.convertTo(f, cv.CV_32F, 1 / 255);
    const b = scope.add(new cv.Mat());
    cv.GaussianBlur(f, b, new cv.Size(0, 0), 1.0, 1.0, cv.BORDER_REPLICATE);
    return b;
  };
  const gA = toAnalysis(golden, true);
  const cA = toAnalysis(capture, true);
  const validA = Uint8Array.from(toAnalysis(valid, false).data);

  const ts = Math.max(80, Math.round(TILE_MM * apm));
  const cols = Math.max(1, Math.floor(aw / ts));
  const rows = Math.max(1, Math.floor(ah / ts));
  const tw = Math.floor(aw / cols);
  const th = Math.floor(ah / rows);
  const total = cols * rows;

  const tdx = new Float32Array(total);
  const tdy = new Float32Array(total);
  const ok = new Uint8Array(total);

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const x0 = c * tw;
      const y0 = r * th;
      // skip tiles that are mostly outside the captured area
      let vc = 0;
      for (let y = y0; y < y0 + th; y += 2) for (let x = x0; x < x0 + tw; x += 2) if (validA[y * aw + x]) vc++;
      if (vc / ((tw / 2) * (th / 2)) < 0.85) continue;

      const rect = new cv.Rect(x0, y0, tw, th);
      const gt = scope.add(gA.roi(rect).clone());
      const ct = scope.add(cA.roi(rect).clone());
      // a tile with (almost) no structure in the reference cannot be registered
      const mean = new cv.Mat();
      const std = new cv.Mat();
      cv.meanStdDev(gt, mean, std);
      const sd = std.data64F[0]!;
      mean.delete();
      std.delete();
      if (sd < 0.08) continue;

      try {
        const warp = scope.add(cv.Mat.eye(2, 3, cv.CV_32F));
        const criteria = new cv.TermCriteria(cv.TermCriteria_COUNT + cv.TermCriteria_EPS, 80, 1e-4);
        const cc = cv.findTransformECC(gt, ct, warp, cv.MOTION_TRANSLATION, criteria, scope.add(new cv.Mat()), 5);
        if (cc >= MIN_TILE_CC) {
          const d = warp.data32F;
          const i = r * cols + c;
          tdx[i] = d[2]!;
          tdy[i] = d[5]!;
          ok[i] = 1;
        }
      } catch {
        /* did not converge: leave this tile invalid */
      }
    }
  }

  const validTiles = ok.reduce((s, v) => s + v, 0);
  if (validTiles < Math.max(2, MIN_VALID_FRACTION * total)) return null;

  // ---- clean the coarse field: fill gaps, reject outliers, clamp, smooth --------------------
  fillGaps(tdx, tdy, ok, cols, rows);
  medianFilter(tdx, cols, rows);
  medianFilter(tdy, cols, rows);
  const maxShiftA = MAX_SHIFT_MM * apm;
  for (let i = 0; i < total; i++) {
    tdx[i] = Math.max(-maxShiftA, Math.min(maxShiftA, tdx[i]!));
    tdy[i] = Math.max(-maxShiftA, Math.min(maxShiftA, tdy[i]!));
  }
  smooth(tdx, cols, rows);
  smooth(tdy, cols, rows);

  // ---- upsample to the working frame (tile centres are the sample points) ---------------------
  const upsample = (coarse: Float32Array): Float32Array => {
    const src = scope.add(new cv.Mat(rows, cols, cv.CV_32FC1));
    src.data32F.set(coarse);
    const dst = scope.add(new cv.Mat());
    cv.resize(src, dst, new cv.Size(w, h), 0, 0, cv.INTER_LINEAR);
    const out = Float32Array.from(dst.data32F);
    for (let i = 0; i < out.length; i++) out[i]! *= ds; // analysis px -> working px
    return out;
  };
  const dx = upsample(tdx);
  const dy = upsample(tdy);

  let maxPx = 0;
  let sum = 0;
  for (let i = 0; i < total; i++) {
    const m = Math.hypot(tdx[i]!, tdy[i]!) * ds;
    maxPx = Math.max(maxPx, m);
    sum += m;
  }
  return { dx, dy, width: w, height: h, maxPx, meanPx: sum / total, validTiles, totalTiles: total };
}

/** Replace invalid cells with the mean of their valid neighbours, repeating until filled. */
function fillGaps(dx: Float32Array, dy: Float32Array, ok: Uint8Array, cols: number, rows: number): void {
  const filled = Uint8Array.from(ok);
  for (let pass = 0; pass < cols + rows; pass++) {
    let changed = false;
    const next = Uint8Array.from(filled);
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const i = r * cols + c;
        if (filled[i]) continue;
        let sx = 0;
        let sy = 0;
        let n = 0;
        for (let rr = Math.max(0, r - 1); rr <= Math.min(rows - 1, r + 1); rr++) {
          for (let cc = Math.max(0, c - 1); cc <= Math.min(cols - 1, c + 1); cc++) {
            const j = rr * cols + cc;
            if (filled[j]) {
              sx += dx[j]!;
              sy += dy[j]!;
              n++;
            }
          }
        }
        if (n > 0) {
          dx[i] = sx / n;
          dy[i] = sy / n;
          next[i] = 1;
          changed = true;
        }
      }
    }
    filled.set(next);
    if (!changed) break;
  }
}

function medianFilter(a: Float32Array, cols: number, rows: number): void {
  const src = Float32Array.from(a);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const vals: number[] = [];
      for (let rr = Math.max(0, r - 1); rr <= Math.min(rows - 1, r + 1); rr++)
        for (let cc = Math.max(0, c - 1); cc <= Math.min(cols - 1, c + 1); cc++) vals.push(src[rr * cols + cc]!);
      vals.sort((x, y) => x - y);
      a[r * cols + c] = vals[vals.length >> 1]!;
    }
  }
}

function smooth(a: Float32Array, cols: number, rows: number): void {
  const src = Float32Array.from(a);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      let s = 0;
      let wsum = 0;
      for (let rr = Math.max(0, r - 1); rr <= Math.min(rows - 1, r + 1); rr++) {
        for (let cc = Math.max(0, c - 1); cc <= Math.min(cols - 1, c + 1); cc++) {
          const wgt = (rr === r ? 2 : 1) * (cc === c ? 2 : 1);
          s += wgt * src[rr * cols + cc]!;
          wsum += wgt;
        }
      }
      a[r * cols + c] = s / wsum;
    }
  }
}
