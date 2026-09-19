import fs from 'fs';
import path from 'path';
import sharp from 'sharp';
import { cvConfig } from '../config';
import { alignCapture, type AlignmentResult, type Quad } from './align';
import { speckleKernel } from './binarize';
import { decodeGrayRegion, writePng, type Rect } from './image';
import { fitCaptureModel } from './model';
import { bytesOf, matFromRaw, scoped, withCv, type CV, type Mat, type MatScope } from './opencv';

export type DefectKind = 'open' | 'short' | 'unclassified';

export interface DefectResult {
  /** In the reference raster's pixel frame. */
  bboxPx: { x: number; y: number; w: number; h: number };
  /** Millimetres from the top-left of the inspected region, y down. */
  bboxMm: { x: number; y: number; w: number; h: number };
  kind: DefectKind;
  /** Blob area in the aligned working frame (see workPxPerMm). */
  areaPx: number;
  areaMm2: number;
  /** Twice the largest inscribed radius: how thick the blob really is (a sliver along an edge is thin). */
  thicknessMm: number;
  /** Thicker than this capture's own detection limit. Thinner blobs are edge-misregistration artifacts. */
  credible: boolean;
  confidence: number;
}

export interface InspectionInput {
  capturePath: string;
  reference: {
    maskPath: string;
    drillMaskPath?: string | null;
    copperValue: 0 | 1;
    pxPerMm: number;
    source: string;
  };
  /** Region of the reference the capture covers: a placement rect, or the whole raster. */
  target: Rect;
  corners?: Quad | undefined;
  /** Absolute path prefix (no extension) for the artifacts this run writes. */
  outputStem: string;
}

export interface CVInspectionResult {
  verdict: 'pass' | 'fail' | 'needs_review';
  effectivePxPerMm: number;
  workPxPerMm: number;
  smallestDetectableMm: number;
  toleranceMm: number;
  alignmentScore: number;
  alignmentMethod: AlignmentResult['method'];
  localCorrection: AlignmentResult['localCorrection'];
  /** Global trace-width offset per side in mm: >0 means the capture's copper is wider than the reference. */
  widthBiasMm: number;
  /** Optical softness of the capture (Gaussian sigma) in mm. */
  opticalBlurMm: number;
  /** Mismatch (fraction of area) against the raw reference vs. against the fitted expectation. */
  modelMismatch: { before: number; after: number };
  /** Spread of the per-tile fit: how uneven the capture is (focus, bow) across the board. */
  modelSpread: { biasMm: [number, number]; blurMm: [number, number]; grid: { cols: number; rows: number } };
  defects: DefectResult[];
  omittedDefects: number;
  maskPath: string;
  previewPath: string;
  warnings: string[];
}

/** A blob must be this many times thicker than the capture's detection limit to be believed: one AT the limit is indistinguishable from noise. */
const CREDIBLE_MARGIN = 1.5;
/** On a low-quality capture only a confirmed short, or a defect this many times the limit, may fail a board. */
const GROSS_MARGIN = 3;

const KIND_COLOR: Record<DefectKind, string> = { open: '#ef4444', short: '#f59e0b', unclassified: '#38bdf8' };

export async function runInspection(input: InspectionInput): Promise<CVInspectionResult> {
  return withCv(async (cv, scope) => {
    const G = input.reference.pxPerMm;
    const T = input.target;
    const align = await alignCapture(cv, scope, {
      capturePath: input.capturePath,
      reference: { maskPath: input.reference.maskPath, copperValue: input.reference.copperValue, pxPerMm: G },
      target: T,
      corners: input.corners,
    });
    const { width: w, height: h } = align.work;
    const wpm = align.workPxPerMm;
    const warnings = [...align.warnings];

    // ---- reference + capture masks in the working frame (copper = 255) ---------------------
    const gBin = scope.add(new cv.Mat());
    cv.threshold(matFromRaw(cv, scope, align.golden, w, h, 1), gBin, 127, 255, cv.THRESH_BINARY);
    const cap = align.captureMask;
    const valid = align.valid;

    // Drilled holes (plan §5.4): the copper raster draws every pad as a solid disc but the real board
    // has a hole through it, so without masking them out every drilled pad reads as missing copper.
    let ignore: Mat | null = null;
    if (input.reference.drillMaskPath) {
      const drill = await decodeGrayRegion(input.reference.drillMaskPath, {
        rect: T,
        outWidth: w,
        outHeight: h,
        // the drill mask uses the same encoding as the copper raster it accompanies
        copperValue: align.referenceInverted ? ((1 - input.reference.copperValue) as 0 | 1) : input.reference.copperValue,
      });
      const d = matFromRaw(cv, scope, drill.data, w, h, 1);
      const dBin = scope.add(new cv.Mat());
      cv.threshold(d, dBin, 127, 255, cv.THRESH_BINARY);
      ignore = scope.add(new cv.Mat());
      cv.dilate(dBin, ignore, scope.add(cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(5, 5))));
    }

    // ---- tolerance-band diff -----------------------------------------------------------------
    // Raw XOR flags every trace edge (etched edges are rough, registration is imperfect). Two-sided
    // band instead: copper that MUST be there (golden shrunk by the tolerance) but is absent, and
    // copper that MUST NOT be there (golden grown by the tolerance) but is present.
    // Fit the global effects (trace-width bias, optical blur) so they are not mistaken for local defects.
    const model = fitCaptureModel(cv, scope, gBin, cap, valid, wpm);
    const gExp = model.expected;
    const widthBiasMm = model.biasPx / wpm;
    const opticalBlurMm = model.sigmaPx / wpm;

    const tolPx = Math.max(cvConfig.toleranceMm * wpm, wpm < cvConfig.minPxPerMm ? 2 : 1);
    const k = Math.max(1, Math.round(tolPx));
    const tolKernel = scope.add(cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(2 * k + 1, 2 * k + 1)));
    const gEro = scope.add(new cv.Mat());
    const gDil = scope.add(new cv.Mat());
    cv.erode(gExp, gEro, tolKernel);
    cv.dilate(gExp, gDil, tolKernel);

    const notCap = scope.add(new cv.Mat());
    const notGDil = scope.add(new cv.Mat());
    cv.bitwise_not(cap, notCap);
    cv.bitwise_not(gDil, notGDil);
    const missing = scope.add(new cv.Mat());
    const excess = scope.add(new cv.Mat());
    cv.bitwise_and(gEro, notCap, missing);
    cv.bitwise_and(cap, notGDil, excess);
    cv.bitwise_and(missing, valid, missing);
    cv.bitwise_and(excess, valid, excess);
    if (ignore) {
      const notIgnore = scope.add(new cv.Mat());
      cv.bitwise_not(ignore, notIgnore);
      cv.bitwise_and(missing, notIgnore, missing);
      cv.bitwise_and(excess, notIgnore, excess);
    }
    for (const m of [gEro, gDil, notCap, notGDil, tolKernel]) scope.free(m);
    if (process.env.CV_DEBUG_DIR) {
      // Diagnostics: the fitted expectation and the raw diff maps, before component analysis.
      const dir = process.env.CV_DEBUG_DIR;
      const base = path.join(dir, path.basename(input.outputStem));
      await Promise.all([
        writePng(`${base}_dbg_expected.png`, bytesOf(gExp), w, h, 1),
        writePng(`${base}_dbg_missing.png`, bytesOf(missing), w, h, 1),
        writePng(`${base}_dbg_excess.png`, bytesOf(excess), w, h, 1),
      ]);
    }
    const kOpen = speckleKernel(wpm);
    if (kOpen > 1) {
      const ko = scope.add(cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(kOpen, kOpen)));
      for (const m of [missing, excess]) {
        const opened = scope.add(new cv.Mat());
        cv.morphologyEx(m, opened, cv.MORPH_OPEN, ko);
        opened.copyTo(m);
      }
    }

    // ---- connected components (plan §5.4) ----------------------------------------------------
    const blobs = (mask: Mat, keepLabels: boolean) => {
      const labels = scope.add(new cv.Mat());
      const stats = scope.add(new cv.Mat());
      const centroids = scope.add(new cv.Mat());
      const n = cv.connectedComponentsWithStats(mask, labels, stats, centroids, 8, cv.CV_32S);
      const s = Int32Array.from(stats.data32S);
      scope.free(stats);
      scope.free(centroids);
      // thickness of each blob = 2 * (largest inscribed radius) - 1, from a distance transform
      const rmax = scoped((t) => {
        const dt = t.add(new cv.Mat());
        cv.distanceTransform(mask, dt, cv.DIST_L2, cv.DIST_MASK_5);
        const d = dt.data32F;
        const lab = labels.data32S;
        const r = new Float32Array(n);
        for (let i = 0; i < d.length; i++) {
          const l = lab[i]!;
          if (l > 0 && d[i]! > r[l]!) r[l] = d[i]!;
        }
        return r;
      });
      if (!keepLabels) scope.free(labels);
      const out: { id: number; x: number; y: number; w: number; h: number; area: number; thickness: number }[] = [];
      for (let i = 1; i < n; i++) {
        const area = s[i * 5 + 4]!;
        if (area >= cvConfig.minDefectAreaPx) {
          out.push({ id: i, x: s[i * 5]!, y: s[i * 5 + 1]!, w: s[i * 5 + 2]!, h: s[i * 5 + 3]!, area, thickness: Math.max(1, 2 * rmax[i]! - 1) });
        }
      }
      return { labels, out };
    };
    const miss = blobs(missing, false);
    const exc = blobs(excess, true);

    // Net analysis: an excess blob that touches >= 2 distinct reference copper nets is a real short.
    const goldNets = scope.add(new cv.Mat());
    cv.connectedComponents(gBin, goldNets, 8, cv.CV_32S);
    const bridged = netsBridged(cv, excess, exc.labels, goldNets, w, h, k + 2);
    scope.free(exc.labels);
    scope.free(goldNets);

    // Without a drill mask, drilled pads read as round voids fully enclosed by reference copper.
    const suppressHoles =
      cvConfig.suppressEnclosedHoles === 'on' || (cvConfig.suppressEnclosedHoles === 'auto' && input.reference.source === 'gerber' && !input.reference.drillMaskPath);
    const gBytes = bytesOf(gBin);
    let suppressed = 0;

    // features narrower than ~2.5 sigma fuse in the capture and cannot be judged, however good the alignment
    const smallestDetectableMm = Math.max(Math.max(2, 2 * k + 1, kOpen) / wpm, (2.5 * model.sigmaPx) / wpm);

    const toDefect = (b: { x: number; y: number; w: number; h: number; area: number; thickness: number }, kind: DefectKind, baseConfidence: number): DefectResult => {
      const thicknessMm = b.thickness / wpm;
      const credible = thicknessMm >= CREDIBLE_MARGIN * smallestDetectableMm;
      return {
        bboxPx: {
        x: Math.round(T.x + (b.x * G) / wpm),
        y: Math.round(T.y + (b.y * G) / wpm),
        w: Math.max(1, Math.round((b.w * G) / wpm)),
        h: Math.max(1, Math.round((b.h * G) / wpm)),
      },
      bboxMm: { x: b.x / wpm, y: b.y / wpm, w: b.w / wpm, h: b.h / wpm },
      kind,
      areaPx: b.area,
      areaMm2: b.area / (wpm * wpm),
      thicknessMm,
      credible,
      confidence: credible ? baseConfidence : baseConfidence * 0.4,
    };
    };

    const all: DefectResult[] = [];
    for (const b of miss.out) {
      if (suppressHoles && looksLikeDrillHole(b, gBytes, w, h, wpm, k + 2)) {
        suppressed++;
        continue;
      }
      all.push(toDefect(b, 'open', 0.6));
    }
    for (const b of exc.out) {
      const nets = bridged.get(b.id) ?? 0;
      all.push(nets >= 2 ? toDefect(b, 'short', 0.9) : toDefect(b, 'unclassified', 0.5));
    }
    // Plan §5.5: a "pass" must never read as stronger evidence than it is - so the limitation is stated whenever
    // it applies, not only when it happened to trigger.
    if (input.reference.source === 'gerber' && !input.reference.drillMaskPath) {
      warnings.push(
        suppressHoles
          ? `No drill mask for this reference: round voids fully enclosed by reference copper are assumed to be drilled holes and ignored${suppressed > 0 ? ` (${suppressed} ignored in this capture)` : ''}. ` +
              'A real pinhole in a pad of that size is a blind spot - supply a drill mask to remove it.'
          : 'No drill mask for this reference: drilled pads may be reported as missing copper.',
      );
    }

    all.sort((a, b) => Number(b.credible) - Number(a.credible) || b.areaPx - a.areaPx);
    const defects = all.slice(0, cvConfig.maxDefects);
    const omittedDefects = all.length - defects.length;
    if (omittedDefects > 0) warnings.push(`${omittedDefects} additional smaller defects were not listed (cap ${cvConfig.maxDefects}); the verdict still counts them.`);

    if (Math.abs(widthBiasMm) >= cvConfig.maxWidthBiasMm) {
      warnings.push(
        `Copper is ${widthBiasMm > 0 ? 'wider' : 'narrower'} than the reference by ${Math.abs(widthBiasMm).toFixed(2)} mm per side. ` +
          'This is a global effect (under/over-etch, or photographic bloom) and was compensated for when looking for local defects, ' +
          'but it means this board does not match the design.',
      );
    }

    // ---- verdict (plan §5.4: defect count + total area against configurable thresholds) ------
    // Only defects thicker than this capture's detection limit may fail a board. Thinner ones are
    // overwhelmingly edge-misregistration slivers - they are listed (low confidence) and force a review,
    // but a verdict of "fail" on 300 hairlines would be noise dressed up as a finding.
    const credible = all.filter((d) => d.credible);
    const doubtful = all.length - credible.length;
    if (doubtful > 0) {
      warnings.push(
        `${doubtful} of ${all.length} detections are thinner than this capture's detection limit (${smallestDetectableMm.toFixed(2)} mm): edge-misregistration slivers, not counted toward a fail.`,
      );
    }
    // At low resolution or weak alignment a marginal defect is not enough evidence to fail a board:
    // only a confirmed short or a defect several times the detection limit may.
    const lowQuality = align.effectivePxPerMm < cvConfig.minPxPerMm || align.score < 0.5;
    const failable = lowQuality ? credible.filter((d) => d.kind === 'short' || d.thicknessMm >= GROSS_MARGIN * smallestDetectableMm) : credible;
    const failableAreaMm2 = failable.reduce((s, d) => s + d.areaMm2, 0);
    let verdict: CVInspectionResult['verdict'];
    if (all.length === 0) verdict = 'pass';
    else if (failable.length > 0 && (failableAreaMm2 >= cvConfig.failTotalAreaMm2 || failable.length >= cvConfig.failDefectCount || failable.some((d) => d.kind === 'short'))) verdict = 'fail';
    else verdict = 'needs_review';
    if (lowQuality && verdict === 'needs_review' && credible.length > 0) {
      warnings.push('Capture quality is low, so marginal defects are held for review rather than failing the board.');
    }

    // A "pass" must never read as stronger evidence than the capture supports (plan §6).
    if (verdict === 'pass') {
      if (Math.abs(widthBiasMm) >= cvConfig.maxWidthBiasMm) {
        verdict = 'needs_review';
      } else if (align.effectivePxPerMm < cvConfig.minPxPerMm) {
        verdict = 'needs_review';
        warnings.push(`Not passed: at ${align.effectivePxPerMm.toFixed(1)} px/mm this capture cannot rule out defects smaller than the stated detection limit.`);
      } else if (align.score < 0.5) {
        verdict = 'needs_review';
        warnings.push(`Not passed: alignment correlation is only ${align.score.toFixed(2)}, so a clean diff is not trustworthy.`);
      }
    }

    // ---- artifacts ---------------------------------------------------------------------------
    const maskPath = `${input.outputStem}_mask.png`;
    await writePng(maskPath, bytesOf(cap), w, h, 1);
    const previewPath = `${input.outputStem}_annotated.jpg`;
    await writePreview(previewPath, align, defects, w, h);

    return {
      verdict,
      effectivePxPerMm: align.effectivePxPerMm,
      workPxPerMm: wpm,
      smallestDetectableMm,
      toleranceMm: k / wpm,
      alignmentScore: align.score,
      alignmentMethod: align.method,
      localCorrection: align.localCorrection,
      widthBiasMm,
      opticalBlurMm,
      modelMismatch: { before: model.mismatchBefore, after: model.mismatchAfter },
      modelSpread: {
        biasMm: [model.biasRangePx[0] / wpm, model.biasRangePx[1] / wpm],
        blurMm: [model.sigmaRangePx[0] / wpm, model.sigmaRangePx[1] / wpm],
        grid: model.grid,
      },
      defects,
      omittedDefects,
      maskPath,
      previewPath,
      warnings,
    };
  });
}

/** For each excess-copper blob, how many distinct reference nets lie within `reach` px of it. */
function netsBridged(cv: CV, excess: Mat, exLabels: Mat, goldNets: Mat, w: number, h: number, reach: number): Map<number, number> {
  const { ex, gr, nets } = scoped((s) => {
    const grown = s.add(new cv.Mat());
    cv.dilate(excess, grown, s.add(cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(2 * reach + 1, 2 * reach + 1))));
    const grownLabels = s.add(new cv.Mat());
    cv.connectedComponents(grown, grownLabels, 8, cv.CV_32S);
    return { ex: Int32Array.from(exLabels.data32S), gr: Int32Array.from(grownLabels.data32S), nets: Int32Array.from(goldNets.data32S) };
  });

  // grown-component id -> set of reference nets under it
  const netsUnder = new Map<number, Set<number>>();
  for (let i = 0; i < w * h; i++) {
    const g = gr[i]!;
    const n = nets[i]!;
    if (g > 0 && n > 0) {
      let s = netsUnder.get(g);
      if (!s) netsUnder.set(g, (s = new Set()));
      s.add(n);
    }
  }
  // excess-blob id -> its grown component
  const result = new Map<number, number>();
  for (let i = 0; i < w * h; i++) {
    const e = ex[i]!;
    if (e > 0 && !result.has(e)) result.set(e, netsUnder.get(gr[i]!)?.size ?? 0);
  }
  return result;
}

/** A round void of plausible drill size whose surroundings are (almost) all reference copper. */
function looksLikeDrillHole(b: { x: number; y: number; w: number; h: number; area: number }, gold: Uint8Array, w: number, h: number, wpm: number, margin: number): boolean {
  const diaMm = Math.max(b.w, b.h) / wpm;
  if (diaMm < 0.25 || diaMm > 8) return false;
  const aspect = b.w / b.h;
  if (aspect < 0.7 || aspect > 1.43) return false;
  // A filled circle occupies pi/4 = 0.785 of its bounding box (0.65-0.85 once pixelated); a RECTANGLE occupies
  // 0.9-1.0. Without the upper bound a rectangular notch cut across a trace passed as a "round hole".
  const fill = b.area / (b.w * b.h);
  if (fill < 0.62 || fill > 0.86) return false;
  const cx = b.x + b.w / 2;
  const cy = b.y + b.h / 2;
  const r = Math.max(b.w, b.h) / 2 + margin;
  let onCopper = 0;
  for (let a = 0; a < 8; a++) {
    const px = Math.round(cx + r * Math.cos((a * Math.PI) / 4));
    const py = Math.round(cy + r * Math.sin((a * Math.PI) / 4));
    if (px >= 0 && py >= 0 && px < w && py < h && gold[py * w + px]) onCopper++;
  }
  return onCopper >= 6;
}

async function writePreview(file: string, align: AlignmentResult, defects: DefectResult[], w: number, h: number): Promise<void> {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  const ps = Math.min(1, 1600 / Math.max(w, h));
  const pw = Math.max(1, Math.round(w * ps));
  const ph = Math.max(1, Math.round(h * ps));
  const stroke = Math.max(2, Math.round(Math.max(pw, ph) / 400));
  const wpm = align.workPxPerMm;
  const boxes = defects
    .map((d, i) => {
      // bboxMm is region-relative mm, so mm * wpm * ps lands on the preview pixel grid.
      const pad = 3;
      const x = Math.max(0, d.bboxMm.x * wpm * ps - pad);
      const y = Math.max(0, d.bboxMm.y * wpm * ps - pad);
      const bw = Math.max(8, d.bboxMm.w * wpm * ps + 2 * pad);
      const bh = Math.max(8, d.bboxMm.h * wpm * ps + 2 * pad);
      const c = KIND_COLOR[d.kind];
      return (
        `<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${bw.toFixed(1)}" height="${bh.toFixed(1)}" fill="none" stroke="${c}" stroke-width="${stroke}"/>` +
        `<text x="${(x + 2).toFixed(1)}" y="${Math.max(12, y - 3).toFixed(1)}" font-size="${stroke * 6}" font-family="sans-serif" font-weight="bold" fill="${c}" stroke="#000" stroke-width="0.6">${i + 1}</text>`
      );
    })
    .join('');
  const svg = Buffer.from(`<svg xmlns="http://www.w3.org/2000/svg" width="${pw}" height="${ph}">${boxes}</svg>`);
  const rgb = Buffer.from(align.warpedRgb.data);
  await sharp(rgb, { raw: { width: w, height: h, channels: 3 } })
    .resize(pw, ph, { kernel: 'lanczos3' })
    .composite([{ input: svg, left: 0, top: 0 }])
    .jpeg({ quality: 85 })
    .toFile(file);
}
