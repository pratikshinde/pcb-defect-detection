import fs from 'fs';
import path from 'path';
import sharp from 'sharp';
import { cvConfig } from '../config';
import { PipelineError } from '../errors';
import { detectOutlineQuad, homographyFromQuads, polyArea, rotateQuad, warpRgb, type Pt, type Quad } from './align';
import { binarizeCopper } from './binarize';
import { decodeRgb, writePng } from './image';
import { bytesOf, matFromRaw, scoped, withCv } from './opencv';

export interface RegistrationInput {
  imagePath: string;
  /** Physical size of the region inside the corners. A photograph has no ground-truth scale (plan §0),
   *  so it must be supplied - plan §4.4: "a user-entered board size in the app". */
  widthMm: number;
  heightMm: number;
  /** Cap on the stored resolution; never exceeds what the photo actually contains. */
  pxPerMm?: number | undefined;
  /** Operator-tapped corners (fractions 0-1 of the oriented photo), TL,TR,BR,BL as the operator sees them. */
  corners?: Quad | undefined;
  /** Absolute path prefix (no extension) for the artifacts written. */
  outputStem: string;
}

export interface CVRegistrationResult {
  widthPx: number;
  heightPx: number;
  pxPerMm: number;
  effectivePxPerMm: number;
  method: 'manual-corners' | 'auto-outline';
  coverage: number;
  contrast: number;
  maskPath: string;
  previewPath: string;
  /** Always empty: see the note in the warnings. Kept for the plan §3 response shape. */
  fiducialsFound: never[];
  warnings: string[];
}

/**
 * Photo -> golden reference (plan §4): find the board, perspective-correct it to a top-down frame of
 * known physical size, binarise, validate the artifact, store.
 *
 * Deliberate deviation from plan §4.2-4.3 (fiducial detection): the old implementation returned
 * three FABRICATED fiducials ({100,100} etc.) as if found, and applied no perspective correction -
 * exactly the "silently degraded artifact" plan §8 calls the worst failure mode. Fiducials also
 * cannot serve here: the panel fiducials in aoi.json all sit in the rails outside every board, so a
 * single-board photo contains none, and three points cannot define a homography anyway. The board
 * outline (or four operator-tapped corners) defines the transform instead.
 */
export async function registerGoldenReference(input: RegistrationInput): Promise<CVRegistrationResult> {
  if (!(input.widthMm > 0) || !(input.heightMm > 0)) throw new PipelineError('BAD_REQUEST', 'widthMm and heightMm must be positive numbers');
  const warnings: string[] = [];

  return withCv(async (cv, scope) => {
    // ---- 1. locate the board ------------------------------------------------------------------
    const preview = await decodeRgb(input.imagePath, 1400);
    const srcW = preview.sourceWidth;
    const srcH = preview.sourceHeight;
    let quadSrc: Quad;
    let method: CVRegistrationResult['method'];
    if (input.corners) {
      quadSrc = input.corners.map(([fx, fy]) => [fx * srcW, fy * srcH] as Pt) as Quad;
      method = 'manual-corners';
    } else {
      const q = scoped((s) => detectOutlineQuad(cv, s, matFromRaw(cv, s, preview.data, preview.width, preview.height, 3)));
      quadSrc = q.map(([x, y]) => [x / preview.scale, y / preview.scale] as Pt) as Quad;
      method = 'auto-outline';
    }

    // ---- 2. resolution gate (plan §4.1, §6) ---------------------------------------------------
    const effNative = Math.sqrt(polyArea(quadSrc) / (input.widthMm * input.heightMm));
    if (effNative < cvConfig.refusePxPerMm) {
      throw new PipelineError(
        'RESOLUTION_TOO_LOW',
        `The board is only ${effNative.toFixed(1)} px/mm in this photo (minimum ${cvConfig.refusePxPerMm}); a reference registered from it could never support a trustworthy inspection. Move closer / use the highest-resolution photo mode.`,
        { effectivePxPerMm: effNative },
      );
    }
    if (effNative < cvConfig.minPxPerMm) {
      warnings.push(`Low resolution: ${effNative.toFixed(1)} px/mm (< ${cvConfig.minPxPerMm}). Inspections against this reference cannot resolve features below ~${(2 / effNative).toFixed(2)} mm.`);
    }
    const pxPerMm = Math.min(effNative, input.pxPerMm ?? cvConfig.maxWorkPxPerMm, cvConfig.maxWorkPxPerMm);
    const outW = Math.round(input.widthMm * pxPerMm);
    const outH = Math.round(input.heightMm * pxPerMm);

    // ---- 3. orientation: the quad's aspect must match the stated size -------------------------
    let k = 0;
    if (method === 'auto-outline') {
      const len = (a: Pt, b: Pt) => Math.hypot(a[0] - b[0], a[1] - b[1]);
      let bestCost = Infinity;
      for (let i = 0; i < 4; i++) {
        const q = rotateQuad(quadSrc, i);
        const cost = Math.abs(Math.log(len(q[0], q[1]) / len(q[1], q[2]) / (input.widthMm / input.heightMm)));
        if (cost < bestCost - 1e-9) {
          bestCost = cost;
          k = i;
        }
      }
      if (bestCost > 0.35) {
        warnings.push(
          `The detected outline's proportions differ from the stated ${input.widthMm} x ${input.heightMm} mm by ${(Math.exp(bestCost) * 100 - 100).toFixed(0)}%. ` +
            'Check the size, or tap the four corners.',
        );
      }
      warnings.push('Orientation of an auto-detected outline is arbitrary (which way is "up" cannot be known from a single photo); later inspections resolve it by correlation.');
    }

    // ---- 4. rectify + binarise ----------------------------------------------------------------
    const capScale = Math.min(1, (pxPerMm * 1.5) / effNative);
    const cap = await decodeRgb(input.imagePath, Math.round(Math.max(srcW, srcH) * capScale));
    const quadCap = quadSrc.map(([x, y]) => [x * cap.scale, y * cap.scale] as Pt) as Quad;
    const capMat = matFromRaw(cv, scope, cap.data, cap.width, cap.height, 3);
    const H = homographyFromQuads(cv, rotateQuad(quadCap, k), [
      [0, 0],
      [outW, 0],
      [outW, outH],
      [0, outH],
    ]);
    const { rgb, valid } = warpRgb(cv, scope, capMat, H, outW, outH);
    scope.free(capMat);
    const validBytes = bytesOf(valid);
    let inside = 0;
    for (let i = 0; i < validBytes.length; i++) if (validBytes[i]) inside++;
    if (inside / validBytes.length < 0.98) {
      throw new PipelineError('ALIGNMENT_FAILED', 'The rectified board extends outside the photo - the board is cut off or the corners are wrong.');
    }
    // Throws BINARIZATION_IMPLAUSIBLE for a blank / uniform / low-contrast result (plan §4.6).
    const bin = binarizeCopper(cv, scope, rgb, { pxPerMm, valid });

    // ---- 5. store -----------------------------------------------------------------------------
    fs.mkdirSync(path.dirname(input.outputStem), { recursive: true });
    const maskPath = `${input.outputStem}_mask.png`;
    await writePng(maskPath, bytesOf(bin.mask), outW, outH, 1);
    const previewPath = `${input.outputStem}_rectified.jpg`;
    await sharp(Buffer.from(rgb.data), { raw: { width: outW, height: outH, channels: 3 } })
      .resize({ width: Math.min(outW, 1600), withoutEnlargement: true })
      .jpeg({ quality: 85 })
      .toFile(previewPath);

    warnings.push('Registered from the board outline, not fiducials (see plan notes): no fiducial positions are stored for this reference.');
    return {
      widthPx: outW,
      heightPx: outH,
      pxPerMm,
      effectivePxPerMm: effNative,
      method,
      coverage: bin.coverage,
      contrast: bin.contrast,
      maskPath,
      previewPath,
      fiducialsFound: [],
      warnings,
    };
  });
}
