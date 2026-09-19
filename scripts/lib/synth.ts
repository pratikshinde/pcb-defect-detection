/**
 * Synthetic "photographs" of a board, with KNOWN ground truth, for validating the pipeline
 * (plan §9 Phase 2: "validated against deliberately-defective sample pairs ... also the negative
 * case: a known-good board must come back clean").
 *
 * A photo is rendered from a copper mask (255 = copper) by: downscaling to the requested px/mm,
 * colourising (orange copper on dark substrate), applying an illumination gradient, optical blur and
 * sensor noise, then perspective-warping the board onto a cluttered canvas and JPEG-encoding it.
 */
import sharp from 'sharp';
import type { Quad } from '../../src/cv/align';
import { getCv, type CV } from '../../src/cv/opencv';

export interface SynthOptions {
  /** resolution of the board in the photo */
  pxPerMm: number;
  seed?: number;
  /** optical blur sigma, in photo pixels */
  blurPx?: number;
  noise?: number;
  /** illumination gradient strength (0 = flat) */
  gradient?: number;
  /** in-plane rotation of the board on the canvas, degrees */
  rotationDeg?: number;
  /** perspective keystone as a fraction of the board size */
  perspective?: number;
  /** extra canvas margin around the board, as a fraction of the board size */
  margin?: number;
  /** how much wider (each side) the copper is than designed, in mm - under-etch */
  underEtchMm?: number;
}

export interface SynthPhoto {
  /** board corners in the photo as fractions of the image, TL,TR,BR,BL as the board was authored */
  corners: Quad;
  width: number;
  height: number;
}

const mulberry32 = (seed: number) => () => {
  let t = (seed += 0x6d2b79f5);
  t = Math.imul(t ^ (t >>> 15), t | 1);
  t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
};

/** @param mask copper = 255, at `srcPxPerMm` */
export async function renderPhoto(mask: Uint8Array, w: number, h: number, srcPxPerMm: number, opts: SynthOptions, outPath: string): Promise<SynthPhoto> {
  const cv = await getCv();
  const rand = mulberry32(opts.seed ?? 1);
  const gauss = () => Math.sqrt(-2 * Math.log(rand() + 1e-12)) * Math.cos(2 * Math.PI * rand());

  const bw = Math.round((w * opts.pxPerMm) / srcPxPerMm);
  const bh = Math.round((h * opts.pxPerMm) / srcPxPerMm);

  // 1. design -> photo scale (area-averaged, like a sensor integrating light)
  let gray: Buffer = await sharp(Buffer.from(mask.buffer, mask.byteOffset, mask.byteLength), { raw: { width: w, height: h, channels: 1 } })
    .resize(bw, bh, { kernel: 'lanczos3' })
    .greyscale()
    .raw()
    .toBuffer();

  // 2. under-etch: copper wider than designed (a real process effect, not a defect)
  const etchPx = (opts.underEtchMm ?? 0) * opts.pxPerMm;
  if (etchPx >= 0.5) {
    const k = Math.max(1, Math.round(etchPx));
    const src = new cv.Mat(bh, bw, cv.CV_8UC1);
    src.data.set(gray);
    const dst = new cv.Mat();
    const kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(2 * k + 1, 2 * k + 1));
    cv.dilate(src, dst, kernel);
    gray = Buffer.from(dst.data);
    src.delete();
    dst.delete();
    kernel.delete();
  }

  // 3. colourise + illumination + noise
  const rgb = new Uint8Array(bw * bh * 3);
  const g = opts.gradient ?? 0.25;
  const ang = rand() * Math.PI * 2;
  const gx = Math.cos(ang);
  const gy = Math.sin(ang);
  for (let y = 0; y < bh; y++) {
    for (let x = 0; x < bw; x++) {
      const i = y * bw + x;
      const t = gray[i]! / 255;
      const light = 1 + g * (((x / bw - 0.5) * gx + (y / bh - 0.5) * gy) * 1.6);
      rgb[i * 3] = Math.max(0, Math.min(255, (30 + 206 * t) * light));
      rgb[i * 3 + 1] = Math.max(0, Math.min(255, (38 + 140 * t) * light));
      rgb[i * 3 + 2] = Math.max(0, Math.min(255, (30 + 66 * t) * light));
    }
  }

  // 4. optical blur
  const boardMat = new cv.Mat(bh, bw, cv.CV_8UC3);
  boardMat.data.set(rgb);
  if ((opts.blurPx ?? 0) > 0.2) {
    const b = new cv.Mat();
    cv.GaussianBlur(boardMat, b, new cv.Size(0, 0), opts.blurPx!, opts.blurPx!, cv.BORDER_REPLICATE);
    b.copyTo(boardMat);
    b.delete();
  }
  {
    const d = boardMat.data;
    const n = opts.noise ?? 4;
    for (let i = 0; i < d.length; i++) d[i] = Math.max(0, Math.min(255, d[i]! + gauss() * n));
  }

  // 5. perspective-warp onto a cluttered canvas
  const margin = opts.margin ?? 0.18;
  const rot = ((opts.rotationDeg ?? 0) * Math.PI) / 180;
  // size the canvas to the ROTATED footprint: a 90-degree turn of a portrait board is wider than it is tall
  const footW = Math.abs(bw * Math.cos(rot)) + Math.abs(bh * Math.sin(rot));
  const footH = Math.abs(bw * Math.sin(rot)) + Math.abs(bh * Math.cos(rot));
  const cw = Math.round(footW * (1 + 2 * margin));
  const ch = Math.round(footH * (1 + 2 * margin));
  const cx = cw / 2;
  const cy = ch / 2;
  const keystone = (opts.perspective ?? 0.03) * Math.min(bw, bh);
  const corner = (sx: number, sy: number, jx: number, jy: number): [number, number] => {
    const x = (sx * bw) / 2 + jx * keystone;
    const y = (sy * bh) / 2 + jy * keystone;
    return [cx + x * Math.cos(rot) - y * Math.sin(rot), cy + x * Math.sin(rot) + y * Math.cos(rot)];
  };
  const dst4 = [corner(-1, -1, 0.6, 0.2), corner(1, -1, -0.4, 0.5), corner(1, 1, -0.5, -0.3), corner(-1, 1, 0.3, -0.6)];
  const srcM = cv.matFromArray(4, 1, cv.CV_32FC2, [0, 0, bw, 0, bw, bh, 0, bh]);
  const dstM = cv.matFromArray(4, 1, cv.CV_32FC2, dst4.flat());
  const H = cv.getPerspectiveTransform(srcM, dstM);
  const canvas = new cv.Mat(ch, cw, cv.CV_8UC3, new cv.Scalar(92, 84, 70, 255)); // desk-like, low saturation
  cv.warpPerspective(boardMat, canvas, H, new cv.Size(cw, ch), cv.INTER_LINEAR, cv.BORDER_TRANSPARENT);
  // clutter that must not be mistaken for the board (kept in the margin: the board spans 13%-87% of the canvas)
  cv.rectangle(canvas, new cv.Point(Math.round(cw * 0.93), Math.round(ch * 0.6)), new cv.Point(cw - 1, ch - 1), new cv.Scalar(126, 158, 183, 255), -1);
  cv.rectangle(canvas, new cv.Point(0, 0), new cv.Point(Math.round(cw * 0.2), Math.round(ch * 0.07)), new cv.Scalar(12, 12, 12, 255), -1);

  await sharp(Buffer.from(canvas.data), { raw: { width: cw, height: ch, channels: 3 } }).jpeg({ quality: 90 }).toFile(outPath);
  [boardMat, srcM, dstM, H, canvas].forEach((m) => m.delete());

  return {
    corners: dst4.map(([x, y]) => [x / cw, y / ch]) as Quad,
    width: cw,
    height: ch,
  };
}

export type { CV };
