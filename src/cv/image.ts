import sharp from 'sharp';
import fs from 'fs';
import path from 'path';
import { PipelineError } from '../errors';

export interface Rect {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface RawImage {
  data: Buffer;
  width: number;
  height: number;
  channels: number;
}

export interface DecodedRgb extends RawImage {
  /** Oriented source dimensions (after EXIF rotation), before any downscale. */
  sourceWidth: number;
  sourceHeight: number;
  /** decoded px per source px (1 when not downscaled). */
  scale: number;
}

/**
 * Decode a photo to raw RGB.
 *
 * `.rotate()` applies the EXIF orientation. Phone cameras store portrait shots sideways plus a tag,
 * and sharp ignores the tag unless asked, so without this a portrait photo is silently processed on
 * its side. (Some capture paths write no tag at all - alignment therefore also searches all four
 * 90-degree orientations rather than trusting this.)
 */
export async function decodeRgb(source: string | Buffer, maxSide?: number): Promise<DecodedRgb> {
  try {
    const meta = await sharp(source, { failOn: 'none' }).metadata();
    if (!meta.width || !meta.height) throw new Error('no dimensions');
    const swap = (meta.orientation ?? 1) >= 5;
    const sourceWidth = swap ? meta.height : meta.width;
    const sourceHeight = swap ? meta.width : meta.height;

    let img = sharp(source, { failOn: 'none' }).rotate().removeAlpha();
    let scale = 1;
    if (maxSide && Math.max(sourceWidth, sourceHeight) > maxSide) {
      scale = maxSide / Math.max(sourceWidth, sourceHeight);
      img = img.resize({
        width: Math.max(1, Math.round(sourceWidth * scale)),
        height: Math.max(1, Math.round(sourceHeight * scale)),
        fit: 'fill',
        kernel: 'lanczos3',
      });
    }
    const { data, info } = await img.toColourspace('srgb').raw().toBuffer({ resolveWithObject: true });
    return { data, width: info.width, height: info.height, channels: info.channels, sourceWidth, sourceHeight, scale };
  } catch (err) {
    throw new PipelineError('IMAGE_UNREADABLE', `Could not decode image: ${(err as Error).message}`);
  }
}

export interface GrayRegionOptions {
  /** Region of the source raster to read, in source pixels. Defaults to the whole raster. */
  rect?: Rect;
  /** Output size; the region is resampled to it (area-averaging when shrinking). */
  outWidth: number;
  outHeight: number;
  /** Pixel value that means "copper" in the source: 1 => 255, 0 => 0. Output is always copper = 255. */
  copperValue: 0 | 1;
}

/**
 * Read (part of) a binary raster resampled to the working resolution, normalised so copper = 255.
 * Normalising here is what makes `copperValue` matter: plan §0.1 says read it, never assume it.
 */
export async function decodeGrayRegion(file: string, opts: GrayRegionOptions): Promise<RawImage> {
  if (!fs.existsSync(file)) throw new PipelineError('REFERENCE_INVALID', `Reference raster is missing on disk: ${file}`);
  try {
    let img = sharp(file, { failOn: 'none' });
    if (opts.rect) {
      img = img.extract({ left: opts.rect.x, top: opts.rect.y, width: opts.rect.w, height: opts.rect.h });
    }
    img = img.resize({ width: opts.outWidth, height: opts.outHeight, fit: 'fill', kernel: 'lanczos3' }).greyscale();
    if (opts.copperValue === 0) img = img.negate({ alpha: false });
    const { data, info } = await img.raw().toBuffer({ resolveWithObject: true });
    return { data, width: info.width, height: info.height, channels: info.channels };
  } catch (err) {
    throw new PipelineError('REFERENCE_INVALID', `Could not read reference raster ${path.basename(file)}: ${(err as Error).message}`);
  }
}

export async function writePng(file: string, data: Uint8Array, width: number, height: number, channels: 1 | 3 | 4): Promise<void> {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  await sharp(Buffer.from(data.buffer, data.byteOffset, data.byteLength), { raw: { width, height, channels } })
    .png({ compressionLevel: 6 })
    .toFile(file);
}
