import sharp from 'sharp';
import { PipelineError } from '../errors';

// gerber-to-svg ships no types
// eslint-disable-next-line @typescript-eslint/no-require-imports
const gerberToSvg = require('gerber-to-svg');

export interface RasterizeGerberOptions {
  copperFileStr: string;
  /** Excellon drill file. Rendered on the SAME frame as the copper (see reframe()). */
  drillFileStr?: string | undefined;
  pxPerMm: number;
  /** Flip left-right: the bottom side is what a camera sees when the board is turned over (plan §7). */
  mirror?: boolean;
}

export interface RasterizeGerberResult {
  /** 8-bit greyscale PNG, copper = 255 (so copperValue = 1). */
  maskBuffer: Buffer;
  /** 8-bit greyscale PNG, drilled hole = 255, identical size/frame to maskBuffer. Kept SEPARATE - it is
   *  subtracted at diff time (plan §5.4), not baked into the copper. */
  drillBuffer?: Buffer;
  widthPx: number;
  heightPx: number;
  widthMm: number;
  heightMm: number;
  warnings: string[];
}

function convertToSvg(gerber: string): Promise<string> {
  return new Promise((resolve, reject) => {
    let svg = '';
    let converter: NodeJS.EventEmitter;
    try {
      converter = gerberToSvg(gerber, {});
    } catch (err) {
      reject(err);
      return;
    }
    converter.on('data', (chunk: string) => (svg += chunk));
    converter.on('end', () => resolve(svg));
    converter.on('error', (err: unknown) => reject(err instanceof Error ? err : new Error(String(err))));
  });
}

interface SvgBox {
  x: number;
  y: number;
  w: number;
  h: number;
  /** unit of the root width/height attributes */
  unit: 'mm' | 'in';
}

/**
 * gerber-to-svg's viewBox is in THOUSANDTHS of the file's unit (a 101 mm board has viewBox width
 * 101000), and the real size is only in the root width/height attributes ("101mm"). The old code read
 * the viewBox as millimetres and asked for a raster 1000x too large.
 */
function parseBox(svg: string): SvgBox {
  const vb = svg.match(/viewBox="([^"]+)"/);
  const wAttr = svg.match(/<svg[^>]*\swidth="([\d.]+)(mm|in)"/);
  if (!vb || !wAttr) throw new PipelineError('BAD_REQUEST', 'The Gerber file produced no drawable geometry (no viewBox/size in the converted SVG).');
  const [x, y, w, h] = vb[1]!.split(/\s+/).map(Number) as [number, number, number, number];
  return { x, y, w, h, unit: wAttr[2] as 'mm' | 'in' };
}

const toMm = (v: number, unit: 'mm' | 'in') => (unit === 'in' ? v * 25.4 : v);

/**
 * Copper and drill are emitted on DIFFERENT tight bounding boxes, so rasterised independently they
 * differ in size and offset (the old code stretched the drill onto the copper's frame, putting every
 * hole in the wrong place). Re-frame both onto the union box. The y-flip group is `translate(0, T)
 * scale(1,-1)` with T = ymin + ymax of the box, so it must be recomputed too.
 */
function reframe(svg: string, box: SvgBox, into: SvgBox): string {
  // UNIT-LESS millimetres on purpose: with mm/in units sharp applies `density` twice (measured: 100 mm
  // at 150 dpi renders 1230 px, not 591), whereas a unit-less width scales linearly as width * density/72.
  const unitSize = (v: number) => `${+toMm(v / 1000, box.unit).toFixed(6)}`;
  const T = 2 * into.y + into.h;
  return svg
    .replace(/(<svg[^>]*\s)width="[^"]*"/, `$1width="${unitSize(into.w)}"`)
    .replace(/(<svg[^>]*\s)height="[^"]*"/, `$1height="${unitSize(into.h)}"`)
    .replace(/viewBox="[^"]*"/, `viewBox="${into.x} ${into.y} ${into.w} ${into.h}"`)
    .replace(/translate\(0,\s*[-\d.]+\)\s*scale\(1,\s*-1\)/, `translate(0,${T}) scale(1,-1)`)
    .replace(/currentColor/g, '#ffffff');
}

async function rasterize(svg: string, pxPerMm: number, mirror: boolean): Promise<{ png: Buffer; width: number; height: number }> {
  // Render the vector directly at the target resolution; resizing a default-DPI raster (the old
  // approach) is blurry and loses thin features.
  // 1 user unit = 1 mm, and sharp renders width * density/72 px, so density = 72 * pxPerMm gives exactly pxPerMm.
  let img = sharp(Buffer.from(svg), { density: 72 * pxPerMm })
    .flatten({ background: '#000000' }) // librsvg output is transparent; unpainted = substrate = black
    .greyscale()
    .threshold(128);
  if (mirror) img = img.flop();
  const { data, info } = await img.png({ compressionLevel: 9 }).toBuffer({ resolveWithObject: true });
  return { png: data, width: info.width, height: info.height };
}

export async function rasterizeGerber(options: RasterizeGerberOptions): Promise<RasterizeGerberResult> {
  const warnings: string[] = [];
  if (!(options.pxPerMm > 0) || options.pxPerMm > 200) {
    throw new PipelineError('BAD_REQUEST', `pxPerMm must be between 0 and 200, got ${options.pxPerMm}`);
  }

  let copperSvg: string;
  try {
    copperSvg = await convertToSvg(options.copperFileStr);
  } catch (err) {
    throw new PipelineError('BAD_REQUEST', `Could not parse the copper Gerber: ${(err as Error).message}`);
  }
  const copperBox = parseBox(copperSvg);

  let drillSvg: string | undefined;
  let frame = copperBox;
  if (options.drillFileStr) {
    try {
      drillSvg = await convertToSvg(options.drillFileStr);
      const d = parseBox(drillSvg);
      if (d.unit !== copperBox.unit) throw new Error(`drill file is in ${d.unit} but copper is in ${copperBox.unit}`);
      const x0 = Math.min(copperBox.x, d.x);
      const y0 = Math.min(copperBox.y, d.y);
      const x1 = Math.max(copperBox.x + copperBox.w, d.x + d.w);
      const y1 = Math.max(copperBox.y + copperBox.h, d.y + d.h);
      frame = { x: x0, y: y0, w: x1 - x0, h: y1 - y0, unit: copperBox.unit };
    } catch (err) {
      drillSvg = undefined;
      warnings.push(`Drill file ignored: ${(err as Error).message}`);
    }
  }

  const mask = await rasterize(reframe(copperSvg, copperBox, frame), options.pxPerMm, options.mirror === true);
  const result: RasterizeGerberResult = {
    maskBuffer: mask.png,
    widthPx: mask.width,
    heightPx: mask.height,
    widthMm: toMm(frame.w / 1000, frame.unit),
    heightMm: toMm(frame.h / 1000, frame.unit),
    warnings,
  };

  if (drillSvg) {
    const drill = await rasterize(reframe(drillSvg, parseBox(drillSvg), frame), options.pxPerMm, options.mirror === true);
    if (drill.width !== mask.width || drill.height !== mask.height) {
      warnings.push(`Drill raster (${drill.width}x${drill.height}) does not match copper (${mask.width}x${mask.height}); drill mask dropped.`);
    } else {
      result.drillBuffer = drill.png;
    }
  }

  // Plan §4.6/§8: validate the artifact, not the step. A blank raster is the most dangerous outcome.
  const { data } = await sharp(mask.png).greyscale().raw().toBuffer({ resolveWithObject: true });
  let lit = 0;
  for (let i = 0; i < data.length; i++) if (data[i]) lit++;
  if (lit === 0) throw new PipelineError('BINARIZATION_IMPLAUSIBLE', 'The rasterised copper is completely blank - refusing to register an empty reference.');
  if (lit === data.length) throw new PipelineError('BINARIZATION_IMPLAUSIBLE', 'The rasterised copper is completely filled - refusing to register it.');
  if (result.drillBuffer) {
    const dd = await sharp(result.drillBuffer).greyscale().raw().toBuffer();
    if (!dd.some((v) => v)) {
      delete result.drillBuffer;
      warnings.push('The drill file rendered completely blank and was dropped rather than registered (a blank drill mask would hide nothing but look like data).');
    }
  }
  return result;
}
