import AdmZip from 'adm-zip';
import crypto from 'crypto';
import fs from 'fs';
import path from 'path';
import sharp from 'sharp';
import { z } from 'zod';
import { paths } from '../config';
import { parseJson, prisma, toJson } from '../db';
import { PipelineError } from '../errors';
import type { GoldenReference } from '../generated/prisma';

/**
 * Ingests the portal's AOI export package (plan §0.1). The manifest is a fixed contract: it is parsed
 * and validated here rather than re-keyed by hand, and only the load-bearing fields are consumed.
 *
 * Rasters are stored AS PROVIDED with the declared `copperValue`, never re-encoded: the copper polarity
 * is applied when the raster is read (image.ts), and cross-checked against the photographed board at
 * inspection time (align.ts) because the shipped export turned out to contradict its own manifest.
 */

const Side = z.enum(['top', 'bottom']);
const Rect = z.object({ x: z.number(), y: z.number(), w: z.number(), h: z.number() });

const ManifestSchema = z.object({
  schemaVersion: z.number(),
  job: z.object({ partNumber: z.string().optional(), revision: z.string().optional(), customer: z.string().optional() }).optional(),
  pxPerMm: z.number().positive(),
  images: z.array(
    z.object({
      file: z.string(),
      kind: z.enum(['copper', 'drill']),
      side: Side,
      mirrored: z.boolean(),
      widthPx: z.number().int().positive(),
      heightPx: z.number().int().positive(),
      copperValue: z.union([z.literal(0), z.literal(1)]),
    }),
  ),
  fiducials: z
    .array(
      z.object({
        id: z.string(),
        mm: z.tuple([z.number(), z.number()]),
        // per-side: the same physical point has a different x in the mirrored bottom image
        px: z.object({ top: z.tuple([z.number(), z.number()]), bottom: z.tuple([z.number(), z.number()]) }),
        diameterMm: z.number(),
      }),
    )
    .default([]),
  placements: z
    .array(
      z.object({
        designId: z.string(),
        row: z.number(),
        col: z.number(),
        mm: Rect,
        px: z.object({ top: Rect, bottom: Rect }),
      }),
    )
    .default([]),
  warnings: z.array(z.string()).default([]),
});
export type AoiManifest = z.infer<typeof ManifestSchema>;

export interface AoiFiles {
  manifest: AoiManifest;
  /** raster / manifest bytes keyed by base file name */
  files: Map<string, Buffer>;
}

const MAX_ENTRY_BYTES = 400 * 1024 * 1024;
const MAX_TOTAL_BYTES = 1024 * 1024 * 1024;

function parseManifest(bytes: Buffer): AoiManifest {
  let raw: unknown;
  try {
    raw = JSON.parse(bytes.toString('utf8'));
  } catch {
    throw new PipelineError('BAD_REQUEST', 'aoi.json is not valid JSON');
  }
  const parsed = ManifestSchema.safeParse(raw);
  if (!parsed.success) {
    throw new PipelineError('BAD_REQUEST', `aoi.json does not match the expected manifest: ${parsed.error.issues.map((i) => `${i.path.join('.')}: ${i.message}`).slice(0, 4).join('; ')}`);
  }
  return parsed.data;
}

export function loadAoiFromDir(dir: string): AoiFiles {
  const manifestPath = path.join(dir, 'aoi.json');
  if (!fs.existsSync(manifestPath)) throw new PipelineError('BAD_REQUEST', `No aoi.json in ${dir}`);
  const manifest = parseManifest(fs.readFileSync(manifestPath));
  const files = new Map<string, Buffer>();
  for (const img of manifest.images) {
    const p = path.join(dir, img.file);
    if (fs.existsSync(p)) files.set(path.basename(img.file), fs.readFileSync(p));
  }
  return { manifest, files };
}

/**
 * Reads the archive in memory. Entries are looked up by BASE name only and never extracted to disk, so
 * a hostile archive ("../../x") cannot write outside our directories; sizes are capped against zip bombs.
 */
export function loadAoiFromZip(zipBytes: Buffer): AoiFiles {
  let zip: AdmZip;
  try {
    zip = new AdmZip(zipBytes);
  } catch {
    throw new PipelineError('BAD_REQUEST', 'The uploaded file is not a readable ZIP archive');
  }
  let total = 0;
  const byBase = new Map<string, AdmZip.IZipEntry>();
  for (const e of zip.getEntries()) {
    if (e.isDirectory) continue;
    total += e.header.size;
    if (e.header.size > MAX_ENTRY_BYTES || total > MAX_TOTAL_BYTES) throw new PipelineError('BAD_REQUEST', 'The archive is too large when decompressed');
    byBase.set(path.posix.basename(e.entryName.replace(/\\/g, '/')), e);
  }
  const manifestEntry = byBase.get('aoi.json');
  if (!manifestEntry) throw new PipelineError('BAD_REQUEST', 'The archive contains no aoi.json');
  const manifest = parseManifest(manifestEntry.getData());
  const files = new Map<string, Buffer>();
  for (const img of manifest.images) {
    const e = byBase.get(path.basename(img.file));
    if (e) files.set(path.basename(img.file), e.getData());
  }
  return { manifest, files };
}

export interface IngestResult {
  references: GoldenReference[];
  created: number;
  warnings: string[];
}

async function storeRaster(bytes: Buffer, side: string, kind: 'copper' | 'drill'): Promise<string> {
  const sha = crypto.createHash('sha256').update(bytes).digest('hex').slice(0, 10);
  fs.mkdirSync(paths.golden, { recursive: true });
  const file = path.join(paths.golden, `aoi_${side}_${kind}_${sha}.png`);
  if (!fs.existsSync(file)) fs.writeFileSync(file, bytes);
  return file;
}

export async function ingestAoi(pkg: AoiFiles, opts: { boardId: number; sides: ('top' | 'bottom')[]; registeredBy: string }): Promise<IngestResult> {
  const { manifest, files } = pkg;
  const warnings = [...manifest.warnings];
  const references: GoldenReference[] = [];
  let created = 0;

  for (const side of opts.sides) {
    const copper = manifest.images.find((i) => i.kind === 'copper' && i.side === side);
    if (!copper) {
      warnings.push(`The package has no ${side} copper raster; skipped.`);
      continue;
    }
    const copperBytes = files.get(path.basename(copper.file));
    if (!copperBytes) throw new PipelineError('BAD_REQUEST', `The package lists ${copper.file} but does not contain it`);

    // Validate the ARTIFACT against its own manifest (plan §8): a mis-sized raster maps every mm wrongly.
    const meta = await sharp(copperBytes).metadata();
    if (meta.width !== copper.widthPx || meta.height !== copper.heightPx) {
      throw new PipelineError('REFERENCE_INVALID', `${copper.file} is ${meta.width}x${meta.height} but the manifest says ${copper.widthPx}x${copper.heightPx}`);
    }

    const maskPath = await storeRaster(copperBytes, side, 'copper');

    let drillMaskPath: string | null = null;
    const drill = manifest.images.find((i) => i.kind === 'drill' && i.side === side);
    if (drill) {
      const drillBytes = files.get(path.basename(drill.file));
      if (!drillBytes) {
        warnings.push(`${drill.file} is listed in the manifest but missing from the package - ${side} has no drill mask.`);
      } else {
        const dm = await sharp(drillBytes).metadata();
        if (dm.width !== copper.widthPx || dm.height !== copper.heightPx) {
          throw new PipelineError('REFERENCE_INVALID', `${drill.file} (${dm.width}x${dm.height}) must match its copper raster (${copper.widthPx}x${copper.heightPx})`);
        }
        drillMaskPath = await storeRaster(drillBytes, side, 'drill');
      }
    } else {
      warnings.push(`No ${side} drill mask in this package: drilled pads may read as missing copper.`);
    }

    // Idempotent: the same raster registered twice is one reference, not two.
    const existing = await prisma.goldenReference.findFirst({ where: { boardId: opts.boardId, side, source: 'gerber', maskPath } });
    if (existing) {
      references.push(existing);
      continue;
    }

    const fiducials = manifest.fiducials.map((f) => ({ id: f.id, mm: { x: f.mm[0], y: f.mm[1] }, px: { x: f.px[side][0], y: f.px[side][1] }, diameterMm: f.diameterMm }));
    const placements = manifest.placements.map((p) => ({ designId: p.designId, row: p.row, col: p.col, mm: p.mm, px: p.px[side] }));

    references.push(
      await prisma.goldenReference.create({
        data: {
          boardId: opts.boardId,
          source: 'gerber',
          side,
          mirrored: copper.mirrored,
          imagePath: maskPath,
          maskPath,
          drillMaskPath,
          copperValue: copper.copperValue,
          pxPerMm: manifest.pxPerMm,
          widthPx: copper.widthPx,
          heightPx: copper.heightPx,
          fiducials: toJson(fiducials),
          placements: toJson(placements),
          registeredBy: opts.registeredBy,
        },
      }),
    );
    created++;
  }
  return { references, created, warnings };
}

export const readPlacements = (g: Pick<GoldenReference, 'placements'>) =>
  parseJson<{ designId: string; row: number; col: number; mm: z.infer<typeof Rect>; px: z.infer<typeof Rect> }[]>(g.placements, []);
