import crypto from 'crypto';
import { Router, type Request } from 'express';
import fs from 'fs';
import multer from 'multer';
import path from 'path';
import { z } from 'zod';
import { ingestAoi, loadAoiFromZip, readPlacements } from './aoi/ingest';
import { paths } from './config';
import type { Quad } from './cv/align';
import { runInspection } from './cv/comparison';
import { rasterizeGerber } from './cv/gerber';
import { registerGoldenReference } from './cv/registration';
import { prisma, toJson } from './db';
import { PipelineError } from './errors';
import { goldenReferenceDto, inspectionDto, inspectionSummaryDto, resultUrl } from './serialize';

const router = Router();

// ---- uploads ------------------------------------------------------------------------------
// Names are generated, never taken from the client: originalname is attacker-controlled, and the
// old `${Date.now()}-${originalname}` also made every derived path depend on the client's extension.
const EXT_BY_MIME: Record<string, string> = {
  'image/jpeg': '.jpg',
  'image/png': '.png',
  'application/zip': '.zip',
  'application/x-zip-compressed': '.zip',
};
const storage = multer.diskStorage({
  destination: (_req, _file, cb) => {
    fs.mkdirSync(paths.uploads, { recursive: true });
    cb(null, paths.uploads);
  },
  filename: (_req, file, cb) => cb(null, `${Date.now()}-${crypto.randomBytes(4).toString('hex')}${EXT_BY_MIME[file.mimetype] ?? '.bin'}`),
});
const acceptOnly = (allowed: (mime: string) => boolean) => (_req: Request, file: Express.Multer.File, cb: multer.FileFilterCallback) =>
  allowed(file.mimetype) ? cb(null, true) : cb(new PipelineError('BAD_REQUEST', `Unsupported file type "${file.mimetype}"`));

const isImage = (m: string) => m === 'image/jpeg' || m === 'image/png';
const isZip = (m: string) => m === 'application/zip' || m === 'application/x-zip-compressed' || m === 'application/octet-stream';
const uploadImage = multer({ storage, limits: { fileSize: 120 * 1024 * 1024, files: 1 }, fileFilter: acceptOnly(isImage) });
const uploadImageOrZip = multer({ storage, limits: { fileSize: 600 * 1024 * 1024, files: 1 }, fileFilter: acceptOnly((m) => isImage(m) || isZip(m)) });
const uploadGerber = multer({ storage, limits: { fileSize: 100 * 1024 * 1024, files: 2 } });

// ---- validation helpers -------------------------------------------------------------------
const Side = z.enum(['top', 'bottom']);
const IdParam = z.coerce.number().int().positive();
const PositiveNumber = z.coerce.number().positive();
const Unit = z.number().min(0).max(1);
const CornersSchema = z.tuple([z.tuple([Unit, Unit]), z.tuple([Unit, Unit]), z.tuple([Unit, Unit]), z.tuple([Unit, Unit])]);

function id(value: unknown, what = 'id'): number {
  const r = IdParam.safeParse(value);
  if (!r.success) throw new PipelineError('BAD_REQUEST', `${what} must be a positive integer`);
  return r.data;
}

/** Operator-tapped corners arrive as a JSON string in a multipart field. */
function parseCorners(raw: unknown): Quad | undefined {
  if (raw === undefined || raw === '') return undefined;
  let value: unknown;
  try {
    value = typeof raw === 'string' ? JSON.parse(raw) : raw;
  } catch {
    throw new PipelineError('BAD_REQUEST', 'corners must be JSON: [[x,y],[x,y],[x,y],[x,y]] as fractions of the image');
  }
  const r = CornersSchema.safeParse(value);
  if (!r.success) throw new PipelineError('BAD_REQUEST', 'corners must be four [x,y] pairs, each in 0..1 (top-left, top-right, bottom-right, bottom-left)');
  const q = r.data;
  let a = 0;
  for (let i = 0; i < 4; i++) a += q[i]![0] * q[(i + 1) % 4]![1] - q[(i + 1) % 4]![0] * q[i]![1];
  if (Math.abs(a) / 2 < 0.05) throw new PipelineError('BAD_REQUEST', 'corners enclose almost no area');
  return q as Quad;
}

async function requireBoard(boardId: number) {
  const board = await prisma.board.findUnique({ where: { id: boardId } });
  if (!board) throw new PipelineError('NOT_FOUND', 'Board not found');
  return board;
}

const stem = (dir: string, prefix: string) => path.join(dir, `${prefix}_${Date.now()}_${crypto.randomBytes(3).toString('hex')}`);

// ---- boards -------------------------------------------------------------------------------
router.get('/boards', async (_req, res) => {
  res.json(await prisma.board.findMany({ orderBy: { id: 'asc' } }));
});

router.post('/boards', async (req, res) => {
  const body = z.object({ name: z.string().trim().min(1), customerRef: z.string().optional(), revision: z.string().optional() }).parse(req.body);
  res.status(201).json(await prisma.board.create({ data: body }));
});

// ---- golden references --------------------------------------------------------------------
router.get('/boards/:id/golden-references', async (req, res) => {
  const boardId = id(req.params.id);
  await requireBoard(boardId);
  const refs = await prisma.goldenReference.findMany({ where: { boardId }, orderBy: { registeredAt: 'desc' } });
  res.json(refs.map(goldenReferenceDto));
});

const RegisterBody = z.object({
  source: z.enum(['photo', 'gerber-seed']),
  side: Side.optional(),
  widthMm: PositiveNumber.optional(),
  heightMm: PositiveNumber.optional(),
  pxPerMm: PositiveNumber.optional(),
  corners: z.string().optional(),
  dryRun: z.enum(['true', 'false']).optional(),
  operatorId: z.string().optional(),
  deviceId: z.string().optional(),
});

router.post('/boards/:id/golden-references', uploadImageOrZip.single('file'), async (req, res) => {
  const boardId = id(req.params.id);
  await requireBoard(boardId);
  if (!req.file) throw new PipelineError('BAD_REQUEST', 'A file is required');
  const body = RegisterBody.parse(req.body);
  const registeredBy = body.operatorId ?? body.deviceId ?? 'api';

  if (body.source === 'gerber-seed') {
    // The portal's AOI ZIP: the manifest is the contract, nothing is re-keyed by hand (plan §3).
    const pkg = loadAoiFromZip(fs.readFileSync(req.file.path));
    const result = await ingestAoi(pkg, { boardId, sides: body.side ? [body.side] : ['top', 'bottom'], registeredBy });
    res.status(result.created > 0 ? 201 : 200).json({ created: result.created, warnings: result.warnings, references: result.references.map(goldenReferenceDto) });
    return;
  }

  if (!isImage(req.file.mimetype)) throw new PipelineError('BAD_REQUEST', 'A photo registration needs a JPEG or PNG image');
  if (!body.side) throw new PipelineError('BAD_REQUEST', 'side is required');
  if (!body.widthMm || !body.heightMm) {
    throw new PipelineError('BAD_REQUEST', 'widthMm and heightMm are required for a photo: it has no ground-truth scale, so the physical size of the region inside the corners must be stated');
  }

  const outStem = stem(paths.results, 'registration');
  const reg = await registerGoldenReference({
    imagePath: req.file.path,
    widthMm: body.widthMm,
    heightMm: body.heightMm,
    pxPerMm: body.pxPerMm,
    corners: parseCorners(body.corners),
    outputStem: outStem,
  });
  const summary = {
    fiducialsFound: reg.fiducialsFound.length,
    widthPx: reg.widthPx,
    heightPx: reg.heightPx,
    pxPerMm: reg.pxPerMm,
    effectivePxPerMm: reg.effectivePxPerMm,
    copperCoverage: reg.coverage,
    method: reg.method,
    previewUrl: resultUrl(reg.previewPath),
    warnings: reg.warnings,
  };

  if (body.dryRun === 'true') {
    // Plan §5.6: show the result BEFORE committing - a registration that silently succeeds with a bad
    // outline would poison every later inspection of this board.
    fs.rmSync(reg.maskPath, { force: true });
    res.json({ dryRun: true, ...summary });
    return;
  }

  fs.mkdirSync(paths.golden, { recursive: true });
  const maskPath = path.join(paths.golden, `photo_${body.side}_${path.basename(reg.maskPath).replace(/^registration_/, '')}`);
  fs.renameSync(reg.maskPath, maskPath);
  const ref = await prisma.goldenReference.create({
    data: {
      boardId,
      source: 'photo',
      side: body.side,
      mirrored: body.side === 'bottom',
      imagePath: req.file.path,
      maskPath,
      copperValue: 1, // the mask is written copper = 255 by construction
      pxPerMm: reg.pxPerMm,
      widthPx: reg.widthPx,
      heightPx: reg.heightPx,
      fiducials: toJson([]),
      registeredBy,
    },
  });
  res.status(201).json({ dryRun: false, goldenReferenceId: ref.id, ...summary });
});

router.post(
  '/boards/:id/golden-references/gerber',
  uploadGerber.fields([
    { name: 'copperFile', maxCount: 1 },
    { name: 'drillFile', maxCount: 1 },
  ]),
  async (req, res) => {
    const boardId = id(req.params.id);
    await requireBoard(boardId);
    const files = req.files as Record<string, Express.Multer.File[]> | undefined;
    const copperFile = files?.copperFile?.[0];
    if (!copperFile) throw new PipelineError('BAD_REQUEST', 'copperFile is required');
    const body = z.object({ side: Side, pxPerMm: PositiveNumber.default(40), operatorId: z.string().optional() }).parse(req.body);

    const result = await rasterizeGerber({
      copperFileStr: fs.readFileSync(copperFile.path, 'utf8'),
      drillFileStr: files?.drillFile?.[0] ? fs.readFileSync(files.drillFile[0].path, 'utf8') : undefined,
      pxPerMm: body.pxPerMm,
      mirror: body.side === 'bottom', // what a camera sees once the board is turned over (plan §7)
    });

    const base = stem(paths.golden, `gerber_${body.side}`);
    fs.mkdirSync(paths.golden, { recursive: true });
    const maskPath = `${base}_copper.png`;
    fs.writeFileSync(maskPath, result.maskBuffer);
    let drillMaskPath: string | null = null;
    if (result.drillBuffer) {
      drillMaskPath = `${base}_drill.png`;
      fs.writeFileSync(drillMaskPath, result.drillBuffer);
    }
    const ref = await prisma.goldenReference.create({
      data: {
        boardId,
        source: 'gerber',
        side: body.side,
        mirrored: body.side === 'bottom',
        maskPath,
        drillMaskPath,
        copperValue: 1, // rasterizeGerber writes copper = 255
        pxPerMm: body.pxPerMm,
        widthPx: result.widthPx,
        heightPx: result.heightPx,
        fiducials: toJson([]),
        registeredBy: body.operatorId ?? 'api',
      },
    });
    res.status(201).json({ ...goldenReferenceDto(ref), widthMm: result.widthMm, heightMm: result.heightMm, warnings: result.warnings });
  },
);

// ---- inspections --------------------------------------------------------------------------
const InspectBody = z.object({
  side: Side,
  placementIndex: z.coerce.number().int().min(0).optional(),
  corners: z.string().optional(),
  deviceId: z.string().optional(),
  operatorId: z.string().optional(),
});

router.post('/boards/:id/inspections', uploadImage.single('file'), async (req, res) => {
  const boardId = id(req.params.id);
  await requireBoard(boardId);
  if (!req.file) throw new PipelineError('BAD_REQUEST', 'An image file is required');
  const body = InspectBody.parse(req.body);

  // The latest reference FOR THIS SIDE: comparing a bottom capture to a top reference is a guaranteed
  // all-red diff (plan §7), so side is required and never inferred.
  const ref = await prisma.goldenReference.findFirst({ where: { boardId, side: body.side }, orderBy: { registeredAt: 'desc' } });
  if (!ref) throw new PipelineError('NOT_FOUND', `No ${body.side}-side golden reference registered for this board`);

  let target = { x: 0, y: 0, w: ref.widthPx, h: ref.heightPx };
  if (body.placementIndex !== undefined) {
    const placements = readPlacements(ref);
    const p = placements[body.placementIndex];
    if (!p) {
      throw new PipelineError(
        'BAD_REQUEST',
        placements.length === 0
          ? 'This reference has no placements: it is a single board, so omit placementIndex'
          : `placementIndex must be 0..${placements.length - 1} for this reference`,
      );
    }
    target = { x: p.px.x, y: p.px.y, w: p.px.w, h: p.px.h };
  }

  const outStem = stem(paths.results, 'inspection');
  const result = await runInspection({
    capturePath: req.file.path,
    reference: {
      maskPath: ref.maskPath,
      drillMaskPath: ref.drillMaskPath,
      copperValue: ref.copperValue === 0 ? 0 : 1,
      pxPerMm: ref.pxPerMm,
      source: ref.source,
    },
    target,
    corners: parseCorners(body.corners),
    outputStem: outStem,
  });

  const saved = await prisma.inspection.create({
    data: {
      boardId,
      goldenReferenceId: ref.id,
      placementIndex: body.placementIndex ?? null,
      capturePath: req.file.path,
      maskPath: result.maskPath, // the real capture mask (was a made-up path to a file that never existed)
      alignedPath: result.previewPath,
      verdict: result.verdict,
      effectivePxPerMm: result.effectivePxPerMm,
      smallestDetectableMm: result.smallestDetectableMm,
      defectCount: result.defects.length + result.omittedDefects,
      alignmentScore: result.alignmentScore,
      warnings: toJson(result.warnings),
      deviceId: body.deviceId ?? null,
      operatorId: body.operatorId ?? null,
      defects: {
        create: result.defects.map((d) => ({
          bboxPx: toJson(d.bboxPx),
          bboxMm: toJson(d.bboxMm),
          kind: d.kind,
          areaPx: d.areaPx,
          confidence: d.confidence,
        })),
      },
    },
    include: { defects: true },
  });
  res.status(201).json(inspectionDto(saved));
});

router.get('/boards/:id/inspections', async (req, res) => {
  const boardId = id(req.params.id);
  await requireBoard(boardId);
  const limit = Math.min(200, Math.max(1, Number(req.query.limit) || 50));
  const rows = await prisma.inspection.findMany({
    where: { boardId },
    orderBy: { createdAt: 'desc' },
    take: limit,
    include: { goldenReference: { select: { side: true } } },
  });
  res.json(rows.map(inspectionSummaryDto));
});

router.get('/inspections/:id', async (req, res) => {
  const row = await prisma.inspection.findUnique({ where: { id: id(req.params.id) }, include: { defects: true } });
  if (!row) throw new PipelineError('NOT_FOUND', 'Inspection not found');
  res.json(inspectionDto(row));
});

export default router;
