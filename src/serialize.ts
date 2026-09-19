import path from 'path';
import { parseJson } from './db';
import type { Defect, GoldenReference, Inspection } from './generated/prisma';

/**
 * Wire format (plan §3). Two bugs lived here before:
 *  - defect bboxes are stored as JSON strings (SQLite) and were returned raw, so the app's
 *    `d.bboxPx.x` was `undefined` on a string and rendered "Pos: (undefined, undefined)";
 *  - the DB row was returned as-is, leaking absolute server file paths and lacking `warnings` and
 *    `alignedPreviewUrl` that the plan's contract promises.
 */

type Box = { x: number; y: number; w: number; h: number };

/** Only /results is served over HTTP; anything else has no public URL. */
export const resultUrl = (file: string | null | undefined): string | null => (file ? `/results/${path.basename(file)}` : null);

export function defectDto(d: Defect) {
  return {
    id: d.id,
    bboxPx: parseJson<Box>(d.bboxPx, { x: 0, y: 0, w: 0, h: 0 }),
    bboxMm: parseJson<Box>(d.bboxMm, { x: 0, y: 0, w: 0, h: 0 }),
    kind: d.kind,
    areaPx: d.areaPx,
    confidence: d.confidence,
  };
}

const summary = (i: Inspection) => ({
  inspectionId: i.id,
  boardId: i.boardId,
  goldenReferenceId: i.goldenReferenceId,
  placementIndex: i.placementIndex,
  verdict: i.verdict as 'pass' | 'fail' | 'needs_review',
  effectivePxPerMm: i.effectivePxPerMm,
  smallestDetectableMm: i.smallestDetectableMm,
  defectCount: i.defectCount,
  alignmentScore: i.alignmentScore,
  alignedPreviewUrl: resultUrl(i.alignedPath),
  warnings: parseJson<string[]>(i.warnings, []),
  createdAt: i.createdAt.toISOString(),
  deviceId: i.deviceId,
  operatorId: i.operatorId,
});

/** History rows: no defect list, plus the side (from the reference used). */
export const inspectionSummaryDto = (i: Inspection & { goldenReference?: { side: string } | null }) => ({
  ...summary(i),
  side: i.goldenReference?.side ?? null,
});

export const inspectionDto = (i: Inspection & { defects: Defect[] }) => ({ ...summary(i), defects: i.defects.map(defectDto) });

export function goldenReferenceDto(g: GoldenReference) {
  return {
    goldenReferenceId: g.id,
    boardId: g.boardId,
    source: g.source,
    side: g.side,
    mirrored: g.mirrored,
    copperValue: g.copperValue,
    pxPerMm: g.pxPerMm,
    widthPx: g.widthPx,
    heightPx: g.heightPx,
    fiducials: parseJson<unknown[]>(g.fiducials, []),
    placements: parseJson<unknown[]>(g.placements, []),
    hasDrillMask: g.drillMaskPath !== null,
    registeredBy: g.registeredBy,
    registeredAt: g.registeredAt.toISOString(),
  };
}
