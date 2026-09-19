/**
 * End-to-end verification with KNOWN ground truth (plan §9 Phase 2: "validated against deliberately
 * defective sample pairs ... a known-good board must come back clean").
 *
 *   npm run verify
 *
 * Every check is a named oracle. Synthetic photographs are rendered from the real Top_Copper.png design,
 * so what is measured is the pipeline on the real copper pattern - only the camera is simulated. Writes
 * annotated previews to a temp directory (printed at the end) for eyeballing, like the plan's other scripts.
 */
import fs from 'fs';
import os from 'os';
import path from 'path';
import sharp from 'sharp';
import type { Quad } from '../src/cv/align';
import { runInspection, type CVInspectionResult } from '../src/cv/comparison';
import { rasterizeGerber } from '../src/cv/gerber';
import { PipelineError } from '../src/errors';
import { findSites, injectDefects } from './lib/sites';
import { renderPhoto } from './lib/synth';

const AOI = path.resolve(import.meta.dirname, '../../data/aoi');
const TMP = fs.mkdtempSync(path.join(os.tmpdir(), 'pcb-aoi-verify-'));
let failures = 0;
let checks = 0;
const check = (name: string, ok: boolean, detail = '') => {
  checks++;
  if (!ok) failures++;
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? `\n        ${detail}` : ''}`);
};
const section = (t: string) => console.log(`\n=== ${t} ===`);
const time = async <T,>(fn: () => Promise<T>): Promise<[T, number]> => {
  const t = Date.now();
  const r = await fn();
  return [r, (Date.now() - t) / 1000];
};

// ---- the real design ---------------------------------------------------------------------------
const manifest = JSON.parse(fs.readFileSync(path.join(AOI, 'aoi.json'), 'utf8'));
const PPM: number = manifest.pxPerMm;
const P = manifest.placements[0].px.top as { x: number; y: number; w: number; h: number };
const TOP = path.join(AOI, 'Top_Copper.png');
const reference = { maskPath: TOP, copperValue: 1 as const, pxPerMm: PPM, source: 'gerber' };

// The shipped raster stores copper as BLACK although aoi.json declares copperValue=1 (measured against a
// photographed board). The photographs below are therefore rendered from the raster's black pixels.
const { data: designRaw } = await sharp(TOP).extract({ left: P.x, top: P.y, width: P.w, height: P.h }).greyscale().raw().toBuffer({ resolveWithObject: true });
const design = new Uint8Array(designRaw.length);
for (let i = 0; i < design.length; i++) design[i] = designRaw[i] === 0 ? 255 : 0;

const inspect = (photo: string, opts: { corners?: Quad; ref?: typeof reference; target?: typeof P; stem: string }) =>
  runInspection({ capturePath: photo, reference: opts.ref ?? reference, target: opts.target ?? P, corners: opts.corners, outputStem: path.join(TMP, opts.stem) });

const center = (d: CVInspectionResult['defects'][number]) => ({ x: d.bboxMm.x + d.bboxMm.w / 2, y: d.bboxMm.y + d.bboxMm.h / 2 });
const distMm = (a: { x: number; y: number }, b: { xMm: number; yMm: number }) => Math.hypot(a.x - b.xMm, a.y - b.yMm);

// =====================================================================================================
section('1. Known-good board at the plan\'s intended resolution (35 px/mm)');
const good35 = path.join(TMP, 'good35.jpg');
const goodShot = await renderPhoto(design, P.w, P.h, PPM, { pxPerMm: 35, seed: 11, blurPx: 1.0, rotationDeg: 4, underEtchMm: 0 }, good35);
{
  const [r, secs] = await time(() => inspect(good35, { corners: goodShot.corners, stem: 'good35' }));
  console.log(`   ${secs.toFixed(1)}s  eff ${r.effectivePxPerMm.toFixed(1)} px/mm  align ${r.alignmentScore.toFixed(3)}  bias ${r.widthBiasMm.toFixed(3)} mm  detectable ${r.smallestDetectableMm.toFixed(2)} mm  defects ${r.defects.length + r.omittedDefects}`);
  check('known-good board: no credible defects (negative case)', r.defects.filter((d) => d.credible).length === 0, r.defects.map((d) => `${d.kind} ${d.areaMm2.toFixed(2)}mm2 thick ${d.thicknessMm.toFixed(2)}`).join('; '));
  check('known-good board: PASS (plan §9 "a known-good board must come back clean")', r.verdict === 'pass', `verdict ${r.verdict}; ${r.warnings.filter((w) => !w.includes('OPPOSITE')).join(' | ')}`);
  check('alignment is tight (correlation >= 0.9)', r.alignmentScore >= 0.9, String(r.alignmentScore));
  check('resolution is MEASURED from the capture (34-36 px/mm), not copied from the 40 px/mm reference', r.effectivePxPerMm > 32 && r.effectivePxPerMm < 37, String(r.effectivePxPerMm));
  check('detection limit reflects the capture (< 0.25 mm at 35 px/mm)', r.smallestDetectableMm < 0.25, `${r.smallestDetectableMm.toFixed(3)} mm`);
  check('raster/manifest polarity contradiction is detected and reported (aoi.json says copper=white; the raster is copper=black)', r.warnings.some((w) => w.includes('OPPOSITE')));
  check('no drill mask -> blind spot is stated, not hidden', r.warnings.some((w) => /drill/i.test(w)));
}

// =====================================================================================================
section('2. Injected defects on the real design - are they actually caught?');
const sites = await findSites(design, P.w, P.h, PPM);
const injected = injectDefects(design, P.w, P.h, PPM, sites);
for (const t of injected.truth) console.log(`   injected ${t.kind.padEnd(6)} at (${t.xMm.toFixed(1)}, ${t.yMm.toFixed(1)}) mm  - ${t.note}`);
const bad35 = path.join(TMP, 'bad35.jpg');
const badShot = await renderPhoto(injected.mask, P.w, P.h, PPM, { pxPerMm: 35, seed: 12, blurPx: 1.0, rotationDeg: -3, underEtchMm: 0 }, bad35);
{
  const [r, secs] = await time(() => inspect(bad35, { corners: badShot.corners, stem: 'bad35' }));
  console.log(`   ${secs.toFixed(1)}s  verdict ${r.verdict}  defects ${r.defects.length + r.omittedDefects} (credible ${r.defects.filter((d) => d.credible).length})`);
  for (const d of r.defects.filter((x) => x.credible)) console.log(`     ${d.kind.padEnd(12)} ${d.areaMm2.toFixed(2)} mm2  at (${center(d).x.toFixed(1)}, ${center(d).y.toFixed(1)}) mm`);
  const credible = r.defects.filter((d) => d.credible);
  const truth = (k: string) => injected.truth.find((t) => t.kind === k)!;
  const hit = (k: string, kinds: string[], within = 2.0) => credible.find((d) => kinds.includes(d.kind) && distMm(center(d), truth(k)) <= within);

  const open = hit('open', ['open']);
  check('severed trace is found, at the right place, classified OPEN', !!open, open ? `${distMm(center(open), truth('open')).toFixed(2)} mm off` : 'not found');
  const shrt = hit('short', ['short']);
  check('bridge between two nets is found and classified SHORT (confirmed via the reference net map)', !!shrt, shrt ? `${distMm(center(shrt), truth('short')).toFixed(2)} mm off` : 'not found');
  const isl = hit('island', ['unclassified', 'short']);
  check('stray copper island is found (extra copper touching no net)', !!isl, isl ? `${distMm(center(isl), truth('island')).toFixed(2)} mm off` : 'not found');
  check('a SHORT makes the verdict FAIL', r.verdict === 'fail', r.verdict);
  const stray = credible.filter((d) => injected.truth.every((t) => distMm(center(d), t) > 3));
  check('no credible defects away from the injected ones (false-positive check)', stray.length === 0, stray.map((d) => `${d.kind} at (${center(d).x.toFixed(1)},${center(d).y.toFixed(1)}) ${d.areaMm2.toFixed(2)}mm2`).join('; '));
  const voidNear = r.defects.find((d) => distMm(center(d), truth('void')) <= 1.5);
  check(
    'a 0.6 mm round void inside solid copper is indistinguishable from a drilled hole without a drill mask: it is suppressed AND that is reported',
    !voidNear && r.warnings.some((w) => /assumed to be drilled holes/.test(w)),
    voidNear ? `unexpectedly listed as ${voidNear.kind}` : r.warnings.find((w) => /assumed to be drilled holes/.test(w))?.slice(0, 110) ?? 'no warning',
  );
}

// =====================================================================================================
section('3. Resolution honesty: a low-resolution capture must never read as strong evidence (plan §6)');
const good20 = path.join(TMP, 'good20.jpg');
const good20Shot = await renderPhoto(design, P.w, P.h, PPM, { pxPerMm: 20, seed: 13, blurPx: 0.7, rotationDeg: 2, underEtchMm: 0 }, good20);
{
  const r = await inspect(good20, { corners: good20Shot.corners, stem: 'good20' });
  console.log(`   eff ${r.effectivePxPerMm.toFixed(1)} px/mm  detectable ${r.smallestDetectableMm.toFixed(2)} mm  verdict ${r.verdict}`);
  check('a clean 20 px/mm capture is NOT passed (below the 30 px/mm floor -> needs_review)', r.verdict === 'needs_review', r.verdict);
  check('and it says why', r.warnings.some((w) => /Low resolution/.test(w)) && r.warnings.some((w) => /Not passed/.test(w)));
  check('detection limit is stated honestly (>= 0.25 mm at 20 px/mm)', r.smallestDetectableMm >= 0.25, `${r.smallestDetectableMm.toFixed(3)} mm`);
  check('no false defects at low resolution either', r.defects.filter((d) => d.credible).length === 0);
}

// =====================================================================================================
section('4. Phone-photo hazards');
{
  // EXIF: Android writes portrait shots sideways plus an orientation tag. sharp ignores the tag unless
  // asked to auto-orient, so the old code processed the photo on its side.
  const exifPhoto = path.join(TMP, 'good35_exif.jpg');
  await sharp(good35).rotate(-90).withMetadata({ orientation: 6 }).jpeg({ quality: 90 }).toFile(exifPhoto);
  const meta = await sharp(exifPhoto).metadata();
  const r = await inspect(exifPhoto, { corners: goodShot.corners, stem: 'good35_exif' });
  check('EXIF-rotated photo (stored sideways + orientation tag) is auto-oriented and inspects clean', meta.orientation === 6 && r.alignmentScore > 0.9 && r.defects.filter((d) => d.credible).length === 0, `stored orientation tag ${meta.orientation}; align ${r.alignmentScore.toFixed(3)}`);
}
{
  // Wrong side: a mirrored board is what you get by comparing the bottom of a board to the top reference
  const mirrored = new Uint8Array(design.length);
  for (let y = 0; y < P.h; y++) for (let x = 0; x < P.w; x++) mirrored[y * P.w + x] = design[y * P.w + (P.w - 1 - x)]!;
  const mir = path.join(TMP, 'mirrored35.jpg');
  const mirShot = await renderPhoto(mirrored, P.w, P.h, PPM, { pxPerMm: 35, seed: 14, blurPx: 1.0, rotationDeg: 2 }, mir);
  let verdict: string | undefined;
  let code: string | undefined;
  try {
    verdict = (await inspect(mir, { corners: mirShot.corners, stem: 'mirrored35' })).verdict;
  } catch (e) {
    code = e instanceof PipelineError ? e.code : String(e);
  }
  check('mirrored (wrong-side) board is refused or failed - never passed (plan §7)', verdict !== 'pass' && (code === 'ALIGNMENT_FAILED' || verdict === 'fail'), `verdict=${verdict} error=${code}`);
}
for (const [label, make] of [
  ['blank grey image', () => sharp({ create: { width: 3000, height: 4000, channels: 3, background: { r: 128, g: 128, b: 128 } } }).jpeg().toBuffer()],
  [
    'random noise',
    async () => {
      const n = Buffer.alloc(1500 * 2000 * 3);
      let s = 12345;
      for (let i = 0; i < n.length; i++) n[i] = ((s = (Math.imul(s, 1664525) + 1013904223) >>> 0) >>> 24) & 255;
      return sharp(n, { raw: { width: 1500, height: 2000, channels: 3 } }).jpeg().toBuffer();
    },
  ],
] as const) {
  const f = path.join(TMP, `${label.replace(/\W+/g, '_')}.jpg`);
  fs.writeFileSync(f, await make());
  let threw: string | undefined;
  let verdict: string | undefined;
  try {
    verdict = (await inspect(f, { corners: [[0.15, 0.15], [0.85, 0.15], [0.85, 0.85], [0.15, 0.85]], stem: 'garbage' })).verdict;
  } catch (e) {
    threw = e instanceof PipelineError ? e.code : `UNEXPECTED ${e}`;
  }
  check(`${label} -> rejected with a reason, never a verdict (plan §8: silently-degraded artifacts)`, !!threw && !threw.startsWith('UNEXPECTED') && verdict === undefined, `error=${threw} verdict=${verdict}`);
}

// =====================================================================================================
section('5. Whole panel, sideways, found automatically (no corners)');
{
  const { data: panelRaw } = await sharp(TOP).greyscale().raw().toBuffer({ resolveWithObject: true });
  const panel = new Uint8Array(panelRaw.length);
  for (let i = 0; i < panel.length; i++) panel[i] = panelRaw[i] === 0 ? 255 : 0;
  const W = manifest.images[0].widthPx as number;
  const H = manifest.images[0].heightPx as number;
  const sideways = path.join(TMP, 'panel_sideways.jpg');
  await renderPhoto(panel, W, H, PPM, { pxPerMm: 10, seed: 21, blurPx: 0.8, rotationDeg: 90, perspective: 0.02 }, sideways);
  const [r, secs] = await time(() => inspect(sideways, { target: { x: 0, y: 0, w: W, h: H }, stem: 'panel' }));
  console.log(`   ${secs.toFixed(1)}s  method ${r.alignmentMethod}  align ${r.alignmentScore.toFixed(3)}  eff ${r.effectivePxPerMm.toFixed(1)} px/mm  verdict ${r.verdict}  defects ${r.defects.length + r.omittedDefects}`);
  check('panel found by its copper outline amid clutter, with the 90-degree turn resolved (no EXIF tag)', r.alignmentMethod === 'auto-outline' && r.alignmentScore > 0.6, `align ${r.alignmentScore.toFixed(3)}`);
  check('10 px/mm panel capture is not passed', r.verdict !== 'pass', r.verdict);
  check('and no credible false defects on a clean panel', r.defects.filter((d) => d.credible).length === 0, `${r.defects.filter((d) => d.credible).length} credible`);
}

// =====================================================================================================
section('6. Drill mask (plan §5.4) on a small design with correct polarity');
{
  const MW = 60 * 40;
  const MH = 40 * 40;
  const copper = new Uint8Array(MW * MH);
  const holes = new Uint8Array(MW * MH);
  const px = (mm: number) => Math.round(mm * 40);
  const rect = (a: Uint8Array, x0: number, y0: number, x1: number, y1: number) => {
    for (let y = px(y0); y <= px(y1); y++) for (let x = px(x0); x <= px(x1); x++) a[y * MW + x] = 255;
  };
  const disc = (a: Uint8Array, cx: number, cy: number, r: number, v: number) => {
    for (let y = px(cy - r); y <= px(cy + r); y++) for (let x = px(cx - r); x <= px(cx + r); x++) if ((x - px(cx)) ** 2 + (y - px(cy)) ** 2 <= px(r) ** 2) a[y * MW + x] = v;
  };
  rect(copper, 1.5, 1.5, 58.5, 2); rect(copper, 1.5, 38, 58.5, 38.5); rect(copper, 1.5, 1.5, 2, 38.5); rect(copper, 58, 1.5, 58.5, 38.5); // frame
  for (let i = 0; i < 5; i++) rect(copper, 18 + i * 5, 6, 18.4 + i * 5, 34); // 0.4 mm traces
  rect(copper, 10, 9.8, 50, 10.2); rect(copper, 10, 29.8, 50, 30.2);
  rect(copper, 30, 14, 33, 15); rect(copper, 12, 18, 15, 22); rect(copper, 44, 20, 46, 27); // asymmetric features
  const pads: [number, number][] = [[10, 10], [50, 10], [10, 30], [50, 30]];
  for (const [x, y] of pads) disc(copper, x, y, 1.5, 255);
  for (const [x, y] of pads) disc(holes, x, y, 0.5, 255);
  const board = Uint8Array.from(copper);
  for (let i = 0; i < board.length; i++) if (holes[i]) board[i] = 0; // the physical board has holes through the pads
  const write = async (name: string, a: Uint8Array) => {
    const f = path.join(TMP, name);
    await sharp(Buffer.from(a), { raw: { width: MW, height: MH, channels: 1 } }).png().toFile(f);
    return f;
  };
  const copperPng = await write('mini_copper.png', copper);
  const drillPng = await write('mini_drill.png', holes);
  const photo = path.join(TMP, 'mini_photo.jpg');
  const shot = await renderPhoto(board, MW, MH, 40, { pxPerMm: 35, seed: 31, blurPx: 1.0, rotationDeg: 3 }, photo);
  const target = { x: 0, y: 0, w: MW, h: MH };
  const run = (drill: string | null, source: string, stem: string) =>
    runInspection({ capturePath: photo, reference: { maskPath: copperPng, drillMaskPath: drill, copperValue: 1, pxPerMm: 40, source }, target, corners: shot.corners, outputStem: path.join(TMP, stem) });

  const withDrill = await run(drillPng, 'gerber', 'mini_drill_on');
  check('with a drill mask: the 4 drilled pads are not defects and the board passes', withDrill.verdict === 'pass' && withDrill.defects.length === 0, `${withDrill.verdict}, ${withDrill.defects.length} defects; ${withDrill.warnings.join(' | ').slice(0, 120)}`);
  check('polarity POSITIVE CONTROL: a correctly-encoded reference raises no polarity warning', !withDrill.warnings.some((w) => w.includes('OPPOSITE')));
  check('with a drill mask there is no drill-mask caveat', !withDrill.warnings.some((w) => /No drill mask/.test(w)));
  const noDrillGerber = await run(null, 'gerber', 'mini_drill_off');
  check('without one (Gerber source): round voids inside pads are assumed to be holes, and the count is reported', noDrillGerber.defects.filter((d) => d.credible).length === 0 && noDrillGerber.warnings.some((w) => /\(4 ignored in this capture\)/.test(w)), noDrillGerber.warnings.find((w) => /drill/i.test(w))?.slice(0, 130) ?? 'no warning');
  const noDrillPhoto = await run(null, 'photo', 'mini_drill_photo');
  const holesAsDefects = noDrillPhoto.defects.filter((d) => d.kind === 'open' && d.credible).length;
  check('WITHOUT any drill handling every drilled pad reads as missing copper - the failure plan §5.4 warns about', holesAsDefects >= 4, `${holesAsDefects} credible 'open' defects, verdict ${noDrillPhoto.verdict}`);

  // wrong design entirely
  let wrong: string | undefined;
  try {
    await inspect(photo, { corners: shot.corners, stem: 'wrong_design' });
  } catch (e) {
    wrong = e instanceof PipelineError ? e.code : `UNEXPECTED ${e}`;
  }
  check('a photo of a DIFFERENT board is refused, not compared', wrong === 'ALIGNMENT_FAILED', `error=${wrong}`);
}

// =====================================================================================================
section('7. Gerber rasteriser against hand-computed geometry (was: 1000x scale error, drill on a different frame)');
{
  const copper = ['%FSLAX34Y34*%', '%MOMM*%', '%ADD10C,1.000*%', 'D10*', 'X100000Y100000D02*', 'X1100000Y100000D01*', 'X100000Y500000D02*', 'X1100000Y500000D01*', 'M02*'].join('\n');
  const drill = ['M48', 'METRIC,TZ', 'T1C0.800', '%', 'T1', 'X20.0Y10.0', 'X60.0Y50.0', 'M30'].join('\n');
  const r = await rasterizeGerber({ copperFileStr: copper, drillFileStr: drill, pxPerMm: 10 });
  const cu = await sharp(r.maskBuffer).greyscale().raw().toBuffer();
  const dr = await sharp(r.drillBuffer!).greyscale().raw().toBuffer();
  const at = (b: Buffer, xmm: number, ymm: number) => b[Math.round((50.5 - ymm) * 10) * r.widthPx + Math.round((xmm - 9.5) * 10)];
  check('raster is 101 x 41 mm at 10 px/mm = 1010 x 410 px', r.widthPx === 1010 && r.heightPx === 410, `${r.widthPx}x${r.heightPx}`);
  check('both holes land ON the copper lines, in the same frame', at(dr, 20, 10) === 255 && at(dr, 60, 50) === 255 && at(cu, 20, 10) === 255 && at(cu, 60, 50) === 255 && at(dr, 40, 30) === 0);
  let n = 0;
  for (let dy = -8; dy <= 8; dy++) for (let dx = -8; dx <= 8; dx++) if (dr[(Math.round((50.5 - 10) * 10) + dy) * r.widthPx + Math.round((20 - 9.5) * 10) + dx]) n++;
  check('a 0.8 mm hole is ~50 px (pi * (0.4 mm * 10 px/mm)^2)', n >= 44 && n <= 58, `${n} px`);
  const m = await rasterizeGerber({ copperFileStr: copper, pxPerMm: 10, mirror: true });
  const mc = await sharp(m.maskBuffer).greyscale().raw().toBuffer();
  check('mirror=true flips left-right (bottom side, plan §7)', mc[5 * m.widthPx + 5] === 255 && mc[5 * m.widthPx + m.widthPx - 6] === 255 && m.widthPx === 1010);
  let empty: string | undefined;
  try {
    await rasterizeGerber({ copperFileStr: ['%FSLAX34Y34*%', '%MOMM*%', 'M02*'].join('\n'), pxPerMm: 10 });
  } catch (e) {
    empty = e instanceof PipelineError ? e.code : String(e);
  }
  check('an empty Gerber is refused rather than registered as a blank reference', empty !== undefined);
}

// =====================================================================================================
section('8. The plan\'s primary flow: golden PHOTO vs test PHOTO (registration -> inspection)');
{
  const { registerGoldenReference } = await import('../src/cv/registration');
  const reg = await registerGoldenReference({ imagePath: good35, widthMm: P.w / PPM, heightMm: P.h / PPM, corners: goodShot.corners, pxPerMm: 35, outputStem: path.join(TMP, 'photo_golden') });
  console.log(`   registered ${reg.widthPx}x${reg.heightPx}px @ ${reg.pxPerMm.toFixed(1)} px/mm, copper coverage ${(reg.coverage * 100).toFixed(1)}%, ${reg.method}`);
  check('registration reports NO fabricated fiducials (was: three invented points returned as "found")', reg.fiducialsFound.length === 0);
  check('registered resolution is measured from the photo, not assumed 40 px/mm', reg.pxPerMm > 32 && reg.pxPerMm < 37, `${reg.pxPerMm.toFixed(2)}`);
  const photoRef = { maskPath: reg.maskPath, copperValue: 1 as const, pxPerMm: reg.pxPerMm, source: 'photo' };
  const tgt = { x: 0, y: 0, w: reg.widthPx, h: reg.heightPx };
  const clean = await inspect(good35, { ref: photoRef, target: tgt, corners: goodShot.corners, stem: 'photo_vs_self' });
  check('the golden photo inspected against itself: pass, alignment ~1', clean.verdict === 'pass' && clean.alignmentScore > 0.97, `${clean.verdict} ${clean.alignmentScore.toFixed(3)}`);
  const bad = await inspect(bad35, { ref: photoRef, target: tgt, corners: badShot.corners, stem: 'photo_vs_bad' });
  const cred = bad.defects.filter((d) => d.credible);
  const near = (k: string, kinds: string[]) => cred.find((d) => kinds.includes(d.kind) && distMm(center(d), injected.truth.find((t) => t.kind === k)!) <= 2);
  check('photo-vs-photo: verdict FAIL on the defective board', bad.verdict === 'fail', bad.verdict);
  check('photo-vs-photo: open, short and island all found in place', !!near('open', ['open']) && !!near('short', ['short']) && !!near('island', ['unclassified', 'short']), cred.map((d) => `${d.kind}@(${center(d).x.toFixed(1)},${center(d).y.toFixed(1)})`).join(' '));
  check('photo-vs-photo: no polarity warning (photo goldens are copper=255 by construction)', !bad.warnings.some((w) => w.includes('OPPOSITE')));
}

// ------------------------------------------------------------------------------------------------
console.log(`\n${failures === 0 ? 'ALL' : failures + ' OF'} ${checks} CHECKS ${failures === 0 ? 'PASSED' : 'FAILED'}`);
console.log(`Artifacts (annotated previews, synthetic photos): ${TMP}`);
process.exit(failures === 0 ? 0 : 1);
