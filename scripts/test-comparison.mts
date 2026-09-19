// npx tsx scripts/test-comparison.mts <capture> <reference-mask> [--drill f] [--copper-value 0|1] [--px-per-mm 40] [--source gerber|photo] [--out dir]
import path from 'path';
import sharp from 'sharp';
import { runInspection } from '../src/cv/comparison';

const args = process.argv.slice(2);
const flag = (name: string, fallback?: string) => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 ? args[i + 1] : fallback;
};
const positional = args.filter((a, i) => !a.startsWith('--') && !(i > 0 && args[i - 1]!.startsWith('--')));
const [capture, reference] = positional;
if (!capture || !reference) {
  console.error('Usage: npx tsx scripts/test-comparison.mts <capture> <reference-mask> [--drill f] [--copper-value 0|1] [--px-per-mm 40] [--source gerber|photo] [--out dir]');
  process.exit(1);
}
const meta = await sharp(path.resolve(reference)).metadata();
try {
  const drill = flag('drill');
  const r = await runInspection({
    capturePath: path.resolve(capture),
    reference: {
      maskPath: path.resolve(reference),
      drillMaskPath: drill ? path.resolve(drill) : null,
      copperValue: flag('copper-value', '1') === '0' ? 0 : 1,
      pxPerMm: Number(flag('px-per-mm', '40')),
      source: flag('source', 'gerber')!,
    },
    target: { x: 0, y: 0, w: meta.width!, h: meta.height! },
    outputStem: path.join(path.resolve(flag('out', '.')!), 'inspection_test'),
  });
  console.log(`Verdict: ${r.verdict.toUpperCase()}   defects: ${r.defects.length}${r.omittedDefects ? ` (+${r.omittedDefects} not listed)` : ''}`);
  console.log(`Capture ${r.effectivePxPerMm.toFixed(1)} px/mm; smallest detectable ${r.smallestDetectableMm.toFixed(2)} mm; alignment ${r.alignmentScore.toFixed(3)} (${r.alignmentMethod})`);
  console.log(`Width bias ${r.widthBiasMm.toFixed(3)} mm/side; blur ${r.opticalBlurMm.toFixed(3)} mm`);
  r.defects.slice(0, 15).forEach((d, i) =>
    console.log(` [${i + 1}] ${d.kind.padEnd(12)} ${d.areaMm2.toFixed(2)} mm2  thickness ${d.thicknessMm.toFixed(2)} mm ${d.credible ? 'CREDIBLE' : 'sliver'}  at (${d.bboxMm.x.toFixed(1)}, ${d.bboxMm.y.toFixed(1)}) mm`),
  );
  r.warnings.forEach((w) => console.log(' !', w));
  console.log(`Annotated preview: ${r.previewPath}`);
} catch (err) {
  console.error('Inspection failed:', err instanceof Error ? `${(err as { code?: string }).code ?? ''} ${err.message}` : err);
  process.exit(1);
}
