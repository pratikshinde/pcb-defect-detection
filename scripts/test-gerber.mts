// npx tsx scripts/test-gerber.mts <copper-gerber> [drill-excellon] [pxPerMm] [--mirror]
import fs from 'fs';
import path from 'path';
import { rasterizeGerber } from '../src/cv/gerber';

const args = process.argv.slice(2).filter((a) => a !== '--mirror');
const mirror = process.argv.includes('--mirror');
const [copperPath, second, third] = args;
if (!copperPath) {
  console.error('Usage: npx tsx scripts/test-gerber.mts <copper-gerber> [drill-excellon] [pxPerMm] [--mirror]');
  process.exit(1);
}
const drillPath = second && Number.isNaN(Number(second)) ? second : undefined;
const pxPerMm = Number(drillPath ? third : second) || 40;
try {
  const r = await rasterizeGerber({
    copperFileStr: fs.readFileSync(path.resolve(copperPath), 'utf8'),
    drillFileStr: drillPath ? fs.readFileSync(path.resolve(drillPath), 'utf8') : undefined,
    pxPerMm,
    mirror,
  });
  fs.writeFileSync('gerber_output_copper.png', r.maskBuffer);
  if (r.drillBuffer) fs.writeFileSync('gerber_output_drill.png', r.drillBuffer);
  console.log(`${r.widthPx}x${r.heightPx}px = ${r.widthMm.toFixed(2)} x ${r.heightMm.toFixed(2)} mm @ ${pxPerMm} px/mm${mirror ? ' (mirrored)' : ''}`);
  r.warnings.forEach((w) => console.warn(' !', w));
} catch (err) {
  console.error('Rasterization failed:', err instanceof Error ? err.message : err);
  process.exit(1);
}
