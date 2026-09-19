// npx tsx scripts/test-registration.mts <photo> <widthMm> <heightMm> [outDir]
import path from 'path';
import { registerGoldenReference } from '../src/cv/registration';

const [image, w, h, out] = process.argv.slice(2);
if (!image || !w || !h) {
  console.error('Usage: npx tsx scripts/test-registration.mts <photo> <widthMm> <heightMm> [outDir]');
  process.exit(1);
}
const outDir = path.resolve(out ?? '.');
try {
  const r = await registerGoldenReference({ imagePath: path.resolve(image), widthMm: Number(w), heightMm: Number(h), outputStem: path.join(outDir, 'registration_test') });
  console.log(JSON.stringify({ ...r, warnings: undefined }, null, 2));
  r.warnings.forEach((x) => console.log(' -', x));
  console.log(`Mask: ${r.maskPath}\nRectified preview: ${r.previewPath}`);
} catch (err) {
  console.error('Registration failed:', err instanceof Error ? `${(err as { code?: string }).code ?? ''} ${err.message}` : err);
  process.exit(1);
}
