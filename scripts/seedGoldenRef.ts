/**
 * Registers the portal's AOI package as golden references.
 *
 *   npm run seed                      # reads ../data/aoi
 *   npm run seed -- path/to/aoi-dir
 *
 * Idempotent and non-destructive: running it twice creates nothing new, and it never deletes a
 * reference (the old script did `deleteMany`, which fails on the foreign key from any existing
 * Inspection - and would have destroyed the audit trail if it hadn't).
 */
import path from 'path';
import { ingestAoi, loadAoiFromDir } from '../src/aoi/ingest';
import { prisma } from '../src/db';

async function main() {
  const dir = path.resolve(process.argv[2] ?? path.join(__dirname, '../../data/aoi'));
  console.log(`Reading AOI package from ${dir}`);
  const pkg = loadAoiFromDir(dir);
  const { job } = pkg.manifest;

  const name = [job?.partNumber, job?.revision ? `Rev ${job.revision}` : undefined].filter(Boolean).join(' ') || 'AOI board';
  let board = await prisma.board.findFirst({ where: { name } });
  if (!board) {
    board = await prisma.board.create({ data: { name, customerRef: job?.customer ?? null, revision: job?.revision ?? null } });
    console.log(`Created board #${board.id} "${name}"`);
  } else {
    console.log(`Using existing board #${board.id} "${name}"`);
  }

  const result = await ingestAoi(pkg, { boardId: board.id, sides: ['top', 'bottom'], registeredBy: 'seed' });
  for (const r of result.references) {
    console.log(`  ${r.side.padEnd(6)} ref #${r.id}  ${r.widthPx}x${r.heightPx}px @ ${r.pxPerMm} px/mm  copperValue=${r.copperValue}  drill=${r.drillMaskPath ? 'yes' : 'no'}`);
  }
  console.log(result.created > 0 ? `Registered ${result.created} new reference(s).` : 'Already registered - nothing to do.');
  for (const w of result.warnings) console.warn(`  ! ${w}`);
}

main()
  .catch((e) => {
    console.error(e);
    process.exit(1);
  })
  .finally(() => prisma.$disconnect());
