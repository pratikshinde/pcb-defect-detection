import path from 'path';

const BACKEND_ROOT = path.resolve(__dirname, '..');

function num(name: string, fallback: number): number {
  const raw = process.env[name];
  if (raw === undefined || raw.trim() === '') return fallback;
  const n = Number(raw);
  if (!Number.isFinite(n)) throw new Error(`Environment variable ${name} must be a number, got "${raw}"`);
  return n;
}

function oneOf<T extends string>(name: string, allowed: readonly T[], fallback: T): T {
  const raw = process.env[name];
  if (raw === undefined || raw.trim() === '') return fallback;
  if (!(allowed as readonly string[]).includes(raw)) {
    throw new Error(`Environment variable ${name} must be one of ${allowed.join(' | ')}, got "${raw}"`);
  }
  return raw as T;
}

/** One place for every path. Previously the multer dir, the Gerber route and the seed script each
 *  resolved `uploads` differently (backend/uploads vs the repo-root uploads). */
const uploads = path.resolve(process.env.UPLOAD_DIR ?? path.join(BACKEND_ROOT, 'uploads'));
export const paths = {
  backendRoot: BACKEND_ROOT,
  uploads,
  golden: path.join(uploads, 'golden'),
  results: path.join(uploads, 'results'), // the ONLY directory served over HTTP
};

export const BINARIZE_CHANNELS = ['gray', 'red', 'green', 'value'] as const;
export type BinarizeChannel = (typeof BINARIZE_CHANNELS)[number];

export const cvConfig = {
  // --- resolution gates (plan §6) ---------------------------------------------------------
  /** Below this a would-be "pass" is downgraded to needs_review: the capture cannot resolve fine features. */
  minPxPerMm: num('CV_MIN_PX_PER_MM', 30),
  /** Below this the capture is rejected outright - nothing meaningful can be measured. */
  refusePxPerMm: num('CV_REFUSE_PX_PER_MM', 8),
  /** Never work above this resolution: it costs memory and the capture rarely has the detail anyway. */
  maxWorkPxPerMm: num('CV_MAX_WORK_PX_PER_MM', 40),

  // --- binarisation (plan §4.5, shared by registration and inspection per §5.3) -----------
  channel: oneOf('CV_BINARIZE_CHANNEL', BINARIZE_CHANNELS, 'gray'),
  /** Plan §4.6: a mask outside this copper-coverage band is a failed artifact, not a result. */
  coverageMin: num('CV_COVERAGE_MIN', 0.03),
  coverageMax: num('CV_COVERAGE_MAX', 0.9),
  /** Minimum copper-vs-substrate level separation (0-255) for a binarisation to be trusted. */
  minContrast: num('CV_MIN_CONTRAST', 25),
  /** Scale of the illumination estimate. Must exceed the widest copper/substrate feature. */
  illuminationSigmaMm: num('CV_ILLUMINATION_SIGMA_MM', 6),

  // --- diff / defects (plan §5.4) ---------------------------------------------------------
  /** Opening kernel on the diff maps. Must stay smaller than the smallest defect you need to catch. */
  morphKernel: num('CV_MORPH_KERNEL_SIZE', 3),
  /** Edge tolerance band. Etched edges are rough and registration is imperfect, so a raw XOR flags
   *  every trace edge; features smaller than this are deliberately not judged. */
  toleranceMm: num('CV_TOLERANCE_MM', 0.05),
  minDefectAreaPx: num('CV_MIN_DEFECT_AREA_PX', 10),
  /** Without a drill mask, drilled pads read as round voids inside reference copper. 'auto' = ignore them
   *  only for Gerber-derived references that have no drill mask (photo references already contain the holes). */
  suppressEnclosedHoles: oneOf('CV_SUPPRESS_ENCLOSED_HOLES', ['auto', 'on', 'off'] as const, 'auto'),
  maxDefects: num('CV_MAX_DEFECTS', 200),

  // --- verdict ----------------------------------------------------------------------------
  /** Global trace-width offset vs the reference (per side) above which a clean "pass" is not issued. */
  maxWidthBiasMm: num('CV_MAX_WIDTH_BIAS_MM', 0.05),
  failTotalAreaMm2: num('CV_FAIL_TOTAL_AREA_MM2', 0.5),
  failDefectCount: num('CV_FAIL_DEFECT_COUNT', 3),

  // --- alignment --------------------------------------------------------------------------
  /** Correlation between the aligned capture mask and the reference below which alignment is rejected. */
  minAlignmentScore: num('CV_MIN_ALIGNMENT_SCORE', 0.3),
};
