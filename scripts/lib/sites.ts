/**
 * Finds places in a REAL copper pattern where a known defect can be injected, and applies it.
 * Everything is deterministic, so the ground truth is reproducible run to run.
 *
 * Coordinates are pixels of the mask (copper = 255). Divide by the mask's px/mm for millimetres from the
 * region's top-left, y down - the same frame the pipeline reports defects in (bboxMm).
 */
import { getCv, scoped } from '../../src/cv/opencv';

export interface Site {
  x: number;
  y: number;
  note: string;
}

export interface DefectSites {
  /** a thin vertical trace, to be cut through */
  open: Site;
  /** the gap between two DIFFERENT nets, to be bridged */
  short: Site;
  /** empty substrate, to receive stray copper */
  island: Site;
  /** the interior of solid copper, to lose a disc of it */
  voidSite: Site;
}

const at = (m: Uint8Array, w: number, x: number, y: number) => m[y * w + x]! > 127;

export async function findSites(mask: Uint8Array, w: number, h: number, pxPerMm: number): Promise<DefectSites> {
  const cv = await getCv();
  const margin = Math.round(8 * pxPerMm); // stay off the board edge
  const cx = w / 2;
  const cy = h / 2;
  const dist = (x: number, y: number) => Math.hypot(x - cx, y - cy);
  const minTrace = Math.round(0.4 * pxPerMm);
  const maxTrace = Math.round(0.7 * pxPerMm);
  const reach = Math.round(2 * pxPerMm); // required straight extent either side of an OPEN site
  // Bridging a gap narrower than ~0.4 mm is not credibly detectable at 35 px/mm (the tolerance band swallows
  // most of it), so shorts are injected across >= 0.5 mm gaps. Progressively relaxed until a site exists.
  const shortPasses = [
    { minGap: 0.5, maxGap: 1.0, minRun: 0.3, reach: 0.9 },
    { minGap: 0.5, maxGap: 1.6, minRun: 0.25, reach: 0.6 },
    { minGap: 0.45, maxGap: 2.5, minRun: 0.2, reach: 0.4 },
  ];

  // connected components of copper: the two sides of a short must be DIFFERENT nets
  const { labels, dtCopper, dtSub } = scoped((s) => {
    const m = s.add(new cv.Mat(h, w, cv.CV_8UC1));
    m.data.set(mask);
    const lab = s.add(new cv.Mat());
    cv.connectedComponents(m, lab, 8, cv.CV_32S);
    const inv = s.add(new cv.Mat());
    cv.bitwise_not(m, inv);
    const dc = s.add(new cv.Mat());
    const ds = s.add(new cv.Mat());
    cv.distanceTransform(m, dc, cv.DIST_L2, cv.DIST_MASK_5);
    cv.distanceTransform(inv, ds, cv.DIST_L2, cv.DIST_MASK_5);
    return { labels: Int32Array.from(lab.data32S), dtCopper: Float32Array.from(dc.data32F), dtSub: Float32Array.from(ds.data32F) };
  });

  const straightCopperRun = (x0: number, x1: number, y: number, r = reach) => {
    for (let dy = -r; dy <= r; dy += 3) for (const x of [x0 + 1, (x0 + x1) >> 1, x1 - 1]) if (!at(mask, w, x, y + dy)) return false;
    return true;
  };

  let open = null as (Site & { d: number }) | null; // assigned inside a closure: `as` stops TS narrowing it to `never`
  let short = null as (Site & { d: number }) | null;

  const scanRows = (pass: (typeof shortPasses)[number] | null) => {
    for (let y = margin; y < h - margin; y += 6) {
      let x = margin;
      while (x < w - margin) {
        if (!at(mask, w, x, y)) {
          x++;
          continue;
        }
        let x1 = x;
        while (x1 < w - margin && at(mask, w, x1, y)) x1++;
        const len = x1 - x;
        if (!pass) {
          if (len >= minTrace && len <= maxTrace && straightCopperRun(x, x1 - 1, y)) {
            const d = dist((x + x1) / 2, y);
            if (!open || d < open.d) open = { x: Math.round((x + x1) / 2), y, d, note: `${(len / pxPerMm).toFixed(2)} mm trace` };
          }
        } else {
          const r = Math.round(pass.reach * pxPerMm);
          const minRun = Math.round(pass.minRun * pxPerMm);
          let g = x1;
          while (g < w - margin && !at(mask, w, g, y)) g++;
          const gap = g - x1;
          if (gap >= Math.round(pass.minGap * pxPerMm) && gap <= Math.round(pass.maxGap * pxPerMm) && len >= minRun) {
            let x2 = g;
            while (x2 < w - margin && at(mask, w, x2, y)) x2++;
            if (x2 - g >= minRun && straightCopperRun(x, x1 - 1, y, r) && straightCopperRun(g, x2 - 1, y, r)) {
              let gapClear = true;
              for (let dy = -r; dy <= r && gapClear; dy += 3) if (at(mask, w, (x1 + g) >> 1, y + dy)) gapClear = false;
              const la = labels[y * w + ((x + x1) >> 1)]!;
              const lb = labels[y * w + ((g + x2) >> 1)]!;
              if (gapClear && la > 0 && lb > 0 && la !== lb) {
                const d = dist((x1 + g) / 2, y);
                if (!short || d < short.d) short = { x: Math.round((x1 + g) / 2), y, d, note: `${(gap / pxPerMm).toFixed(2)} mm gap between nets ${la}/${lb}` };
              }
            }
          }
        }
        x = x1 + 1;
      }
    }
  };
  scanRows(null);
  for (const pass of shortPasses) {
    if (short) break;
    scanRows(pass);
  }

  const bestPixel = (score: Float32Array, minScore: number): Site | null => {
    let best: { x: number; y: number; d: number } | null = null;
    for (let y = margin; y < h - margin; y += 8) {
      for (let x = margin; x < w - margin; x += 8) {
        if (score[y * w + x]! >= minScore) {
          const d = dist(x, y);
          if (!best || d < best.d) best = { x, y, d };
        }
      }
    }
    return best ? { x: best.x, y: best.y, note: '' } : null;
  };
  const island = bestPixel(dtSub, 1.1 * pxPerMm);
  const voidSite = bestPixel(dtCopper, 0.7 * pxPerMm);

  if (!open || !short || !island || !voidSite) {
    throw new Error(`could not find defect sites (open=${!!open} short=${!!short} island=${!!island} void=${!!voidSite})`);
  }
  return {
    open: { x: open.x, y: open.y, note: open.note },
    short: { x: short.x, y: short.y, note: short.note },
    island: { ...island, note: 'isolated substrate' },
    voidSite: { ...voidSite, note: 'inside solid copper' },
  };
}

export interface InjectedDefects {
  mask: Uint8Array;
  /** ground truth in mm from the region's top-left, y down */
  truth: { kind: 'open' | 'short' | 'island' | 'void'; xMm: number; yMm: number; note: string }[];
}

/** Draws each defect into a copy of the mask. Sizes are chosen to be judgeable at >= 30 px/mm. */
export function injectDefects(mask: Uint8Array, w: number, h: number, pxPerMm: number, sites: DefectSites): InjectedDefects {
  const out = Uint8Array.from(mask);
  const fillRect = (x0: number, y0: number, x1: number, y1: number, v: number) => {
    for (let y = Math.max(0, y0); y <= Math.min(h - 1, y1); y++) for (let x = Math.max(0, x0); x <= Math.min(w - 1, x1); x++) out[y * w + x] = v;
  };
  const fillDisc = (cx: number, cy: number, r: number, v: number) => {
    for (let y = Math.max(0, cy - r); y <= Math.min(h - 1, cy + r); y++) for (let x = Math.max(0, cx - r); x <= Math.min(w - 1, cx + r); x++) if ((x - cx) ** 2 + (y - cy) ** 2 <= r * r) out[y * w + x] = v;
  };
  const half = Math.round(0.6 * pxPerMm); // 1.2 mm along the trace
  const inRun = (x: number, y: number, copper: boolean) => (mask[y * w + x]! > 127) === copper;
  const extent = (x: number, y: number, copper: boolean): [number, number] => {
    let a = x;
    let b = x;
    while (a > 0 && inRun(a - 1, y, copper)) a--;
    while (b < w - 1 && inRun(b + 1, y, copper)) b++;
    return [a, b];
  };
  // open: erase exactly the trace's width (+2 px) so neighbouring copper is untouched - a wider cut
  // makes a rectangular hole in a copper region, which is a different (and ambiguous) defect
  const [ox0, ox1] = extent(sites.open.x, sites.open.y, true);
  fillRect(ox0 - 2, sites.open.y - half, ox1 + 2, sites.open.y + half, 0);
  // short: copper across exactly the gap, overlapping 3 px into each conductor
  const [gx0, gx1] = extent(sites.short.x, sites.short.y, false);
  fillRect(gx0 - 3, sites.short.y - half, gx1 + 3, sites.short.y + half, 255);
  // island: stray copper, 0.9 mm across
  fillDisc(sites.island.x, sites.island.y, Math.round(0.45 * pxPerMm), 255);
  // void: 0.6 mm disc lost from solid copper
  fillDisc(sites.voidSite.x, sites.voidSite.y, Math.round(0.3 * pxPerMm), 0);

  const mm = (s: Site) => ({ xMm: s.x / pxPerMm, yMm: s.y / pxPerMm });
  return {
    mask: out,
    truth: [
      { kind: 'open', ...mm(sites.open), note: sites.open.note },
      { kind: 'short', ...mm(sites.short), note: sites.short.note },
      { kind: 'island', ...mm(sites.island), note: sites.island.note },
      { kind: 'void', ...mm(sites.voidSite), note: sites.voidSite.note },
    ],
  };
}
