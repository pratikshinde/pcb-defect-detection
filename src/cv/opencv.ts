import type * as CVTypes from '@techstark/opencv-js';

export type CV = typeof CVTypes;
export type Mat = CVTypes.Mat;

let ready: Promise<CV> | undefined;

/**
 * OpenCV.js compiles its WASM asynchronously: right after `require`, `cv.Mat` is undefined and the
 * export is a thenable (measured: ~60 ms until usable). The old code called `new cv.Mat()`
 * synchronously, so it threw "Mat is not a constructor" whenever it ran before init finished -
 * always, in the CLI scripts; on a cold start, in the server. Always go through getCv().
 */
export function getCv(): Promise<CV> {
  ready ??= (async () => {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const mod = require('@techstark/opencv-js');
    const cv = typeof mod.then === 'function' ? await mod : mod;
    if (typeof cv.Mat !== 'function') {
      await new Promise<void>((resolve) => {
        cv.onRuntimeInitialized = () => resolve();
      });
    }
    return cv as CV;
  })();
  return ready;
}

type Deletable = { delete(): void };

/** Owns every Mat created during one pipeline run. WASM memory is not garbage collected, so the old
 *  code leaked every Mat on any error path; a scope frees them in `finally`. */
export class MatScope {
  private items: Deletable[] = [];

  add<T extends Deletable>(item: T): T {
    this.items.push(item);
    return item;
  }

  /** Delete one item now (rather than at the end of the run) and forget it. */
  free(item: Deletable): void {
    const i = this.items.indexOf(item);
    if (i < 0) return;
    this.items.splice(i, 1);
    try {
      item.delete();
    } catch {
      /* already freed */
    }
  }

  dispose(): void {
    for (let i = this.items.length - 1; i >= 0; i--) {
      try {
        this.items[i]!.delete();
      } catch {
        /* already freed */
      }
    }
    this.items = [];
  }
}

/**
 * The WASM heap is capped at ~1 GB (measured), and a single 12 MP float image is 47 MB. Holding every
 * intermediate until the end of a run exhausts it ("Insufficient memory"). Wrap each stage in scoped()
 * so its temporaries are freed as soon as the stage returns; only plain data (or Mats added to an
 * outer scope) may escape.
 */
export function scoped<T>(fn: (s: MatScope) => T): T {
  const s = new MatScope();
  try {
    return fn(s);
  } finally {
    s.dispose();
  }
}

/** Run `fn` with OpenCV ready and a scope that is always disposed. Return plain data only:
 *  any Mat created inside is freed when this resolves. */
export async function withCv<T>(fn: (cv: CV, scope: MatScope) => Promise<T> | T): Promise<T> {
  const cv = await getCv();
  const scope = new MatScope();
  try {
    return await fn(cv, scope);
  } catch (err) {
    throw translateCvException(cv, err);
  } finally {
    scope.dispose();
  }
}

/** OpenCV.js throws C++ exceptions as a bare WASM pointer (a number), which says nothing. */
function translateCvException(cv: CV, err: unknown): unknown {
  if (typeof err !== 'number') return err;
  try {
    const ex = (cv as unknown as { exceptionFromPtr?: (p: number) => { msg?: string } }).exceptionFromPtr?.(err);
    if (ex?.msg) return new Error(`OpenCV: ${ex.msg}`);
  } catch {
    /* fall through */
  }
  return new Error(`OpenCV threw a native exception (pointer ${err}) with no message available`);
}

export function matFromRaw(
  cv: CV,
  scope: MatScope,
  data: Uint8Array,
  width: number,
  height: number,
  channels: 1 | 3 | 4,
): Mat {
  const type = channels === 1 ? cv.CV_8UC1 : channels === 3 ? cv.CV_8UC3 : cv.CV_8UC4;
  const mat = scope.add(new cv.Mat(height, width, type));
  mat.data.set(data);
  return mat;
}

/**
 * `mat.data` is a VIEW into the WASM heap. Any later allocation can grow the heap, which detaches
 * every view taken before it ("Cannot perform %TypedArray%.prototype.set on a detached ArrayBuffer").
 * So: never hold a `.data` view across an allocation - take a copy with bytesOf()/floatsOf() first.
 */
export function bytesOf(mat: Mat): Uint8Array {
  return Uint8Array.from(mat.data);
}

export function floatsOf(mat: Mat): Float32Array {
  return Float32Array.from(mat.data32F);
}

/** Copies out of WASM memory. */
export function bufferFromMat(mat: Mat): Buffer {
  return Buffer.from(mat.data);
}
