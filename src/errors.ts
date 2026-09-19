export type PipelineErrorCode =
  | 'BAD_REQUEST'
  | 'NOT_FOUND'
  | 'IMAGE_UNREADABLE'
  | 'REFERENCE_INVALID'
  | 'ALIGNMENT_FAILED'
  | 'RESOLUTION_TOO_LOW'
  | 'BINARIZATION_IMPLAUSIBLE';

const STATUS: Record<PipelineErrorCode, number> = {
  BAD_REQUEST: 400,
  NOT_FOUND: 404,
  IMAGE_UNREADABLE: 400,
  REFERENCE_INVALID: 422,
  ALIGNMENT_FAILED: 422,
  RESOLUTION_TOO_LOW: 422,
  BINARIZATION_IMPLAUSIBLE: 422,
};

/**
 * A pipeline stage refused to produce a result. Plan §8: a degraded artifact must be dropped with an
 * explanation rather than returned, because a confident-looking wrong result is worse than an error.
 */
export class PipelineError extends Error {
  readonly status: number;
  constructor(
    readonly code: PipelineErrorCode,
    message: string,
    readonly details?: unknown,
  ) {
    super(message);
    this.name = 'PipelineError';
    this.status = STATUS[code];
  }
}
