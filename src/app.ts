import express, { type ErrorRequestHandler } from 'express';
import fs from 'fs';
import multer from 'multer';
import { ZodError } from 'zod';
import { paths } from './config';
import { PipelineError } from './errors';
import routes from './routes';

/** Built as a function so tests can start the real HTTP layer on an ephemeral port. */
export function createApp() {
  for (const dir of [paths.uploads, paths.golden, paths.results]) fs.mkdirSync(dir, { recursive: true });

  const app = express();
  app.disable('x-powered-by');

  // The Expo web preview runs on another origin. No auth exists yet (plan §11.1 is undecided), so this is
  // opt-out in production rather than a security boundary.
  const origin = process.env.CORS_ORIGIN ?? (process.env.NODE_ENV === 'production' ? '' : '*');
  if (origin) {
    app.use((req, res, next) => {
      res.setHeader('Access-Control-Allow-Origin', origin);
      res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
      res.setHeader('Access-Control-Allow-Methods', 'GET,POST,OPTIONS');
      if (req.method === 'OPTIONS') {
        res.sendStatus(204);
        return;
      }
      next();
    });
  }

  app.use(express.json());
  app.use(express.urlencoded({ extended: true }));

  app.get('/health', (_req, res) => {
    res.json({ status: 'ok', uptime: process.uptime() });
  });

  // ONLY annotated results/previews are public. Uploads (customer boards) and golden references are not.
  app.use('/results', express.static(paths.results, { index: false, dotfiles: 'deny', maxAge: '1h' }));

  app.use('/', routes);

  const onError: ErrorRequestHandler = (err, _req, res, _next) => {
    if (err instanceof PipelineError) {
      res.status(err.status).json({ error: err.message, code: err.code, details: err.details });
      return;
    }
    if (err instanceof ZodError) {
      res.status(400).json({ error: 'Invalid request', code: 'BAD_REQUEST', details: err.issues.map((i) => `${i.path.join('.') || 'body'}: ${i.message}`) });
      return;
    }
    if (err instanceof multer.MulterError) {
      res.status(err.code === 'LIMIT_FILE_SIZE' ? 413 : 400).json({ error: `Upload rejected: ${err.message}`, code: 'BAD_REQUEST' });
      return;
    }
    // A real bug. It used to be reported as a 400, indistinguishable from a bad request.
    console.error('Unhandled error:', err);
    res.status(500).json({
      error: 'Internal Server Error',
      code: 'INTERNAL',
      details: process.env.NODE_ENV === 'production' ? undefined : (err as Error)?.message,
    });
  };
  app.use(onError);
  return app;
}
