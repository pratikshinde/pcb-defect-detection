import { PrismaClient } from './generated/prisma';

export const prisma = new PrismaClient();

// SQLite has no Json column type in Prisma 5, so structured fields are stored as JSON strings.
// Writing a raw array/object into one of them makes Prisma throw ("Expected String, provided
// Object") - which is exactly how golden-reference registration used to fail. Always go through these.
export const toJson = (value: unknown): string => JSON.stringify(value);

export function parseJson<T>(text: string | null | undefined, fallback: T): T {
  if (text === null || text === undefined || text === '') return fallback;
  try {
    return JSON.parse(text) as T;
  } catch {
    return fallback;
  }
}
