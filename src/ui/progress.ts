/** Progress persistence: localStorage + cookie so a new session resumes. */

import { ALL_LEVELS } from "../levels";

export const STORAGE_KEY = "learn-ml-progress-v1";
export const COOKIE_KEY = "learn_ml_progress";
const COOKIE_MAX_AGE = 60 * 60 * 24 * 400;

export interface LevelProgress {
  solved: boolean;
  bestCommands?: number;
  solvedAt?: string;
}

export interface PersistBlob {
  progress: Record<string, LevelProgress>;
  savedAt?: string;
}

export interface CurriculumItem {
  id: string;
  name: string;
  worldTitle: string;
  learning: string[];
  bestCommands?: number;
}

export interface CurriculumSummary {
  solvedCount: number;
  total: number;
  learned: CurriculumItem[];
  remaining: CurriculumItem[];
  next: CurriculumItem | null;
  percent: number;
}

function readCookie(): string | null {
  if (typeof document === "undefined") return null;
  const parts = document.cookie.split(";");
  for (const part of parts) {
    const [rawKey, ...rest] = part.trim().split("=");
    if (rawKey !== COOKIE_KEY) continue;
    try {
      return decodeURIComponent(rest.join("="));
    } catch {
      return rest.join("=");
    }
  }
  return null;
}

function writeCookie(payload: string): void {
  if (typeof document === "undefined") return;
  const encoded = encodeURIComponent(payload);
  document.cookie = `${COOKIE_KEY}=${encoded}; path=/; max-age=${COOKIE_MAX_AGE}; SameSite=Lax`;
}

function parseBlob(raw: string | null): Record<string, LevelProgress> | null {
  if (!raw) return null;
  try {
    const parsed = JSON.parse(raw) as PersistBlob | Record<string, LevelProgress>;
    if (parsed && typeof parsed === "object" && "progress" in parsed) {
      return (parsed as PersistBlob).progress ?? {};
    }
    return parsed as Record<string, LevelProgress>;
  } catch {
    return null;
  }
}

export function loadProgress(): Record<string, LevelProgress> {
  let fromLocal: Record<string, LevelProgress> | null = null;
  let fromCookie: Record<string, LevelProgress> | null = null;
  try {
    fromLocal = parseBlob(localStorage.getItem(STORAGE_KEY));
  } catch {
    fromLocal = null;
  }
  try {
    fromCookie = parseBlob(readCookie());
  } catch {
    fromCookie = null;
  }

  const merged: Record<string, LevelProgress> = {};
  for (const src of [fromCookie ?? {}, fromLocal ?? {}]) {
    for (const [id, prog] of Object.entries(src)) {
      if (!prog) continue;
      const prev = merged[id];
      merged[id] = {
        solved: Boolean(prog.solved || prev?.solved),
        bestCommands:
          prev?.bestCommands === undefined
            ? prog.bestCommands
            : prog.bestCommands === undefined
              ? prev.bestCommands
              : Math.min(prev.bestCommands, prog.bestCommands),
        solvedAt: prog.solvedAt ?? prev?.solvedAt,
      };
    }
  }
  return merged;
}

export function saveProgress(progress: Record<string, LevelProgress>): void {
  const blob: PersistBlob = {
    progress,
    savedAt: new Date().toISOString(),
  };
  const payload = JSON.stringify(blob);
  try {
    localStorage.setItem(STORAGE_KEY, payload);
  } catch {
    // private mode — cookie still helps
  }
  writeCookie(payload);
}

export function summarizeCurriculum(
  progress: Record<string, LevelProgress>,
): CurriculumSummary {
  const learned: CurriculumItem[] = [];
  const remaining: CurriculumItem[] = [];
  let next: CurriculumItem | null = null;

  for (const level of ALL_LEVELS) {
    const item: CurriculumItem = {
      id: level.id,
      name: level.title,
      worldTitle: level.worldTitle,
      learning: [...(level.learning ?? [])],
      bestCommands: progress[level.id]?.bestCommands,
    };
    if (progress[level.id]?.solved) {
      learned.push(item);
    } else {
      remaining.push(item);
      if (!next) next = item;
    }
  }

  return {
    solvedCount: learned.length,
    total: ALL_LEVELS.length,
    learned,
    remaining,
    next,
    percent: ALL_LEVELS.length
      ? Math.round((learned.length / ALL_LEVELS.length) * 100)
      : 0,
  };
}

export function resumeLine(summary: CurriculumSummary): string {
  if (!summary.solvedCount) {
    return `No saved progress yet (${summary.total} levels waiting). Start with \`levels\`.`;
  }
  const learnedTitles = summary.learned
    .map((l) => `${l.name} (${l.id})`)
    .join(" · ");
  const nextText = summary.next
    ? `Next up: ${summary.next.name} — open level ${summary.next.id}`
    : "All levels cleared.";
  return [
    `Welcome back — progress saved: ${summary.solvedCount}/${summary.total} levels (${summary.percent}%).`,
    `Learned so far: ${learnedTitles}`,
    nextText,
  ].join("\n");
}
