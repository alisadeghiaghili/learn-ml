/**
 * Level registry — worlds compose into the rail and `levels` command.
 */

import type { Level } from "../engine/types";
import { WORLD1_LEVELS } from "./world1";
import { WORLD2_LEVELS } from "./world2";

export const ALL_LEVELS: Level[] = [...WORLD1_LEVELS, ...WORLD2_LEVELS];

export function findLevel(id: string): Level | undefined {
  return ALL_LEVELS.find((l) => l.id === id);
}

export { WORLD1_LEVELS, WORLD2_LEVELS };
