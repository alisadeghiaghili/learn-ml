/**
 * Level registry — worlds compose into the rail and `levels` command.
 */

import type { Level } from "../engine/types";
import { WORLD1_LEVELS } from "./world1";
import { WORLD2_LEVELS } from "./world2";
import { WORLD3_LEVELS } from "./world3";
import { WORLD4_LEVELS } from "./world4";
import { WORLD5_LEVELS } from "./world5";
import { WORLD6_LEVELS } from "./world6";
import { WORLD7_LEVELS } from "./world7";
import { WORLD8_LEVELS } from "./world8";
import { WORLD9_LEVELS } from "./world9";
import { WORLD10_LEVELS } from "./world10";
import { WORLD11_LEVELS } from "./world11";
import { WORLD12_LEVELS } from "./world12";
import { WORLD13_LEVELS } from "./world13";

export const ALL_LEVELS: Level[] = [
  ...WORLD1_LEVELS,
  ...WORLD2_LEVELS,
  ...WORLD3_LEVELS,
  ...WORLD4_LEVELS,
  ...WORLD5_LEVELS,
  ...WORLD6_LEVELS,
  ...WORLD7_LEVELS,
  ...WORLD8_LEVELS,
  ...WORLD9_LEVELS,
  ...WORLD10_LEVELS,
  ...WORLD11_LEVELS,
  ...WORLD12_LEVELS,
  ...WORLD13_LEVELS,
];

export function findLevel(id: string): Level | undefined {
  return ALL_LEVELS.find((l) => l.id === id);
}

export {
  WORLD1_LEVELS,
  WORLD2_LEVELS,
  WORLD3_LEVELS,
  WORLD4_LEVELS,
  WORLD5_LEVELS,
  WORLD6_LEVELS,
  WORLD7_LEVELS,
  WORLD8_LEVELS,
  WORLD9_LEVELS,
  WORLD10_LEVELS,
  WORLD11_LEVELS,
  WORLD12_LEVELS,
  WORLD13_LEVELS,
};
