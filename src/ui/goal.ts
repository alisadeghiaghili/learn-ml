import type { SessionSnapshot } from "../engine/types";

/** UI-facing goal checklist step (mirrors Level.steps). */
export interface UiGoalStep {
  id: string;
  label: string;
  detail: string;
  command?: string;
  check: (state: SessionSnapshot) => boolean;
}

export type { SessionSnapshot };
