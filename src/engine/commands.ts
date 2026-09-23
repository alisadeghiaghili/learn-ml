/**
 * Command DSL parser and dispatcher for the terminal dock.
 */

import { describeDatasets, DATASET_NAMES } from "./datasets";
import { MODEL_NAMES, modelSklearnName, defaultParams } from "./estimators";
import type { CommandResult, Session } from "./session";
import type { DatasetName, ModelName, ScalerName } from "./types";

export interface DispatchExtras {
  onShowLevels?: () => string[];
  onShowGoal?: () => string[];
  onHint?: () => string[];
  onRunLevel?: (id: string) => string[];
  onShow?: (what: string) => string[];
}

const HELP_LINES = [
  "LearnML commands (scikit-learn mental model)",
  "",
  "  load <dataset>           load a built-in toy dataset",
  "  split test_size=0.2      hold out a test set (do this before scale/fit)",
  "  scale [standard|minmax]  fit a scaler on train only after split",
  "  fit <model> [k=v ...]    fit an estimator on train",
  "  predict                  predict the test set",
  "  score [train|test]       compute metrics",
  "  cm                       confusion matrix (classification)",
  "  show data|pipeline|metrics|code",
  "  goal | hint | levels | run <id> | reset | undo | clear | help",
  "",
  `datasets: ${DATASET_NAMES.join(", ")}`,
  `models:   ${MODEL_NAMES.join(", ")}`,
  "",
  "Each command prints the scikit-learn equivalent below the dock.",
];

function parseKwargs(tokens: string[]): Record<string, number | string> {
  const out: Record<string, number | string> = {};
  for (const tok of tokens) {
    const eq = tok.indexOf("=");
    if (eq <= 0) {
      throw new Error(`Expected key=value, got '${tok}'`);
    }
    const key = tok.slice(0, eq);
    const raw = tok.slice(eq + 1);
    const num = Number(raw);
    out[key] = Number.isFinite(num) && raw.trim() !== "" && !Number.isNaN(num) && /^-?\d+(\.\d+)?$/.test(raw)
      ? num
      : raw;
  }
  return out;
}

function ok(lines: string[], snapshot: CommandResult["snapshot"], sklearn: string | null = null): CommandResult {
  return { lines, sklearn, status: "ok", snapshot };
}

/**
 * Execute one command line against a Session.
 *
 * Args:
 *   session: Mutable sandbox session.
 *   line: Raw user input.
 *   extras: UI callbacks for levels/goal/hint/show.
 *
 * Returns:
 *   CommandResult with echo lines, optional sklearn equivalent, and snapshot.
 *
 * Raises:
 *   Error: On unknown commands or invalid arguments (caught by the terminal UI).
 *
 * Examples:
 *   >>> dispatch(session, "load blobs").lines[0]
 *   'Loaded blobs'
 */
export function dispatch(
  session: Session,
  line: string,
  extras: DispatchExtras = {},
): CommandResult {
  const trimmed = line.trim();
  if (!trimmed) {
    return ok([], session.snapshot());
  }
  const tokens = trimmed.split(/\s+/);
  const cmd = (tokens[0] ?? "").toLowerCase();
  const args = tokens.slice(1);

  switch (cmd) {
    case "help":
    case "?":
      return ok([...HELP_LINES], session.snapshot());
    case "levels":
      return ok(extras.onShowLevels?.() ?? ["No level list wired."], session.snapshot());
    case "goal":
      return ok(extras.onShowGoal?.() ?? ["No goal wired."], session.snapshot());
    case "hint":
      return ok(extras.onHint?.() ?? ["No hints wired."], session.snapshot());
    case "run": {
      const id = args[0] ?? "";
      return ok(extras.onRunLevel?.(id) ?? [`Unknown level '${id}'`], session.snapshot());
    }
    case "show": {
      const what = args[0] ?? "data";
      if (what === "data") {
        session.markInspected();
      }
      return ok(extras.onShow?.(what) ?? [`show ${what}: not wired`], session.snapshot());
    }
    case "clear":
    case "reset":
      return session.reset();
    case "undo":
      return session.undo();
    case "load": {
      const name = args[0] as DatasetName;
      if (!DATASET_NAMES.includes(name)) {
        throw new Error(`Unknown dataset '${args[0] ?? ""}'. Try: ${DATASET_NAMES.join(", ")}`);
      }
      return session.load(name);
    }
    case "split": {
      const kw = parseKwargs(args);
      const testSize = Number(kw.test_size ?? 0.2);
      const seed = Number(kw.seed ?? kw.random_state ?? 42);
      return session.split_(testSize, seed);
    }
    case "scale": {
      const raw = (args[0] ?? "standard").toLowerCase();
      if (raw !== "standard" && raw !== "minmax") {
        throw new Error("scale accepts `standard` or `minmax`");
      }
      return session.scale(raw as ScalerName);
    }
    case "fit": {
      const name = (args[0] ?? "").toLowerCase() as ModelName;
      if (!MODEL_NAMES.includes(name)) {
        throw new Error(
          `Unknown model '${args[0] ?? ""}'. Try: ${MODEL_NAMES.join(", ")}`,
        );
      }
      const params = parseKwargs(args.slice(1));
      return session.fit(name, params);
    }
    case "predict":
      return session.predict();
    case "cm":
    case "confusion":
      return session.showConfusion();
    case "score": {
      const on = (args[0] ?? "test").toLowerCase();
      if (on !== "train" && on !== "test") {
        throw new Error("score accepts `train` or `test`");
      }
      return session.score(on);
    }
    case "sklearn":
    case "code": {
      return ok(sklearnCheatSheet(), session.snapshot());
    }
    default:
      throw new Error(`Unknown command '${cmd}'. Type 'help'.`);
  }
}

function sklearnCheatSheet(): string[] {
  const lines = ["scikit-learn map", ""];
  for (const name of MODEL_NAMES) {
    const params = defaultParams(name);
    const py = Object.entries(params)
      .map(([k, v]) => (typeof v === "string" ? `${k}='${v}'` : `${k}=${v}`))
      .join(", ");
    lines.push(`  fit ${name}  →  ${modelSklearnName(name)}(${py}).fit(X_train, y_train)`);
  }
  lines.push("");
  for (const d of describeDatasets()) {
    lines.push(`  load ${d.name}  →  # ${d.task}: ${d.blurb}`);
  }
  return lines;
}
