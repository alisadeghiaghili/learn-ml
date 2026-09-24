/**
 * World 4 — Real tables (impute, encode, pipeline order).
 */

import type { ClassificationMetrics, Level, SessionSnapshot, WinResult } from "../engine/types";

function fail(feedback: string): WinResult {
  return { won: false, feedback };
}

function win(feedback = "Clear. Concept locked."): WinResult {
  return { won: true, feedback };
}

function isClass(
  m: SessionSnapshot["metrics"] | SessionSnapshot["trainMetrics"],
): m is ClassificationMetrics {
  return Boolean(m && "accuracy" in m && "confusion" in m);
}

export const WORLD4_LEVELS: Level[] = [
  {
    id: "4.1",
    world: "w4",
    worldTitle: "Real tables",
    title: "Missing values are data",
    concept: {
      title: "Impute from train, never from the future",
      body:
        "mixed_table has ~12% missing spend. Dropping rows throws away signal; filling with 0 invents poverty. SimpleImputer estimates mean/median from train and fills both train and test with *those* numbers. If you fit the imputer on all rows, test distribution is inside your features again — quiet leakage.",
      whatHappens:
        "`load mixed_table` → `split` → `impute mean` (fits on train only) → `fit logistic` → `score test`. The pipeline graph shows SimpleImputer before the estimator.",
      why:
        "Missingness is often informative. For v1 you learn the hygiene (train-only statistics). Production adds missingness indicators and model-native handling.",
      callout: "Every statistic that touches the data is part of the model.",
    },
    goal: "On mixed_table: split, impute mean (train stats), fit logistic, test accuracy ≥ 0.7.",
    hints: [
      "`load mixed_table` → `split` → `impute mean` → `fit logistic` → `score test`",
    ],
    learning: ["SimpleImputer", "train-only statistics", "missing data hygiene"],
    seedDataset: "mixed_table",
    steps: [
      {
        id: "split",
        label: "Split before any statistic",
        detail: "Protect the future.",
        command: "split test_size=0.25",
        check: (s) => Boolean(s.split),
      },
      {
        id: "impute",
        label: "Impute from train",
        detail: "mean/median/constant from train only.",
        command: "impute mean",
        check: (s) => Boolean(s.imputed),
      },
      {
        id: "fit",
        label: "Fit logistic and score test",
        detail: "Accuracy ≥ 0.70.",
        command: "score test",
        check: (s) =>
          s.fitted &&
          isClass(s.metrics) &&
          s.scoredOn === "test" &&
          s.metrics.accuracy >= 0.7,
      },
    ],
    win: (s) => {
      if (!s.split) return fail("Split first.");
      if (!s.imputed) return fail("Run `impute mean` (or median) after the split.");
      if (s.scaleLeaked) return fail("Scaler leaked. Keep every fitted step after split.");
      if (!isClass(s.metrics) || s.scoredOn !== "test") return fail("`score test` after fit.");
      if (s.metrics.accuracy < 0.7) {
        return fail(`Test accuracy ${s.metrics.accuracy.toFixed(3)} < 0.70. Check impute + fit order.`);
      }
      return win("Missingness handled with train statistics. No peeking.");
    },
  },
  {
    id: "4.2",
    world: "w4",
    worldTitle: "Real tables",
    title: "Categories are not integers",
    concept: {
      title: "One-hot beats fake ordinals",
      body:
        "region ∈ {0,1,2,3} is not a quantity. Linear models treat 3 as 'three times 1'. OneHotEncoder turns it into orthogonal flags so 'west' is a direction, not a size. On mixed_table, encoding is what lets logistic use region without inventing order.",
      whatHappens:
        "`impute` then `encode onehot` then `fit logistic`. The design matrix grows columns per region level. Score test — compare to leaving region as a fake integer (skipping encode).",
      why:
        "Wrong encoding is a silent geometry bug. Trees tolerate integers; distance and linear models do not. Encode for the estimator you ship.",
      callout: "Country code 840 is not twice 420.",
    },
    goal: "On mixed_table: impute + encode onehot + logistic, test accuracy ≥ 0.75.",
    hints: [
      "`split` → `impute mean` → `encode onehot` → `fit logistic` → `score test`",
    ],
    learning: ["OneHotEncoder", "categorical geometry", "ColumnTransformer mental model"],
    seedDataset: "mixed_table",
    steps: [
      {
        id: "prep",
        label: "Split + impute",
        detail: "Clean numerics first.",
        command: "impute mean",
        check: (s) => Boolean(s.split) && Boolean(s.imputed),
      },
      {
        id: "enc",
        label: "One-hot the region column",
        detail: "Orthogonal flags, not fake order.",
        command: "encode onehot",
        check: (s) => s.encoded === "onehot",
      },
      {
        id: "score",
        label: "Logistic test accuracy ≥ 0.75",
        detail: "Encoding must pay rent.",
        command: "score test",
        check: (s) =>
          s.model === "logistic" &&
          isClass(s.metrics) &&
          s.scoredOn === "test" &&
          s.metrics.accuracy >= 0.75,
      },
    ],
    win: (s) => {
      if (s.encoded !== "onehot") return fail("Run `encode onehot` — integers lie to linear models.");
      if (!s.imputed) return fail("Impute before fitting.");
      if (!isClass(s.metrics) || s.scoredOn !== "test") return fail("`score test`.");
      if (s.metrics.accuracy < 0.75) {
        return fail(`Test accuracy ${s.metrics.accuracy.toFixed(3)} < 0.75.`);
      }
      return win("Categoricals encoded without fake order. That is a real preprocessing column.");
    },
  },
  {
    id: "4.3",
    world: "w4",
    worldTitle: "Real tables",
    title: "Pipeline order is a safety property",
    concept: {
      title: "The order of transforms is the product",
      body:
        "Correct pipeline: split → impute(train) → encode → scale(train) → fit(train) → score(test). Swap two steps and you either leak or learn garbage geometry. Production sklearn `Pipeline` / `ColumnTransformer` exist to freeze this order in code review.",
      whatHappens:
        "Build the full chain on mixed_table with scaler after encode. The left pipeline graph is the reviewable artifact. Any red edge means a failed property, not a style nit.",
      why:
        "Teams do not fail ML because they chose the wrong loss. They fail because a transform saw the hold-out. Encode that as a checklist item forever.",
      formula: "split → impute → encode → scale → fit → score(test)",
      callout: "If a step estimates anything, it is fit on train only.",
    },
    goal: "Full clean chain on mixed_table (impute+encode+scale+knn or logistic) with test accuracy ≥ 0.75 and no leak flag.",
    hints: [
      "`split` → `impute mean` → `encode onehot` → `scale standard` → `fit logistic` → `score test`",
    ],
    learning: ["end-to-end pipeline order", "ColumnTransformer as a contract", "leak-free production checklist"],
    seedDataset: "mixed_table",
    steps: [
      {
        id: "split",
        label: "Split first",
        detail: "Always.",
        command: "split test_size=0.25",
        check: (s) => Boolean(s.split) && !s.scaleLeaked,
      },
      {
        id: "prep",
        label: "Impute + encode + scale",
        detail: "All stats from train.",
        command: "scale standard",
        check: (s) => Boolean(s.imputed) && Boolean(s.encoded) && Boolean(s.scaled) && !s.scaleLeaked,
      },
      {
        id: "fit",
        label: "Fit and score test ≥ 0.75",
        detail: "One honest number.",
        command: "score test",
        check: (s) =>
          s.fitted &&
          isClass(s.metrics) &&
          s.scoredOn === "test" &&
          s.metrics.accuracy >= 0.75 &&
          !s.scaleLeaked,
      },
    ],
    win: (s) => {
      if (s.scaleLeaked) return fail("Leakage. Split before every fitted transform.");
      if (!s.imputed || !s.encoded || !s.scaled) return fail("Complete impute + encode + scale after split.");
      if (!isClass(s.metrics) || s.scoredOn !== "test") return fail("`score test` at the end.");
      if (s.metrics.accuracy < 0.75) {
        return fail(`Test accuracy ${s.metrics.accuracy.toFixed(3)} < 0.75.`);
      }
      return win("Full leak-free chain. This is the artifact a reviewer should demand.");
    },
  },
];
