/**
 * World 12 — Metrics pack and leakage-aware encodings.
 */

import type { ClassificationMetrics, Level, SessionSnapshot, WinResult } from "../engine/types";

function fail(feedback: string): WinResult {
  return { won: false, feedback };
}

function win(feedback = "Clear. Concept locked."): WinResult {
  return { won: true, feedback };
}

function isClass(m: SessionSnapshot["metrics"] | SessionSnapshot["trainMetrics"]): m is ClassificationMetrics {
  return Boolean(m && "accuracy" in m && "rocAuc" in m);
}

export const WORLD12_LEVELS: Level[] = [
  {
    id: "12.1",
    world: "w12",
    worldTitle: "Metrics pack",
    title: "ROC-AUC and PR-AUC",
    concept: {
      title: "Threshold-free ranking quality",
      body:
        "ROC-AUC is the probability a random positive scores above a random negative (Mann–Whitney). PR-AUC focuses on the positive class under imbalance. Accuracy is one point; AUC is the ranking. Use ROC-AUC when both classes matter; PR-AUC when positives are rare and you care about precision.",
      whatHappens:
        "Score logistic on moons and read `roc_auc` and `pr_auc` on the metrics panel. Then `roc` for the threshold curve. Prefer the metric that matches the cost structure before you tune.",
      why:
        "Leaderboards optimize the wrong scalar all the time. Declare the ranking metric up front.",
      formula: "ROC-AUC = P(score⁺ > score⁻)  ·  PR-AUC = area under precision–recall",
      callout: "AUC is a ranking claim. Calibration is a different claim.",
    },
    goal: "On moons: score test (roc_auc printed) and run `roc`.",
    hints: [
      "`load moons` → `split` → `fit logistic` → `score test` → `roc`",
    ],
    learning: ["ROC-AUC", "PR-AUC", "ranking vs threshold"],
    seedDataset: "moons",
    steps: [
      {
        id: "score",
        label: "Score test with AUCs",
        detail: "roc_auc + pr_auc.",
        command: "score test",
        check: (s) => isClass(s.metrics) && s.metrics.rocAuc > 0.5 && s.metrics.prAuc > 0,
      },
      {
        id: "roc",
        label: "Threshold sweep",
        detail: "Connect AUC to product cutoff.",
        command: "roc",
        check: (s) => s.inspectedRoc,
      },
    ],
    win: (s) => {
      if (!isClass(s.metrics)) return fail("Score a classifier first.");
      return win(
        `roc_auc=${s.metrics.rocAuc.toFixed(3)} pr_auc=${s.metrics.prAuc.toFixed(3)}. Ranking is not accuracy.`,
      );
    },
  },
  {
    id: "12.2",
    world: "w12",
    worldTitle: "Metrics pack",
    title: "Target encoding without leakage",
    concept: {
      title: "y in a feature is radioactive",
      body:
        "TargetEncoder replaces a category with the train mean of y. It is powerful and dangerous: fit it on all rows and test labels leak into features. Correct: fit encoder on train only (`fe target` after split). Regularization/smoothing in production also helps on rare categories.",
      whatHappens:
        "On mixed_table: split → `fe target` → `fit logistic` → `score test`. If you FE before split in spirit (stats on all data), the protocol fails. Our FE fit uses train only when split exists.",
      why:
        "Target encoding is the #1 silent leakage in Kaggle-to-prod pipelines. Protocol is the product.",
      callout: "If the feature knows y, the split must protect y.",
    },
    goal: "On mixed_table: split then `fe target`, fit logistic, score test.",
    hints: [
      "`load mixed_table` → `split` → `impute mean` → `fe target` → `fit logistic` → `score test`",
    ],
    learning: ["TargetEncoder", "train-only encoding stats", "rare category risk"],
    seedDataset: "mixed_table",
    steps: [
      {
        id: "fe",
        label: "Target encode after split",
        detail: "Train means only.",
        command: "fe target",
        check: (s) => s.feMode === "target" && Boolean(s.split),
      },
      {
        id: "score",
        label: "Fit and score test",
        detail: "No y leakage.",
        command: "score test",
        check: (s) => s.fitted && s.scoredOn === "test" && s.feMode === "target",
      },
    ],
    win: (s) => {
      if (s.feMode !== "target") return fail("Run `fe target` after split.");
      if (s.scaleLeaked) return fail("Leakage elsewhere in the chain.");
      if (s.scoredOn !== "test") return fail("`score test` at the end.");
      return win("Target encoding with train-only statistics. Radioactive, but contained.");
    },
  },
  {
    id: "12.3",
    world: "w12",
    worldTitle: "Metrics pack",
    title: "Binning changes geometry",
    concept: {
      title: "Discretization is an assumption",
      body:
        "KBinsDiscretizer turns a smooth feature into ordinal bins. Linear models then fit piecewise-constant effects; trees may not care. `fe bin` fits quantile bins on train and appends the bin index. Expect test metrics to move — that is a modeling choice, not free lunch.",
      whatHappens:
        "On noisy_line or poly_curve: `fe bin` → fit linear → score test. Binning can stabilize outliers at the cost of resolution.",
      why:
        "Feature engineering is hypothesis injection. Every bin edge is a claim about discontinuities.",
      callout: "Bins are assumptions with footnotes.",
    },
    goal: "On poly_curve: split, `fe bin`, fit linear, score test.",
    hints: [
      "`load poly_curve` → `split` → `fe bin` → `fit linear` → `score test`",
    ],
    learning: ["binning", "quantile discretization", "FE as assumptions"],
    seedDataset: "poly_curve",
    steps: [
      {
        id: "fe",
        label: "Bin the feature",
        detail: "Quantile bins on train.",
        command: "fe bin",
        check: (s) => s.feMode === "bin",
      },
      {
        id: "score",
        label: "Fit + score test",
        detail: "Measure the choice.",
        command: "score test",
        check: (s) => s.fitted && s.scoredOn === "test",
      },
    ],
    win: (s) => {
      if (s.feMode !== "bin") return fail("`fe bin` after split.");
      if (s.scoredOn !== "test") return fail("`score test`.");
      return win("You discretized on purpose and measured it. FE is design, not decoration.");
    },
  },
];
