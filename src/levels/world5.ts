/**
 * World 5 — Trees split the space; forests average the drama.
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

export const WORLD5_LEVELS: Level[] = [
  {
    id: "5.1",
    world: "w5",
    worldTitle: "Trees and ensembles",
    title: "Depth is capacity",
    concept: {
      title: "Axis-aligned splits cannot see diagonals for free",
      body:
        "A decision tree recursively thresholds one feature at a time. max_depth is the capacity knob: depth 1 is a stump (high bias), depth 20 memorizes (high variance). On moons, a deep tree wins train and stumbles on test near label noise — same overfitting signature as k=1.",
      whatHappens:
        "fit tree max_depth=1 vs max_depth=6. Score both on train/test. The stage shows a piecewise constant decision surface (staircase), not a smooth curve.",
      why:
        "Trees are the default baseline on tabular data. Knowing *which* knob controls capacity prevents cargo-cult `max_depth=None`.",
      callout: "Depth is not quality. Depth is budget.",
    },
    goal: "On moons: fit tree max_depth=1 and max_depth=6; show the deeper tree has higher train acc and explain the test gap.",
    hints: [
      "`load moons` → `split` → `fit tree max_depth=1` → `score train`/`score test`",
      "`fit tree max_depth=6` → compare",
    ],
    learning: ["recursive partitioning", "max_depth as capacity", "staircase boundaries"],
    seedDataset: "moons",
    steps: [
      {
        id: "shallow",
        label: "Stump: max_depth=1",
        detail: "High bias baseline.",
        command: "fit tree max_depth=1",
        check: (s) => s.model === "tree" && Number(s.modelParams.max_depth ?? 3) === 1,
      },
      {
        id: "deep",
        label: "Deep: max_depth=6",
        detail: "High variance path.",
        command: "fit tree max_depth=6",
        check: (s) => s.model === "tree" && Number(s.modelParams.max_depth ?? 3) === 6,
      },
      {
        id: "compare",
        label: "Score train and test on the deep tree",
        detail: "Name the gap.",
        command: "score test",
        check: (s) =>
          isClass(s.trainMetrics) &&
          isClass(s.metrics) &&
          s.scoredOn === "test" &&
          s.trainMetrics.accuracy > s.metrics.accuracy,
      },
    ],
    win: (s) => {
      if (!isClass(s.trainMetrics) || !isClass(s.metrics)) {
        return fail("Score train and test so the gap is visible.");
      }
      if (s.trainMetrics.accuracy <= s.metrics.accuracy) {
        return fail(
          `Train ${s.trainMetrics.accuracy.toFixed(3)} should exceed test ${s.metrics.accuracy.toFixed(3)} on a deep tree with label noise.`,
        );
      }
      return win(
        `Gap ${(s.trainMetrics.accuracy - s.metrics.accuracy).toFixed(3)} on a deep tree. Capacity again — different estimator, same law.`,
      );
    },
  },
  {
    id: "5.2",
    world: "w5",
    worldTitle: "Trees and ensembles",
    title: "Forests buy variance reduction",
    concept: {
      title: "Bagging is insurance against a dramatic tree",
      body:
        "One deep tree is a high-variance artist. RandomForest bootstrap-sample rows, grow many deep trees, and majority-vote. Each tree is still overfit-ish; their errors decorrelate. The forest usually beats one tree on test without a regularization formula in sight.",
      whatHappens:
        "fit forest n_estimators=15 max_depth=4 on moons. Score test and compare to the single deep tree from 5.1. Decision surface becomes less shard-like.",
      why:
        "Ensembling is not magic: it is variance reduction via averaging unstable fits. That is why forests dominate tabular leaderboards.",
      callout: "Average the drama. Keep the signal.",
    },
    goal: "On moons: forest test accuracy ≥ single tree test accuracy (use forest and score test).",
    hints: [
      "`load moons` → `split` → `fit forest n_estimators=15 max_depth=4` → `score test`",
    ],
    learning: ["bagging", "RandomForestClassifier", "variance reduction by averaging"],
    seedDataset: "moons",
    steps: [
      {
        id: "fit",
        label: "Fit RandomForest",
        detail: "Bootstrap + majority vote.",
        command: "fit forest n_estimators=15 max_depth=4",
        check: (s) => s.model === "forest",
      },
      {
        id: "score",
        label: "Score test",
        detail: "Compare to a lone deep tree.",
        command: "score test",
        check: (s) => isClass(s.metrics) && s.scoredOn === "test" && s.metrics.accuracy >= 0.7,
      },
    ],
    win: (s) => {
      if (s.model !== "forest") return fail("Fit `forest`.");
      if (!isClass(s.metrics) || s.scoredOn !== "test") return fail("`score test`.");
      if (s.metrics.accuracy < 0.7) {
        return fail(`Test accuracy ${s.metrics.accuracy.toFixed(3)} < 0.70. Raise n_estimators or keep max_depth moderate.`);
      }
      return win(
        `Forest test accuracy ${s.metrics.accuracy.toFixed(3)}. Averaging unstable trees is a statistical technique, not folklore.`,
      );
    },
  },
  {
    id: "5.3",
    world: "w5",
    worldTitle: "Trees and ensembles",
    title: "When linear still wins",
    concept: {
      title: "Match the inductive bias to the geometry",
      body:
        "On blobs (linearly separable), logistic is enough. Forest can match it but pays in complexity and loses interpretability. Choosing the simplest model that clears the product constraint is not being lazy — it is being honest about risk.",
      whatHappens:
        "Fit logistic and forest on blobs. Both should be strong. Notice logistic coefficients (weights) give a readable story; forest does not.",
      why:
        "Leaderboard accuracy is one axis. Reviewability, calibration, and ops cost are others. A 9-capability curriculum teaches *choice*, not just fitting.",
      callout: "The best model is the simplest one that clears the constraint.",
    },
    goal: "On blobs: prove logistic test acc ≥ 0.9 and still fit forest — then acknowledge simplicity in the win check.",
    hints: [
      "`load blobs` → `split` → `fit logistic` → `score test`",
      "Optionally `fit forest` and score — compare, do not worship.",
    ],
    learning: ["inductive bias", "parsimony", "when not to ensemble"],
    seedDataset: "blobs",
    steps: [
      {
        id: "log",
        label: "Logistic on blobs ≥ 0.90",
        detail: "Simple linear story.",
        command: "score test",
        check: (s) =>
          s.model === "logistic" &&
          isClass(s.metrics) &&
          s.scoredOn === "test" &&
          s.metrics.accuracy >= 0.9,
      },
      {
        id: "alt",
        label: "Also evaluate forest",
        detail: "Evidence of choice, not cargo cult.",
        command: "fit forest n_estimators=11 max_depth=4",
        check: (s) => s.model === "forest" || s.model === "logistic",
      },
    ],
    win: (s) => {
      if (!isClass(s.metrics) || s.scoredOn !== "test") return fail("Score test.");
      if (s.metrics.accuracy < 0.9) return fail("Need a strong test score first.");
      return win(
        "You can use a forest when the geometry demands it. On blobs, linear already told the truth. Choice is the skill.",
      );
    },
  },
];
