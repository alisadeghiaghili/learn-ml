/**
 * World 2 levels — Metrics lie.
 *
 * Concept briefs force the learner to treat accuracy as one number among many.
 */

import type { ClassificationMetrics, Level, SessionSnapshot, WinResult } from "../engine/types";

function fail(feedback: string): WinResult {
  return { won: false, feedback };
}

function win(feedback = "Clear. Concept locked."): WinResult {
  return { won: true, feedback };
}

function isClass(m: SessionSnapshot["metrics"] | SessionSnapshot["trainMetrics"]): m is ClassificationMetrics {
  return Boolean(m && "accuracy" in m && "confusion" in m);
}

export const WORLD2_LEVELS: Level[] = [
  {
    id: "2.1",
    world: "w2",
    worldTitle: "Metrics lie",
    title: "Accuracy is not the goal",
    concept: {
      title: "A majority vote can look smart",
      body:
        "If 90% of rows are class 0, a model that always predicts 0 scores 0.90 accuracy and knows nothing. Dummy baseline first: fit dummy, score test, then beat that number for real. The lesson is comparative — absolute accuracy is meaningless without a baseline.",
      callout: "If you cannot beat the dummy, you do not have a model. You have a prior.",
    },
    goal: "On blobs: score dummy on test, then fit logistic and beat dummy accuracy on test.",
    hints: [
      "`load blobs` → `split` → `fit dummy` → `score test` (record it)",
      "Then `fit logistic` → `score test` — logistic must score strictly higher.",
    ],
    seedDataset: "blobs",
    win: (s) => {
      if (!s.split) {
        return fail("Split first. Baselines without a hold-out are theater.");
      }
      if (!s.trainMetrics && !s.metrics) {
        return fail("Score dummy on test first, then fit logistic and score again.");
      }
      // Require logistic fitted and test metrics better than a constant prior.
      if (s.model !== "logistic") {
        return fail("Fit logistic and score test. (Dummy is only the baseline to beat.)");
      }
      if (s.scoredOn !== "test" || !isClass(s.metrics)) {
        return fail("Score logistic on test.");
      }
      if (s.metrics.accuracy < 0.85) {
        return fail(
          `Test accuracy ${s.metrics.accuracy.toFixed(3)} is weak on blobs. Logistic should be near 1.0.`,
        );
      }
      return win(
        `Test accuracy ${s.metrics.accuracy.toFixed(3)} beats a constant baseline on blobs. Always compare, never celebrate a raw score.`,
      );
    },
  },
  {
    id: "2.2",
    world: "w2",
    worldTitle: "Metrics lie",
    title: "The confusion matrix",
    concept: {
      title: "Errors are not interchangeable",
      body:
        "Accuracy lumps all mistakes together. A confusion matrix splits them: false positives vs false negatives. Precision asks how many predicted positives are real; recall asks how many real positives you found. Run `cm` after scoring and read all four cells.",
      formula: "precision = TP / (TP + FP) ·  recall = TP / (TP + FN)",
      callout: "Pick the metric that matches the cost of the mistake you can least afford.",
    },
    goal: "On moons with logistic: score test and open `cm`. Test accuracy ≥ 0.8 and you must have viewed the confusion matrix.",
    hints: [
      "`load moons` → `split` → `fit logistic` → `score test` → `cm`",
      "Moons carry label noise and are not linearly separable — do not demand 1.0 accuracy.",
    ],
    seedDataset: "moons",
    win: (s) => {
      if (s.model !== "logistic") {
        return fail("Use logistic so the decision boundary is inspectable.");
      }
      if (s.scoredOn !== "test" || !isClass(s.metrics)) {
        return fail("Score on test first.");
      }
      if (s.metrics.accuracy < 0.7) {
        return fail(
          `Test accuracy ${s.metrics.accuracy.toFixed(3)} < 0.70 on moons with logistic. Check that you split before fit.`,
        );
      }
      if (!s.steps.some((st) => st.label === "confusion matrix")) {
        return fail("Run `cm` and read TP, FP, FN, TN — accuracy alone is not the lesson.");
      }
      const m = s.metrics;
      return win(
        `confusion [[${m.confusion[0]?.join(", ")}], [${m.confusion[1]?.join(", ")}]] · precision ${m.precision.toFixed(2)} · recall ${m.recall.toFixed(2)}. Same accuracy can hide different failure modes.`,
      );
    },
  },
  {
    id: "2.3",
    world: "w2",
    worldTitle: "Metrics lie",
    title: "Train score is flattery",
    concept: {
      title: "The number on train is not a product claim",
      body:
        "Reporting training accuracy is like grading homework with the answer key open. Always quote the hold-out. This level fails if you only score train — the interface will insist you produce a test number next to it.",
      callout: "If a slide only shows train metrics, treat the model as untested.",
    },
    goal: "On scale_trap: clean pipeline (split → scale → knn) and report BOTH train and test scores.",
    hints: [
      "`load scale_trap` → `split` → `scale` → `fit knn`",
      "`score train` and `score test` — both must exist on the metrics panel.",
    ],
    seedDataset: "scale_trap",
    win: (s) => {
      if (!s.trainMetrics) {
        return fail("Run `score train` as well. You need both numbers side by side.");
      }
      if (!s.metrics || s.scoredOn !== "test") {
        return fail("Now `score test` — the hold-out is the only number that counts.");
      }
      if (s.scaleLeaked) {
        return fail("Leakage again. Split first, then scale on train.");
      }
      if (s.model !== "knn") {
        return fail("Use knn on scale_trap so scaling actually matters.");
      }
      const trainAcc = isClass(s.trainMetrics) ? s.trainMetrics.accuracy : 0;
      const testAcc = isClass(s.metrics) ? s.metrics.accuracy : 0;
      return win(
        `train ${trainAcc.toFixed(3)} vs test ${testAcc.toFixed(3)} (gap ${Math.abs(trainAcc - testAcc).toFixed(3)}). Publish the test column.`,
      );
    },
  },
];
