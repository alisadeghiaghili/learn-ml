/**
 * World 2 levels — Metrics lie.
 */

import type {
  ClassificationMetrics,
  Level,
  SessionSnapshot,
  WinResult,
} from "../engine/types";

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

export const WORLD2_LEVELS: Level[] = [
  {
    id: "2.1",
    world: "w2",
    worldTitle: "Metrics lie",
    title: "Accuracy is not the goal",
    concept: {
      title: "A majority vote can look smart",
      body:
        "Accuracy is the fraction of correct labels. That sounds like intelligence and is often just a prior. If 90% of rows are class 0, a model that always predicts 0 scores 0.90 and knows nothing about individuals. The only way to know whether a number means anything is to compare it to a baseline that uses no features.",
      whatHappens:
        "`fit dummy` predicts the training mean/majority for every row — no features consulted. Score it. Then `fit logistic`, which actually reads x1 and x2. On blobs, logistic should beat dummy by a wide margin. That margin is the evidence of learning; the raw accuracy is not.",
      why:
        "Absolute metrics are folklore. Relative metrics against a dumb baseline are evidence. Always ask: compared to what?",
      callout: "If you cannot beat the dummy, you do not have a model. You have a prior.",
    },
    goal: "On blobs: score dummy on test, then fit logistic and beat dummy accuracy on test.",
    hints: [
      "`load blobs` → `split` → `fit dummy` → `score test` (record it)",
      "Then `fit logistic` → `score test` — logistic must score strictly higher.",
    ],
    learning: [
      "accuracy as a base rate trap",
      "DummyRegressor/DummyClassifier baselines",
      "comparative evaluation",
    ],
    seedDataset: "blobs",
    steps: [
      {
        id: "split",
        label: "Load blobs and split",
        detail: "Same protocol for dummy and logistic.",
        command: "split test_size=0.2",
        check: (s) => Boolean(s.dataset) && Boolean(s.split),
      },
      {
        id: "dummy",
        label: "Fit dummy and score test",
        detail: "Record the no-feature baseline.",
        command: "fit dummy",
        check: (s) => s.model === "dummy" && Boolean(s.metrics),
      },
      {
        id: "logistic",
        label: "Fit logistic and beat dummy on test",
        detail: "Strictly higher test accuracy.",
        command: "fit logistic",
        check: (s) =>
          s.model === "logistic" &&
          isClass(s.metrics) &&
          s.scoredOn === "test" &&
          s.metrics.accuracy >= 0.85,
      },
    ],
    win: (s) => {
      if (!s.split) {
        return fail("Split first. Baselines without a hold-out are theater.");
      }
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
        "Accuracy lumps every mistake into one bucket. A confusion matrix splits them: false positives (you cried wolf) versus false negatives (you missed the fire). Precision asks how many predicted positives are real. Recall asks how many real positives you found. The right metric is a business question about which mistake is more expensive.",
      whatHappens:
        "After `score test`, run `cm`. You get the 2×2 table [[TN, FP], [FN, TP]] plus precision, recall, and F1. On moons (label noise + nonlinear boundary) logistic cannot be perfect — the matrix shows *how* it fails, not just that it fails.",
      why:
        "Two models can share accuracy and have opposite failure modes. A fraud detector with high recall and low precision floods the ops team; high precision and low recall misses fraud. Publish the matrix, not the slogan.",
      formula: "precision = TP / (TP + FP)  ·  recall = TP / (TP + FN)",
      callout: "Pick the metric that matches the cost of the mistake you can least afford.",
    },
    goal: "On moons with logistic: score test and open `cm`. Test accuracy ≥ 0.7 and you must have viewed the confusion matrix.",
    hints: [
      "`load moons` → `split` → `fit logistic` → `score test` → `cm`",
      "Moons carry label noise — do not demand 1.0 accuracy.",
    ],
    learning: [
      "confusion matrix cells",
      "precision vs recall tradeoffs",
      "F1 as a harmonic compromise",
    ],
    seedDataset: "moons",
    steps: [
      {
        id: "splitfit",
        label: "Split and fit logistic on moons",
        detail: "Inspectable linear decision surface.",
        command: "fit logistic",
        check: (s) => Boolean(s.split) && s.model === "logistic",
      },
      {
        id: "score",
        label: "Score on test (acc ≥ 0.70)",
        detail: "Honest number on noisy moons.",
        command: "score test",
        check: (s) =>
          s.scoredOn === "test" && isClass(s.metrics) && s.metrics.accuracy >= 0.7,
      },
      {
        id: "cm",
        label: "Open the confusion matrix",
        detail: "Read TP, FP, FN, TN — not just accuracy.",
        command: "cm",
        check: (s) => s.steps.some((st) => st.label === "confusion matrix"),
      },
    ],
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
        return fail(
          "Run `cm` and read TP, FP, FN, TN — accuracy alone is not the lesson.",
        );
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
        "Training accuracy answers a question nobody asked: can the model reproduce data it already optimized against? Reporting it alone is grading homework with the answer key open. You may show train metrics as a diagnostic (overfitting gap), never as the headline.",
      whatHappens:
        "Build a clean pipeline on scale_trap (split → scale → knn), then run both `score train` and `score test`. The side-by-side numbers teach the only publishable claim: the hold-out column. If they diverge wildly, that is the overfitting lesson again — still not a reason to hide the test number.",
      why:
        "Slides full of train metrics are how bad models reach production. The discipline is simple: every score that influences a decision comes from rows the model never fit on.",
      callout: "If a slide only shows train metrics, treat the model as untested.",
    },
    goal: "On scale_trap: clean pipeline (split → scale → knn) and report BOTH train and test scores.",
    hints: [
      "`load scale_trap` → `split` → `scale` → `fit knn`",
      "`score train` and `score test` — both must exist on the metrics panel.",
    ],
    learning: [
      "train metric vs hold-out metric",
      "why headlines must quote test/CV",
      "clean pipeline recap",
    ],
    seedDataset: "scale_trap",
    steps: [
      {
        id: "pipe",
        label: "Clean pipeline on scale_trap",
        detail: "split → scale → knn, no leakage.",
        command: "fit knn n_neighbors=5",
        check: (s) =>
          Boolean(s.split) &&
          Boolean(s.scaled) &&
          !s.scaleLeaked &&
          s.model === "knn",
      },
      {
        id: "train",
        label: "Score train",
        detail: "Diagnostic only.",
        command: "score train",
        check: (s) => Boolean(s.trainMetrics),
      },
      {
        id: "test",
        label: "Score test (the publishable number)",
        detail: "Both metrics must be visible.",
        command: "score test",
        check: (s) => Boolean(s.metrics) && s.scoredOn === "test",
      },
    ],
    win: (s) => {
      if (!s.trainMetrics) {
        return fail(
          "Run `score train` as well. You need both numbers side by side.",
        );
      }
      if (!s.metrics || s.scoredOn !== "test") {
        return fail(
          "Now `score test` — the hold-out is the only number that counts.",
        );
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
