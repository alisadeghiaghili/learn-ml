/**
 * World 1 levels — Fit is not understanding.
 *
 * Win predicates inspect SessionSnapshot so they stay free of DOM and
 * command-string matching wherever possible.
 */

import type { ClassificationMetrics, Level, SessionSnapshot, WinResult } from "../engine/types";

function fail(feedback: string): WinResult {
  return { won: false, feedback };
}

function win(feedback = "Clear. Concept locked."): WinResult {
  return { won: true, feedback };
}

function isClass(m: SessionSnapshot["metrics"]): m is ClassificationMetrics {
  return Boolean(m && "accuracy" in m);
}

export const WORLD1_LEVELS: Level[] = [
  {
    id: "1.1",
    world: "w1",
    worldTitle: "Fit is not understanding",
    title: "X and y",
    concept: {
      title: "Everything is a table and a target",
      body:
        "Supervised learning starts with a design matrix X (rows = examples, columns = features) and a target y. The model never sees 'the world' — only these numbers. Before any fit, load a dataset and read its shape and feature names.",
      callout: "If you cannot name X and y, you cannot debug a model.",
    },
    goal: "Load any dataset and inspect it with `show data`.",
    hints: [
      "Try `load blobs`.",
      "Then `show data` — the stage prints n and feature names.",
    ],
    seedDataset: "blobs",
    win: (s) => {
      if (!s.dataset) {
        return fail("No dataset yet. `load blobs` to create X and y.");
      }
      if (!s.inspectedData) {
        return fail("Run `show data` and read n, feature names, and target.");
      }
      return win("X and y exist. That is the whole substrate of supervised learning.");
    },
  },
  {
    id: "1.2",
    world: "w1",
    worldTitle: "Fit is not understanding",
    title: "Fit is optimization, not magic",
    concept: {
      title: "fit(X, y) searches for parameters that lower a loss",
      body:
        "A linear model chooses slope and intercept to minimize squared error on the training rows. That is all fit does: an optimizer over parameters. On clean noisy_line data, Ordinary Least Squares should recover a strong positive slope.",
      formula: "min_{w,b}  Σ (y_i − (w x_i + b))²",
      callout: "Training loss can look great while the model is useless. Hold that thought.",
    },
    goal: "Load noisy_line, split, fit linear, and score on test with r2 ≥ 0.85.",
    hints: [
      "`load noisy_line` → `split test_size=0.2` → `fit linear` → `score test`",
      "No scaler needed — one feature, honest units.",
    ],
    seedDataset: "noisy_line",
    win: (s) => {
      if (!s.fitted || s.model !== "linear") {
        return fail("Fit LinearRegression on the training split.");
      }
      if (!s.split) {
        return fail("Hold out a test set before trusting any number.");
      }
      if (s.scoredOn !== "test" || !s.metrics || "accuracy" in s.metrics) {
        return fail("Run `score test` and report regression metrics.");
      }
      if (s.metrics.r2 < 0.85) {
        return fail(
          `Test r2 is ${s.metrics.r2.toFixed(3)}. A linear model on noisy_line should exceed 0.85 — did you fit on the wrong set?`,
        );
      }
      return win("Parameters landed near the data-generating slope. That is optimization, not insight yet.");
    },
  },
  {
    id: "1.3",
    world: "w1",
    worldTitle: "Fit is not understanding",
    title: "The hold-out is sacred",
    concept: {
      title: "Train/test split is a promise about the future",
      body:
        "You fit on train to estimate parameters. You score on test to estimate generalization. The moment test rows influence fit — directly, or through a scaler — the estimate is fiction.",
      callout: "Never fit on all rows 'just to see', then quote test accuracy.",
    },
    goal: "On blobs: split first, fit logistic, score on test with accuracy ≥ 0.9. Do not fit before splitting.",
    hints: [
      "`load blobs` → `split test_size=0.2` → `fit logistic` → `score test`",
      "Fitting before split is leakage in spirit even if metrics still look high.",
    ],
    seedDataset: "blobs",
    win: (s) => {
      if (!s.split) {
        return fail("Create the split first. That is the whole lesson.");
      }
      const fitStep = s.steps.find((st) => st.kind === "fit");
      const splitStep = s.steps.find((st) => st.kind === "split");
      if (fitStep && splitStep) {
        const fi = s.steps.indexOf(fitStep);
        const si = s.steps.indexOf(splitStep);
        if (fi < si) {
          return fail("You fit before splitting. Undo and split first.");
        }
      }
      if (s.scaleLeaked) {
        return fail("Preprocessing saw the test rows. Split first, then scale on train.");
      }
      if (!s.fitted || s.model !== "logistic") {
        return fail("Fit logistic on the training partition.");
      }
      if (s.scoredOn !== "test" || !isClass(s.metrics)) {
        return fail("Score on test — train accuracy is not the promise.");
      }
      if (s.metrics.accuracy < 0.9) {
        return fail(
          `Test accuracy ${s.metrics.accuracy.toFixed(3)} < 0.90. blobs is linearly separable; logistic should clear this.`,
        );
      }
      return win("You measured generalization on rows the model never trained on. That is the discipline.");
    },
  },
  {
    id: "1.4",
    world: "w1",
    worldTitle: "Fit is not understanding",
    title: "Overfitting, on purpose",
    concept: {
      title: "Capacity is a lever, not a virtue",
      body:
        "KNN with k=1 memorizes every training row — including label noise. Train accuracy locks at 1.0 while test accuracy falls apart. That gap is the overfitting signature: high capacity bought a recording of the data, not the decision rule.",
      callout: "If train ≫ test, you do not have a great model. You have a recording.",
    },
    goal: "On moons with k=1: train accuracy ≥ 0.99 and a train−test gap of at least 3 points.",
    hints: [
      "`load moons` → `split` → `fit knn n_neighbors=1`",
      "`score train` then `score test` — look at the gap, not just one number.",
    ],
    seedDataset: "moons",
    win: (s) => {
      if (s.model !== "knn" || Number(s.modelParams.n_neighbors ?? 5) !== 1) {
        return fail("Use KNeighborsClassifier with n_neighbors=1.");
      }
      if (!s.trainMetrics || !isClass(s.trainMetrics)) {
        return fail("Run `score train` — you need the training number too.");
      }
      if (s.trainMetrics.accuracy < 0.99) {
        return fail(
          `Train accuracy ${s.trainMetrics.accuracy.toFixed(3)} is not memorization yet. k=1 should hit 1.00 on train.`,
        );
      }
      if (s.scoredOn !== "test" || !isClass(s.metrics)) {
        return fail("Now `score test` and compare.");
      }
      const gap = s.trainMetrics.accuracy - s.metrics.accuracy;
      if (gap < 0.03) {
        return fail(
          `Gap is only ${(gap * 100).toFixed(1)} points (train − test). Re-split with another seed or compare against k=15 — we need a visible generalization gap.`,
        );
      }
      return win(
        `Gap = ${(gap * 100).toFixed(1)} points (train − test). Capacity bought memorization, not knowledge.`,
      );
    },
  },
  {
    id: "1.5",
    world: "w1",
    worldTitle: "Fit is not understanding",
    title: "Leakage in the pipeline",
    concept: {
      title: "Preprocessing is part of the model",
      body:
        "StandardScaler estimates mean and variance. If you fit it on all rows, the test set's distribution is already inside those statistics. The model cheats without knowing. Correct order: split → scaler.fit on train → transform train and test → estimator.fit on train.",
      formula: "correct: split → fit_transform(train) → transform(test) → model.fit(train)",
      callout: "If a step can look at y or at all of X, it belongs after the split.",
    },
    goal: "On scale_trap with KNN: scale AFTER split (no leakage), fit knn, test accuracy ≥ 0.9.",
    hints: [
      "`load scale_trap` → `split` → `scale standard` → `fit knn n_neighbors=5` → `score test`",
      "If you scale before split, the pipeline edge turns red and the level will not clear.",
    ],
    seedDataset: "scale_trap",
    win: (s) => {
      if (!s.dataset || s.dataset.name !== "scale_trap") {
        return fail("Use the scale_trap dataset — it has features in different units.");
      }
      if (s.scaleLeaked) {
        return fail(
          "Leakage detected: the scaler was fit before the split. Reset, split first, then scale on train.",
        );
      }
      if (s.scaled !== "standard" && s.scaled !== "minmax") {
        return fail("Fit a scaler after the split (`scale standard`).");
      }
      if (s.model !== "knn") {
        return fail("Use KNN — a distance model that actually suffers without scaling.");
      }
      if (s.scoredOn !== "test" || !isClass(s.metrics)) {
        return fail("Score on the test split.");
      }
      if (s.metrics.accuracy < 0.9) {
        return fail(
          `Test accuracy ${s.metrics.accuracy.toFixed(3)} < 0.90. Check order: split → scale → fit → score test.`,
        );
      }
      return win("Clean pipeline. Test statistics never touched the scaler. That is production hygiene.");
    },
  },
];
