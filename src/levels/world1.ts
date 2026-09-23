/**
 * World 1 levels — Fit is not understanding.
 *
 * Each level: deep concept brief, checklist steps with a current-step signal,
 * and a win predicate that encodes the idea (not a magic command string).
 */

import type {
  ClassificationMetrics,
  GoalStep,
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
        "Supervised learning does not start with an algorithm. It starts with two arrays: X and y. X is the design matrix — every row is one example you have already seen, every column is a feature the model is allowed to use. y is the target — the answer you want the model to reproduce for those examples, then generalize to new ones.",
      whatHappens:
        "When you `load blobs`, the sandbox materializes a toy table: 120 rows, two numeric columns (x1, x2), and a class label in {A, B}. Nothing is trained yet. `show data` is not a formality — it is how you confirm shapes, feature names, and the target column before any fit can hide a mistake.",
      why:
        "If you cannot name X and y, every later bug (wrong column as target, leakage, silent NaN) becomes invisible. Production sklearn code is literally `model.fit(X_train, y_train)` — knowing which array is which is the whole setup.",
      callout: "If you cannot name X and y, you cannot debug a model.",
    },
    goal: "Load any dataset and inspect it with `show data`.",
    hints: [
      "Try `load blobs`.",
      "Then `show data` — read n, feature names, and the target.",
    ],
    learning: ["design matrix X vs target y", "features, rows, and labels"],
    seedDataset: "blobs",
    steps: [
      {
        id: "load",
        label: "Load a dataset",
        detail: "Materialize X and y in the session.",
        command: "load blobs",
        check: (s) => Boolean(s.dataset),
      },
      {
        id: "inspect",
        label: "Inspect the table",
        detail: "Confirm n, feature names, and target.",
        command: "show data",
        check: (s) => s.inspectedData,
      },
    ],
    win: (s) => {
      if (!s.dataset) {
        return fail("No dataset yet. `load blobs` to create X and y.");
      }
      if (!s.inspectedData) {
        return fail("Run `show data` and read n, feature names, and target.");
      }
      return win(
        "X and y exist. Rows are examples, columns are features, y is the story the model must retell.",
      );
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
        "A linear model does not 'learn the data'. It picks numbers (slope w, intercept b) that make predictions ŷ = w·x + b as close as possible to the observed y, under a squared-error loss. That search — gradient descent or the closed-form normal equations — is all `fit` does. There is no understanding inside the optimizer, only arithmetic that reduces a scalar loss.",
      whatHappens:
        "On `noisy_line`, y is generated from y ≈ 2.5x + 1 plus small noise. When you fit `LinearRegression`, the engine solves for w and b on the training rows only. The stage draws that line over the scatter. `score test` then measures how well those frozen parameters explain hold-out rows they never saw.",
      why:
        "Training loss can look perfect while the model is useless on new data. You fit to estimate parameters; you score to estimate generalization. Mixing those jobs is how demos lie to teams.",
      formula: "min_{w,b}  Σ (yᵢ − (w xᵢ + b))²",
      callout: "Parameters landing near the data-generating slope is optimization working — not yet wisdom.",
    },
    goal: "Load noisy_line, split, fit linear, and score on test with r2 ≥ 0.85.",
    hints: [
      "`load noisy_line` → `split test_size=0.2` → `fit linear` → `score test`",
      "One feature in honest units — skip the scaler.",
    ],
    learning: [
      "loss functions and parameter search",
      "linear regression as constrained curve fitting",
      "train fit vs test score",
    ],
    seedDataset: "noisy_line",
    steps: [
      {
        id: "load",
        label: "Load noisy_line",
        detail: "y ≈ 2.5x + 1 with noise.",
        command: "load noisy_line",
        check: (s) => s.dataset?.name === "noisy_line",
      },
      {
        id: "split",
        label: "Hold out a test set",
        detail: "Reserve rows the optimizer never sees.",
        command: "split test_size=0.2",
        check: (s) => Boolean(s.split),
      },
      {
        id: "fit",
        label: "Fit LinearRegression",
        detail: "Solve for slope and intercept on train.",
        command: "fit linear",
        check: (s) => s.fitted && s.model === "linear",
      },
      {
        id: "score",
        label: "Score on test (r2 ≥ 0.85)",
        detail: "Generalization estimate on hold-out rows.",
        command: "score test",
        check: (s) =>
          s.scoredOn === "test" &&
          Boolean(s.metrics) &&
          "r2" in (s.metrics ?? {}) &&
          (s.metrics as { r2: number }).r2 >= 0.85,
      },
    ],
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
      return win(
        "Parameters landed near the data-generating slope. That is optimization doing its job — keep the test set sacred.",
      );
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
        "You split because the real question is not 'how well does the model recall the training table?' but 'how well will it act on rows that do not exist yet?'. Train is for fitting parameters. Test is for one honest estimate of that future. The moment test rows influence fitting — directly, or through a scaler — the estimate becomes fiction with a number attached.",
      whatHappens:
        "`split` shuffles row indices with a seed, carves out a test fraction, and freezes two partitions. `fit logistic` optimizes only on train. `score test` runs the frozen model on the hold-out. Accuracy on blobs should be high because the two Gaussian blobs are linearly separable — the point is not the number, it is *where the number came from*.",
      why:
        "Teams ship on train accuracy all the time. The metric looks fine in the notebook and dies in production because the hold-out never existed. This level fails if you fit before splitting, even if accuracy is high.",
      callout: "Never fit on all rows 'just to see', then quote test accuracy.",
    },
    goal: "On blobs: split first, fit logistic, score on test with accuracy ≥ 0.9. Do not fit before splitting.",
    hints: [
      "`load blobs` → `split test_size=0.2` → `fit logistic` → `score test`",
      "Order matters more than the score.",
    ],
    learning: [
      "train/test as future simulation",
      "why fitting before splitting is cheating",
      "LogisticRegression fit/predict on hold-out",
    ],
    seedDataset: "blobs",
    steps: [
      {
        id: "load",
        label: "Load blobs",
        detail: "Linearly separable two-class toy.",
        command: "load blobs",
        check: (s) => s.dataset?.name === "blobs",
      },
      {
        id: "split",
        label: "Split before any fit",
        detail: "test_size=0.2, fixed seed.",
        command: "split test_size=0.2",
        check: (s) => Boolean(s.split) && s.steps.findIndex((x) => x.kind === "fit") < 0,
      },
      {
        id: "fit",
        label: "Fit logistic on train",
        detail: "Parameters from train only.",
        command: "fit logistic",
        check: (s) => s.fitted && s.model === "logistic",
      },
      {
        id: "score",
        label: "Score on test (acc ≥ 0.90)",
        detail: "One honest generalization number.",
        command: "score test",
        check: (s) =>
          s.scoredOn === "test" &&
          isClass(s.metrics) &&
          s.metrics.accuracy >= 0.9,
      },
    ],
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
      return win(
        "You measured generalization on rows the model never trained on. That is the discipline.",
      );
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
        "KNN with k=1 does not learn a rule — it stores the training set and answers new points by looking up the nearest stored row. Train accuracy locks at 1.0 because every training point's nearest neighbor is itself. When labels carry noise (this moons set does), the model memorizes the wrong labels too. Test accuracy falls apart exactly where the recording contradicts the true geometry.",
      whatHappens:
        "`fit knn n_neighbors=1` stores the scaled/untransformed train rows as the model. `score train` will look perfect. `score test` will show a visible gap. Watch the decision surface: with k=1 it is a jagged Voronoi partition, not a smooth boundary.",
      why:
        "High capacity buys a recording of the sample, not the decision rule. The gap (train − test) is the signature. If you only publish train metrics, you are showing how well the model can look up homework.",
      callout: "If train ≫ test, you do not have a great model. You have a recording.",
    },
    goal: "On moons with k=1: train accuracy ≥ 0.99 and a train−test gap of at least 3 points.",
    hints: [
      "`load moons` → `split` → `fit knn n_neighbors=1`",
      "`score train` then `score test` — look at the gap, not just one number.",
    ],
    learning: [
      "k-NN as memorization",
      "label noise and capacity",
      "train/test gap as overfitting signature",
    ],
    seedDataset: "moons",
    steps: [
      {
        id: "load",
        label: "Load moons (with label noise)",
        detail: "Noise makes memorization visible.",
        command: "load moons",
        check: (s) => s.dataset?.name === "moons",
      },
      {
        id: "split",
        label: "Split",
        detail: "Hold out honest test rows.",
        command: "split test_size=0.25",
        check: (s) => Boolean(s.split),
      },
      {
        id: "fit",
        label: "Fit knn with n_neighbors=1",
        detail: "Maximum capacity — pure memorization.",
        command: "fit knn n_neighbors=1",
        check: (s) => s.model === "knn" && Number(s.modelParams.n_neighbors ?? 5) === 1,
      },
      {
        id: "train",
        label: "Score train (expect ~1.00)",
        detail: "The flattery number.",
        command: "score train",
        check: (s) => Boolean(s.trainMetrics) && isClass(s.trainMetrics) && s.trainMetrics.accuracy >= 0.99,
      },
      {
        id: "test",
        label: "Score test and compare the gap",
        detail: "Train − test should be ≥ 3 points.",
        command: "score test",
        check: (s) => {
          if (!isClass(s.trainMetrics) || !isClass(s.metrics)) return false;
          return s.scoredOn === "test" && s.trainMetrics.accuracy - s.metrics.accuracy >= 0.03;
        },
      },
    ],
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
          `Gap is only ${(gap * 100).toFixed(1)} points (train − test). Re-split with another seed — we need a visible generalization gap.`,
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
        "StandardScaler estimates mean and variance from data. If you fit it on all rows, the test set's distribution is already inside those statistics. The estimator then trains on features that 'know' the future. The model cheats without knowing — metrics look fine, production does not.",
      whatHappens:
        "`scale_trap` has features in wildly different units (budget ~0–1, units ~0–1000). Distance-based KNN is dominated by the large-scale feature until you standardize. Correct order: `split` → `scale` (fit on train only) → `fit knn` → `score test`. The pipeline graph turns red and the level refuses a leaky path even if accuracy looks acceptable.",
      why:
        "Any step that estimates statistics from data is part of the model. Fit it inside the train fold only, then transform test. Production `Pipeline` / `ColumnTransformer` exist precisely to make this order hard to get wrong.",
      formula: "correct: split → fit_transform(train) → transform(test) → model.fit(train)",
      callout: "If a step can look at y or at all of X, it belongs after the split.",
    },
    goal: "On scale_trap with KNN: scale AFTER split (no leakage), fit knn, test accuracy ≥ 0.9.",
    hints: [
      "`load scale_trap` → `split` → `scale standard` → `fit knn n_neighbors=5` → `score test`",
      "If you scale before split, the pipeline edge turns red.",
    ],
    learning: [
      "feature scaling and distance models",
      "why scalers must fit on train only",
      "pipeline order as a safety property",
    ],
    seedDataset: "scale_trap",
    steps: [
      {
        id: "load",
        label: "Load scale_trap",
        detail: "Features in incompatible units.",
        command: "load scale_trap",
        check: (s) => s.dataset?.name === "scale_trap",
      },
      {
        id: "split",
        label: "Split first",
        detail: "Never scale before this step.",
        command: "split test_size=0.2",
        check: (s) => Boolean(s.split) && !s.scaleLeaked,
      },
      {
        id: "scale",
        label: "Scale on train only",
        detail: "StandardScaler or MinMaxScaler.",
        command: "scale standard",
        check: (s) => Boolean(s.scaled) && !s.scaleLeaked,
      },
      {
        id: "fit",
        label: "Fit knn",
        detail: "Distance model that actually needs scaling.",
        command: "fit knn n_neighbors=5",
        check: (s) => s.model === "knn",
      },
      {
        id: "score",
        label: "Score on test (acc ≥ 0.90)",
        detail: "Clean generalization estimate.",
        command: "score test",
        check: (s) =>
          s.scoredOn === "test" &&
          isClass(s.metrics) &&
          s.metrics.accuracy >= 0.9 &&
          !s.scaleLeaked,
      },
    ],
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
      return win(
        "Clean pipeline. Test statistics never touched the scaler. That is production hygiene.",
      );
    },
  },
];

export type { GoalStep };
