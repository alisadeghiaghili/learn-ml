/**
 * World 9 — Statistical honesty (intervals, probabilities, log-loss).
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

export const WORLD9_LEVELS: Level[] = [
  {
    id: "9.1",
    world: "w9",
    worldTitle: "Statistical honesty",
    title: "Point estimates are rumors",
    concept:
      {
        title: "Bootstrap an interval before you celebrate",
        body:
          "A test score is a random variable — it depends on the draw. Bootstrap resamples the test predictions with replacement and re-scores. The 2.5% and 97.5% percentiles form a 95% interval. Overlapping intervals mean 'we cannot tell', no matter which mean is higher.",
        whatHappens:
          "`bootstrap 200` after scoring prints [lo, hi]. If hi − lo is wide, your A/B 'win' may be noise. Report the interval in the share text and in design docs.",
        why:
          "Leadership remembers the point estimate. Engineering remembers how often it would flip if the dataset shifted a little.",
        formula: "bootstrap 95% CI = quantiles of score on resampled test indices",
        callout: "No interval, no conclusion.",
      },
    goal: "On blobs: fit logistic, score test, run bootstrap and keep a CI.",
    hints: [
      "`load blobs` → `split` → `fit logistic` → `score test` → `bootstrap 200`",
    ],
    learning: ["bootstrap CI", "sampling variability", "why points estimates lie"],
    seedDataset: "blobs",
    steps: [
      {
        id: "score",
        label: "Score test",
        detail: "Need predictions to resample.",
        command: "score test",
        check: (s) => Boolean(s.metrics),
      },
      {
        id: "boot",
        label: "Bootstrap 95% CI",
        detail: "Resample test indices.",
        command: "bootstrap 200",
        check: (s) => Boolean(s.bootstrapCi),
      },
    ],
    win: (s) => {
      if (!s.bootstrapCi) return fail("Run `bootstrap` after scoring.");
      return win(
        `95% CI [${s.bootstrapCi[0].toFixed(3)}, ${s.bootstrapCi[1].toFixed(3)}]. Quote this, not a naked number.`,
      );
    },
  },
  {
    id: "9.2",
    world: "w9",
    worldTitle: "Statistical honesty",
    title: "Probabilities need log-loss",
    concept:
      {
        title: "Hard labels hide confidence",
        body:
          "Accuracy treats 0.51 and 0.99 the same if the threshold cuts both to 1. Log-loss scores the probability: −log p(y). Overconfident wrong answers are punished. If you will threshold later or price risk, fit for probabilities, not just labels.",
        whatHappens:
          "Score logistic on moons and read `logloss` on the metrics panel (and `show metrics`). Compare to a deep tree — trees are often worse calibrated even with similar accuracy.",
        why:
          "Credit, fraud, and triage systems buy rank + calibrated probability. Accuracy is a UI concern.",
        formula: "logloss = −(1/n) Σ y log p + (1−y) log(1−p)",
        callout: "0.51 is not 0.99.",
      },
    goal: "On moons: score test and read logloss (must be finite and < 1.0) plus open `roc` for the threshold story.",
    hints: [
      "`load moons` → `split` → `fit logistic` → `score test` → `roc`",
    ],
    learning: ["log-loss", "probability vs label", "threshold as a product knob"],
    seedDataset: "moons",
    steps: [
      {
        id: "score",
        label: "Score test with logloss",
        detail: "Probabilities matter.",
        command: "score test",
        check: (s) => isClass(s.metrics) && s.metrics.logLoss > 0 && s.metrics.logLoss < 1,
      },
      {
        id: "roc",
        label: "Inspect threshold sweep",
        detail: "Move the cutoff consciously.",
        command: "roc",
        check: (s) => s.inspectedRoc,
      },
    ],
    win: (s) => {
      if (!isClass(s.metrics)) return fail("Score a classifier first.");
      if (!(s.metrics.logLoss > 0 && s.metrics.logLoss < 1)) {
        return fail(`logloss ${s.metrics.logLoss.toFixed(3)} should be a proper probability score < 1.`);
      }
      if (!s.inspectedRoc) return fail("Run `roc` — thresholds are product decisions.");
      return win(`logloss ${s.metrics.logLoss.toFixed(3)}. You scored uncertainty, not just hits.`);
    },
  },
  {
    id: "9.3",
    world: "w9",
    worldTitle: "Statistical honesty",
    title: "Stratify when classes are rare",
    concept:
      {
        title: "Random split can starve a class",
        body:
          "On imbalanced or small classes, `train_test_split` without stratify can put almost all positives in one side. `split strategy=stratified` preserves the class ratio in train and test. For time-ordered data, `strategy=time` puts the future last — the only honest temporal test.",
        whatHappens:
          "On mixed_table or blobs: `split strategy=stratified` then fit/score. Then try `strategy=time` and watch metrics change — that is distribution shift, not magic.",
        why:
          "Split strategy is part of the experiment design. Default random is not always innocent.",
        callout: "How you split is a hypothesis about deployment.",
      },
    goal: "On blobs: split with strategy=stratified, fit logistic, score test ≥ 0.9.",
    hints: [
      "`load blobs` → `split test_size=0.25 strategy=stratified` → `fit logistic` → `score test`",
    ],
    learning: ["stratified splits", "temporal hold-out", "experiment design"],
    seedDataset: "blobs",
    steps: [
      {
        id: "split",
        label: "Stratified split",
        detail: "Keep class ratios honest.",
        command: "split test_size=0.25 strategy=stratified",
        check: (s) => s.splitStrategy === "stratified",
      },
      {
        id: "score",
        label: "Fit and score test ≥ 0.9",
        detail: "Protocol intact.",
        command: "score test",
        check: (s) =>
          isClass(s.metrics) &&
          s.scoredOn === "test" &&
          s.metrics.accuracy >= 0.9 &&
          s.splitStrategy === "stratified",
      },
    ],
    win: (s) => {
      if (s.splitStrategy !== "stratified") {
        return fail("Use `split strategy=stratified`.");
      }
      if (!isClass(s.metrics) || s.scoredOn !== "test") return fail("`score test` after fit.");
      if (s.metrics.accuracy < 0.9) return fail(`Test accuracy ${s.metrics.accuracy.toFixed(3)} < 0.90.`);
      return win("Class ratios survived the split. That is experiment design, not a default.");
    },
  },
];
