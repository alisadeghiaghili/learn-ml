/**
 * World 10 — Capacity tax paid in production (boosting, parsimony, temporal risk).
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

export const WORLD10_LEVELS: Level[] = [
  {
    id: "10.1",
    world: "w10",
    worldTitle: "Capacity tax",
    title: "Boosting fits residuals",
    concept:
      {
        title: "Additive trees fight the last errors",
        body:
          "Gradient boosting adds shallow trees that chase current residuals (logistic loss). It is high capacity with a learning-rate leash. On moons with noise it often beats a single tree and can rival forests — then overfit if n_estimators is huge and lr is 1.",
        whatHappens:
          "`fit boost n_estimators=30 max_depth=2 lr=0.3` → score train/test. Compare to `tree max_depth=6`. Same bias–variance law, different update rule.",
        why:
          "GBDT is the tabular default for a reason. Know the knobs: n_estimators, max_depth, learning_rate.",
        callout: "Boosting is serial variance hunting — budget it.",
      },
    goal: "On moons: fit boost and score test acc ≥ 0.75.",
    hints: [
      "`load moons` → `split` → `fit boost n_estimators=30 max_depth=2 lr=0.3` → `score test`",
    ],
    learning: ["gradient boosting", "shrinkage/learning rate", "ensemble update rules"],
    seedDataset: "moons",
    steps: [
      {
        id: "fit",
        label: "Fit GradientBoosting",
        detail: "Shallow trees on residuals.",
        command: "fit boost n_estimators=30 max_depth=2 lr=0.3",
        check: (s) => s.model === "boost",
      },
      {
        id: "score",
        label: "Test acc ≥ 0.75",
        detail: "Generalization only.",
        command: "score test",
        check: (s) => isClass(s.metrics) && s.scoredOn === "test" && s.metrics.accuracy >= 0.75,
      },
    ],
    win: (s) => {
      if (s.model !== "boost") return fail("Fit `boost`.");
      if (!isClass(s.metrics) || s.scoredOn !== "test") return fail("`score test`.");
      if (s.metrics.accuracy < 0.75) {
        return fail(`Test accuracy ${s.metrics.accuracy.toFixed(3)} < 0.75. Tune n_estimators/lr.`);
      }
      return win(`Boost test accuracy ${s.metrics.accuracy.toFixed(3)}. Residual chasing paid off — still watch variance.`);
    },
  },
  {
    id: "10.2",
    world: "w10",
    worldTitle: "Capacity tax",
    title: "Temporal leakage is still leakage",
    concept:
      {
        title: "The future is not a random row",
        body:
          "Random splits assume exchangeability. Time-ordered data breaks that: if train includes the future relative to test, you have look-ahead. `split strategy=time` puts the chronologically last rows in test. Metrics usually drop — that is the honest number.",
        whatHappens:
          "On poly_curve or noisy_line (x roughly ordered): `split strategy=time` then fit linear/ridge and score. Compare mentally to random split. The time split is the one that survives a 6 a.m. production meeting.",
        why:
          "Fraud, demand, churn — all temporal. If your CV is random on time series, your model has read tomorrow's newspaper.",
        callout: "Shuffle is a crime when order is meaning.",
      },
    goal: "On noisy_line: split strategy=time, fit linear, score test r2 ≥ 0.5.",
    hints: [
      "`load noisy_line` → `split test_size=0.3 strategy=time` → `fit linear` → `score test`",
    ],
    learning: ["temporal hold-out", "look-ahead bias", "exchangeability"],
    seedDataset: "noisy_line",
    steps: [
      {
        id: "split",
        label: "Temporal split",
        detail: "Future rows last.",
        command: "split test_size=0.3 strategy=time",
        check: (s) => s.splitStrategy === "time",
      },
      {
        id: "score",
        label: "Fit and score test r2 ≥ 0.5",
        detail: "Honest forward estimate.",
        command: "score test",
        check: (s) => {
          const m = s.metrics;
          return (
            s.splitStrategy === "time" &&
            Boolean(m) &&
            m !== null &&
            "r2" in m &&
            s.scoredOn === "test" &&
            m.r2 >= 0.5
          );
        },
      },
    ],
    win: (s) => {
      if (s.splitStrategy !== "time") return fail("`split strategy=time` is required here.");
      const m = s.metrics;
      if (!m || !("r2" in m) || s.scoredOn !== "test") return fail("`score test` after fit.");
      if (m.r2 < 0.5) return fail(`Test r2 ${m.r2.toFixed(3)} < 0.5 on a temporal hold-out.`);
      return win("You measured the future with a future-shaped test. That is deployable evidence.");
    },
  },
  {
    id: "10.3",
    world: "w10",
    worldTitle: "Capacity tax",
    title: "End-to-end claim",
    concept:
      {
        title: "Graduation: one sentence you would ship",
        body:
          "Synthesis level. Build a leak-free pipeline on mixed_table, choose a model with CV/search discipline, attach an interval, save the Pipeline. The win is not a magic score — it is a complete, defensible story: what you fit, on what, how you measured, how uncertain it is, and what artifact you deploy.",
        whatHappens:
          "Run the checklist. Each step is a prior lesson. If you skip bootstrap or save, the win check fails — production is the sum of the boring parts.",
        why:
          "Courses often end at accuracy. Products end at an artifact and a claim with error bars.",
        callout: "Score + CI + Pipeline.save = a claim an engineer can defend.",
      },
    goal: "On mixed_table: split → impute → encode → fit → score test → bootstrap → pipeline → save.",
    hints: [
      "`split` → `impute mean` → `encode onehot` → `fit logistic` → `score test` → `bootstrap 100` → `pipeline` → `save`",
    ],
    learning: ["end-to-end ML protocol", "claim + interval + artifact"],
    seedDataset: "mixed_table",
    steps: [
      {
        id: "prep",
        label: "Leak-free prep",
        detail: "split, impute, encode.",
        command: "encode onehot",
        check: (s) => Boolean(s.split) && Boolean(s.imputed) && Boolean(s.encoded) && !s.scaleLeaked,
      },
      {
        id: "model",
        label: "Fit and score test",
        detail: "One honest number.",
        command: "score test",
        check: (s) => s.fitted && s.scoredOn === "test",
      },
      {
        id: "unc",
        label: "Bootstrap interval",
        detail: "Uncertainty attached.",
        command: "bootstrap 100",
        check: (s) => Boolean(s.bootstrapCi),
      },
      {
        id: "ship",
        label: "Pipeline lesson + save",
        detail: "Deployable object.",
        command: "save",
        check: (s) => s.pipelineSaved,
      },
    ],
    win: (s) => {
      if (s.scaleLeaked) return fail("Leakage in prep. Restart the protocol.");
      if (s.scoredOn !== "test") return fail("Score test.");
      if (!s.bootstrapCi) return fail("Attach a bootstrap CI.");
      if (!s.pipelineSaved) return fail("`save` the Pipeline.");
      return win(
        `Claim ready: score + CI [${s.bootstrapCi[0].toFixed(3)}, ${s.bootstrapCi[1].toFixed(3)}] + model.joblib. That is ML engineering, not a demo.`,
      );
    },
  },
];
