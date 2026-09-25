/**
 * World 14 — Multiclass and probability quality.
 */

import type { ClassificationMetrics, Level, SessionSnapshot, WinResult } from "../engine/types";

function fail(feedback: string): WinResult {
  return { won: false, feedback };
}

function win(feedback = "Clear. Concept locked."): WinResult {
  return { won: true, feedback };
}

function isClass(m: SessionSnapshot["metrics"] | SessionSnapshot["trainMetrics"]): m is ClassificationMetrics {
  return Boolean(m && "accuracy" in m);
}

export const WORLD14_LEVELS: Level[] = [
  {
    id: "14.1",
    world: "w14",
    worldTitle: "Multiclass & probability",
    title: "One-vs-rest without lying",
    concept: {
      title: "K binary questions, one argmax answer",
      body:
        "OneVsRestClassifier fits one logistic per class (y = 1 vs rest) and predicts argmax score. Softmax multinomial is the other main family. OvR is transparent and scales; probabilities need renormalization if you want them to sum to 1. Accuracy on K classes is harsher than binary — a wrong confusion story matters.",
      whatHappens:
        "On clusters (3 labels): `fit ovr_logistic` → `score test` → `cm`. You get a multi-class accuracy; confusion is 2×2 only in binary — here we report overall accuracy of argmax. Compare to kmeans which has no y during fit.",
      why:
        "Product taxonomies are multiclass. Forcing them into one binary cut destroys policy options (e.g. churn vs dormancy vs fraud).",
      callout: "Argmax is a decision rule. Scores underneath are a different artifact.",
    },
    goal: "On clusters: fit ovr_logistic and score test accuracy ≥ 0.8.",
    hints: [
      "`load clusters` → `split` → `fit ovr_logistic` → `score test`",
    ],
    learning: ["OneVsRest", "argmax decoding", "multiclass accuracy"],
    seedDataset: "clusters",
    steps: [
      {
        id: "fit",
        label: "Fit OvR logistic",
        detail: "One binary model per class.",
        command: "fit ovr_logistic",
        check: (s) => s.model === "ovr_logistic",
      },
      {
        id: "score",
        label: "Test acc ≥ 0.8",
        detail: "Argmax quality.",
        command: "score test",
        check: (s) => isClass(s.metrics) && s.scoredOn === "test" && s.metrics.accuracy >= 0.8,
      },
    ],
    win: (s) => {
      if (s.model !== "ovr_logistic") return fail("Fit `ovr_logistic`.");
      if (!isClass(s.metrics) || s.scoredOn !== "test") return fail("`score test`.");
      if (s.metrics.accuracy < 0.8) return fail(`Test acc ${s.metrics.accuracy.toFixed(3)} < 0.80.`);
      return win(`Multiclass test acc ${s.metrics.accuracy.toFixed(3)}. OvR decoded by argmax — that is the contract.`);
    },
  },
  {
    id: "14.2",
    world: "w14",
    worldTitle: "Multiclass & probability",
    title: "Calibration is not accuracy",
    concept: {
      title: "When p=0.8, 80% should be positive",
      body:
        "A model can rank well (high AUC) and be badly calibrated. `calib` bins predicted probabilities and compares mean p to empirical frequency. Off-diagonal bins mean your risk numbers are fiction. Fix with Platt/isotonic (CalibratedClassifierCV) or better loss — never by retuning accuracy alone.",
      whatHappens:
        "On moons/blobs with logistic: score test, then `calib`. Read bins. If rate ≈ p, probabilities are usable for thresholds and pricing.",
      why:
        "Fraud scores, medical risk, and churn dollars are multiplied by p. Calibration errors become budget errors.",
      formula: "reliability: bin by p, plot mean(p) vs mean(y)",
      callout: "Do not put a price on an uncalibrated probability.",
    },
    goal: "On blobs: fit logistic, score test, run `calib` (curve produced).",
    hints: [
      "`load blobs` → `split` → `fit logistic` → `score test` → `calib`",
    ],
    learning: ["calibration_curve", "ranking vs probability", "when to calibrate"],
    seedDataset: "blobs",
    steps: [
      {
        id: "score",
        label: "Score logistic",
        detail: "Need p̂ first.",
        command: "score test",
        check: (s) => s.fitted && s.scoredOn === "test",
      },
      {
        id: "calib",
        label: "Reliability bins",
        detail: "p̂ vs empirical rate.",
        command: "calib",
        check: (s) => Boolean(s.calibCurve && s.calibCurve.length >= 5),
      },
    ],
    win: (s) => {
      if (!s.calibCurve) return fail("Run `calib` after scoring a classifier.");
      return win("You separated ranking from reliability. That is probability literacy.");
    },
  },
  {
    id: "14.3",
    world: "w14",
    worldTitle: "Multiclass & probability",
    title: "Threshold is a product object",
    concept: {
      title: "Move the cutoff before the dashboard",
        body:
        "The default 0.5 threshold is a convenience, not a strategy. Cost(FN)/cost(FP) sets the optimal cut. `roc` shows the sweep; `calib` tells you if p can enter the cost formula at all. Document the threshold in the same place as the model version.",
      whatHappens:
        "On moons: `fit logistic` → `roc` → `score test` → `calib`. Choose a threshold story (e.g. high recall for churn). That sentence is a product decision with a number.",
      why:
        "Models do not deploy. Decisions deploy. The threshold is the last mile of every classifier.",
      callout: "Ship the decision rule with the weights.",
    },
    goal: "On moons: run `roc` and `calib` after a logistic fit (both artifacts exist).",
    hints: [
      "`load moons` → `split` → `fit logistic` → `roc` → `calib`",
    ],
    learning: ["threshold policy", "cost-sensitive cuts", "decision + model packaging"],
    seedDataset: "moons",
    steps: [
      {
        id: "roc",
        label: "Threshold sweep",
        detail: "ROC points.",
        command: "roc",
        check: (s) => s.inspectedRoc,
      },
      {
        id: "calib",
        label: "Calibration",
        detail: "Can p enter the cost fn?",
        command: "calib",
        check: (s) => Boolean(s.calibCurve),
      },
    ],
    win: (s) => {
      if (!s.inspectedRoc || !s.calibCurve) return fail("Need both `roc` and `calib`.");
      return win("Threshold + calibration considered. That is a shipping checklist, not a demo.");
    },
  },
];
