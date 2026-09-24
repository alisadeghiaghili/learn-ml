/**
 * World 8 — Honest model selection protocol (curves, nested thinking, Pipeline API).
 */

import type { ClassificationMetrics, Level, RegressionMetrics, SessionSnapshot, WinResult } from "../engine/types";

function fail(feedback: string): WinResult {
  return { won: false, feedback };
}

function win(feedback = "Clear. Concept locked."): WinResult {
  return { won: true, feedback };
}

function isClass(m: SessionSnapshot["metrics"] | SessionSnapshot["trainMetrics"]): m is ClassificationMetrics {
  return Boolean(m && "accuracy" in m && "confusion" in m);
}

function isReg(m: SessionSnapshot["metrics"] | SessionSnapshot["trainMetrics"]): m is RegressionMetrics {
  return Boolean(m && "r2" in m && "mse" in m);
}

export const WORLD8_LEVELS: Level[] = [
  {
    id: "8.1",
    world: "w8",
    worldTitle: "Honest selection",
    title: "Learning curves diagnose the gap",
    concept: {
      title: "Two failure modes, two curves",
      body:
        "A learning curve plots train score and validation score against training size. High train + low valid that slowly meet: variance — add data or simplify. Both low and tight: bias — richer features or model. This turns 'the model is bad' into a prescription.",
      whatHappens:
        "`curve 5` fits the current model on increasing slices of train and reports train/valid. Read the last gap and the absolute level. That is your variance/bias report.",
      why:
        "Without curves you guess. With curves you allocate: data budget, regularization, or feature work.",
      callout: "Gap is variance. Floor is bias.",
    },
    goal: "On poly_curve poly 12: fit ridge, run `curve 5`, and produce a learning curve with both train and valid scores.",
    hints: [
      "`load poly_curve` → `split` → `poly 12` → `fit ridge alpha=5` → `curve 5`",
    ],
    learning: ["learning curves", "bias–variance prescription", "data size vs capacity"],
    seedDataset: "poly_curve",
    steps: [
      {
        id: "prep",
        label: "High-capacity basis + ridge",
        detail: "poly 12 with a leash.",
        command: "fit ridge alpha=5",
        check: (s) => s.polyDegree === 12 && s.model === "ridge",
      },
      {
        id: "curve",
        label: "Run learning curve",
        detail: "Train/valid vs n.",
        command: "curve 5",
        check: (s) => Boolean(s.learningCurve && s.learningCurve.length >= 3),
      },
    ],
    win: (s) => {
      if (!s.learningCurve || s.learningCurve.length < 3) {
        return fail("Run `curve 5` after fitting.");
      }
      return win(
        `Curve points: ${s.learningCurve.map((r) => `(${r.n}:${r.valid.toFixed(2)})`).join(" ")}. Name the regime before you tune.`,
      );
    },
  },
  {
    id: "8.2",
    world: "w8",
    worldTitle: "Honest selection",
    title: "Nested roles: search, freeze, test once",
    concept: {
      title: "Selection uses CV; reporting uses test",
      body:
        "Nested CV (outer loop estimate, inner loop search) is the textbook gold standard. In product work the minimal honest version is: search on train CV only, freeze hyperparameters, fit, score test once. Never re-enter search after looking at test — that is test-set training.",
      whatHappens:
        "search ridge alphas on poly_curve, fit the winner, `cv 5` for an inner-style estimate, `bootstrap` for an interval, then `score test` as the only final claim.",
      why:
        "Every adaptive decision after seeing test burns a little of the hold-out's honesty. Budget those looks. Ideally zero.",
      callout: "Test is a one-shot camera, not a dashboard.",
    },
    goal: "On poly_curve: search + fit winner + cv 5 + bootstrap + score test once.",
    hints: [
      "`load poly_curve` → `split` → `poly 3`",
      "`search ridge alpha=0.01,0.1,1,10` → `fit ridge alpha=...` → `cv 5` → `bootstrap 100` → `score test`",
    ],
    learning: ["nested roles of CV vs test", "freeze after search", "intervals before claims"],
    seedDataset: "poly_curve",
    steps: [
      {
        id: "search",
        label: "Search on train CV only",
        detail: "Inner loop.",
        command: "search ridge alpha=0.01,0.1,1,10",
        check: (s) => Boolean(s.searchBest),
      },
      {
        id: "cv",
        label: "cv 5 on the frozen model",
        detail: "Still not the final claim.",
        command: "cv 5",
        check: (s) => Boolean(s.cvScores && s.cvScores.length >= 5),
      },
      {
        id: "boot",
        label: "Bootstrap CI",
        detail: "Uncertainty, not vibes.",
        command: "bootstrap 100",
        check: (s) => Boolean(s.bootstrapCi),
      },
      {
        id: "test",
        label: "Score test once",
        detail: "Outer report.",
        command: "score test",
        check: (s) => s.scoredOn === "test" && Boolean(s.searchBest) && Boolean(s.bootstrapCi),
      },
    ],
    win: (s) => {
      if (!s.searchBest) return fail("Search first on train CV.");
      if (!s.cvScores) return fail("Run `cv 5`.");
      if (!s.bootstrapCi) return fail("Run `bootstrap` for an interval.");
      if (s.scoredOn !== "test") return fail("Finally `score test` once.");
      return win(`Frozen params ${JSON.stringify(s.searchBest)} · CI [${s.bootstrapCi[0].toFixed(3)}, ${s.bootstrapCi[1].toFixed(3)}].`);
    },
  },
  {
    id: "8.3",
    world: "w8",
    worldTitle: "Honest selection",
    title: "Ship the Pipeline, not the notebook",
    concept: {
      title: "Preprocessing and model are one artifact",
      body:
        "sklearn `Pipeline` chains transforms and an estimator into a single fit/predict object. `joblib.dump` saves that object. If you export only the classifier and reimplement impute/scale in app code, you will drift — and the drift will be silent.",
      whatHappens:
        "`pipeline` prints the production composition (ColumnTransformer + Pipeline). `save` records a joblib-style save after fit. The checklist is the deployment contract.",
      why:
        "MLOps failures are often 'the app used mean=0 while training used mean=17'. One object, one protocol.",
      formula: "Pipeline = transforms + estimator, fit once, predict as a unit",
      callout: "If it is not in the pickle, it is not in production.",
    },
    goal: "On mixed_table: clean prep + fit + `pipeline` lesson + `save`.",
    hints: [
      "`load mixed_table` → `split` → `impute` → `encode` → `fit logistic` → `pipeline` → `save`",
    ],
    learning: ["sklearn Pipeline", "joblib deployment", "artifact boundary"],
    seedDataset: "mixed_table",
    steps: [
      {
        id: "fit",
        label: "Fit a full prep + model",
        detail: "impute/encode then estimator.",
        command: "fit logistic",
        check: (s) => Boolean(s.split) && s.fitted,
      },
      {
        id: "pipe",
        label: "Read Pipeline composition",
        detail: "ColumnTransformer + Pipeline.",
        command: "pipeline",
        check: (s) => s.steps.some((st) => st.label === "Pipeline.compose"),
      },
      {
        id: "save",
        label: "joblib.save pipeline",
        detail: "One artifact.",
        command: "save",
        check: (s) => s.pipelineSaved,
      },
    ],
    win: (s) => {
      if (!s.fitted) return fail("Fit first — a Pipeline needs an estimator.");
      if (!s.pipelineSaved) return fail("Run `save` (joblib semantics).");
      return win("You defined the deployable object. Notebook cells do not ship.");
    },
  },
];

void isClass;
void isReg;
