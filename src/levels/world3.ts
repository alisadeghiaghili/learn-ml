/**
 * World 3 — Complexity is a budget (polynomial, residuals, regularization, search).
 */

import type { Level, RegressionMetrics, SessionSnapshot, WinResult } from "../engine/types";

function fail(feedback: string): WinResult {
  return { won: false, feedback };
}

function win(feedback = "Clear. Concept locked."): WinResult {
  return { won: true, feedback };
}

function isReg(m: SessionSnapshot["metrics"] | SessionSnapshot["trainMetrics"]): m is RegressionMetrics {
  return Boolean(m && "r2" in m && "mse" in m);
}

export const WORLD3_LEVELS: Level[] = [
  {
    id: "3.1",
    world: "w3",
    worldTitle: "Complexity is a budget",
    title: "Residuals tell the truth",
    concept: {
      title: "A model class, not a score, is what you pick",
      body:
        "Before chasing metrics, look at residuals e = y − ŷ. On poly_curve (a cubic), linear residuals form a bow: negative ends, positive middle (or the reverse). That shape is the model class telling you it is wrong — no amount of tuning a linear fit fixes a nonlinear truth.",
      whatHappens:
        "Fit linear on poly_curve, run `residuals`, and inspect mean/sd. Then `poly 3` and `fit linear` again — residuals collapse toward noise. The design matrix gained columns x², x³; the estimator is still LinearRegression.",
      why:
        "ML folklore says 'try more models'. Engineering says 'look at error structure first'. Residual plots are the cheapest diagnostic you own.",
      formula: "residuals = y_train − ŷ_train",
      callout: "A patterned residual is a bug report from the data.",
    },
    goal: "On poly_curve: fit linear, open residuals, then poly 3 + fit linear and prove r2 improves on test.",
    hints: [
      "`load poly_curve` → `split` → `fit linear` → `residuals`",
      "`poly 3` → `fit linear` → `score test`",
    ],
    learning: [
      "residual diagnostics",
      "model class misspecification",
      "PolynomialFeatures as a design choice",
    ],
    seedDataset: "poly_curve",
    steps: [
      {
        id: "linear",
        label: "Fit linear and inspect residuals",
        detail: "See structured error on a cubic truth.",
        command: "residuals",
        check: (s) => s.inspectedResiduals && s.model === "linear",
      },
      {
        id: "poly",
        label: "Expand to degree 3",
        detail: "x, x², x³ become the design matrix.",
        command: "poly 3",
        check: (s) => s.polyDegree === 3,
      },
      {
        id: "refit",
        label: "Refit and score test (r2 up)",
        detail: "Same estimator, richer features.",
        command: "score test",
        check: (s) => s.fitted && isReg(s.metrics) && s.scoredOn === "test" && s.metrics.r2 >= 0.8,
      },
    ],
    win: (s) => {
      if (!s.inspectedResiduals) {
        return fail("Run `residuals` after a linear fit — diagnose before you upgrade.");
      }
      if (s.polyDegree !== 3) {
        return fail("Use `poly 3` so the cubic truth is representable.");
      }
      if (!isReg(s.metrics) || s.scoredOn !== "test") {
        return fail("Fit and `score test` after the expansion.");
      }
      if (s.metrics.r2 < 0.8) {
        return fail(`Test r2 ${s.metrics.r2.toFixed(3)} is weak after poly 3. Check you refit after expanding.`);
      }
      return win(
        "Same LinearRegression, better basis. Complexity went into the features — and test r2 rose honestly.",
      );
    },
  },
  {
    id: "3.2",
    world: "w3",
    worldTitle: "Complexity is a budget",
    title: "High degree buys variance",
    concept: {
      title: "Capacity without a leash memorizes noise",
      body:
        "Degree 12 on 80 noisy points is a perfect interpolator and a terrible scientist. Train r2 ≈ 1 while test r2 collapses. Regularization (Ridge/Lasso) is a leash: it pays a penalty for large weights so the curve cannot whip through every jitter.",
      whatHappens:
        "poly 12 → fit linear → score train/test (gap). Then poly 12 → fit ridge alpha=10 (or search alpha) and watch test r2 recover. Lasso can zero some of the useless powers of x.",
      why:
        "This is bias–variance with one knob visible. Unregularized high-degree OLS has zero bias on train and huge variance; α reopens bias to buy stability.",
      formula: "Ridge: min Σ (y−ŷ)² + α ‖w‖₂²  ·  Lasso: + α ‖w‖₁",
      callout: "If train is perfect and test is tragic, the problem is variance — not 'more data' slogans.",
    },
    goal: "On poly_curve with poly 12: show a train/test gap on plain linear, then improve test r2 with ridge.",
    hints: [
      "`load poly_curve` → `split` → `poly 12` → `fit linear` → `score train` + `score test`",
      "`fit ridge alpha=10` → `score test` (must beat the unregularized test score)",
    ],
    learning: [
      "overfitting via polynomial degree",
      "L1/L2 penalties",
      "α as a variance budget",
    ],
    seedDataset: "poly_curve",
    steps: [
      {
        id: "gap",
        label: "poly 12 + linear: show the gap",
        detail: "Train nearly 1.0, test far below.",
        command: "score test",
        check: (s) =>
          s.polyDegree === 12 &&
          isReg(s.trainMetrics) &&
          isReg(s.metrics) &&
          s.trainMetrics.r2 - s.metrics.r2 > 0.05,
      },
      {
        id: "ridge",
        label: "Regularize with Ridge",
        detail: "Penalty shrinks wild coefficients.",
        command: "fit ridge alpha=10",
        check: (s) => s.model === "ridge",
      },
      {
        id: "better",
        label: "Beat the unregularized test r2",
        detail: "Generalization is the only score.",
        command: "score test",
        check: (s) => s.model === "ridge" && isReg(s.metrics) && s.scoredOn === "test",
      },
    ],
    win: (s) => {
      if (!isReg(s.trainMetrics) || !isReg(s.metrics)) {
        return fail("Score both train and test on a poly 12 model first.");
      }
      if (s.model !== "ridge" && s.model !== "lasso") {
        return fail("Now fit ridge (or lasso) and score test again.");
      }
      if (!isReg(s.metrics)) {
        return fail("Score the regularized model on test.");
      }
      return win(
        `Regularized test r2=${s.metrics.r2.toFixed(3)}. The leash bought stability at the price of a little train fit — that is the trade.`,
      );
    },
  },
  {
    id: "3.3",
    world: "w3",
    worldTitle: "Complexity is a budget",
    title: "Lasso zeros useless features",
    concept: {
      title: "L1 is a feature selection prior",
      body:
        "Ridge shrinks weights toward 0 but rarely to 0. Lasso's L1 diamond constraint hits corners: some weights become exactly 0. On dup_features (copies of one signal + noise), Lasso should keep one copy and drop the rest more aggressively than Ridge.",
      whatHappens:
        "fit lasso alpha=0.2 on dup_features. Read the printed weights story via residuals/score — only a few coordinates stay large. Compare ridge alpha=0.2: weights spread across the copies.",
      why:
        "Interpretability and compression are real product constraints. Sparsity is not cosmetic — it is a deployment budget.",
      callout: "If three columns are the same signal, Lasso tells you so.",
    },
    goal: "On dup_features: fit lasso, score test r2 ≥ 0.2, and run residuals (weights must include exact zeros).",
    hints: [
      "`load dup_features` → `split` → `fit lasso alpha=0.2` → `residuals` → `score test`",
      "Compare mentally with ridge — same basis, different geometry.",
    ],
    learning: ["L1 vs L2 geometry", "sparsity", "redundant features"],
    seedDataset: "dup_features",
    steps: [
      {
        id: "fit",
        label: "Fit Lasso on dup_features",
        detail: "L1 penalty on correlated copies.",
        command: "fit lasso alpha=0.2",
        check: (s) => s.model === "lasso",
      },
      {
        id: "diag",
        label: "Residual diagnostics",
        detail: "Confirm you looked at error structure.",
        command: "residuals",
        check: (s) => s.inspectedResiduals,
      },
      {
        id: "score",
        label: "Test r2 ≥ 0.2",
        detail: "Sparse but honest.",
        command: "score test",
        check: (s) => isReg(s.metrics) && s.scoredOn === "test" && s.metrics.r2 >= 0.2,
      },
    ],
    win: (s) => {
      if (s.model !== "lasso") {
        return fail("Fit `lasso` — the L1 path is the point.");
      }
      if (!isReg(s.metrics) || s.scoredOn !== "test") {
        return fail("`score test` after fitting.");
      }
      if (s.metrics.r2 < 0.2) {
        return fail(`r2 ${s.metrics.r2.toFixed(3)} is too low. Try alpha around 0.05–0.5.`);
      }
      return win("Sparse model, stable test score. You bought interpretability with L1.");
    },
  },
];
