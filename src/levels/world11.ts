/**
 * World 11 — Inference: coefficients, assumptions, MLE.
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

export const WORLD11_LEVELS: Level[] = [
  {
    id: "11.1",
    world: "w11",
    worldTitle: "Inference",
    title: "Coefficients with error bars",
    concept: {
      title: "A weight without SE is a story, not an estimate",
      body:
        "Linear regression as Gaussian MLE gives coefficients and (under assumptions) standard errors. A |z| ≳ 2 is the textbook 'significant' band. This level is not p-value worship — it is learning that estimates are distributions. `infer` prints coef / se / z for the fitted linear model.",
      whatHappens:
        "On noisy_line: fit linear, `infer`, read the table. Slope should be far from 0 (true 2.5). On a noise-only feature, z should be small.",
      why:
        "Product decisions ('spend drives churn') need uncertainty. Point estimates create confident wrong roadmaps.",
      formula: "z = coef / SE  ·  OLS is argmax Gaussian likelihood",
      callout: "If you cannot reject 0, do not ship the feature story.",
    },
    goal: "On noisy_line: fit linear and run `infer` (coef report produced).",
    hints: [
      "`load noisy_line` → `split` → `fit linear` → `infer`",
    ],
    learning: ["OLS as MLE", "SE and z", "coefficient claims"],
    seedDataset: "noisy_line",
    steps: [
      {
        id: "fit",
        label: "Fit linear on noisy_line",
        detail: "Parameters first.",
        command: "fit linear",
        check: (s) => s.fitted && s.model === "linear",
      },
      {
        id: "infer",
        label: "Coefficient summary",
        detail: "coef / SE / z.",
        command: "infer",
        check: (s) => Boolean(s.coefReport && s.coefReport.length >= 2),
      },
    ],
    win: (s) => {
      if (!s.coefReport) return fail("Run `infer` after fitting a linear-like model.");
      return win(
        `z(x1) ≈ ${s.coefReport[1]?.z.toFixed(2) ?? "?"}. You reported uncertainty, not folklore.`,
      );
    },
  },
  {
    id: "11.2",
    world: "w11",
    worldTitle: "Inference",
    title: "Assumptions are load-bearing",
    concept: {
      title: "Residuals: mean 0, no structure, finite variance",
      body:
        "OLS is BLUE under linearity, independent errors, homoscedasticity, and normality for exact inference. You will not prove these with one number. `residuals` gives mean/sd; add scatter inspection mentally. Heteroscedastic patterns mean robust SEs or a different model class.",
      whatHappens:
        "Fit linear on outlier_line. Residuals will show the two outliers dominating. Fit ridge — slope shrinks. That is robustness, not magic.",
      why:
        "Most production linear bugs are assumption bugs: missing interactions, nonlinear truth, or contaminated labels.",
      callout: "Assume nothing. Look at e.",
    },
    goal: "On outlier_line: fit linear, `residuals`, then ridge and `residuals` again.",
    hints: [
      "`load outlier_line` → `split` → `fit linear` → `residuals`",
      "`fit ridge alpha=5` → `residuals`",
    ],
    learning: ["OLS assumptions", "outlier influence", "regularization as robustness"],
    seedDataset: "outlier_line",
    steps: [
      {
        id: "ols",
        label: "OLS residuals",
        detail: "See contaminated error.",
        command: "residuals",
        check: (s) => s.inspectedResiduals && s.model === "linear",
      },
      {
        id: "ridge",
        label: "Ridge residuals",
        detail: "Shrink the drama.",
        command: "residuals",
        check: (s) => s.model === "ridge" || s.model === "lasso" || s.model === "elasticnet",
      },
    ],
    win: (s) => {
      if (!s.inspectedResiduals) return fail("Open residuals first.");
      if (s.model === "linear") return fail("Now fit ridge (or lasso) and inspect again.");
      return win("You watched assumptions fail and applied a leash. That is diagnostic ML.");
    },
  },
  {
    id: "11.3",
    world: "w11",
    worldTitle: "Inference",
    title: "ElasticNet mixes geometries",
    concept: {
      title: "L1 + L2 in one penalty",
      body:
        "ElasticNet = α·l1_ratio·‖w‖₁ + α·(1−l1_ratio)/2·‖w‖₂². It keeps Lasso's sparsity while stabilizing correlated features (Ridge behavior). On dup_features this is the honest middle: zeros *and* not a knife-edge among copies.",
      whatHappens:
        "fit elasticnet alpha=0.1 l1_ratio=0.5 on dup_features. `infer` or residuals + score. Compare mental model with pure Lasso.",
      why:
        "Real features are correlated. Pure L1 arbitrarily picks one; ElasticNet spreads more stably. That is an engineering choice about explanation.",
      formula: "α·l1_ratio·|w| + α·(1−l1_ratio)/2·w²",
      callout: "Sparsity and stability are different gifts.",
    },
    goal: "On dup_features: fit elasticnet, score test, and run `infer` or residuals.",
    hints: [
      "`load dup_features` → `split` → `fit elasticnet alpha=0.1 l1_ratio=0.5` → `score test` → `infer`",
    ],
    learning: ["ElasticNet penalty", "correlated features", "sparse + stable"],
    seedDataset: "dup_features",
    steps: [
      {
        id: "fit",
        label: "Fit ElasticNet",
        detail: "L1+L2 mix.",
        command: "fit elasticnet alpha=0.1 l1_ratio=0.5",
        check: (s) => s.model === "elasticnet",
      },
      {
        id: "score",
        label: "Score or infer",
        detail: "Look at the weights story.",
        command: "score test",
        check: (s) => s.scoredOn === "test" || Boolean(s.coefReport),
      },
    ],
    win: (s) => {
      if (s.model !== "elasticnet") return fail("Fit `elasticnet`.");
      return win("You mixed L1/L2 on purpose. That is regularization literacy.");
    },
  },
];

void isReg;
