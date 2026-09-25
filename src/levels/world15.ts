/**
 * World 15 — Statistical depth: MLE, prediction intervals, power.
 */

import type { Level, WinResult } from "../engine/types";

function fail(feedback: string): WinResult {
  return { won: false, feedback };
}

function win(feedback = "Clear. Concept locked."): WinResult {
  return { won: true, feedback };
}

export const WORLD15_LEVELS: Level[] = [
  {
    id: "15.1",
    world: "w15",
    worldTitle: "Statistical depth",
    title: "MLE is the engine under fit",
    concept: {
      title: "Loss functions are negative log-likelihoods",
      body:
        "Squared error is Gaussian MLE. Logistic loss is Bernoulli MLE. When you `fit linear` or `fit logistic`, you are maximizing likelihood (or minimizing the equivalent loss). `infer` then uses the Hessian/SE story under assumptions. This is why different losses are different worldviews about noise.",
      whatHappens:
        "Fit logistic on blobs, `infer` is linear-only — instead `score test` + `calib` and reread the loss. Fit linear on noisy_line and `infer` for Gaussian SEs. Connect the two columns: loss ↔ distribution.",
      why:
        "If you pick MSE on heavy tails, you baked in a Gaussian lie. MLE makes the lie explicit.",
      formula: "MSE = −log N(y|ŷ,σ²)  ·  log-loss = −log Bern(y|p)",
      callout: "Every loss is a noise model with a reputation.",
    },
    goal: "On noisy_line: fit linear, `infer`, and connect to MLE story (coef report exists).",
    hints: [
      "`load noisy_line` → `split` → `fit linear` → `infer`",
      "Then `fit logistic` on blobs and recall Bernoulli MLE.",
    ],
    learning: ["MLE as fit", "loss as distribution", "SE under likelihood"],
    seedDataset: "noisy_line",
    steps: [
      {
        id: "infer",
        label: "Gaussian MLE + SE",
        detail: "infer on linear.",
        command: "infer",
        check: (s) => Boolean(s.coefReport),
      },
    ],
    win: (s) => {
      if (!s.coefReport) return fail("Run `infer` on a linear fit.");
      return win("Loss ↔ likelihood mapped. That is statistical engineering, not button mashing.");
    },
  },
  {
    id: "15.2",
    world: "w15",
    worldTitle: "Statistical depth",
    title: "Predictions need intervals",
    concept: {
      title: "CI for the mean is not PI for a point",
        body:
        "A confidence interval for E[y|x] shrinks with n. A prediction interval for a new y includes irreducible σ and stays wide. `predci` prints ŷ ± 1.96·σ_resid on test rows. Ops teams need PIs; research tables often show CIs. Mixing them causes dangerous optimism.",
      whatHappens:
        "On noisy_line: fit linear, `predci`, read the bands. Compare with `bootstrap` for score CIs — different objects: model prediction vs performance estimate.",
      why:
        "Saying 'demand will be 120 ± 3' (CI) when the true PI is ±40 is how supply chains die.",
      formula: "PI ≈ ŷ ± z·σ_resid  (simplified; ignores leverage)",
      callout: "Widen the band until operations can act on it.",
    },
    goal: "On noisy_line: fit linear and run `predci` (intervals printed).",
    hints: [
      "`load noisy_line` → `split` → `fit linear` → `predci`",
    ],
    learning: ["prediction vs confidence intervals", "irreducible error", "σ_resid"],
    seedDataset: "noisy_line",
    steps: [
      {
        id: "fit",
        label: "Fit linear",
        detail: "Need residuals for σ.",
        command: "fit linear",
        check: (s) => s.fitted,
      },
      {
        id: "pi",
        label: "Prediction intervals",
        detail: "ŷ ± zσ.",
        command: "predci",
        check: (s) => Boolean(s.predCi && s.predCi.length > 0),
      },
    ],
    win: (s) => {
      if (!s.predCi) return fail("Run `predci` after fitting a regressor.");
      return win("You printed a band a planner can use. That is statistics with a job.");
    },
  },
  {
    id: "15.3",
    world: "w15",
    worldTitle: "Statistical depth",
    title: "Power before the A/B",
    concept: {
      title: "You cannot A/B your way out of noise without n",
        body:
        "Minimum detectable effect shrinks as n grows; variance raises the bar. A 'failed' test with 80 users is underpowered, not proof of no effect. Use `bootstrap` CI width as a proxy: if the CI on the metric is huge, your product test will not see a 2% lift. Plan n first; score second.",
      whatHappens:
        "On blobs/scale_trap: score + `bootstrap 300`. Notice CI width vs n (smaller test_size → wider). That width is the resolution of your experiment.",
      why:
        "Most product teams underpower tests and overfit narrative. CI width is the invoice for uncertainty.",
      callout: "If the noise bar is taller than the effect, stop the meeting.",
    },
    goal: "On blobs: score test + `bootstrap 300` (CI exists).",
    hints: [
      "`load blobs` → `split test_size=0.15` → `fit logistic` → `score test` → `bootstrap 300`",
    ],
    learning: ["power / MDE", "CI width as resolution", "planning n"],
    seedDataset: "blobs",
    steps: [
      {
        id: "score",
        label: "Score on a small hold-out",
        detail: "Feel the resolution.",
        command: "score test",
        check: (s) => Boolean(s.metrics),
      },
      {
        id: "boot",
        label: "Bootstrap CI",
        detail: "Width is the invoice.",
        command: "bootstrap 300",
        check: (s) => Boolean(s.bootstrapCi),
      },
    ],
    win: (s) => {
      if (!s.bootstrapCi) return fail("Run `bootstrap` after scoring.");
      const w = s.bootstrapCi[1] - s.bootstrapCi[0];
      return win(`CI width ≈ ${w.toFixed(3)}. Plan n so the effect is taller than that bar.`);
    },
  },
];
