/**
 * World 6 — Model selection without worshipping CV or leaking test.
 */

import type { ClassificationMetrics, Level, SessionSnapshot, WinResult } from "../engine/types";

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

export const WORLD6_LEVELS: Level[] = [
  {
    id: "6.1",
    world: "w6",
    worldTitle: "Model selection",
    title: "CV lives inside train",
    concept: {
      title: "cross_val_score is a train-only instrument",
      body:
        "k-fold CV re-splits *train* into folds to estimate generalization without burning test. It is a better train score — still not the final claim. You may use CV to pick a model; you may use test once to report.",
      whatHappens:
        "`cv 5` after fit prints fold scores and their mean on the training partition only. Test remains untouched until `score test`.",
      why:
        "Teams that CV on the full table (including test) launder leakage through a 'scientific' tool. CV without a hold-out is just fancier train accuracy.",
      callout: "CV chooses. Test convicts.",
    },
    goal: "On noisy_line or scale_trap: fit a model, run `cv 5`, then score test exactly once at the end.",
    hints: [
      "`load noisy_line` → `split` → `fit ridge alpha=1` → `cv 5` → `score test`",
    ],
    learning: ["k-fold on train only", "CV vs hold-out roles"],
    seedDataset: "noisy_line",
    steps: [
      {
        id: "fit",
        label: "Fit on a proper split",
        detail: "Test stays sealed.",
        command: "fit ridge alpha=1",
        check: (s) => Boolean(s.split) && s.fitted,
      },
      {
        id: "cv",
        label: "Run 5-fold CV on train",
        detail: "Estimate without burning test.",
        command: "cv 5",
        check: (s) => Boolean(s.cvScores && s.cvScores.length >= 5),
      },
      {
        id: "test",
        label: "Score test once",
        detail: "The publishable number.",
        command: "score test",
        check: (s) => s.scoredOn === "test" && Boolean(s.metrics),
      },
    ],
    win: (s) => {
      if (!s.cvScores || s.cvScores.length < 5) {
        return fail("Run `cv 5` first — instrument the train partition.");
      }
      if (s.scoredOn !== "test" || !s.metrics) {
        return fail("Now `score test` once.");
      }
      return win("CV estimated, test reported. Roles stayed clean.");
    },
  },
  {
    id: "6.2",
    world: "w6",
    worldTitle: "Model selection",
    title: "Search, then freeze",
    concept: {
      title: "GridSearchCV must not meet the test set",
      body:
        "Hyperparameter search peeks at CV folds. If those folds include test, you have trained your *process* on the exam. `search` here only touches train CV. After it returns best params, fit a fresh model and `score test` once.",
      whatHappens:
        "`search ridge alpha=0.01,0.1,1,10` runs a 3-fold sweep on train and prints the winner. Then `fit ridge alpha=<best>` and score test. The score test result is allowed to disagree with CV — that is information, not failure.",
      why:
        "Nested CV is the fully honest version (outer loop for estimate, inner for search). Learn the rule first: never let search or selection touch the final hold-out.",
      callout: "Overfitting the validation protocol is still overfitting.",
    },
    goal: "On poly_curve: search ridge alphas, fit the winner, score test. Must record a searchBest and use test only after search.",
    hints: [
      "`load poly_curve` → `split` → `poly 3` → `search ridge alpha=0.01,0.1,1,10`",
      "`fit ridge alpha=<printed best>` → `score test`",
    ],
    learning: [
      "GridSearchCV protocol",
      "search must not see test",
      "CV score vs test score can disagree",
    ],
    seedDataset: "poly_curve",
    steps: [
      {
        id: "prep",
        label: "Split + poly 3",
        detail: "Rich basis first.",
        command: "poly 3",
        check: (s) => Boolean(s.split) && s.polyDegree === 3,
      },
      {
        id: "search",
        label: "Search α on train CV",
        detail: "Never on test.",
        command: "search ridge alpha=0.01,0.1,1,10",
        check: (s) => Boolean(s.searchBest && Object.keys(s.searchBest).length > 0),
      },
      {
        id: "freeze",
        label: "Fit winner and score test",
        detail: "One final claim.",
        command: "score test",
        check: (s) => s.fitted && s.scoredOn === "test" && Boolean(s.searchBest),
      },
    ],
    win: (s) => {
      if (!s.searchBest) return fail("Run `search ridge alpha=...` before scoring test.");
      if (s.scoredOn !== "test") return fail("Fit the winner, then `score test`.");
      return win(
        `Search chose ${JSON.stringify(s.searchBest)}. Test was a spectator, not a participant.`,
      );
    },
  },
  {
    id: "6.3",
    world: "w6",
    worldTitle: "Model selection",
    title: "Metric shopping is fraud",
    concept: {
      title: "Declare the metric before you look",
      body:
        "If you try twenty metrics and publish the best, you have selected a flattering instrument. Declare accuracy or r2 up front (the level goal does). On moons with noise, also demand `cm` — a single number cannot carry the story.",
      whatHappens:
        "Pick logistic on moons, score test with the predeclared accuracy floor, then open `cm`. If you switch to 'whatever looks best', the conceptual error is identical to leaking test.",
      why:
        "Science has a name for post-hoc metric selection: it is HARKing. Engineering has a name for it too — a surprise outage.",
      callout: "Pre-register the metric. Then look.",
    },
    goal: "On moons: logistic + score test (acc ≥ 0.7) + `cm`. No requirement to hunt metrics.",
    hints: [
      "`load moons` → `split` → `fit logistic` → `score test` → `cm`",
    ],
    learning: ["pre-declared metrics", "metric shopping", "confusion as the second story"],
    seedDataset: "moons",
    steps: [
      {
        id: "fit",
        label: "Fit logistic on moons",
        detail: "Declared model and metric.",
        command: "fit logistic",
        check: (s) => s.model === "logistic",
      },
      {
        id: "score",
        label: "Score test (declared accuracy)",
        detail: "Floor 0.70 — look once.",
        command: "score test",
        check: (s) => isClass(s.metrics) && s.scoredOn === "test" && s.metrics.accuracy >= 0.7,
      },
      {
        id: "cm",
        label: "Open confusion matrix",
        detail: "Failure modes, not slogans.",
        command: "cm",
        check: (s) => s.steps.some((st) => st.label === "confusion matrix"),
      },
    ],
    win: (s) => {
      if (!isClass(s.metrics) || s.scoredOn !== "test") return fail("Score test with the declared metric.");
      if (!s.steps.some((st) => st.label === "confusion matrix")) return fail("Run `cm` as well.");
      return win("You reported a pre-declared metric and the failure modes. That is publishable.");
    },
  },
];
