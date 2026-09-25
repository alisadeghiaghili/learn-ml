/**
 * World 16 — Feature geometry and cluster validation depth.
 */

import type { Level, WinResult } from "../engine/types";

function fail(feedback: string): WinResult {
  return { won: false, feedback };
}

function win(feedback = "Clear. Concept locked."): WinResult {
  return { won: true, feedback };
}

export const WORLD16_LEVELS: Level[] = [
  {
    id: "16.1",
    world: "w16",
    worldTitle: "Geometry of features",
    title: "Interactions are hypotheses",
    concept: {
      title: "x1·x2 is a claim that the joint matters",
      body:
        "Adding interactions injects a multiplicative story. On poly-like or mixed data, `fe interact` appends x0·x1. Fit linear again and compare test r2/mse. If the interaction helps test, the world has structure the main effects missed; if only train improves, you bought variance.",
      whatHappens:
        "On poly_curve or mixed_table: `fe interact` → `fit linear|ridge` → `score test`. Remember FE fit is train-only when split exists — the same leakage law as scalers.",
      why:
        "Product metrics are full of interactions (price × segment). Blind mains underfit; blind interactions overfit. Measure both.",
      callout: "Interactions are cheap to add and expensive to explain.",
    },
    goal: "On mixed_table or poly_curve: `fe interact`, fit linear/ridge, score test.",
    hints: [
      "`load poly_curve` → `split` → `fe interact` → `fit ridge alpha=1` → `score test`",
    ],
    learning: ["interaction features", "measured FE", "train-fit FE stats"],
    seedDataset: "poly_curve",
    steps: [
      {
        id: "fe",
        label: "Add interactions",
        detail: "x0·x1 from train FE.",
        command: "fe interact",
        check: (s) => s.feMode === "interact",
      },
      {
        id: "score",
        label: "Fit + score test",
        detail: "Hypothesis accepted/rejected.",
        command: "score test",
        check: (s) => s.fitted && s.scoredOn === "test",
      },
    ],
    win: (s) => {
      if (s.feMode !== "interact") return fail("`fe interact` after split.");
      if (s.scoredOn !== "test") return fail("`score test`.");
      return win("You treated interaction as a hypothesis and measured it. That is FE science.");
    },
  },
  {
    id: "16.2",
    world: "w16",
    worldTitle: "Geometry of features",
    title: "Compare clusterings, do not decorate them",
    concept: {
      title: "Silhouette + shape + stability",
      body:
        "k-means assumes spheres; `dbscan` finds density shapes and flags noise (−1). `sil` scores k-means quality. Use both: silhouette on kmeans to choose k; dbscan to discover non-globular groups. Stability under seeds/eps is the third check in production. Never name clusters from a single plot.",
      whatHappens:
        "On clusters: `fit kmeans n_clusters=3` → `sil`. Then `fit dbscan` (optional) and note noise points. Two methods disagreeing is information.",
      why:
        "Segments drive pricing and ops. Aesthetic clusters become expensive segments.",
      callout: "Validate before you name.",
    },
    goal: "On clusters: kmeans + `sil`, and also fit dbscan at least once.",
    hints: [
      "`load clusters` → `split` → `fit kmeans n_clusters=3` → `sil`",
      "`fit dbscan eps=0.8 min_samples=4`",
    ],
    learning: ["kmeans vs dbscan", "silhouette", "noise points"],
    seedDataset: "clusters",
    steps: [
      {
        id: "km",
        label: "kmeans + silhouette",
        detail: "Cohesion metric.",
        command: "sil",
        check: (s) => s.model === "kmeans" && s.silhouette !== null,
      },
      {
        id: "db",
        label: "Try dbscan",
        detail: "Density alternative.",
        command: "fit dbscan eps=0.8 min_samples=4",
        check: (s) => s.model === "dbscan" || Boolean(s.silhouette),
      },
    ],
    win: (s) => {
      if (s.silhouette === null && s.model !== "dbscan") return fail("Need silhouette or dbscan.");
      return win("Two geometries compared. Names come after evidence.");
    },
  },
  {
    id: "16.3",
    world: "w16",
    worldTitle: "Geometry of features",
    title: "Binning + target encoding tax",
    concept: {
      title: "Every FE transform has a bias and a leak surface",
      body:
        "`fe bin` adds quantile steps; `fe target` adds y-mean per category. Both fit on train only. Combined with scale/impute/encode they form a real ColumnTransformer. The tax: target encoding is high-variance on rare categories; binning loses resolution. Measure test, document the transform list in Pipeline.",
      whatHappens:
        "On mixed_table: split → impute → `fe target` → `fe bin` is not stacked in v1 (one FE mode); instead run target then logistic. In `pipeline` see where FE belongs in the ColumnTransformer.",
      why:
        "This is the last mile before 9+ in FE: knowing what you paid for each column you invented.",
      callout: "Invented columns are liabilities until they score.",
    },
    goal: "On mixed_table: impute + `fe target` + fit + score test (FE story complete).",
    hints: [
      "`load mixed_table` → `split` → `impute mean` → `fe target` → `fit logistic` → `score test` → `pipeline`",
    ],
    learning: ["FE tax", "rare category variance", "Pipeline placement of FE"],
    seedDataset: "mixed_table",
    steps: [
      {
        id: "prep",
        label: "Impute + target FE",
        detail: "Train-only stats.",
        command: "fe target",
        check: (s) => Boolean(s.imputed) && s.feMode === "target",
      },
      {
        id: "score",
        label: "Score test",
        detail: "Pay the tax consciously.",
        command: "score test",
        check: (s) => s.fitted && s.scoredOn === "test",
      },
      {
        id: "pipe",
        label: "See Pipeline composition",
        detail: "Where FE lives.",
        command: "pipeline",
        check: (s) => s.steps.some((st) => st.label === "Pipeline.compose") || Boolean(s.pipelineSaved),
      },
    ],
    win: (s) => {
      if (s.feMode !== "target") return fail("`fe target` after split.");
      if (s.scoredOn !== "test") return fail("`score test`.");
      return win("FE costed and placed in the pipeline. That is engineering-grade feature work.");
    },
  },
];
