/**
 * World 13 — Structure, importance, nested CV, and the estimator contract.
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

export const WORLD13_LEVELS: Level[] = [
  {
    id: "13.1",
    world: "w13",
    worldTitle: "Structure and contracts",
    title: "Importance is not causality",
    concept: {
      title: "feature_importances_ is a fit artifact",
      body:
        "Trees and forests expose impurity or permutation importance. High importance means the model relied on the feature — not that changing it moves y. Correlated features share or steal credit. Always pair importance with a domain story or an experiment.",
      whatHappens:
        "`importance` on forest/boost prints proxy scores. In sklearn use `permutation_importance` on a hold-out fold. Treat the output as a debugging lens, not a policy file.",
      why:
        "Executives love importance bars. Data scientists must add 'association ≠ intervention' in the same breath.",
      callout: "Importance ranks model dependency, not business levers.",
    },
    goal: "On scale_trap or blobs: fit forest/boost/tree and run `importance`.",
    hints: [
      "`load scale_trap` → `split` → `scale` → `fit forest` → `importance`",
    ],
    learning: ["feature_importances_", "permutation importance", "causal caution"],
    seedDataset: "scale_trap",
    steps: [
      {
        id: "fit",
        label: "Fit a tree ensemble",
        detail: "Forest/boost/tree.",
        command: "fit forest n_estimators=15 max_depth=4",
        check: (s) => s.model === "forest" || s.model === "tree" || s.model === "boost",
      },
      {
        id: "imp",
        label: "Read importance",
        detail: "Dependency, not causality.",
        command: "importance",
        check: (s) => Boolean(s.importance && s.importance.length > 0),
      },
    ],
    win: (s) => {
      if (!s.importance) return fail("Run `importance` after fitting a tree ensemble.");
      return win("You treated importance as a diagnostic. Keep the causal footnote attached.");
    },
  },
  {
    id: "13.2",
    world: "w13",
    worldTitle: "Structure and contracts",
    title: "Silhouette chooses k with geometry",
    concept: {
      title: "Inertia always falls as k rises",
      body:
        "Elbow plots lie by monotonicity. Silhouette compares intra-cluster cohesion to nearest-cluster separation in [−1, 1]. On `clusters` with true k=3, k=3 should beat a lazy k=8. Run `sil` after `fit kmeans`.",
      whatHappens:
        "fit kmeans n_clusters=3 → `sil`. Try n_clusters=8 and compare. Density methods (`dbscan`) find variable shapes but need eps/min_samples.",
      why:
        "Unsupervised validation is still validation. Naming a segment without a stability metric is astrology.",
      formula: "silhouette = (b − a) / max(a, b)",
      callout: "Never pick k because the slide looked full.",
    },
    goal: "On clusters: fit kmeans n_clusters=3 and run `sil` (score produced).",
    hints: [
      "`load clusters` → `split` → `fit kmeans n_clusters=3` → `sil`",
    ],
    learning: ["silhouette", "choosing k", "dbscan alternative"],
    seedDataset: "clusters",
    steps: [
      {
        id: "fit",
        label: "k-means k=3",
        detail: "Matches the geometry.",
        command: "fit kmeans n_clusters=3",
        check: (s) => s.model === "kmeans" || s.model === "dbscan",
      },
      {
        id: "sil",
        label: "Silhouette score",
        detail: "Cohesion vs separation.",
        command: "sil",
        check: (s) => s.silhouette !== null,
      },
    ],
    win: (s) => {
      if (s.silhouette === null) return fail("Run `sil` after clustering.");
      return win(`silhouette ≈ ${s.silhouette.toFixed(3)}. Geometry judged, not vibes.`);
    },
  },
  {
    id: "13.3",
    world: "w13",
    worldTitle: "Structure and contracts",
    title: "Nested CV and the estimator contract",
    concept: {
      title: "fit/transform/predict + honest selection",
      body:
        "sklearn estimators implement fit/transform/predict; check_is_fitted guards predict-before-fit; clone() resets; get_params/set_params drive search. Nested CV estimates the *procedure*: inner search, outer score. `nested ridge` runs that on train only. Combine with `pipeline` + `save` for the full contract.",
      whatHappens:
        "`nested ridge` prints outer scores. `pipeline` prints the composition. `save` persists. The graduation is a defensible selection procedure + artifact.",
      why:
        "Teams that search on the test set and ship a notebook have no procedure. Nested CV is how you audit the procedure, not the model.",
      callout: "Clone. Fit. Predict. Never share a fitted transformer across folds.",
    },
    goal: "On poly_curve or blobs: run `nested ridge` (outer scores) and `save` after fit (pipeline saved).",
    hints: [
      "`load poly_curve` → `split` → `poly 3` → `nested ridge` → `fit ridge alpha=1` → `save`",
    ],
    learning: [
      "sklearn estimator contract",
      "nested CV outer estimate",
      "clone / check_is_fitted / joblib",
    ],
    seedDataset: "poly_curve",
    steps: [
      {
        id: "nested",
        label: "Nested CV outer scores",
        detail: "Estimate the procedure.",
        command: "nested ridge",
        check: (s) => Boolean(s.nestedCvOuter && s.nestedCvOuter.length >= 2),
      },
      {
        id: "fit",
        label: "Fit a frozen model",
        detail: "After selection.",
        command: "fit ridge alpha=1",
        check: (s) => s.fitted,
      },
      {
        id: "save",
        label: "joblib.save Pipeline",
        detail: "The artifact.",
        command: "save",
        check: (s) => s.pipelineSaved,
      },
    ],
    win: (s) => {
      if (!s.nestedCvOuter) return fail("Run `nested ridge` first.");
      if (!s.pipelineSaved) return fail("`save` the pipeline at the end.");
      return win(
        `Outer scores ${s.nestedCvOuter.map((x) => x.toFixed(3)).join(", ")}. Procedure audited; artifact saved.`,
      );
    },
  },
];

void isClass;
