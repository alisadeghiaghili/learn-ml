/**
 * World 7 — Structure without y (PCA) and grouping without labels (k-means).
 */

import type { ClassificationMetrics, Level, WinResult } from "../engine/types";

function fail(feedback: string): WinResult {
  return { won: false, feedback };
}

function win(feedback = "Clear. Concept locked."): WinResult {
  return { won: true, feedback };
}

function isClass(m: unknown): m is ClassificationMetrics {
  return Boolean(m && typeof m === "object" && "accuracy" in m && "confusion" in m);
}

export const WORLD7_LEVELS: Level[] = [
  {
    id: "7.1",
    world: "w7",
    worldTitle: "Unsupervised structure",
    title: "PCA finds the axes of variance",
    concept: {
      title: "Compression is not prediction",
      body:
        "PCA rotates the feature space into directions of maximal variance, then optionally keeps the first k. It never looks at y. On dup_features, three columns are copies of one signal: PC1 should explain almost all variance. That is redundancy detection, not a classifier.",
      whatHappens:
        "`fit pca_knn n_components=1` (or kmeans) shows a reduced space. Compare score of knn on raw vs compressed features — compression can denoise or destroy. You must know which.",
      why:
        "Dimensionality reduction is a modeling choice with a loss. If the discarded direction holds y, you threw away the answer.",
      callout: "PCA maximizes variance, not your KPI.",
    },
    goal: "On dup_features: fit pca_knn n_components=1 and score test r2/acc is not the point — fit is. Then score and pass r2 ≥ 0 or acc ≥ 0.5 to prove the reduced model runs.",
    hints: [
      "`load dup_features` → `split` → `fit pca_knn n_components=1` → `score test`",
    ],
    learning: ["PCA as rotation", "rank / redundancy", "unsupervised ≠ free lunch"],
    seedDataset: "dup_features",
    steps: [
      {
        id: "fit",
        label: "PCA (1 component) + knn",
        detail: "Look at X only, then classify.",
        command: "fit pca_knn n_components=1",
        check: (s) => s.model === "pca_knn" && Number(s.modelParams.n_components ?? 1) === 1,
      },
      {
        id: "score",
        label: "Score test on the reduced model",
        detail: "Prove the pipeline runs.",
        command: "score test",
        check: (s) => Boolean(s.metrics) && s.scoredOn === "test",
      },
    ],
    win: (s) => {
      if (s.model !== "pca_knn") return fail("Fit `pca_knn` with n_components=1.");
      if (s.scoredOn !== "test") return fail("`score test` on the reduced model.");
      return win(
        "You rotated the space, then predicted. Remember: variance ≠ relevance to y.",
      );
    },
  },
  {
    id: "7.2",
    world: "w7",
    worldTitle: "Unsupervised structure",
    title: "k-means finds blobs, not meaning",
    concept:
      {
        title: "Clusters are a geometry claim",
        body:
          "k-means assigns each point to the nearest of k centroids (Lloyd iterations). It assumes spherical clusters and cares about Euclidean distance — so scale matters, and k is a prior. On clusters (3 Gaussians), k=3 should recover the geometry; k=2 will lie confidently.",
        whatHappens:
          "`fit kmeans n_clusters=3` labels points 0/1/2. `score test` against the true cluster id is a sanity check when labels exist; on real unsupervised data you would use silhouette or domain review instead.",
        why:
          "Unsupervised methods still have failure modes and hyperparameters. Treating k-means output as truth is how segments get executive names and no validation.",
        callout: "A cluster is a hypothesis with coordinates.",
      },
    goal: "On clusters: fit kmeans n_clusters=3 and score test against the 3-class labels (acc ≥ 0.8).",
    hints: [
      "`load clusters` → `split` → `fit kmeans n_clusters=3` → `score test`",
      "Use k=2 first to see a worse score — then k=3.",
    ],
    learning: ["Lloyd's algorithm", "k as a prior", "validation of unsupervised output"],
    seedDataset: "clusters",
    steps: [
      {
        id: "fit",
        label: "k-means with k=3",
        detail: "Three spherical blobs.",
        command: "fit kmeans n_clusters=3",
        check: (s) => s.model === "kmeans" && Number(s.modelParams.n_clusters ?? 0) === 3,
      },
      {
        id: "score",
        label: "Score test acc ≥ 0.8",
        detail: "Labels exist here as a rare gift.",
        command: "score test",
        check: (s) =>
          isClass(s.metrics) &&
          s.scoredOn === "test" &&
          s.metrics.accuracy >= 0.8,
      },
    ],
    win: (s) => {
      if (s.model !== "kmeans") return fail("Fit `kmeans n_clusters=3`.");
      if (!isClass(s.metrics) || s.scoredOn !== "test") return fail("`score test`.");
      if (s.metrics.accuracy < 0.8) {
        return fail(`Accuracy ${s.metrics.accuracy.toFixed(3)} < 0.80. Check k=3 and scaling if needed.`);
      }
      return win("Geometry recovered. Naming those clusters is a separate, riskier job.");
    },
  },
  {
    id: "7.3",
    world: "w7",
    worldTitle: "Unsupervised structure",
    title: "Reduce, then predict",
    concept:
      {
        title: "Pipeline PCA into a learner — and watch for leakage",
        body:
          "PCA estimates mean and components. If you fit PCA on all rows (including test), the basis saw the future. Correct: split → fit PCA on train → transform both → fit knn on train. Our pca_knn bakes the projection; the lesson is the order in your head and in `Pipeline`.",
        whatHappens:
          "Use scale_trap or dup_features: split, optional scale on train, fit pca_knn n_components=2, score test. Any pre-split fitted projection is leakage even when accuracy looks fine.",
        why:
          "Unsupervised steps are still fitted estimators. The word 'unsupervised' does not mean 'exempt from the split protocol'.",
        callout: "Unsupervised is not un-fittable. It still estimates parameters from data.",
      },
    goal: "On scale_trap: split first (no leak), fit pca_knn n_components=2, score test acc ≥ 0.85.",
    hints: [
      "`load scale_trap` → `split` → `fit pca_knn n_components=2` → `score test`",
    ],
    learning: ["PCA in a train-only pipeline", "unsupervised leakage", "compose transforms and a learner"],
    seedDataset: "scale_trap",
    steps: [
      {
        id: "split",
        label: "Split before PCA",
        detail: "Components are parameters.",
        command: "split test_size=0.2",
        check: (s) => Boolean(s.split) && !s.scaleLeaked,
      },
      {
        id: "fit",
        label: "PCA(2) + knn",
        detail: "Reduce then learn.",
        command: "fit pca_knn n_components=2",
        check: (s) => s.model === "pca_knn",
      },
      {
        id: "score",
        label: "Test acc ≥ 0.85",
        detail: "Honest hold-out.",
        command: "score test",
        check: (s) =>
          isClass(s.metrics) &&
          s.scoredOn === "test" &&
          s.metrics.accuracy >= 0.85 &&
          !s.scaleLeaked,
      },
    ],
    win: (s) => {
      if (s.scaleLeaked) return fail("Leakage — split before any fitted projection.");
      if (s.model !== "pca_knn") return fail("Fit `pca_knn n_components=2`.");
      if (!isClass(s.metrics) || s.scoredOn !== "test") return fail("`score test`.");
      if (s.metrics.accuracy < 0.85) {
        return fail(`Test accuracy ${s.metrics.accuracy.toFixed(3)} < 0.85. Try n_components=2 after scale.`);
      }
      return win("Unsupervised projection inside the train protocol. That is production PCA.");
    },
  },
];
