/**
 * Built-in toy datasets that make specific concepts visible in 2-D.
 */

import { matrix, vector } from "./matrix";
import type { Dataset, DatasetName } from "./types";

/** Deterministic PRNG (mulberry32). */
export function makeRng(seed: number): () => number {
  let t = seed >>> 0;
  return () => {
    t += 0x6d2b79f5;
    let r = Math.imul(t ^ (t >>> 15), 1 | t);
    r ^= r + Math.imul(r ^ (r >>> 7), 61 | r);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

function gauss(rng: () => number, mu = 0, sigma = 1): number {
  // Box-Muller, one draw per call (fine for toy data).
  const u1 = Math.max(rng(), 1e-9);
  const u2 = rng();
  const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  return mu + sigma * z;
}

/**
 * Two anisotropic Gaussian blobs — clean linear separability.
 */
function makeBlobs(seed = 42): Dataset {
  const rng = makeRng(seed);
  const rows: number[][] = [];
  const labels: number[] = [];
  for (let i = 0; i < 120; i += 1) {
    const cls = i % 2;
    const cx = cls === 0 ? -1.4 : 1.4;
    const cy = cls === 0 ? -1.0 : 1.0;
    rows.push([cx + gauss(rng, 0, 0.55), cy + gauss(rng, 0, 0.55)]);
    labels.push(cls);
  }
  return {
    name: "blobs",
    task: "classification",
    featureNames: ["x1", "x2"],
    targetName: "class",
    X: matrix(rows),
    y: vector(labels),
    classNames: ["A", "B"],
  };
}

/**
 * Two interleaving half-circles with label noise — k=1 memorizes the flips
 * (train accuracy 1.0) while test accuracy collapses. The overfitting demo.
 */
function makeMoons(seed = 7): Dataset {
  const rng = makeRng(seed);
  const rows: number[][] = [];
  const labels: number[] = [];
  for (let i = 0; i < 80; i += 1) {
    const t = (i / 79) * Math.PI;
    rows.push([Math.cos(t) + gauss(rng, 0, 0.1), Math.sin(t) + gauss(rng, 0, 0.1)]);
    labels.push(0);
  }
  for (let i = 0; i < 80; i += 1) {
    const t = (i / 79) * Math.PI;
    rows.push([
      1 - Math.cos(t) + gauss(rng, 0, 0.1),
      1 - Math.sin(t) - 0.5 + gauss(rng, 0, 0.1),
    ]);
    labels.push(1);
  }
  // ~20% label flips: a 1-NN model will lock onto the wrong labels on train.
  for (let i = 0; i < labels.length; i += 1) {
    if (rng() < 0.2) {
      labels[i] = labels[i] === 1 ? 0 : 1;
    }
  }
  return {
    name: "moons",
    task: "classification",
    featureNames: ["x1", "x2"],
    targetName: "class",
    X: matrix(rows),
    y: vector(labels),
    classNames: ["A", "B"],
  };
}

/**
 * y = 2.5x + 1 with light noise — honest linear fit.
 */
function makeNoisyLine(seed = 11): Dataset {
  const rng = makeRng(seed);
  const rows: number[][] = [];
  const labels: number[] = [];
  for (let i = 0; i < 60; i += 1) {
    const x = -2 + (4 * i) / 59;
    rows.push([x]);
    labels.push(2.5 * x + 1 + gauss(rng, 0, 0.45));
  }
  return {
    name: "noisy_line",
    task: "regression",
    featureNames: ["x"],
    targetName: "y",
    X: matrix(rows),
    y: vector(labels),
  };
}

/**
 * Mostly linear with two heavy outliers — makes overfitting / robustness visible.
 */
function makeOutlierLine(seed = 3): Dataset {
  const rng = makeRng(seed);
  const rows: number[][] = [];
  const labels: number[] = [];
  for (let i = 0; i < 40; i += 1) {
    const x = -2 + (4 * i) / 39;
    rows.push([x]);
    labels.push(1.5 * x + 0.2 + gauss(rng, 0, 0.2));
  }
  rows.push([1.8], [1.9]);
  labels.push(12, -10);
  return {
    name: "outlier_line",
    task: "regression",
    featureNames: ["x"],
    targetName: "y",
    X: matrix(rows),
    y: vector(labels),
  };
}

/**
 * Two features with wildly different units — scaling trap for distance models.
 * x1 in ~[0, 1], x2 in ~[0, 1000]. Class is mostly a function of x2.
 */
function makeScaleTrap(seed = 19): Dataset {
  const rng = makeRng(seed);
  const rows: number[][] = [];
  const labels: number[] = [];
  for (let i = 0; i < 100; i += 1) {
    const x1 = rng();
    const x2 = rng() * 1000;
    const cls = x2 > 500 ? 1 : 0;
    rows.push([x1, x2 + gauss(rng, 0, 20)]);
    labels.push(cls);
  }
  return {
    name: "scale_trap",
    task: "classification",
    featureNames: ["budget", "units"],
    targetName: "churn",
    X: matrix(rows),
    y: vector(labels),
    classNames: ["stay", "churn"],
  };
}

const REGISTRY: Record<DatasetName, (seed?: number) => Dataset> = {
  blobs: makeBlobs,
  moons: makeMoons,
  noisy_line: makeNoisyLine,
  outlier_line: makeOutlierLine,
  scale_trap: makeScaleTrap,
  poly_curve: makePolyCurve,
  mixed_table: makeMixedTable,
  clusters: makeClusters,
  dup_features: makeDupFeatures,
};

export const DATASET_NAMES = Object.keys(REGISTRY) as DatasetName[];

/**
 * Construct a built-in dataset.
 *
 * Args:
 *   name: Dataset identifier.
 *   seed: PRNG seed for noise (optional; dataset default when omitted).
 *
 * Returns:
 *   A fresh Dataset instance (mutable arrays are copies).
 *
 * Raises:
 *   Error: If name is unknown.
 *
 * Examples:
 *   >>> loadDataset("blobs").task
 *   'classification'
 */
export function loadDataset(name: DatasetName, seed?: number): Dataset {
  const factory = REGISTRY[name];
  if (!factory) {
    throw new Error(`Unknown dataset '${name}'`);
  }
  return factory(seed);
}

/**
 * y = 0.4x^3 - 1.2x + 1 + noise — linear fit fails; polynomial fits.
 */
function makePolyCurve(seed = 23): Dataset {
  const rng = makeRng(seed);
  const rows: number[][] = [];
  const labels: number[] = [];
  for (let i = 0; i < 60; i += 1) {
    const x = -1.6 + (3.2 * i) / 59;
    rows.push([x]);
    labels.push(0.4 * x ** 3 - 1.2 * x + 1 + gauss(rng, 0, 0.45));
  }
  return {
    name: "poly_curve",
    task: "regression",
    featureNames: ["x"],
    targetName: "y",
    X: matrix(rows),
    y: vector(labels),
  };
}

/**
 * Mixed table: numeric spend + categorical region (+ missing spend).
 * Target is churn driven mostly by spend and region.
 */
function makeMixedTable(seed = 29): Dataset {
  const rng = makeRng(seed);
  const rows: number[][] = [];
  const labels: number[] = [];
  const regions = ["north", "south", "east", "west"];
  for (let i = 0; i < 120; i += 1) {
    const regionIdx = Math.floor(rng() * 4);
    const region = regions[regionIdx] ?? "north";
    // Encode region as numeric index 0..3 in column 1 — engine OneHot expands it.
    let spend = Math.abs(gauss(rng, 40 + regionIdx * 15, 12));
    if (rng() < 0.12) {
      spend = Number.NaN;
    }
    const loyalty = rng() < 0.35 ? 1 : 0;
    const risk = (Number.isFinite(spend) ? (100 - spend) / 100 : 0.7) + (regionIdx === 3 ? 0.15 : 0);
    rows.push([spend, regionIdx, loyalty]);
    labels.push(risk + gauss(rng, 0, 0.08) > 0.55 ? 1 : 0);
    void region;
  }
  return {
    name: "mixed_table",
    task: "classification",
    featureNames: ["spend", "region", "loyalty"],
    targetName: "churn",
    X: matrix(rows),
    y: vector(labels),
    classNames: ["stay", "churn"],
  };
}

/**
 * Three Gaussian clusters with two informative dims (for PCA / k-means).
 */
function makeClusters(seed = 31): Dataset {
  const rng = makeRng(seed);
  const rows: number[][] = [];
  const labels: number[] = [];
  const centers = [
    [0, 0],
    [4, 1],
    [1, 4],
  ];
  for (let c = 0; c < 3; c += 1) {
    for (let i = 0; i < 50; i += 1) {
      const cx = centers[c]?.[0] ?? 0;
      const cy = centers[c]?.[1] ?? 0;
      rows.push([cx + gauss(rng, 0, 0.55), cy + gauss(rng, 0, 0.55)]);
      labels.push(c);
    }
  }
  return {
    name: "clusters",
    task: "classification",
    featureNames: ["x1", "x2"],
    targetName: "cluster",
    X: matrix(rows),
    y: vector(labels),
    classNames: ["c0", "c1", "c2"],
  };
}

/**
 * Two redundant copies of the same signal — PCA should find rank-1 structure.
 */
function makeDupFeatures(seed = 37): Dataset {
  const rng = makeRng(seed);
  const rows: number[][] = [];
  const labels: number[] = [];
  for (let i = 0; i < 100; i += 1) {
    const s = gauss(rng, 0, 1);
    rows.push([s, s + gauss(rng, 0, 0.05), s + gauss(rng, 0, 0.05), gauss(rng, 0, 0.2)]);
    labels.push(0);
  }
  return {
    name: "dup_features",
    task: "regression",
    featureNames: ["a", "a_copy1", "a_copy2", "noise"],
    targetName: "probe",
    X: matrix(rows),
    y: vector(labels),
  };
}

/**
 * List datasets with short descriptions for `help` / level copy.
 */
export function describeDatasets(): { name: DatasetName; task: string; blurb: string }[] {
  return [
    { name: "blobs", task: "classification", blurb: "Two clean Gaussian blobs. Linear models work." },
    { name: "moons", task: "classification", blurb: "Two moons with label noise. k=1 memorizes; watch the gap." },
    { name: "noisy_line", task: "regression", blurb: "y ≈ 2.5x + 1 with noise. Ordinary linear regression." },
    { name: "outlier_line", task: "regression", blurb: "Line plus two outliers. Compare linear vs ridge." },
    { name: "scale_trap", task: "classification", blurb: "Features in different units. Distance models need scaling." },
    { name: "poly_curve", task: "regression", blurb: "Cubic curve. Linear fails; polynomial + regularization." },
    { name: "mixed_table", task: "classification", blurb: "Numeric + categorical + missing. Encode and impute." },
    { name: "clusters", task: "classification", blurb: "Three clusters. PCA and k-means." },
    { name: "dup_features", task: "regression", blurb: "Copies of one signal. PCA rank / redundancy." },
  ];
}
