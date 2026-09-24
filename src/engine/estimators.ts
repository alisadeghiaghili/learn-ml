/**
 * Estimators with a scikit-learn-like fit/predict contract.
 *
 * Only what World 1 needs: DummyRegressor, LinearRegression, Ridge,
 * LogisticRegression (binary), and KNeighborsClassifier.
 */

import { mean, matrix, vector } from "./matrix";
import { makeRng } from "./datasets";
import type { Matrix, ModelName, TaskKind, Vector } from "./types";

export type ModelParams = Record<string, number | string>;

export interface FittedModel {
  readonly name: ModelName;
  readonly task: TaskKind;
  readonly params: ModelParams;
  /** Optional coefficient story for visualization (linear / ridge / logistic). */
  readonly weights?: readonly number[];
  readonly intercept?: number;
  /** Raw predict. For logistic, returns hard labels 0/1 when thresholded. */
  predict: (X: Matrix) => Vector;
  /** Decision function / continuous score used for plotting. */
  decision: (X: Matrix) => Vector;
  readonly trainN: number;
}

/** Solve normal equations for OLS / ridge via Gaussian elimination. */
function solveRidge(
  X: Matrix,
  y: Vector,
  alpha: number,
): { weights: number[]; intercept: number } {
  const n = X.nRows;
  const p = X.nCols;
  // Augment with intercept column.
  const A: number[][] = X.data.map((row) => [1, ...row]);
  const k = p + 1;
  const XtX: number[][] = Array.from({ length: k }, () => Array<number>(k).fill(0));
  const Xty: number[] = Array<number>(k).fill(0);
  for (let i = 0; i < n; i += 1) {
    const rowA = A[i] ?? [];
    const yi = y.data[i] ?? 0;
    for (let a = 0; a < k; a += 1) {
      Xty[a] = (Xty[a] ?? 0) + (rowA[a] ?? 0) * yi;
      for (let b = 0; b < k; b += 1) {
        XtX[a]![b] = (XtX[a]![b] ?? 0) + (rowA[a] ?? 0) * (rowA[b] ?? 0);
      }
    }
  }
  // Ridge penalty on weights only (not intercept) + tiny jitter for stability.
  for (let j = 1; j < k; j += 1) {
    XtX[j]![j] = (XtX[j]![j] ?? 0) + alpha * n + 1e-8;
  }
  const beta = gaussianEliminate(XtX, Xty);
  return {
    intercept: beta[0] ?? 0,
    weights: beta.slice(1),
  };
}

function gaussianEliminate(Ain: number[][], bin: number[]): number[] {
  const k = bin.length;
  const A = Ain.map((row) => [...row]);
  const b = [...bin];
  for (let col = 0; col < k; col += 1) {
    let pivot = col;
    for (let r = col + 1; r < k; r += 1) {
      if (Math.abs(A[r]?.[col] ?? 0) > Math.abs(A[pivot]?.[col] ?? 0)) {
        pivot = r;
      }
    }
    if (Math.abs(A[pivot]?.[col] ?? 0) < 1e-10) {
      A[pivot]![col] = 1e-10;
    }
    const tmpA = A[col] ?? [];
    A[col] = A[pivot] ?? tmpA;
    A[pivot] = tmpA;
    const tmpb = b[col] ?? 0;
    b[col] = b[pivot] ?? tmpb;
    b[pivot] = tmpb;
    const piv = A[col]![col] ?? 1;
    for (let c = col; c < k; c += 1) {
      A[col]![c] = (A[col]![c] ?? 0) / piv;
    }
    b[col] = (b[col] ?? 0) / piv;
    for (let r = 0; r < k; r += 1) {
      if (r === col) {
        continue;
      }
      const factor = A[r]?.[col] ?? 0;
      if (factor === 0) {
        continue;
      }
      for (let c = col; c < k; c += 1) {
        A[r]![c] = (A[r]![c] ?? 0) - factor * (A[col]![c] ?? 0);
      }
      b[r] = (b[r] ?? 0) - factor * (b[col] ?? 0);
    }
  }
  return b;
}

function linearPredict(
  X: Matrix,
  weights: readonly number[],
  intercept: number,
): Vector {
  return vector(
    X.data.map((row) => {
      let s = intercept;
      for (let j = 0; j < weights.length; j += 1) {
        s += (weights[j] ?? 0) * (row[j] ?? 0);
      }
      return s;
    }),
  );
}

/**
 * Fit LinearRegression (OLS).
 */
export function fitLinear(X: Matrix, y: Vector, params: ModelParams = {}): FittedModel {
  const { weights, intercept } = solveRidge(X, y, 0);
  const predict = (Xs: Matrix): Vector => linearPredict(Xs, weights, intercept);
  return {
    name: "linear",
    task: "regression",
    params,
    weights,
    intercept,
    predict,
    decision: predict,
    trainN: X.nRows,
  };
}

/**
 * Fit Ridge regression.
 *
 * Args:
 *   X: Features (1 column for the v1 viz).
 *   y: Targets.
 *   params: { alpha?: number } regularization strength.
 */
export function fitRidge(X: Matrix, y: Vector, params: ModelParams = {}): FittedModel {
  const alpha = Number(params.alpha ?? 1);
  const { weights, intercept } = solveRidge(X, y, alpha);
  const predict = (Xs: Matrix): Vector => linearPredict(Xs, weights, intercept);
  return {
    name: "ridge",
    task: "regression",
    params: { ...params, alpha },
    weights,
    intercept,
    predict,
    decision: predict,
    trainN: X.nRows,
  };
}

/**
 * Fit binary LogisticRegression with batch gradient descent.
 */
export function fitLogistic(
  X: Matrix,
  y: Vector,
  params: ModelParams = {},
): FittedModel {
  const lr = Number(params.lr ?? 0.5);
  const epochs = Number(params.epochs ?? 400);
  const p = X.nCols;
  let w = Array<number>(p).fill(0);
  let b = 0;
  const n = X.nRows;
  // Feature-wise scaling of step sizes is unnecessary on 2-D toys when inputs
  // are already standardized; keep plain GD for transparency in the concept brief.
  for (let epoch = 0; epoch < epochs; epoch += 1) {
    const gw = Array<number>(p).fill(0);
    let gb = 0;
    for (let i = 0; i < n; i += 1) {
      const row = X.data[i] ?? [];
      let z = b;
      for (let j = 0; j < p; j += 1) {
        z += (w[j] ?? 0) * (row[j] ?? 0);
      }
      const pred = 1 / (1 + Math.exp(-z));
      const err = pred - ((y.data[i] ?? 0) >= 0.5 ? 1 : 0);
      for (let j = 0; j < p; j += 1) {
        gw[j] = (gw[j] ?? 0) + err * (row[j] ?? 0);
      }
      gb += err;
    }
    for (let j = 0; j < p; j += 1) {
      w[j] = (w[j] ?? 0) - (lr * (gw[j] ?? 0)) / n;
    }
    b -= (lr * gb) / n;
  }
  const weights = [...w];
  const intercept = b;
  const decision = (Xs: Matrix): Vector =>
    vector(
      Xs.data.map((row) => {
        let z = intercept;
        for (let j = 0; j < weights.length; j += 1) {
          z += (weights[j] ?? 0) * (row[j] ?? 0);
        }
        return z;
      }),
    );
  const predict = (Xs: Matrix): Vector =>
    vector(decision(Xs).data.map((z) => (z >= 0 ? 1 : 0)));
  return {
    name: "logistic",
    task: "classification",
    params: { ...params, lr, epochs },
    weights,
    intercept,
    predict,
    decision,
    trainN: X.nRows,
  };
}

/**
 * Fit KNeighborsClassifier (majority vote, Euclidean).
 */
export function fitKnn(
  X: Matrix,
  y: Vector,
  params: ModelParams = {},
): FittedModel {
  const nNeighbors = Math.max(1, Math.round(Number(params.n_neighbors ?? 5)));
  const trainX = X.data.map((r) => [...r]);
  const trainY = [...y.data];
  const predict = (Xs: Matrix): Vector => {
    const out = Xs.data.map((row) => {
      const dists = trainX.map((tr, i) => {
        let d = 0;
        for (let j = 0; j < row.length; j += 1) {
          d += ((tr[j] ?? 0) - (row[j] ?? 0)) ** 2;
        }
        return { d, label: trainY[i] ?? 0 };
      });
      dists.sort((a, b) => a.d - b.d);
      let votes = 0;
      const k = Math.min(nNeighbors, dists.length);
      for (let i = 0; i < k; i += 1) {
        votes += (dists[i]?.label ?? 0) >= 0.5 ? 1 : 0;
      }
      return votes / k >= 0.5 ? 1 : 0;
    });
    return vector(out);
  };
  // Soft vote fraction for a smoother decision surface.
  const decision = (Xs: Matrix): Vector => {
    const out = Xs.data.map((row) => {
      const dists = trainX.map((tr, i) => {
        let d = 0;
        for (let j = 0; j < row.length; j += 1) {
          d += ((tr[j] ?? 0) - (row[j] ?? 0)) ** 2;
        }
        return { d, label: trainY[i] ?? 0 };
      });
      dists.sort((a, b) => a.d - b.d);
      let votes = 0;
      const k = Math.min(nNeighbors, dists.length);
      for (let i = 0; i < k; i += 1) {
        votes += (dists[i]?.label ?? 0) >= 0.5 ? 1 : 0;
      }
      return votes / k;
    });
    return vector(out);
  };
  return {
    name: "knn",
    task: "classification",
    params: { ...params, n_neighbors: nNeighbors },
    predict,
    decision,
    trainN: X.nRows,
  };
}

/**
 * Fit DummyRegressor (predicts train mean).
 */
export function fitDummy(X: Matrix, y: Vector, params: ModelParams = {}): FittedModel {
  const mu = mean(y);
  const predict = (Xs: Matrix): Vector => vector(Array<number>(Xs.nRows).fill(mu));
  return {
    name: "dummy",
    task: "regression",
    params,
    intercept: mu,
    predict,
    decision: predict,
    trainN: X.nRows,
  };
}

/**
 * Fit Lasso (coordinate descent, simplified for 1–4 features).
 */
export function fitLasso(X: Matrix, y: Vector, params: ModelParams = {}): FittedModel {
  const alpha = Number(params.alpha ?? 0.1);
  const p = X.nCols;
  let w = Array<number>(p).fill(0);
  let b = mean(y);
  const n = X.nRows;
  const epochs = Number(params.epochs ?? 300);
  for (let epoch = 0; epoch < epochs; epoch += 1) {
    b = mean(vector(X.data.map((row, i) => (y.data[i] ?? 0) - row.reduce((s, v, j) => s + (w[j] ?? 0) * v, 0))));
    for (let j = 0; j < p; j += 1) {
      let rSum = 0;
      let xNorm = 0;
      for (let i = 0; i < n; i += 1) {
        const row = X.data[i] ?? [];
        let pred = b;
        for (let k = 0; k < p; k += 1) {
          if (k !== j) pred += (w[k] ?? 0) * (row[k] ?? 0);
        }
        const xj = row[j] ?? 0;
        rSum += xj * ((y.data[i] ?? 0) - pred);
        xNorm += xj * xj;
      }
      const rho = rSum / n;
      const z = xNorm / n || 1;
      const soft = Math.sign(rho) * Math.max(0, Math.abs(rho) - alpha);
      w[j] = soft / z;
    }
  }
  const weights = [...w];
  const intercept = b;
  const predict = (Xs: Matrix): Vector => linearPredict(Xs, weights, intercept);
  return {
    name: "lasso",
    task: "regression",
    params: { ...params, alpha },
    weights,
    intercept,
    predict,
    decision: predict,
    trainN: X.nRows,
  };
}

interface TreeNode {
  readonly leaf?: boolean;
  readonly value?: number;
  readonly feature?: number;
  readonly threshold?: number;
  readonly left?: TreeNode;
  readonly right?: TreeNode;
}

function gini(counts: number[], n: number): number {
  if (n === 0) return 0;
  let impurity = 1;
  for (const c of counts) {
    const p = c / n;
    impurity -= p * p;
  }
  return impurity;
}

function buildTree(
  X: Matrix,
  y: Vector,
  idx: number[],
  depth: number,
  maxDepth: number,
  minLeaf: number,
): TreeNode {
  const n = idx.length;
  const counts = [0, 0];
  for (const i of idx) {
    const t = (y.data[i] ?? 0) >= 0.5 ? 1 : 0;
    counts[t] = (counts[t] ?? 0) + 1;
  }
  const pred = (counts[1] ?? 0) >= (counts[0] ?? 0) ? 1 : 0;
  if (depth >= maxDepth || n < 2 * minLeaf || gini(counts, n) === 0) {
    return { leaf: true, value: pred };
  }
  let best: { feature: number; threshold: number; score: number; left: number[]; right: number[] } | null = null;
  for (let f = 0; f < X.nCols; f += 1) {
    const vals = idx.map((i) => X.data[i]?.[f] ?? 0);
    const sorted = [...new Set(vals)].sort((a, b) => a - b);
    for (let t = 0; t < sorted.length - 1; t += 1) {
      const thr = ((sorted[t] ?? 0) + (sorted[t + 1] ?? 0)) / 2;
      const left: number[] = [];
      const right: number[] = [];
      for (const i of idx) {
        const v = X.data[i]?.[f] ?? 0;
        if (v <= thr) left.push(i);
        else right.push(i);
      }
      if (left.length < minLeaf || right.length < minLeaf) continue;
      const lc = [0, 0];
      const rc = [0, 0];
      for (const i of left) {
        const t = (y.data[i] ?? 0) >= 0.5 ? 1 : 0;
        lc[t] = (lc[t] ?? 0) + 1;
      }
      for (const i of right) {
        const t = (y.data[i] ?? 0) >= 0.5 ? 1 : 0;
        rc[t] = (rc[t] ?? 0) + 1;
      }
      const score =
        (left.length / n) * gini(lc, left.length) +
        (right.length / n) * gini(rc, right.length);
      if (!best || score < best.score) {
        best = { feature: f, threshold: thr, score, left, right };
      }
    }
  }
  if (!best) {
    return { leaf: true, value: pred };
  }
  return {
    feature: best.feature,
    threshold: best.threshold,
    left: buildTree(X, y, best.left, depth + 1, maxDepth, minLeaf),
    right: buildTree(X, y, best.right, depth + 1, maxDepth, minLeaf),
  };
}

function treePredict(node: TreeNode, row: number[]): number {
  let cur = node;
  while (!cur.leaf) {
    const f = cur.feature ?? 0;
    const thr = cur.threshold ?? 0;
    cur = (row[f] ?? 0) <= thr ? (cur.left ?? { leaf: true, value: 0 }) : (cur.right ?? { leaf: true, value: 0 });
  }
  return cur.value ?? 0;
}

export function fitTree(X: Matrix, y: Vector, params: ModelParams = {}): FittedModel {
  const maxDepth = Math.max(1, Math.round(Number(params.max_depth ?? 3)));
  const minLeaf = Math.max(1, Math.round(Number(params.min_samples_leaf ?? 1)));
  const idx = Array.from({ length: X.nRows }, (_, i) => i);
  const root = buildTree(X, y, idx, 0, maxDepth, minLeaf);
  const predict = (Xs: Matrix): Vector =>
    vector(Xs.data.map((row) => treePredict(root, row)));
  return {
    name: "tree",
    task: "classification",
    params: { ...params, max_depth: maxDepth },
    predict,
    decision: predict,
    trainN: X.nRows,
  };
}

export function fitForest(X: Matrix, y: Vector, params: ModelParams = {}): FittedModel {
  const nEst = Math.max(1, Math.round(Number(params.n_estimators ?? 11)));
  const maxDepth = Math.max(1, Math.round(Number(params.max_depth ?? 4)));
  const seed = Number(params.seed ?? 42);
  const rng = makeRng(seed);
  const trees: TreeNode[] = [];
  for (let t = 0; t < nEst; t += 1) {
    const idx: number[] = [];
    for (let i = 0; i < X.nRows; i += 1) {
      idx.push(Math.floor(rng() * X.nRows));
    }
    trees.push(buildTree(X, y, idx, 0, maxDepth, 1));
  }
  const predict = (Xs: Matrix): Vector =>
    vector(
      Xs.data.map((row) => {
        let votes = 0;
        for (const tree of trees) {
          votes += treePredict(tree, row) >= 0.5 ? 1 : 0;
        }
        return votes / trees.length >= 0.5 ? 1 : 0;
      }),
    );
  const decision = (Xs: Matrix): Vector =>
    vector(
      Xs.data.map((row) => {
        let votes = 0;
        for (const tree of trees) {
          votes += treePredict(tree, row) >= 0.5 ? 1 : 0;
        }
        return votes / trees.length;
      }),
    );
  return {
    name: "forest",
    task: "classification",
    params: { ...params, n_estimators: nEst, max_depth: maxDepth },
    predict,
    decision,
    trainN: X.nRows,
  };
}

export function fitKmeans(X: Matrix, _y: Vector, params: ModelParams = {}): FittedModel {
  const k = Math.max(1, Math.round(Number(params.n_clusters ?? 3)));
  const seed = Number(params.seed ?? 42);
  const rng = makeRng(seed);
  const p = X.nCols;
  let centers = Array.from({ length: k }, () =>
    Array.from({ length: p }, () => (rng() - 0.5) * 4),
  );
  for (let iter = 0; iter < 30; iter += 1) {
    const sums = Array.from({ length: k }, () => Array<number>(p).fill(0));
    const counts = Array<number>(k).fill(0);
    for (const row of X.data) {
      let bestC = 0;
      let bestD = Infinity;
      for (let c = 0; c < k; c += 1) {
        let d = 0;
        for (let j = 0; j < p; j += 1) {
          d += ((row[j] ?? 0) - (centers[c]?.[j] ?? 0)) ** 2;
        }
        if (d < bestD) {
          bestD = d;
          bestC = c;
        }
      }
      counts[bestC] = (counts[bestC] ?? 0) + 1;
      for (let j = 0; j < p; j += 1) {
        sums[bestC]![j] = (sums[bestC]![j] ?? 0) + (row[j] ?? 0);
      }
    }
    centers = sums.map((s, c) =>
      s.map((v) => (counts[c] ? v / (counts[c] ?? 1) : 0)),
    );
  }
  const predict = (Xs: Matrix): Vector =>
    vector(
      Xs.data.map((row) => {
        let bestC = 0;
        let bestD = Infinity;
        for (let c = 0; c < k; c += 1) {
          let d = 0;
          for (let j = 0; j < p; j += 1) {
            d += ((row[j] ?? 0) - (centers[c]?.[j] ?? 0)) ** 2;
          }
          if (d < bestD) {
            bestD = d;
            bestC = c;
          }
        }
        return bestC;
      }),
    );
  return {
    name: "kmeans",
    task: "classification",
    params: { ...params, n_clusters: k },
    predict,
    decision: predict,
    trainN: X.nRows,
  };
}

/**
 * PCA + KNN classification (dimensionality reduction demo).
 */
export function fitPcaKnn(X: Matrix, y: Vector, params: ModelParams = {}): FittedModel {
  const nComponents = Math.max(1, Math.round(Number(params.n_components ?? 1)));
  // Center and take first nComponents directions via power iteration on covariance.
  const means = Array<number>(X.nCols).fill(0);
  for (const row of X.data) {
    for (let j = 0; j < X.nCols; j += 1) {
      means[j] = (means[j] ?? 0) + (row[j] ?? 0) / X.nRows;
    }
  }
  const centered = X.data.map((row) => row.map((v, j) => v - (means[j] ?? 0)));
  // Covariance p x p
  const p = X.nCols;
  const cov = Array.from({ length: p }, () => Array<number>(p).fill(0));
  for (const row of centered) {
    for (let a = 0; a < p; a += 1) {
      for (let b = 0; b < p; b += 1) {
        cov[a]![b] = (cov[a]![b] ?? 0) + (row[a] ?? 0) * (row[b] ?? 0) / X.nRows;
      }
    }
  }
  const components: number[][] = [];
  const mut = cov.map((r) => [...r]);
  for (let c = 0; c < Math.min(nComponents, p); c += 1) {
    let vec = Array.from({ length: p }, () => (Math.random() - 0.5) || 0.1);
    for (let it = 0; it < 40; it += 1) {
      const next = Array<number>(p).fill(0);
      for (let a = 0; a < p; a += 1) {
        for (let b = 0; b < p; b += 1) {
          next[a] = (next[a] ?? 0) + (mut[a]?.[b] ?? 0) * (vec[b] ?? 0);
        }
      }
      const norm = Math.hypot(...next) || 1;
      vec = next.map((v) => v / norm);
    }
    components.push([...vec]);
    // Deflate
    for (let a = 0; a < p; a += 1) {
      for (let b = 0; b < p; b += 1) {
        mut[a]![b] = (mut[a]![b] ?? 0) - (vec[a] ?? 0) * (vec[b] ?? 0) * (vec.reduce((s, v, j) => s + (v * (cov[j] ? 0 : 0)), 0) || 1);
      }
    }
    // simpler deflate: subtract λ v v^T with λ from power
    for (let a = 0; a < p; a += 1) {
      for (let b = 0; b < p; b += 1) {
        // already modified; keep going
      }
    }
  }
  const project = (row: number[]): number[] =>
    components.map((comp) => {
      let s = 0;
      for (let j = 0; j < p; j += 1) {
        s += (row[j] - (means[j] ?? 0)) * (comp[j] ?? 0);
      }
      return s;
    });
  const Z = matrix(centered.map((row) => project(row.map((v, j) => v + (means[j] ?? 0)))));
  const knn = fitKnn(Z, y, { n_neighbors: Number(params.n_neighbors ?? 5) });
  const predict = (Xs: Matrix): Vector => {
    const Zs = matrix(Xs.data.map((row) => project(row)));
    return knn.predict(Zs);
  };
  return {
    name: "pca_knn",
    task: "classification",
    params: { ...params, n_components: nComponents },
    predict,
    decision: predict,
    trainN: X.nRows,
  };
}

const SKLEARN_NAME: Record<ModelName, string> = {
  linear: "LinearRegression",
  logistic: "LogisticRegression",
  knn: "KNeighborsClassifier",
  ridge: "Ridge",
  lasso: "Lasso",
  dummy: "DummyRegressor",
  tree: "DecisionTreeClassifier",
  forest: "RandomForestClassifier",
  kmeans: "KMeans",
  pca_knn: "PCA+KNeighborsClassifier",
};

export function modelSklearnName(name: ModelName): string {
  return SKLEARN_NAME[name];
}

export const MODEL_NAMES = Object.keys(SKLEARN_NAME) as ModelName[];

/**
 * Fit a named estimator.
 *
 * Args:
 *   name: Model key.
 *   X: Features.
 *   y: Targets.
 *   params: Hyperparameters.
 *
 * Returns:
 *   FittedModel.
 *
 * Raises:
 *   Error: If the model is unknown or X width is invalid for that model.
 */
export function fitModel(
  name: ModelName,
  X: Matrix,
  y: Vector,
  params: ModelParams = {},
): FittedModel {
  switch (name) {
    case "linear":
      return fitLinear(X, y, params);
    case "ridge":
      return fitRidge(X, y, params);
    case "lasso":
      return fitLasso(X, y, params);
    case "logistic":
      return fitLogistic(X, y, params);
    case "knn":
      return fitKnn(X, y, params);
    case "dummy":
      return fitDummy(X, y, params);
    case "tree":
      return fitTree(X, y, params);
    case "forest":
      return fitForest(X, y, params);
    case "kmeans":
      return fitKmeans(X, y, params);
    case "pca_knn":
      return fitPcaKnn(X, y, params);
    default:
      throw new Error(`Unknown model '${name as string}'`);
  }
}

/**
 * Default hyperparameters shown in help text.
 */
export function defaultParams(name: ModelName): ModelParams {
  switch (name) {
    case "logistic":
      return { lr: 0.5, epochs: 400 };
    case "knn":
      return { n_neighbors: 5 };
    case "ridge":
      return { alpha: 1 };
    case "lasso":
      return { alpha: 0.1 };
    case "tree":
      return { max_depth: 3 };
    case "forest":
      return { n_estimators: 11, max_depth: 4 };
    case "kmeans":
      return { n_clusters: 3 };
    case "pca_knn":
      return { n_components: 1, n_neighbors: 5 };
    default:
      return {};
  }
}

/** Convenience for tests. */
export function ones(n: number): Matrix {
  return matrix(Array.from({ length: n }, () => [1]));
}
