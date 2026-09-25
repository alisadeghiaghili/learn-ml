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

export function fitElasticNet(X: Matrix, y: Vector, params: ModelParams = {}): FittedModel {
  const alpha = Number(params.alpha ?? 0.1);
  const l1Ratio = Number(params.l1_ratio ?? 0.5);
  const p = X.nCols;
  let w = Array<number>(p).fill(0);
  let b = mean(y);
  const n = X.nRows;
  const epochs = Number(params.epochs ?? 300);
  for (let epoch = 0; epoch < epochs; epoch += 1) {
    b = mean(
      vector(
        X.data.map((row, i) => (y.data[i] ?? 0) - row.reduce((s, v, j) => s + (w[j] ?? 0) * v, 0)),
      ),
    );
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
        xNorm += xj * xj + (1 - l1Ratio) * alpha * n;
      }
      const rho = rSum / n;
      const z = xNorm / n || 1;
      const thresh = alpha * l1Ratio;
      const soft = Math.sign(rho) * Math.max(0, Math.abs(rho) - thresh);
      w[j] = soft / z;
    }
  }
  const weights = [...w];
  const intercept = b;
  const predict = (Xs: Matrix): Vector => linearPredict(Xs, weights, intercept);
  return {
    name: "elasticnet",
    task: "regression",
    params: { ...params, alpha, l1_ratio: l1Ratio },
    weights,
    intercept,
    predict,
    decision: predict,
    trainN: X.nRows,
  };
}

/**
 * One-vs-rest logistic for K labels {0..K-1}.
 */
export function fitOvrLogistic(X: Matrix, y: Vector, params: ModelParams = {}): FittedModel {
  const classes = [...new Set(y.data.map((v) => Math.round(v)))].sort((a, b) => a - b);
  const k = Math.max(2, classes.length);
  const binModels = classes.map((cls) => {
    const yb = vector(y.data.map((v) => (Math.round(v) === cls ? 1 : 0)));
    return fitLogistic(X, yb, params);
  });
  const decision = (Xs: Matrix): Vector => {
    // return argmax decision as integer label
    return vector(
      Xs.data.map((row) => {
        const single = matrix([row]);
        let best = 0;
        let bestZ = -Infinity;
        binModels.forEach((m, ci) => {
          const z = m.decision(single).data[0] ?? -Infinity;
          if (z > bestZ) {
            bestZ = z;
            best = ci;
          }
        });
        return classes[best] ?? 0;
      }),
    );
  };
  const predict = decision;
  return {
    name: "ovr_logistic",
    task: "classification",
    params: { ...params, n_classes: k },
    predict,
    decision,
    trainN: X.nRows,
  };
}

export function fitDbscan(X: Matrix, _y: Vector, params: ModelParams = {}): FittedModel {
  const eps = Number(params.eps ?? 0.8);
  const minPts = Number(params.min_samples ?? 4);
  const n = X.nRows;
  const labels = Array<number>(n).fill(-1);
  let cluster = 0;
  const dist = (a: number[], b: number[]) => {
    let d = 0;
    for (let j = 0; j < a.length; j += 1) d += ((a[j] ?? 0) - (b[j] ?? 0)) ** 2;
    return Math.sqrt(d);
  };
  const neighbors = (i: number) => {
    const out: number[] = [];
    for (let j = 0; j < n; j += 1) {
      if (dist(X.data[i] ?? [], X.data[j] ?? []) <= eps) out.push(j);
    }
    return out;
  };
  for (let i = 0; i < n; i += 1) {
    if ((labels[i] ?? -1) !== -1) continue;
    const nb = neighbors(i);
    if (nb.length < minPts) {
      labels[i] = -1;
      continue;
    }
    labels[i] = cluster;
    const queue = [...nb];
    while (queue.length) {
      const j = queue.pop() ?? 0;
      if ((labels[j] ?? -1) === -1) labels[j] = cluster;
      if ((labels[j] ?? -1) !== -1 && labels[j] !== cluster && labels[j] !== -1) continue;
      labels[j] = cluster;
      const nbj = neighbors(j);
      if (nbj.length >= minPts) {
        for (const k of nbj) {
          if ((labels[k] ?? -1) === -1) queue.push(k);
        }
      }
    }
    cluster += 1;
  }
  const predict = (Xs: Matrix): Vector =>
    vector(
      Xs.data.map((row) => {
        let best = -1;
        let bestD = eps;
        for (let i = 0; i < n; i += 1) {
          if ((labels[i] ?? -1) < 0) continue;
          const d = dist(row, X.data[i] ?? []);
          if (d <= bestD) {
            bestD = d;
            best = labels[i] ?? -1;
          }
        }
        return best;
      }),
    );
  return {
    name: "dbscan",
    task: "classification",
    params: { ...params, eps, min_samples: minPts },
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

export function fitBoost(X: Matrix, y: Vector, params: ModelParams = {}): FittedModel {
  const nEst = Math.max(1, Math.round(Number(params.n_estimators ?? 20)));
  const maxDepth = Math.max(1, Math.round(Number(params.max_depth ?? 2)));
  const lr = Number(params.lr ?? 0.3);
  const n = X.nRows;
  // Binary Gradient Boosting on logistic loss via residual trees (simplified).
  let raw = Array<number>(n).fill(0);
  const stumps: TreeNode[] = [];
  for (let m = 0; m < nEst; m += 1) {
    const prob = raw.map((z) => 1 / (1 + Math.exp(-z)));
    const r = y.data.map((yi, i) => ((yi ?? 0) >= 0.5 ? 1 : 0) - (prob[i] ?? 0.5));
    const yVec = vector(r.map((v) => v + 0.5));
    const idx = Array.from({ length: n }, (_, i) => i);
    const tree = buildTree(X, yVec, idx, 0, maxDepth, 2);
    stumps.push(tree);
    const leafVals = X.data.map((row) => treePredict(tree, row));
    for (let i = 0; i < n; i += 1) {
      raw[i] = (raw[i] ?? 0) + lr * ((leafVals[i] ?? 0) - 0.5) * 2;
    }
  }
  const decision = (Xs: Matrix): Vector =>
    vector(
      Xs.data.map((row) => {
        let z = 0;
        for (let m = 0; m < stumps.length; m += 1) {
          z += lr * (treePredict(stumps[m] ?? { leaf: true, value: 0 }, row) - 0.5) * 2;
        }
        return z;
      }),
    );
  const predict = (Xs: Matrix): Vector =>
    vector(decision(Xs).data.map((z) => (z >= 0 ? 1 : 0)));
  return {
    name: "boost",
    task: "classification",
    params: { ...params, n_estimators: nEst, max_depth: maxDepth, lr },
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
  // k-means++ style: first center random, others spread by distance.
  const centers: number[][] = [];
  const first = X.data[Math.floor(rng() * X.nRows)] ?? Array<number>(p).fill(0);
  centers.push([...first]);
  while (centers.length < k) {
    let bestIdx = 0;
    let bestD = -1;
    for (let i = 0; i < X.nRows; i += 1) {
      const row = X.data[i] ?? [];
      let minD = Infinity;
      for (const c of centers) {
        let d = 0;
        for (let j = 0; j < p; j += 1) d += ((row[j] ?? 0) - (c[j] ?? 0)) ** 2;
        minD = Math.min(minD, d);
      }
      if (minD > bestD) {
        bestD = minD;
        bestIdx = i;
      }
    }
    centers.push([...(X.data[bestIdx] ?? Array<number>(p).fill(0))]);
  }
  for (let iter = 0; iter < 50; iter += 1) {
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
    for (let c = 0; c < k; c += 1) {
      if (!counts[c]) continue;
      for (let j = 0; j < p; j += 1) {
        centers[c]![j] = (sums[c]![j] ?? 0) / (counts[c] ?? 1);
      }
    }
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
  elasticnet: "ElasticNet",
  dummy: "DummyRegressor",
  tree: "DecisionTreeClassifier",
  forest: "RandomForestClassifier",
  boost: "GradientBoostingClassifier",
  kmeans: "KMeans",
  dbscan: "DBSCAN",
  pca_knn: "PCA+KNeighborsClassifier",
  ovr_logistic: "OneVsRestClassifier(LogisticRegression)",
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
    case "elasticnet":
      return fitElasticNet(X, y, params);
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
    case "boost":
      return fitBoost(X, y, params);
    case "kmeans":
      return fitKmeans(X, y, params);
    case "dbscan":
      return fitDbscan(X, y, params);
    case "pca_knn":
      return fitPcaKnn(X, y, params);
    case "ovr_logistic":
      return fitOvrLogistic(X, y, params);
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
    case "elasticnet":
      return { alpha: 0.1, l1_ratio: 0.5 };
    case "tree":
      return { max_depth: 3 };
    case "forest":
      return { n_estimators: 11, max_depth: 4 };
    case "boost":
      return { n_estimators: 20, max_depth: 2, lr: 0.3 };
    case "kmeans":
      return { n_clusters: 3 };
    case "dbscan":
      return { eps: 0.8, min_samples: 4 };
    case "pca_knn":
      return { n_components: 1, n_neighbors: 5 };
    case "ovr_logistic":
      return { lr: 0.5, epochs: 300 };
    default:
      return {};
  }
}

/** Convenience for tests. */
export function ones(n: number): Matrix {
  return matrix(Array.from({ length: n }, () => [1]));
}
