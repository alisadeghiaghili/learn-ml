/**
 * Estimators with a scikit-learn-like fit/predict contract.
 *
 * Only what World 1 needs: DummyRegressor, LinearRegression, Ridge,
 * LogisticRegression (binary), and KNeighborsClassifier.
 */

import { mean, matrix, vector } from "./matrix";
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

function assertCols(X: Matrix, nCols: number, who: string): void {
  if (X.nCols !== nCols) {
    throw new Error(`${who} expects ${nCols} feature column(s), got ${X.nCols}`);
  }
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
  // Ridge penalty on weights only (not intercept).
  for (let j = 1; j < k; j += 1) {
    XtX[j]![j] = (XtX[j]![j] ?? 0) + alpha * n;
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
  assertCols(X, 1, "LinearRegression");
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
  assertCols(X, 1, "Ridge");
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

const SKLEARN_NAME: Record<ModelName, string> = {
  linear: "LinearRegression",
  logistic: "LogisticRegression",
  knn: "KNeighborsClassifier",
  ridge: "Ridge",
  dummy: "DummyRegressor",
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
    case "logistic":
      return fitLogistic(X, y, params);
    case "knn":
      return fitKnn(X, y, params);
    case "dummy":
      return fitDummy(X, y, params);
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
    default:
      return {};
  }
}

/** Convenience for tests. */
export function ones(n: number): Matrix {
  return matrix(Array.from({ length: n }, () => [1]));
}
