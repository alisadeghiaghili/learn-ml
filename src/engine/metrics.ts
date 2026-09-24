/**
 * Train/test split and evaluation metrics.
 */

import { mean, take, takeRows, totalSumOfSquares } from "./matrix";
import { makeRng } from "./datasets";
import type {
  ClassificationMetrics,
  Matrix,
  Metrics,
  RegressionMetrics,
  Split,
  TaskKind,
  Vector,
} from "./types";

/**
 * Shuffle indices and cut a hold-out test set.
 *
 * Args:
 *   X: Feature matrix.
 *   y: Target vector.
 *   testSize: Fraction of rows assigned to test, in (0, 1).
 *   seed: PRNG seed for reproducible splits.
 *
 * Returns:
 *   Split with X/y train and test partitions.
 *
 * Raises:
 *   Error: If shapes mismatch or testSize is out of range.
 *
 * Examples:
 *   >>> const s = trainTestSplit(X, y, 0.2, 42); s.XTest.nRows > 0
 *   true
 */
export function trainTestSplit(
  X: Matrix,
  y: Vector,
  testSize = 0.2,
  seed = 42,
): Split {
  if (X.nRows !== y.data.length) {
    throw new Error("X and y must have the same number of rows");
  }
  if (!(testSize > 0 && testSize < 1)) {
    throw new Error("testSize must be in (0, 1)");
  }
  const n = X.nRows;
  const idx = Array.from({ length: n }, (_, i) => i);
  const rng = makeRng(seed);
  for (let i = n - 1; i > 0; i -= 1) {
    const j = Math.floor(rng() * (i + 1));
    const tmp = idx[i] ?? 0;
    idx[i] = idx[j] ?? 0;
    idx[j] = tmp;
  }
  const nTest = Math.max(1, Math.round(n * testSize));
  const testIdx = idx.slice(0, nTest);
  const trainIdx = idx.slice(nTest);
  return {
    XTrain: takeRows(X, trainIdx),
    XTest: takeRows(X, testIdx),
    yTrain: take(y, trainIdx),
    yTest: take(y, testIdx),
    testSize,
    seed,
  };
}

/**
 * Classification metrics for binary labels {0, 1}, including log-loss.
 *
 * Args:
 *   yTrue: Ground-truth labels.
 *   yPred: Predicted labels (hard 0/1).
 *   yProba: Optional positive-class probabilities for log-loss.
 *
 * Returns:
 *   Accuracy, precision, recall, F1, log-loss, and 2x2 confusion [[TN, FP], [FN, TP]].
 */
export function classificationMetrics(
  yTrue: Vector,
  yPred: Vector,
  yProba?: Vector,
): ClassificationMetrics {
  let tp = 0;
  let fp = 0;
  let tn = 0;
  let fn = 0;
  let correct = 0;
  const n = yTrue.data.length;
  let llSum = 0;
  for (let i = 0; i < n; i += 1) {
    const t = (yTrue.data[i] ?? 0) >= 0.5 ? 1 : 0;
    const p = (yPred.data[i] ?? 0) >= 0.5 ? 1 : 0;
    if (t === p) {
      correct += 1;
    }
    if (t === 1 && p === 1) {
      tp += 1;
    } else if (t === 0 && p === 1) {
      fp += 1;
    } else if (t === 0 && p === 0) {
      tn += 1;
    } else {
      fn += 1;
    }
    const prob = yProba?.data[i] ?? (p === 1 ? 0.9 : 0.1);
    const clamped = Math.min(1 - 1e-6, Math.max(1e-6, prob));
    llSum += t === 1 ? -Math.log(clamped) : -Math.log(1 - clamped);
  }
  const precision = tp + fp === 0 ? 0 : tp / (tp + fp);
  const recall = tp + fn === 0 ? 0 : tp / (tp + fn);
  const f1 = precision + recall === 0 ? 0 : (2 * precision * recall) / (precision + recall);
  return {
    accuracy: n === 0 ? 0 : correct / n,
    precision,
    recall,
    f1,
    logLoss: n === 0 ? 0 : llSum / n,
    confusion: [
      [tn, fp],
      [fn, tp],
    ],
    n,
  };
}

/**
 * Regression metrics MAE, MSE, and R^2.
 *
 * Args:
 *   yTrue: Ground-truth targets.
 *   yPred: Predictions.
 *
 * Returns:
 *   RegressionMetrics.
 *
 * Raises:
 *   Error: If lengths differ.
 */
export function regressionMetrics(yTrue: Vector, yPred: Vector): RegressionMetrics {
  if (yTrue.data.length !== yPred.data.length) {
    throw new Error("yTrue and yPred length mismatch");
  }
  const n = yTrue.data.length;
  if (n === 0) {
    return { mae: 0, mse: 0, r2: 0, n: 0 };
  }
  let abs = 0;
  let sq = 0;
  for (let i = 0; i < n; i += 1) {
    const e = (yTrue.data[i] ?? 0) - (yPred.data[i] ?? 0);
    abs += Math.abs(e);
    sq += e * e;
  }
  const mae = abs / n;
  const mse = sq / n;
  const ssTot = totalSumOfSquares(yTrue);
  const r2 = ssTot === 0 ? 0 : 1 - sq / ssTot;
  return { mae, mse, r2, n };
}

/**
 * Dispatch metrics by task type.
 */
export function computeMetrics(
  task: TaskKind,
  yTrue: Vector,
  yPred: Vector,
  yProba?: Vector,
): Metrics {
  return task === "classification"
    ? classificationMetrics(yTrue, yPred, yProba)
    : regressionMetrics(yTrue, yPred);
}

/**
 * Format metrics as short terminal lines.
 */
export function formatMetrics(metrics: Metrics, label: string): string[] {
  const head = `${label}:`;
  if ("accuracy" in metrics) {
    return [
      head,
      `  accuracy ${metrics.accuracy.toFixed(3)}  precision ${metrics.precision.toFixed(3)}`,
      `  recall   ${metrics.recall.toFixed(3)}  f1        ${metrics.f1.toFixed(3)}`,
      `  logloss  ${metrics.logLoss.toFixed(3)}  (n=${metrics.n})`,
    ];
  }
  return [
    head,
    `  mae ${metrics.mae.toFixed(3)}  mse ${metrics.mse.toFixed(3)}  r2 ${metrics.r2.toFixed(3)}  (n=${metrics.n})`,
  ];
}

/**
 * Baseline constant predictor (mean for regression, majority for classification).
 */
export function baselinePredict(
  task: TaskKind,
  yTrain: Vector,
  n: number,
): Vector {
  if (task === "classification") {
    const meanLabel = mean(yTrain);
    const label = meanLabel >= 0.5 ? 1 : 0;
    return { data: Array<number>(n).fill(label) };
  }
  const mu = mean(yTrain);
  return { data: Array<number>(n).fill(mu) };
}
