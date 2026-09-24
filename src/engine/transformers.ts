/**
 * Feature transformers in the scikit-learn sense: fit on train, transform anywhere.
 */

import { colMeans, colMinMax, colStds, scaleShift } from "./matrix";
import type { EncoderName, ImputerName, Matrix, ScalerName, Vector } from "./types";
import { matrix } from "./matrix";

export interface Scaler {
  readonly name: ScalerName;
  readonly scale: Vector;
  readonly shift: Vector;
  readonly fittedOnRows: number;
}

/**
 * Fit a StandardScaler (z-score) or MinMaxScaler.
 *
 * Args:
 *   X: Training features.
 *   name: Scaler kind.
 *
 * Returns:
 *   Fitted scaler with per-column scale/shift such that transform is X*scale+shift.
 *
 * Raises:
 *   Error: If X is empty or name is unknown.
 */
export function fitScaler(X: Matrix, name: ScalerName): Scaler {
  if (X.nRows === 0) {
    throw new Error("fitScaler requires at least one row");
  }
  if (name === "standard") {
    const means = colMeans(X);
    const stds = colStds(X, means);
    // z = (x - mu) / s  =>  scale = 1/s, shift = -mu/s
    const scale = { data: stds.data.map((s) => 1 / s) };
    const shift = {
      data: means.data.map((m, j) => -m / (stds.data[j] ?? 1)),
    };
    return { name, scale, shift, fittedOnRows: X.nRows };
  }
  if (name === "minmax") {
    const { mins, maxs } = colMinMax(X);
    const scale = {
      data: mins.data.map((lo, j) => 1 / ((maxs.data[j] ?? 1) - lo)),
    };
    const shift = {
      data: mins.data.map((lo, j) => -lo / ((maxs.data[j] ?? 1) - lo)),
    };
    return { name, scale, shift, fittedOnRows: X.nRows };
  }
  throw new Error(`Unknown scaler '${name as string}'`);
}

/**
 * Apply a fitted scaler to features.
 *
 * Args:
 *   X: Features to transform.
 *   scaler: Fitted scaler.
 *
 * Returns:
 *   Scaled matrix with the same shape as X.
 */
export function transform(X: Matrix, scaler: Scaler): Matrix {
  return scaleShift(X, scaler.scale, scaler.shift);
}

/**
 * Human label for the pipeline graph.
 */
export function scalerLabel(name: ScalerName): string {
  return name === "standard" ? "StandardScaler" : "MinMaxScaler";
}

/**
 * Fit a simple imputer (mean/median/constant=0) column-wise.
 */
export function fitImputer(
  X: Matrix,
  name: ImputerName,
): { fill: number[]; name: ImputerName } {
  const fill: number[] = [];
  for (let j = 0; j < X.nCols; j += 1) {
    const col = X.data.map((r) => r[j] ?? 0).filter((v) => Number.isFinite(v));
    if (col.length === 0) {
      fill.push(0);
      continue;
    }
    if (name === "mean") {
      fill.push(col.reduce((a, b) => a + b, 0) / col.length);
    } else if (name === "median") {
      const sorted = [...col].sort((a, b) => a - b);
      fill.push(sorted[Math.floor(sorted.length / 2)] ?? 0);
    } else {
      fill.push(0);
    }
  }
  return { fill, name };
}

export function applyImputer(X: Matrix, imp: { fill: number[] }): Matrix {
  return matrix(
    X.data.map((row) =>
      row.map((v, j) => (Number.isFinite(v) ? v : (imp.fill[j] ?? 0))),
    ),
  );
}

/**
 * One-hot expand a categorical index column (first matching col).
 * v1 mixed_table uses col 1 as region index 0..3.
 */
export function oneHotColumn(X: Matrix, col: number): Matrix {
  let maxIdx = 0;
  for (const row of X.data) {
    maxIdx = Math.max(maxIdx, Math.round(row[col] ?? 0));
  }
  const levels = Math.max(1, maxIdx + 1);
  const rows = X.data.map((row) => {
    const out: number[] = [];
    for (let j = 0; j < X.nCols; j += 1) {
      if (j === col) {
        for (let k = 0; k < levels; k += 1) {
          out.push(Math.round(row[j] ?? -1) === k ? 1 : 0);
        }
      } else {
        out.push(row[j] ?? 0);
      }
    }
    return out;
  });
  return matrix(rows);
}

export interface PolyScaler {
  mid: number;
  scale: number;
  degree: number;
}

/**
 * Fit the [-1, 1] map for polynomial expansion on training x.
 */
export function fitPolyScaler(X: Matrix, degree: number): PolyScaler {
  let minX = Infinity;
  let maxX = -Infinity;
  for (const row of X.data) {
    const x = row[0] ?? 0;
    if (Number.isFinite(x)) {
      minX = Math.min(minX, x);
      maxX = Math.max(maxX, x);
    }
  }
  if (!Number.isFinite(minX)) {
    minX = 0;
    maxX = 1;
  }
  return {
    mid: (maxX + minX) / 2,
    scale: Math.max(1e-6, (maxX - minX) / 2),
    degree,
  };
}

/**
 * Polynomial features [x, x^2, ...] for a single column.
 * Uses a train-fitted scale so test never leaks min/max.
 */
export function polyExpand(X: Matrix, scaler: PolyScaler): Matrix {
  const degree = scaler.degree;
  if (degree < 1) {
    throw new Error("poly degree must be >= 1");
  }
  return matrix(
    X.data.map((row) => {
      const x = ((row[0] ?? 0) - scaler.mid) / scaler.scale;
      const out: number[] = [];
      for (let d = 1; d <= degree; d += 1) {
        out.push(x ** d);
      }
      return out;
    }),
  );
}

export function imputerLabel(name: ImputerName): string {
  return `SimpleImputer(strategy='${name}')`;
}

export function encoderLabel(name: EncoderName): string {
  return name === "onehot" ? "OneHotEncoder" : "OrdinalEncoder";
}

/**
 * Feature engineering transforms: interaction, binning, target encoding.
 */
export type FeMode = "interact" | "bin" | "target";

export interface FeState {
  mode: FeMode;
  bins: number[];
  targetMap: Record<string, number>;
}

/**
 * Fit FE params on train only (bins / target means).
 */
export function fitFe(
  X: Matrix,
  y: Vector,
  mode: FeMode,
  feature = 0,
): FeState {
  if (mode === "bin") {
    const col = X.data.map((r) => r[feature] ?? 0).filter(Number.isFinite).sort((a, b) => a - b);
    const bins = [25, 50, 75].map((p) => col[Math.min(col.length - 1, Math.floor((p / 100) * col.length))] ?? 0);
    return { mode, bins, targetMap: {} };
  }
  if (mode === "target") {
    const map: Record<string, number> = {};
    const sums: Record<string, number> = {};
    const cnt: Record<string, number> = {};
    for (let i = 0; i < X.nRows; i += 1) {
      const key = String(Math.round(X.data[i]?.[feature] ?? 0));
      sums[key] = (sums[key] ?? 0) + (y.data[i] ?? 0);
      cnt[key] = (cnt[key] ?? 0) + 1;
    }
    for (const k of Object.keys(sums)) {
      map[k] = (sums[k] ?? 0) / (cnt[k] ?? 1);
    }
    return { mode, bins: [], targetMap: map };
  }
  return { mode, bins: [], targetMap: {} };
}

export function applyFe(X: Matrix, state: FeState, feature = 0): Matrix {
  if (state.mode === "interact") {
    return matrix(
      X.data.map((row) => {
        const a = row[0] ?? 0;
        const b = row[1] ?? a;
        return [...row, a * b];
      }),
    );
  }
  if (state.mode === "bin") {
    return matrix(
      X.data.map((row) => {
        const v = row[feature] ?? 0;
        const bin = state.bins.filter((b) => v > b).length;
        return [...row, bin];
      }),
    );
  }
  if (state.mode === "target") {
    return matrix(
      X.data.map((row) => {
        const key = String(Math.round(row[feature] ?? 0));
        return [...row, state.targetMap[key] ?? 0];
      }),
    );
  }
  return X;
}

export function feLabel(mode: FeMode): string {
  if (mode === "interact") return "PolynomialFeatures(interaction_only=True)";
  if (mode === "bin") return "KBinsDiscretizer(n_bins=4, encode='ordinal')";
  return "TargetEncoder()";
}
