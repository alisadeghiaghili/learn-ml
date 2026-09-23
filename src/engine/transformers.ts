/**
 * Feature transformers in the scikit-learn sense: fit on train, transform anywhere.
 */

import { colMeans, colMinMax, colStds, scaleShift } from "./matrix";
import type { Matrix, ScalerName, Vector } from "./types";

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
