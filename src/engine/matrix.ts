/**
 * Numeric matrix helpers shared across the engine.
 *
 * Args, Returns, Raises, Examples follow project documentation standards.
 */

import type { Matrix, Vector } from "./types";

/**
 * Create a matrix from row-major data.
 *
 * Args:
 *   rows: Matrix rows as number arrays.
 *
 * Returns:
 *   A Matrix with consistent column count.
 *
 * Raises:
 *   Error: If rows are empty or ragged.
 *
 * Examples:
 *   >>> matrix([[1, 2], [3, 4]]).nRows
 *   2
 */
export function matrix(rows: number[][]): Matrix {
  if (rows.length === 0) {
    throw new Error("matrix() requires at least one row");
  }
  const nCols = rows[0]?.length ?? 0;
  if (nCols === 0) {
    throw new Error("matrix() requires at least one column");
  }
  for (const row of rows) {
    if (row.length !== nCols) {
      throw new Error("matrix() requires a dense, non-ragged grid");
    }
  }
  return { nRows: rows.length, nCols, data: rows.map((r) => [...r]) };
}

/**
 * Create a vector from values.
 *
 * Args:
 *   values: 1-D values.
 *
 * Returns:
 *   A Vector copy of the input.
 */
export function vector(values: number[]): Vector {
  return { data: [...values] };
}

/**
 * Build an identity-like zero matrix.
 *
 * Args:
 *   nRows: Row count.
 *   nCols: Column count.
 *
 * Returns:
 *   Matrix of zeros.
 */
export function zeros(nRows: number, nCols: number): Matrix {
  return matrix(Array.from({ length: nRows }, () => Array(nCols).fill(0)));
}

/**
 * Select a subset of matrix rows by index.
 *
 * Args:
 *   X: Source matrix.
 *   indices: Row indices to keep.
 *
 * Returns:
 *   New matrix with selected rows in the given order.
 *
 * Raises:
 *   Error: If any index is out of range.
 */
export function takeRows(X: Matrix, indices: readonly number[]): Matrix {
  const rows = indices.map((i) => {
    const row = X.data[i];
    if (!row) {
      throw new Error(`takeRows: index ${i} out of range`);
    }
    return [...row];
  });
  return matrix(rows);
}

/**
 * Select a subset of vector entries by index.
 *
 * Args:
 *   y: Source vector.
 *   indices: Entry indices to keep.
 *
 * Returns:
 *   New vector with selected entries in the given order.
 */
export function take(y: Vector, indices: readonly number[]): Vector {
  return vector(indices.map((i) => y.data[i] ?? 0));
}

/**
 * Column means of a matrix.
 *
 * Args:
 *   X: Input matrix.
 *
 * Returns:
 *   Vector of length nCols.
 */
export function colMeans(X: Matrix): Vector {
  const sums = Array<number>(X.nCols).fill(0);
  for (const row of X.data) {
    for (let j = 0; j < X.nCols; j += 1) {
      sums[j] = (sums[j] ?? 0) + (row[j] ?? 0);
    }
  }
  return vector(sums.map((s) => s / X.nRows));
}

/**
 * Column population standard deviations (ddof=0).
 *
 * Args:
 *   X: Input matrix.
 *   means: Optional precomputed column means.
 *
 * Returns:
 *   Vector of length nCols. Zero-variance columns become 1 to avoid divide-by-zero.
 */
export function colStds(X: Matrix, means?: Vector): Vector {
  const mu = means ?? colMeans(X);
  const vars = Array<number>(X.nCols).fill(0);
  for (const row of X.data) {
    for (let j = 0; j < X.nCols; j += 1) {
      const d = (row[j] ?? 0) - (mu.data[j] ?? 0);
      vars[j] = (vars[j] ?? 0) + d * d;
    }
  }
  return vector(
    vars.map((v) => {
      const s = Math.sqrt(v / X.nRows);
      return s === 0 ? 1 : s;
    }),
  );
}

/**
 * Column min and max.
 *
 * Args:
 *   X: Input matrix.
 *
 * Returns:
 *   Tuple [mins, maxs] as vectors.
 */
export function colMinMax(X: Matrix): { mins: Vector; maxs: Vector } {
  const mins = Array<number>(X.nCols).fill(Number.POSITIVE_INFINITY);
  const maxs = Array<number>(X.nCols).fill(Number.NEGATIVE_INFINITY);
  for (const row of X.data) {
    for (let j = 0; j < X.nCols; j += 1) {
      const v = row[j] ?? 0;
      mins[j] = Math.min(mins[j] ?? v, v);
      maxs[j] = Math.max(maxs[j] ?? v, v);
    }
  }
  return {
    mins: vector(mins),
    maxs: vector(maxs.map((m, j) => (m === (mins[j] ?? 0) ? (mins[j] ?? 0) + 1 : m))),
  };
}

/**
 * Elementwise linear map with per-column scale/shift.
 *
 * Args:
 *   X: Input matrix.
 *   scale: Multipliers.
 *   shift: Additive terms applied after scale (X * scale + shift).
 *
 * Returns:
 *   Transformed matrix.
 */
export function scaleShift(X: Matrix, scale: Vector, shift: Vector): Matrix {
  return matrix(
    X.data.map((row) => row.map((v, j) => v * (scale.data[j] ?? 1) + (shift.data[j] ?? 0))),
  );
}

/**
 * Mean of a vector.
 *
 * Args:
 *   y: Input vector.
 *
 * Returns:
 *   Arithmetic mean, or 0 when empty.
 */
export function mean(y: Vector): number {
  if (y.data.length === 0) {
    return 0;
  }
  return y.data.reduce((a, b) => a + b, 0) / y.data.length;
}

/**
 * Sum of squared residuals against a constant mean baseline.
 *
 * Args:
 *   y: Targets.
 *
 * Returns:
 *   Total sum of squares.
 */
export function totalSumOfSquares(y: Vector): number {
  const mu = mean(y);
  return y.data.reduce((acc, v) => acc + (v - mu) ** 2, 0);
}
