/**
 * Canvas charts: scatter, decision surface, regression fit, residual strip.
 */

import type { Dataset, Matrix, Split, TaskKind, Vector } from "../engine/types";
import type { FittedModel } from "../engine/estimators";

export type ChartKind = "empty" | "data" | "fit" | "residual";

const COLORS = {
  class0: "#5b8def",
  class1: "#f0b429",
  class2: "#9b7ede",
  signal: "#f0b429",
  alarm: "#e23d51",
  calm: "#2a9d8f",
  chalk: "#c5ced6",
  chalkDim: "#7d8b96",
  void: "#0a1018",
  grid: "#1a2c3a",
};

export interface DrawState {
  dataset: Dataset | null;
  split: Split | null;
  model: FittedModel | null;
  /** Optional transformed features for plotting (post-scaler), same rows as dataset. */
  scaledX: Matrix | null;
  kind: ChartKind;
  /** 0..1 animation progress for fit bloom. */
  fitProgress: number;
}

interface Bounds {
  xMin: number;
  xMax: number;
  yMin: number;
  yMax: number;
}

function computeBounds(X: Matrix, y?: Vector): Bounds {
  let xMin = Infinity;
  let xMax = -Infinity;
  let yMin = Infinity;
  let yMax = -Infinity;
  if (X.nCols === 1) {
    for (const row of X.data) {
      xMin = Math.min(xMin, row[0] ?? 0);
      xMax = Math.max(xMax, row[0] ?? 0);
    }
    const ys = y?.data ?? X.data.map((r) => r[0] ?? 0);
    for (const v of ys) {
      yMin = Math.min(yMin, v);
      yMax = Math.max(yMax, v);
    }
  } else {
    for (const row of X.data) {
      xMin = Math.min(xMin, row[0] ?? 0);
      xMax = Math.max(xMax, row[0] ?? 0);
      yMin = Math.min(yMin, row[1] ?? 0);
      yMax = Math.max(yMax, row[1] ?? 0);
    }
  }
  const padX = (xMax - xMin) * 0.12 || 1;
  const padY = (yMax - yMin) * 0.12 || 1;
  return {
    xMin: xMin - padX,
    xMax: xMax + padX,
    yMin: yMin - padY,
    yMax: yMax + padY,
  };
}

function project(
  x: number,
  y: number,
  b: Bounds,
  w: number,
  h: number,
  m: number,
): [number, number] {
  const px = m + ((x - b.xMin) / (b.xMax - b.xMin)) * (w - 2 * m);
  const py = h - m - ((y - b.yMin) / (b.yMax - b.yMin)) * (h - 2 * m);
  return [px, py];
}

/**
 * Draw the active stage chart.
 *
 * Args:
 *   ctx: Canvas 2D context (already sized to CSS pixels).
 *   width: CSS width.
 *   height: CSS height.
 *   state: What to draw.
 */
export function drawStage(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  state: DrawState,
): void {
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = COLORS.void;
  ctx.fillRect(0, 0, width, height);

  const ds = state.dataset;
  if (!ds) {
    drawEmpty(ctx, width, height);
    return;
  }

  const isClass = ds.task === "classification";
  const plotX = state.scaledX ?? ds.X;

  if (ds.task === "regression" && plotX.nCols === 1) {
    drawRegression(ctx, width, height, plotX, ds.y, state);
    return;
  }

  drawClassification(ctx, width, height, plotX, ds, state, isClass);
}

function drawEmpty(ctx: CanvasRenderingContext2D, w: number, h: number): void {
  ctx.fillStyle = COLORS.chalkDim;
  ctx.font = "13px 'IBM Plex Mono', monospace";
  ctx.textAlign = "center";
  ctx.fillText("awaiting X, y — try `load blobs`", w / 2, h / 2);
}

function drawGrid(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  m: number,
): void {
  ctx.strokeStyle = COLORS.grid;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(m, m);
  ctx.lineTo(m, h - m);
  ctx.lineTo(w - m, h - m);
  ctx.stroke();
}

function drawClassification(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  X: Matrix,
  ds: Dataset,
  state: DrawState,
  _isClass: boolean,
): void {
  const m = 28;
  const b = computeBounds(X);
  drawGrid(ctx, w, h, m);

  // Decision surface from model.decision on a coarse grid.
  if (state.model && state.model.task === "classification") {
    const steps = 36;
    const cellW = (w - 2 * m) / steps;
    const cellH = (h - 2 * m) / steps;
    for (let i = 0; i < steps; i += 1) {
      for (let j = 0; j < steps; j += 1) {
        const x = b.xMin + ((i + 0.5) / steps) * (b.xMax - b.xMin);
        const y = b.yMin + ((j + 0.5) / steps) * (b.yMax - b.yMin);
        // Input to the model is the 2-D feature row as plotted.
        const row = [[x, y]];
        const s = state.model.decision({ nRows: 1, nCols: 2, data: row }).data[0] ?? 0;
        // decision may be logit or vote fraction; normalize softly
        const t = 1 / (1 + Math.exp(-s * 3));
        const alpha = (0.08 + 0.22 * Math.abs(t - 0.5) * 2) * state.fitProgress;
        const [px, py] = project(x, y, b, w, h, m);
        // y increases upward in data space; canvas rows go down
        ctx.fillStyle =
          t >= 0.5
            ? `rgba(240, 180, 41, ${alpha})`
            : `rgba(91, 141, 239, ${alpha})`;
        ctx.fillRect(px - cellW / 2, py - cellH / 2 - cellH, cellW + 1, cellH + 1);
      }
    }
  }

  // Points
  const trainIdx = new Set<number>();
  const testIdx = new Set<number>();
  if (state.split) {
    // Reconstruct membership via matching rows is fragile; draw all points,
    // then overlay split markers using split matrices directly.
  }

  for (let i = 0; i < X.nRows; i += 1) {
    const x = X.data[i]?.[0] ?? 0;
    const y = X.data[i]?.[1] ?? 0;
    const label = ds.y.data[i] ?? 0;
    const [px, py] = project(x, y, b, w, h, m);
    ctx.beginPath();
    ctx.arc(px, py, 4, 0, Math.PI * 2);
    ctx.fillStyle = label >= 0.5 ? COLORS.class1 : COLORS.class0;
    ctx.fill();
    ctx.strokeStyle = COLORS.void;
    ctx.lineWidth = 1;
    ctx.stroke();
  }

  // Overlay test points with a ring (from split).
  if (state.split) {
    for (const row of state.split.XTest.data) {
      const [px, py] = project(row[0] ?? 0, row[1] ?? 0, b, w, h, m);
      ctx.beginPath();
      ctx.arc(px, py, 6.5, 0, Math.PI * 2);
      ctx.strokeStyle = COLORS.chalk;
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }
  }

  drawLegend(ctx, w, h, ds, state);
  void trainIdx;
  void testIdx;
}

function drawRegression(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  X: Matrix,
  y: Vector,
  state: DrawState,
): void {
  const m = 28;
  const b = computeBounds(X, y);
  drawGrid(ctx, w, h, m);

  // Points
  for (let i = 0; i < X.nRows; i += 1) {
    const x = X.data[i]?.[0] ?? 0;
    const yv = y.data[i] ?? 0;
    const [px, py] = project(x, yv, b, w, h, m);
    const isTest = isRowInSplitTest(state.split, x);
    ctx.beginPath();
    ctx.arc(px, py, isTest ? 5.5 : 4, 0, Math.PI * 2);
    ctx.fillStyle = isTest ? COLORS.chalk : COLORS.class0;
    ctx.fill();
    if (isTest) {
      ctx.strokeStyle = COLORS.void;
      ctx.lineWidth = 1;
      ctx.stroke();
    }
  }

  // Fit line
  if (state.model && state.model.task === "regression") {
    const x0 = b.xMin;
    const x1 = b.xMax;
    const y0 = state.model.decision({ nRows: 1, nCols: 1, data: [[x0]] }).data[0] ?? 0;
    const y1 = state.model.decision({ nRows: 1, nCols: 1, data: [[x1]] }).data[0] ?? 0;
    // Animate draw
    const prog = state.fitProgress;
    const xe = x0 + (x1 - x0) * prog;
    const ye = y0 + (y1 - y0) * prog;
    const [px0, py0] = project(x0, y0, b, w, h, m);
    const [px1, py1] = project(xe, ye, b, w, h, m);
    ctx.beginPath();
    ctx.moveTo(px0, py0);
    ctx.lineTo(px1, py1);
    ctx.strokeStyle = COLORS.signal;
    ctx.lineWidth = 2.5;
    ctx.stroke();
  }

  ctx.fillStyle = COLORS.chalkDim;
  ctx.font = "11px 'IBM Plex Mono', monospace";
  ctx.textAlign = "left";
  ctx.fillText("x", w - m + 6, h - m + 4);
  ctx.fillText("y", m - 18, m - 8);
}

function isRowInSplitTest(split: Split | null, x: number): boolean {
  if (!split) {
    return false;
  }
  return split.XTest.data.some((r) => Math.abs((r[0] ?? 0) - x) < 1e-6);
}

function drawLegend(
  ctx: CanvasRenderingContext2D,
  _w: number,
  h: number,
  ds: Dataset,
  state: DrawState,
): void {
  const names = ds.classNames ?? ["0", "1"];
  ctx.font = "11px 'IBM Plex Mono', monospace";
  ctx.textAlign = "left";
  const items: [string, string][] = [
    [names[0] ?? "0", COLORS.class0],
    [names[1] ?? "1", COLORS.class1],
  ];
  let x = 12;
  const y = h - 12;
  for (const [label, color] of items) {
    ctx.beginPath();
    ctx.arc(x + 4, y - 4, 4, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
    ctx.fillStyle = COLORS.chalk;
    ctx.fillText(label, x + 12, y);
    x += 48;
  }
  if (state.split) {
    ctx.strokeStyle = COLORS.chalk;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(x + 4, y - 4, 6, 0, Math.PI * 2);
    ctx.stroke();
    ctx.fillStyle = COLORS.chalkDim;
    ctx.fillText("test", x + 14, y);
  }
}

/**
 * Classification / regression plot type helper for UI copy.
 */
export function chartKindFor(task: TaskKind, fitted: boolean): ChartKind {
  if (fitted) {
    return "fit";
  }
  return task === "classification" ? "data" : "data";
}

export const CHART_COLORS = COLORS;
