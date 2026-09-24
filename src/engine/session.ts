/**
 * Session state machine: the sandbox workspace a level or free play runs in.
 */

import { loadDataset } from "./datasets";
import {
  fitModel,
  modelSklearnName,
  defaultParams,
  type FittedModel,
  type ModelParams,
} from "./estimators";
import { computeMetrics, trainTestSplit, formatMetrics } from "./metrics";
import {
  applyImputer,
  fitImputer,
  fitPolyScaler,
  fitScaler,
  oneHotColumn,
  polyExpand,
  scalerLabel,
  transform,
  type PolyScaler,
  type Scaler,
} from "./transformers";
import type {
  Dataset,
  DatasetName,
  EncoderName,
  ImputerName,
  Metrics,
  ModelName,
  PipelineStep,
  ScalerName,
  SessionSnapshot,
  Split,
  Vector,
} from "./types";

export interface CommandResult {
  readonly lines: string[];
  readonly sklearn: string | null;
  readonly status: "ok" | "error" | "warn";
  readonly snapshot: SessionSnapshot;
}

/** Full internal state for reliable undo. */
interface SessionState {
  dataset: Dataset | null;
  split: Split | null;
  scaler: Scaler | null;
  scaleLeaked: boolean;
  scalePhase: "none" | "before_split" | "after_split";
  encoded: EncoderName | null;
  imputed: ImputerName | null;
  polyDegree: number | null;
  model: FittedModel | null;
  predictions: Vector | null;
  metrics: Metrics | null;
  trainMetrics: Metrics | null;
  cvScores: number[] | null;
  searchBest: Record<string, number | string> | null;
  scoredOn: "train" | "test" | null;
  steps: PipelineStep[];
  commandCount: number;
  inspectedData: boolean;
  inspectedResiduals: boolean;
  inspectedRoc: boolean;
}

export class Session {
  private dataset: Dataset | null = null;
  private split: Split | null = null;
  private scaler: Scaler | null = null;
  private scaleLeaked = false;
  private scalePhase: "none" | "before_split" | "after_split" = "none";
  private encoded: EncoderName | null = null;
  private imputed: ImputerName | null = null;
  private polyDegree: number | null = null;
  private model: FittedModel | null = null;
  private predictions: Vector | null = null;
  private metrics: Metrics | null = null;
  private trainMetrics: Metrics | null = null;
  private cvScores: number[] | null = null;
  private searchBest: Record<string, number | string> | null = null;
  private scoredOn: "train" | "test" | null = null;
  private steps: PipelineStep[] = [];
  private commandCount = 0;
  private history: SessionState[] = [];
  private inspectedData = false;
  private inspectedResiduals = false;
  private inspectedRoc = false;

  snapshot(): SessionSnapshot {
    return {
      dataset: this.dataset,
      split: this.split,
      scaled: this.scaler?.name ?? null,
      scaleLeaked: this.scaleLeaked,
      encoded: this.encoded,
      imputed: this.imputed,
      polyDegree: this.polyDegree,
      model: this.model?.name ?? null,
      modelParams: this.model?.params ?? {},
      fitted: this.model !== null,
      predictions: this.predictions,
      metrics: this.metrics,
      trainMetrics: this.trainMetrics,
      cvScores: this.cvScores ? [...this.cvScores] : null,
      searchBest: this.searchBest ? { ...this.searchBest } : null,
      steps: [...this.steps],
      commandCount: this.commandCount,
      scoredOn: this.scoredOn,
      inspectedData: this.inspectedData,
      inspectedResiduals: this.inspectedResiduals,
      inspectedRoc: this.inspectedRoc,
    };
  }

  /** Mark that the learner inspected X/y via `show data`. */
  markInspected(): void {
    this.inspectedData = true;
    this.commandCount += 1;
  }

  markResiduals(): void {
    this.inspectedResiduals = true;
    this.commandCount += 1;
  }

  markRoc(): void {
    this.inspectedRoc = true;
    this.commandCount += 1;
  }

  get inspected(): boolean {
    return this.inspectedData;
  }

  private imputerFill: number[] | null = null;
  private polyScaler: PolyScaler | null = null;

  /** Build the design matrix actually used for fit (post transforms). */
  private designMatrix(X: Split["XTrain"]): Split["XTrain"] {
    let out = X;
    if (this.imputed && this.imputerFill) {
      out = applyImputer(out, { fill: this.imputerFill });
    }
    if (this.encoded === "onehot" && this.dataset?.name === "mixed_table") {
      out = oneHotColumn(out, 1);
    }
    if (this.polyDegree) {
      if (!this.polyScaler) {
        this.polyScaler = fitPolyScaler(out, this.polyDegree);
      }
      out = polyExpand(out, this.polyScaler);
    }
    if (this.scaler) {
      out = transform(out, this.scaler);
    }
    return out;
  }

  private capture(): SessionState {
    return {
      dataset: this.dataset,
      split: this.split,
      scaler: this.scaler,
      scaleLeaked: this.scaleLeaked,
      scalePhase: this.scalePhase,
      encoded: this.encoded,
      imputed: this.imputed,
      polyDegree: this.polyDegree,
      model: this.model,
      predictions: this.predictions,
      metrics: this.metrics,
      trainMetrics: this.trainMetrics,
      cvScores: this.cvScores,
      searchBest: this.searchBest,
      scoredOn: this.scoredOn,
      steps: [...this.steps],
      commandCount: this.commandCount,
      inspectedData: this.inspectedData,
      inspectedResiduals: this.inspectedResiduals,
      inspectedRoc: this.inspectedRoc,
    };
  }

  private restore(state: SessionState): void {
    this.dataset = state.dataset;
    this.split = state.split;
    this.scaler = state.scaler;
    this.scaleLeaked = state.scaleLeaked;
    this.scalePhase = state.scalePhase;
    this.encoded = state.encoded;
    this.imputed = state.imputed;
    this.polyDegree = state.polyDegree;
    this.model = state.model;
    this.predictions = state.predictions;
    this.metrics = state.metrics;
    this.trainMetrics = state.trainMetrics;
    this.cvScores = state.cvScores;
    this.searchBest = state.searchBest;
    this.scoredOn = state.scoredOn;
    this.steps = [...state.steps];
    this.commandCount = state.commandCount;
    this.inspectedData = state.inspectedData;
    this.inspectedResiduals = state.inspectedResiduals;
    this.inspectedRoc = state.inspectedRoc;
  }

  private pushHistory(): void {
    this.history.push(this.capture());
    if (this.history.length > 40) {
      this.history.shift();
    }
  }

  private addStep(step: PipelineStep): void {
    this.steps = [...this.steps, step];
  }

  private requireDataset(): Dataset {
    if (!this.dataset) {
      throw new Error("No dataset loaded. Try `load blobs`.");
    }
    return this.dataset;
  }

  load(name: DatasetName): CommandResult {
    this.pushHistory();
    this.dataset = loadDataset(name);
    this.split = null;
    this.scaler = null;
    this.scaleLeaked = false;
    this.scalePhase = "none";
    this.model = null;
    this.predictions = null;
    this.metrics = null;
    this.trainMetrics = null;
    this.scoredOn = null;
    this.inspectedData = false;
    this.encoded = null;
    this.imputed = null;
    this.polyDegree = null;
    this.cvScores = null;
    this.searchBest = null;
    this.inspectedResiduals = false;
    this.inspectedRoc = false;
    this.steps = [
      {
        kind: "load",
        label: `load ${name}`,
        sklearn: `# built-in toy dataset '${name}'\nfrom sklearn.datasets import make_...`,
        status: "ok",
      },
    ];
    this.commandCount += 1;
    const ds = this.dataset;
    const lines = [
      `Loaded ${name} · ${ds.task} · n=${ds.X.nRows} · features=[${ds.featureNames.join(", ")}]`,
      `Target: ${ds.targetName}`,
    ];
    return {
      lines,
      sklearn: `X, y = load_${name}()  # conceptually — use the lesson dataset`,
      status: "ok",
      snapshot: this.snapshot(),
    };
  }

  split_(testSize = 0.2, seed = 42): CommandResult {
    this.pushHistory();
    const ds = this.requireDataset();
    if (this.scalePhase === "before_split") {
      this.scaleLeaked = true;
    }
    this.split = trainTestSplit(ds.X, ds.y, testSize, seed);
    this.model = null;
    this.predictions = null;
    this.metrics = null;
    this.trainMetrics = null;
    this.scoredOn = null;
    this.addStep({
      kind: "split",
      label: `split test=${testSize}`,
      sklearn: `train_test_split(X, y, test_size=${testSize}, random_state=${seed})`,
      status: this.scaleLeaked ? "error" : "ok",
      detail: this.scaleLeaked
        ? "Scaler was fit on all rows — test information leaked into preprocessing."
        : undefined,
    });
    this.commandCount += 1;
    const lines = [
      `train ${this.split.XTrain.nRows} rows · test ${this.split.XTest.nRows} rows · seed=${seed}`,
    ];
    if (this.scaleLeaked) {
      lines.push(
        "WARNING: preprocessing was fit before the split. Metrics will be optimistic (leakage).",
      );
    }
    return {
      lines,
      sklearn: `X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=${testSize}, random_state=${seed})`,
      status: this.scaleLeaked ? "warn" : "ok",
      snapshot: this.snapshot(),
    };
  }

  scale(name: ScalerName = "standard"): CommandResult {
    this.pushHistory();
    const ds = this.requireDataset();
    this.scaler = fitScaler(ds.X, name);
    this.scalePhase = "before_split";
    this.scaleLeaked = Boolean(this.split);
    if (this.split) {
      // Correct order: fit on train only.
      this.scaler = fitScaler(this.split.XTrain, name);
      this.scalePhase = "after_split";
      this.scaleLeaked = false;
      this.addStep({
        kind: "scale",
        label: scalerLabel(name),
        sklearn: `scaler = ${scalerLabel(name)}()\nX_train = scaler.fit_transform(X_train)`,
        status: "ok",
        detail: "Scaler fit on train only — no leakage.",
      });
    } else {
      this.scaleLeaked = true;
      this.addStep({
        kind: "scale",
        label: scalerLabel(name),
        sklearn: `scaler = ${scalerLabel(name)}()\nX = scaler.fit_transform(X)  # too early`,
        status: "error",
        detail: "Fitting the scaler on all rows before split leaks test statistics.",
      });
    }
    this.model = null;
    this.predictions = null;
    this.metrics = null;
    this.trainMetrics = null;
    this.scoredOn = null;
    this.commandCount += 1;
    const lines = [`Applied ${scalerLabel(name)}.`];
    if (this.scaleLeaked) {
      lines.push(
        "Leakage: scaler saw test rows. Split first, then `scale` (fit on train).",
      );
    }
    return {
      lines,
      sklearn: this.split
        ? `scaler = ${scalerLabel(name)}()\nX_train = scaler.fit_transform(X_train)\nX_test = scaler.transform(X_test)`
        : `scaler = ${scalerLabel(name)}()\nX = scaler.fit_transform(X)  # leaky`,
      status: this.scaleLeaked ? "warn" : "ok",
      snapshot: this.snapshot(),
    };
  }

  fit(name: ModelName, params: ModelParams = {}): CommandResult {
    this.pushHistory();
    this.requireDataset();
    if (!this.split) {
      throw new Error("Split first: `split test_size=0.2`");
    }
    const modelParams = { ...defaultParams(name), ...params };
    const XTrain = this.designMatrix(this.split.XTrain);
    const yTrain = this.split.yTrain;
    this.model = fitModel(name, XTrain, yTrain, modelParams);
    this.predictions = null;
    this.metrics = null;
    this.trainMetrics = null;
    this.scoredOn = null;
    this.addStep({
      kind: "fit",
      label: modelSklearnName(name),
      sklearn: `model = ${modelSklearnName(name)}(${formatPyParams(modelParams)})\nmodel.fit(X_train, y_train)`,
      status: "ok",
      detail: `Fit on ${XTrain.nRows} training rows.`,
    });
    this.commandCount += 1;
    return {
      lines: [`Fitted ${modelSklearnName(name)} on train (n=${XTrain.nRows}).`],
      sklearn: `model = ${modelSklearnName(name)}(${formatPyParams(modelParams)})\nmodel.fit(X_train, y_train)`,
      status: "ok",
      snapshot: this.snapshot(),
    };
  }

  predict(): CommandResult {
    this.pushHistory();
    this.requireDataset();
    if (!this.model || !this.split) {
      throw new Error("Fit a model on a split first.");
    }
    const XTest = this.designMatrix(this.split.XTest);
    this.predictions = this.model.predict(XTest);
    this.addStep({
      kind: "predict",
      label: "predict",
      sklearn: "y_pred = model.predict(X_test)",
      status: "ok",
    });
    this.commandCount += 1;
    return {
      lines: [`Predicted ${this.predictions.data.length} test rows.`],
      sklearn: "y_pred = model.predict(X_test)",
      status: "ok",
      snapshot: this.snapshot(),
    };
  }

  score(on: "train" | "test" = "test"): CommandResult {
    this.pushHistory();
    const ds = this.requireDataset();
    if (!this.model || !this.split) {
      throw new Error("Fit a model on a split first.");
    }
    const X = this.designMatrix(on === "test" ? this.split.XTest : this.split.XTrain);
    const y = on === "test" ? this.split.yTest : this.split.yTrain;
    const pred = this.model.predict(X);
    const metrics = computeMetrics(ds.task, y, pred);
    if (on === "test") {
      this.metrics = metrics;
      this.scoredOn = "test";
    } else {
      this.trainMetrics = metrics;
      this.scoredOn = "train";
    }
    this.predictions = on === "test" ? pred : this.predictions;
    this.addStep({
      kind: "score",
      label: `score(${on})`,
      sklearn: `model.score(${on === "test" ? "X_test" : "X_train"}, ${on === "test" ? "y_test" : "y_train"})`,
      status: "ok",
    });
    this.commandCount += 1;
    return {
      lines: formatMetrics(metrics, `score[${on}]`),
      sklearn: `model.score(${on === "test" ? "X_test" : "X_train"}, ${on === "test" ? "y_test" : "y_train"})`,
      status: "ok",
      snapshot: this.snapshot(),
    };
  }

  /** Print the last confusion matrix (classification only). */
  showConfusion(): CommandResult {
    this.pushHistory();
    const m = this.metrics ?? this.trainMetrics;
    if (!m || !("confusion" in m)) {
      throw new Error("No classification metrics yet. `score test` first.");
    }
    const [tnr, fpr] = m.confusion[0] ?? [0, 0];
    const [fnr, tpr] = m.confusion[1] ?? [0, 0];
    this.addStep({
      kind: "score",
      label: "confusion matrix",
      sklearn: "from sklearn.metrics import confusion_matrix\nconfusion_matrix(y_test, y_pred)",
      status: "ok",
    });
    this.commandCount += 1;
    return {
      lines: [
        "confusion matrix [[TN, FP], [FN, TP]]",
        `  [[${tnr}, ${fpr}],`,
        `   [${fnr}, ${tpr}]]`,
        `  precision ${m.precision.toFixed(3)}  recall ${m.recall.toFixed(3)}  f1 ${m.f1.toFixed(3)}`,
      ],
      sklearn: "confusion_matrix(y_test, y_pred)",
      status: "ok",
      snapshot: this.snapshot(),
    };
  }

  /** One-hot encode categorical region column (mixed_table). */
  encode(name: EncoderName = "onehot"): CommandResult {
    this.pushHistory();
    this.requireDataset();
    this.encoded = name;
    this.model = null;
    this.addStep({
      kind: "encode",
      label: name === "onehot" ? "OneHotEncoder" : "OrdinalEncoder",
      sklearn: `${name === "onehot" ? "OneHotEncoder" : "OrdinalEncoder"}(handle_unknown='ignore')`,
      status: "ok",
    });
    this.commandCount += 1;
    return {
      lines: [`Applied ${name} encoding to categorical columns.`],
      sklearn: `preprocessor = ColumnTransformer([('cat', OneHotEncoder(), ['region'])])`,
      status: "ok",
      snapshot: this.snapshot(),
    };
  }

  /** Impute missing numeric values using train statistics only. */
  impute(name: ImputerName = "mean"): CommandResult {
    this.pushHistory();
    this.requireDataset();
    if (!this.split) {
      // Still allowed, but flag as leaky statistics if later used wrong.
    }
    const source = this.split?.XTrain ?? this.requireDataset().X;
    const imp = fitImputer(source, name);
    this.imputerFill = imp.fill;
    this.imputed = name;
    this.model = null;
    this.addStep({
      kind: "impute",
      label: `SimpleImputer(${name})`,
      sklearn: `SimpleImputer(strategy='${name}').fit(X_train)`,
      status: this.split ? "ok" : "warn",
      detail: this.split ? "Filled from train stats." : "Fit before split — stats may leak.",
    });
    this.commandCount += 1;
    return {
      lines: [`Imputed missing values with strategy='${name}'.`],
      sklearn: `X_train = SimpleImputer(strategy='${name}').fit_transform(X_train)`,
      status: "ok",
      snapshot: this.snapshot(),
    };
  }

  /** Expand polynomial degree for regression. */
  poly(degree = 2): CommandResult {
    this.pushHistory();
    this.requireDataset();
    this.polyDegree = degree;
    this.polyScaler = null;
    this.model = null;
    this.addStep({
      kind: "poly",
      label: `PolynomialFeatures(${degree})`,
      sklearn: `PolynomialFeatures(degree=${degree}, include_bias=False)`,
      status: "ok",
    });
    this.commandCount += 1;
    return {
      lines: [`Expanded features with degree=${degree}.`],
      sklearn: `X_poly = PolynomialFeatures(degree=${degree}).fit_transform(X)`,
      status: "ok",
      snapshot: this.snapshot(),
    };
  }

  /** K-fold CV scores on train only. */
  cv(folds = 5): CommandResult {
    this.pushHistory();
    const ds = this.requireDataset();
    if (!this.split || !this.model) {
      throw new Error("Fit a model on a split first, then `cv`.");
    }
    const X = this.designMatrix(this.split.XTrain);
    const y = this.split.yTrain;
    const k = Math.max(2, folds);
    const n = X.nRows;
    const foldSize = Math.floor(n / k);
    const scores: number[] = [];
    for (let f = 0; f < k; f += 1) {
      const idx = Array.from({ length: n }, (_, i) => i);
      const testIdx = idx.slice(f * foldSize, f === k - 1 ? n : (f + 1) * foldSize);
      const trainIdx = idx.filter((i) => !testIdx.includes(i));
      const Xtr = { nRows: trainIdx.length, nCols: X.nCols, data: trainIdx.map((i) => [...(X.data[i] ?? [])]) };
      const ytr = { data: trainIdx.map((i) => y.data[i] ?? 0) };
      const Xte = { nRows: testIdx.length, nCols: X.nCols, data: testIdx.map((i) => [...(X.data[i] ?? [])]) };
      const yte = { data: testIdx.map((i) => y.data[i] ?? 0) };
      const m = fitModel(this.model!.name, Xtr, ytr, this.model!.params);
      const pred = m.predict(Xte);
      const met = computeMetrics(ds.task, yte, pred);
      scores.push("accuracy" in met ? met.accuracy : met.r2);
    }
    this.cvScores = scores;
    this.addStep({
      kind: "cv",
      label: `cross_val_score(k=${k})`,
      sklearn: `cross_val_score(model, X_train, y_train, cv=${k})`,
      status: "ok",
    });
    this.commandCount += 1;
    const mean = scores.reduce((a, b) => a + b, 0) / (scores.length || 1);
    return {
      lines: [`cv(${k}) scores: ${scores.map((s) => s.toFixed(3)).join(" ")}  mean=${mean.toFixed(3)}`],
      sklearn: `cross_val_score(model, X_train, y_train, cv=${k})`,
      status: "ok",
      snapshot: this.snapshot(),
    };
  }

  /** Grid-style search over a small param grid using train CV. */
  search(model: ModelName, grid: Record<string, (number | string)[]>): CommandResult {
    this.pushHistory();
    const ds = this.requireDataset();
    if (!this.split) {
      throw new Error("Split first. Search must not see test.");
    }
    const keys = Object.keys(grid);
    if (keys.length === 0) {
      throw new Error("search needs a param grid");
    }
    let best: { params: Record<string, number | string>; score: number } | null = null;
    const combos: Record<string, number | string>[] = [{}];
    for (const key of keys) {
      const next: Record<string, number | string>[] = [];
      for (const base of combos) {
        for (const v of grid[key] ?? []) {
          next.push({ ...base, [key]: v });
        }
      }
      combos.length = 0;
      combos.push(...next);
    }
    for (const params of combos) {
      const X = this.designMatrix(this.split.XTrain);
      const y = this.split.yTrain;
      const n = X.nRows;
      const foldSize = Math.max(1, Math.floor(n / 3));
      const scores: number[] = [];
      for (let f = 0; f < 3; f += 1) {
        const idx = Array.from({ length: n }, (_, i) => i);
        const testIdx = idx.slice(f * foldSize, f === 2 ? n : (f + 1) * foldSize);
        const trainIdx = idx.filter((i) => !testIdx.includes(i));
        const Xtr = { nRows: trainIdx.length, nCols: X.nCols, data: trainIdx.map((i) => [...(X.data[i] ?? [])]) };
        const ytr = { data: trainIdx.map((i) => y.data[i] ?? 0) };
        const Xte = { nRows: testIdx.length, nCols: X.nCols, data: testIdx.map((i) => [...(X.data[i] ?? [])]) };
        const yte = { data: testIdx.map((i) => y.data[i] ?? 0) };
        try {
          const m = fitModel(model, Xtr, ytr, params);
          const met = computeMetrics(ds.task, yte, m.predict(Xte));
          scores.push("accuracy" in met ? met.accuracy : met.r2);
        } catch {
          scores.push(0);
        }
      }
      const mean = scores.reduce((a, b) => a + b, 0) / (scores.length || 1);
      if (!best || mean > best.score) {
        best = { params, score: mean };
      }
    }
    this.searchBest = best?.params ?? {};
    this.addStep({
      kind: "search",
      label: `GridSearchCV(${model})`,
      sklearn: `GridSearchCV(${modelSklearnName(model)}(), param_grid=${JSON.stringify(grid)}, cv=3)`,
      status: "ok",
      detail: `best=${JSON.stringify(best?.params ?? {})} cv=${(best?.score ?? 0).toFixed(3)}`,
    });
    this.commandCount += 1;
    return {
      lines: [
        `searched ${combos.length} candidates on train CV only`,
        `best params ${JSON.stringify(best?.params ?? {})}  score=${(best?.score ?? 0).toFixed(3)}`,
        "Fit a fresh model with those params and `score test` once.",
      ],
      sklearn: `GridSearchCV(${modelSklearnName(model)}(), param_grid=...).fit(X_train, y_train)`,
      status: "ok",
      snapshot: this.snapshot(),
    };
  }

  /** Residual summary for regression diagnostics. */
  residuals(): CommandResult {
    this.pushHistory();
    const ds = this.requireDataset();
    if (!this.model || !this.split || ds.task !== "regression") {
      throw new Error("Need a fitted regression model.");
    }
    const X = this.designMatrix(this.split.XTrain);
    const pred = this.model.predict(X);
    const res = this.split.yTrain.data.map((y, i) => y - (pred.data[i] ?? 0));
    const meanR = res.reduce((a, b) => a + b, 0) / (res.length || 1);
    const sd = Math.sqrt(res.reduce((a, b) => a + (b - meanR) ** 2, 0) / (res.length || 1));
    this.markResiduals();
    this.addStep({
      kind: "score",
      label: "residuals",
      sklearn: "resid = y_train - model.predict(X_train)",
      status: "ok",
    });
    return {
      lines: [
        `residuals mean=${meanR.toFixed(3)}  sd=${sd.toFixed(3)}  n=${res.length}`,
        "Healthy: mean near 0, no funnel pattern. Systematic bias ⇒ wrong model class.",
      ],
      sklearn: "resid = y_train - model.predict(X_train)",
      status: "ok",
      snapshot: this.snapshot(),
    };
  }

  /** ROC / threshold story for binary classifiers. */
  roc(): CommandResult {
    this.pushHistory();
    const ds = this.requireDataset();
    if (!this.model || !this.split || ds.task !== "classification") {
      throw new Error("Need a fitted classifier.");
    }
    const X = this.designMatrix(this.split.XTest);
    const y = this.split.yTest;
    const score = this.model.decision(X);
    let tp = 0;
    let fp = 0;
    const nPos = y.data.filter((v) => (v >= 0.5 ? 1 : 0)).length || 1;
    const pairs = score.data.map((s, i) => ({ s, y: (y.data[i] ?? 0) >= 0.5 ? 1 : 0 }));
    pairs.sort((a, b) => b.s - a.s);
    const pts: string[] = [];
    for (let i = 0; i <= 10; i += 1) {
      const thr = 1 - i / 10;
      tp = 0;
      fp = 0;
      for (const p of pairs) {
        const pred = p.s >= thr ? 1 : 0;
        if (pred === 1 && p.y === 1) tp += 1;
        if (pred === 1 && p.y === 0) fp += 1;
      }
      pts.push(`  tpr=${(tp / nPos).toFixed(2)} fpr=${(fp / pairs.length || 1).toFixed(2)} thr=${thr.toFixed(1)}`);
    }
    this.markRoc();
    this.addStep({
      kind: "score",
      label: "roc",
      sklearn: "roc_curve(y_test, model.decision_function(X_test))",
      status: "ok",
    });
    return {
      lines: ["threshold sweep (test):", ...pts, "Move threshold before shipping — accuracy is one point on this tradeoff."],
      sklearn: "fpr, tpr, thr = roc_curve(y_test, scores)",
      status: "ok",
      snapshot: this.snapshot(),
    };
  }

  reset(): CommandResult {
    this.pushHistory();
    this.dataset = null;
    this.split = null;
    this.scaler = null;
    this.scaleLeaked = false;
    this.scalePhase = "none";
    this.model = null;
    this.predictions = null;
    this.metrics = null;
    this.trainMetrics = null;
    this.scoredOn = null;
    this.steps = [];
    this.commandCount = 0;
    this.inspectedData = false;
    this.encoded = null;
    this.imputed = null;
    this.imputerFill = null;
    this.polyDegree = null;
    this.polyScaler = null;
    this.cvScores = null;
    this.searchBest = null;
    this.inspectedResiduals = false;
    this.inspectedRoc = false;
    return {
      lines: ["Session cleared."],
      sklearn: null,
      status: "ok",
      snapshot: this.snapshot(),
    };
  }

  undo(): CommandResult {
    const prev = this.history.pop();
    if (!prev) {
      return {
        lines: ["Nothing to undo."],
        sklearn: null,
        status: "warn",
        snapshot: this.snapshot(),
      };
    }
    this.restore(prev);
    return {
      lines: ["Undo."],
      sklearn: null,
      status: "ok",
      snapshot: this.snapshot(),
    };
  }
}

function formatPyParams(params: ModelParams): string {
  const parts = Object.entries(params).map(([k, v]) => {
    if (typeof v === "string") {
      return `${k}='${v}'`;
    }
    return `${k}=${v}`;
  });
  return parts.join(", ");
}
