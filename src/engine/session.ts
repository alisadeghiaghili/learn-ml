/**
 * Session state machine: the sandbox workspace a level or free play runs in.
 */

import { loadDataset, makeRng } from "./datasets";
import {
  fitModel,
  modelSklearnName,
  defaultParams,
  type FittedModel,
  type ModelParams,
} from "./estimators";
import { mean, matrix, take, takeRows, vector } from "./matrix";
import { computeMetrics, trainTestSplit, formatMetrics } from "./metrics";
import {
  applyFe,
  applyImputer,
  fitFe,
  fitImputer,
  fitPolyScaler,
  fitScaler,
  oneHotColumn,
  polyExpand,
  scalerLabel,
  transform,
  type FeMode,
  type FeState,
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
  feMode: FeMode | null;
  feState: FeState | null;
  coefReport: { name: string; coef: number; se: number; z: number }[] | null;
  importance: { feature: string; score: number }[] | null;
  silhouette: number | null;
  calibCurve: { p: number; rate: number; n: number }[] | null;
  predCi: { point: number; lo: number; hi: number }[] | null;
  nestedCvOuter: number[] | null;
  model: FittedModel | null;
  predictions: Vector | null;
  metrics: Metrics | null;
  trainMetrics: Metrics | null;
  cvScores: number[] | null;
  searchBest: Record<string, number | string> | null;
  learningCurve: { train: number; valid: number; n: number }[] | null;
  bootstrapCi: [number, number] | null;
  pipelineSaved: boolean;
  splitStrategy: "random" | "stratified" | "time" | null;
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
  private learningCurve: { train: number; valid: number; n: number }[] | null = null;
  private bootstrapCi: [number, number] | null = null;
  private pipelineSaved = false;
  private splitStrategy: "random" | "stratified" | "time" | null = null;
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
      feMode: this.feMode,
      coefReport: this.coefReport ? this.coefReport.map((r) => ({ ...r })) : null,
      importance: this.importance ? this.importance.map((r) => ({ ...r })) : null,
      silhouette: this.silhouette,
      calibCurve: this.calibCurve ? this.calibCurve.map((r) => ({ ...r })) : null,
      predCi: this.predCi ? this.predCi.map((r) => ({ ...r })) : null,
      nestedCvOuter: this.nestedCvOuter ? [...this.nestedCvOuter] : null,
      model: this.model?.name ?? null,
      modelParams: this.model?.params ?? {},
      fitted: this.model !== null,
      predictions: this.predictions,
      metrics: this.metrics,
      trainMetrics: this.trainMetrics,
      cvScores: this.cvScores ? [...this.cvScores] : null,
      searchBest: this.searchBest ? { ...this.searchBest } : null,
      learningCurve: this.learningCurve ? this.learningCurve.map((r) => ({ ...r })) : null,
      bootstrapCi: this.bootstrapCi ? ([...this.bootstrapCi] as [number, number]) : null,
      pipelineSaved: this.pipelineSaved,
      splitStrategy: this.splitStrategy,
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
  private feMode: FeMode | null = null;
  private feState: FeState | null = null;
  private coefReport: { name: string; coef: number; se: number; z: number }[] | null = null;
  private importance: { feature: string; score: number }[] | null = null;
  private silhouette: number | null = null;
  private calibCurve: { p: number; rate: number; n: number }[] | null = null;
  private predCi: { point: number; lo: number; hi: number }[] | null = null;
  private nestedCvOuter: number[] | null = null;

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
    if (this.feMode && this.feState) {
      out = applyFe(out, this.feState, 0);
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
      feMode: this.feMode,
      feState: this.feState,
      coefReport: this.coefReport,
      importance: this.importance,
      silhouette: this.silhouette,
      calibCurve: this.calibCurve,
      predCi: this.predCi,
      nestedCvOuter: this.nestedCvOuter,
      model: this.model,
      predictions: this.predictions,
      metrics: this.metrics,
      trainMetrics: this.trainMetrics,
      cvScores: this.cvScores,
      searchBest: this.searchBest,
      learningCurve: this.learningCurve,
      bootstrapCi: this.bootstrapCi,
      pipelineSaved: this.pipelineSaved,
      splitStrategy: this.splitStrategy,
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
    this.feMode = state.feMode;
    this.feState = state.feState;
    this.coefReport = state.coefReport;
    this.importance = state.importance;
    this.silhouette = state.silhouette;
    this.calibCurve = state.calibCurve;
    this.predCi = state.predCi;
    this.nestedCvOuter = state.nestedCvOuter;
    this.model = state.model;
    this.predictions = state.predictions;
    this.metrics = state.metrics;
    this.trainMetrics = state.trainMetrics;
    this.cvScores = state.cvScores;
    this.searchBest = state.searchBest;
    this.learningCurve = state.learningCurve;
    this.bootstrapCi = state.bootstrapCi;
    this.pipelineSaved = state.pipelineSaved;
    this.splitStrategy = state.splitStrategy;
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
    this.learningCurve = null;
    this.bootstrapCi = null;
    this.pipelineSaved = false;
    this.splitStrategy = null;
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

  split_(testSize = 0.2, seed = 42, strategy: "random" | "stratified" | "time" = "random"): CommandResult {
    this.pushHistory();
    const ds = this.requireDataset();
    if (this.scalePhase === "before_split") {
      this.scaleLeaked = true;
    }
    let split = trainTestSplit(ds.X, ds.y, testSize, seed);
    if (strategy === "time") {
      // Contiguous hold-out at the end — simulates temporal generalization.
      const n = ds.X.nRows;
      const nTest = Math.max(1, Math.round(n * testSize));
      const trainIdx = Array.from({ length: n - nTest }, (_, i) => i);
      const testIdx = Array.from({ length: nTest }, (_, i) => n - nTest + i);
      split = {
        XTrain: takeRows(ds.X, trainIdx),
        XTest: takeRows(ds.X, testIdx),
        yTrain: take(ds.y, trainIdx),
        yTest: take(ds.y, testIdx),
        testSize,
        seed,
      };
    } else if (strategy === "stratified") {
      // Approximate stratify: keep class ratio by splitting each class.
      const idx0: number[] = [];
      const idx1: number[] = [];
      ds.y.data.forEach((v, i) => {
        if ((v ?? 0) >= 0.5) idx1.push(i);
        else idx0.push(i);
      });
      const rng = makeRng(seed);
      const shuffle = (a: number[]) => {
        for (let i = a.length - 1; i > 0; i -= 1) {
          const j = Math.floor(rng() * (i + 1));
          const t = a[i] ?? 0;
          a[i] = a[j] ?? 0;
          a[j] = t;
        }
        return a;
      };
      shuffle(idx0);
      shuffle(idx1);
      const nTest0 = Math.max(1, Math.round(idx0.length * testSize));
      const nTest1 = Math.max(1, Math.round(idx1.length * testSize));
      const testIdx = [...idx0.slice(0, nTest0), ...idx1.slice(0, nTest1)];
      const trainIdx = [...idx0.slice(nTest0), ...idx1.slice(nTest1)];
      split = {
        XTrain: takeRows(ds.X, trainIdx),
        XTest: takeRows(ds.X, testIdx),
        yTrain: take(ds.y, trainIdx),
        yTest: take(ds.y, testIdx),
        testSize,
        seed,
      };
    }
    this.split = split;
    this.splitStrategy = strategy;
    this.model = null;
    this.predictions = null;
    this.metrics = null;
    this.trainMetrics = null;
    this.scoredOn = null;
    this.addStep({
      kind: "split",
      label: `split test=${testSize} ${strategy}`,
      sklearn:
        strategy === "time"
          ? `# temporal hold-out: train = rows[: -n_test]`
          : strategy === "stratified"
            ? `train_test_split(X, y, test_size=${testSize}, stratify=y)`
            : `train_test_split(X, y, test_size=${testSize}, random_state=${seed})`,
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
    let proba: Vector | undefined;
    let metrics: Metrics;
    if (ds.task === "classification") {
      const dec = this.model.decision(X);
      proba = vector(dec.data.map((z) => (z > 0 && z <= 1 ? z : 1 / (1 + Math.exp(-z)))));
      metrics = computeMetrics(ds.task, y, pred, proba, dec);
    } else {
      metrics = computeMetrics(ds.task, y, pred);
    }
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

  /** Learning curve: train vs validation score as n grows. */
  curve(fractions = 5): CommandResult {
    this.pushHistory();
    const ds = this.requireDataset();
    if (!this.split || !this.model) {
      throw new Error("Fit a model on a split first, then `curve`.");
    }
    const X = this.designMatrix(this.split.XTrain);
    const y = this.split.yTrain;
    const n = X.nRows;
    const steps = Math.max(2, fractions);
    const rows: { train: number; valid: number; n: number }[] = [];
    for (let s = 1; s <= steps; s += 1) {
      const nUse = Math.max(4, Math.floor((n * s) / steps));
      const idx = Array.from({ length: nUse }, (_, i) => i);
      const cut = Math.floor(nUse * 0.75);
      const tr = idx.slice(0, cut);
      const va = idx.slice(cut);
      const Xtr = matrix(tr.map((i) => [...(X.data[i] ?? [])]));
      const ytr = vector(tr.map((i) => y.data[i] ?? 0));
      const Xva = matrix(va.map((i) => [...(X.data[i] ?? [])]));
      const yva = vector(va.map((i) => y.data[i] ?? 0));
      try {
        const m = fitModel(this.model.name, Xtr, ytr, this.model.params);
        const trM = computeMetrics(ds.task, ytr, m.predict(Xtr));
        const vaM = computeMetrics(ds.task, yva, m.predict(Xva));
        rows.push({
          train: "accuracy" in trM ? trM.accuracy : trM.r2,
          valid: "accuracy" in vaM ? vaM.accuracy : vaM.r2,
          n: nUse,
        });
      } catch {
        rows.push({ train: 0, valid: 0, n: nUse });
      }
    }
    this.learningCurve = rows;
    this.addStep({
      kind: "curve",
      label: `learning_curve(${steps})`,
      sklearn: `learning_curve(model, X_train, y_train, train_sizes=...)`,
      status: "ok",
    });
    this.commandCount += 1;
    return {
      lines: [
        "learning curve (train vs holdout-in-train):",
        ...rows.map(
          (r) => `  n=${String(r.n).padStart(3)}  train=${r.train.toFixed(3)}  valid=${r.valid.toFixed(3)}`,
        ),
        "Gap high → variance (more data / simpler model). Both low → bias (richer model).",
      ],
      sklearn: "learning_curve(model, X_train, y_train)",
      status: "ok",
      snapshot: this.snapshot(),
    };
  }

  /** Bootstrap CI for a scalar score (test accuracy or r2). */
  bootstrap(reps = 200): CommandResult {
    this.pushHistory();
    const ds = this.requireDataset();
    if (!this.model || !this.split) {
      throw new Error("Fit a model and score before bootstrap.");
    }
    const X = this.designMatrix(this.split.XTest);
    const y = this.split.yTest;
    const pred = this.model.predict(X);
    const rng = makeRng(42);
    const scores: number[] = [];
    for (let b = 0; b < reps; b += 1) {
      const idx = Array.from({ length: X.nRows }, () => Math.floor(rng() * X.nRows));
      const yb = vector(idx.map((i) => y.data[i] ?? 0));
      const pb = vector(idx.map((i) => pred.data[i] ?? 0));
      const met = computeMetrics(ds.task, yb, pb);
      scores.push("accuracy" in met ? met.accuracy : met.r2);
    }
    scores.sort((a, b) => a - b);
    const lo = scores[Math.floor(reps * 0.025)] ?? 0;
    const hi = scores[Math.floor(reps * 0.975)] ?? 1;
    this.bootstrapCi = [lo, hi];
    this.addStep({
      kind: "score",
      label: `bootstrap_${reps}`,
      sklearn: `# bootstrap resample test metrics  (or scipy.stats.bootstrap)`,
      status: "ok",
    });
    this.commandCount += 1;
    return {
      lines: [`bootstrap 95% CI: [${lo.toFixed(3)}, ${hi.toFixed(3)}]  reps=${reps}`, "A point estimate without an interval is a rumor."],
      sklearn: "# bootstrap CI around test score",
      status: "ok",
      snapshot: this.snapshot(),
    };
  }

  /** Persist fitted Pipeline (joblib semantics — local object registry). */
  savePipeline(): CommandResult {
    this.pushHistory();
    if (!this.model) {
      throw new Error("Fit a model first — a Pipeline saves transforms + estimator.");
    }
    this.pipelineSaved = true;
    this.addStep({
      kind: "pipeline",
      label: "Pipeline.save",
      sklearn: "import joblib\njoblib.dump(pipeline, 'model.joblib')",
      status: "ok",
    });
    this.commandCount += 1;
    return {
      lines: [
        "Pipeline (imputer → encoder → scaler → model) saved as model.joblib",
        "Deploy the pipeline, never a bare estimator with silent preprocessing.",
      ],
      sklearn: "joblib.dump(pipeline, 'model.joblib')",
      status: "ok",
      snapshot: this.snapshot(),
    };
  }

  /** Show sklearn Pipeline construction (syntax lesson). */
  pipelineCode(): CommandResult {
    this.pushHistory();
    this.addStep({
      kind: "pipeline",
      label: "Pipeline.compose",
      sklearn: "from sklearn.pipeline import Pipeline",
      status: "ok",
    });
    this.commandCount += 1;
    return {
      lines: [
        "from sklearn.compose import ColumnTransformer",
        "from sklearn.pipeline import Pipeline",
        "from sklearn.impute import SimpleImputer",
        "from sklearn.preprocessing import OneHotEncoder, StandardScaler",
        "",
        "prep = ColumnTransformer([",
        "  ('num', Pipeline([('imp', SimpleImputer()), ('sc', StandardScaler())]), numeric_cols),",
        "  ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')),",
        "                   ('oh', OneHotEncoder(handle_unknown='ignore'))]), cat_cols),",
        "])",
        "pipe = Pipeline([('prep', prep), ('model', LogisticRegression())])",
        "pipe.fit(X_train, y_train)",
        "joblib.dump(pipe, 'model.joblib')",
      ],
      sklearn: "Pipeline([('prep', prep), ('model', clf)]).fit(X_train, y_train)",
      status: "ok",
      snapshot: this.snapshot(),
    };
  }

  /** Feature engineering (train-fitted). */
  fe(mode: FeMode = "interact"): CommandResult {
    this.pushHistory();
    this.requireDataset();
    if (!this.split) {
      throw new Error("Split first — FE statistics must not see test.");
    }
    this.feState = fitFe(this.split.XTrain, this.split.yTrain, mode, 0);
    this.feMode = mode;
    this.model = null;
    this.addStep({
      kind: "fe",
      label:
        mode === "interact"
          ? "interaction features"
          : mode === "bin"
            ? "KBinsDiscretizer"
            : "TargetEncoder",
      sklearn:
        mode === "interact"
          ? "PolynomialFeatures(interaction_only=True)"
          : mode === "bin"
            ? "KBinsDiscretizer(n_bins=4, strategy='quantile')"
            : "TargetEncoder()  # fit on train only",
      status: "ok",
    });
    this.commandCount += 1;
    return {
      lines: [`Applied ${mode} feature engineering on train.`],
      sklearn: mode === "target" ? "TargetEncoder().fit(X_train, y_train)" : "KBins/Polynomial FE",
      status: "ok",
      snapshot: this.snapshot(),
    };
  }

  /** Coefficient / SE / z for linear-like models (Gaussian error approx). */
  infer(): CommandResult {
    this.pushHistory();
    const ds = this.requireDataset();
    if (!this.model || !this.split) {
      throw new Error("Fit a linear-like model first.");
    }
    if (!this.model.weights) {
      throw new Error("`infer` supports linear/ridge/lasso/elasticnet weights.");
    }
    const X = this.designMatrix(this.split.XTrain);
    const pred = this.model.predict(X);
    const res = this.split.yTrain.data.map((y, i) => y - (pred.data[i] ?? 0));
    const n = X.nRows;
    const p = X.nCols;
    const sigma2 = res.reduce((a, e) => a + e * e, 0) / Math.max(1, n - p - 1);
    // Diagonal of (X'X)^-1 approx via ridge-free inverse of Gram matrix
    const XtX = Array.from({ length: p + 1 }, () => Array<number>(p + 1).fill(0));
    for (const row of X.data) {
      const a = [1, ...row];
      for (let i = 0; i <= p; i += 1) {
        for (let j = 0; j <= p; j += 1) {
          XtX[i]![j] = (XtX[i]![j] ?? 0) + (a[i] ?? 0) * (a[j] ?? 0);
        }
      }
    }
    const invDiag = XtX.map((row, i) => {
      // crude SE via 1/diag(X'X) * sigma2
      const d = row[i] ?? 1;
      return sigma2 / (d || 1);
    });
    const names = ["intercept", ...Array.from({ length: p }, (_, j) => `x${j + 1}`)];
    const coefs = [this.model.intercept ?? 0, ...(this.model.weights ?? [])];
    this.coefReport = names.map((name, i) => {
      const coef = coefs[i] ?? 0;
      const se = Math.sqrt(invDiag[i] ?? 1);
      return { name, coef, se, z: se === 0 ? 0 : coef / se };
    });
    this.addStep({
      kind: "infer",
      label: "coef summary",
      sklearn: "import statsmodels.api as sm\nsm.OLS(y, sm.add_constant(X)).fit().summary()",
      status: "ok",
    });
    this.commandCount += 1;
    return {
      lines: [
        "coefficients (approx SE / z):",
        ...this.coefReport.map(
          (r) =>
            `  ${r.name.padEnd(12)} coef=${r.coef.toFixed(4)}  se=${r.se.toFixed(4)}  z=${r.z.toFixed(2)}`,
        ),
        "MLE view: OLS is Gaussian MLE; logistic is Bernoulli MLE. SEs here are diagonal-approx.",
        `task=${ds.task}`,
      ],
      sklearn: "sm.OLS(y, X).fit().summary()",
      status: "ok",
      snapshot: this.snapshot(),
    };
  }

  /** Impurity-based feature importance for tree/forest/boost. */
  featureImportance(): CommandResult {
    this.pushHistory();
    this.requireDataset();
    if (!this.model || !this.split) {
      throw new Error("Fit a tree/forest/boost model first.");
    }
    const X = this.designMatrix(this.split.XTrain);
    const y = this.split.yTrain;
    const scores = Array<number>(X.nCols).fill(0);
    // Permutation-ish: correlate |x_j| variation with residual drop using univariate splits.
    for (let j = 0; j < X.nCols; j += 1) {
      let ss = 0;
      for (let i = 0; i < X.nRows; i += 1) {
        ss += (X.data[i]?.[j] ?? 0) ** 2;
      }
      // covariance with y as a cheap importance proxy
      let cov = 0;
      for (let i = 0; i < X.nRows; i += 1) {
        cov += (X.data[i]?.[j] ?? 0) * ((y.data[i] ?? 0) - mean(y));
      }
      scores[j] = Math.abs(cov) / (Math.sqrt(ss) + 1e-9);
    }
    this.importance = scores.map((score, j) => ({ feature: `f${j}`, score }));
    this.addStep({
      kind: "score",
      label: "feature_importances_",
      sklearn: "model.feature_importances_  # or permutation_importance",
      status: "ok",
    });
    this.commandCount += 1;
    return {
      lines: [
        "feature importance (proxy scores):",
        ...this.importance
          .slice()
          .sort((a, b) => b.score - a.score)
          .map((r) => `  ${r.feature}  ${r.score.toFixed(3)}`),
        "In sklearn prefer permutation_importance on a hold-out fold.",
      ],
      sklearn: "from sklearn.inspection import permutation_importance",
      status: "ok",
      snapshot: this.snapshot(),
    };
  }

  /** Silhouette score for a clustering model. */
  sil(): CommandResult {
    this.pushHistory();
    this.requireDataset();
    if (!this.model || (this.model.name !== "kmeans" && this.model.name !== "dbscan")) {
      throw new Error("Fit kmeans or dbscan first.");
    }
    const X = this.designMatrix(this.split?.XTrain ?? this.dataset!.X);
    const lab = this.model.predict(X);
    const n = X.nRows;
    const dist = (i: number, j: number) => {
      let d = 0;
      const ri = X.data[i] ?? [];
      const rj = X.data[j] ?? [];
      for (let k = 0; k < X.nCols; k += 1) d += ((ri[k] ?? 0) - (rj[k] ?? 0)) ** 2;
      return Math.sqrt(d);
    };
    let total = 0;
    let used = 0;
    for (let i = 0; i < n; i += 1) {
      const li = lab.data[i] ?? -1;
      if (li < 0) continue;
      let a = 0;
      let aCount = 0;
      const clusterD = new Map<number, { sum: number; cnt: number }>();
      for (let j = 0; j < n; j += 1) {
        if (i === j) continue;
        const lj = lab.data[j] ?? -1;
        const d = dist(i, j);
        if (lj === li) {
          a += d;
          aCount += 1;
        } else if (lj >= 0) {
          const rec = clusterD.get(lj) ?? { sum: 0, cnt: 0 };
          rec.sum += d;
          rec.cnt += 1;
          clusterD.set(lj, rec);
        }
      }
      a = aCount ? a / aCount : 0;
      let b = Infinity;
      for (const rec of clusterD.values()) {
        if (rec.cnt > 0) b = Math.min(b, rec.sum / rec.cnt);
      }
      if (!Number.isFinite(b) || aCount === 0) continue;
      total += (b - a) / Math.max(a, b, 1e-9);
      used += 1;
    }
    this.silhouette = used ? total / used : 0;
    this.addStep({
      kind: "score",
      label: "silhouette",
      sklearn: "from sklearn.metrics import silhouette_score\nsilhouette_score(X, labels)",
      status: "ok",
    });
    this.commandCount += 1;
    return {
      lines: [`silhouette ≈ ${this.silhouette.toFixed(3)} (range -1..1; higher is tighter)`, "Compare k values; never ship k on inertia alone."],
      sklearn: "silhouette_score(X, labels)",
      status: "ok",
      snapshot: this.snapshot(),
    };
  }

  /** Outer-loop estimate + inner search (minimal nested CV). */
  nestedCv(model: ModelName, grid: Record<string, (number | string)[]>, outer = 3): CommandResult {
    this.pushHistory();
    const ds = this.requireDataset();
    if (!this.split) {
      throw new Error("Split first. Nested CV runs on train only.");
    }
    const X = this.designMatrix(this.split.XTrain);
    const y = this.split.yTrain;
    const n = X.nRows;
    const fold = Math.floor(n / outer);
    const outerScores: number[] = [];
    for (let o = 0; o < outer; o += 1) {
      const testIdx = Array.from({ length: o === outer - 1 ? n - o * fold : fold }, (_, i) => o * fold + i);
      const trainIdx = Array.from({ length: n }, (_, i) => i).filter((i) => !testIdx.includes(i));
      // inner 2-fold search
      let best: Record<string, number | string> = {};
      let bestScore = -Infinity;
      const keys = Object.keys(grid);
      let combos: Record<string, number | string>[] = [{}];
      for (const key of keys) {
        const next: Record<string, number | string>[] = [];
        for (const base of combos) {
          for (const v of grid[key] ?? []) next.push({ ...base, [key]: v });
        }
        combos = next;
      }
      for (const params of combos) {
        let s = 0;
        for (let inner = 0; inner < 2; inner += 1) {
          const cut = Math.floor(trainIdx.length / 2);
          const va = inner === 0 ? trainIdx.slice(0, cut) : trainIdx.slice(cut);
          const tr = inner === 0 ? trainIdx.slice(cut) : trainIdx.slice(0, cut);
          const Xtr = matrix(tr.map((i) => [...(X.data[i] ?? [])]));
          const ytr = vector(tr.map((i) => y.data[i] ?? 0));
          const Xva = matrix(va.map((i) => [...(X.data[i] ?? [])]));
          const yva = vector(va.map((i) => y.data[i] ?? 0));
          try {
            const m = fitModel(model, Xtr, ytr, params);
            const met = computeMetrics(ds.task, yva, m.predict(Xva));
            s += "accuracy" in met ? met.accuracy : met.r2;
          } catch {
            s += 0;
          }
        }
        s /= 2;
        if (s > bestScore) {
          bestScore = s;
          best = params;
        }
      }
      const Xtr = matrix(trainIdx.map((i) => [...(X.data[i] ?? [])]));
      const ytr = vector(trainIdx.map((i) => y.data[i] ?? 0));
      const Xte = matrix(testIdx.map((i) => [...(X.data[i] ?? [])]));
      const yte = vector(testIdx.map((i) => y.data[i] ?? 0));
      try {
        const m = fitModel(model, Xtr, ytr, best);
        const met = computeMetrics(ds.task, yte, m.predict(Xte));
        outerScores.push("accuracy" in met ? met.accuracy : met.r2);
      } catch {
        outerScores.push(0);
      }
    }
    this.nestedCvOuter = outerScores;
    this.searchBest = {};
    this.addStep({
      kind: "search",
      label: `nested_cv(${outer})`,
      sklearn: "# outer KFold + inner GridSearchCV",
      status: "ok",
    });
    this.commandCount += 1;
    const mu = outerScores.reduce((a, b) => a + b, 0) / (outerScores.length || 1);
    return {
      lines: [
        `nested outer scores: ${outerScores.map((s) => s.toFixed(3)).join(" ")}  mean=${mu.toFixed(3)}`,
        "Outer scores estimate the *selection procedure*, not one lucky grid cell.",
      ],
      sklearn: "# nested cross validation",
      status: "ok",
      snapshot: this.snapshot(),
    };
  }

  /** Reliability bins: predicted p vs empirical frequency. */
  calib(): CommandResult {
    this.pushHistory();
    const ds = this.requireDataset();
    if (!this.model || !this.split || ds.task !== "classification") {
      throw new Error("Need a fitted classifier.");
    }
    const X = this.designMatrix(this.split.XTest);
    const y = this.split.yTest;
    const dec = this.model.decision(X);
    const probs = dec.data.map((z) => (z > 0 && z <= 1 ? z : 1 / (1 + Math.exp(-z))));
    const bins = Array.from({ length: 5 }, () => ({ sum: 0, cnt: 0, pos: 0 }));
    for (let i = 0; i < probs.length; i += 1) {
      const p = probs[i] ?? 0;
      const b = Math.min(4, Math.floor(p * 5));
      const bin = bins[b]!;
      bin.sum += p;
      bin.cnt += 1;
      if ((y.data[i] ?? 0) >= 0.5) bin.pos += 1;
    }
    this.calibCurve = bins.map((b, i) => ({
      p: b.cnt ? b.sum / b.cnt : (i + 0.5) / 5,
      rate: b.cnt ? b.pos / b.cnt : 0,
      n: b.cnt,
    }));
    this.addStep({
      kind: "score",
      label: "calibration_curve",
      sklearn: "from sklearn.calibration import calibration_curve\ny_prob, y_rate = calibration_curve(y_test, p, n_bins=5)",
      status: "ok",
    });
    this.commandCount += 1;
    return {
      lines: [
        "calibration bins (mean p vs empirical rate):",
        ...this.calibCurve.map(
          (r) => `  p=${r.p.toFixed(2)}  rate=${r.rate.toFixed(2)}  n=${r.n}`,
        ),
        "Diagonal is perfect calibration. Large gaps need CalibratedClassifierCV or isotonic.",
      ],
      sklearn: "calibration_curve(y_test, p, n_bins=5)",
      status: "ok",
      snapshot: this.snapshot(),
    };
  }

  /** Prediction intervals via residual sd (regression). */
  predInterval(): CommandResult {
    this.pushHistory();
    const ds = this.requireDataset();
    if (!this.model || !this.split || ds.task !== "regression") {
      throw new Error("Need a fitted regressor.");
    }
    const X = this.designMatrix(this.split.XTrain);
    const pred = this.model.predict(X);
    const res = this.split.yTrain.data.map((y, i) => y - (pred.data[i] ?? 0));
    const n = res.length || 1;
    const mu = res.reduce((a, b) => a + b, 0) / n;
    const sd = Math.sqrt(res.reduce((a, b) => a + (b - mu) ** 2, 0) / n);
    const Xt = this.designMatrix(this.split.XTest);
    const pt = this.model.predict(Xt);
    this.predCi = pt.data.slice(0, 8).map((point) => ({
      point,
      lo: point - 1.96 * sd,
      hi: point + 1.96 * sd,
    }));
    this.addStep({
      kind: "infer",
      label: "prediction_interval",
      sklearn: "# ŷ ± 1.96 * σ_residual  (simplified)",
      status: "ok",
    });
    this.commandCount += 1;
    return {
      lines: [
        `residual sd=${sd.toFixed(3)}`,
        "prediction intervals (first rows):",
        ...this.predCi.map(
          (r) => `  ŷ=${r.point.toFixed(2)}  [${r.lo.toFixed(2)}, ${r.hi.toFixed(2)}]`,
        ),
        "Intervals are for a future observation — wider than a CI for the mean.",
      ],
      sklearn: "# prediction interval from residual sd",
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
    this.feMode = null;
    this.feState = null;
    this.coefReport = null;
    this.importance = null;
    this.silhouette = null;
    this.calibCurve = null;
    this.predCi = null;
    this.nestedCvOuter = null;
    this.cvScores = null;
    this.searchBest = null;
    this.learningCurve = null;
    this.bootstrapCi = null;
    this.pipelineSaved = false;
    this.splitStrategy = null;
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
