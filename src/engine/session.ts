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
import { fitScaler, scalerLabel, transform, type Scaler } from "./transformers";
import type {
  Dataset,
  DatasetName,
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
  model: FittedModel | null;
  predictions: Vector | null;
  metrics: Metrics | null;
  trainMetrics: Metrics | null;
  scoredOn: "train" | "test" | null;
  steps: PipelineStep[];
  commandCount: number;
  inspectedData: boolean;
}

export class Session {
  private dataset: Dataset | null = null;
  private split: Split | null = null;
  private scaler: Scaler | null = null;
  private scaleLeaked = false;
  private scalePhase: "none" | "before_split" | "after_split" = "none";
  private model: FittedModel | null = null;
  private predictions: Vector | null = null;
  private metrics: Metrics | null = null;
  private trainMetrics: Metrics | null = null;
  private scoredOn: "train" | "test" | null = null;
  private steps: PipelineStep[] = [];
  private commandCount = 0;
  private history: SessionState[] = [];
  private inspectedData = false;

  snapshot(): SessionSnapshot {
    return {
      dataset: this.dataset,
      split: this.split,
      scaled: this.scaler?.name ?? null,
      scaleLeaked: this.scaleLeaked,
      model: this.model?.name ?? null,
      modelParams: this.model?.params ?? {},
      fitted: this.model !== null,
      predictions: this.predictions,
      metrics: this.metrics,
      trainMetrics: this.trainMetrics,
      steps: [...this.steps],
      commandCount: this.commandCount,
      scoredOn: this.scoredOn,
      inspectedData: this.inspectedData,
    };
  }

  /** Mark that the learner inspected X/y via `show data`. */
  markInspected(): void {
    this.inspectedData = true;
    this.commandCount += 1;
  }

  get inspected(): boolean {
    return this.inspectedData;
  }

  private capture(): SessionState {
    return {
      dataset: this.dataset,
      split: this.split,
      scaler: this.scaler,
      scaleLeaked: this.scaleLeaked,
      scalePhase: this.scalePhase,
      model: this.model,
      predictions: this.predictions,
      metrics: this.metrics,
      trainMetrics: this.trainMetrics,
      scoredOn: this.scoredOn,
      steps: [...this.steps],
      commandCount: this.commandCount,
      inspectedData: this.inspectedData,
    };
  }

  private restore(state: SessionState): void {
    this.dataset = state.dataset;
    this.split = state.split;
    this.scaler = state.scaler;
    this.scaleLeaked = state.scaleLeaked;
    this.scalePhase = state.scalePhase;
    this.model = state.model;
    this.predictions = state.predictions;
    this.metrics = state.metrics;
    this.trainMetrics = state.trainMetrics;
    this.scoredOn = state.scoredOn;
    this.steps = [...state.steps];
    this.commandCount = state.commandCount;
    this.inspectedData = state.inspectedData;
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
    let XTrain = this.split.XTrain;
    let yTrain = this.split.yTrain;
    if (this.scaler) {
      XTrain = transform(XTrain, this.scaler);
    }
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
    let XTest = this.split.XTest;
    if (this.scaler) {
      XTest = transform(XTest, this.scaler);
    }
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
    let X = on === "test" ? this.split.XTest : this.split.XTrain;
    let y = on === "test" ? this.split.yTest : this.split.yTrain;
    if (this.scaler) {
      X = transform(X, this.scaler);
    }
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
