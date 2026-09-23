/**
 * Shared engine types for the LearnML sandbox.
 *
 * Mirrors the scikit-learn estimator / transformer mental model without
 * depending on a real scikit-learn runtime.
 */

export type DatasetName = "blobs" | "moons" | "noisy_line" | "outlier_line" | "scale_trap";

export type ModelName = "linear" | "logistic" | "knn" | "ridge" | "dummy";

export type ScalerName = "standard" | "minmax";

export type TaskKind = "classification" | "regression";

export interface Matrix {
  readonly nRows: number;
  readonly nCols: number;
  readonly data: number[][];
}

export interface Vector {
  readonly data: number[];
}

export interface Dataset {
  readonly name: DatasetName;
  readonly task: TaskKind;
  readonly featureNames: readonly string[];
  readonly targetName: string;
  readonly X: Matrix;
  readonly y: Vector;
  readonly classNames?: readonly string[];
}

export interface Split {
  readonly XTrain: Matrix;
  readonly XTest: Matrix;
  readonly yTrain: Vector;
  readonly yTest: Vector;
  readonly testSize: number;
  readonly seed: number;
}

export interface ClassificationMetrics {
  readonly accuracy: number;
  readonly precision: number;
  readonly recall: number;
  readonly f1: number;
  readonly confusion: readonly (readonly number[])[];
  readonly n: number;
}

export interface RegressionMetrics {
  readonly mae: number;
  readonly mse: number;
  readonly r2: number;
  readonly n: number;
}

export type Metrics = ClassificationMetrics | RegressionMetrics;

export type StepKind =
  | "load"
  | "split"
  | "scale"
  | "fit"
  | "predict"
  | "score"
  | "reset";

export interface PipelineStep {
  readonly kind: StepKind;
  readonly label: string;
  readonly sklearn: string;
  readonly status: "ok" | "warn" | "error";
  readonly detail?: string;
}

export interface WinResult {
  readonly won: boolean;
  readonly feedback: string;
}

export interface ConceptBrief {
  readonly title: string;
  readonly body: string;
  readonly whatHappens: string;
  readonly why: string;
  readonly formula?: string;
  readonly callout?: string;
}

export interface GoalStep {
  readonly id: string;
  readonly label: string;
  readonly detail: string;
  readonly command?: string;
  readonly check: (state: SessionSnapshot) => boolean;
}

export interface Level {
  readonly id: string;
  readonly world: string;
  readonly worldTitle: string;
  readonly title: string;
  readonly concept: ConceptBrief;
  readonly goal: string;
  readonly hints: readonly string[];
  readonly learning: readonly string[];
  readonly steps: readonly GoalStep[];
  readonly seedDataset?: DatasetName;
  readonly win: (state: SessionSnapshot) => WinResult;
}

export interface SessionSnapshot {
  readonly dataset: Dataset | null;
  readonly split: Split | null;
  readonly scaled: ScalerName | null;
  /** True when the scaler was fit on train only (the correct order). */
  readonly scaleLeaked: boolean;
  readonly model: ModelName | null;
  readonly modelParams: Readonly<Record<string, number | string>>;
  readonly fitted: boolean;
  readonly predictions: Vector | null;
  readonly metrics: Metrics | null;
  readonly trainMetrics: Metrics | null;
  readonly steps: readonly PipelineStep[];
  readonly commandCount: number;
  readonly scoredOn: "train" | "test" | null;
  /** True after the learner ran `show data` on the current dataset. */
  readonly inspectedData: boolean;
}
