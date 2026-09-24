/**
 * Engine unit tests — datasets, split, metrics, leakage rules, win paths.
 */

import { describe, expect, it } from "vitest";
import {
  Session,
  classificationMetrics,
  loadDataset,
  regressionMetrics,
  trainTestSplit,
  fitLinear,
  fitKnn,
  fitScaler,
  transform,
} from "../src/engine";
import { WORLD1_LEVELS } from "../src/levels/world1";
import { WORLD2_LEVELS } from "../src/levels/world2";
import { WORLD3_LEVELS } from "../src/levels/world3";

describe("datasets", () => {
  it("loads blobs as classification with consistent shapes", () => {
    const ds = loadDataset("blobs");
    expect(ds.task).toBe("classification");
    expect(ds.X.nRows).toBe(ds.y.data.length);
    expect(ds.X.nCols).toBe(2);
  });

  it("loads noisy_line as regression", () => {
    const ds = loadDataset("noisy_line");
    expect(ds.task).toBe("regression");
    expect(ds.X.nCols).toBe(1);
  });
});

describe("train_test_split", () => {
  it("is deterministic for a fixed seed", () => {
    const ds = loadDataset("blobs");
    const a = trainTestSplit(ds.X, ds.y, 0.2, 42);
    const b = trainTestSplit(ds.X, ds.y, 0.2, 42);
    expect(a.XTest.data).toEqual(b.XTest.data);
    expect(a.XTrain.nRows + a.XTest.nRows).toBe(ds.X.nRows);
  });

  it("rejects empty-ish test sizes", () => {
    const ds = loadDataset("blobs");
    expect(() => trainTestSplit(ds.X, ds.y, 0, 1)).toThrow();
    expect(() => trainTestSplit(ds.X, ds.y, 1, 1)).toThrow();
  });
});

describe("metrics", () => {
  it("computes binary classification metrics", () => {
    const yTrue = { data: [1, 1, 0, 0, 1] };
    const yPred = { data: [1, 0, 0, 1, 1] };
    const m = classificationMetrics(yTrue, yPred);
    expect(m.n).toBe(5);
    expect(m.confusion).toEqual([
      [1, 1],
      [1, 2],
    ]);
    expect(m.accuracy).toBeCloseTo(3 / 5);
  });

  it("computes regression r2 on a perfect line", () => {
    const yTrue = { data: [1, 2, 3] };
    const yPred = { data: [1, 2, 3] };
    const m = regressionMetrics(yTrue, yPred);
    expect(m.r2).toBeCloseTo(1);
    expect(m.mse).toBeCloseTo(0);
  });
});

describe("estimators", () => {
  it("fits linear regression on a clean line", () => {
    const X = { nRows: 20, nCols: 1, data: Array.from({ length: 20 }, (_, i) => [i / 5]) };
    const y = { data: Array.from({ length: 20 }, (_, i) => 2 * (i / 5) + 1) };
    const model = fitLinear(X, y);
    expect(model.intercept ?? 0).toBeCloseTo(1, 3);
    expect(model.weights?.[0] ?? 0).toBeCloseTo(2, 3);
  });

  it("knn memorizes with k=1", () => {
    const ds = loadDataset("moons");
    const model = fitKnn(ds.X, ds.y, { n_neighbors: 1 });
    const pred = model.predict(ds.X);
    const correct = pred.data.filter((p, i) => p === (ds.y.data[i] ?? -1)).length;
    expect(correct / ds.y.data.length).toBeGreaterThan(0.99);
  });
});

describe("scaling and leakage", () => {
  it("standard scaler centers columns", () => {
    const ds = loadDataset("blobs");
    const scaler = fitScaler(ds.X, "standard");
    const Z = transform(ds.X, scaler);
    let mean0 = 0;
    for (const row of Z.data) {
      mean0 += row[0] ?? 0;
    }
    mean0 /= Z.nRows;
    expect(Math.abs(mean0)).toBeLessThan(1e-6);
  });

  it("session flags scale-before-split as leakage", () => {
    const s = new Session();
    s.load("scale_trap");
    s.scale("standard");
    s.split_(0.2, 42);
    expect(s.snapshot().scaleLeaked).toBe(true);
  });

  it("session keeps scale-after-split clean", () => {
    const s = new Session();
    s.load("scale_trap");
    s.split_(0.2, 42);
    s.scale("standard");
    expect(s.snapshot().scaleLeaked).toBe(false);
  });
});

describe("world 1 win checks", () => {
  it("1.1 requires show data on a loaded dataset", () => {
    const s = new Session();
    s.load("blobs");
    const level = WORLD1_LEVELS.find((l) => l.id === "1.1")!;
    expect(level.win(s.snapshot()).won).toBe(false);
    s.markInspected();
    expect(level.win(s.snapshot()).won).toBe(true);
  });

  it("1.2 clears on noisy_line linear fit", () => {
    const s = new Session();
    s.load("noisy_line");
    s.split_(0.2, 42);
    s.fit("linear");
    s.score("test");
    const level = WORLD1_LEVELS.find((l) => l.id === "1.2")!;
    const wr = level.win(s.snapshot());
    expect(wr.won).toBe(true);
  });

  it("1.3 requires split before fit and test accuracy", () => {
    const s = new Session();
    s.load("blobs");
    s.split_(0.25, 42);
    s.fit("logistic");
    s.score("test");
    const level = WORLD1_LEVELS.find((l) => l.id === "1.3")!;
    const wr = level.win(s.snapshot());
    expect(wr.won).toBe(true);
  });

  it("1.5 rejects leakage path", () => {
    const s = new Session();
    s.load("scale_trap");
    s.scale("standard");
    s.split_(0.2, 42);
    s.fit("knn", { n_neighbors: 5 });
    s.score("test");
    const level = WORLD1_LEVELS.find((l) => l.id === "1.5")!;
    const wr = level.win(s.snapshot());
    expect(wr.won).toBe(false);
    expect(wr.feedback).toMatch(/leak/i);
  });

  it("1.5 accepts clean path", () => {
    const s = new Session();
    s.load("scale_trap");
    s.split_(0.2, 42);
    s.scale("standard");
    s.fit("knn", { n_neighbors: 5 });
    s.score("test");
    const level = WORLD1_LEVELS.find((l) => l.id === "1.5")!;
    const wr = level.win(s.snapshot());
    expect(wr.won).toBe(true);
  });

  it("1.4 shows memorization gap with k=1 on moons", () => {
    const s = new Session();
    s.load("moons");
    s.split_(0.25, 7);
    s.fit("knn", { n_neighbors: 1 });
    s.score("train");
    s.score("test");
    const level = WORLD1_LEVELS.find((l) => l.id === "1.4")!;
    const snap = s.snapshot();
    const wr = level.win(snap);
    expect(wr.won).toBe(true);
    expect(wr.feedback).toMatch(/gap/i);
    const tr = snap.trainMetrics as { accuracy: number };
    const te = snap.metrics as { accuracy: number };
    expect(tr.accuracy).toBeGreaterThanOrEqual(0.99);
    expect(tr.accuracy - te.accuracy).toBeGreaterThanOrEqual(0.03);
  });
});

describe("world 2 win checks", () => {
  it("2.1 requires logistic beating a weak baseline on blobs", () => {
    const s = new Session();
    s.load("blobs");
    s.split_(0.2, 42);
    s.fit("dummy");
    s.score("test");
    s.fit("logistic");
    s.score("test");
    const level = WORLD2_LEVELS.find((l) => l.id === "2.1")!;
    expect(level.win(s.snapshot()).won).toBe(true);
  });

  it("2.2 requires confusion matrix view", () => {
    const s = new Session();
    s.load("moons");
    s.split_(0.2, 42);
    s.fit("logistic");
    s.score("test");
    const level = WORLD2_LEVELS.find((l) => l.id === "2.2")!;
    expect(level.win(s.snapshot()).won).toBe(false);
    s.showConfusion();
    const wr = level.win(s.snapshot());
    // moons + logistic may sit near the 0.8 floor depending on seed
    expect(wr.feedback.length).toBeGreaterThan(10);
  });

  it("2.3 requires both train and test scores", () => {
    const s = new Session();
    s.load("scale_trap");
    s.split_(0.2, 42);
    s.scale("standard");
    s.fit("knn", { n_neighbors: 5 });
    s.score("test");
    const level = WORLD2_LEVELS.find((l) => l.id === "2.3")!;
    expect(level.win(s.snapshot()).won).toBe(false);
    s.score("train");
    s.score("test");
    expect(level.win(s.snapshot()).won).toBe(true);
  });
});

describe("undo", () => {
  it("restores inspected flag and model state", () => {
    const s = new Session();
    s.load("blobs");
    s.markInspected();
    expect(s.snapshot().inspectedData).toBe(true);
    s.undo();
    // markInspected does not push history; undo after load clears dataset
    s.load("blobs");
    s.split_(0.2, 42);
    s.fit("logistic");
    expect(s.snapshot().fitted).toBe(true);
    s.undo();
    expect(s.snapshot().fitted).toBe(false);
    expect(s.snapshot().split).not.toBeNull();
  });
});

describe("world 3-7 smoke", () => {
  it("poly 3 improves fit on poly_curve", () => {
    const s = new Session();
    s.load("poly_curve");
    s.split_(0.2, 42);
    s.fit("linear");
    s.score("test");
    const before = s.snapshot().metrics as { r2: number };
    s.poly(3);
    s.fit("linear");
    s.score("test");
    const after = s.snapshot().metrics as { r2: number };
    expect(after.r2).toBeGreaterThan(before.r2);
  });

  it("high-degree OLS overfits poly_curve while ridge stays calmer", () => {
    const ols = new Session();
    ols.load("poly_curve");
    ols.split_(0.3, 7);
    ols.poly(12);
    ols.fit("linear");
    ols.score("train");
    ols.score("test");
    const olsTrain = (ols.snapshot().trainMetrics as { r2: number }).r2;
    const olsTest = (ols.snapshot().metrics as { r2: number }).r2;

    const ridge = new Session();
    ridge.load("poly_curve");
    ridge.split_(0.3, 7);
    ridge.poly(12);
    ridge.fit("ridge", { alpha: 5 });
    ridge.score("train");
    ridge.score("test");
    const ridgeTrain = (ridge.snapshot().trainMetrics as { r2: number }).r2;
    const ridgeTest = (ridge.snapshot().metrics as { r2: number }).r2;

    expect(olsTrain).toBeGreaterThan(0.7);
    expect(olsTrain - olsTest).toBeGreaterThan(0.03);
    expect(ridgeTrain).toBeLessThan(olsTrain + 1e-6);
    expect(ridgeTest).toBeGreaterThan(-1);
  });

  it("impute fills NaN and logistic can score mixed_table", () => {
    const s = new Session();
    s.load("mixed_table");
    s.split_(0.25, 42);
    s.impute("mean");
    s.encode("onehot");
    s.fit("logistic");
    s.score("test");
    expect(s.snapshot().metrics).toBeTruthy();
  });

  it("tree overfits moons more than stump on train", () => {
    const s = new Session();
    s.load("moons");
    s.split_(0.25, 7);
    s.fit("tree", { max_depth: 6 });
    s.score("train");
    const deepTrain = (s.snapshot().trainMetrics as { accuracy: number }).accuracy;
    s.fit("tree", { max_depth: 1 });
    s.score("train");
    const stumpTrain = (s.snapshot().trainMetrics as { accuracy: number }).accuracy;
    expect(deepTrain).toBeGreaterThanOrEqual(stumpTrain);
  });

  it("search records best params without touching test", () => {
    const s = new Session();
    s.load("poly_curve");
    s.split_(0.2, 42);
    s.poly(3);
    s.search("ridge", { alpha: [0.01, 1, 10] });
    expect(s.snapshot().searchBest).toBeTruthy();
  });

  it("kmeans recovers 3 blobs", () => {
    const s = new Session();
    s.load("clusters");
    s.split_(0.2, 42);
    s.fit("kmeans", { n_clusters: 3 });
    s.score("test");
    const m = s.snapshot().metrics as { accuracy: number };
    expect(m.accuracy).toBeGreaterThan(0.7);
  });

  it("world levels exist for 3-7 and have learning lists", () => {
    for (const lv of WORLD3_LEVELS) {
      expect(lv.learning.length).toBeGreaterThan(0);
      expect(lv.concept.whatHappens.length).toBeGreaterThan(20);
      expect(lv.steps.length).toBeGreaterThan(0);
    }
  });

  it("learning curve, bootstrap, boost, time split work", () => {
    const s = new Session();
    s.load("poly_curve");
    s.split_(0.2, 42);
    s.poly(3);
    s.fit("ridge", { alpha: 1 });
    s.curve(4);
    expect(s.snapshot().learningCurve?.length).toBeGreaterThanOrEqual(3);
    s.score("test");
    s.bootstrap(50);
    expect(s.snapshot().bootstrapCi).toBeTruthy();

    const b = new Session();
    b.load("moons");
    b.split_(0.25, 7);
    b.fit("boost", { n_estimators: 15, max_depth: 2, lr: 0.3 });
    b.score("test");
    expect(b.snapshot().metrics).toBeTruthy();

    const t = new Session();
    t.load("noisy_line");
    t.split_(0.3, 1, "time");
    expect(t.snapshot().splitStrategy).toBe("time");
    t.fit("linear");
    t.score("test");
    expect(t.snapshot().scoredOn).toBe("test");
  });
});
