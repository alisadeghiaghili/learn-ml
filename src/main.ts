/**
 * LearnML shell: chrome, level rail, concept brief, stage, command dock.
 */

import "./styles/app.css";
import {
  Session,
  dispatch,
  fitModel,
  fitScaler,
  transform,
  formatMetrics,
  type CommandResult,
  type FittedModel,
  type Level,
  type ModelName,
  type ModelParams,
  type SessionSnapshot,
} from "./engine";
import { ALL_LEVELS, findLevel } from "./levels";
import { drawStage, type DrawState } from "./ui/charts";

type Mode = "level" | "sandbox";

const SOLVED_KEY = "learnml.solved.v1";

function loadSolved(): Set<string> {
  try {
    const raw = localStorage.getItem(SOLVED_KEY);
    if (!raw) {
      return new Set();
    }
    return new Set(JSON.parse(raw) as string[]);
  } catch {
    return new Set();
  }
}

function saveSolved(ids: Set<string>): void {
  localStorage.setItem(SOLVED_KEY, JSON.stringify([...ids]));
}

interface AppView {
  mode: Mode;
  levelId: string;
  session: Session;
  solved: Set<string>;
  fitProgress: number;
  liveModel: FittedModel | null;
  lastLines: { kind: "cmd" | "out" | "err" | "sk"; text: string }[];
  winFeedback: string | null;
  winOk: boolean;
}

function h<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  props: Record<string, string> = {},
  ...children: (Node | string)[]
): HTMLElementTagNameMap[K] {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (k === "class") {
      el.className = v;
    } else if (k === "text") {
      el.textContent = v;
    } else {
      el.setAttribute(k, v);
    }
  }
  for (const c of children) {
    el.append(c);
  }
  return el;
}

function currentLevel(view: AppView): Level | null {
  return findLevel(view.levelId) ?? null;
}

function maybeWin(view: AppView, result: CommandResult): void {
  const level = currentLevel(view);
  if (!level || view.mode !== "level") {
    return;
  }
  const wr = level.win(result.snapshot);
  view.winFeedback = wr.feedback;
  view.winOk = wr.won;
  if (wr.won) {
    view.solved.add(level.id);
    saveSolved(view.solved);
  }
}

function rebuildModel(snap: SessionSnapshot): FittedModel | null {
  if (!snap.fitted || !snap.model || !snap.split || !snap.dataset) {
    return null;
  }
  let XTrain = snap.split.XTrain;
  if (snap.scaled) {
    XTrain = transform(XTrain, fitScaler(snap.split.XTrain, snap.scaled));
  }
  try {
    return fitModel(snap.model as ModelName, XTrain, snap.split.yTrain, {
      ...(snap.modelParams as ModelParams),
    });
  } catch {
    return null;
  }
}

export function mount(root: HTMLElement): void {
  const view: AppView = {
    mode: "level",
    levelId: ALL_LEVELS[0]?.id ?? "1.1",
    session: new Session(),
    solved: loadSolved(),
    fitProgress: 0,
    liveModel: null,
    lastLines: [],
    winFeedback: null,
    winOk: false,
  };

  const seed = () => currentLevel(view)?.seedDataset;
  if (seed()) {
    const r = view.session.load(seed()!);
    view.lastLines.push(...r.lines.map((text) => ({ kind: "out" as const, text })));
  }

  const elChrome = h("header", { class: "chrome" });
  const elBrand = h("div", { class: "brand" }, "Learn", h("span", {}, "ML"));
  const elNav = h("nav", { class: "chrome-nav" });
  const btnLevels = h("button", { type: "button", class: "is-active", text: "levels" });
  const btnSandbox = h("button", { type: "button", text: "sandbox" });
  const btnHelp = h("button", { type: "button", text: "help" });
  elNav.append(btnLevels, btnSandbox, btnHelp);
  const elMeta = h("div", { class: "chrome-meta", text: "scikit-learn mental model" });
  elChrome.append(elBrand, elNav, elMeta);

  const elBody = h("div", { class: "body" });
  const elRail = h("aside", { class: "rail" });
  const elMain = h("div", { class: "main" });
  const elBrief = h("section", { class: "brief" });
  const elStage = h("section", { class: "stage" });
  const elViz = h("div", { class: "stage-viz" });
  const canvas = document.createElement("canvas");
  elViz.append(canvas);
  const elSide = h("div", { class: "stage-side" });
  elStage.append(elViz, elSide);
  elMain.append(elBrief, elStage);
  elBody.append(elRail, elMain);

  const elDock = h("footer", { class: "dock" });
  const elLog = h("div", { class: "dock-log" });
  const elInputRow = h("div", { class: "dock-input-row" });
  const elPrompt = h("span", { class: "dock-prompt", text: "$" });
  const input = h("input", {
    class: "dock-input",
    type: "text",
    spellcheck: "false",
    autocomplete: "off",
    placeholder: "load blobs",
    "aria-label": "command",
  }) as HTMLInputElement;
  elInputRow.append(elPrompt, input);
  const elSk = h("div", { class: "dock-sklearn" });
  elDock.append(elLog, elInputRow, elSk);

  root.append(elChrome, elBody, elDock);

  function setSklearn(text: string | null): void {
    elSk.replaceChildren();
    if (!text) {
      elSk.append(h("span", { class: "lbl", text: "scikit-learn · " }), "ready");
      return;
    }
    elSk.append(h("span", { class: "lbl", text: "scikit-learn · " }), text);
  }

  function renderLog(): void {
    elLog.replaceChildren();
    for (const line of view.lastLines.slice(-80)) {
      const cls =
        line.kind === "cmd"
          ? "line-cmd"
          : line.kind === "err"
            ? "line-err"
            : line.kind === "sk"
              ? "line-sk"
              : "line-out";
      elLog.append(h("div", { class: cls, text: line.text }));
    }
    elLog.scrollTop = elLog.scrollHeight;
  }

  function renderRail(): void {
    elRail.replaceChildren();
    const worlds = new Map<string, Level[]>();
    for (const lv of ALL_LEVELS) {
      const list = worlds.get(lv.world) ?? [];
      list.push(lv);
      worlds.set(lv.world, list);
    }
    for (const [world, levels] of worlds) {
      const label =
        world === "w1"
          ? "WORLD 1 · FIT IS NOT UNDERSTANDING"
          : world === "w2"
            ? "WORLD 2 · METRICS LIE"
            : world.toUpperCase();
      elRail.append(h("div", { class: "rail-world", text: label }));
      for (const lv of levels) {
        const solved = view.solved.has(lv.id);
        const active = view.mode === "level" && view.levelId === lv.id;
        const btn = h(
          "button",
          {
            type: "button",
            class: `level-btn${solved ? " is-solved" : ""}${active ? " is-active" : ""}`,
          },
          h("span", { class: "id", text: lv.id }),
          h("span", { class: "title", text: lv.title }),
          h("span", { class: "mark", text: solved ? "✓" : "" }),
        );
        btn.addEventListener("click", () => openLevel(lv.id));
        elRail.append(btn);
      }
    }
  }

  function renderBrief(): void {
    elBrief.replaceChildren();
    if (view.mode === "sandbox") {
      elBrief.append(
        h("div", { class: "brief-kicker", text: "SANDBOX" }),
        h("h1", { text: "Mess with the pipeline" }),
        h("p", {
          class: "brief-body",
          text: "Free play. Load a dataset, split, scale in the right order, fit, and score. Type help for the command map. Levels are where concepts get teeth.",
        }),
        h(
          "div",
          { class: "brief-goal" },
          h("span", { class: "goal-chip", text: "goal" }),
          "Build one honest pipeline end to end.",
        ),
      );
      return;
    }
    const lv = currentLevel(view);
    if (!lv) {
      return;
    }
    const nodes: (Node | string)[] = [
      h("div", { class: "brief-kicker", text: `${lv.worldTitle}  ·  ${lv.id}` }),
      h("h1", { text: lv.concept.title }),
      h("p", { class: "brief-body", text: lv.concept.body }),
    ];
    if (lv.concept.formula) {
      nodes.push(h("div", { class: "brief-formula mono", text: lv.concept.formula }));
    }
    if (lv.concept.callout) {
      nodes.push(h("div", { class: "brief-callout", text: lv.concept.callout }));
    }
    nodes.push(
      h("div", { class: "brief-goal" }, h("span", { class: "goal-chip", text: "goal" }), lv.goal),
    );
    if (view.winFeedback) {
      nodes.push(
        h("div", {
          class: `win-feedback ${view.winOk ? "is-win" : "is-fail"}`,
          text: view.winFeedback,
        }),
      );
    }
    elBrief.append(...nodes);
  }

  function renderSide(snap: SessionSnapshot): void {
    elSide.replaceChildren();

    const pipePanel = h("div", { class: "panel" }, h("h2", { text: "PIPELINE" }));
    const pipe = h("div", { class: "pipeline" });
    if (snap.steps.length === 0) {
      pipe.append(h("div", { class: "pipe-node" }, "no steps yet"));
    }
    snap.steps.forEach((step, i) => {
      const active = i === snap.steps.length - 1;
      pipe.append(
        h(
          "div",
          { class: `pipe-node is-${step.status}${active ? " is-active" : ""}` },
          h("span", { class: "dot" }),
          step.label,
        ),
      );
      if (step.detail) {
        pipe.append(h("div", { class: "pipe-detail", text: step.detail }));
      }
    });
    pipePanel.append(pipe);

    const metPanel = h("div", { class: "panel" }, h("h2", { text: "METRICS" }));
    const table = h("table", { class: "metrics-table" });
    const push = (k: string, v: string) => {
      const tr = h("tr");
      tr.append(h("td", { text: k }), h("td", { text: v }));
      table.append(tr);
    };
    if (snap.dataset) {
      push("n", String(snap.dataset.X.nRows));
      push("task", snap.dataset.task);
    }
    if (snap.split) {
      push("train/test", `${snap.split.XTrain.nRows}/${snap.split.XTest.nRows}`);
    }
    if (snap.scaled) {
      push("scale", snap.scaleLeaked ? `${snap.scaled} ⚠` : snap.scaled);
    }
    if (snap.model) {
      push("model", snap.model);
    }
    if (snap.trainMetrics) {
      if ("accuracy" in snap.trainMetrics) {
        push("train acc", snap.trainMetrics.accuracy.toFixed(3));
      } else {
        push("train r2", snap.trainMetrics.r2.toFixed(3));
      }
    }
    if (snap.metrics) {
      if ("accuracy" in snap.metrics) {
        push("test acc", snap.metrics.accuracy.toFixed(3));
        push("test f1", snap.metrics.f1.toFixed(3));
      } else {
        push("test r2", snap.metrics.r2.toFixed(3));
        push("test mae", snap.metrics.mae.toFixed(3));
      }
    }
    metPanel.append(table);

    const lv = currentLevel(view);
    if (view.mode === "level" && lv) {
      const hintPanel = h("div", { class: "panel" }, h("h2", { text: "HINTS" }));
      const ul = h("ul", { class: "hint-list" });
      for (const hint of lv.hints) {
        ul.append(h("li", { text: hint }));
      }
      hintPanel.append(ul);
      elSide.append(pipePanel, metPanel, hintPanel);
    } else {
      elSide.append(pipePanel, metPanel);
    }

    elViz.querySelector(".leak-banner")?.remove();
    if (snap.scaleLeaked) {
      elViz.prepend(
        h("div", {
          class: "leak-banner",
          text: "LEAKAGE · scaler fit before split — test statistics already in the model",
        }),
      );
    }
  }

  let raf = 0;
  function paint(): void {
    const snap = view.session.snapshot();
    const rect = elViz.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(320, Math.floor(rect.width));
    const ht = Math.max(220, Math.floor(rect.height));
    canvas.width = Math.floor(w * dpr);
    canvas.height = Math.floor(ht * dpr);
    canvas.style.width = `${w}px`;
    canvas.style.height = `${ht}px`;
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      return;
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    let scaledX = null;
    if (snap.dataset && snap.scaled) {
      try {
        const scaler =
          snap.split && !snap.scaleLeaked
            ? fitScaler(snap.split.XTrain, snap.scaled)
            : fitScaler(snap.dataset.X, snap.scaled);
        scaledX = transform(snap.dataset.X, scaler);
      } catch {
        scaledX = null;
      }
    }

    const state: DrawState = {
      dataset: snap.dataset,
      split: snap.split,
      model: view.liveModel,
      scaledX,
      kind: snap.fitted ? "fit" : "data",
      fitProgress: view.fitProgress,
    };
    drawStage(ctx, w, ht, state);
  }

  function schedulePaint(): void {
    if (raf) {
      return;
    }
    raf = requestAnimationFrame(() => {
      raf = 0;
      paint();
    });
  }

  function bloomFit(): void {
    view.fitProgress = 0;
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      view.fitProgress = 1;
      paint();
      return;
    }
    const t0 = performance.now();
    const dur = 420;
    const tick = () => {
      const t = Math.min(1, (performance.now() - t0) / dur);
      view.fitProgress = t;
      paint();
      if (t < 1) {
        requestAnimationFrame(tick);
      }
    };
    requestAnimationFrame(tick);
  }

  function renderAll(): void {
    renderRail();
    renderBrief();
    renderSide(view.session.snapshot());
    renderLog();
    schedulePaint();
  }

  function showWhat(what: string): string[] {
    const snap = view.session.snapshot();
    switch (what) {
      case "data": {
        if (!snap.dataset) {
          return ["No dataset. `load blobs`"];
        }
        const ds = snap.dataset;
        return [
          `dataset ${ds.name}  task=${ds.task}`,
          `X: n=${ds.X.nRows}  p=${ds.X.nCols}  features=[${ds.featureNames.join(", ")}]`,
          `y: ${ds.targetName}${ds.classNames ? `  classes=[${ds.classNames.join(", ")}]` : ""}`,
          `x1 range: ${colRange(ds.X, 0)}`,
          ...(ds.X.nCols > 1 ? [`x2 range: ${colRange(ds.X, 1)}`] : []),
          `y range: ${vecRange(ds.y)}`,
        ];
      }
      case "pipeline": {
        if (snap.steps.length === 0) {
          return ["pipeline empty"];
        }
        return snap.steps.map((s) => `  [${s.status}] ${s.label}`);
      }
      case "metrics": {
        const out: string[] = [];
        if (snap.trainMetrics) {
          out.push(...formatMetrics(snap.trainMetrics, "train"));
        }
        if (snap.metrics) {
          out.push(...formatMetrics(snap.metrics, "test"));
        }
        if (out.length === 0) {
          out.push("No scores yet. `score test`");
        }
        return out;
      }
      case "code":
      case "sklearn":
        return [
          "scikit-learn order of operations",
          "  1. train_test_split(X, y, test_size=0.2)",
          "  2. scaler.fit_transform(X_train); scaler.transform(X_test)",
          "  3. model.fit(X_train, y_train)",
          "  4. model.predict(X_test) / model.score(X_test, y_test)",
        ];
      default:
        return [`show ${what}: try data|pipeline|metrics|code`];
    }
  }

  function openLevel(id: string): void {
    const lv = findLevel(id);
    if (!lv) {
      view.lastLines.push({ kind: "err", text: `unknown level ${id}` });
      renderAll();
      return;
    }
    view.mode = "level";
    view.levelId = id;
    view.session = new Session();
    view.liveModel = null;
    view.fitProgress = 0;
    view.winFeedback = null;
    view.winOk = false;
    view.lastLines.push({ kind: "out", text: `— level ${lv.id}: ${lv.title} —` });
    const s = lv.seedDataset;
    if (s) {
      const r = view.session.load(s);
      view.lastLines.push(...r.lines.map((text) => ({ kind: "out" as const, text })));
    }
    btnLevels.classList.add("is-active");
    btnSandbox.classList.remove("is-active");
    renderAll();
  }

  function openSandbox(): void {
    view.mode = "sandbox";
    view.session = new Session();
    view.liveModel = null;
    view.fitProgress = 0;
    view.winFeedback = null;
    view.lastLines.push({ kind: "out", text: "— sandbox —" });
    btnSandbox.classList.add("is-active");
    btnLevels.classList.remove("is-active");
    renderAll();
  }

  function runCommand(raw: string): void {
    const line = raw.trim();
    if (!line) {
      return;
    }
    view.lastLines.push({ kind: "cmd", text: `$ ${line}` });
    try {
      const result = dispatch(view.session, line, {
        onShowLevels: () =>
          ALL_LEVELS.map((lv) => {
            const mark = view.solved.has(lv.id) ? "✓" : "·";
            return `  ${lv.id}  ${mark}  ${lv.title}`;
          }),
        onShowGoal: () => {
          const lv = currentLevel(view);
          if (!lv) {
            return ["Sandbox: build an honest pipeline (split → scale → fit → score test)."];
          }
          return [`[${lv.id}] ${lv.goal}`, `Concept: ${lv.concept.title}`];
        },
        onHint: () => {
          const lv = currentLevel(view);
          return [...(lv?.hints ?? ["No hints in sandbox."])];
        },
        onRunLevel: (id) => {
          openLevel(id);
          return [`Opened level ${id}.`];
        },
        onShow: (what) => showWhat(what),
      });
      for (const text of result.lines) {
        view.lastLines.push({ kind: "out", text });
      }
      if (result.sklearn) {
        view.lastLines.push({
          kind: "sk",
          text: `→ ${result.sklearn.split("\n")[0] ?? ""}`,
        });
        setSklearn(result.sklearn);
      }
      const snap = result.snapshot;
      if (!snap.fitted) {
        view.liveModel = null;
      } else {
        view.liveModel = rebuildModel(snap);
      }
      if (line.toLowerCase().startsWith("fit ")) {
        bloomFit();
      } else {
        view.fitProgress = snap.fitted ? 1 : 0;
      }
      maybeWin(view, result);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      view.lastLines.push({ kind: "err", text: `error: ${msg}` });
    }
    renderAll();
    input.value = "";
  }

  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      runCommand(input.value);
    }
  });

  btnHelp.addEventListener("click", () => runCommand("help"));
  btnLevels.addEventListener("click", () => openLevel(view.levelId));
  btnSandbox.addEventListener("click", openSandbox);

  window.addEventListener("resize", schedulePaint);
  setSklearn(null);
  renderAll();
  input.focus();
}

function colRange(X: { data: number[][] }, col: number): string {
  let lo = Infinity;
  let hi = -Infinity;
  for (const row of X.data) {
    const v = row[col] ?? 0;
    lo = Math.min(lo, v);
    hi = Math.max(hi, v);
  }
  return `[${lo.toFixed(2)}, ${hi.toFixed(2)}]`;
}

function vecRange(y: { data: number[] }): string {
  let lo = Infinity;
  let hi = -Infinity;
  for (const v of y.data) {
    lo = Math.min(lo, v);
    hi = Math.max(hi, v);
  }
  return `[${lo.toFixed(2)}, ${hi.toFixed(2)}]`;
}

const root = document.getElementById("app");
if (root) {
  mount(root);
}
