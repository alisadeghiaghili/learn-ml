/**
 * LearnML shell: chrome, level rail, concept brief, stage, command dock,
 * goal checklist with current-step neon, and level-complete celebration.
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
import { TerminalView, type LogLine } from "./ui/terminal";
import { escapeHtml, renderMarkdown, showModal } from "./ui/dialog";
import { launchConfetti, playFanfare } from "./ui/confetti";
import {
  buildShareTargets,
  shareWithClipboard,
  COFFEE_BUTTON_HTML,
  REPO_URL,
  LIVE_URL,
} from "./ui/share";
import {
  loadProgress,
  saveProgress,
  summarizeCurriculum,
  resumeLine,
  type LevelProgress,
} from "./ui/progress";

type Mode = "level" | "sandbox";

const CHEERS = [
  "Locked in. That concept is yours now.",
  "Clean win. The pipeline told the truth.",
  "Good — you measured the future, not the homework.",
  "Nice. That is production hygiene, not notebook luck.",
];

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

function nextLevel(id: string): Level | undefined {
  const idx = ALL_LEVELS.findIndex((l) => l.id === id);
  return idx >= 0 ? ALL_LEVELS[idx + 1] : undefined;
}

export function mount(root: HTMLElement): void {
  const progress: Record<string, LevelProgress> = loadProgress();
  let mode: Mode = "level";
  let levelId = ALL_LEVELS[0]?.id ?? "1.1";
  let session = new Session();
  let liveModel: FittedModel | null = null;
  let fitProgress = 0;
  let winFeedback: string | null = null;
  let winOk = false;
  let celebratedFor = "";
  let lastCmdCount = 0;

  function currentLevel(): Level | null {
    return findLevel(levelId) ?? null;
  }

  function persistProgress(id: string, cmds: number): void {
    const prev = progress[id] ?? { solved: false };
    progress[id] = {
      solved: true,
      bestCommands:
        prev.bestCommands === undefined ? cmds : Math.min(prev.bestCommands, cmds),
      solvedAt: prev.solvedAt ?? new Date().toISOString(),
    };
    saveProgress(progress);
  }

  const elChrome = document.createElement("header");
  elChrome.className = "chrome";
  elChrome.innerHTML = `
    <div class="brand">Learn<span>ML</span></div>
    <nav class="chrome-nav">
      <button type="button" data-nav="levels" class="is-active">levels</button>
      <button type="button" data-nav="sandbox">sandbox</button>
      <button type="button" data-nav="help">help</button>
    </nav>
    <div class="chrome-meta">scikit-learn mental model · progress in cookie</div>
  `;

  const elBody = document.createElement("div");
  elBody.className = "body";
  const elRail = document.createElement("aside");
  elRail.className = "rail";
  const elMain = document.createElement("div");
  elMain.className = "main";
  const elBrief = document.createElement("section");
  elBrief.className = "brief";
  const elStage = document.createElement("section");
  elStage.className = "stage";
  const elViz = document.createElement("div");
  elViz.className = "stage-viz";
  const canvas = document.createElement("canvas");
  elViz.append(canvas);
  const elSide = document.createElement("div");
  elSide.className = "stage-side";
  elStage.append(elViz, elSide);
  elMain.append(elBrief, elStage);
  elBody.append(elRail, elMain);

  const elDock = document.createElement("footer");
  elDock.className = "dock";
  const termRoot = document.createElement("div");
  termRoot.className = "term-root";
  const elSk = document.createElement("div");
  elSk.className = "dock-sklearn";
  elDock.append(termRoot, elSk);

  root.append(elChrome, elBody, elDock);

  const terminal = new TerminalView(termRoot, (cmd) => runCommand(cmd));

  function setSklearn(text: string | null): void {
    elSk.innerHTML = text
      ? `<span class="lbl">scikit-learn · </span>${escapeHtml(text)}`
      : `<span class="lbl">scikit-learn · </span>ready`;
  }

  function pushLines(kind: LogLine["kind"], texts: string[]): void {
    for (const text of texts) {
      terminal.push(kind, text);
    }
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
      const head = document.createElement("div");
      head.className = "rail-world";
      head.textContent = label;
      elRail.append(head);
      for (const lv of levels) {
        const solved = Boolean(progress[lv.id]?.solved);
        const active = mode === "level" && levelId === lv.id;
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = `level-btn${solved ? " is-solved" : ""}${active ? " is-active" : ""}`;
        btn.innerHTML = `<span class="id">${escapeHtml(lv.id)}</span><span class="title">${escapeHtml(lv.title)}</span><span class="mark">${solved ? "✓" : ""}</span>`;
        btn.addEventListener("click", () => openLevel(lv.id));
        elRail.append(btn);
      }
    }
  }

  function renderSide(snap: SessionSnapshot): void {
    // Rebuild side fully so checklist current-step stays in sync.
    elSide.replaceChildren();

    if (mode === "level") {
      const lv = currentLevel();
      if (lv) {
        const stepHtml = (lv.steps ?? []).map((st) => {
          const met = st.check(snap);
          const firstUnmet = (lv.steps ?? []).find((s) => !s.check(snap));
          const current = firstUnmet?.id === st.id;
          const cls = met ? "met" : current ? "current" : "";
          return `<li class="${cls}">
            <div class="g-label">${escapeHtml(st.label)}</div>
            <div class="g-detail">${escapeHtml(st.detail)}</div>
            ${st.command ? `<code class="g-cmd">${escapeHtml(st.command)}</code>` : ""}
          </li>`;
        });
        const goalBox = document.createElement("div");
        goalBox.className = "panel goal-panel";
        goalBox.innerHTML = `<h2>CHECKLIST</h2><ol class="goal-list">${stepHtml.join("")}</ol>`;
        elSide.append(goalBox);
      }
    }

    const pipePanel = document.createElement("div");
    pipePanel.className = "panel";
    pipePanel.innerHTML = `<h2>PIPELINE</h2>`;
    const pipe = document.createElement("div");
    pipe.className = "pipeline";
    if (snap.steps.length === 0) {
      pipe.append(Object.assign(document.createElement("div"), { className: "pipe-node", textContent: "no steps yet" }));
    }
    snap.steps.forEach((step, i) => {
      const active = i === snap.steps.length - 1;
      const node = document.createElement("div");
      node.className = `pipe-node is-${step.status}${active ? " is-active" : ""}`;
      node.innerHTML = `<span class="dot"></span>${escapeHtml(step.label)}`;
      pipe.append(node);
      if (step.detail) {
        const d = document.createElement("div");
        d.className = "pipe-detail";
        d.textContent = step.detail;
        pipe.append(d);
      }
    });
    pipePanel.append(pipe);

    const metPanel = document.createElement("div");
    metPanel.className = "panel";
    metPanel.innerHTML = `<h2>METRICS</h2>`;
    const table = document.createElement("table");
    table.className = "metrics-table";
    const push = (k: string, v: string) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `<td>${escapeHtml(k)}</td><td>${escapeHtml(v)}</td>`;
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

    elSide.append(pipePanel, metPanel);

    const lv = currentLevel();
    if (mode === "level" && lv) {
      const hintPanel = document.createElement("div");
      hintPanel.className = "panel";
      hintPanel.innerHTML = `<h2>HINTS</h2>`;
      const ul = document.createElement("ul");
      ul.className = "hint-list";
      for (const hint of lv.hints) {
        const li = document.createElement("li");
        li.textContent = hint;
        ul.append(li);
      }
      hintPanel.append(ul);
      elSide.append(hintPanel);
    }

    elViz.querySelector(".leak-banner")?.remove();
    if (snap.scaleLeaked) {
      const ban = document.createElement("div");
      ban.className = "leak-banner";
      ban.textContent = "LEAKAGE · scaler fit before split — test statistics already in the model";
      elViz.prepend(ban);
    }
  }

  let raf = 0;
  function paint(): void {
    const snap = session.snapshot();
    const rect = elViz.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(320, Math.floor(rect.width));
    const ht = Math.max(220, Math.floor(rect.height));
    canvas.width = Math.floor(w * dpr);
    canvas.height = Math.floor(ht * dpr);
    canvas.style.width = `${w}px`;
    canvas.style.height = `${ht}px`;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
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
      model: liveModel,
      scaledX,
      kind: snap.fitted ? "fit" : "data",
      fitProgress,
    };
    drawStage(ctx, w, ht, state);
  }

  function schedulePaint(): void {
    if (raf) return;
    raf = requestAnimationFrame(() => {
      raf = 0;
      paint();
    });
  }

  function bloomFit(): void {
    fitProgress = 0;
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      fitProgress = 1;
      paint();
      return;
    }
    const t0 = performance.now();
    const dur = 420;
    const tick = () => {
      const t = Math.min(1, (performance.now() - t0) / dur);
      fitProgress = t;
      paint();
      if (t < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }

  function refreshHint(): void {
    const lv = currentLevel();
    if (mode !== "level" || !lv) {
      terminal.setHint(null);
      return;
    }
    const snap = session.snapshot();
    const next = (lv.steps ?? []).find((s) => !s.check(snap));
    terminal.setHint(next?.command ?? null);
    terminal.setExtraCompletions(
      (lv.steps ?? []).map((s) => s.command).filter((c): c is string => Boolean(c)),
    );
  }

  function renderAll(): void {
    const snap = session.snapshot();
    renderRail();
    renderSide(snap);
    // brief depends on side being built for checklist — render brief content only
    renderBriefContent(snap);
    refreshHint();
    schedulePaint();
  }

  function renderBriefContent(snap: SessionSnapshot): void {
    elBrief.replaceChildren();
    if (mode === "sandbox") {
      elBrief.innerHTML = `
        <div class="brief-kicker">SANDBOX</div>
        <h1>Mess with the pipeline</h1>
        <p class="brief-body">Free play. Load a dataset, split, scale in the right order, fit, and score. Type help for the command map. Levels are where concepts get teeth.</p>
        <div class="brief-goal"><span class="goal-chip">goal</span>Build one honest pipeline end to end.</div>
      `;
      return;
    }
    const lv = currentLevel();
    if (!lv) return;
    elBrief.innerHTML = `
      <div class="brief-kicker">${escapeHtml(lv.worldTitle)}  ·  ${escapeHtml(lv.id)}</div>
      <h1>${escapeHtml(lv.concept.title)}</h1>
      <p class="brief-body">${escapeHtml(lv.concept.body)}</p>
      <p class="brief-body"><strong>What happens.</strong> ${escapeHtml(lv.concept.whatHappens)}</p>
      <p class="brief-body"><strong>Why it matters.</strong> ${escapeHtml(lv.concept.why)}</p>
      ${lv.concept.formula ? `<div class="brief-formula mono">${escapeHtml(lv.concept.formula)}</div>` : ""}
      ${lv.concept.callout ? `<div class="brief-callout">${escapeHtml(lv.concept.callout)}</div>` : ""}
      <div class="brief-goal"><span class="goal-chip">goal</span>${escapeHtml(lv.goal)}</div>
      ${winFeedback ? `<div class="win-feedback ${winOk ? "is-win" : "is-fail"}">${escapeHtml(winFeedback)}</div>` : ""}
    `;
    void snap;
  }

  function showWhat(what: string): string[] {
    const snap = session.snapshot();
    switch (what) {
      case "data": {
        if (!snap.dataset) return ["No dataset. `load blobs`"];
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
        if (snap.steps.length === 0) return ["pipeline empty"];
        return snap.steps.map((s) => `  [${s.status}] ${s.label}`);
      }
      case "metrics": {
        const out: string[] = [];
        if (snap.trainMetrics) out.push(...formatMetrics(snap.trainMetrics, "train"));
        if (snap.metrics) out.push(...formatMetrics(snap.metrics, "test"));
        if (out.length === 0) out.push("No scores yet. `score test`");
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

  function maybeCelebrate(prevWon: boolean, result: CommandResult): void {
    const lv = currentLevel();
    if (!lv || mode !== "level") return;
    const wr = lv.win(result.snapshot);
    winFeedback = wr.feedback;
    winOk = wr.won;
    if (!wr.won) return;

    persistProgress(lv.id, session.snapshot().commandCount);
    if (celebratedFor === lv.id && prevWon) return;
    celebratedFor = lv.id;
    offerCelebration(lv);
  }

  function offerCelebration(level: Level): void {
    const cmds = session.snapshot().commandCount || null;
    const curriculum = summarizeCurriculum(progress);
    const share = buildShareTargets({
      levelName: level.title,
      levelId: level.id,
      commands: cmds,
      curriculum,
    });
    const next = nextLevel(level.id);
    const cheer = CHEERS[Math.floor(Math.random() * CHEERS.length)] ?? CHEERS[0]!;
    const learnedPreview = curriculum.learned
      .map((l) => `<li>${escapeHtml(l.worldTitle)}: ${escapeHtml(l.name)}</li>`)
      .join("");
    const bodyHtml = `
      <div class="celebrate" aria-live="polite">
        <div class="celebrate-visual" aria-hidden="true">
          <div class="celebrate-ring"></div>
          <div class="celebrate-star">★</div>
        </div>
        <div class="celebrate-badge">LEVEL CLEARED</div>
        <h3 class="celebrate-title">${escapeHtml(level.title)}</h3>
        <p class="celebrate-sub">${escapeHtml(level.worldTitle)} · <code>${escapeHtml(level.id)}</code></p>
        <p class="celebrate-cheer">${escapeHtml(cheer)}</p>
        <div class="celebrate-stats">${renderMarkdown(
          cmds ? `**${cmds}** commands this level.` : "Cleared.",
        )}</div>
        <div class="celebrate-progress">
          <div class="prog-track"><div class="prog-fill" style="width:${curriculum.percent}%"></div></div>
          <div class="par-note">${curriculum.solvedCount} / ${curriculum.total} levels solved · progress saved in this browser</div>
        </div>
        <div class="share-block">
          <div class="next-title">Share what you learned</div>
          <div class="learned-preview">
            <div class="par-note">In the post body:</div>
            <ul>${learnedPreview || `<li>Solve more levels to build the outline.</li>`}</ul>
          </div>
          <div class="share-row" role="group" aria-label="Share">
            <button type="button" class="share-btn linkedin" data-share="linkedin">LinkedIn</button>
            <button type="button" class="share-btn x" data-share="x">X</button>
            <button type="button" class="share-btn facebook" data-share="facebook">Facebook</button>
            <button type="button" class="share-btn copy" data-share="copy">Copy post</button>
          </div>
          <div class="share-status" data-share-status hidden></div>
        </div>
        <div class="celebrate-next">${renderMarkdown(
          next
            ? `Next: **${next.id} ${next.title}**`
            : "World cleared. Sandbox is still open for practice.",
        )}</div>
        <div class="celebrate-next">${COFFEE_BUTTON_HTML}</div>
      </div>
    `;

    const actions = [
      {
        label: "Bask in it",
        className: "ghost",
        onClick: () => terminal.focus(),
      },
    ];
    if (next) {
      actions.push({
        label: `Celebrate on ${next.id}`,
        className: "primary",
        onClick: () => openLevel(next.id),
      });
    } else {
      actions.push({
        label: "Browse levels",
        className: "primary",
        onClick: () => {
          mode = "level";
          renderAll();
          terminal.focus();
        },
      });
    }

    const confetti = launchConfetti(4800);
    playFanfare();

    const modal = showModal({
      title: "Level complete",
      bodyHtml,
      variant: "celebrate",
      actions: actions.map((a) => ({
        ...a,
        onClick: () => {
          confetti?.stop();
          modal.close();
          a.onClick();
        },
      })),
      onClose: () => {
        confetti?.stop();
        terminal.focus();
      },
    });

    modal.el.querySelectorAll<HTMLButtonElement>("[data-share]").forEach((btn) => {
      btn.addEventListener("click", async (ev) => {
        ev.preventDefault();
        const kind = (btn.dataset.share ?? "copy") as
          | "linkedin"
          | "facebook"
          | "x"
          | "copy";
        const status = modal.el.querySelector<HTMLElement>("[data-share-status]");
        const result = await shareWithClipboard(kind, share);
        if (!status) return;
        status.hidden = false;
        if (kind === "copy") {
          status.textContent = result.copied ? "Copied." : "Copy failed.";
          return;
        }
        status.textContent = result.copied
          ? "Share window opened. Post text also copied as fallback."
          : "Share window opened.";
      });
    });

    modal.el.querySelector(".modal")?.addEventListener("keydown", (ev) => {
      const key = (ev as KeyboardEvent).key;
      if (key === "Enter" || key === "Tab") {
        ev.preventDefault();
        ev.stopPropagation();
      }
    });
  }

  function openLevel(id: string): void {
    const lv = findLevel(id);
    if (!lv) {
      terminal.push("err", `unknown level ${id}`);
      renderAll();
      return;
    }
    mode = "level";
    levelId = id;
    session = new Session();
    liveModel = null;
    fitProgress = 0;
    winFeedback = null;
    winOk = false;
    celebratedFor = "";
    lastCmdCount = 0;
    terminal.push("out", `— level ${lv.id}: ${lv.title} —`);
    terminal.push("meta", lv.goal);
    const s = lv.seedDataset;
    if (s) {
      const r = session.load(s);
      pushLines("out", r.lines);
    }
    elChrome.querySelectorAll("[data-nav]").forEach((b) => {
      b.classList.toggle("is-active", (b as HTMLElement).dataset.nav === "levels");
    });
    renderAll();
    terminal.focus();
  }

  function openSandbox(): void {
    mode = "sandbox";
    session = new Session();
    liveModel = null;
    fitProgress = 0;
    winFeedback = null;
    celebratedFor = "";
    terminal.push("out", "— sandbox —");
    elChrome.querySelectorAll("[data-nav]").forEach((b) => {
      b.classList.toggle("is-active", (b as HTMLElement).dataset.nav === "sandbox");
    });
    renderAll();
    terminal.focus();
  }

  function runCommand(raw: string): void {
    const line = raw.trim();
    if (!line) return;
    terminal.push("cmd", line);
    const wasWon = winOk;
    try {
      const result = dispatch(session, line, {
        onShowLevels: () =>
          ALL_LEVELS.map((lv) => {
            const mark = progress[lv.id]?.solved ? "✓" : "·";
            return `  ${lv.id}  ${mark}  ${lv.title}`;
          }),
        onShowGoal: () => {
          const lv = currentLevel();
          if (!lv) {
            return ["Sandbox: build an honest pipeline (split → scale → fit → score test)."];
          }
          const snap = session.snapshot();
          const steps = (lv.steps ?? []).map((st) => {
            const mark = st.check(snap) ? "✓" : "·";
            return `  ${mark} ${st.label}`;
          });
          return [`[${lv.id}] ${lv.goal}`, ...steps];
        },
        onHint: () => {
          const lv = currentLevel();
          return [...(lv?.hints ?? ["No hints in sandbox."])];
        },
        onRunLevel: (id) => {
          openLevel(id);
          return [`Opened level ${id}.`];
        },
        onShow: (what) => showWhat(what),
      });
      pushLines("out", result.lines);
      if (result.sklearn) {
        terminal.push("sk", `→ ${result.sklearn.split("\n")[0] ?? ""}`);
        setSklearn(result.sklearn);
      }
      const snap = result.snapshot;
      if (!snap.fitted) {
        liveModel = null;
      } else {
        liveModel = rebuildModel(snap);
      }
      if (line.toLowerCase().startsWith("fit ")) {
        bloomFit();
      } else {
        fitProgress = snap.fitted ? 1 : 0;
      }
      maybeCelebrate(wasWon, result);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      terminal.push("err", `error: ${msg}`);
    }
    renderAll();
    terminal.focus();
    lastCmdCount = session.snapshot().commandCount;
    void lastCmdCount;
  }

  elChrome.querySelector("[data-nav='levels']")?.addEventListener("click", () =>
    openLevel(levelId),
  );
  elChrome.querySelector("[data-nav='sandbox']")?.addEventListener("click", openSandbox);
  elChrome.querySelector("[data-nav='help']")?.addEventListener("click", () =>
    runCommand("help"),
  );

  window.addEventListener("resize", schedulePaint);
  setSklearn(null);

  // Resume line when progress exists.
  const summary = summarizeCurriculum(progress);
  terminal.push("out", "LearnML · scikit-learn mental model sandbox");
  if (summary.solvedCount > 0) {
    pushLines("meta", resumeLine(summary).split("\n"));
  }
  openLevel(levelId);
  if (summary.solvedCount > 0 && summary.next) {
    terminal.push("meta", `Resume with \`run ${summary.next.id}\`.`);
  }
  void LIVE_URL;
  void REPO_URL;
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
