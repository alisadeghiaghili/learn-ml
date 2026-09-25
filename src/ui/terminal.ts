/**
 * Terminal view: log, history (↑/↓), word-by-word Tab completion, ghost suffix.
 * Empty input uses placeholder only so two texts never stack on the caret.
 */

export type LogKind = "cmd" | "out" | "err" | "meta" | "ok" | "sk";

export interface LogLine {
  kind: LogKind;
  text: string;
}

function escapeHtml(s: string): string {
  return s.replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");
}

const BASE_COMMANDS = [
  "load blobs",
  "load moons",
  "load noisy_line",
  "load outlier_line",
  "load scale_trap",
  "load poly_curve",
  "load mixed_table",
  "load clusters",
  "load dup_features",
  "show data",
  "show pipeline",
  "show metrics",
  "show code",
  "split test_size=0.2",
  "split test_size=0.25",
  "split test_size=0.3",
  "scale standard",
  "scale minmax",
  "fit linear",
  "fit logistic",
  "fit knn",
  "fit ridge",
  "fit dummy",
  "fit knn n_neighbors=1",
  "fit knn n_neighbors=5",
  "fit ridge alpha=1",
  "fit lasso alpha=0.2",
  "fit tree max_depth=1",
  "fit tree max_depth=6",
  "fit forest n_estimators=15 max_depth=4",
  "fit kmeans n_clusters=3",
  "fit pca_knn n_components=1",
  "fit pca_knn n_components=2",
  "fit boost n_estimators=30 max_depth=2 lr=0.3",
  "fit ovr_logistic",
  "fit elasticnet alpha=0.1 l1_ratio=0.5",
  "fit dbscan eps=0.8 min_samples=4",
  "impute mean",
  "encode onehot",
  "fe interact",
  "fe bin",
  "fe target",
  "infer",
  "calib",
  "predci",
  "importance",
  "sil",
  "nested ridge",
  "poly 3",
  "poly 12",
  "cv 5",
  "curve 5",
  "bootstrap 200",
  "pipeline",
  "save",
  "search ridge alpha=0.01,0.1,1,10",
  "residuals",
  "roc",
  "split test_size=0.25 strategy=stratified",
  "split test_size=0.3 strategy=time",
  "predict",
  "score train",
  "score test",
  "cm",
  "goal",
  "hint",
  "levels",
  "run 8.1",
  "run 9.1",
  "run 10.1",
  "reset",
  "undo",
  "clear",
  "help",
];

interface WordState {
  head: string[];
  current: string;
  afterSpace: boolean;
}

function parseLine(value: string): WordState {
  const endsWithSpace = /\s$/.test(value);
  const trimmed = value.replace(/\s+$/, "");
  if (!trimmed) {
    return { head: [], current: "", afterSpace: endsWithSpace };
  }
  const parts = trimmed.split(/\s+/);
  if (endsWithSpace) {
    return { head: parts, current: "", afterSpace: true };
  }
  return {
    head: parts.slice(0, -1),
    current: parts[parts.length - 1] ?? "",
    afterSpace: false,
  };
}

export class TerminalView {
  private logEl: HTMLElement;
  private inputEl: HTMLInputElement;
  private wrapEl: HTMLElement;
  private ghostEl: HTMLElement;
  private hintEl: HTMLElement;
  private lines: LogLine[] = [];
  private history: string[] = [];
  private historyIdx = -1;
  private draft = "";
  private hint = "";
  private extraCompletions: string[] = [];
  private wordCycle: string[] = [];
  private wordIdx = 0;
  private wordKey = "";
  private measureCtx: CanvasRenderingContext2D | null = null;
  private onSubmit: (cmd: string) => void;

  constructor(root: HTMLElement, onSubmit: (cmd: string) => void) {
    this.onSubmit = onSubmit;
    root.innerHTML = `
      <div class="term-log" id="term-log" role="log" aria-live="polite"></div>
      <div class="term-hint" id="term-hint" hidden></div>
      <div class="term-input-row">
        <label class="prompt" for="term-input">ml $</label>
        <div class="term-input-wrap" id="term-input-wrap">
          <div class="term-ghost" id="term-ghost" aria-hidden="true"></div>
          <input id="term-input" class="term-input" autocomplete="off" spellcheck="false"
            placeholder="" aria-label="Command input" />
        </div>
      </div>
    `;
    this.logEl = root.querySelector("#term-log")!;
    this.inputEl = root.querySelector("#term-input")!;
    this.wrapEl = root.querySelector("#term-input-wrap")!;
    this.ghostEl = root.querySelector("#term-ghost")!;
    this.hintEl = root.querySelector("#term-hint")!;
    this.inputEl.addEventListener("keydown", (e) => this.onKey(e));
    this.inputEl.addEventListener("input", () => this.syncGhost());
  }

  focus(): void {
    if (document.querySelector(".overlay .modal")) return;
    this.inputEl.focus();
    const len = this.inputEl.value.length;
    try {
      this.inputEl.setSelectionRange(len, len);
    } catch {
      // ignore
    }
  }

  setLog(lines: LogLine[]): void {
    this.lines = lines;
    this.render();
  }

  push(kind: LogKind, text: string): void {
    if (kind === "cmd" && text) {
      this.history.push(text);
      this.historyIdx = this.history.length;
    }
    this.lines.push({ kind, text });
    if (this.lines.length > 400) this.lines = this.lines.slice(-300);
    this.render();
  }

  clear(): void {
    this.lines = [];
    this.render();
  }

  private render(): void {
    const html = this.lines
      .map((l) => {
        const prefix = l.kind === "cmd" ? "$ " : "";
        return `<div class="${l.kind}">${prefix}${escapeHtml(l.text)}</div>`;
      })
      .join("");
    this.logEl.innerHTML = html;
    this.logEl.scrollTop = this.logEl.scrollHeight;
  }

  setHint(command: string | null): void {
    this.hint = command ?? "";
    this.inputEl.placeholder = this.hint
      ? `next: ${this.hint}`
      : "Type a command — help · levels · hint · goal";
    this.hintEl.hidden = !this.hint;
    if (this.hint) {
      this.hintEl.innerHTML = `Next: <code>${escapeHtml(this.hint)}</code> <span class="par-note">· Tab completes one word</span>`;
    } else {
      this.hintEl.textContent = "";
    }
    this.syncGhost();
  }

  setExtraCompletions(commands: string[]): void {
    this.extraCompletions = commands.filter(Boolean);
  }

  private allCompletions(): string[] {
    return [
      ...new Set<string>([
        ...this.extraCompletions,
        ...BASE_COMMANDS,
        ...this.history.slice().reverse(),
      ]),
    ];
  }

  private matchingCommands(head: string[], current: string): string[] {
    const cur = current.toLowerCase();
    return this.allCompletions().filter((cmd) => {
      const words = cmd.split(/\s+/);
      if (words.length <= head.length) {
        if (head.length && words.length === head.length) {
          return words.every((w, i) => w === head[i]);
        }
        return false;
      }
      for (let i = 0; i < head.length; i += 1) {
        if (words[i] !== head[i]) return false;
      }
      if (!cur) return true;
      return (words[head.length] ?? "").toLowerCase().startsWith(cur);
    });
  }

  private nextWords(head: string[], current: string): string[] {
    const matches = this.matchingCommands(head, current);
    const words: string[] = [];
    const push = (w: string | undefined) => {
      if (!w) return;
      if (!words.includes(w)) words.push(w);
    };
    if (this.hint) {
      const hw = this.hint.split(/\s+/);
      const okHead = head.every((h, i) => hw[i] === h);
      if (okHead) push(hw[head.length]);
    }
    for (const cmd of matches) {
      push(cmd.split(/\s+/)[head.length]);
    }
    return words.filter((w) => !current || w.toLowerCase().startsWith(current.toLowerCase()));
  }

  private measureText(text: string): number {
    if (!this.measureCtx) {
      this.measureCtx = document.createElement("canvas").getContext("2d");
    }
    const ctx = this.measureCtx;
    if (!ctx) return text.length * 7.2;
    const font = getComputedStyle(this.inputEl).font;
    ctx.font = font || "13px 'IBM Plex Mono', Consolas, monospace";
    return ctx.measureText(text).width;
  }

  /** Ghost shows only the rest of the current word (or the next word after a space). */
  private syncGhost(): void {
    const value = this.inputEl.value;
    this.ghostEl.dataset.visible = "0";
    this.ghostEl.textContent = "";
    this.wrapEl.classList.remove("has-ghost");

    if (!value) return;

    const { head, current, afterSpace } = parseLine(value);
    const words = this.nextWords(head, afterSpace ? "" : current);
    const first = words[0];
    if (!first) return;

    if (afterSpace) {
      this.ghostEl.textContent = first;
      this.ghostEl.style.left = `${this.measureText(value)}px`;
      this.ghostEl.dataset.visible = "1";
      this.wrapEl.classList.add("has-ghost");
      return;
    }

    if (!first.toLowerCase().startsWith(current.toLowerCase()) || first.length <= current.length) {
      return;
    }

    this.ghostEl.textContent = first.slice(current.length);
    this.ghostEl.style.left = `${this.measureText(value)}px`;
    this.ghostEl.dataset.visible = "1";
    this.wrapEl.classList.add("has-ghost");
  }

  /** Real-terminal Tab: complete the current word only; cycle on repeat. */
  private applyTab(e: KeyboardEvent): void {
    e.preventDefault();
    const value = this.inputEl.value;
    const { head, current, afterSpace } = parseLine(value);
    const cycleKey = `${head.join(" ")}|${afterSpace ? "" : current}`;

    if (!value && this.hint) {
      const firstWord = this.hint.split(/\s+/)[0] ?? "";
      this.inputEl.value = firstWord;
      this.wordCycle = [firstWord];
      this.wordIdx = 0;
      this.wordKey = firstWord;
      this.focus();
      this.syncGhost();
      return;
    }

    const options = this.nextWords(head, afterSpace ? "" : current);
    if (!options.length) {
      this.syncGhost();
      return;
    }

    if (cycleKey !== this.wordKey || !this.wordCycle.length) {
      this.wordKey = cycleKey;
      this.wordCycle = options;
      this.wordIdx = 0;
    } else {
      this.wordIdx = (this.wordIdx + 1) % this.wordCycle.length;
    }

    const chosen = this.wordCycle[this.wordIdx] ?? options[0] ?? "";
    const headText = head.length ? `${head.join(" ")} ` : "";
    // Complete ONE word only — never dump a full command line.
    this.inputEl.value = `${headText}${chosen}`;
    this.focus();
    this.syncGhost();

    if (this.wordCycle.length > 1) {
      const preview = this.wordCycle.slice(0, 6).join(" · ");
      this.hintEl.hidden = false;
      this.hintEl.innerHTML = `Tab word <strong>${this.wordIdx + 1}/${this.wordCycle.length}</strong>: <code>${escapeHtml(preview)}</code>${
        this.wordCycle.length > 6 ? " …" : ""
      }`;
    } else if (this.hint) {
      this.hintEl.innerHTML = `Next: <code>${escapeHtml(this.hint)}</code> <span class="par-note">· Tab completes one word</span>`;
    }
  }

  private onKey(e: KeyboardEvent): void {
    if (e.key === "Tab") {
      this.applyTab(e);
      return;
    }
    if (e.key === "Escape") {
      e.preventDefault();
      this.inputEl.value = "";
      this.wordCycle = [];
      this.wordKey = "";
      this.syncGhost();
      return;
    }
    if (e.key === "Enter") {
      e.preventDefault();
      if (document.querySelector(".overlay .modal")) return;
      const value = this.inputEl.value;
      this.inputEl.value = "";
      const trimmed = value.trim();
      if (trimmed) {
        this.history.push(trimmed);
        this.historyIdx = this.history.length;
      }
      this.wordCycle = [];
      this.wordKey = "";
      this.onSubmit(value);
      this.focus();
      this.syncGhost();
      return;
    }
    if (e.key === "ArrowUp") {
      e.preventDefault();
      if (!this.history.length) return;
      if (this.historyIdx === this.history.length) this.draft = this.inputEl.value;
      this.historyIdx = Math.max(0, this.historyIdx - 1);
      this.inputEl.value = this.history[this.historyIdx] ?? "";
      this.wordCycle = [];
      this.syncGhost();
      this.focus();
      return;
    }
    if (e.key === "ArrowDown") {
      e.preventDefault();
      if (!this.history.length) return;
      this.historyIdx = Math.min(this.history.length, this.historyIdx + 1);
      this.inputEl.value =
        this.historyIdx >= this.history.length ? this.draft : (this.history[this.historyIdx] ?? "");
      this.wordCycle = [];
      this.syncGhost();
      this.focus();
    }
  }
}

export { BASE_COMMANDS };
