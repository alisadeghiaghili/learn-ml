# LearnML — Product & Design Spec

Interactive ML sandbox and leveled tutorial in the spirit of
[learnGitBranching](https://github.com/pcottle/learnGitBranching).
Curriculum and API vocabulary follow scikit-learn. The product teaches
**concepts first**, then the estimator code that expresses them.

## Product thesis

learnGitBranching wins because Git is invisible on a commit graph and a
terminal. Machine learning has the same problem: `fit` is a black box, data
leakage is silent, and train/test mistakes look like good scores. LearnML makes
those states visible and challengeable.

Three modes, same shell:

1. **Sandbox** — free command play, like LGB sandbox.
2. **Levels** — short challenges with a win condition and a concept brief.
3. **Concept labs** — parameter playgrounds (bias-variance, threshold, scale)
   where the lesson is watching a surface change.

Every user action is a pipeline step. The stage always shows:

- data space (scatter / residual)
- model state (boundary, line, clusters)
- evaluation panel
- pipeline graph (the analog of LGB’s commit tree)
- the scikit-learn Python equivalent of the last command

## Out of scope (v1)

- Real CPython / scikit-learn execution (Pyodide is a later backend).
- Deep learning, text, time series.
- User accounts, remote progress sync.

v1 ships a faithful **scikit-learn mental model** engine in TypeScript:
`fit` / `predict` / `transform` / `Pipeline` / `train_test_split` / metrics.
Code panels show the real Python API the learner will meet in production.

## Information architecture

```
Chrome:  LearnML · levels · sandbox · goal
Left:    level rail (worlds → levels, solved markers)
Center:  concept brief (paper) + visual stage (dark lab)
Bottom:  command dock + sklearn equivalent + status
```

### Level schema

```ts
interface Level {
  id: string;
  title: string;
  world: string;
  concept: ConceptBrief;      // why this matters
  goal: string;               // one sentence win condition
  hints: string[];
  win: (session: SessionState) => WinResult;
  seed?: {
    dataset?: DatasetName;
    prefit?: boolean;
  };
}
```

### Command grammar (maps 1:1 to sklearn)

```
help | levels | goal | hint | reset | undo | clear
load <dataset>
show data|pipeline|metrics|code
split test_size=0.2 [seed=42]
scale [standard|minmax]
fit <model> [param=value ...]
  models: linear | logistic | knn | ridge | dummy
predict
score
cm                         # confusion matrix
run <level-id>
```

Example equivalence taught in the dock:

```text
split test_size=0.2
→ X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```

## Pedagogy rules

1. Concept brief opens every level in plain language (no jargon dump).
2. Win checks encode the *idea*, not a magic command sequence
   (e.g. “test accuracy ≥ 0.8 **and** scaler fit on train only”).
3. Failure feedback is diagnostic: name the misconception, not “wrong”.
4. The sklearn line is always visible after a DSL command.
5. Golf counters (commands used) exist as a stretch goal, not the main score.

## World map (v1 content)

**World 1 — Fit is not understanding**

| id | concept | win sketch |
|----|---------|------------|
| 1.1 | X and y, features vs target | load + shape callouts |
| 1.2 | Fit = optimize loss, not magic | fit linear on clean data, see residual drop |
| 1.3 | Train/test split | split then fit; refuse score-on-train-only as win |
| 1.4 | Overfitting | high-capacity model beats train, loses test |
| 1.5 | Leakage | scale before split fails; scale-after-split wins |

**World 2 — Metrics lie**

| id | concept | win sketch |
|----|---------|------------|
| 2.1 | Baseline first | beat dummy on blobs with logistic |
| 2.2 | Confusion matrix / P-R | score + `cm` on moons |
| 2.3 | Train score is flattery | both train and test published |

## Visual system

**Style anchor:** laboratory instrument face + cool editorial notebook.
Not SaaS cards, not warm cream editorial, not neon-on-black.

**Palette**

| token | hex | role |
|-------|-----|------|
| void | `#0A1018` | stage / terminal ground |
| surface | `#13202C` | elevated dark panels |
| paper | `#E6E9E4` | concept brief (cool sage paper) |
| ink | `#0F1518` | text on paper |
| chalk | `#C5CED6` | text on dark |
| signal | `#F0B429` | primary accent — active / fitted |
| alarm | `#E23D51` | leakage, errors |
| calm | `#2A9D8F` | solved, valid |
| class0 | `#5B8DEF` | class / series A |
| class1 | `#F0B429` | class / series B |
| class2 | `#9B7EDE` | class / series C |

**Typography**

- Display / concept titles: `Newsreader`
- UI / body labels: `IBM Plex Sans`
- Code / terminal / metrics: `IBM Plex Mono`

Scale: concept title 28–32px / 500; brief body 16px / 1.55; UI 13–14px;
mono 12–13px. Max measure for brief text ≈ 62ch.

**Layout rhythm**

8px base. Shell is a full-viewport instrument panel:

```
┌──────────────────────────────────────────────────┐
│ top chrome (48px)                                │
├──────────┬───────────────────────────────────────┤
│ rail     │ brief (auto, paper)                   │
│ 220px    ├───────────────────────────────────────┤
│          │ stage (flex, void)  canvas + pipeline │
├──────────┴───────────────────────────────────────┤
│ dock (120px) command · sklearn equivalent        │
└──────────────────────────────────────────────────┘
```

**Signature moments**

1. **Fit bloom** — on successful `fit`, the decision surface / regression line
   draws in (~400ms) and the model node on the pipeline graph ignites `signal`.
2. **Leakage strike** — illegal order (e.g. `scale` on all data before `split`)
   paints the pipeline edge `alarm` and opens a short concept callout.

Motion respects `prefers-reduced-motion`. Focus rings are 2px `signal`.

## Architecture

```
src/engine   pure TS state machine (datasets, transformers, estimators,
             metrics, pipeline, command parser). No DOM.
src/ui       DOM + canvas renderers (terminal, stage, charts, pipeline graph).
src/levels   level definitions + win predicates.
tests/       engine unit tests (vitest).
```

Dependency inversion: UI depends on `engine` public API only. Levels depend on
`SessionState` read model. Engine never imports UI.

## Tech

- Vite + TypeScript (strict)
- Vitest for engine tests
- No UI framework — deliberate: terminal + canvas control matters more than
  component bookkeeping; keeps the bundle close to LGB’s simplicity.
- Fonts via `@fontsource` or system stack fallback chain.

## Success criteria for v1

1. Sandbox usable end-to-end for classification and regression.
2. World 1 (5 levels) completable with concept briefs and honest win checks.
3. Every command prints its scikit-learn equivalent.
4. `npm test` green; `npm run build` produces a static `dist/`.
5. README explains pedagogy and how to add a level.
