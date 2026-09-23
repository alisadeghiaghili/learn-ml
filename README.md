# LearnML

Interactive machine learning sandbox and leveled tutorial in the spirit of
[learnGitBranching](https://github.com/pcottle/learnGitBranching).

**Live:** https://alisadeghiaghili.github.io/learn-ml/

LearnML teaches the **scikit-learn mental model** — not just which function to
call. Every level opens with a concept brief, the stage visualizes data and
model state, the pipeline graph shows order of operations (including leakage),
and each command prints the real scikit-learn Python equivalent.

## Why

Git is invisible until you draw commits. ML is invisible until you see
overfitting, leakage, and metric lies as *shapes*. LearnML makes those states
the game.

## Quick start

```bash
npm install
npm run dev      # http://localhost:5173
npm test
npm run build
```

## Modes

- **Levels** — World 1 (*Fit is not understanding*) has five challenges with
  honest win checks: X/y, optimization, train/test discipline, overfitting,
  preprocessing leakage. World 2 (*Metrics lie*) covers baselines, confusion
  matrices, and the train-vs-test discipline.
- **Sandbox** — free play with the same command surface.

## Command surface

```text
load blobs|moons|noisy_line|outlier_line|scale_trap
show data|pipeline|metrics|code
split test_size=0.2 seed=42
scale standard|minmax
fit linear|logistic|knn|ridge|dummy [params]
predict
score train|test
cm
goal | hint | levels | help | undo | reset
```

Example — the pipeline hygiene the levels enforce:

```text
$ load scale_trap
$ split test_size=0.2
$ scale standard
$ fit knn n_neighbors=5
$ score test
```

Under the dock you always see the scikit-learn form:

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)
```

## Pedagogy rules

1. Concept before code.
2. Win checks encode the idea (e.g. test accuracy ≥ threshold *and* no leakage),
   not a single magic command string.
3. Failure feedback names the misconception.
4. scikit-learn equivalence is always one glance away.

## Architecture

```text
src/engine   pure TypeScript sandbox (datasets, estimators, metrics, session)
src/ui       canvas charts + shell wiring
src/levels   level definitions and win predicates
tests/       vitest engine tests
```

`src/engine` never touches the DOM. Add a level by appending to
`src/levels/world1.ts` (or a new world module) with a `Level` object and a
`win(snapshot)` predicate.

## Scope

v1 is a **client-side simulation** of the scikit-learn workflow (fit/predict/
transform, train_test_split, StandardScaler, metrics). The dock teaches the
API you will use in production Python. A Pyodide / real-scikit-learn backend
is a deliberate later step, not a silent claim in the UI.

## License

MIT
