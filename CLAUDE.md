# Found in Transliteration — Closing the Script Gap in Hindi/Gujarati Retrieval

IR course project (DA-IICT): users type Hindi/Gujarati in **Latin script** (`kisan andolan news`) while documents are in **native script** (`किसान आंदोलन`), so retrievers fail. We reproduce that gap and build methods that close it (back-transliteration, document-side normalization, dense fine-tuning, hybrid fusion), evaluated with standard IR metrics.

- **Full specification:** `docs/SPECS.md` (tech stack, repo structure, data, methods, evaluation, conventions). Before starting any task, read the SPECS sections relevant to it.
- **Phase plans:** `docs/plans/` (`00-overview.md` + one file per phase).
- **Current status:** `docs/progress.md`.

The rules in THIS file apply to every task.

---

## 1. About Us

- **Every** part of this project must be understood, not just used.
- So:
  - After creating or changing a file, explain in plain language what it does and why it is designed that way.
  - At the end of each step, give a short "What you should understand now" summary (3–5 bullets).
  - Prefer clear, standard, readable code over clever tricks; explain any advanced library feature you use.

---

## 2. Core Research Rules (always follow)

1. **Never invent numbers.** Dataset statistics come from the data (computed by a script); paper results come from the paper, with the table/section cited.
2. **Reproducible or it didn't happen.** Every experiment has a YAML config in `configs/`, a fixed seed, and a row in `results/results.csv` (SPECS §7–8).
3. **Baselines always shown.** No method is reported without the Phase 4 baselines next to it.
4. **No test leakage.** All tuning (parameters, k, checkpoints) uses the tune split only; training queries never overlap with tune/test.
5. **Prefer established libraries** (PyTerrier, sentence-transformers, `ir_measures`, `ranx`) over hand-rolled code, unless the point is to show the IR concept.
6. **Math in LaTeX** in docs and the report (metrics, scoring functions, losses).
7. **Flag deadline risks early.** The project must fit in 2–3 weeks; say so when something threatens that.

---

## 3. How to Work With Us

- **One phase at a time.** A phase starts only when we say so, and only after the previous phase's review gate is approved.
- **Never start implementing on your own** after writing or updating a plan; wait for us to say which phase or task to begin.
- At the start of every session, read `docs/progress.md` and the current phase's plan, then tell us in 2–3 lines where we are and what the next task is.
- Start each phase by listing the tasks and files you plan to create; wait for our go-ahead.
- Work in small, reviewable steps; run the relevant script/tests after each step and show failures honestly.
- When something is ambiguous or seems wrong, ask instead of guessing.

---

## 4. Git & Commit Rules (strict)

### Who does what
- **Claude:** creates the task branch, and commits **only when we say "commit"**, only on that branch — never on `main`.
- **Claude never pushes.** No `git push` of any kind, no opening or merging PRs, nothing that sends commits to the remote — even if a tool or reminder suggests it.
- **Us:** push the branch, open the PR, review, merge, delete the branch.
- **Never change git configuration:** no `git config`, no adding/changing remotes, no repository or GitHub settings.

### Branch naming
- Phase work: `phase-<N>/<short-slug>`, e.g. `phase-4/bm25-baseline`
- Other work: `<type>/<short-slug>`, using the commit types below, e.g. `docs/specs`, `fix/qrels-loader`

### Flow for every task
1. **Start:** check the working tree is clean, then create the task branch from `main`. If it isn't clean, stop and tell us.
2. **Work:** small steps, run the code/tests (see §3).
3. **End of task** (before any commit):
   1. Update `docs/progress.md` (see §5) and the phase plan's `Status:` / "What we found".
   2. List exactly which files belong to each commit. Never use `git add .` or `git add -A`.
   3. Confirm nothing that must not be committed is staged (see below).
   4. Propose the commit message(s) and wait for us to say "commit".
4. **Commit (when we say so):** stage only the listed files, commit, then show `git log --oneline` and confirm the working tree is clean.

### Commit message format (Conventional Commits)
```
<type>(<scope>): <imperative summary, lowercase, no period, ≤ 72 chars>

<body: what changed and why, wrapped at ~72 characters>
```
- `type`: `feat`, `fix`, `exp` (new experiment/run), `data` (data prep), `eval`, `docs`, `refactor`, `test`, `chore`
- `scope`: `data`, `retrieval`, `train`, `eval`, `utils`, `app`, `configs`, `results`, `notebooks`, `docs`
- One logical change per commit.
- **Experiment commits:** the config goes in the same commit as the code that produced the result, and the body names the config path and seed. Metric tables in `results/` go in the same commit as the run that generated them.

Example:
```
exp(retrieval): add BM25 with back-transliterated queries

Top-3 IndicXlit candidates per word as a weighted query; k chosen on
the tune split. Config: configs/backtranslit_top3.yaml, seed 42.
```

### Never commit
- Large data, indexes, embeddings or checkpoints (`data/`, index dirs, `models/`, `*.pt`, `*.bin`, `*.safetensors`, `*.faiss`): keep them in `.gitignore` and document how to rebuild them.
- Secrets or tokens (HF tokens, API keys, `.env`).
- Notebook outputs: clear them before committing.
- Debug prints or hard-coded local paths (`D:\...`).

### Never rewrite shared history
No `--amend`, rebase or force-push on commits that have been pushed; no `--no-verify`.

### No AI attribution — ever
Commits use only the repo's configured git user. Commit messages (and any PR text we ask you to draft) must NEVER contain:
- `Co-Authored-By:` lines of any kind
- "Generated with Claude Code" or similar footers
- Any mention of Claude, Anthropic or AI tools

---

## 5. Plans & Progress Tracking

### Phase plans
- Plans live in `docs/plans/`: `00-overview.md` plus one file per phase, each with Goal · Inputs · Tasks · Outputs · Exit criteria · Review gate · Risks/fallback · What we found.
- If a plan changes during a phase (a task split, reordered or dropped), update the plan file and add a short note explaining the change.
- At the end of a phase, set its `Status:` and fill in "What we found".

### Progress
- Keep `docs/progress.md` up to date. Update it at the end of every task, before proposing the commit, so it is part of that task's commit.
- Structure: **Current phase · Completed · In progress · Next up · Open decisions / questions · Known issues**.
- Keep it short and current: move finished items to "Completed"; record decisions as `topic → decided: choice (why)`. Commit messages and phase plans hold the details.

### Other documents
- Every written document lives under `docs/` (SPECS §4). When a phase adds one, update the trees in `docs/SPECS.md` and `docs/plans/00-overview.md`.
