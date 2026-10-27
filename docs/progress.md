# Progress

## Current phase
Planning is complete. **Phase 1 — Setup & groundwork** has not started (plan:
docs/plans/01-setup.md); it starts when the team says so.

## Completed
- [x] Project brief and conventions in CLAUDE.md
- [x] Commit rules: Claude commits locally only when asked, never pushes or opens PRs,
  and never appears as a contributor
- [x] Phase-wise plan approved and saved (2026-10-17): docs/plans/00-overview.md + one
  file per phase (01–11), each with goal, tasks, outputs, exit criteria and a review gate
- [x] Plans moved to docs/plans/; every written document now lives under docs/
- [x] Docs split (2026-10-27): CLAUDE.md slimmed down to working rules, full
  specification in docs/SPECS.md, progress tracked here
- [x] Initial commit on `main` (2026-10-27): CLAUDE.md + docs/ (one-time exception to
  "never commit on main", since the repo had no commits to branch from)

## In progress
- (nothing)

## Next up
- Before Phase 1: push `main`; provide the reference paper (PDF in a git-ignored
  folder, or an open-access copy); confirm the Python 3.11 env approach (conda)
- Phase 1 — Setup & groundwork: read the reference paper into docs/paper_notes.md,
  Python 3.11 env + pinned requirements.txt, repo skeleton + .gitignore, src/utils
  (seed, config, run-file I/O), environment smoke test

## Open decisions / questions
- Languages → decided: Hindi core, Gujarati as a separate droppable phase (Phase 10,
  cut line Day 12)
- Plan layout → decided: docs/plans/00-overview.md + one file per phase
- Python → decided: separate Python 3.11 env (system Python 3.13 is too new for several
  IR/transliteration libraries)
- BM25 library → decided: PyTerrier (works with the installed Java 17)
- Hindi collection → open: MIRACL-hi is the primary candidate; confirm against the
  paper in Phase 2
- Natural romanization tool → open: decide in Phase 3
- Starting dense model → open: multilingual-e5-small or -base; decide in Phase 1
- Back-transliteration tool → open: IndicXlit first, fallback decided at the start of
  Phase 5
- Gujarati collection → open: decide in Phase 10
- Demo stack → open: Streamlit or Gradio; decide in Phase 11

## Known issues
- GitHub ruleset `protect-main` exists but is NOT enforced: rulesets on a private repo
  need a paid plan (GitHub Pro via the Student Developer Pack, or a public repo)
- Python 3.11 is not installed (launcher has 3.12 and 3.14; default `python` is 3.13);
  conda 25.5.1 is available to create a 3.11 env
- IndicXlit depends on fairseq, which is known to be hard to install on Windows; may
  need WSL (Phase 5)
