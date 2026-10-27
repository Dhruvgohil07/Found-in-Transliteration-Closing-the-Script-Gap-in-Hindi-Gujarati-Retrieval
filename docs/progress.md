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
- [x] `main` pushed to GitHub
- [x] Reference paper available locally at `papers/lost-in-transliteration.pdf`;
  `papers/` is git-ignored (new `.gitignore`, extended in Phase 1)
- [x] Conda env `script-gap` created (2026-10-27): Python 3.11.18 + pip from
  conda-forge; no project packages installed yet

## In progress
- (nothing)

## Next up
- Phase 1 — Setup & groundwork: read the reference paper into docs/paper_notes.md,
  pinned requirements.txt installed into `script-gap`, repo skeleton + .gitignore,
  src/utils (seed, config, run-file I/O), environment smoke test

## Open decisions / questions
- Languages → decided: Hindi core, Gujarati as a separate droppable phase (Phase 10,
  cut line Day 12)
- Plan layout → decided: docs/plans/00-overview.md + one file per phase
- Python → decided: conda env `script-gap` with Python 3.11 (a conservative version the
  whole core stack supports; keeps the project out of the base Python 3.13). Created from
  conda-forge with `--override-channels`, so no Anaconda channel terms were involved
- BM25 library → decided: PyTerrier (works with the installed Java 17)
- Hindi collection → open: MIRACL-hi is the primary candidate. The paper itself used
  mMARCO/v2 (Chinese, Russian), so check whether mMARCO/v2 has Hindi and choose
  between them in Phase 2
- Natural romanization tool → open: decide in Phase 3. The paper romanized with Uroman
  (rule-based); candidate for our rule-based condition
- Starting dense model → open: decide in Phase 1. The paper used BGE-M3 (single-vector)
  with an mT5 reranker; our plan has multilingual-e5-small/-base for the 8 GB GPU
- Metric cutoffs → open: the paper reports MRR@10 + R@1000 (mMARCO) and nDCG@20 +
  R@1000 (neuCLIR); our plan has nDCG@10, MRR@10, R@100. Decide in Phase 1
- Fine-tuning setups → open: the paper compares native-only (N), 50/50 mixed (50) and
  romanized-only (T) training; consider mirroring all three in Phase 7
- Back-transliteration tool → open: IndicXlit first, fallback decided at the start of
  Phase 5
- Gujarati collection → open: decide in Phase 10
- Demo stack → open: Streamlit or Gradio; decide in Phase 11

## Known issues
- GitHub ruleset `protect-main` exists but is NOT enforced: rulesets on a private repo
  need a paid plan (GitHub Pro via the Student Developer Pack, or a public repo)
- IndicXlit depends on fairseq, which is known to be hard to install on Windows and may
  not support Python 3.11; it may need its own older env or WSL (Phase 5)
