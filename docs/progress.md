# Progress

## Current phase
**Phase 1 — Setup & groundwork: done** (review gate approved 2026-11-06). Phase 2 — Data
(docs/plans/02-data.md) is next and starts on our go-ahead.

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
- [x] Phase 1 task 1 (2026-10-27): reference paper setup and results in
  docs/paper_notes.md (Tables 1–2 checked number by number against the paper text)
- [x] Phase 1 task 2 (2026-11-06): pinned requirements.txt installed into `script-gap`;
  `pip check` clean, all libraries import, torch sees the RTX 4060
- [x] Phase 1 task 3 (2026-11-06): repo skeleton from SPECS §4 plus `tests/`; `src`
  subpackages import from the repo root
- [x] Phase 1 task 4 (2026-11-06): .gitignore covers data, models, checkpoints, indexes,
  embeddings, secrets and caches; checked with `git check-ignore` on 39 sample paths
  (top-level rules anchored with `/` so `src/data/` and `docs/data/` stay tracked)
- [x] Phase 1 task 5 (2026-11-06): src/utils `seed.py`, `config.py`, `runs.py` with 13
  pytest tests in tests/test_utils.py; all pass, ruff clean
- [x] Phase 1 task 6 (2026-11-06): `python -m scripts.check_env` passes 6/6 (Python
  3.11, pinned versions, imports, GPU fp16, Java 17 + Terrier 5.11, Hindi BM25 end to
  end); the Hindi check was confirmed to fail when the tokeniser drops Devanagari
- [x] Phase 1 task 7 (2026-11-06): `scripts/bench_dense_models.py` measured size,
  encoding speed and fine-tuning memory of e5-small, e5-base and BGE-M3 on the
  RTX 4060 (numbers in docs/plans/01-setup.md); dense model and metric cutoffs decided
- [x] Phase 1 merged into `main` (2026-11-06)
- [x] Phase 1 review gate approved (2026-11-06): evaluation setup agreed, dense model
  confirmed; "What we found" filled in docs/plans/01-setup.md

## In progress
- Nothing

## Next up
- Phase 2 — Data (docs/plans/02-data.md), on our go-ahead

## Open decisions / questions
- Languages → decided: Hindi core, Gujarati as a separate droppable phase (Phase 10,
  cut line Day 12)
- Plan layout → decided: docs/plans/00-overview.md + one file per phase
- Python → decided: conda env `script-gap` with Python 3.11 (a conservative version the
  whole core stack supports; keeps the project out of the base Python 3.13). Created from
  conda-forge with `--override-channels`, so no Anaconda channel terms were involved
- BM25 library → decided: PyTerrier (works with the installed Java 17)
- torch build → decided: 2.14.1+cu126 from the PyTorch index (the driver supports up to
  CUDA 13.1; 12.6 is the most widely supported of the offered builds)
- Pinning → decided: requirements.txt pins our direct dependencies plus the transitive
  ones that can change results (pyterrier, transformers, tokenizers, huggingface-hub,
  numpy, scipy); a full freeze would be ~150 lines of noise
- Import layout → decided: `src` is a package imported from the repo root
  (`python -m scripts.<name>`, `python -m pytest`); no pyproject or install step needed
- Run-file I/O → decided: read with `ir_measures.read_trec_run`, write our own (ties
  broken by docid, scores at full precision so files round-trip exactly)
- Lint rules → decided: ruff 0.17.3 defaults, no project config (the version is pinned,
  so everyone runs the same rules)
- Hindi collection → open: MIRACL-hi is the primary candidate. The paper itself used
  mMARCO/v2 (Chinese, Russian), so check whether mMARCO/v2 has Hindi and choose
  between them in Phase 2
- Natural romanization tool → open: decide in Phase 3. The paper romanized with Uroman
  (rule-based); candidate for our rule-based condition
- Starting dense model → decided: multilingual-e5-base as the main model (fine-tunes in
  ~5.3 GB, leaving headroom), e5-small for quick debugging runs, BGE-M3 off-the-shelf
  only as a reference point to the paper (its fine-tuning step needs 10.6 GB, more than
  the 8 GB card has)
- Metric cutoffs → decided: nDCG@10, MRR@10, R@100 (main) plus R@1000 (secondary, so
  recall is comparable with the paper, which reports R@1000); runs are therefore 1000
  deep (`DEFAULT_TOP_K = 1000` in src/utils/runs.py)
- Reproduction scope → decided: the paper's setup on Hindi (native vs romanized versions
  of the same queries against native documents), compared by pattern, not by number (our
  language, collection and model all differ from the paper's)
- Significance correction → decided: paired t-test with Bonferroni, as in the paper
- Paper's code repository → decided: consult it at the start of the phase that needs a
  missing detail (Uroman options in Phase 3, training setup in Phase 7), not upfront
- Fine-tuning setups → open: the paper compares native-only (N), 50/50 mixed (50) and
  romanized-only (T) training; consider mirroring all three in Phase 7
- Back-transliteration tool → open: IndicXlit first, fallback decided at the start of
  Phase 5
- Gujarati collection → open: decide in Phase 10
- Run files in git → open: `results/runs/*.trec` are currently tracked (text, and they
  back every number in results.csv); runs are now 1000 deep, so they are 10× larger than
  first planned. Check their real size in Phase 4 and decide then
- Demo stack → open: Streamlit or Gradio; decide in Phase 11

## Known issues
- GitHub ruleset `protect-main` exists but is NOT enforced: rulesets on a private repo
  need a paid plan (GitHub Pro via the Student Developer Pack, or a public repo)
- **Terrier tokeniser drops Devanagari by default** (found 2026-11-06): the default
  English tokeniser, and also the string `tokeniser="UTF"` (wrong case, silently
  ignored), produce an index with 0 terms and only a warning. Every Hindi/Gujarati
  index must use `tokeniser="utf"` (lowercase) or `pt.terrier.TerrierTokeniser.utf`;
  queries then use the same tokeniser automatically. scripts/check_env.py guards this
- Windows console is cp1252: printing Devanagari from Python crashes with
  `UnicodeEncodeError`; scripts that print Hindi need `sys.stdout.reconfigure(
  encoding="utf-8")` or should write to UTF-8 files instead (Phase 2)
- On Windows the JVM may keep Terrier index files open after `close()`, so deleting a
  temporary index can fail; harmless for the smoke test (cleanup errors are ignored)
- GPU memory overflow on Windows does not always raise OOM (found 2026-11-06): the
  driver can spill into system RAM instead, so a too-big job keeps running but many
  times slower (the BGE-M3 fine-tuning benchmark took ~36 min at 0% GPU use). Watch for
  this in Phase 7 training; the benchmark script now flags peaks above the GPU's memory
- IndicXlit depends on fairseq, which is known to be hard to install on Windows and may
  not support Python 3.11; it may need its own older env or WSL (Phase 5)
