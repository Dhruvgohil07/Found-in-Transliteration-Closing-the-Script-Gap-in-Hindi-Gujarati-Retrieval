# Progress

## Current phase
**Phase 4 — Baselines & eval harness** (plan: docs/plans/04-baselines.md); not started.
Phase 3 is done (review gate approved 2026-11-08) on branch `phase-3/romanized-queries`,
not merged yet.

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
- [x] Git workflow rules added to CLAUDE.md §4 (2026-11-06): after "merged", sync `main`
  and delete local branches whose commits are all in `main`; commit less often, with at
  most one commit per type per batch and the changes listed point by point
- [x] Phase 2 task 1 (2026-11-06): collection chosen (MIRACL-hi) and split sizes fixed
  (100 tune / 250 test from the 350 dev queries, seed 42)
- [x] Phase 2 task 2 (2026-11-06): `src/data/load.py` loads MIRACL-hi end to end (350 dev
  + 1,169 train + 819 test-b queries, 506,264 passages, in 76 s); `src/data/splits.py`
  for fixed-seed splits and the overlap check; 12 new tests (25 in total), ruff clean
- [x] Phase 2 tasks 3–6 (2026-11-06): `scripts/prepare_data.py` writes the tune/test/train
  files (100 / 250 / 1,168 queries; byte-identical on re-run); one train query dropped as
  a text duplicate of a test query; `scripts/data_stats.py` fills the statistics in
  docs/data/data_card_hi.md; `scripts/sample_queries.py` writes 20 tune queries with
  their relevant passages to `data/processed/sample_hi.md`
- [x] Phase 2 merged into `main` (2026-11-06)
- [x] Phase 2 review gate approved (2026-11-07): MIRACL-hi is a good fit and 250 test
  queries are enough; limits recorded in docs/plans/02-data.md
- [x] Phase 3 tasks 1–4, code (2026-11-07): `src/data/romanize.py` (Uroman, Gemini
  wrapper, edit distance, CER), `scripts/romanize_queries.py`,
  `scripts/human_romanization.py`, `scripts/romanization_quality.py`,
  `configs/romanize_hi.yaml`, `.env` loading; 30 new tests (56 in total), ruff clean
- [x] Phase 3 rule-based set (2026-11-07): Uroman files for tune/test/train written to
  `data/romanized/` and frozen; the hand-romanization sheet (50 test queries) made
- [x] Phase 3 natural set (2026-11-07): Gemini (`gemini-3.5-flash`, answered by
  `gemini-3.6-flash`) romanized all 1,518 queries over two days (free-tier quota);
  files for tune/test/train frozen in `data/romanized/`; sanity checks and the 30-query
  review file from `scripts/romanization_quality.py`
- [x] Phase 3 hand-romanized set (2026-11-08): 50 test queries typed once (version A),
  frozen in `data/romanized/test_queries_roman_human_a.tsv`; CER against it: natural
  0.096, rule 0.236; "What we found" filled in docs/plans/03-romanized-queries.md
- [x] Phase 3 review gate approved (2026-11-08): the natural set is the main condition;
  points carried forward are listed in docs/plans/03-romanized-queries.md

## In progress
- (none)

## Next up
- Commit and merge Phase 3, then Phase 4 — Baselines & eval harness: agree on the task
  list and files (open decisions for Phase 4 are listed below)

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
- Hindi collection → decided: MIRACL-hi (native-written queries and passages, about 10
  judgments per query, 506K passages). `mmarco/v2/hi` (the paper's collection) exists in
  ir_datasets but has 8.8M machine-translated passages and about 1 judgment per query
- Tune/test split → decided: 100 tune / 250 test from MIRACL-hi dev (350 queries, the
  only split with public qrels), seed 42
- Raw data location → decided: the ir_datasets cache (`~/.ir_datasets`), outside the
  repo; `data/processed/` holds our own split files
- Train/test leakage → decided: train queries that repeat a tune/test query (same id or
  same normalized text) are dropped from train, so the evaluation splits stay intact
- Query case → open, decide in Phase 4: Terrier lower-cases queries and documents itself,
  but the e5 tokenizer is case-sensitive (`Raja` and `raja` become different tokens), and
  the human set keeps the capitals as typed. Candidate: lower-case every query set the same
  way before dense retrieval (a fixed preprocessing step, not tuned on test)
- Terrier stemming and stopwords → open, decide in Phase 4: the default pipeline applies the
  English Porter stemmer to Latin-script words (`roy` was indexed as `roi`); for romanized
  Hindi it should probably be switched off (`stemmer=None, stopwords=None`), for every run
- Shallow judgments → open: about 10 judged passages per query, so report Judged@10
  (`ir_measures`) next to the main metrics in Phase 4; decide there
- Phase 7 query length → open: train queries reach 41 e5 tokens, so the 32-token query
  length from the Phase 1 benchmark is too short; plan for about 48 in Phase 7
- Natural romanization → decided: Gemini API (`google-genai`), fixed prompt, temperature 0,
  output frozen (IndicXlit ruled out: Phase 5 uses it to undo romanization)
- Rule-based romanization → decided: Uroman, called as in the paper's code (`lang=hin`,
  raw text), instead of ITRANS/ISO
- Hand-romanized set → decided: 50 test queries (seed 42), one version (A); version B
  dropped, so no human-vs-human CER
- Secrets → decided: API keys go in a git-ignored `.env` at the repo root, read by
  `load_env_file()` in src/utils/config.py (conda env variables only apply after
  `conda activate`, which direct calls to the env's python skip)
- Known LLM errors → decided: kept in the frozen natural set, not hand-corrected (2 inserted
  words and 1 dropped word found in 1,518 queries; listed in docs/plans/03)
- Romanized files in git → decided: `data/romanized/` is versioned (small text; the LLM
  and hand-typed sets cannot be regenerated exactly); the rest of `data/` stays ignored
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
- Gemini free tier allows only 20 requests per day per model (found 2026-11-07); with
  25 queries per request the natural set needs 61 requests
- GitHub ruleset `protect-main` exists but is NOT enforced: rulesets on a private repo
  need a paid plan (GitHub Pro via the Student Developer Pack, or a public repo)
- **Terrier tokeniser drops Devanagari by default** (found 2026-11-06): the default
  English tokeniser, and also the string `tokeniser="UTF"` (wrong case, silently
  ignored), produce an index with 0 terms and only a warning. Every Hindi/Gujarati
  index must use `tokeniser="utf"` (lowercase) or `pt.terrier.TerrierTokeniser.utf`;
  queries then use the same tokeniser automatically. scripts/check_env.py guards this
- **Python must run in UTF-8 mode on Windows** (found 2026-11-06): ir_datasets opens
  some files without naming an encoding, so reading MIRACL-hi queries under the default
  cp1252 fails with `UnicodeDecodeError`. Fixed: `PYTHONUTF8=1` is set as a conda env
  variable of `script-gap` (active after `conda activate`; setup line added to
  requirements.txt); this also fixes printing Devanagari to the console. The loaders
  and `scripts/check_env.py` (now 7 checks) stop with the fix in the message if it is off
- On Windows the JVM may keep Terrier index files open after `close()`, so deleting a
  temporary index can fail; harmless for the smoke test (cleanup errors are ignored)
- GPU memory overflow on Windows does not always raise OOM (found 2026-11-06): the
  driver can spill into system RAM instead, so a too-big job keeps running but many
  times slower (the BGE-M3 fine-tuning benchmark took ~36 min at 0% GPU use). Watch for
  this in Phase 7 training; the benchmark script now flags peaks above the GPU's memory
- IndicXlit depends on fairseq, which is known to be hard to install on Windows and may
  not support Python 3.11; it may need its own older env or WSL (Phase 5)
