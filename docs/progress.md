# Progress

## Current phase
**Phase 6 — Doc-side romanization & normalization** (plan: docs/plans/06-doc-side-normalization.md);
started 2026-11-09 on branch `phase-6/doc-side-normalization`; done and review gate approved
(2026-11-11), not committed yet.

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

- [x] Phase 3 merged into `main` (2026-11-08)
- [x] Phase 4 step 1 (2026-11-08): evaluation harness, `src/eval/evaluate.py` (ir_measures
  metrics, per-query scores, results.csv rows) and `src/eval/significance.py` (paired t-test,
  Bonferroni); 13 tests with hand-computed values (69 in total)
- [x] Phase 4 step 2 (2026-11-09): BM25 index of all 506,264 passages (title + text, utf
  tokeniser, no stemming, no stopwords; 213 MB, built in 3.8 min); 4 tests (73 in total)
- [x] Phase 4 step 3 (2026-11-09): BM25 baseline runs on tune and test; queries tokenised
  before search, IDF floored at 0 and k1 = 1.2, b = 0.3 tuned on tune; test native
  nDCG@10 0.381 (MIRACL's published Lucene BM25: 0.458), all romanized sets ≈ 0
- [x] Phase 4 step 4 (2026-11-09): dense baseline with multilingual-e5-base; corpus encoded
  once (16.5 min, 778 MB cache); test native nDCG@10 0.607 (mE5 report: 58.4 on all of
  dev), romanized 0.000-0.013; 5 tests (84 in total)
- [x] Phase 4 step 5 (2026-11-09): script-gap table (`results/tables/gap_hi_test.md`);
  nDCG@10 drops 99.5-100% for BM25 and 97.8-100% for dense, all significant after
  Bonferroni; 4 tests (88 in total)
- [x] Phase 4 step 6 (2026-11-09): reproducibility check passed for both baselines: BM25
  with the index rebuilt from scratch and dense with a re-encoded shard give byte-identical
  run files and identical per-query scores (`scripts/check_reproducibility.py`)
- [x] Phase 4 step 7 (2026-11-09): "What we found" written: the script gap is 94.6-100%
  for BM25 and 85.9-100% for dense on every metric, matching the paper's 97% MRR@10
  drop for BGE-M3 on Chinese
- [x] Phase 4 review gate approved (2026-11-09): gap visible and matching the paper's trend;
  BM25 accepted below MIRACL's Lucene BM25 for documented reasons
- [x] Phase 4 merged into `main` (2026-11-09)
- [x] Phase 5 step 1 (2026-11-09): IndicXlit in a pinned Docker image; 10 scored candidates
  for each of the 2,441 romanized words (tune + test), ranking identical to IndicXlit's,
  frozen in `data/backtranslit/candidates_hi.tsv`; 3 tests (91 in total)
- [x] Phase 5 step 2 (2026-11-09): top-1 and weighted top-k query builders
  (`src/data/backtranslit.py`), top-1 query files for every romanized set on tune and test;
  9 tests (97 in total)
- [x] Phase 5 step 3 (2026-11-09): top-1 runs on tune: BM25 recovers 90.2% (natural) and
  82.3% (rule) of the nDCG@10 gap, dense 101.3% and 87.1%; test runs deferred to step 6;
  `gap_recovered` added (98 tests)
- [x] Phase 5 step 4 (2026-11-09): weighted top-k BM25 (`retrieve_weighted`, Terrier
  `term^weight`); k = 5 chosen on tune (tune nDCG@10 0.3656 natural, 0.3475 rule; the
  significant gain is recall); 3 tests (101 in total)
- [x] Phase 5 step 5 (2026-11-09): keeping English words in Latin tested on tune and
  rejected (BM25 natural nDCG@10 −0.120, p < 0.001; keeping both scripts −0.045,
  p = 0.003); 4 tests (105 in total)
- [x] Phase 5 step 6 (2026-11-09): test results in `results/tables/backtranslit_hi_test.md`
  (generic `scripts/method_table.py`): BM25 top-5 recovers 98.1% (natural), 88.3% (rule) and
  84.0% (human) of the nDCG@10 gap, dense top-1 92.3%, 80.9% and 88.2%; all significant
  against the romanized baselines after Bonferroni
- [x] Phase 5 step 7 (2026-11-09): failure analysis of 10 tune queries (5 are valid
  Devanagari spelling variants, e.g. आंख / आँख, both common in the corpus); "What we found"
  written
- [x] Phase 5 review gate approved (2026-11-09): back-transliteration recovers 81-98% of the
  gap; remaining failures are mostly Devanagari spelling variants (input to Phase 6)
- [x] Phase 5 merged into `main` (2026-11-09)
- [x] Phase 6 task list agreed (2026-11-09): a Devanagari normalization key added as task 1
  (plan change noted in docs/plans/06-doc-side-normalization.md)

- [x] Phase 6 task 1 (2026-11-09): Devanagari normalization key in `src/data/normalize.py`
  with six switchable rules (zero_width, nukta, chandrabindu, nasal, final_virama, digits;
  zero_width added after finding that Terrier splits words at a zero-width joiner); 29 tests
  (134 in total). `scripts/normalization_report.py`: all rules change 4.46% of the corpus's
  35.2M token occurrences and shrink the vocabulary by 5.62% (755,382 → 712,955); the
  largest merges are true variants (फिल्म / फ़िल्म, हिन्दी / हिंदी, गाँव / गांव); on tune the
  key resolves 28 of the 77 native words that top-1 back-transliteration misses

- [x] Phase 6 task 2 (2026-11-09): normalization in the index build, BM25 searchers and
  `tune_bm25.py` (rules recorded per index and checked against the queries; 3 tests, 137 in
  total). Leave-one-out ablation on tune (`scripts/tune_normalization.py`): the decision rule
  keeps chandrabindu, nasal, digits; tested together they beat no normalization on every tune
  set (nDCG@10 native 0.3954 → 0.4113, natural top-5 0.3656 → 0.3807, rule top-5 0.3475 →
  0.3607; p = 0.09-0.29, not significant). zero_width / final_virama / nukta "hurt" through the
  weighted query (they fold IndicXlit junk candidates such as `किस्‌` back into common question
  words), not through wrong merges. Index `data/indexes/hi_native_norm`; k1/b re-tuned on tune:
  1.2 / 0.3 as in Phase 4; tune runs `bm25_norm` and `bm25_bt5_norm` in results.csv (test runs
  in task 8)
- [x] Phase 6 task 3 (2026-11-09): `scripts/romanize_corpus.py` (Uroman as in Phase 3, optional
  worker processes) and `docs_file` support in `scripts/build_bm25_index.py`; 3 tests (140 in
  total). 1% sample (5,063 passages, seed 42): 20-21 ms per passage in one process, 4.7 ms with
  8 (byte-identical output), so the full corpus takes ≤ 40 min, ~0.23 GB of text and an index
  of ≤ ~0.5 GB; no subset needed

- [x] Phase 6 task 4 (2026-11-10): full corpus romanized with Uroman (25.8 min, 8 processes,
  238 MB, git-ignored `data/corpus_roman/`); romanized index `data/indexes/hi_roman` (207 MB);
  k1/b tuned on tune with the rule queries (natural ones give no signal): 1.2 / 0.3. Tune
  nDCG@10 on the romanized index: rule queries 0.4109 (same romanizer on both sides; above
  native 0.3954), natural queries 0.0078 (Uroman spells `upanissad`, users type `upanishad`)

- [x] Phase 6 task 5 (2026-11-11): romanized key `normalize_romanized` in `src/data/normalize.py`
  with 13 switchable rules derived from aligned natural/Uroman tune words; index records and
  checks romanized rules too; `scripts/roman_key_report.py` (word matches 32.8% → 90.1%,
  vocabulary −23.9%) and `scripts/tune_roman_normalization.py`; 184 tests. Leave-one-out kept
  11 rules (z_j, y_glide dropped). Index `data/indexes/hi_roman_norm`, k1/b 1.2 / 0.3; tune
  nDCG@10 natural 0.0078 → 0.3163, rule 0.4109 → 0.4203 (runs in results.csv)

- [x] Phase 6 task 6 (2026-11-11): `char_ngrams` and the shared `text_key` (Devanagari key →
  romanized key → n-grams) used by the index build and all searches; index records the n-gram
  size; 4 tests (188 in total). 3-, 4- and 5-gram indexes, k1/b tuned per index on natural tune
  queries (grid widened where the best was at its edge). Tune nDCG@10 natural: words 0.3163,
  3-grams 0.3089, **4-grams 0.3642** (R@1000 0.884), 5-grams 0.3561; rule queries on 4-grams
  0.4296. Chosen: 4-grams, k1 2.0, b 0.4 (runs in results.csv)

- [x] Phase 6 task 7 (2026-11-11): `src/retrieval/dual_index.py` (RRF, k = 60, checked against
  ranx) and `scripts/run_fusion.py`; 5 tests (193 in total). Dual index on tune (natural):
  nDCG@10 0.3792, level with the native-normalized route alone (0.3807, p = 0.94), but
  R@1000 0.9686 vs 0.9188 (p = 0.010); for rule queries the 4-gram route alone stays best

- [x] Phase 6 task 8 (2026-11-11): every Phase 6 method run on test once; tables
  `results/tables/docside_hi_tune.md`, `docside_hi_test.md`, `docside_hi_test_recall.md`
  (`scripts/method_table.py` gained a "p vs Phase 5" column). Test nDCG@10 natural: normalized
  native index + top-5 0.3870 (Phase 5 0.3737, p = 0.055); 4-gram romanized 0.3353; dual 0.3674
  with R@1000 0.9351 vs 0.8874 (p < 0.001); human set best with 4-grams (0.4145). "What we
  found" written in docs/plans/06-doc-side-normalization.md

- [x] Phase 6 review gate approved (2026-11-11): Devanagari rules chandrabindu, nasal, digits;
  11 romanized rules; the dual index helps recall, not top precision; carried into Phase 8: the
  dual index and top-5 back-transliteration on the normalized native index

## In progress
- (none)

## Next up
- Commit Phase 6 (when asked), then Phase 7 — dense fine-tuning: agree on the task list and files

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
- Query case → decided: queries go to the dense model as typed (off-the-shelf baseline =
  model + raw query); normalization is left to later methods, where it can be tested
- mE5 and MIRACL → note: multilingual-e5 was fine-tuned on MIRACL (mE5 report, Table 2),
  so the native dense baseline is in-domain; for Phase 7 only the romanized training
  queries are new to the model
- Terrier stemming and stopwords → decided: both off for every index and run (the English
  defaults change romanized Hindi, `roy` -> `roi`, and do nothing for Devanagari)
- Shallow judgments → decided: Judged@10 is logged with every run as a diagnostic
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
- Top-k expansion → decided: k = 5 candidates per word (tied with k = 10 on tune nDCG@10,
  better recall, fewer terms)
- English words in queries → decided: back-transliterated like any word; keeping them in
  Latin (or in both scripts) hurt on tune, since Hindi Wikipedia writes loanwords in Devanagari
- Spelling variants → note for Phase 6: most back-transliteration failures are valid
  Devanagari variants (anusvara / chandrabindu / half-nasal, २ / 2); a normalization key on
  both sides should address them
- Character n-grams → decided: 4-grams on the normalized romanized index, k1 2.0 / b 0.4 (best of
  words and 3-, 4-, 5-grams on tune, each with its own tuned k1/b)
- Romanized normalization rules → decided: ch, sh, w_v, ph_f, anusvara, long_vowels, doubled,
  ei_e, eh_ah, schwa, final_a (leave-one-out on tune, decision on natural nDCG@10); the key
  merges some different words (काम/कम), but it lifts retrieval on both query sets
- Devanagari normalization rules → decided: chandrabindu, nasal, digits (leave-one-out on tune
  with a decision rule fixed before the run: mean romanized nDCG@10); zero_width, nukta and
  final_virama dropped although their merges are correct, since with the weighted query they
  over-weight common question words
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
- Back-transliteration tool → decided: IndicXlit in Docker (Python 3.8, pinned image in
  `docker/indicxlit/`), run once and frozen; fairseq does not build on Windows/Python 3.11,
  no reliable HF substitute exists, and Docker Desktop was already installed
- Gujarati collection → open: decide in Phase 10
- Run files in git → decided: `results/runs/*.trec` are git-ignored (1-17 MB each, rebuilt
  from their configs); `results/results.csv` and `results/per_query/*.csv` are versioned
- BM25 setup → decided: queries tokenised with the index tokeniser, IDF floored at 0
  (terms in > half the passages dropped), k1 = 1.2 and b = 0.3 tuned on tune
- Hindi normalization for BM25 → partly addressed in Phase 6: the Devanagari key raises native
  BM25 test nDCG@10 from 0.381 to 0.404 (p = 0.003); still below MIRACL's Lucene BM25 with the
  Hindi analyzer (0.458), which also stems
- Demo stack → open: Streamlit or Gradio; decide in Phase 11

## Known issues
- Terrier drops terms longer than 20 characters (found 2026-11-10): romanization lengthens words
  (चमत्कारचिन्तामणि -> `camatkaaracintaamanni`), so 10,041 of 35.2M romanized passage tokens
  (0.029%) are not indexed and 2 passages become empty; no tune query and 1 test query token is
  that long. Left at Terrier's default (negligible)
- pytest prints "Windows fatal exception: access violation" stack dumps when a test starts
  Java (found 2026-11-09): the JVM raises and handles these itself and pytest's faulthandler
  reports them; the tests still pass. Harmless; faulthandler is left on to catch real crashes
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
- IndicXlit/fairseq do not install on Windows with Python 3.11 (confirmed 2026-11-09);
  solved with the Docker image in `docker/indicxlit/`, needed only to regenerate candidates
