# Phase 2 — Data

**Status:** In review · **Days:** 1–2

## Goal
Load a native-script Hindi test collection (documents, queries, relevance judgments), verify its real statistics, and create clean, fixed splits for tuning, testing and training.

## Inputs
- Phase 1 env and utilities
- Paper notes in `docs/paper_notes.md` (which datasets the paper used)

## Tasks
1. **Choose the Hindi collection.** Primary candidate is **MIRACL-hi** via `ir_datasets` (native Wikipedia passages, human-written queries and qrels). Confirm it matches or is comparable to the paper's setup.
   _Decided (2026-11-06): **MIRACL-hi**. The `ir_datasets` registry also has `mmarco/v2/hi`, the paper's own collection in Hindi. Its bundled metadata lists 8,841,823 passages (vs 506,264 for MIRACL-hi), about 1 judgment per dev/small query, and machine-translated text. MIRACL-hi was chosen for native-written queries and passages, deeper judgments and a laptop-sized corpus; this fits the Phase 1 decision to reproduce the paper's setup, not its numbers. Only MIRACL-hi `dev` has public qrels (`test-b` has none), so tune/test are carved from its 350 dev queries: **100 tune / 250 test, seed 42**. Counts here are from `ir_datasets` metadata; the data card holds our own computed ones._
2. **Write loaders** in `src/data/` that return docs, queries and qrels in one consistent format.
3. **Compute statistics from the data** (never copied or estimated):
   - number of documents and queries
   - judged / relevant docs per query
   - query and document lengths (tokens)
   Write them to `docs/data/data_card_hi.md`.
   _Change (2026-11-06): done after tasks 4–5, so the card reports statistics per split. `scripts/data_stats.py` writes the statistics block of the card itself._
4. **Define splits:**
   - If only one labeled split has public qrels, carve a **fixed-seed tune/test split** from it.
   - Tuning (BM25 params, RRF k, expansion k, …) uses **only** the tune split.
5. **Identify training data** for Phase 7 (e.g. the MIRACL-hi train split), and check that no training query overlaps with tune/test.
   _Change (2026-11-06): the check compares query texts as well as ids, and it found one leak in MIRACL itself: train `677356#0` repeats the text of dev `1270408#0` (in test), sharing 2 relevant passages. Train queries that repeat a tune/test query are now dropped from train by `scripts/prepare_data.py`, leaving 1,168 train queries._
6. Look at ~20 random queries with their relevant passages to get a feel for the data.
   _Change (2026-11-06): sampled from the tune split only, never test (`scripts/sample_queries.py`, written to the git-ignored `data/processed/sample_hi.md`)._

## Outputs
- `src/data/load.py` (loaders and processed-file I/O), `src/data/splits.py` (splits and leakage checks), `tests/test_data.py`
- `scripts/prepare_data.py`, `scripts/data_stats.py`, `scripts/sample_queries.py`
- `data/processed/{tune,test,train}_queries.tsv`, `data/processed/{tune,test,train}_qrels.tsv`
- `docs/data/data_card_hi.md` (source, licence, statistics, split seed)
- `configs/data_hi.yaml`

## Exit criteria
- Loaders run end-to-end; statistics in the data card are computed by a script.
- Splits are saved, the split seed is recorded, and the train/test overlap check passes.

## Review gate
- Review the data card and the sample queries.
- Is the collection a good fit (enough queries, enough relevant docs per query)?

## Risks / fallback
- Corpus too large for the laptop → use a documented subset that keeps all judged passages plus a fixed-seed sample of the rest.
- MIRACL-hi doesn't fit → fall back to the dataset the paper used, or mMARCO-hi / IndicMSMARCO-hi (verify before choosing).

## What we found

**Exit criteria: met.** The loaders run end to end; every statistic in `docs/data/data_card_hi.md` is written by `scripts/data_stats.py`; the splits are saved with seed 42 (two runs give byte-identical files) and the overlap check passes.

**Numbers (all from `docs/data/data_card_hi.md`)**
- 506,264 passages from 148,107 articles, the same counts as the MIRACL dataset card.
- Splits: 100 tune / 250 test / 1,168 train queries, with about 10 judged and 2 relevant passages per query; every query has at least one relevant passage, and every judged passage is in the corpus.

**Surprises**
- **ir_datasets cannot read Hindi on Windows by default:** it opens files without naming an encoding (cp1252). Fixed by running Python in UTF-8 mode (`PYTHONUTF8=1`, now set in the `script-gap` env); the loaders and `scripts/check_env.py` check it.
- **MIRACL leaks one query between dev and train:** the same question appears as dev `1270408#0` (in our test split) and train `677356#0`, sharing 2 relevant passages. Only a text comparison finds it; the ids differ. The train copy is dropped.
- **7.7% of passages are longer than 256 e5 tokens** (95th percentile 301), so the dense model truncates them at the planned length.
- **Queries reach 41 e5 tokens** (train), more than the 32 tokens used in the Phase 1 memory benchmark; Phase 7 needs a longer query length (about 48).
- **The native queries are almost all pure Devanagari:** only 9 of 1,518 contain Latin letters, so the romanized query sets in Phase 3 are what create the script gap.

**Decisions**
- Collection: MIRACL-hi (task 1 note). Split: 100 tune / 250 test from dev, seed 42.
- Train queries that repeat a tune/test query (by id or normalized text) are dropped from train.
- Raw data stays in the ir_datasets cache; `data/processed/` holds our split files.
