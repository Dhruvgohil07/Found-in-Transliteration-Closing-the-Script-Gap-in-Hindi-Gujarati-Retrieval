# Phase 2 — Data

**Status:** Not started · **Days:** 1–2

## Goal
Load a native-script Hindi test collection (documents, queries, relevance judgments), verify its real statistics, and create clean, fixed splits for tuning, testing and training.

## Inputs
- Phase 1 env and utilities
- Paper notes in `docs/paper_notes.md` (which datasets the paper used)

## Tasks
1. **Choose the Hindi collection.** Primary candidate is **MIRACL-hi** via `ir_datasets` (native Wikipedia passages, human-written queries and qrels). Confirm it matches or is comparable to the paper's setup.
2. **Write loaders** in `src/data/` that return docs, queries and qrels in one consistent format.
3. **Compute statistics from the data** (never copied or estimated):
   - number of documents and queries
   - judged / relevant docs per query
   - query and document lengths (tokens)
   Write them to `docs/data/data_card_hi.md`.
4. **Define splits:**
   - If only one labeled split has public qrels, carve a **fixed-seed tune/test split** from it.
   - Tuning (BM25 params, RRF k, expansion k, …) uses **only** the tune split.
5. **Identify training data** for Phase 7 (e.g. the MIRACL-hi train split), and check that no training query overlaps with tune/test.
6. Look at ~20 random queries with their relevant passages to get a feel for the data.

## Outputs
- `src/data/load.py` (or similar)
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
_(fill in at the end of the phase)_
