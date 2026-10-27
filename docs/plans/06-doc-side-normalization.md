# Phase 6 — Document-side romanization & phonetic normalization

**Status:** Not started · **Days:** 6–7

## Goal
Close the gap on the **document side**: also index the documents in Latin script, and make matching robust to spelling variation using a phonetic normalization key and character n-grams.

## Inputs
- Phase 2 corpus
- Phase 3 romanized query sets
- Phase 5 back-transliterated queries (for the dual-index run)

## Tasks
1. **Time it first:** romanize and index a 1% sample of the corpus to estimate time and disk for the full corpus.
2. **Romanize the corpus** with the fast rule-based romanizer, and build a **second BM25 index** (Latin script).
3. **Phonetic / spelling normalization key**, applied identically to queries and documents. For example:
   - lowercase; collapse long vowels (`aa→a`, `ee/ii→i`, `oo/uu→u`)
   - merge common variants (`w/v`, `z/j`, `ph/f`)
   - drop the final inherent vowel (schwa: `kisana→kisan`)
   - de-duplicate repeated letters
   Run BM25 on the normalized index with normalized queries.
4. **Character n-gram tokenization** (e.g. 3–4-grams) on the romanized index, so near-miss spellings still share tokens.
5. **Dual-index retrieval:** combine
   - native index ← back-transliterated query (Phase 5)
   - romanized/normalized index ← raw romanized query
   using a simple fusion (RRF; full fusion study is Phase 8).
6. Ablation: raw romanized index vs + normalization vs + n-grams, on the tune split.

## Outputs
- `src/data/normalize.py` (normalization key, unit-tested on examples)
- `src/retrieval/dual_index.py`
- `configs/docside_*.yaml`
- New rows in `results/results.csv` + ablation table in "What we found"

## Exit criteria
- Romanized index built (full corpus, or a documented subset).
- Ablation shows the effect of each step; dual-index result reported next to the baselines and Phase 5.

## Review gate
- Which normalization rules help, which hurt? Keep only those that help on the tune split.
- Is the dual index better than Phase 5 alone?

## Risks / fallback
- Full corpus too slow/large → documented fixed-seed subset that keeps all judged passages (same subset for **all** methods in the comparison).
- Normalization over-merges different words → keep the rule list small and justified.

## What we found
_(fill in at the end of the phase)_
