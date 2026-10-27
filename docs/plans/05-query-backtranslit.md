# Phase 5 — Query-side back-transliteration

**Status:** Not started · **Days:** 5–6

## Goal
Close the gap on the **query side**: convert romanized queries back into native script (Latin → Devanagari) before retrieval, and use multiple candidates as query expansion to handle spelling ambiguity.

## Inputs
- Phase 3 romanized query sets
- Phase 4 harness, BM25 index and dense baseline

## Tasks
1. **Install/choose the transliterator (decide on day 1 of this phase):**
   - First choice: **IndicXlit** (AI4Bharat), which supports Hindi and Gujarati and returns top-k candidates.
   - If fairseq won't install on Windows: run it in WSL, or use an HF seq2seq transliteration model.
2. **Top-1 back-transliteration:** replace each romanized word with its best native candidate, then run BM25 and dense retrieval.
3. **Top-k expansion (core IR idea):** use the top-k candidates per word as a weighted query, so spelling variants all get a chance to match. Weight each candidate by its transliteration score (normalized per word):
   $$q' = \bigcup_{w \in q} \big\{ (c_{w,j},\; p_{w,j}) \big\}_{j=1}^{k}, \qquad \sum_{j=1}^{k} p_{w,j} = 1$$
   where $c_{w,j}$ is the $j$-th native candidate for word $w$ and $p_{w,j}$ its weight in the BM25 query.
   Tune k ∈ {1, 3, 5} on the **tune split** only.
4. Handle code-mixed queries: words that are English (e.g. `news`) should not be forced into Devanagari. Try a simple rule (keep words found in an English wordlist) and report whether it helps.
5. Evaluate everything on all romanized query sets, next to the Phase 4 baselines.

## Outputs
- `src/data/backtranslit.py`
- `data/processed/queries_backtranslit_{natural,rule,human}_top{k}.tsv` (frozen)
- `configs/backtranslit_*.yaml`
- New rows in `results/results.csv`

## Exit criteria
- Top-1 and top-k runs are evaluated on every romanized query set.
- The best k is chosen on the tune split, and test results are reported with significance against BM25-romanized.

## Review gate
- How much of the gap does back-transliteration recover?
- Look at 10 queries where it fails: wrong candidate, English word, or named entity?

## Risks / fallback
- Transliterator setup eats more than half a day → switch to the fallback immediately and note it.

## What we found
_(fill in at the end of the phase)_
