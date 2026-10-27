# Phase 4 — Baselines & evaluation harness ★ milestone

**Status:** Not started · **Days:** 3–4

## Goal
Build the evaluation harness that every later phase uses, and **reproduce the script gap**: show how much retrieval quality drops when native queries are replaced by romanized ones.

## Inputs
- Phase 2 corpus, splits and qrels
- Phase 3 query sets (native, natural, rule, human)

## Tasks
1. **Evaluation harness** (`src/eval/`), using `ir_measures`:
   - nDCG@10, MRR@10 and Recall@100 for a run file + qrels
   - per-query scores saved alongside the averages
   - paired significance test between two runs (paired t-test, with a correction when comparing many runs)
   - append one row per run to `results/results.csv`: method, query set, split, config path, seed, metrics
2. **BM25 index** of the native corpus with PyTerrier (record the tokenizer/stemmer settings in the config).
3. **Baseline runs** on the test split:
   | Run | Query set | Role |
   |---|---|---|
   | BM25 | native | upper bound |
   | BM25 | natural / rule / human | shows the gap |
   | Dense (off-the-shelf, e.g. `multilingual-e5-base`) | native | dense upper bound |
   | Dense (off-the-shelf) | natural / rule / human | dense gap |
4. **Gap table:** relative drop per metric, e.g. $\Delta = \frac{M_{\text{native}} - M_{\text{roman}}}{M_{\text{native}}}$.
5. **Reproducibility check:** re-run one baseline from scratch with the same config and seed and confirm identical metrics.

### Metric definitions (for the report)
$$\text{nDCG@}k = \frac{1}{\text{IDCG@}k}\sum_{i=1}^{k} \frac{2^{rel_i}-1}{\log_2(i+1)}$$
$$\text{MRR@}k = \frac{1}{|Q|}\sum_{q \in Q} \frac{1}{\text{rank}_q}\quad(\text{0 if no relevant doc in top } k)$$
$$\text{Recall@}k = \frac{|\text{relevant} \cap \text{top-}k|}{|\text{relevant}|}$$

## Outputs
- `src/eval/evaluate.py`, `src/eval/significance.py`
- `src/retrieval/bm25.py`, `src/retrieval/dense.py`
- `configs/baseline_*.yaml`
- `results/runs/*.trec`, `results/per_query/*.csv`, `results/results.csv`
- Gap table in "What we found"

## Exit criteria
- All baseline runs are evaluated and logged.
- The reproducibility check gives identical numbers.
- BM25 native→native is in a plausible range for this collection (compared with **verified** published numbers).

## Review gate
- Is the gap clearly visible and does its trend match the paper?
- Are the numbers sane? If BM25-native looks off, fix it here before building anything on top.

## Risks / fallback
- Dense encoding of the full corpus is slow → encode once, cache embeddings to disk (git-ignored), and reuse them in later phases.

## What we found
_(fill in at the end of the phase)_
