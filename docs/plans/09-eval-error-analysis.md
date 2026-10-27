# Phase 9 — Evaluation & error analysis

**Status:** Not started · **Days:** 11–12

## Goal
Produce the final results table and understand **why** methods succeed or fail, not just which one has the highest number.

## Inputs
- All runs and per-query scores from Phases 4–8

## Tasks
1. **Final results table:** every method × every query set (native / natural / rule / human) × metrics (nDCG@10, MRR@10, Recall@100), on the test split.
2. **Significance:** paired tests of each method vs BM25-romanized and vs the best single method, with a multiple-comparison correction.
3. **Gap recovered:** for each method,
   $$\text{GapRecovered} = \frac{M_{\text{method}} - M_{\text{roman}}}{M_{\text{native}} - M_{\text{roman}}}$$
   (how much of the native-vs-romanized gap is closed).
4. **Per-query analysis:** win/loss/tie counts vs the baseline; scatter plots of per-query nDCG.
5. **Failure buckets** (tag a sample of ~50–100 queries, manually or with simple rules):
   - spelling variation (high CER between romanizations)
   - code-mixed queries (English words inside)
   - named entities (people, places, organizations)
   - query length (short vs long)
   Report metrics per bucket.
6. **Qualitative examples:** 5–10 queries with the top results from each method, explaining what went wrong or right.

## Outputs
- `notebooks/error_analysis.ipynb` (outputs cleared before commit)
- `results/final_table.csv` (+ a Markdown/LaTeX version for the report)
- `results/figures/*.png`

## Exit criteria
- Final table complete, every number traceable to a run file and config.
- Each failure bucket has a metric breakdown and at least one example.

## Review gate
- Agree on the 3–4 main findings the report will lead with.

## Risks / fallback
- Manual tagging takes too long → tag 50 queries only; use simple rules for code-mixing and length.

## What we found
_(fill in at the end of the phase)_
