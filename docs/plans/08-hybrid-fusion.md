# Phase 8 — Hybrid fusion

**Status:** Not started · **Days:** 10

## Goal
Combine the best lexical and the best dense method, since they tend to fail on different queries, and check whether fusion beats either one alone.

## Inputs
- Best lexical run(s) from Phases 5–6
- Best dense run from Phase 7 (and the off-the-shelf dense run from Phase 4)

## Tasks
1. **Reciprocal Rank Fusion (RRF)**, using `ranx` or a short implementation:
   $$\text{RRF}(d) = \sum_{r \in R} \frac{1}{k + \text{rank}_r(d)}$$
   Tune $k$ (e.g. {10, 30, 60, 100}) on the **tune split** only.
2. **Score interpolation** as a comparison: normalize scores (min-max per query), then
   $$s(d) = \alpha \, \hat{s}_{\text{lex}}(d) + (1-\alpha)\, \hat{s}_{\text{dense}}(d)$$
   with $\alpha$ tuned on the tune split.
3. Fusion combinations to try:
   - BM25 + back-translit (lexical) ⊕ fine-tuned dense
   - dual index (Phase 6) ⊕ fine-tuned dense
   - best lexical ⊕ off-the-shelf dense (shows how much fine-tuning adds inside a hybrid)
4. Report test results with significance against the best single method.

## Outputs
- `src/retrieval/fusion.py`
- `configs/fusion_*.yaml`
- New rows in `results/results.csv`

## Exit criteria
- Fusion parameters picked on tune; test results logged with significance tests.

## Review gate
- Does fusion help, and on which kinds of queries?
- Pick the final "best system" for the demo and the report.

## Risks / fallback
- Fusion gives no gain → report it honestly; analyse the overlap of the two methods' retrieved sets in Phase 9.

## What we found
_(fill in at the end of the phase)_
