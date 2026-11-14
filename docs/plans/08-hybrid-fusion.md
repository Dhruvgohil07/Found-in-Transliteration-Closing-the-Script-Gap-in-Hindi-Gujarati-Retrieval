# Phase 8 — Hybrid fusion

**Status:** Done; review gate approved (2026-11-14) · **Days:** 10

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

**Plan changes (2026-11-14)**
- *Combinations.* Phase 7 found that the best dense method is still query-side: back-transliteration with the off-the-shelf model, with the N model level. Fine-tuning only helps raw romanized queries (T). So "fine-tuned dense" became two things: the N model with top-1 back-transliteration (hybrid B), and the T model on raw romanized queries, fused with the 4-gram romanized index (hybrid D, new). D is a hybrid that needs no transliteration model. The four combinations, lexical run first:
  - **A**: top-5 back-transliteration on the normalized native index ⊕ off-the-shelf e5 with top-1 back-transliteration (best lexical ⊕ best dense)
  - **B**: the same lexical run ⊕ fine-tuned N with top-1 back-transliteration
  - **C**: dual index (Phase 6) ⊕ off-the-shelf e5 with top-1 back-transliteration
  - **D**: romanized 4-gram index, raw query ⊕ fine-tuned T, raw query
- *RRF grid.* The first run put the best k of A, B and C at the edge of {10, 30, 60, 100}, so the grid was widened downward to {1, 5, 10, 20, 30, 60, 100}, as for k1/b in Phase 6.
- *Code.* RRF moved from `src/retrieval/dual_index.py` into `src/retrieval/fusion.py`, next to interpolation; the Phase 6 dual-index runs are reproduced byte for byte.
- *Added.* Per-query wins/losses against the components (`scripts/fusion_wins.py`) for the review gate's "on which kinds of queries", and a comparison of the hybrids with each other (`configs/compare_fusion_hi.yaml`) to pick the final system.

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

**Exit criteria: met.**
- k and α were chosen per hybrid on tune by natural-query nDCG@10, a rule fixed before the run (`configs/tune_fusion_hi.yaml`, `results/tuning/tune_fusion_hi.csv`). The table for tune is in `results/tables/fusion_hi_tune.md`.
- Test was run once afterwards: `results/tables/fusion_hi_test.md`. The table shows the Phase 4 baselines, the components and the best single method, which is the Phase 5 dense e5 with top-1 back-transliteration. The paired t-test against that method is Bonferroni-corrected over the table's 24 comparisons.

**Chosen parameters (tune, natural nDCG@10)**

| Hybrid | RRF k | α (weight of the lexical run) |
|---|---|---|
| A | 10 | 0.3 |
| B | 10 | 0.4 |
| C | 5 | 0.3 |
| D | 30 | 0.6 |

**Test results** (nDCG@10; 250 / 250 / 50 queries; † = significant against Phase 5 dense after Bonferroni, difference in brackets)

| Run | natural | rule | human |
|---|---|---|---|
| Phase 5: dense e5 + top-1 back-transliteration (best single method) | 0.5608 | 0.4907 | 0.5496 |
| A, RRF | 0.5833 | 0.5243 | 0.5374 |
| **A, interpolation** | **0.6115†** (+0.051) | 0.5532† (+0.063) | **0.5922** (+0.043, p = 0.18) |
| B, RRF | 0.5853 | 0.5323 | 0.5396 |
| **B, interpolation** | **0.6185†** (+0.058) | 0.5597† (+0.069) | 0.5698 |
| C, RRF | 0.5712 | 0.5535† | 0.5747 |
| **C, interpolation** | **0.6101†** (+0.049) | **0.5869†** (+0.096) | 0.5882 (+0.039, p = 0.15) |
| D, RRF (no transliteration model) | 0.3932† (−0.168) | 0.3644† | 0.4154 |
| D, interpolation (no transliteration model) | 0.4002† (−0.161) | 0.4491 | 0.4315 |
| *native queries, dense e5 (Phase 4 upper bound)* | *0.6067* | *0.6067* | *0.6220* |
| *Phase 6: normalized BM25 top-5 (A's lexical run)* | *0.3870* | *0.3384* | *0.3425* |

Recall: A interpolation reaches R@100 0.9569 and R@1000 0.9974 on natural queries, against 0.8962 and 0.9934 for Phase 5 dense.

**Findings**
- **Fusion helps, if the stronger run keeps more weight.** Interpolation raises natural nDCG@10 from 0.5608 to 0.61–0.62 with every hybrid that uses back-transliteration (p ≤ 0.001). This is above dense with *native* queries (0.6067; p vs native 0.48–0.80, so level), i.e. 100.6–102.0% of the gap recovered. RRF gains only +0.010 to +0.025 and is not significant on natural queries. RRF weights both lists equally, while dense is far stronger here (0.56 vs 0.39). Interpolation gives BM25 only 0.3–0.4 of the weight and keeps the dense score margins.
- **The lexical run rescues the queries where dense fails** (`results/tables/fusion_wins_hi_test.md`, A on natural queries). On the 61 queries where BM25 alone beat dense, the hybrid lifts dense from 0.31 to 0.52. On the 142 where dense was better, it gives back only 0.011 (0.729 → 0.718). Against dense it wins 100 queries, ties 107 and loses 43. The pattern is the same for B and C and on every query set.
- **Long queries gain most.** For A on natural queries, the gain over dense is +0.031 for queries of up to 8 words, +0.037 for 9–12 words and +0.097 for 13+ words. Dense quality falls with query length (0.63 → 0.50), and BM25 does not (0.34 → 0.40).
- **Fine-tuning adds nothing inside a hybrid.** B (fine-tuned N) and A (off-the-shelf) are level: natural +0.007 (p = 0.52), rule +0.007 (p = 0.60), human −0.022 (p = 0.35) (`results/tables/fusion_compare_hi_test.md`).
- **The dual index pays off for Uroman-like spellings.** C beats A on rule queries (0.5869 vs 0.5532, p < 0.001, significant after Bonferroni over 24 comparisons) and is level on natural (−0.001) and human (−0.004) queries. Its romanized route covers queries that IndicXlit transliterates badly.
- **A hybrid without a transliteration model stays well behind.** D (4-gram romanized BM25 ⊕ fine-tuned T) improves on both of its parts on natural queries (0.4002 vs 0.3353 and 0.2259), but stays 0.16 below Phase 5 dense. Back-transliteration remains the key step.

**Decisions**
- Own fusion code (`src/retrieval/fusion.py`), checked against `ranx` in the tests (RRF, and min-max + weighted sum). A passage missing from a run gets 0 from it. A list whose scores are all equal is normalized to 1, where ranx gives 0.
- One parameter per hybrid and method, chosen on natural tune queries and applied to every query set.
- Interpolation is the fusion method for every hybrid; RRF is reported as the comparison.

**Surprises**
- The tune split under-states the gain: A interpolation +0.037 on tune (p = 0.052) but +0.051 on test (p < 0.001).
- Hybrids with back-transliteration on natural romanized queries match dense retrieval with the *original* native queries.
- RRF's best k is small (5–10), not the usual default of 60. A small k trusts the top of each list more, which here lifts the dense run's top hits over BM25's lower ranks.

**Limits**
- Two-run fusion only; a three-way hybrid (both BM25 routes ⊕ dense) was not tried.
- α was searched in steps of 0.1. The tune curve is flat between 0.2 and 0.4 for A (0.6166 / 0.6170 / 0.6055).
- The human set has 50 queries, so no hybrid is significantly better than Phase 5 there after correction (A: +0.043, p = 0.18).

**Review gate (approved 2026-11-14)**
- *Does fusion help, and on which kinds of queries?* Yes, with interpolation: about +0.05 nDCG@10 over the best single method on natural and rule queries. The gain comes from queries where dense fails and BM25 succeeds, and from long queries.
- *Final "best system":* **hybrid A with interpolation (α = 0.3)**, i.e. top-5 back-transliteration on the normalized native BM25 index ⊕ off-the-shelf e5 with top-1 back-transliteration (`configs/fusion_A_interp.yaml`). It is level with B and C on natural and hand-romanized queries, and needs no fine-tuned model. C stays the documented alternative for Uroman-like spellings (rule set +0.034). Carried into Phase 9 (error analysis) and Phase 11 (demo and report).
