# Phase 9 — Evaluation & error analysis

**Status:** Done; review gate approved (2026-11-14) · **Days:** 11–12

## Goal
Produce the final results table and understand **why** methods succeed or fail, not just which one has the highest number.

## Inputs
- All runs and per-query scores from Phases 4–8

## Tasks
1. **Final results table:** every method × every query set (native / natural / rule / human) × metrics (nDCG@10, MRR@10, Recall@100, plus Recall@1000 as secondary), on the test split.
2. **Significance:** paired tests of each method vs BM25-romanized and vs the best single method, with a Bonferroni correction (SPECS §7.2).
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

**Plan changes (2026-11-14)**
- *Added task: the final system on native queries.* Hybrid A was also run with native (Devanagari) queries (`configs/fusion_A_interp_native.yaml`: normalized native BM25 ⊕ dense e5, same α = 0.3, nothing tuned). A user may type in either script, and this gives the native counterpart of the final system.
- *GapRecovered:* one pair of baselines for every row of the final table (dense e5, native vs romanized), so rows are comparable; earlier phase tables used each family's own baselines.
- *Spelling bucket:* measured as the number of native words (after the Phase 6 Devanagari key) missing from the top-1 back-transliteration, not as CER between romanizations. This measures what the retrieval pipeline actually loses.
- *Code-mixing bucket:* the Phase 5 English word list also counts Indian names that are frequent in English text (`bharat` 29 times, `gandhi`, `rajasthan`), which mixes code-mixing with named entities. A second list counts only words that are frequent in their lower-case form (`configs/english_common_words_hi.yaml`: 77 words instead of 122). The Phase 5 list is unchanged.
- *Named entities:* 100 test queries (seed 42) tagged by hand by the model assistant with a written rule (`configs/failure_buckets_hi.yaml`), versioned in `data/annotations/entities_hi_test.tsv` together with the names counted, for spot-checking.
- *Scope:* buckets and examples use the natural romanized test queries (the main condition). Test is used because every method choice is already fixed; nothing feeds back into a method.
- *Notebook:* thin. Every number comes from a script and its config (`scripts/final_table.py`, `plot_per_query.py`, `failure_buckets.py`, `qualitative_examples.py`); the notebook only walks through their outputs.

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

**Exit criteria: met.**
- Final table complete: `results/final_table.csv` (one row per method and query set, with its run file and config), `results/tables/final_table_hi.md` and `final_table_hi.tex`. 12 methods × 4 query sets; paired t-tests against BM25 with romanized queries and against the best single method, Bonferroni over all 71 comparisons (p < 0.0007).
- Each failure bucket has a metric breakdown (`results/tables/failure_buckets_hi_test.md`) and at least one example (`results/tables/examples_hi_test.md`). Figures are in `results/figures/`.

**Final table** (test, nDCG@10, GapRecovered in brackets; † significantly better than the best single method, −† significantly worse; every metric in `results/tables/final_table_hi.md`)

| Phase | Method | native | natural | rule | human |
|---|---|---|---|---|---|
| 4 | BM25 | 0.3810 | 0.0014 | 0.0017 | 0.0000 |
| 4 | Dense e5 | 0.6067 | 0.0133 | 0.0000 | 0.0106 |
| 5 | Dense e5 + top-1 back-transliteration (best single method) | | 0.5608 (92%) | 0.4907 (81%) | 0.5496 (88%) |
| 6 | Normalized BM25 (+ top-5 back-transliteration) | 0.4040 | 0.3870 (63%) | 0.3384 (56%) | 0.3425 (54%) |
| 6 | Romanized 4-gram index | | 0.3353 (54%) | 0.4479 (74%) | 0.4145 (66%) |
| 7 | Fine-tuned T (raw query) | 0.5468 −† | 0.2259 (36%) | 0.0614 (10%) | 0.1595 (24%) |
| 8 | **Hybrid A (final system)** | **0.6473 †** | **0.6115 † (101%)** | 0.5532 † (91%) | **0.5922 (95%)** |
| 8 | Hybrid C (dual index + dense) | | 0.6101 † (101%) | **0.5869 † (97%)** | 0.5882 (94%) |
| 8 | Hybrid D (no transliteration model) | | 0.4002 −† (65%) | 0.4491 (74%) | 0.4315 (69%) |

**The final system on native queries:** 0.6473 against 0.6067 for dense e5 (+0.041, p < 0.001; 100 wins, 103 ties, 47 losses). Fusion helps native queries as much as romanized ones. So hybrid A with natural romanized queries (0.6115) is level with the native dense baseline, but 0.036 below hybrid A with native queries.

**Failure buckets** (natural test queries, nDCG@10; `results/tables/failure_buckets_hi_test.md`)

| Bucket | n | BM25 top-5 norm. | dense top-1 | hybrid A | dense native | hybrid A native |
|---|---|---|---|---|---|---|
| 0 native words missed by top-1 back-transliteration | 144 | 0.3852 | 0.6010 | 0.6410 | 0.6243 | 0.6557 |
| 1 missed | 70 | 0.3832 | 0.5305 | 0.5946 | 0.5962 | 0.6433 |
| 2+ missed | 36 | 0.4015 | 0.4586 | 0.5263 | 0.5570 | 0.6212 |
| code-mixed (a common English word) | 57 | 0.3967 | 0.6129 | 0.6212 | 0.6342 | 0.6777 |
| no English word | 193 | 0.3842 | 0.5454 | 0.6086 | 0.5986 | 0.6383 |
| 1–8 words | 78 | 0.3440 | 0.6290 | 0.6600 | 0.6532 | 0.6783 |
| 9–12 words | 106 | 0.4076 | 0.5480 | 0.5846 | 0.6095 | 0.6503 |
| 13+ words | 66 | 0.4048 | 0.5006 | 0.5974 | 0.5475 | 0.6057 |
| named entity (sample) | 47 | 0.5190 | 0.6076 | 0.6646 | 0.6684 | 0.7209 |
| no named entity (sample) | 53 | 0.2971 | 0.5085 | 0.5574 | 0.5621 | 0.5986 |

**Findings**
- **What remains of the gap is transliteration error.** When top-1 back-transliteration recovers every native word (144 queries, 58%), hybrid A is only 0.015 below hybrid A with native queries (0.6410 vs 0.6557). When it misses two or more words (36 queries, 14%), the distance is 0.095 (0.5263 vs 0.6212). The weighted top-5 BM25 query is flat across these buckets (0.385–0.402): its alternative spellings absorb the errors, so the lexical route helps most where transliteration fails.
- **Typical failures: short words read as acronyms and the reverse** (examples 4 and 5): `goa` → जीओए (the letters G-O-A) and `phd` → फड़ (a word). Names spelled the English way also miss (`portugal` → पोर्तुगाल, against पुर्तगाल).
- **English words are not the problem.** Code-mixed queries score *higher* with dense + back-transliteration (0.6129 vs 0.5454): IndicXlit writes loanwords the way Hindi Wikipedia does (डिग्री, रिसर्च).
- **Named entities help lexical matching.** BM25 top-5 scores 0.5190 on queries with a named entity and 0.2971 without; rare names are highly discriminative terms (example 6: `vikatkavi` finds the Tenali Rama page, which dense misses). Dense retrieval confuses related entities (example 1: other "seven wonders" pages for Chichen Itza).
- **Long queries hurt dense retrieval, not BM25.** Dense + back-transliteration falls from 0.6290 (1–8 words) to 0.5006 (13+ words), and dense with native queries falls the same way (0.6532 → 0.5475). Hybrid A holds 0.5974 on long queries, above dense with native queries.
- **Some losses are judgment artefacts.** In the hybrid's largest loss (example 2) its new rank-1 passage is unjudged: MIRACL judges about 10 passages per query, so passages that only the hybrid finds are often unjudged. In example 3 every system scores 0 because the native query itself is cut (च्चतम for उच्चतम न्यायालय).

**Decisions**
- Hybrid A on native queries is reported as the native counterpart of the final system.
- One GapRecovered reference (dense e5) for the whole final table.
- Code-mixing counted with lower-case English words only.

**Surprises**
- Fusion lifts native queries about as much as romanized ones (+0.041 vs +0.051). "100% of the gap" against the single-method native baseline is therefore not the same as matching the best native system.
- Code-mixed queries are easier, not harder.

**Limits**
- Buckets are reported for the natural set only and without significance tests (36–193 queries per bucket).
- Named entities are tagged on a 100-query sample by one model-assisted annotator.
- The English word rule is frequency-based; `mere` (Hindi "my") still counts as English once.
- Examples show the top 3 passages only; judgments are shallow, so unjudged passages may be relevant.

**Review gate (approved 2026-11-14): the lead findings for the report**
1. **The script gap is near-total and reproduces the paper.** With natural romanized queries nDCG@10 drops 99.6% for BM25 and 97.8% for dense e5 (Phase 4), the pattern of the paper's 97% MRR@10 drop.
2. **Query-side back-transliteration is the key step.** Dense e5 + top-1 back-transliteration closes 92% of the gap; fine-tuning on 1,168 romanized queries closes 36%, document-side romanization 54–74% (BM25).
3. **Hybrid fusion closes the rest on natural queries.** Interpolating normalized BM25 with dense retrieval reaches 0.6115, 101% of the gap and level with dense on native queries. Fusion lifts native queries too (0.6473), so 0.036 remains against the best native system.
4. **What remains is transliteration error, not code-mixing or query length.** The 14% of queries whose back-transliteration misses two or more words lose 0.095 against the native system; acronyms and English-spelled names are typical failures.
