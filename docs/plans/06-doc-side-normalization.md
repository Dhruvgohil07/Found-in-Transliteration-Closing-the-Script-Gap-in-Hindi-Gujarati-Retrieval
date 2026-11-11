# Phase 6 — Document-side romanization & phonetic normalization

**Status:** Done (2026-11-11), review gate approved · **Days:** 6–7

> **Plan change (2026-11-09, before the phase started):** the Phase 5 failure analysis
> showed that most remaining failures are valid *Devanagari* spelling variants
> (anusvara / chandrabindu / half-nasal, final virama, numeral script), not wrong
> transliterations. A **Devanagari normalization key** on the native index and the
> back-transliterated queries was added as task 1 and placed first: it targets the
> largest failure class and is cheap (the native index builds in about 4 minutes).
> The original tasks follow in their old order. If time runs short, the character
> n-grams are dropped before falling back to a corpus subset.

## Goal
Close the gap on the **document side**: normalize Devanagari spelling variants on both sides, also index the documents in Latin script, and make matching robust to spelling variation using a phonetic normalization key and character n-grams.

## Inputs
- Phase 2 corpus
- Phase 3 romanized query sets
- Phase 5 back-transliterated queries (for the normalized native index and the dual-index run)

## Tasks
1. **Devanagari normalization key**, applied identically to the native passages and to the native / back-transliterated queries. Each rule has a name and can be switched off for the ablation; they always run in this order:
   - `zero_width`: delete the zero-width joiner / non-joiner (`क्‍या→क्या`). *Added during the task:* Terrier's `utf` tokeniser splits a word at a zero-width joiner (`क्` + `या`), checked in `tests/test_normalize.py`
   - `nukta`: Unicode NFC, then drop the nukta (`क़→क`)
   - `chandrabindu`: chandrabindu → anusvara (`आँख→आंख`)
   - `nasal`: nasal consonant + virama before a consonant of its own class → anusvara (`आरम्भिक→आरंभिक`, `पम्प→पंप`)
   - `final_virama`: drop a word-final virama (`सम्राट्→सम्राट`)
   - `digits`: Devanagari digits → ASCII digits (`२→2`)

   `हुयी / हुई` (य ↔ ई) is left out on purpose: such a rule risks merging different words. Checks (`scripts/normalization_report.py`): do the Phase 5 failure pairs now share a key; how much does the vocabulary shrink, and which groups merge most (over-merging).
2. **Normalized native index** + runs: native queries and BM25 top-5 back-transliteration with normalized queries; each rule ablated on tune, only helpful rules kept; k1/b re-tuned on tune. *Done:* leave-one-out (`scripts/tune_normalization.py`) kept `chandrabindu`, `nasal`, `digits`; the kept set was then evaluated as its own index (added to the config after the first run); k1/b unchanged (1.2 / 0.3).
3. **Time it first:** romanize and index a 1% sample of the corpus to estimate time and disk for the full corpus. *Done:* `scripts/romanize_corpus.py --sample` (5,063 passages, seed 42): Uroman takes 20-21 ms per passage in one process (~3 h for the corpus) and 4.7 ms with 8 processes (≤ 40 min; byte-identical output); the JSON-lines text is ~0.23 GB and the sample index 4.7 MB in 0.1 min (≤ ~0.5 GB and ~10 min at full size, an upper bound since the vocabulary grows slower than the corpus). The full corpus fits, so no subset is needed.
4. **Romanize the corpus** with the fast rule-based romanizer, and build a **second BM25 index** (Latin script). *Done:* Uroman, 506,264 passages in 25.8 min with 8 processes (238 MB); index `data/indexes/hi_roman` (705,882 terms, 207 MB, 3.1 min). k1/b tuned on tune with the **rule** queries (the natural ones score 0.005-0.010 on the whole grid, no signal): 1.2 / 0.3 again. Tune nDCG@10: rule 0.4109 (above native on the native index, 0.3954), natural 0.0078.
5. **Phonetic / spelling normalization key** on the romanized side, applied identically to queries and documents. For example:
   - lowercase; collapse long vowels (`aa→a`, `ee/ii→i`, `oo/uu→u`)
   - merge common variants (`w/v`, `z/j`, `ph/f`)
   - drop the final inherent vowel (schwa: `kisana→kisan`)
   - de-duplicate repeated letters
   Run BM25 on the normalized index with normalized queries.
   *Done:* rules derived from the 995 aligned word pairs of the natural and Uroman tune queries (32.8% identical): `ch`, `sh`, `w_v`, `ph_f`, `z_j`, `anusvara` (Uroman writes the anusvara as `m`), `long_vowels`, `doubled`, `ei_e`, `eh_ah`, `y_glide`, `schwa` (V C a C V), `final_a`; the last three before `schwa` were *added during the task* from the remaining mismatches (`mein`/`mem`, `pehle`/`pahale`, `liye`/`lie`). All 13 give 90.1% same-key pairs but shrink the corpus vocabulary by 23.9% with visible over-merges (काम/कम, शक्ति/सकती). Leave-one-out on tune (decision: natural nDCG@10, fixed before the run) kept 11 rules (dropped `z_j`, `y_glide`); `doubled`, `schwa`, `final_a`, `anusvara` matter most (p < 0.05 each). The run was cut by a machine restart once and resumed from the finished indexes. k1/b re-tuned on natural queries: 1.2 / 0.3.
6. **Character n-gram tokenization** (e.g. 3–4-grams) on the romanized index, so near-miss spellings still share tokens.
   *Done:* `char_ngrams` (words padded with `0`, which Terrier keeps, unlike `_`; numbers kept whole) on top of the task 5 key, in one shared `text_key` for passages and queries. k1/b tuned per index on natural tune queries. *Plan change:* 5-grams were added, and the k1 grid was widened, because the first best (4-grams, k1 = 2.0) was at the edge of both ranges. Tune nDCG@10 (natural): words 0.3163 (k1 1.2, b 0.3), 3-grams 0.3089 (1.2, 0.75), **4-grams 0.3642 (2.0, 0.4)**, 5-grams 0.3561 (0.6, 0.6). Chosen: 4-grams (`data/indexes/hi_roman_ng4`, 127,938 terms, 234 tokens per passage).
7. **Dual-index retrieval:** combine
   - native (normalized) index ← back-transliterated query (Phase 5)
   - romanized/normalized index ← raw romanized query
   using a simple fusion (RRF with k = 60; the full fusion study, including tuning k, is Phase 8).
   *Done:* `src/retrieval/dual_index.py` (RRF written out, ties broken by docid as in our run files; checked against `ranx.fuse`, which is not used for the runs because its tie order is undocumented and it needs every run to hold the same queries) and `scripts/run_fusion.py`. Components: `docside_bt5_norm_bm25` + `docside_bm25_romindex_ng4`. Tune, natural: nDCG@10 0.3792 (native-normalized route 0.3807, p = 0.94; 4-gram route 0.3642; Phase 5 0.3656), but R@1000 0.9686 vs 0.9188 (p = 0.010) and R@100 0.8004 vs 0.7000 for the 4-gram route (p = 0.001). Rule queries: the 4-gram route alone is best (MRR@10 0.4755 vs fused 0.4086, p = 0.021), since they share the passages' romanizer.
8. Ablation: raw romanized index vs + normalization vs + n-grams, on the tune split; test table next to the Phase 4 baselines and Phase 5. *Done:* `scripts/method_table.py` (new optional `reference` column and `also_show` rows) with `configs/table_docside_hi_tune.yaml`, `table_docside_hi.yaml` and `table_docside_hi_recall.yaml`.

## Outputs
- `src/data/normalize.py` (Devanagari and romanized normalization keys, n-grams; unit-tested on examples)
- `src/retrieval/dual_index.py`
- `configs/docside_*.yaml`
- New rows in `results/results.csv` + ablation table in "What we found"

## Exit criteria
- Romanized index built (full corpus, or a documented subset).
- Ablation shows the effect of each step; dual-index result reported next to the baselines and Phase 5.

## Review gate
- Which normalization rules help, which hurt (Devanagari and romanized)? Keep only those that help on the tune split.
- Is the dual index better than Phase 5 alone?

## Risks / fallback
- Full corpus too slow/large → documented fixed-seed subset that keeps all judged passages (same subset for **all** methods in the comparison).
- Normalization over-merges different words → keep the rule list small and justified. Known Hindi minimal pairs exist for two rules: `nukta` (राज "rule" / राज़ "secret", जंग "war" / ज़ंग "rust") and `chandrabindu` (हंस "swan" / हँस "laugh"); the per-rule ablation on tune decides whether the gain outweighs them.

## What we found

**Exit criteria: met.**
- The romanized index covers the full corpus (506,264 passages); no subset was needed.
- Every step is ablated on tune (`results/tables/docside_hi_tune.md`). Every choice (rules, n, k1/b) was made there, and test was run once afterwards (`results/tables/docside_hi_test.md`, and `docside_hi_test_recall.md` with the p-values on recall@1000). All results sit next to the Phase 4 baselines and Phase 5.

**Test results** (BM25 nDCG@10, % = GapRecovered on nDCG@10; 250 / 250 / 50 queries)

| Method | natural | rule | human |
|---|---|---|---|
| romanized (Phase 4 baseline) | 0.0014 | 0.0017 | 0.0000 |
| Phase 5: top-5 back-transliteration, native index | 0.3737 (98.1%) | 0.3365 (88.3%) | 0.3294 (84.0%) |
| romanized index (Uroman), raw query | 0.0233 (5.8%) | 0.3933 (103.2%) | 0.0188 (4.8%) |
| + romanized key | 0.3077 (80.7%) | 0.4225 (110.9%) | 0.3054 (77.9%) |
| + romanized key + 4-grams | 0.3353 (88.0%) | **0.4479** (117.6%) | **0.4145** (105.7%) |
| top-5 back-transliteration, normalized native index | **0.3870** (101.6%) | 0.3384 (88.8%) | 0.3425 (87.3%) |
| dual index (RRF of the two lines above) | 0.3674 (96.4%) | 0.3948 (103.6%) | 0.3748 (95.5%) |
| native (Phase 4 upper bound) | 0.3810 | 0.3810 | 0.3923 |
| native, normalized native index | 0.4040 | 0.4040 | 0.4004 |
| *dense mE5 top-1 back-transliteration (Phase 5)* | *0.5608* | *0.4907* | *0.5496* |

Recall@1000 of the dual index: 0.9351 / 0.9569 / 0.9303, against 0.8874 / 0.8531 / 0.8790 for Phase 5 (p < 0.001 / < 0.001 / 0.058, paired t-test).

**Tune ablation** (nDCG@10, natural / rule; `results/tables/docside_hi_tune.md`): romanized index raw 0.0078 / 0.4109 → + key 0.3163 / 0.4203 → + 4-grams 0.3642 / 0.4296; normalized native index with top-5 back-transliteration 0.3807 / 0.3607; dual index 0.3792 / 0.3865.

**Findings**
- **The Devanagari key improves BM25 itself.** Native queries on the normalized index: nDCG@10 0.3810 → 0.4040 on test (p = 0.003; MRR@10, R@100 and R@1000 all p < 0.05). With top-5 back-transliteration it reaches 0.3870 for natural queries, slightly above the raw native upper bound (101.6%), though not significantly above Phase 5 (p = 0.055).
- **A romanized index only works with a romanized key.** Uroman spells passages differently from how people type (`raassttrapati` / `rashtrapati`), so raw romanized queries score 0.0233. The 11-rule key lifts them to 0.3077, and 4-grams to 0.3353.
- **The romanized route suits real typing and Uroman-like queries.** It is the best BM25 run for the hand-romanized queries (0.4145, above native 0.3923; p = 0.050 vs Phase 5, 50 queries) and for the rule set (0.4479, p < 0.001). It needs no transliteration model.
- **For the natural (LLM) queries, back-transliteration stays ahead.** 4-grams reach 0.3353 against 0.3870 on the normalized native index. They also fell from tune (0.3642) to test, unlike the native route (0.3807 → 0.3870). This suggests the romanized rules, derived from tune word pairs, partly fit the tune queries.
- **The dual index buys recall, not top precision.** nDCG@10 is level with Phase 5 (0.3674 vs 0.3737, p = 0.70), but R@1000 rises by 0.048 (p < 0.001) and R@100 from 0.7301 to 0.8220. Its R@1000 is even above native queries on the native index (0.8888, p = 0.004): each route finds relevant passages the other misses.
- **Dense retrieval is still far ahead** (0.5608 natural). Every Phase 6 gain is within BM25.

**Decisions**
- Devanagari key: `chandrabindu`, `nasal`, `digits` (leave-one-out on tune, decision fixed before the run: mean romanized nDCG@10). `zero_width`, `nukta` and `final_virama` were dropped. Their merges are correct, but with the weighted query they fold IndicXlit junk candidates (`किस्‌`, `ख़ेल`) into common question words and over-weight them.
- Romanized key: 11 rules (`z_j` and `y_glide` dropped, no effect). `doubled`, `schwa`, `final_a` and `anusvara` carry most of the gain (p < 0.05 each when removed).
- Character 4-grams (best of words, 3-, 4- and 5-grams on tune); k1 / b tuned per index: 1.2 / 0.3 for every word index, 2.0 / 0.4 for 4-grams.
- RRF with k = 60 over the two routes; tuning k and weights is left to Phase 8.

**Surprises**
- Terrier splits a word at a zero-width joiner and at `_`, and drops terms longer than 20 characters. Romanization lengthens words, so 0.029% of romanized passage tokens are lost.
- `ranx.fuse` needs every run to hold the same queries and does not document its tie order. RRF is written out in `src/retrieval/dual_index.py` and checked against it.
- Rule (Uroman) queries on the raw romanized index beat native queries on the native index (tune 0.4109 vs 0.3954). Uroman normalizes Devanagari spelling variants on its own.
- A machine restart cut the romanized ablation once. It resumed from the finished indexes, and the `none` rows reproduce the earlier runs exactly.

**Limits**
- The romanized rules were derived from the tune queries' word pairs, and their test gain is smaller than on tune.
- On 100 tune queries most single-rule differences are not significant; the decision rules sort small differences.
- BM25 only. Normalizing or fusing the dense model is not tested.
- The human set has 50 queries; its differences between methods are rarely significant.

**Review gate (approved 2026-11-11)**
- *Which rules help, which hurt?* Devanagari: `chandrabindu`, `nasal`, `digits` help; `zero_width`, `nukta`, `final_virama` hurt only through the weighted query. Romanized: 11 of 13 help, four of them clearly.
- *Is the dual index better than Phase 5 alone?* Not at the top of the ranking (nDCG@10 level), but clearly for recall (R@1000 +0.048, p < 0.001). Carried into Phase 8: the dual index (recall) and top-5 back-transliteration on the normalized native index (the best natural-query nDCG@10).
