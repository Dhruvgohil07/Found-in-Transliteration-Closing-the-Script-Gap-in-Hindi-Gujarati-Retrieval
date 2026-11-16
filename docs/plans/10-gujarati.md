# Phase 10 — Gujarati extension (droppable)

**Status:** Done (2026-11-17, branch `phase-10/gujarati`); review gate approved 2026-11-17 · **Days:** 12–14

> **Cut line:** if we are behind schedule on Day 12, this phase becomes "future work" in the report and the time goes to Phases 9 and 11.

## Goal
Show that the approach transfers to a second language and script (Gujarati), using the pipeline already built for Hindi.

## Inputs
- The full Hindi pipeline (Phases 2–8) and the final best system from Phase 8

## Tasks
1. **Find a Gujarati collection with qrels.** MIRACL has no Gujarati, so check candidates such as AI4Bharat **IndicIRSuite / IndicMSMARCO** (Gujarati) or another collection. Verify what is actually available before deciding; note that translated collections are less natural than native ones.
2. **Data card** for the Gujarati collection (statistics computed from the data), same splits procedure as Phase 2.
3. **Romanized queries:** natural + rule-based (same tools as Phase 3, Gujarati mode). Human-romanized subset (~50 queries) if time allows.
4. **Runs — limited set only:**
   - BM25 native and BM25 romanized (the gap)
   - back-transliteration (IndicXlit Gujarati)
   - the best final system from Phase 8 (fine-tuned model applied zero-shot, unless time allows Gujarati fine-tuning)
5. Add a Hindi vs Gujarati comparison to the final table.

**Plan changes (2026-11-16)**
- *Collection (task 1):* the Gujarati part of IndicQA retrieval (`mteb/IndicQARetrieval`, CC0-1.0), with native Gujarati questions over Gujarati Wikipedia paragraphs. IndicMARCO-gu has training queries only and no dev/test qrels. FIRE 2011/12 Gujarati needs a registration request. Both were rejected (docs/data/data_card_gu.md, "Source").
- *Distractors (added):* IndicQA-gu judges only 248 passages, which would make retrieval close to trivial. Unjudged passages from the Gujarati Wikipedia dump (2023-11-01) are added. Articles sharing a word 8-gram with a judged passage are dropped, the rest is packed into 130–300-word passages to match the judged passages' median length, and mostly-English passages and exact repeats are dropped. `scripts/build_collection_gu.py`, `src/data/distractors.py`.
- *Splits:* 100 tune and a seeded sample of 500 test queries (seed 42), so natural romanization fits the Gemini free tier in one day. No train split exists, so there is no Gujarati fine-tuning: Phase 7 models are applied unchanged. Tune and test share relevant passages (each passage answers several questions). This is kept, because only global parameters are tuned (data card, "Splits").
- *Romanization (task 3):* same tools and settings as Hindi (`configs/romanize_gu.yaml`): Uroman with `lang=guj`, and Gemini with the Hindi prompt adapted to Gujarati (Gujarati example, `ahmedabad` as the name example). Uroman 1.3.1.1 leaves the Gujarati visarga (ઃ) unchanged, though it maps the Devanagari one to `h`. The same `h` is applied afterwards, which affects one test query (`src/data/romanize.py`). `check_romanized` now also rejects leftover Gujarati letters. Natural set: 8 requests of 75. Gemini answered as `gemini-3.6-flash`, the same version as for Hindi. One answer with a stray Devanagari word was rejected and retried automatically. `scripts/romanization_quality.py`: rule 0 / natural 149 of 600 queries with a word count different from the native query. 145 of these have more words, mostly a case ending split from an English-spelled noun (`france ma`, `muslim league e`, `association na`), which is common in Gujarati typing. Native Gujarati attaches it (ફ્રાન્સમાં), which may hurt word-by-word back-transliteration in task 4. As for Hindi, errors are kept, not corrected by hand (e.g. `hoti` for હતી). CER natural vs rule 0.241. The human-romanized subset is not done (optional).
- *Final system (task 4):* "best final system from Phase 8" = hybrid A with interpolation (normalized native BM25 with top-k back-transliteration ⊕ dense e5 with top-1 back-transliteration). It uses no fine-tuned model, so no Phase 7 model is run on Gujarati. Every setting of the recipe is re-chosen on the Gujarati tune split with the Hindi procedure and grid. Results: BM25 k1 1.5 / b 0.6; k 3; normalization rules `gu_nukta`; α 0.5.
- *Dense passage length (task 4, added):* Gujarati needs about 2.2 e5 tokens per word, so 54% of the passages exceed the 256 tokens used for Hindi. 256 and 512 were compared on tune (native nDCG@10 0.4358 vs 0.3908), and 256 was kept (`configs/dense_len{256,512}_gu.yaml`).
- *Gujarati normalization rules (task 4):* Gujarati twins of the Devanagari rules (`gu_nukta`, `gu_chandrabindu`, `gu_nasal`, `gu_final_virama`, `gu_digits`), built by shifting the characters into the Gujarati block (`src/data/normalize.py`). Chosen by the Phase 6 leave-one-out on tune (`configs/tune_docside_normalization_gu.yaml`).
- *Comparison (task 5):* the Gujarati final table (`configs/final_table_gu.yaml`) and a Hindi vs Gujarati table with each method's rank and Kendall's τ-b between the rankings (`scripts/compare_languages.py`, `configs/compare_languages_hi_gu.yaml`). It is a separate table, so the Hindi final table stays as approved in Phase 9.

## Outputs
- `configs/data_gu.yaml` and Gujarati versions of the Phase 3–8 configs used
- `docs/data/data_card_gu.md`
- New rows in `results/results.csv` (language column = `gu`)

## Exit criteria
- Gap reproduced on Gujarati, and at least one gap-closing method evaluated.

## Review gate
- Does the method ranking from Hindi hold for Gujarati?

## Risks / fallback
- No suitable Gujarati collection with qrels → drop the phase and document the search in the report.

## What we found

**Exit criteria: met.** The gap is reproduced on Gujarati for BM25 and dense e5, and every gap-closing method of the limited set was evaluated. Every choice was made on tune and test was run once at the end. Tables: `results/tables/final_table_gu.md` (Bonferroni over 23 comparisons) and `results/tables/compare_hi_gu_test.md`.

**Collection.** IndicQA-gu (248 judged Wikipedia paragraphs, native questions) + 50,881 unjudged Gujarati Wikipedia distractors = 51,129 passages; 100 tune / 500 test queries (`docs/data/data_card_gu.md`).

**Gujarati test, nDCG@10 (GapRecovered against dense e5 in brackets)**

| Method | native | natural | rule |
|---|---|---|---|
| BM25 | 0.3564 | 0.0129 (0%) | 0.0083 (1%) |
| Dense e5 | 0.4325 | 0.0134 (0%) | 0.0020 (0%) |
| BM25 + top-3 back-transliteration | | 0.3593 (83%) | 0.3105 (72%) |
| Dense e5 + top-1 back-transliteration | | 0.3371 (77%) | 0.2963 (68%) |
| Normalized BM25 (+ top-3 back-transliteration) | 0.3564 | 0.3593 (83%) | 0.3105 (72%) |
| Hybrid A (α 0.5) | **0.4938** | **0.4744 (110%)** | **0.4287 (99%)** |

- **The gap is as large as in Hindi.** Romanized queries score about 0.01 with BM25 and dense, against 0.36 / 0.43 native. Off-the-shelf e5 on natural romanized queries scores 0.0134 in Gujarati and 0.0133 in Hindi.
- **Back-transliteration is again the key step.** It lifts natural queries to 77–83% of the gap with a single method. Top-3 BM25 on natural queries (0.3593) is level with BM25 on native queries (0.3564).
- **Hybrid A closes the whole gap on natural queries (110%).** It beats native-query dense e5 (0.4744 vs 0.4325) and is significantly better than the best single method on every query set. It also lifts native queries (0.4938 vs 0.4325, p < 0.001), as in Hindi.
- **Normalization barely matters for Gujarati.** Leave-one-out kept only `gu_nukta`, which is +0.002 on tune over no normalization, and it changes nothing on test. In Hindi the key was worth +0.02 (p = 0.003). The questions were written from the passages, so native queries already share the passages' spelling, and IndicXlit's Gujarati output rarely differs from Wikipedia's in the variants the rules cover.

**Review gate (approved 2026-11-17): does the Hindi method ranking hold for Gujarati?** Mostly yes (`results/tables/compare_hi_gu_test.md`). Kendall's τ-b between the two rankings is +0.91 on native queries and +0.69 on natural and on rule queries. In both languages hybrid A is first on every query set and the raw-query baselines are last. The one swap is in the middle. In Hindi, dense + top-1 back-transliteration beats lexical top-k (0.5608 vs 0.3870 natural); in Gujarati lexical top-k is ahead (0.3593 vs 0.3371), though not significantly. Likely reasons (not tested): e5 is weaker on Gujarati than on Hindi even with native queries (0.43 vs 0.61), half the Gujarati passages are cut at 256 tokens, and IndicQA questions share many words with their passages, which favours BM25.

**Limits.** The test set is 500 sampled IndicQA questions with one relevant passage each, and distractors are unjudged (`judged@10` can be at most 0.1 and is 0.037–0.064 for the methods that work, in results/results.csv). Absolute numbers are therefore not comparable with Hindi's MIRACL numbers; ranks and GapRecovered are. There is no hand-romanized Gujarati set and no Gujarati fine-tuning.
