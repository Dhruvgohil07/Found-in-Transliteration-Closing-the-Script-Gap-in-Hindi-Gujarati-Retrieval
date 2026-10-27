# Phase 10 — Gujarati extension (droppable)

**Status:** Not started · **Days:** 12–14

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
_(fill in at the end of the phase)_
