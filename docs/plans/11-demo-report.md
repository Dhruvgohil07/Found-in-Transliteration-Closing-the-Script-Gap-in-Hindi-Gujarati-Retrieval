# Phase 11 — Demo & report

**Status:** Not started · **Days:** 14–16

## Goal
Make the results easy to see (a live search demo) and easy to follow (a clear report).

## Inputs
- Final systems and results from Phases 4–10
- Final table and figures from Phase 9

## Tasks
### Demo
1. **Stack:** Streamlit (or Gradio); decide at the start of this phase.
2. Type a romanized query → see native-script results **side by side** from:
   - BM25 with the raw romanized query (baseline)
   - BM25 with back-transliteration
   - fine-tuned dense
   - hybrid (final best system)
3. Show the back-transliterated query and highlight matched terms where possible.
4. Load prebuilt indexes and embeddings only; no indexing or training inside the app.
5. Include a few ready example queries (native, romanized, code-mixed, named entity).

### Report
1. Sections: problem & motivation → related work (incl. the reference paper) → data → methods → experimental setup → results → error analysis → limitations → future work.
2. All formulas in LaTeX (BM25, nDCG, MRR, Recall, InfoNCE loss, RRF, gap recovered).
3. Every number comes from `results/` and matches the run files.

## Outputs
- `app/` (demo) with a short README on how to launch it
- `docs/report/` (report source + PDF)

## Exit criteria
- Demo runs from a clean checkout following its README (given the prebuilt artefacts).
- Report complete and cross-checked against `results/`.

## Review gate
- Full run-through of the demo from a clean start, and a final read of the report.

## Risks / fallback
- Dense model too slow on the laptop for live search → precompute results for the example queries and show those.

## What we found
_(fill in at the end of the phase)_
