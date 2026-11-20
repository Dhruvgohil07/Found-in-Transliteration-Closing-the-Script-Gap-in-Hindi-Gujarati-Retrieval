# Phase 11 — Demo & report

**Status:** In progress (started 2026-11-17, branch `phase-11/demo-report`; demo and report draft done, review gate pending) · **Days:** 14–16

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

**Plan changes (2026-11-17)**
- *Stack:* Streamlit 1.66.1 (pinned; installed without changing any existing package).
- *Live back-transliteration (added):* IndicXlit runs only in the Docker image, so the demo talks to it as a small HTTP service (`scripts/indicxlit/serve.py`, standard library only). The scoring code moved to `scripts/indicxlit/xlit_scoring.py`, shared with the frozen generator; regenerating the Gujarati candidates with it gives a byte-identical file. Words seen in the experiments use the frozen candidates; only new words go to the service (live scores differ from frozen ones by at most 1.2e-7 because of float32 batching, and the candidates and their order are the same, checked on 50 words per language).
- *Languages:* Hindi and Gujarati. Column 3 is the fine-tuned N model for Hindi and off-the-shelf e5 for Gujarati (no Gujarati fine-tuning).
- *Native-script queries:* skip back-transliteration and run the native-query configs of the experiments.
- *Passage store (added):* SQLite files (`scripts/build_demo_store.py`), since scanning the corpus per query is too slow.
- *Report build:* LaTeX in `docs/report/`, compiled on Overleaf (no local TeX installation).

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

**Demo (2026-11-17).**
- Every column is an existing experiment config (`configs/demo.yaml`). On judged tune queries (one romanized and one native per language) each column's top 10 equals the experiment run file; the top-1000 overlap is 0.997–1.0 (dense batch size 1 vs 64 moves a few tail positions).
- Examples are tune queries, one per kind, chosen by a rule: hybrid A nDCG@10 ≥ 0.5 in `results/per_query`. The code-mixed Hindi example shows the gap well: the baseline finds no relevant passage in its top 1000, the three back-transliterating methods rank it first.
- Run-through in the browser: both languages, an unjudged query with live words (`kisan andolan news` → किसान आंदोलन न्यूज़), native-script queries, language switch, and the service stopped (clear message; examples still work). First search per language takes 20–40 s (loading), later searches well under a second per method.

**Report (2026-11-17).**
- `docs/report/report.tex` (+ `references.bib`, `README.md`): problem, related work, data (Hindi and Gujarati), methods, setup, results, error analysis, demo, limitations, future work, conclusion; about 3,700 words plus 5 tables and 2 figures. All formulas in LaTeX (BM25, weighted back-transliteration, InfoNCE, RRF, interpolation, nDCG, MRR, Recall, GapRecovered, Kendall's τ_b).
- Tables and figures are generated: `scripts/build_report_assets.py` copies `results/tables/*.tex` and `results/figures/*.png` into `docs/report/generated/` (Overleaf compiles one folder) and `--check` finds stale copies. `scripts/compare_languages.py` gained a LaTeX output (4 decimals, as in its CSV source, so no double rounding).
- `scripts/check_report_numbers.py`: every number with 4+ decimals in the text must appear exactly in `results/`, the paper notes or the data cards (137 numbers, all traced); p-values are recomputed from `results/per_query/`; the two CER values are listed with their command. The first version accepted roundings and matched three numbers by coincidence (a computed difference and the CERs); the stricter version fixed that, and the difference was replaced by its two source numbers.
- Not compiled here (no local TeX): citations, references, inputs, figure paths and environments were checked by script, and pandoc parses the file. The bibliography is from the project's notes and from memory; entries without a doubtful detail were kept, uncertain page numbers and co-authors removed. To be checked against the publishers' pages.

**Review gate:** _(pending: demo run-through from a clean start, final read of the compiled report)_
