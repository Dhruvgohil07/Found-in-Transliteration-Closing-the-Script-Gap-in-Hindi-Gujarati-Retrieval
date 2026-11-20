# Report

`report.tex` is the final report (LaTeX), with `references.bib`. The folder is
self-contained, so it can be uploaded to Overleaf as it is.

## Build on Overleaf

1. Refresh the generated tables and figures, and check every number:
   ```bash
   python -m scripts.build_report_assets --config configs/report.yaml
   python -m scripts.check_report_numbers --config configs/report.yaml
   ```
2. Upload the whole `docs/report/` folder (including `generated/`) to a new Overleaf project.
3. In Overleaf, set **Menu → Compiler → XeLaTeX** (the report shows Devanagari and Gujarati
   text, which needs XeLaTeX and OpenType fonts) and compile.
4. Download the PDF and save it here as `report.pdf`.

The report uses the fonts **Noto Sans Devanagari** and **Noto Sans Gujarati**. If the
compiler reports one of them as missing, download it from Google Fonts (SIL Open Font
License), upload the `.ttf` files into the project, and change the two `\newfontfamily`
lines in `report.tex` to load them from the project folder, e.g.
`\newfontfamily\devfont{NotoSansDevanagari-Regular.ttf}[Path=./, Script=Devanagari]`.

## Where the content comes from

- `generated/` holds unchanged copies of `results/tables/*.tex` and `results/figures/*.png`,
  made by `scripts/build_report_assets.py` (never edit them by hand; `--check` finds stale
  copies).
- Every metric value in the text is checked by `scripts/check_report_numbers.py` against
  `results/`, the reference paper's notes (`docs/paper_notes.md`) and the data cards. p-values
  are recomputed from `results/per_query/`; the two CER values are listed in
  `configs/report.yaml` with the command that computes them.
