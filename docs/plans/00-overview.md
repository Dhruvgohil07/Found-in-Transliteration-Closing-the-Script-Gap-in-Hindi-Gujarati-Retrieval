# 00 — Project Plan Overview

**Project:** Found in Transliteration — Closing the Script Gap in Hindi/Gujarati Retrieval
**Scope:** Hindi is the core language; Gujarati is a separate, droppable phase (Phase 10).
**Timeline:** ~16 working days (2–3 weeks).

## Why phase-wise
Each phase builds on the outputs of the previous one (data → query sets → baselines → methods → analysis). If an early phase is wrong (bad splits, broken romanization, wrong metric), everything after it is wrong too. So we go **one phase at a time** and review each phase before the next one starts.

## Rules for every phase
1. **Review gate:** a phase starts only after the previous phase's review gate has been approved by the team.
2. **Status:** each phase file has a `Status:` line (`Not started` → `In progress` → `In review` → `Done`).
3. **Findings:** at the end of a phase, fill in its "What we found" section (numbers, surprises, decisions).
4. **Results logging:** every evaluated run appends a row to `results/results.csv` with method, query set, config path, seed and metrics.
5. **Baselines always shown:** no method is reported without the Phase 4 baselines next to it.
6. **Reproducibility:** every run has a config file in `configs/` plus a fixed seed (see `docs/SPECS.md` §8).
7. **Tuning only on the tune split:** test numbers are never used to pick parameters.
8. **Documentation goes in `docs/`:** any written document a phase produces goes under `docs/` (see below), never into `data/`, which is git-ignored.

## Where documents go (`docs/`)
```
docs/
  SPECS.md          full project specification (tech stack, data, methods, evaluation)
  progress.md       where the project stands (updated at the end of every task)
  plans/            these phase plans (00-overview + one file per phase)
  paper_notes.md    reference paper setup and headline results (Phase 1)
  data/             data cards: data_card_hi.md (Phase 2), data_card_gu.md (Phase 10)
  report/           final report source + PDF (Phase 11)
```
New documents get added here as phases need them, and this tree is updated when they are.

## Phases
| # | Phase | File | Days |
|---|---|---|---|
| 1 | Setup & groundwork | [01-setup.md](01-setup.md) | 1 |
| 2 | Data | [02-data.md](02-data.md) | 1–2 |
| 3 | Romanized query sets | [03-romanized-queries.md](03-romanized-queries.md) | 2–3 |
| 4 | Baselines & eval harness ★ | [04-baselines.md](04-baselines.md) | 3–4 |
| 5 | Query-side back-transliteration | [05-query-backtranslit.md](05-query-backtranslit.md) | 5–6 |
| 6 | Doc-side romanization & normalization | [06-doc-side-normalization.md](06-doc-side-normalization.md) | 6–7 |
| 7 | Dense retriever fine-tuning | [07-dense-finetune.md](07-dense-finetune.md) | 7–10 |
| 8 | Hybrid fusion | [08-hybrid-fusion.md](08-hybrid-fusion.md) | 10 |
| 9 | Evaluation & error analysis | [09-eval-error-analysis.md](09-eval-error-analysis.md) | 11–12 |
| 10 | Gujarati extension (droppable) | [10-gujarati.md](10-gujarati.md) | 12–14 |
| 11 | Demo & report | [11-demo-report.md](11-demo-report.md) | 14–16 |

★ Phase 4 is the key milestone: once the gap is reproduced, the project is on solid ground.

## Dependency flow
```
1 Setup → 2 Data → 3 Romanized queries → 4 Baselines ★
                                            ├─ 5 Query back-translit ─┐
                                            ├─ 6 Doc-side / normalize ┼→ 8 Hybrid → 9 Analysis → 10 Gujarati → 11 Demo & report
                                            └─ 7 Dense fine-tuning ───┘
```

## Environment (checked)
- GPU: RTX 4060 Laptop, **8 GB VRAM** → small/base models, fp16.
- System Python is 3.13.5 → use a separate **Python 3.11** env (several IR/transliteration libraries lag behind 3.13).
- Java 17 → fine for PyTerrier (recent Pyserini needs Java 21).

## Main risks
| Risk | Mitigation | Phase |
|---|---|---|
| IndicXlit/fairseq install on Windows | WSL, or an HF seq2seq transliteration model; decided at the start of the phase | 5 |
| 8 GB VRAM for fine-tuning | small/base models, fp16, `CachedMultipleNegativesRankingLoss`, max_len ~256 | 7 |
| Corpus too large to romanize/index quickly | time it on a sample first; use a documented fixed-seed subset if needed | 6 |
| Synthetic romanization ≠ real users | human-romanized test subset | 3 |
| Gujarati data availability | optional phase with a cut line on Day 12 | 10 |

## End-to-end verification
- Each phase's exit criteria are checked at its review gate.
- A single script plus config reproduces the Phase 4 baseline numbers from scratch with the same seed, giving identical metrics.
- The BM25 native→native number is in the range the reference/dataset papers report for that collection (we verify those reference numbers, never assume them).

## Open decisions
Tracked in one place: the "Open decisions / questions" section of [`docs/progress.md`](../progress.md).
