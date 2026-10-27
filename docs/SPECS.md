# Found in Transliteration — Project Specification

## 1. Problem & Vision

In Indian web search, users often type Hindi or Gujarati in **Latin script** ("Hinglish" / romanized, e.g. `kisan andolan news`), while the documents they want are written in **native script** (Devanagari / Gujarati, e.g. `किसान आंदोलन`). Query and document tokens then don't overlap, so lexical retrievers like BM25 fail. Multilingual dense retrievers also lose much of their quality on romanized input.

This project **reproduces that script gap** and **builds and compares methods that close it**, using standard IR evaluation and a clear error analysis.

- **Reference paper:** "Lost in Transliteration" — Chari, Ounis & MacAvaney, SIGIR 2025 (short paper). It provides the baseline framing and the evaluation setup. Its exact setup is recorded in `docs/paper_notes.md` (Phase 1); nothing from the paper is assumed before that.
- **Context:** Information Retrieval course project, DA-IICT. Instructor: Sandip Modha.

### Goals
1. Reproduce the problem: measure how much retrieval quality drops when native-script queries are replaced by romanized ones.
2. Build and compare methods that close the gap (query-side, document-side, dense fine-tuning, hybrid).
3. Report results with standard IR metrics, significance tests and an error analysis.
4. Show it in a simple side-by-side search demo.

### Non-goals (do NOT build these)
- No LLM-wrapper solution as the main method: the methods must rest on core IR concepts. An LLM may be used only as a tool (e.g. to generate romanized queries).
- No production search engine (no serving infrastructure, scaling or crawling).
- No languages beyond Hindi and Gujarati.
- No pre-training of models from scratch; fine-tuning existing models only.

---

## 2. Constraints

| Constraint | Value |
|---|---|
| Timeline | 2–3 weeks, ~16 working days (see `docs/plans/00-overview.md`) |
| Languages | Hindi = core; Gujarati = separate, droppable phase (cut line: Day 12) |
| GPU | NVIDIA RTX 4060 Laptop, **8 GB VRAM** → small/base encoders, fp16 |
| OS | Windows 11 (WSL allowed as a fallback for tools that don't run natively) |
| Python | System Python is 3.13 → project uses a separate **Python 3.11** environment |
| Java | 17 (needed by PyTerrier; recent Pyserini would need Java 21) |

---

## 3. Tech Stack

| Layer | Choice | Status |
|---|---|---|
| Language | Python 3.11, type hints on functions | decided |
| Environment | `venv` + `pip` with a pinned `requirements.txt` (conda only if a package needs it) | decided |
| Datasets | `ir_datasets` | decided |
| BM25 / indexing | PyTerrier (Terrier BM25, Java 17) | decided |
| Dense retrieval | sentence-transformers + PyTorch (CUDA) | decided |
| Starting dense model | `intfloat/multilingual-e5-small` or `-base` | confirm in Phase 1 |
| Vector search | exact search over cached embeddings (NumPy / torch); FAISS only if needed | decided |
| Back-transliteration (Latin → native) | IndicXlit (AI4Bharat); fallback: WSL or an HF seq2seq model | confirm in Phase 5 |
| Rule-based romanization | `indic_transliteration` (ITRANS / ISO) | decided |
| Natural romanization | a transliteration model or an LLM, outputs frozen to files | confirm in Phase 3 |
| Evaluation | `ir_measures` (metrics), `ranx` (fusion + significance tests) | decided |
| Data handling | pandas | decided |
| Configs | YAML files in `configs/` (PyYAML) | decided |
| Figures | matplotlib | decided |
| Notebooks | Jupyter (exploration + error analysis only) | decided |
| Tests / lint | pytest (for pure functions), ruff | decided |
| Demo | Streamlit (or Gradio) | confirm in Phase 11 |

**Ask before adding any dependency not listed here.**

---

## 4. Repository Structure

```
data/                 raw + processed corpora, queries, qrels (git-ignored)
  raw/
  processed/
src/
  data/               loading, splits, romanization, back-transliteration, normalization
  retrieval/          bm25, dense, dual index, fusion
  train/              training-pair building + dense retriever fine-tuning
  eval/               metrics, significance tests, result tables
  utils/              seeds, config loading, run-file I/O
scripts/              small entry-point scripts (e.g. environment check)
configs/              one YAML file per experiment
results/
  runs/               TREC run files
  per_query/          per-query metric CSVs
  figures/
  results.csv         one row per evaluated run
models/               fine-tuned checkpoints (git-ignored)
notebooks/            exploration and error analysis
app/                  demo search UI
docs/
  SPECS.md            this file
  progress.md         where the project stands
  plans/              phase-wise plans (00-overview + one file per phase)
  paper_notes.md      reference paper setup and headline results (Phase 1)
  data/               data cards (Phase 2, Phase 10)
  report/             final report source + PDF (Phase 11)
```

- All written documentation lives under `docs/`. When a phase needs a new document, add it there and update this tree and the one in `docs/plans/00-overview.md`.
- `src/` code is importable modules; `scripts/` and notebooks call into `src/`, never the other way around.

---

## 5. Data

### 5.1 Collections
| Language | Primary candidate | Notes |
|---|---|---|
| Hindi | **MIRACL-hi** via `ir_datasets` (native Wikipedia passages, human queries + qrels) | confirm against the paper in Phase 2 |
| Gujarati | AI4Bharat IndicIRSuite / IndicMSMARCO (Gujarati) or another verified collection | MIRACL has no Gujarati; decide in Phase 10 |

All statistics (documents, queries, judgments, lengths) are **computed from the data** by a script and written to the data card (`docs/data/data_card_hi.md`, `docs/data/data_card_gu.md`). Never copy or estimate them.

### 5.2 Splits
- **tune:** used for every parameter choice (BM25 params, expansion k, RRF k, checkpoint selection).
- **test:** used only for final numbers, never for tuning.
- **train:** used only for Phase 7 fine-tuning; checked to have no query overlap with tune/test.
- If a collection has only one labeled split with public qrels, tune/test are carved from it with a **fixed seed**, recorded in the data card.

### 5.3 Query sets
Every query set contains the **same query IDs** as the native set:

| File (`data/processed/`) | Content |
|---|---|
| `queries_native.tsv` | original native-script queries |
| `queries_roman_natural.tsv` | casual "Hinglish"-style romanization (main condition) |
| `queries_roman_rule.tsv` | deterministic ITRANS/ISO romanization (contrast) |
| `queries_roman_human_a.tsv`, `queries_roman_human_b.tsv` | ~50–100 test queries romanized by hand, as two independent versions (A and B) |

Generated query sets are produced once and **frozen**; regeneration only on purpose, with a new file name.

### 5.4 File formats
- Queries: TSV, `qid<TAB>text`, UTF-8, no header.
- Qrels: TREC format, `qid 0 docid relevance`.
- Runs: TREC format, `qid Q0 docid rank score run_tag`, 100 results per query by default (Recall@100).

---

## 6. Methods

### 6.1 Baselines (Phase 4) — always reported next to every method
| Run | Role |
|---|---|
| BM25, native queries → native docs | lexical upper bound |
| BM25, romanized queries → native docs | shows the gap |
| Off-the-shelf multilingual dense model, native queries | dense upper bound |
| Off-the-shelf multilingual dense model, romanized queries | dense gap |

BM25 scoring:
$$\text{BM25}(q,d) = \sum_{t \in q} \text{IDF}(t)\,\frac{tf_{t,d}\,(k_1+1)}{tf_{t,d} + k_1\left(1 - b + b\,\frac{|d|}{\text{avgdl}}\right)}$$

### 6.2 Gap-closing methods
| Phase | Method | Core IR idea |
|---|---|---|
| 5 | Query-side back-transliteration (top-1, and top-k candidates as a weighted query) | query reformulation / expansion |
| 6 | Document-side romanized index + phonetic normalization key + character n-grams; dual index | index-side normalization, robust tokenization |
| 7 | Dense retriever fine-tuned on (romanized query, native passage) pairs with BM25 hard negatives | contrastive representation learning |
| 8 | Hybrid fusion of the best lexical and best dense runs (RRF; score interpolation as comparison) | rank fusion |

Fine-tuning loss (InfoNCE with in-batch and hard negatives):
$$\mathcal{L} = -\log \frac{e^{s(q,d^+)/\tau}}{e^{s(q,d^+)/\tau} + \sum_{d^- \in \mathcal{N}(q)} e^{s(q,d^-)/\tau}}$$

Reciprocal Rank Fusion:
$$\text{RRF}(d) = \sum_{r \in R} \frac{1}{k + \text{rank}_r(d)}$$

Details, variants and ablations for each method are in its phase plan in `docs/plans/`.

---

## 7. Evaluation

### 7.1 Metrics (via `ir_measures`)
$$\text{nDCG@}k = \frac{1}{\text{IDCG@}k}\sum_{i=1}^{k} \frac{2^{rel_i}-1}{\log_2(i+1)}$$
$$\text{MRR@}k = \frac{1}{|Q|}\sum_{q \in Q} \frac{1}{\text{rank}_q}\quad(0 \text{ if no relevant document in the top } k)$$
$$\text{Recall@}k = \frac{|\text{relevant} \cap \text{top-}k|}{|\text{relevant}|}$$

Reported cutoffs: **nDCG@10, MRR@10, Recall@100**.

### 7.2 Comparing methods
- Paired significance test per query (paired t-test), with a multiple-comparison correction when several methods are compared to one baseline.
- **Gap recovered** — how much of the native-vs-romanized gap a method closes:
  $$\text{GapRecovered} = \frac{M_{\text{method}} - M_{\text{roman}}}{M_{\text{native}} - M_{\text{roman}}}$$
- Per-query scores are always saved, so wins/losses and failure buckets (spelling variation, code-mixing, named entities, query length) can be analysed (Phase 9).

### 7.3 Results table — `results/results.csv`
One row per evaluated run, appended by the evaluation code (never edited by hand):

| Column | Example |
|---|---|
| `date` | `2026-11-02` |
| `phase` | `4` |
| `lang` | `hi` |
| `method` | `bm25` |
| `query_set` | `roman_natural` |
| `split` | `test` |
| `config` | `configs/baseline_bm25.yaml` |
| `seed` | `42` |
| `ndcg@10`, `mrr@10`, `recall@100` | metric values |
| `run_file` | `results/runs/hi_bm25_roman_natural_test.trec` |

---

## 8. Experiment Conventions

- **One YAML config per experiment** in `configs/`; scripts take `--config` and nothing that changes results is hard-coded.
- **Fixed seeds** (default `42`) for Python, NumPy and torch, set through `src/utils`.
- **Run naming:** `{lang}_{method}_{query_set}_{split}` — used for run files, run tags and per-query files.
- **Expensive artefacts are cached and git-ignored:** indexes, embeddings, checkpoints. Each one's config records how to rebuild it.
- **No hard-coded local paths** (e.g. `D:\...`); paths come from configs, relative to the repo root.

Example config:
```yaml
experiment: hi_bm25_roman_natural
lang: hi
seed: 42
data:
  queries: data/processed/queries_roman_natural.tsv
  qrels: data/processed/test_qrels.tsv
retrieval:
  method: bm25
  index: data/indexes/hi_native
  k1: 1.2
  b: 0.75
  top_k: 100
output:
  run_file: results/runs/hi_bm25_roman_natural_test.trec
```

---

## 9. Demo (Phase 11)

- Type a romanized query → see native-script results **side by side** from: BM25 with the raw romanized query, BM25 with back-transliteration, fine-tuned dense, and the final hybrid.
- Show the back-transliterated query; include ready example queries (romanized, code-mixed, named entity).
- Loads prebuilt indexes and embeddings only; no indexing or training inside the app.

---

## 10. Roadmap

| # | Phase | Plan |
|---|---|---|
| 1 | Setup & groundwork | `docs/plans/01-setup.md` |
| 2 | Data | `docs/plans/02-data.md` |
| 3 | Romanized query sets | `docs/plans/03-romanized-queries.md` |
| 4 | Baselines & evaluation harness ★ | `docs/plans/04-baselines.md` |
| 5 | Query-side back-transliteration | `docs/plans/05-query-backtranslit.md` |
| 6 | Doc-side romanization & normalization | `docs/plans/06-doc-side-normalization.md` |
| 7 | Dense retriever fine-tuning | `docs/plans/07-dense-finetune.md` |
| 8 | Hybrid fusion | `docs/plans/08-hybrid-fusion.md` |
| 9 | Evaluation & error analysis | `docs/plans/09-eval-error-analysis.md` |
| 10 | Gujarati extension (droppable) | `docs/plans/10-gujarati.md` |
| 11 | Demo & report | `docs/plans/11-demo-report.md` |

Each phase ends with a review gate; the next phase starts only after it is approved. Timeline, dependencies and risks are in `docs/plans/00-overview.md`; current status is in `docs/progress.md`.
