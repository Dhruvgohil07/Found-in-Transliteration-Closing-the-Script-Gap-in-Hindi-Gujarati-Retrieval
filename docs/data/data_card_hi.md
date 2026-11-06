# Data Card — Hindi (MIRACL-hi)

## Source

| | |
|---|---|
| Collection | MIRACL, Hindi (`miracl/hi` in `ir_datasets`) |
| Paper | Zhang, Thakur, Ogundepo, Kamalloo, Alfonso-Hermelo, Li, Liu, Rezagholizadeh, Lin. *Making a MIRACL: Multilingual Information Retrieval Across a Continuum of Languages.* arXiv:2210.09984 (2022) |
| Documents | Passages of Hindi Wikipedia, dump `hiwiki-20220301`; plain text only, images and tables removed (dataset card of `miracl/miracl-corpus`) |
| Queries and judgments | Written and judged by native speakers (dataset card of `miracl/miracl`) |
| Licence | Apache-2.0 (Hugging Face dataset cards of `miracl/miracl-corpus` and `miracl/miracl`) |
| Download | Corpus from `miracl/miracl-corpus` on Hugging Face; queries and qrels from `macavaney/miracl-noauth`, an `ir_datasets` mirror that avoids the login the official topics need. Both are fetched automatically by `ir_datasets` into its cache (`~/.ir_datasets`), outside the repo |

## Splits

| Our split | Source | Use |
|---|---|---|
| `tune` | 100 queries of `miracl/hi/dev`, fixed seed 42 | every parameter choice (BM25, k, checkpoints) |
| `test` | the other 250 queries of `miracl/hi/dev` | final numbers only, never tuning |
| `train` | `miracl/hi/train`, minus repeats of tune/test queries | Phase 7 fine-tuning only |

- `dev` is the only Hindi split with public qrels (`test-b` has queries but no judgments), so tune and test are carved from it (SPECS §5.2).
- **One train query was dropped as a leak:** train `677356#0` has exactly the same text as dev `1270408#0` (now in `test`), and the two share 2 relevant passages. MIRACL repeats this question across splits under different ids. Train queries matching a tune/test query by id or by text (case and whitespace normalized) are removed by `scripts/prepare_data.py`, which lists each one.
- No query id or query text appears in two splits (checked by `src/data/splits.py`).

## Statistics

Computed from the data by `python -m scripts.data_stats --config configs/data_hi.yaml`, which rewrites the block below. Do not edit it by hand.

<!-- stats:start -->
Tokenizer: `intfloat/multilingual-e5-base`. Passages are title + text, as indexed.

**Collection**

| | Value |
|---|---|
| Passages | 506,264 |
| Articles (distinct article ids) | 148,107 |
| Passage length, words: mean / median / 95th pct / max | 68.9 / 49 / 191 / 9,421 |
| Passage length, e5 tokens: mean / median / 95th pct / max | 110.8 / 80 / 301 / 16,047 |
| Passages longer than 256 e5 tokens | 39,085 (7.7%) |

**Splits**

| | tune | test | train |
|---|---|---|---|
| Queries | 100 | 250 | 1,168 |
| Judgments | 1,000 | 2,494 | 11,658 |
| Judged passages per query: mean (min–max) | 10.0 (10–10) | 10.0 (6–10) | 10.0 (7–10) |
| Relevant passages per query: mean / median (min–max) | 2.4 / 2 (1–9) | 2.0 / 2 (1–7) | 2.1 / 2 (1–9) |
| Queries with no relevant passage | 0 | 0 | 0 |
| Judged passages missing from the corpus | 0 | 0 | 0 |
| Query length, words: mean (max) | 10.0 (20) | 10.6 (24) | 10.1 (24) |
| Query length, e5 tokens: mean (max) | 14.2 (28) | 15.5 (34) | 15.0 (41) |
| Queries containing Latin letters | 1 | 1 | 7 |
<!-- stats:end -->

## How to rebuild

```bash
python -m scripts.prepare_data --config configs/data_hi.yaml
python -m scripts.data_stats --config configs/data_hi.yaml
```

The first command downloads MIRACL-hi (about 97 MB) on first use and writes `data/processed/{tune,test,train}_{queries,qrels}.tsv`; running it again gives byte-identical files. Python must run in UTF-8 mode (`PYTHONUTF8=1`, see `requirements.txt`).
