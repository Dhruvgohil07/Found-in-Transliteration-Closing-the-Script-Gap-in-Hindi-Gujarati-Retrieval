# Data Card — Gujarati (IndicQA-gu + Wikipedia distractors)

## Source

| | |
|---|---|
| Judged part | Gujarati files of `mteb/IndicQARetrieval` on Hugging Face (revision `206ea3c`): IndicQA questions, each paired with the Wikipedia paragraph that answers it |
| Paper | Doddapaneni, Aralikatte, Ramesh, Goyal, Khapra, Kunchukuttan, Kumar. *Towards Leaving No Indic Language Behind: Building Monolingual Corpora, Benchmark and Models for Indic Languages.* ACL 2023 (arXiv:2212.05409) |
| What the dataset card says | IndicQA is "a manually curated cloze-style reading comprehension dataset", repurposed for retrieval by MTEB. The Gujarati queries are full questions in Gujarati script (see `data/processed/gu/sample_gu.md`) |
| Distractor part | Gujarati Wikipedia dump of 2023-11-01, `wikimedia/wikipedia`, config `20231101.gu` (revision `b04c8d1`) |
| Licences | IndicQARetrieval: CC0-1.0; Wikipedia: CC-BY-SA-3.0 / GFDL (Hugging Face dataset cards) |
| Download | Both are fetched by `scripts/build_collection_gu.py` with `huggingface_hub` into its cache (`~/.cache/huggingface`), outside the repo |

MIRACL, our Hindi collection, has no Gujarati. The other options we checked were IndicMARCO-gu (machine-translated MS MARCO, with training queries only and no dev/test qrels) and FIRE 2011/12 Gujarati (needs a registration request). Neither fits the project's time frame (docs/plans/10-gujarati.md, task 1).

## Why distractors, and how they are made

IndicQA-gu judges only a few hundred passages, and every passage answers several questions. Searching that set alone would be close to trivial, and Recall@100/1000 would mean little. So the judged passages are mixed with unjudged passages from Gujarati Wikipedia, making the search space similar in kind to MIRACL-hi. `scripts/build_collection_gu.py` (rules in `src/data/distractors.py`, settings in `configs/data_gu.yaml`):

1. **Short ids.** IndicQA ids are 64-character hashes, too long for Terrier's docno field. Judged passages become `indicqa-001` …, queries `gq-0001` …, numbered in the sorted order of the original ids (`data/processed/gu/id_map.tsv` keeps both).
2. **Remove copies of the judged passages.** The judged paragraphs are themselves taken from Wikipedia. Any article that shares a word 8-gram with a judged passage is dropped whole. Otherwise a copy of the right passage, or its answer, would sit in the collection unjudged and count as wrong.
3. **Clean and pack.** `<ref>` footnotes and HTML tags are removed. Lines under 5 words (headings, short list items) are skipped. The remaining lines are packed in order into passages of 130–300 words (a longer line is first split at sentence ends). These bounds put the distractors' median length close to the judged passages' (100–250 words gave a clearly lower median), so length alone gives little hint which passage is judged. Lengths per kind are in the table below; the distractor mean stays lower because many articles are shorter than one passage.
4. **Filter.** Passages where more than half of the Latin + Gujarati letters are Latin (untranslated English articles) are dropped. Romanized queries could match them by accident, which is not the effect we study. Exact repeats of an earlier passage (templated village stubs) are dropped too.
5. **No titles.** IndicQA passages have none, so distractors get none either. Otherwise the two kinds of passage would look different to the retrievers.

One small surface difference remains: footnote markers such as `[૪૫]` occur in 17 of the 248 judged passages and in 359 distractors (counted on the built corpus). Queries never contain them, so they hardly affect matching.

Running the script again gives byte-identical files.

**Limitation:** distractors are unjudged. A distractor from an article we kept may still answer a question (for example a history article that mentions the same event), and it then counts as non-relevant. Step 2 removes the clearest cases. The rest is the same kind of incompleteness as MIRACL's pooled judgments, and the `judged@10` column of `results/results.csv` shows how much of each top 10 is judged.

## Splits

| Our split | Source | Use |
|---|---|---|
| `tune` | 100 IndicQA-gu queries, fixed seed 42 | every parameter choice |
| `test` | 500 of the remaining queries, a seeded sample (seed 42) | final numbers only, never tuning |
| — | the other IndicQA-gu queries | not used |
| `train` | none | IndicQA has one labeled split, so there is no Gujarati fine-tuning; fine-tuned models from Hindi are applied unchanged |

- Test is a sample, not all remaining queries, so that natural (LLM) romanization of tune + test fits the Gemini free tier in one day (Phase 3 procedure).
- No query id or query text appears in two splits (checked by `src/data/splits.py`).
- **Tune and test share passages.** Each IndicQA passage answers several questions, and the split is by query, so many test queries have a relevant passage that is also relevant to some tune query (last row of the split table). Tuning only sets a few global parameters (BM25 k1/b, back-transliteration k, fusion α), and nothing is trained on tune, so we keep the query-level split. A split by passage would leave tune with only about a dozen passages.

## Statistics

Computed from the data by `python -m scripts.data_stats --config configs/data_gu.yaml`, which rewrites the block below. Do not edit it by hand.

<!-- stats:start -->
Tokenizer: `intfloat/multilingual-e5-base`. Passages are title + text, as indexed.

**How the collection was built**

| | Value |
|---|---|
| Judged passages / queries / judgments (`mteb/IndicQARetrieval`) | 248 / 2,015 / 2,017 |
| Wikipedia articles in the dump (`20231101.gu/train-00000-of-00001.parquet`) | 30,445 |
| Articles dropped: share a word 8-gram with a judged passage | 63 |
| Articles left with no passage (too short, or all passages filtered) | 3,405 |
| Passages dropped: mostly Latin letters | 798 |
| Passages dropped: exact repeat of an earlier passage | 1,966 |
| Articles used / distractor passages | 26,977 / 50,881 |
| Corpus passages (judged + distractors) | 51,129 |

**Collection**

| | Value |
|---|---|
| Passages | 51,129 |
| Articles (distinct article ids) | 27,225 |
| Passage length, words: mean / median / 95th pct / max | 125.8 / 131 / 232 / 517 |
| Passage length, e5 tokens: mean / median / 95th pct / max | 281.9 / 283 / 523 / 1,073 |
| Passages longer than 256 e5 tokens | 27,596 (54.0%) |
| judged (IndicQA): passages | 248 |
| judged (IndicQA): length, words: mean / median / 95th pct / max | 152.7 / 136 / 276 / 517 |
| judged (IndicQA): length, e5 tokens: mean / median / 95th pct / max | 334.3 / 300 / 618 / 1,073 |
| distractors (Wikipedia): passages | 50,881 |
| distractors (Wikipedia): length, words: mean / median / 95th pct / max | 125.6 / 131 / 231 / 300 |
| distractors (Wikipedia): length, e5 tokens: mean / median / 95th pct / max | 281.7 / 283 / 522 / 924 |

**Splits**

| | tune | test |
|---|---|---|
| Queries | 100 | 500 |
| Judgments | 100 | 500 |
| Judged passages per query: mean (min–max) | 1.0 (1–1) | 1.0 (1–1) |
| Relevant passages per query: mean / median (min–max) | 1.0 / 1 (1–1) | 1.0 / 1 (1–1) |
| Queries with no relevant passage | 0 | 0 |
| Judged passages missing from the corpus | 0 | 0 |
| Query length, words: mean (max) | 9.5 (26) | 9.7 (25) |
| Query length, e5 tokens: mean (max) | 21.1 (61) | 20.2 (56) |
| Queries containing Latin letters | 2 | 6 |
| Queries with a relevant passage that is also relevant to a tune query | – | 171 |
<!-- stats:end -->

## How to rebuild

```bash
python -m scripts.build_collection_gu --config configs/data_gu.yaml
python -m scripts.prepare_data --config configs/data_gu.yaml
python -m scripts.data_stats --config configs/data_gu.yaml
```

The first command downloads the two parquet files (about 41 MB together) on first use and writes `data/processed/gu/corpus.jsonl`, `labeled_{queries,qrels}.tsv`, `id_map.tsv` and `collection_meta.json`. The second writes `data/processed/gu/{tune,test}_{queries,qrels}.tsv`. Both give byte-identical files when run again. Python must run in UTF-8 mode (`PYTHONUTF8=1`, see `requirements.txt`).
