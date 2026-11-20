# Search demo

Type a query the way people do, in Latin script (`kisan andolan news`), and see four
methods search the same native-script collection side by side:

| Column | Hindi (MIRACL-hi) | Gujarati (IndicQA-gu + Wikipedia) |
|---|---|---|
| 1 | BM25 with the query as typed (the baseline) | same |
| 2 | BM25 + weighted top-5 back-transliteration | same with top-3 |
| 3 | fine-tuned dense e5 (N), top-1 back-transliteration | off-the-shelf e5 (no Gujarati fine-tuning) |
| 4 | hybrid A, the final system | same |

The page shows the top-1 back-transliteration of the query and highlights each method's
query terms in the passages. If the query is one of the judged tune/test queries, its
relevant passages are marked ✅ and each column says where the first one ranks. A query
in native script (Devanagari / Gujarati) skips back-transliteration and runs the
native-query versions of the same methods.

Every setting comes from `configs/demo.yaml`, which points at the experiment configs
behind the reported results. On judged tune queries, each column's top 10 is the same as
in the experiment's run file (checked for one romanized and one native query per
language). Nothing is indexed or trained in the app.

## 1. Prebuilt artefacts

The app only loads files built during the experiments (all git-ignored except the
candidate files). Each is rebuilt by the command shown, from the repo root inside the
conda env `script-gap` (`conda activate script-gap`; see `requirements.txt`).

| Needed for | Artefact | Rebuild |
|---|---|---|
| both | `data/backtranslit/candidates_{hi,gu}.tsv` (versioned) | already in the repo |
| Hindi | `data/indexes/hi_native`, `data/indexes/hi_native_norm` | `python -m scripts.build_bm25_index --config configs/index_bm25_hi.yaml` (and `index_bm25_hi_norm.yaml`) |
| Hindi | `data/embeddings/hi_e5base` | `python -m scripts.encode_corpus --config configs/baseline_dense.yaml` (~20 min) |
| Hindi | `models/e5base_hi_N/checkpoint-39`, `data/embeddings/hi_e5ft_N` | `python -m scripts.train_dense --config configs/train_dense_N.yaml`, then `python -m scripts.encode_corpus --config configs/finetuned_dense_N.yaml` |
| Hindi | `data/processed/*`, `data/romanized/*` | `python -m scripts.prepare_data --config configs/data_hi.yaml` (romanized sets are versioned) |
| Gujarati | `data/processed/gu/*` | `python -m scripts.build_collection_gu --config configs/data_gu.yaml`, then `python -m scripts.prepare_data --config configs/data_gu.yaml` |
| Gujarati | `data/indexes/gu_native`, `data/indexes/gu_native_norm` | `python -m scripts.build_bm25_index --config configs/index_bm25_gu.yaml` (and `index_bm25_gu_norm.yaml`) |
| Gujarati | `data/embeddings/gu_e5base_len256` | `python -m scripts.encode_corpus --config configs/dense_len256_gu.yaml` |

Then the passage stores (SQLite, about 0.6 GB for Hindi and 0.14 GB for Gujarati,
a few seconds each):

```bash
python -m scripts.build_demo_store --config configs/demo.yaml
```

## 2. Back-transliteration service (for new words)

Words that occur in the experiment queries are back-transliterated from the frozen
candidate files. Any other word needs IndicXlit, which runs in the project's Docker
image (`docker/indicxlit/`, built once with
`docker build -t script-gap-indicxlit docker/indicxlit`). With Docker Desktop
running, start the service from the repo root (Git Bash; it loads the model in about
a minute and then answers in well under a second):

```bash
docker run -d --rm --name script-gap-xlit -p 127.0.0.1:8765:8765 -v "$(pwd -W):/work" -v "$(pwd -W)/data/xlit_cache:/usr/local/lib/python3.8/site-packages/ai4bharat/transliteration/transformer/models/en2indic/v1.0" script-gap-indicxlit python scripts/indicxlit/serve.py
```

Check it with `curl http://127.0.0.1:8765/health`, and stop it with
`docker stop script-gap-xlit`. Without the service the example queries still work; a
query with a new word shows an error that says how to start it.

## 3. Launch

```bash
streamlit run app/app.py
```

Streamlit opens the page at http://localhost:8501. The first search in each language
loads the indexes, models and embeddings (about 20–40 s with a GPU); later searches take
well under a second per method. The dense columns hold their embedding matrices on the
GPU (about 3.1 GB for Hindi, which needs two of them; 0.16 GB for Gujarati); without a GPU everything runs on the
CPU, more slowly.

Python must run in UTF-8 mode on Windows: the conda env sets `PYTHONUTF8=1`, otherwise
use `python -X utf8 -m streamlit run app/app.py`.
