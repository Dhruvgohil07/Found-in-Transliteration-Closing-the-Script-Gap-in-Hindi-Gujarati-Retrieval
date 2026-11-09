"""Compare ways of handling English words in back-transliterated queries, on tune.

Run from the repo root inside the conda env `script-gap`:

    python -m scripts.tune_backtranslit_english --config configs/tune_backtranslit_english_hi.yaml

For each romanized set and each ``english_mode`` (``src.data.backtranslit``),
BM25 is run with weighted top-k queries and the dense model with top-1 queries,
both with their baseline settings. The table of all metrics is saved to
``output``, and every mode is compared with ``"none"`` by a paired t-test.
Only the tune split is allowed (SPECS §5.2); nothing goes to results/results.csv.
"""

import argparse

import pandas as pd
import torch

from src.data.backtranslit import (
    load_candidates,
    load_english,
    top1_query,
    weighted_query,
)
from src.data.load import read_qrels, read_queries
from src.data.splits import subset_qrels
from src.eval.evaluate import mean_scores, per_query_scores
from src.eval.significance import compare
from src.retrieval import bm25, dense
from src.utils.config import load_config, resolve_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    cfg = load_config(parser.parse_args().config)
    if cfg["split"] != "tune":
        raise ValueError(f"tuning is only allowed on the tune split, not {cfg['split']!r}")

    bcfg, dcfg = load_config(cfg["bm25_config"]), load_config(cfg["dense_config"])
    candidates = load_candidates(resolve_path(cfg["candidates_file"]))
    english = load_english(resolve_path(cfg["english_file"]))
    all_qrels = read_qrels(resolve_path(cfg["qrels"]))

    index = bm25.open_index(resolve_path(bcfg["index_dir"]))
    model = dense.load_model(dcfg["model"], fp16=dcfg["fp16"])
    docnos, vectors = dense.load_corpus(resolve_path(dcfg["embeddings_dir"]))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    doc_vectors = torch.from_numpy(vectors).to(device).float()

    def run_bm25(queries, mode):
        params, k = bcfg["bm25"], bcfg["backtranslit"]["k"]
        weighted = {q: weighted_query(t, candidates, k, english, mode) for q, t in queries.items()}
        return bm25.retrieve_weighted(index, weighted, k1=params["k1"], b=params["b"],
                                      top_k=bcfg["top_k"], floor_idf=params["floor_idf"])

    def run_dense(queries, mode):
        texts = {q: top1_query(t, candidates, english, mode) for q, t in queries.items()}
        return dense.retrieve(model, texts, doc_vectors, docnos, max_length=dcfg["query_max_length"],
                              batch_size=dcfg["batch_size"], top_k=dcfg["top_k"])

    rows, per_query = [], {}
    for query_set, path in cfg["query_sets"].items():
        queries = read_queries(resolve_path(path))
        qrels = subset_qrels(all_qrels, queries)
        n_english = sum(t in english for text in queries.values() for t in text.lower().split())
        for method, runner in [("bm25_bt", run_bm25), ("e5base_bt1", run_dense)]:
            for mode in cfg["english_modes"]:
                scores = per_query_scores(runner(queries, mode), qrels)
                per_query[method, query_set, mode] = scores
                rows.append({"method": method, "query_set": query_set, "english_mode": mode,
                             **mean_scores(scores)})
        print(f"{query_set}: about {n_english} English word occurrences in {len(queries)} queries")
    index.close()

    table = pd.DataFrame(rows)
    out = resolve_path(cfg["output"])
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False, float_format="%.4f", lineterminator="\n")
    print(f"\n{table.round(4).to_string(index=False)}\n")

    print("paired t-test against english_mode = none:")
    for (method, query_set, mode), scores in per_query.items():
        if mode == "none":
            continue
        base = per_query[method, query_set, "none"]
        for metric in cfg["metrics_tested"]:
            c = compare(scores, base, metric)
            print(f"  {method:11} {query_set:14} {mode:8} {metric:11} {c.diff:+.4f}  p={c.p:.3f}")


if __name__ == "__main__":
    main()
