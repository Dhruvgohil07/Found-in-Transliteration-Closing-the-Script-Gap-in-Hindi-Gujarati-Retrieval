"""Choose k, the number of IndicXlit candidates per word in the weighted BM25 query.

Run from the repo root inside the conda env `script-gap`:

    python -m scripts.tune_backtranslit_k --config configs/tune_backtranslit_k_hi.yaml

For every romanized query set and every k, each query becomes the weighted query
of ``src.data.backtranslit.weighted_query`` and is searched with the Phase 4 BM25
baseline settings. All metrics are saved to ``output``; the best k per set by
``metric`` is printed. Only the tune split is allowed (SPECS §5.2). Rows are not
written to results/results.csv, which holds reported runs.
"""

import argparse
import itertools

import pandas as pd

from src.data.backtranslit import load_candidates, weighted_query
from src.data.load import read_qrels, read_queries
from src.data.splits import subset_qrels
from src.eval.evaluate import mean_scores, per_query_scores
from src.retrieval.bm25 import open_index, retrieve_weighted
from src.utils.config import load_config, resolve_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    cfg = load_config(parser.parse_args().config)
    if cfg["split"] != "tune":
        raise ValueError(f"tuning is only allowed on the tune split, not {cfg['split']!r}")

    bm25_cfg = load_config(cfg["bm25_config"])
    params = bm25_cfg["bm25"]
    index = open_index(resolve_path(bm25_cfg["index_dir"]))
    candidates = load_candidates(resolve_path(cfg["candidates_file"]))
    all_qrels = read_qrels(resolve_path(cfg["qrels"]))

    rows = []
    for (query_set, path), k in itertools.product(cfg["query_sets"].items(), cfg["ks"]):
        queries = read_queries(resolve_path(path))
        weighted = {qid: weighted_query(text, candidates, k) for qid, text in queries.items()}
        run = retrieve_weighted(index, weighted, k1=params["k1"], b=params["b"],
                                top_k=bm25_cfg["top_k"], floor_idf=params["floor_idf"])
        scores = mean_scores(per_query_scores(run, subset_qrels(all_qrels, queries)))
        rows.append({"query_set": query_set, "k": k, **scores})
        print(f"{query_set:14} k={k}  {cfg['metric']}={scores[cfg['metric']]:.4f}")
    index.close()

    table = pd.DataFrame(rows)
    out = resolve_path(cfg["output"])
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False, float_format="%.4f", lineterminator="\n")

    print(f"\n{table.round(4).to_string(index=False)}")
    for query_set, group in table.groupby("query_set", sort=False):
        best = group.loc[group[cfg["metric"]].idxmax()]  # ties go to the smaller k
        print(f"best k for {query_set} by {cfg['metric']}: {int(best['k'])}")


if __name__ == "__main__":
    main()
