"""Grid search of the BM25 parameters k1 and b on the tune split.

Run from the repo root inside the conda env `script-gap`:

    python -m scripts.tune_bm25 --config configs/tune_bm25_hi.yaml

Every (k1, b) pair of the grid is run on the tune queries and scored; the table
of all pairs is saved to ``output`` and the best pair by ``metric`` is printed.
Only the tune split is allowed here (SPECS §5.2: no tuning on test). The
grid rows are not written to results/results.csv, which holds reported runs.
"""

import argparse
import itertools

import pandas as pd

from src.data.load import read_qrels, read_queries
from src.data.splits import subset_qrels
from src.eval.evaluate import mean_scores, per_query_scores
from src.retrieval.bm25 import open_index, retrieve
from src.utils.config import load_config, resolve_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    cfg = load_config(parser.parse_args().config)
    if cfg["split"] != "tune":
        raise ValueError(f"tuning is only allowed on the tune split, not {cfg['split']!r}")

    queries = read_queries(resolve_path(cfg["queries"]))
    qrels = subset_qrels(read_qrels(resolve_path(cfg["qrels"])), queries)
    index = open_index(resolve_path(cfg["index_dir"]))

    rows = []
    for k1, b in itertools.product(cfg["grid"]["k1"], cfg["grid"]["b"]):
        run = retrieve(index, queries, k1=k1, b=b, top_k=cfg["top_k"], floor_idf=cfg["floor_idf"])
        rows.append({"k1": k1, "b": b, **mean_scores(per_query_scores(run, qrels))})
        print(f"k1={k1:<4} b={b:<5} {cfg['metric']}={rows[-1][cfg['metric']]:.4f}")
    index.close()

    table = pd.DataFrame(rows)
    out = resolve_path(cfg["output"])
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False, float_format="%.4f", lineterminator="\n")

    # Highest metric wins; ties go to the pair listed first in the grid.
    best = table.loc[table[cfg["metric"]].idxmax()]
    print(f"\nbest on {cfg['split']} by {cfg['metric']}: k1={best['k1']}, b={best['b']}")
    print(best.to_string())
    pivot = table.pivot(index="k1", columns="b", values=cfg["metric"])
    print(f"\n{cfg['metric']} by k1 (rows) and b (columns):\n{pivot.round(4).to_string()}")


if __name__ == "__main__":
    main()
