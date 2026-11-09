"""Run a retrieval baseline for every split and query set in a config, and evaluate it.

Run from the repo root inside the conda env `script-gap`:

    python -m scripts.run_baseline --config configs/baseline_bm25.yaml
    python -m scripts.run_baseline --config configs/baseline_bm25.yaml --split tune

For each (split, query set) pair the script:

1. reads the queries and the qrels of exactly those queries (the 50 hand-romanized
   queries are scored against the qrels of those 50, not of all 250);
2. retrieves the top ``top_k`` passages per query;
3. writes the TREC run file ``{run_dir}/{run_name}.trec``;
4. scores it and saves per-query scores plus one row in ``results/results.csv``.

Run names follow SPECS §8: ``{lang}_{method}_{query_set}_{split}``. An existing
run file is not overwritten unless ``--force`` is given (then the run is redone
and logged again; tables use the latest row of each run).
"""

import argparse
from collections.abc import Callable
from pathlib import Path

from src.data.load import Queries, read_qrels, read_queries
from src.data.splits import subset_qrels
from src.eval.evaluate import RESULTS_FILE, evaluate_run
from src.utils.config import load_config, resolve_path
from src.utils.runs import Run, run_name, write_run
from src.utils.seed import set_seed

Searcher = Callable[[Queries], Run]


def bm25_searcher(cfg: dict) -> Searcher:
    from src.retrieval.bm25 import open_index, retrieve

    index = open_index(resolve_path(cfg["index_dir"]))
    params = cfg["bm25"]

    def search(queries: Queries) -> Run:
        return retrieve(
            index, queries, k1=params["k1"], b=params["b"], top_k=cfg["top_k"],
            floor_idf=params["floor_idf"],
        )

    return search


def bm25_backtranslit_searcher(cfg: dict) -> Searcher:
    """BM25 with weighted top-k back-transliterated queries (Phase 5): each romanized
    query becomes ``src.data.backtranslit.weighted_query`` with ``backtranslit.k``."""
    from src.data.backtranslit import load_candidates, weighted_query
    from src.retrieval.bm25 import open_index, retrieve_weighted

    index = open_index(resolve_path(cfg["index_dir"]))
    params, k = cfg["bm25"], cfg["backtranslit"]["k"]
    candidates = load_candidates(resolve_path(cfg["backtranslit"]["candidates_file"]))

    def search(queries: Queries) -> Run:
        weighted = {qid: weighted_query(text, candidates, k) for qid, text in queries.items()}
        return retrieve_weighted(index, weighted, k1=params["k1"], b=params["b"],
                                 top_k=cfg["top_k"], floor_idf=params["floor_idf"])

    return search


def dense_searcher(cfg: dict) -> Searcher:
    import torch

    from src.retrieval.dense import load_corpus, load_model, retrieve

    model = load_model(cfg["model"], fp16=cfg["fp16"])
    docnos, vectors = load_corpus(resolve_path(cfg["embeddings_dir"]))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    doc_vectors = torch.from_numpy(vectors).to(device).float()  # float32 scores: fewer ties

    def search(queries: Queries) -> Run:
        return retrieve(model, queries, doc_vectors, docnos, max_length=cfg["query_max_length"],
                        batch_size=cfg["batch_size"], top_k=cfg["top_k"])

    return search


SEARCHERS: dict[str, Callable[[dict], Searcher]] = {
    "e5base": dense_searcher,
    "bm25": bm25_searcher,
    "bm25_classic": bm25_searcher,
    "bm25_idf0": bm25_searcher,
    "bm25_bt5": bm25_backtranslit_searcher,
}


def run_all(cfg: dict, config_path: str, splits: list[str], *, force: bool = False,
            results_file: str | Path = RESULTS_FILE) -> dict[str, dict[str, float]]:
    """Run, write and evaluate every (split, query set) of ``cfg``; returns the
    mean scores per run name. Output locations come from ``cfg["output"]``."""
    search = SEARCHERS[cfg["method"]](cfg)
    all_means = {}
    print("| run | queries | ndcg@10 | mrr@10 | recall@100 | recall@1000 | judged@10 |")
    print("|---|---|---|---|---|---|---|")
    for split in splits:
        all_qrels = read_qrels(resolve_path(cfg["qrels"].format(split=split)))
        for query_set, pattern in cfg["query_sets"].items():
            queries_path = resolve_path(pattern.format(split=split))
            if not queries_path.exists():
                continue  # e.g. the hand-romanized set exists for test only
            name = run_name(cfg["lang"], cfg["method"], query_set, split)
            run_file = Path(cfg["output"]["run_dir"]) / f"{name}.trec"
            if resolve_path(run_file).exists() and not force:
                print(f"| {name} | (run file exists, skipped; --force to redo) |")
                continue

            queries = read_queries(queries_path)
            missing = sorted(set(queries) - set(all_qrels))
            if missing:
                raise ValueError(f"{name}: {len(missing)} queries have no qrels, e.g. {missing[0]}")
            qrels = subset_qrels(all_qrels, queries)

            run = search(queries)
            write_run(run, resolve_path(run_file), tag=name, top_k=cfg["top_k"])
            means = evaluate_run(
                run, qrels,
                per_query_file=Path(cfg["output"]["per_query_dir"]) / f"{name}.csv",
                log={
                    "phase": cfg["phase"], "lang": cfg["lang"], "method": cfg["method"],
                    "query_set": query_set, "split": split, "config": config_path,
                    "seed": cfg["seed"], "run_file": run_file.as_posix(),
                },
                results_file=results_file,
            )
            all_means[name] = means
            print(f"| {name} | {len(queries)} | " + " | ".join(f"{v:.4f}" for v in means.values()) + " |")
    return all_means


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", help="run only this split (it must be in the config)")
    parser.add_argument("--force", action="store_true", help="redo runs whose file exists")
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    splits = cfg["splits"]
    if args.split:
        if args.split not in splits:
            raise ValueError(f"split {args.split!r} is not in the config ({splits})")
        splits = [args.split]
    run_all(cfg, args.config, splits, force=args.force)


if __name__ == "__main__":
    main()
