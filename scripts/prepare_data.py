"""Create the fixed tune/test/train split files for a collection.

Run from the repo root inside the conda env `script-gap`:

    python -m scripts.prepare_data --config configs/data_hi.yaml

The labeled split (MIRACL-hi dev) is divided into tune and test with a fixed
seed. Train queries that repeat a tune/test query (same id or same text) are
dropped from train and listed. Before anything is written, the script checks
that every labeled query has qrels and that no query id or query text appears
in two splits. Output goes to ``output_dir`` (git-ignored); running it
again with the same config produces identical files.
"""

import argparse

from src.data.load import load_qrels, load_queries, write_qrels, write_queries
from src.data.splits import (
    check_no_overlap,
    drop_overlapping,
    split_ids,
    subset_qrels,
    subset_queries,
)
from src.utils.config import load_config, resolve_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    cfg = load_config(parser.parse_args().config)

    labeled_queries = load_queries(cfg["dataset"]["labeled"])
    labeled_qrels = load_qrels(cfg["dataset"]["labeled"])
    missing = sorted(set(labeled_queries) - set(labeled_qrels))
    if missing:
        raise ValueError(f"{len(missing)} labeled queries have no qrels, e.g. {missing[0]!r}")

    tune_ids, test_ids = split_ids(labeled_queries, cfg["split"]["n_tune"], cfg["seed"])
    tune = subset_queries(labeled_queries, tune_ids)
    test = subset_queries(labeled_queries, test_ids)
    # MIRACL itself repeats some questions across dev and train; the training
    # copy is dropped so tune/test stay intact.
    train, dropped = drop_overlapping(load_queries(cfg["dataset"]["train"]), [tune, test])
    for qid in dropped:
        print(f"dropped train query {qid}: same id or text as a tune/test query")

    splits = {"tune": tune, "test": test, "train": train}
    qrels = {
        "tune": subset_qrels(labeled_qrels, tune_ids),
        "test": subset_qrels(labeled_qrels, test_ids),
        "train": subset_qrels(load_qrels(cfg["dataset"]["train"]), train),
    }
    check_no_overlap(splits)

    out = resolve_path(cfg["output_dir"])
    for name, queries in splits.items():
        write_queries(queries, out / f"{name}_queries.tsv")
        write_qrels(qrels[name], out / f"{name}_qrels.tsv")
        n_judged = sum(len(v) for v in qrels[name].values())
        print(f"{name}: {len(queries)} queries, {n_judged} judgments")
    print(f"no id or text overlap between {', '.join(splits)}; written to {cfg['output_dir']}/")


if __name__ == "__main__":
    main()
