"""Scoring a run against relevance judgments, and logging the result (SPECS §7).

Metrics come from ``ir_measures`` (never hand-rolled):

- **nDCG@10**: are the relevant passages near the top? (graded, position-discounted)
- **MRR@10**: how high is the *first* relevant passage? (``RR@10`` per query)
- **Recall@100**, **Recall@1000**: what share of the relevant passages is found at all?
- **Judged@10**: what share of the top 10 has a relevance judgment? A diagnostic,
  not a quality metric: MIRACL judged only about 10 passages per query, so a
  method that finds *new* passages has them counted as non-relevant. A low
  Judged@10 means the other metrics may under-rate that run.

Every query in the qrels counts. A query the run has no results for scores 0
on every metric (it is not silently left out of the average). Queries in the
run that have no qrels are ignored.

Each evaluation keeps the per-query scores (for significance tests and error
analysis) and appends one row to ``results/results.csv``.
"""

import csv
from datetime import date
from pathlib import Path

import ir_measures
import pandas as pd
from ir_measures import RR, Judged, R, nDCG

from src.data.load import Qrels
from src.utils.config import resolve_path
from src.utils.runs import Run

# ir_measures measure -> our column name (the names used in SPECS §7.3).
MEASURES = {
    nDCG @ 10: "ndcg@10",
    RR @ 10: "mrr@10",
    R @ 100: "recall@100",
    R @ 1000: "recall@1000",
    Judged @ 10: "judged@10",
}
METRICS = list(MEASURES.values())

RESULTS_FILE = "results/results.csv"
RESULT_COLUMNS = [
    "date", "phase", "lang", "method", "query_set", "split", "config", "seed",
    "n_queries", *METRICS, "run_file",
]


def per_query_scores(run: Run, qrels: Qrels) -> pd.DataFrame:
    """One row per query in ``qrels`` (index ``qid``), one column per metric.

    Queries without results in ``run`` get 0 everywhere.
    """
    scores = pd.DataFrame(0.0, index=sorted(qrels), columns=METRICS)
    scores.index.name = "qid"
    for metric in ir_measures.iter_calc(list(MEASURES), qrels, run):
        if metric.query_id in qrels:
            scores.loc[metric.query_id, MEASURES[metric.measure]] = metric.value
    return scores


def mean_scores(per_query: pd.DataFrame) -> dict[str, float]:
    """The average of each metric over all queries (the number reported in tables)."""
    return {metric: float(per_query[metric].mean()) for metric in METRICS}


def write_per_query(per_query: pd.DataFrame, path: str | Path) -> None:
    """Save per-query scores as CSV (``qid`` plus one column per metric)."""
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    per_query.to_csv(path, float_format="%.6f", lineterminator="\n")


def read_per_query(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(resolve_path(path), index_col="qid", dtype={"qid": str})


def append_result(row: dict, path: str | Path = RESULTS_FILE) -> None:
    """Append one row to the results table, creating it (with a header) if needed.

    The columns are fixed (``RESULT_COLUMNS``); a missing or unknown key, or an
    existing file with a different header, raises ``ValueError`` so the table
    can never end up with shifted columns.
    """
    if set(row) != set(RESULT_COLUMNS):
        missing, extra = set(RESULT_COLUMNS) - set(row), set(row) - set(RESULT_COLUMNS)
        raise ValueError(f"result row: missing {sorted(missing)}, unknown {sorted(extra)}")
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    if not new_file:
        with open(path, encoding="utf-8", newline="") as f:
            header = next(csv.reader(f), None)
        if header != RESULT_COLUMNS:
            raise ValueError(f"{path} has columns {header}, expected {RESULT_COLUMNS}")
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS, lineterminator="\n")
        if new_file:
            writer.writeheader()
        writer.writerow({k: f"{v:.4f}" if k in METRICS else v for k, v in row.items()})


def evaluate_run(
    run: Run,
    qrels: Qrels,
    *,
    per_query_file: str | Path,
    log: dict,
    results_file: str | Path = RESULTS_FILE,
) -> dict[str, float]:
    """Score a run, save its per-query scores, log one results row, return the means.

    ``log`` holds the descriptive columns of the row (phase, lang, method,
    query_set, split, config, seed, run_file); date, number of queries and the
    metrics are filled in here.
    """
    per_query = per_query_scores(run, qrels)
    write_per_query(per_query, per_query_file)
    means = mean_scores(per_query)
    append_result(
        {**log, "date": date.today().isoformat(), "n_queries": len(per_query), **means},
        results_file,
    )
    return means
