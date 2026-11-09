"""Tests for src/eval: metric values worked out by hand on a toy example."""

import csv
import math

import pandas as pd
import pytest

from src.eval.evaluate import (
    METRICS,
    RESULT_COLUMNS,
    append_result,
    evaluate_run,
    mean_scores,
    per_query_scores,
    read_per_query,
)
from src.eval.significance import bonferroni_significant, compare, paired_ttest

# Toy judgments: q1 has two relevant docs (d1, d3) and one judged non-relevant (d2).
QRELS = {"q1": {"d1": 1, "d2": 0, "d3": 1}, "q2": {"d9": 1}, "q3": {"d5": 1}}
RUN = {
    "q1": {"d2": 3.0, "d1": 2.0, "dx": 1.0},  # ranks: d2 (non-rel), d1 (rel), dx (unjudged)
    "q2": {"d9": 1.0},                         # the relevant doc at rank 1
    # q3: no results at all -> must score 0, not be dropped
    "q4": {"d1": 5.0},                         # not in the qrels -> ignored
}

# q1 by hand:
#   DCG@10  = 1/log2(1+2)                  (d1 at rank 2, gain 2^1 - 1 = 1)
#   IDCG@10 = 1/log2(2) + 1/log2(3)        (both relevant docs at ranks 1 and 2)
Q1_NDCG = (1 / math.log2(3)) / (1 + 1 / math.log2(3))


def test_per_query_scores_by_hand():
    scores = per_query_scores(RUN, QRELS)
    assert list(scores.index) == ["q1", "q2", "q3"]  # q4 ignored, q3 kept
    assert list(scores.columns) == METRICS
    assert scores.loc["q1", "ndcg@10"] == pytest.approx(Q1_NDCG)
    assert scores.loc["q1", "mrr@10"] == pytest.approx(1 / 2)       # first relevant at rank 2
    assert scores.loc["q1", "recall@100"] == pytest.approx(1 / 2)   # found d1, missed d3
    assert scores.loc["q1", "judged@10"] == pytest.approx(2 / 3)    # d2, d1 judged; dx not
    assert scores.loc["q2"].tolist() == pytest.approx([1.0] * len(METRICS))
    assert scores.loc["q3"].tolist() == [0.0] * len(METRICS)


def test_mean_scores_average_over_all_judged_queries():
    means = mean_scores(per_query_scores(RUN, QRELS))
    assert means["ndcg@10"] == pytest.approx((Q1_NDCG + 1 + 0) / 3)
    assert means["mrr@10"] == pytest.approx((0.5 + 1 + 0) / 3)
    assert means["recall@1000"] == pytest.approx((0.5 + 1 + 0) / 3)


def test_recall_cutoff_matters():
    # The relevant doc at rank 150: outside the top 100, inside the top 1000.
    run = {"q2": {f"n{i}": 1000.0 - i for i in range(149)} | {"d9": 1.0}}
    scores = per_query_scores(run, {"q2": {"d9": 1}})
    assert scores.loc["q2", "recall@100"] == 0.0
    assert scores.loc["q2", "recall@1000"] == 1.0
    assert scores.loc["q2", "mrr@10"] == 0.0


# --- results table ------------------------------------------------------------

def make_row(**overrides):
    row = {c: "x" for c in RESULT_COLUMNS} | {m: 0.5 for m in METRICS}
    return row | overrides


def test_append_result_writes_header_once(tmp_path):
    path = tmp_path / "results.csv"
    append_result(make_row(method="bm25"), path)
    append_result(make_row(method="dense"), path)
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    assert rows[0] == RESULT_COLUMNS
    assert [r[RESULT_COLUMNS.index("method")] for r in rows[1:]] == ["bm25", "dense"]
    assert rows[1][RESULT_COLUMNS.index("ndcg@10")] == "0.5000"  # 4 decimals


def test_append_result_rejects_wrong_columns(tmp_path):
    with pytest.raises(ValueError, match="missing"):
        append_result({"method": "bm25"}, tmp_path / "results.csv")
    old = tmp_path / "old.csv"
    old.write_text("date,method\n", encoding="utf-8")
    with pytest.raises(ValueError, match="has columns"):
        append_result(make_row(), old)


def test_evaluate_run_saves_per_query_and_logs(tmp_path):
    log = {"phase": 4, "lang": "hi", "method": "bm25", "query_set": "native", "split": "test",
           "config": "configs/x.yaml", "seed": 42, "run_file": "results/runs/x.trec"}
    means = evaluate_run(RUN, QRELS, per_query_file=tmp_path / "pq.csv", log=log,
                         results_file=tmp_path / "results.csv")
    saved = read_per_query(tmp_path / "pq.csv")
    assert list(saved.index) == ["q1", "q2", "q3"]
    assert mean_scores(saved) == pytest.approx(means, abs=1e-6)
    table = pd.read_csv(tmp_path / "results.csv")
    assert table.loc[0, "n_queries"] == 3
    assert table.loc[0, "ndcg@10"] == pytest.approx(means["ndcg@10"], abs=1e-4)


# --- significance -------------------------------------------------------------

def test_paired_ttest_by_hand():
    a = pd.Series([1.0, 2.0, 3.0, 4.0], index=["q1", "q2", "q3", "q4"])
    b = pd.Series([0.0, 2.0, 2.0, 2.0], index=["q1", "q2", "q3", "q4"])
    # differences d = [1, 0, 1, 2]: mean 1, sample std sqrt(2/3), n = 4
    # t = 1 / (sqrt(2/3) / sqrt(4)) = sqrt(6)
    t, p = paired_ttest(a, b)
    assert t == pytest.approx(math.sqrt(6))
    assert 0.05 < p < 0.1  # not significant at alpha = 0.05 with only 4 queries


def test_paired_ttest_aligns_query_order_and_checks_queries():
    a = pd.Series([1.0, 2.0, 3.0], index=["q1", "q2", "q3"])
    b_shuffled = pd.Series([2.0, 3.0, 4.0], index=["q2", "q3", "q4"])
    with pytest.raises(ValueError, match="different queries"):
        paired_ttest(a, b_shuffled)
    same_reordered = a.iloc[::-1]
    assert paired_ttest(a, same_reordered) == (0.0, 1.0)  # identical runs


def test_compare_reports_means_and_difference():
    pa = pd.DataFrame({"ndcg@10": [0.9, 0.8, 0.7]}, index=["q1", "q2", "q3"])
    pb = pd.DataFrame({"ndcg@10": [0.1, 0.2, 0.3]}, index=["q1", "q2", "q3"])
    result = compare(pa, pb, "ndcg@10")
    assert result.n_queries == 3
    assert result.diff == pytest.approx(0.6)
    assert result.p < 0.05


@pytest.mark.parametrize(
    ("p", "m", "expected"),
    [(0.02, 1, True), (0.02, 3, False), (0.01, 3, True), (float("nan"), 1, False)],
)
def test_bonferroni(p, m, expected):
    # with m comparisons the threshold is 0.05 / m: 0.05 for m = 1, 0.0167 for m = 3
    assert bonferroni_significant(p, m) is expected


# --- the script gap -----------------------------------------------------------

def test_gap_uses_the_same_queries_and_relative_drop():
    from src.eval.gap import gap_rows

    native = pd.DataFrame({"ndcg@10": [1.0, 0.5, 0.0, 0.5]}, index=["q1", "q2", "q3", "q4"])
    natural = pd.DataFrame({"ndcg@10": [0.5, 0.25, 0.0, 0.25]}, index=["q1", "q2", "q3", "q4"])
    human = pd.DataFrame({"ndcg@10": [0.0, 0.0]}, index=["q1", "q2"])  # subset, like the 50
    rows = {r.query_set: r for r in gap_rows(native, {"natural": natural, "human": human},
                                             ["ndcg@10"])}
    assert rows["natural"].n_queries == 4
    assert rows["natural"].native == pytest.approx(0.5)      # mean over all 4
    assert rows["natural"].drop == pytest.approx(0.5)        # (0.5 - 0.25) / 0.5
    assert rows["human"].n_queries == 2
    assert rows["human"].native == pytest.approx(0.75)       # native mean over q1, q2 only
    assert rows["human"].drop == pytest.approx(1.0)          # everything lost


def test_gap_bonferroni_family_is_the_number_of_romanized_sets():
    from src.eval.gap import gap_rows

    qids = [f"q{i}" for i in range(6)]
    native = pd.DataFrame({"m": [0.9, 0.8, 0.9, 0.7, 0.8, 0.9]}, index=qids)
    close = pd.DataFrame({"m": [0.8, 0.8, 0.8, 0.7, 0.7, 0.85]}, index=qids)  # p = 0.0335
    one = gap_rows(native, {"a": close}, ["m"])[0]
    three = gap_rows(native, {"a": close, "b": close, "c": close}, ["m"])[0]
    assert one.p == three.p                     # same test ...
    assert 0.05 / 3 < one.p < 0.05              # ... between the two thresholds
    assert one.significant and not three.significant


def test_gap_rejects_queries_missing_from_native():
    from src.eval.gap import gap_rows

    native = pd.DataFrame({"m": [1.0]}, index=["q1"])
    with pytest.raises(ValueError, match="not in the native run"):
        gap_rows(native, {"x": pd.DataFrame({"m": [0.0]}, index=["q9"])}, ["m"])


def test_gap_drop_is_nan_when_native_is_zero():
    from src.eval.gap import GapRow

    row = GapRow("x", "m", 1, native=0.0, roman=0.0, p=1.0, significant=False)
    assert math.isnan(row.drop)
