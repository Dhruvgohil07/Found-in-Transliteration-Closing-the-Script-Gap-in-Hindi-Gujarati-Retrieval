"""Tests for src/data: processed-file I/O, splits and leakage checks."""

import pytest

from src.data.load import (
    full_text,
    read_qrels,
    read_queries,
    write_qrels,
    write_queries,
)
from src.data.splits import check_no_overlap, drop_overlapping, split_ids, subset_qrels

# --- load.py: processed files ---------------------------------------------


def test_queries_round_trip_with_devanagari(tmp_path):
    queries = {"2": "किसान आंदोलन क्या है?", "10": "भारत की राजधानी"}
    path = tmp_path / "queries.tsv"
    write_queries(queries, path)
    assert read_queries(path) == queries


def test_write_queries_rejects_tab_in_text(tmp_path):
    with pytest.raises(ValueError, match="tab or newline"):
        write_queries({"1": "a\tb"}, tmp_path / "q.tsv")


def test_read_queries_rejects_duplicate_id(tmp_path):
    path = tmp_path / "q.tsv"
    path.write_text("1\tfirst\n1\tsecond\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate query id"):
        read_queries(path)


def test_qrels_round_trip_keeps_non_relevant(tmp_path):
    qrels = {"1": {"d1": 1, "d2": 0}, "2": {"d3": 1}}
    path = tmp_path / "qrels.tsv"
    write_qrels(qrels, path)
    assert path.read_text(encoding="utf-8").splitlines()[0] == "1\t0\td1\t1"
    assert read_qrels(path) == qrels


def test_full_text_joins_title_and_text():
    assert full_text({"docno": "d", "title": "भारत", "text": "एक देश है"}) == "भारत एक देश है"
    assert full_text({"docno": "d", "title": "", "text": "only text"}) == "only text"


# --- splits.py ----------------------------------------------------------------


def test_split_is_deterministic_and_order_independent():
    ids = [str(i) for i in range(50)]
    first = split_ids(ids, 20, seed=42)
    assert split_ids(reversed(ids), 20, seed=42) == first
    assert split_ids(ids, 20, seed=7) != first


def test_split_sizes_and_disjoint():
    tune, test = split_ids([str(i) for i in range(350)], 100, seed=42)
    assert len(tune) == 100
    assert len(test) == 250
    assert not set(tune) & set(test)


def test_split_rejects_bad_size():
    with pytest.raises(ValueError):
        split_ids(["a", "b"], 2, seed=42)


def test_subset_qrels_skips_unjudged_queries():
    qrels = {"1": {"d1": 1}, "2": {"d2": 1}}
    assert subset_qrels(qrels, ["1", "3"]) == {"1": {"d1": 1}}


def test_overlap_check_passes_on_disjoint_splits():
    check_no_overlap({"train": {"1": "a b"}, "test": {"2": "c d"}})


def test_overlap_check_catches_shared_id():
    with pytest.raises(ValueError, match="shared ids"):
        check_no_overlap({"train": {"1": "a"}, "test": {"1": "b"}})


def test_drop_overlapping_removes_repeated_id_and_text():
    train = {"1": "same question", "2": "unique", "3": "Same  Question", "7": "other"}
    test = {"7": "different text", "9": "same question"}
    kept, dropped = drop_overlapping(train, [test])
    assert kept == {"2": "unique"}
    assert dropped == ["1", "3", "7"]
    check_no_overlap({"train": kept, "test": test})


def test_overlap_check_catches_same_text_under_new_id():
    with pytest.raises(ValueError, match="shared texts"):
        check_no_overlap({"train": {"1": "Kisan  Andolan"}, "test": {"9": "kisan andolan"}})
