"""Tests for src/retrieval/dual_index.py (Reciprocal Rank Fusion, Phase 6)."""

import pytest

from src.retrieval.dual_index import ranked, rrf

# Two toy runs (scores on different scales, as from two indexes)
RUN_A = {"q1": {"d1": 12.0, "d2": 9.5, "d3": 3.1}}
RUN_B = {"q1": {"d3": 0.9, "d4": 0.8, "d1": 0.2}, "q2": {"d5": 1.0}}


def test_rrf_matches_hand_computed_scores():
    fused = rrf([RUN_A, RUN_B], k=60)
    # d1: rank 1 in A, rank 3 in B; d2: rank 2 in A only; d3: rank 3 in A, rank 1 in B; d4: rank 2 in B
    assert fused["q1"] == pytest.approx({
        "d1": 1 / 61 + 1 / 63,
        "d2": 1 / 62,
        "d3": 1 / 63 + 1 / 61,
        "d4": 1 / 62,
    })
    assert fused["q2"] == pytest.approx({"d5": 1 / 61})      # a query in one run only


def test_rrf_uses_ranks_not_scores():
    scaled = {"q1": {d: s * 1000 for d, s in RUN_A["q1"].items()}}
    assert rrf([scaled, RUN_B]) == rrf([RUN_A, RUN_B])     # same ranks: identical fused scores


def test_ties_are_broken_by_docid():
    assert ranked({"b": 1.0, "a": 1.0, "c": 2.0}) == ["c", "a", "b"]
    # the dict order of tied documents does not change the fused scores
    assert rrf([{"q": {"b": 1.0, "a": 1.0}}]) == rrf([{"q": {"a": 1.0, "b": 1.0}}])


def test_top_k_and_k():
    fused = rrf([RUN_A, RUN_B], top_k=2)
    assert set(fused["q1"]) == {"d1", "d3"}                  # the two found by both runs
    small_k = rrf([RUN_A], k=0)["q1"]
    assert small_k["d1"] == pytest.approx(1.0) and small_k["d2"] == pytest.approx(0.5)
    with pytest.raises(ValueError):
        rrf([RUN_A], k=-1)


def test_rrf_agrees_with_ranx():
    """The established implementation gives the same scores on runs without ties.
    (ranx also needs every run to hold the same queries, which our runs do not
    always: Terrier returns nothing for a query none of whose terms match.)"""
    from ranx import Run, fuse

    run_b = {"q1": RUN_B["q1"]}
    ours = rrf([RUN_A, run_b], k=60, top_k=None)
    theirs = fuse([Run(RUN_A), Run(run_b)], method="rrf", params={"k": 60}).to_dict()
    assert set(theirs) == set(ours)
    for qid in ours:
        assert theirs[qid] == pytest.approx(ours[qid])
