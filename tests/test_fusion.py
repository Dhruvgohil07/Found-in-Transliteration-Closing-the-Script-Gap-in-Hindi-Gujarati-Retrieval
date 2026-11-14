"""Tests for src/retrieval/fusion.py (RRF and score interpolation, Phases 6 and 8)."""

import pytest

from src.retrieval.fusion import fuse, interpolate, min_max, ranked, rrf

# Two toy runs (scores on different scales, as from BM25 and a dense model)
RUN_A = {"q1": {"d1": 12.0, "d2": 9.5, "d3": 3.1}}
RUN_B = {"q1": {"d3": 0.9, "d4": 0.8, "d1": 0.2}, "q2": {"d5": 1.0}}


# --- RRF ---------------------------------------------------------------------

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
    from ranx import Run
    from ranx import fuse as ranx_fuse

    run_b = {"q1": RUN_B["q1"]}
    ours = rrf([RUN_A, run_b], k=60, top_k=None)
    theirs = ranx_fuse([Run(RUN_A), Run(run_b)], method="rrf", params={"k": 60}).to_dict()
    assert set(theirs) == set(ours)
    for qid in ours:
        assert theirs[qid] == pytest.approx(ours[qid])


# --- score interpolation -----------------------------------------------------

def test_min_max():
    assert min_max({"a": 10.0, "b": 5.0, "c": 0.0}) == pytest.approx({"a": 1.0, "b": 0.5, "c": 0.0})
    assert min_max({"a": 3.0}) == {"a": 1.0}                 # all equal: each is the best
    assert min_max({}) == {}


def test_interpolate_matches_hand_computed_scores():
    fused = interpolate([RUN_A, RUN_B], weights=[0.3, 0.7], top_k=None)
    # A normalized: d1 1, d2 (9.5-3.1)/8.9, d3 0;  B normalized: d3 1, d4 0.6/0.7, d1 0
    assert fused["q1"] == pytest.approx({
        "d1": 0.3 * 1 + 0.7 * 0,
        "d2": 0.3 * (6.4 / 8.9),               # missing from B: gets 0 from it
        "d3": 0.3 * 0 + 0.7 * 1,
        "d4": 0.7 * (0.6 / 0.7),
    })
    assert fused["q2"] == pytest.approx({"d5": 0.7})           # single passage normalized to 1


def test_interpolate_extremes_keep_one_runs_order():
    only_b = interpolate([RUN_A, RUN_B], weights=[0.0, 1.0], top_k=None)["q1"]
    assert ranked(only_b)[:3] == ranked(RUN_B["q1"])          # alpha = 0: the dense order
    only_a = interpolate([RUN_A, RUN_B], weights=[1.0, 0.0], top_k=None)["q1"]
    assert ranked(only_a)[:3] == ranked(RUN_A["q1"])          # alpha = 1: the lexical order


def test_interpolate_rejects_bad_weights():
    with pytest.raises(ValueError):
        interpolate([RUN_A, RUN_B], weights=[0.5])
    with pytest.raises(ValueError):
        interpolate([RUN_A, RUN_B], weights=[0.6, 0.6])
    with pytest.raises(ValueError):
        interpolate([RUN_A, RUN_B], weights=[1.5, -0.5])


def test_interpolate_agrees_with_ranx():
    """ranx's min-max normalization + weighted sum gives the same scores (on
    lists with more than one distinct score, where the two conventions agree)."""
    from ranx import Run
    from ranx import fuse as ranx_fuse

    run_b = {"q1": RUN_B["q1"]}
    ours = interpolate([RUN_A, run_b], weights=[0.3, 0.7], top_k=None)
    theirs = ranx_fuse([Run(RUN_A), Run(run_b)], norm="min-max", method="wsum",
                       params={"weights": [0.3, 0.7]}).to_dict()
    assert theirs["q1"] == pytest.approx(ours["q1"])


def test_fuse_dispatches_by_method():
    assert fuse([RUN_A, RUN_B], "rrf", k=10) == rrf([RUN_A, RUN_B], k=10)
    assert fuse([RUN_A, RUN_B], "interp", alpha=0.3) == interpolate([RUN_A, RUN_B], [0.3, 0.7])
    with pytest.raises(ValueError):
        fuse([RUN_A, RUN_B, RUN_A], "interp", alpha=0.3)     # alpha only for two runs
    with pytest.raises(ValueError):
        fuse([RUN_A], "combmnz")
