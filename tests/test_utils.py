"""Unit tests for src/utils: seeds, config loading and run-file I/O."""

import random

import numpy as np
import pytest
import torch

from src.utils.config import REPO_ROOT, load_config, resolve_path
from src.utils.runs import read_run, run_name, write_run
from src.utils.seed import set_seed

# --- seed -------------------------------------------------------------------

def _draw() -> tuple[float, float, list[float]]:
    return random.random(), float(np.random.rand()), torch.rand(3).tolist()


def test_set_seed_repeats_draws():
    set_seed(42)
    first = _draw()
    set_seed(42)
    assert _draw() == first


def test_different_seeds_give_different_draws():
    set_seed(1)
    first = _draw()
    set_seed(2)
    assert _draw() != first


# --- config -----------------------------------------------------------------

def test_load_config_reads_yaml(tmp_path):
    path = tmp_path / "exp.yaml"
    path.write_text("experiment: hi_bm25\nseed: 42\nretrieval:\n  k1: 1.2\n", encoding="utf-8")
    config = load_config(path)
    assert config == {"experiment": "hi_bm25", "seed": 42, "retrieval": {"k1": 1.2}}


def test_load_config_rejects_missing_seed(tmp_path):
    path = tmp_path / "exp.yaml"
    path.write_text("experiment: hi_bm25\n", encoding="utf-8")
    with pytest.raises(ValueError, match="seed"):
        load_config(path)


def test_load_config_rejects_non_mapping(tmp_path):
    path = tmp_path / "exp.yaml"
    path.write_text("- just\n- a list\n", encoding="utf-8")
    with pytest.raises(TypeError, match="mapping"):
        load_config(path)


def test_resolve_path_is_relative_to_repo_root(tmp_path):
    assert resolve_path("configs/x.yaml") == REPO_ROOT / "configs" / "x.yaml"
    assert resolve_path(tmp_path) == tmp_path


# --- run files --------------------------------------------------------------

def test_run_name_follows_convention():
    assert run_name("hi", "bm25", "roman_natural", "test") == "hi_bm25_roman_natural_test"


def test_write_then_read_round_trips(tmp_path):
    run = {"q1": {"d1": 2.5, "d2": 0.1 + 0.2}, "q2": {"d3": -1.25}}
    path = tmp_path / "runs" / "x.trec"
    write_run(run, path, tag="x")
    assert read_run(path) == run  # exact: scores are written at full precision


def test_write_run_ranks_by_score_and_breaks_ties_by_docid(tmp_path):
    run = {"q1": {"d3": 1.0, "d1": 1.0, "d2": 5.0}}
    path = tmp_path / "x.trec"
    write_run(run, path, tag="x")
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines == [
        "q1 Q0 d2 1 5.0 x",
        "q1 Q0 d1 2 1.0 x",
        "q1 Q0 d3 3 1.0 x",
    ]


def test_write_run_keeps_top_k(tmp_path):
    run = {"q1": {f"d{i}": float(i) for i in range(10)}}
    path = tmp_path / "x.trec"
    write_run(run, path, tag="x", top_k=3)
    assert read_run(path) == {"q1": {"d9": 9.0, "d8": 8.0, "d7": 7.0}}


def test_write_run_accepts_numpy_scores(tmp_path):
    path = tmp_path / "x.trec"
    write_run({"q1": {"d1": np.float32(0.5)}}, path, tag="x")
    assert read_run(path) == {"q1": {"d1": 0.5}}


def test_write_run_rejects_tag_with_space(tmp_path):
    with pytest.raises(ValueError, match="tag"):
        write_run({"q1": {"d1": 1.0}}, tmp_path / "x.trec", tag="bm25 roman")


def test_read_run_rejects_duplicate_doc(tmp_path):
    path = tmp_path / "x.trec"
    path.write_text("q1 Q0 d1 1 2.0 x\nq1 Q0 d1 2 1.0 x\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        read_run(path)
