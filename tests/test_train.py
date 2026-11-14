"""Tests for src/train/build_pairs.py (Phase 7 training examples), on toy queries and qrels."""

import pytest

from src.train.build_pairs import CONFIGS, attach_texts, build_examples, query_scripts

NATIVE = {f"q{i}": f"प्रश्न {i}" for i in range(10)}
ROMAN = {f"q{i}": f"prashn {i}" for i in range(10)}
QRELS = {qid: {f"{qid}_p1": 1, f"{qid}_p2": 2, f"{qid}_n1": 0, f"{qid}_n2": 0, f"{qid}_n3": 0}
         for qid in NATIVE}
QRELS["q0"] = {"q0_p1": 1, "q0_n1": 0}               # one positive, one negative
QRELS["q9"] = {"q9_p1": 1}                            # no judged negative: skipped


def test_configs_differ_only_in_the_query_script():
    built = {c: build_examples(NATIVE, ROMAN, QRELS, c, seed=42) for c in CONFIGS}
    keys = {c: [(e["qid"], e["pos_docno"], e["neg_docno"]) for e in ex] for c, ex in built.items()}
    assert keys["N"] == keys["50"] == keys["T"]       # same examples, same negatives
    assert all(e["query"] == NATIVE[e["qid"]] for e in built["N"])
    assert all(e["query"] == ROMAN[e["qid"]] for e in built["T"])
    assert len(built["N"]) == 1 + 8 * 2               # q0: 1 positive; q1-q8: 2; q9 skipped


def test_fifty_splits_queries_in_half_by_query():
    scripts = query_scripts(NATIVE, "50", seed=42)
    assert sorted(scripts.values()).count("native") == 5
    ex = build_examples(NATIVE, ROMAN, QRELS, "50", seed=42)
    for e in ex:                                       # every example of a query has the same script
        assert e["script"] == scripts[e["qid"]]
        assert e["query"] == (NATIVE if e["script"] == "native" else ROMAN)[e["qid"]]


def test_negatives_are_judged_non_relevant_and_seeded():
    ex = build_examples(NATIVE, ROMAN, QRELS, "N", seed=42)
    for e in ex:
        assert QRELS[e["qid"]][e["pos_docno"]] > 0 and QRELS[e["qid"]][e["neg_docno"]] == 0
    by_query = {}
    for e in ex:
        by_query.setdefault(e["qid"], []).append(e["neg_docno"])
    assert all(len(set(negs)) == len(negs) for negs in by_query.values())   # different negatives
    assert build_examples(NATIVE, ROMAN, QRELS, "N", seed=42) == ex         # same seed, same choice
    reversed_native = dict(reversed(list(NATIVE.items())))
    assert build_examples(reversed_native, ROMAN, QRELS, "N", seed=42) == ex  # order does not matter
    assert build_examples(NATIVE, ROMAN, QRELS, "N", seed=7) != ex           # (with 3 negatives each)


def test_load_examples_keeps_the_column_order_the_loss_expects(tmp_path):
    import json

    from src.train.train_dense import PROMPTS, load_examples

    rows = [{"qid": "q1", "script": "roman", "query": "kisan", "pos_docno": "p", "neg_docno": "n",
             "positive": "किसान आंदोलन", "negative": "मौसम"}]
    path = tmp_path / "train.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    dataset = load_examples(path)
    assert dataset.column_names == ["anchor", "positive", "negative"]   # anchor first, then d+, d-
    assert dataset[0] == {"anchor": "kisan", "positive": "किसान आंदोलन", "negative": "मौसम"}
    assert PROMPTS == {"anchor": "query: ", "positive": "passage: ", "negative": "passage: "}


def test_checkpoints_are_found_in_training_order(tmp_path):
    from scripts.select_checkpoint import checkpoints

    for name in ["checkpoint-117", "checkpoint-39", "checkpoint-78", "runs", "checkpoint-x"]:
        (tmp_path / name).mkdir()
    assert [step for step, _ in checkpoints(tmp_path)] == [39, 78, 117]   # numeric, not text order


def test_errors_and_texts():
    with pytest.raises(ValueError):
        query_scripts(NATIVE, "M", seed=42)
    with pytest.raises(ValueError):
        build_examples(NATIVE, {"q0": "x"}, QRELS, "N", seed=42)
    ex = build_examples({"q0": "a"}, {"q0": "b"}, QRELS, "T", seed=42)
    rows = attach_texts(ex, {"q0_p1": "positive text", "q0_n1": "negative text"})
    assert rows == [{"qid": "q0", "script": "roman", "query": "b", "pos_docno": "q0_p1",
                     "neg_docno": "q0_n1", "positive": "positive text", "negative": "negative text"}]
    with pytest.raises(KeyError):
        attach_texts(ex, {"q0_p1": "positive text"})
