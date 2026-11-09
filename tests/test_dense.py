"""Tests for src/retrieval/dense.py. Search and caching use toy vectors; one
test loads the small e5 model on the CPU to check prefixes and normalisation."""

import numpy as np
import pytest
import torch

from src.retrieval.dense import (
    PASSAGE_PREFIX,
    QUERY_PREFIX,
    encode,
    encode_corpus,
    load_corpus,
    search,
)


def unit(rows: list[list[float]]) -> np.ndarray:
    array = np.array(rows, dtype=np.float32)
    return array / np.linalg.norm(array, axis=1, keepdims=True)


def test_search_ranks_by_dot_product():
    docs = torch.from_numpy(unit([[1, 0], [0.8, 0.6], [0, 1]]))
    queries = unit([[1, 0], [0, 1]])
    run = search(queries, ["qa", "qb"], docs, ["d1", "d2", "d3"], top_k=2)
    assert list(run["qa"]) == ["d1", "d2"]            # cos 1.0, then 0.8
    assert run["qa"]["d2"] == pytest.approx(0.8)
    assert list(run["qb"]) == ["d3", "d2"]            # cos 1.0, then 0.6
    assert "d3" not in run["qa"]                      # only the top 2 are kept


def test_search_in_chunks_gives_the_same_run():
    rng = np.random.default_rng(42)
    docs = unit(rng.normal(size=(50, 8)).tolist())
    queries = unit(rng.normal(size=(7, 8)).tolist())
    qids, docnos = [f"q{i}" for i in range(7)], [f"d{i}" for i in range(50)]
    whole = search(queries, qids, torch.from_numpy(docs), docnos, top_k=10, chunk=64)
    chunked = search(queries, qids, torch.from_numpy(docs), docnos, top_k=10, chunk=3)
    # Same rankings; scores may differ in the last float32 bit (~1e-7), because a
    # different matrix shape changes the order in which products are summed.
    # Bit-identical runs therefore need a fixed chunk size (the default, 64).
    for qid in qids:
        assert list(whole[qid]) == list(chunked[qid])
        assert list(whole[qid].values()) == pytest.approx(list(chunked[qid].values()), abs=1e-6)


class FakeModel:
    """Stands in for a sentence-transformers model: records calls, returns fixed vectors."""

    def __init__(self):
        self.calls = []
        self.max_seq_length = None

    def encode(self, texts, prompt, batch_size, normalize_embeddings, convert_to_numpy,
               show_progress_bar):
        self.calls.append((prompt, list(texts), self.max_seq_length))
        return np.ones((len(texts), 4), dtype=np.float32) / 2  # unit length


def test_encode_corpus_writes_shards_and_resumes(tmp_path):
    docs = [{"docno": f"d{i}", "text": f"text {i}"} for i in range(5)]
    model = FakeModel()
    assert encode_corpus(model, docs, tmp_path, shard_size=2, max_length=256, batch_size=8) == 3
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "shard_000.npy", "shard_000.txt", "shard_001.npy", "shard_001.txt",
        "shard_002.npy", "shard_002.txt",
    ]
    assert model.calls[0] == (PASSAGE_PREFIX, ["text 0", "text 1"], 256)

    (tmp_path / "shard_001.txt").unlink()  # an interrupted shard: vectors without docnos
    model.calls.clear()
    assert encode_corpus(model, docs, tmp_path, shard_size=2, max_length=256, batch_size=8) == 1
    assert [call[1] for call in model.calls] == [["text 2", "text 3"]]  # only the missing shard

    docnos, vectors = load_corpus(tmp_path)
    assert docnos == [f"d{i}" for i in range(5)]
    assert vectors.shape == (5, 4) and vectors.dtype == np.float16


def test_load_corpus_without_shards_fails(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_corpus(tmp_path)


def test_e5_prefixes_and_unit_length():
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("intfloat/multilingual-e5-small", device="cpu")
    texts = ["किसान आंदोलन", "kisan andolan"]
    vectors = encode(model, texts, prefix=QUERY_PREFIX, max_length=32, batch_size=2)
    manual = model.encode([QUERY_PREFIX + t for t in texts], normalize_embeddings=True)
    assert vectors.dtype == np.float16
    assert np.linalg.norm(vectors.astype(np.float32), axis=1) == pytest.approx([1, 1], abs=1e-3)
    assert np.allclose(vectors, manual, atol=1e-3)  # prefix applied exactly once
