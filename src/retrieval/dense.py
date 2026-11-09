"""Dense retrieval with a sentence-embedding model (SPECS §6.1).

A **bi-encoder** turns every passage and every query into one vector
(768 numbers for multilingual-e5-base). Relevance is the cosine similarity of
the two vectors; with unit-length vectors that is just the dot product

    s(q, d) = <e_q, e_d>,   ||e_q|| = ||e_d|| = 1

so search is one matrix product of the query vectors with all passage vectors,
followed by taking the top-k scores (exact search, no approximation).

e5 models were trained with text prefixes and expect them at use time:
``"query: "`` before queries and ``"passage: "`` before passages. The model
files ship with *empty* default prompts, so the prefixes are always passed
explicitly here.

The corpus is encoded once and cached on disk in shards (``shard_000.npy`` +
``shard_000.txt`` with its docnos). Shards that exist are skipped, so an
interrupted encoding resumes where it stopped. Vectors are stored as float16
(half the disk space) and searched in float32, so scores have enough precision
that ties between passages are rare.
"""

from collections.abc import Iterable, Iterator
from itertools import islice
from pathlib import Path

import numpy as np
import torch

from src.data.load import Queries
from src.utils.runs import DEFAULT_TOP_K, Run

QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "


def load_model(name: str, device: str = "cuda", fp16: bool = True):
    """Load a sentence-transformers model, in half precision on the GPU by default."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(name, device=device)
    if fp16:
        model.half()
    return model


def encode(model, texts: list[str], *, prefix: str, max_length: int, batch_size: int,
           progress: bool = False) -> np.ndarray:
    """Unit-length float16 vectors for ``texts``; longer inputs are cut at ``max_length`` tokens."""
    model.max_seq_length = max_length
    vectors = model.encode(
        texts, prompt=prefix, batch_size=batch_size, normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=progress,
    )
    return vectors.astype(np.float16)


def _batches(items: Iterable, size: int) -> Iterator[list]:
    iterator = iter(items)
    while batch := list(islice(iterator, size)):
        yield batch


def encode_corpus(model, docs: Iterable[dict[str, str]], out_dir: str | Path, *,
                  shard_size: int, max_length: int, batch_size: int) -> int:
    """Encode ``{"docno", "text"}`` dicts into shards in ``out_dir``; returns the
    number of shards encoded now (existing shards are kept and skipped)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    encoded = 0
    for i, shard in enumerate(_batches(docs, shard_size)):
        vectors_path, docnos_path = out_dir / f"shard_{i:03d}.npy", out_dir / f"shard_{i:03d}.txt"
        if vectors_path.exists() and docnos_path.exists():
            continue
        vectors = encode(model, [d["text"] for d in shard], prefix=PASSAGE_PREFIX,
                         max_length=max_length, batch_size=batch_size, progress=True)
        np.save(vectors_path, vectors)
        # The docnos file is written last: a shard counts as done only when both exist.
        docnos_path.write_text("".join(f"{d['docno']}\n" for d in shard), encoding="utf-8")
        encoded += 1
        print(f"shard {i}: {len(shard):,} passages encoded")
    return encoded


def load_corpus(out_dir: str | Path) -> tuple[list[str], np.ndarray]:
    """All cached shards, in order: (docnos, float16 matrix of shape [n_docs, dim])."""
    out_dir = Path(out_dir)
    shard_files = sorted(out_dir.glob("shard_*.npy"))
    if not shard_files:
        raise FileNotFoundError(f"no encoded shards in {out_dir} (encode the corpus first)")
    docnos: list[str] = []
    vectors = []
    for vectors_path in shard_files:
        shard_docnos = vectors_path.with_suffix(".txt").read_text(encoding="utf-8").splitlines()
        shard_vectors = np.load(vectors_path)
        if len(shard_docnos) != len(shard_vectors):
            raise ValueError(f"{vectors_path.name}: {len(shard_vectors)} vectors, {len(shard_docnos)} docnos")
        docnos += shard_docnos
        vectors.append(shard_vectors)
    return docnos, np.concatenate(vectors)


def search(query_vectors: np.ndarray, qids: list[str], doc_vectors: torch.Tensor,
           docnos: list[str], top_k: int = DEFAULT_TOP_K, chunk: int = 64) -> Run:
    """Exact top-k search by dot product. ``doc_vectors`` is a float32 tensor
    (on the GPU for speed); queries are processed ``chunk`` at a time."""
    run: Run = {}
    k = min(top_k, len(docnos))
    for start in range(0, len(qids), chunk):
        q = torch.from_numpy(query_vectors[start:start + chunk].astype(np.float32))
        scores = q.to(doc_vectors.device) @ doc_vectors.T
        top_scores, top_index = torch.topk(scores, k, dim=1)
        for row, qid in enumerate(qids[start:start + chunk]):
            run[qid] = {
                docnos[j]: float(s)
                for j, s in zip(top_index[row].tolist(), top_scores[row].tolist(), strict=True)
            }
    return run


def retrieve(model, queries: Queries, doc_vectors: torch.Tensor, docnos: list[str], *,
             max_length: int, batch_size: int, top_k: int = DEFAULT_TOP_K) -> Run:
    """Encode the queries (as given, with the query prefix) and search the corpus."""
    qids = list(queries)
    vectors = encode(model, [queries[q] for q in qids], prefix=QUERY_PREFIX,
                     max_length=max_length, batch_size=batch_size)
    return search(vectors, qids, doc_vectors, docnos, top_k)
