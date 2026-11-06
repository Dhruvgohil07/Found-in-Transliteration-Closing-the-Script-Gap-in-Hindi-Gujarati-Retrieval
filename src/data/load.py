"""Loading collections from ir_datasets, and reading/writing our processed files.

Everything is returned in one format, whatever the source:

- queries: ``{qid: text}``
- qrels:   ``{qid: {docid: relevance}}``, the shape ``ir_measures`` and ``ranx`` use
- docs:    an iterator of ``{"docno", "title", "text"}`` dicts (``docno`` is the
  field name PyTerrier's indexer expects)

Raw downloads stay in the ir_datasets cache (``~/.ir_datasets`` by default),
outside the repo. Our own processed files (SPECS §5.4) are UTF-8 TSV:

- queries: ``qid<TAB>text``
- qrels:   ``qid<TAB>0<TAB>docid<TAB>relevance``, TREC qrels with tabs, so the
  file is valid TSV and still reads with ``ir_measures`` and ``trec_eval``

ir_datasets opens some text files without naming an encoding, so on Windows
(default cp1252) Devanagari queries fail to decode. Python must run in UTF-8
mode (``PYTHONUTF8=1`` or ``python -X utf8``); the loaders check this first.
"""

import sys
from collections.abc import Iterator
from pathlib import Path

import ir_datasets
import ir_measures

Queries = dict[str, str]
Qrels = dict[str, dict[str, int]]


def require_utf8_mode() -> None:
    """Fail early, with the fix in the message, instead of a decode error halfway."""
    if not sys.flags.utf8_mode:
        raise RuntimeError(
            "Python is not in UTF-8 mode, so ir_datasets cannot read Hindi text on "
            "Windows. Set PYTHONUTF8=1 in the environment or run 'python -X utf8 ...'."
        )


def load_queries(dataset_id: str) -> Queries:
    """Queries of an ir_datasets dataset, e.g. ``miracl/hi/dev``."""
    require_utf8_mode()
    return {q.query_id: q.text for q in ir_datasets.load(dataset_id).queries_iter()}


def load_qrels(dataset_id: str) -> Qrels:
    """Relevance judgments of an ir_datasets dataset, including non-relevant (0) ones."""
    require_utf8_mode()
    qrels: Qrels = {}
    for qrel in ir_datasets.load(dataset_id).qrels_iter():
        qrels.setdefault(qrel.query_id, {})[qrel.doc_id] = qrel.relevance
    return qrels


def iter_docs(dataset_id: str) -> Iterator[dict[str, str]]:
    """Documents of an ir_datasets dataset, one dict at a time (the corpus is not
    held in memory)."""
    require_utf8_mode()
    for doc in ir_datasets.load(dataset_id).docs_iter():
        yield {"docno": doc.doc_id, "title": doc.title, "text": doc.text}


def get_docs(dataset_id: str, docnos: set[str]) -> dict[str, dict[str, str]]:
    """The requested documents, by docno, found with one pass over the corpus.

    A single scan is simpler than building ir_datasets' random-access docstore,
    and fast enough for occasional lookups.
    """
    return {doc["docno"]: doc for doc in iter_docs(dataset_id) if doc["docno"] in docnos}


def full_text(doc: dict[str, str]) -> str:
    """Title and passage text joined: what gets indexed and encoded.

    MIRACL passages are short pieces of Wikipedia articles; the article title
    often holds the key entity, so it is kept with the passage.
    """
    return f"{doc['title']} {doc['text']}".strip()


def _check_field(value: str, what: str) -> None:
    if "\t" in value or "\n" in value or "\r" in value:
        raise ValueError(f"{what} contains a tab or newline: {value!r}")


def write_queries(queries: Queries, path: str | Path) -> None:
    """Write queries as ``qid<TAB>text``, sorted by qid."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for qid in sorted(queries):
            _check_field(qid, "query id")
            _check_field(queries[qid], f"query {qid}")
            f.write(f"{qid}\t{queries[qid]}\n")


def read_queries(path: str | Path) -> Queries:
    """Read a ``qid<TAB>text`` file; raises ``ValueError`` on a malformed or repeated line."""
    queries: Queries = {}
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, start=1):
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 2:
                raise ValueError(f"{path}:{n}: expected 'qid<TAB>text', got {line!r}")
            qid, text = parts
            if qid in queries:
                raise ValueError(f"{path}:{n}: duplicate query id {qid!r}")
            queries[qid] = text
    return queries


def write_qrels(qrels: Qrels, path: str | Path) -> None:
    """Write qrels as ``qid<TAB>0<TAB>docid<TAB>relevance``, sorted by qid then docid."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for qid in sorted(qrels):
            f.writelines(f"{qid}\t0\t{docid}\t{qrels[qid][docid]}\n" for docid in sorted(qrels[qid]))


def read_qrels(path: str | Path) -> Qrels:
    """Read a qrels file (tab- or space-separated TREC format)."""
    qrels: Qrels = {}
    for qrel in ir_measures.read_trec_qrels(str(path)):
        qrels.setdefault(qrel.query_id, {})[qrel.doc_id] = qrel.relevance
    return qrels
