"""BM25 with PyTerrier: building and opening an inverted index (SPECS §6.1).

An **inverted index** maps every term to the documents that contain it (with
term frequencies), plus per-document lengths. BM25 then scores a document for a
query from exactly these statistics:

    BM25(q, d) = sum over t in q of  IDF(t) * tf(t,d) * (k1 + 1)
                                      / (tf(t,d) + k1 * (1 - b + b * |d| / avgdl))

How text becomes terms is fixed when the index is built, and the same
pipeline is applied to queries automatically:

- ``tokeniser="utf"``: splits on non-letter characters and keeps every script.
  Terrier's default English tokeniser drops all Devanagari (found in Phase 1).
- ``stemmer`` and ``stopwords`` both ``none`` (decided in Phase 4): Terrier's
  defaults are the English Porter stemmer and an English stopword list, which
  change romanized Hindi (``roy`` -> ``roi``) and drop words such as ``the``
  (in Hinglish the verb "were"). Hindi has no stemmer or stopwords in Terrier
  either way, so terms are indexed as written, only lower-cased.

**Queries** are first split with the same ``utf`` tokeniser
(``pt.rewrite.tokenise``) and re-joined with spaces. Terrier's query parser
treats characters such as ``+ - ^ ( ) : "`` as operators; a query like
``co2 (carbon) : test`` was parsed as *empty* and returned nothing, without an
error. Tokenising first removes all punctuation, so every query reaches BM25 as
plain terms, cut exactly the way the documents were.

**Negative IDF.** Terrier's BM25 uses the classic IDF
``log((N - df + 0.5) / (df + 0.5))``, which is *negative* for a term that occurs
in more than half of all passages. In Hindi that hits the most common function
words (के in 75.7% of MIRACL-hi passages, में, है, की, और), so a native query
containing them pushes down every passage that contains them. With
``floor_idf=True`` such terms are dropped from the query, which is the same as
flooring their IDF at 0 (a corpus-derived stopword list, no outside resource).
"""

from collections.abc import Iterable, Iterator
from pathlib import Path

import pandas as pd
import pyterrier as pt

from src.data.load import Queries
from src.utils.runs import DEFAULT_TOP_K, Run


def start_java() -> None:
    if not pt.java.started():
        pt.java.init()


def build_index(
    docs: Iterable[dict[str, str]],
    index_dir: str | Path,
    *,
    tokeniser: str = "utf",
    stemmer: str = "none",
    stopwords: str = "none",
    docno_length: int = 32,
) -> str:
    """Index ``{"docno", "text"}`` dicts into ``index_dir`` and return the index path.

    ``docno_length`` is the space Terrier reserves for each docno; a longer
    docno raises ``ValueError`` instead of being cut off silently.
    """
    start_java()

    def checked(docs: Iterable[dict[str, str]]) -> Iterator[dict[str, str]]:
        for doc in docs:
            if len(doc["docno"]) > docno_length:
                raise ValueError(f"docno {doc['docno']!r} is longer than {docno_length}")
            yield doc

    indexer = pt.terrier.IterDictIndexer(
        str(Path(index_dir).resolve()),
        meta={"docno": docno_length},
        tokeniser=tokeniser,
        stemmer=getattr(pt.terrier.TerrierStemmer, stemmer),
        stopwords=getattr(pt.terrier.TerrierStopwords, stopwords),
    )
    return str(indexer.index(checked(docs)))


def open_index(index_dir: str | Path):
    """Open an existing index (raises if ``index_dir`` holds none)."""
    start_java()
    properties = Path(index_dir) / "data.properties"
    if not properties.exists():
        raise FileNotFoundError(f"no Terrier index in {index_dir} (build it first)")
    return pt.IndexFactory.of(str(Path(index_dir).resolve()))


def index_stats(index) -> dict[str, float]:
    """The collection statistics BM25 uses: documents, distinct terms, tokens,
    and the average document length (avgdl)."""
    stats = index.getCollectionStatistics()
    return {
        "documents": stats.getNumberOfDocuments(),
        "unique_terms": stats.getNumberOfUniqueTerms(),
        "tokens": stats.getNumberOfTokens(),
        "avg_doc_length": stats.getAverageDocumentLength(),
    }


def retrieve(
    index,
    queries: Queries,
    *,
    k1: float,
    b: float,
    top_k: int = DEFAULT_TOP_K,
    tokeniser: str = "utf",
    floor_idf: bool = False,
) -> Run:
    """BM25 search for every query; returns ``{qid: {docno: score}}``.

    ``k1`` controls how fast extra occurrences of a term stop adding score
    (term-frequency saturation); ``b`` how strongly long passages are
    penalised (0 = no length normalisation, 1 = full). ``floor_idf`` drops
    query terms with negative IDF (see the module docstring).
    """
    start_java()
    topics = pd.DataFrame({"qid": list(queries), "query": list(queries.values())})
    topics = pt.rewrite.tokenise(tokeniser)(topics)
    if floor_idf:
        common = negative_idf_terms(index, {t for q in topics["query"] for t in q.split()})
        topics["query"] = [" ".join(t for t in q.split() if t not in common) for q in topics["query"]]
    topics = topics[topics["query"].str.strip() != ""]  # nothing left to search for
    if topics.empty:
        return {}
    retriever = pt.terrier.Retriever(
        index,
        wmodel="BM25",
        num_results=top_k,
        controls={"bm25.k_1": str(k1), "bm25.b": str(b)},
    )
    results = retriever(topics)
    run: Run = {}
    for qid, docno, score in zip(results["qid"], results["docno"], results["score"], strict=True):
        run.setdefault(qid, {})[docno] = float(score)
    return run


def negative_idf_terms(index, terms: set[str]) -> set[str]:
    """The given terms that occur in more than half of all passages, i.e. whose
    classic BM25 IDF ``log((N - df + 0.5) / (df + 0.5))`` is below 0."""
    n_docs = index.getCollectionStatistics().getNumberOfDocuments()
    lexicon = index.getLexicon()
    common = set()
    for term in terms:
        entry = lexicon.getLexiconEntry(term)
        if entry is not None and entry.getDocumentFrequency() > n_docs / 2:
            common.add(term)
    return common
