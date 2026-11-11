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

import json
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


NORMALIZATION_FILE = "normalization.json"


def write_index_normalization(index_dir: str | Path, rules: Iterable[str],
                              roman_rules: Iterable[str] = (), ngrams: int | None = None) -> None:
    """Record next to the index which normalization its text went through
    (Phase 6): Devanagari ``rules``, romanized ``roman_rules`` and character
    ``ngrams`` (None: whole words), so a search can check that its queries get
    the same treatment."""
    path = Path(index_dir) / NORMALIZATION_FILE
    record = {"devanagari_rules": list(rules), "roman_rules": list(roman_rules), "ngrams": ngrams}
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")


def _read_record(index_dir: str | Path) -> dict:
    path = Path(index_dir) / NORMALIZATION_FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def read_index_normalization(index_dir: str | Path) -> list[str]:
    """The Devanagari rules recorded by ``write_index_normalization``; an index
    without the file (built before Phase 6) holds the raw text, i.e. no rules."""
    return _read_record(index_dir).get("devanagari_rules", [])


def read_index_roman_normalization(index_dir: str | Path) -> list[str]:
    """The romanized rules recorded by ``write_index_normalization`` (none if not recorded)."""
    return _read_record(index_dir).get("roman_rules", [])


def read_index_ngrams(index_dir: str | Path) -> int | None:
    """The character n-gram size recorded by ``write_index_normalization`` (None: words)."""
    return _read_record(index_dir).get("ngrams")


def check_normalization(index_dir: str | Path, query_rules: Iterable[str],
                        roman_rules: Iterable[str] = (), ngrams: int | None = None) -> None:
    """Raise ``ValueError`` unless the queries are normalized with exactly the
    rules the index was built with (Devanagari and romanized) and cut into the
    same character n-grams. A mismatch would not fail on its own: the variant
    spellings would simply stop matching, and the scores would drop."""
    if read_index_ngrams(index_dir) != ngrams:
        raise ValueError(f"index {index_dir} holds n-grams of size {read_index_ngrams(index_dir)}, "
                         f"but the queries use {ngrams}")
    for kind, index_rules, used in [
        ("rules", read_index_normalization(index_dir), query_rules),
        ("romanized rules", read_index_roman_normalization(index_dir), roman_rules),
    ]:
        if set(index_rules) != set(used):
            raise ValueError(f"index {index_dir} was built with {kind} {sorted(index_rules)}, "
                             f"but the queries use {sorted(set(used))}")


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


WeightedQueries = dict[str, dict[str, float]]  # {qid: {term: weight}}


def retrieve_weighted(
    index,
    queries: WeightedQueries,
    *,
    k1: float,
    b: float,
    top_k: int = DEFAULT_TOP_K,
    tokeniser: str = "utf",
    floor_idf: bool = False,
) -> Run:
    """BM25 search with weighted query terms (Phase 5 top-k expansion).

    Each term is first cut by the index tokeniser (so it matches the index and
    contains no query-language operators), weights of identical tokens are
    added, and the query is sent in Terrier's ``term^weight`` syntax. How
    Terrier applies the weights (checked on the MIRACL-hi index): it divides
    every weight by the largest one in the query, so only ratios matter, and
    BM25 then scales the term's score by ``(k3 + 1) x / (k3 + x)`` with
    ``k3 = 8``, where ``x`` is that relative weight; nearly linear, and the
    order of the weights is always kept. (Terrier's other query language,
    ``#combine``, cannot parse Devanagari.)
    """
    start_java()
    terms = sorted({term for query in queries.values() for term in query})
    tokenised = pt.rewrite.tokenise(tokeniser)(pd.DataFrame({"qid": terms, "query": terms}))
    tokens_of = dict(zip(tokenised["qid"], tokenised["query"].str.split(), strict=True))

    weighted: dict[str, dict[str, float]] = {}
    for qid, query in queries.items():
        merged: dict[str, float] = {}
        for term, weight in query.items():
            for token in tokens_of[term]:
                merged[token] = merged.get(token, 0.0) + weight
        weighted[qid] = merged
    if floor_idf:
        common = negative_idf_terms(index, {t for q in weighted.values() for t in q})
        weighted = {qid: {t: w for t, w in q.items() if t not in common} for qid, q in weighted.items()}

    topics = pd.DataFrame({
        "qid": [qid for qid, q in weighted.items() if q],
        "query": [" ".join(f"{t}^{w:.6f}" for t, w in sorted(q.items())) for q in weighted.values() if q],
    })
    if topics.empty:
        return {}
    retriever = pt.terrier.Retriever(index, wmodel="BM25", num_results=top_k,
                                     controls={"bm25.k_1": str(k1), "bm25.b": str(b)})
    results = retriever(topics)
    run: Run = {}
    for qid, docno, score in zip(results["qid"], results["docno"], results["score"], strict=True):
        run.setdefault(qid, {})[docno] = float(score)
    return run
