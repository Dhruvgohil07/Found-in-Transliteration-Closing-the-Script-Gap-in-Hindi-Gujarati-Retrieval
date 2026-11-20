"""The demo's search pipeline: one query, four methods side by side (Phase 11).

Every column is defined in ``configs/demo.yaml`` by an existing experiment
config, so the demo runs exactly the settings behind the reported results
(index, BM25 k1/b, normalization rules, k, model, embeddings, fusion alpha):

- ``{"config": path, "query": kind}``: one retriever; ``kind`` says which form
  of the query it gets: ``raw`` (as typed), ``bt1`` (top-1 back-transliteration)
  or ``weighted`` (weighted top-k back-transliteration, BM25 only, k from the
  config's ``backtranslit.k``);
- ``{"fusion": path}``: a hybrid; its components and fusion method come from
  the fusion config, and each component gets the query form the experiments
  gave it (a component with query set ``{set}_bt1`` gets ``bt1``, a BM25
  config with a ``backtranslit`` block gets ``weighted``, any other ``raw``).

Each column has a ``romanized`` and a ``native`` definition: a query typed in
native script (Devanagari / Gujarati) needs no back-transliteration and runs the
native-query configs of the experiments instead.

Retrievers are loaded once and kept (an index, a model or an embedding matrix
can be shared by several columns); within one search, a component used by two
columns (e.g. hybrid A's lexical run) is computed once.
"""

import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from src.data.backtranslit import (
    Candidates,
    load_candidates,
    top1_query,
    weighted_query,
)
from src.data.load import read_qrels, read_queries
from src.data.normalize import normalize_weighted, text_key
from src.data.romanize import NATIVE_SCRIPT
from src.demo.xlit_client import DEFAULT_URL, candidates_for
from src.retrieval.fusion import fuse
from src.utils.config import load_config, resolve_path

QUERY_KINDS = ("raw", "bt1", "weighted")
QID = "q"   # the single query of a demo search
# A passage word: a run of characters that are not whitespace or punctuation
# (Python's \w would split Indic words at their vowel signs).
_WORD = re.compile(r"[^\s.,;:!?()\[\]{}\"'“”‘’/\\|।॥\-–—]+")


def is_native(query: str) -> bool:
    """True if the query contains Devanagari or Gujarati letters."""
    return bool(NATIVE_SCRIPT.search(query))


def component_kind(component, cfg: dict) -> str:
    """The query form a fusion component got in the experiments (see module docstring)."""
    if isinstance(component, dict) and str(component.get("query_set", "")).endswith("_bt1"):
        return "bt1"
    return "weighted" if "backtranslit" in cfg else "raw"


def highlight(text: str, terms: set[str], rules: list[str] = ()) -> list[tuple[str, bool]]:
    """Split ``text`` into ``(segment, matched)`` pieces: a word is matched if its
    normalized form (``rules``, lower-cased) is one of ``terms``. The pieces join
    back to ``text`` exactly."""
    out, last = [], 0
    for m in _WORD.finditer(text):
        if m.start() > last:
            out.append((text[last:m.start()], False))
        key = text_key(m.group(), rules).lower() if rules else m.group().lower()
        out.append((m.group(), key in terms))
        last = m.end()
    if last < len(text):
        out.append((text[last:], False))
    return out


@dataclass
class ColumnResult:
    title: str
    ranking: list[tuple[str, float]]          # (docno, score), best first
    seconds: float
    terms: set[str] = field(default_factory=set)   # query terms to highlight
    rules: list[str] = field(default_factory=list)  # normalization the terms went through


@dataclass
class SearchResult:
    query: str
    native: bool
    bt1: str                                  # top-1 back-transliteration ("" for native queries)
    live_words: list[str]                     # words back-transliterated by the live service
    columns: list[ColumnResult]
    judged: dict | None = None                # {"qid", "split", "relevant"} if the query is a known one


class Retrievers:
    """Loads and keeps BM25 indexes, dense models and embedding matrices."""

    def __init__(self) -> None:
        self.indexes: dict[str, object] = {}
        self.models: dict[str, object] = {}
        self.embeddings: dict[str, tuple] = {}

    def index(self, index_dir: str):
        if index_dir not in self.indexes:
            from src.retrieval.bm25 import open_index
            self.indexes[index_dir] = open_index(resolve_path(index_dir))
        return self.indexes[index_dir]

    def dense(self, cfg: dict):
        import torch

        from src.retrieval.dense import load_corpus, load_model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if cfg["model"] not in self.models:
            self.models[cfg["model"]] = load_model(cfg["model"], device=device, fp16=cfg["fp16"] and device == "cuda")
        if cfg["embeddings_dir"] not in self.embeddings:
            docnos, vectors = load_corpus(resolve_path(cfg["embeddings_dir"]))
            self.embeddings[cfg["embeddings_dir"]] = (docnos, torch.from_numpy(vectors).to(device).float())
        return self.models[cfg["model"]], self.embeddings[cfg["embeddings_dir"]]


class Pipeline:
    """Search one language with the columns of its ``configs/demo.yaml`` entry."""

    def __init__(self, lang: str, lang_cfg: dict, xlit_url: str = DEFAULT_URL,
                 retrievers: Retrievers | None = None, fetch=None) -> None:
        self.lang = lang
        self.cfg = lang_cfg
        self.xlit_url = xlit_url
        self.fetch = fetch                      # None: the live service (replaceable in tests)
        self.frozen: Candidates = load_candidates(resolve_path(lang_cfg["candidates_file"]))
        self.retrievers = retrievers or Retrievers()
        self.configs: dict[str, dict] = {}
        self.known = self._known_queries(lang_cfg.get("known_queries", {}))

    # --- configs and known queries ---------------------------------------------

    def config(self, path: str) -> dict:
        if path not in self.configs:
            self.configs[path] = load_config(path)
        return self.configs[path]

    @staticmethod
    def _known_queries(known_cfg: dict) -> dict[str, dict]:
        """``{query text (lower-cased, spaces collapsed): {"qid", "split", "relevant"}}``
        for every tune/test query in any of its forms (native and romanized)."""
        known: dict[str, dict] = {}
        for split, spec in known_cfg.items():
            qrels = read_qrels(resolve_path(spec["qrels"]))
            for path in spec["queries"]:
                if not Path(resolve_path(path)).exists():
                    continue
                for qid, text in read_queries(resolve_path(path)).items():
                    relevant = sorted(d for d, r in qrels.get(qid, {}).items() if r > 0)
                    known[" ".join(text.lower().split())] = {"qid": qid, "split": split, "relevant": relevant}
        return known

    # --- one retriever ----------------------------------------------------------

    def _run(self, path: str, kind: str, query: str, bt1: str, candidates: Candidates,
             top_k: int) -> tuple[dict[str, float], set[str], list[str]]:
        """Scores of one retriever, the terms it searched for, and their normalization."""
        if kind not in QUERY_KINDS:
            raise ValueError(f"unknown query kind {kind!r}; known: {QUERY_KINDS}")
        cfg = self.config(path)
        text = bt1 if kind == "bt1" else query
        if "embeddings_dir" in cfg:                                   # dense
            from src.retrieval.dense import retrieve
            model, (docnos, vectors) = self.retrievers.dense(cfg)
            run = retrieve(model, {QID: text}, vectors, docnos, max_length=cfg["query_max_length"],
                           batch_size=1, top_k=top_k)
            return run.get(QID, {}), {w.lower() for w in _WORD.findall(text)}, []
        from src.retrieval.bm25 import retrieve, retrieve_weighted
        index, params = self.retrievers.index(cfg["index_dir"]), cfg["bm25"]
        rules = list(cfg.get("normalize", []))
        if kind == "weighted":
            weighted = weighted_query(query, candidates, cfg["backtranslit"]["k"])
            if rules:
                weighted = normalize_weighted(weighted, rules)
            run = retrieve_weighted(index, {QID: weighted}, k1=params["k1"], b=params["b"],
                                    top_k=top_k, floor_idf=params["floor_idf"])
            return run.get(QID, {}), {t.lower() for t in weighted}, rules
        roman_rules, ngrams = cfg.get("roman_normalize", []), cfg.get("ngrams")
        if rules or roman_rules or ngrams:
            text = text_key(text, rules, roman_rules, ngrams)
        run = retrieve(index, {QID: text}, k1=params["k1"], b=params["b"], top_k=top_k,
                       floor_idf=params["floor_idf"])
        return run.get(QID, {}), {w.lower() for w in _WORD.findall(text)}, rules

    # --- a search ---------------------------------------------------------------

    def search(self, query: str, top_k: int = 1000) -> SearchResult:
        query = " ".join(query.split())
        native = is_native(query)
        candidates: Candidates = {}
        live: list[str] = []
        bt1 = ""
        if not native:
            kwargs = {"fetch": self.fetch} if self.fetch else {}
            candidates, live = candidates_for(query, self.lang, self.frozen, self.xlit_url, **kwargs)
            bt1 = top1_query(query, candidates)

        cache: dict[tuple[str, str], tuple] = {}

        def run(path: str, kind: str):
            if (path, kind) not in cache:
                start = time.perf_counter()
                result = self._run(path, kind, query, bt1, candidates, top_k)
                cache[path, kind] = (*result, time.perf_counter() - start)
            return cache[path, kind]

        columns = []
        for column in self.cfg["columns"]:
            spec = column["native" if native else "romanized"]
            if "fusion" in spec:
                fcfg = self.config(spec["fusion"])
                parts = []
                for component in fcfg["components"]:
                    path = component["config"] if isinstance(component, dict) else component
                    parts.append(run(path, component_kind(component, self.config(path))))
                start = time.perf_counter()
                params = {k: v for k, v in fcfg["fusion"].items() if k != "method"}
                fused = fuse([{QID: p[0]} for p in parts], fcfg["fusion"]["method"], top_k=top_k, **params)
                scores = fused.get(QID, {})
                terms = set().union(*(p[1] for p in parts))
                rules = sorted({r for p in parts for r in p[2]})
                # its components' own time (even if another column computed them) + fusing
                seconds = sum(p[3] for p in parts) + time.perf_counter() - start
            else:
                scores, terms, rules, seconds = run(spec["config"], spec["query"])
            ranking = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
            columns.append(ColumnResult(column["title"], ranking, seconds, terms, rules))

        judged = self.known.get(query.lower())
        return SearchResult(query, native, bt1, live, columns, judged)
