"""A passage store for the demo: docno -> title and text, in one SQLite file (Phase 11).

Search returns docnos; the demo must show the passages. Scanning the corpus for
every query (``src.data.load.get_docs``) takes about a minute for MIRACL-hi's
506K passages, so the passages are copied once into a SQLite table with the
docno as primary key, and each lookup is an index search. SQLite ships with
Python (``sqlite3``), so nothing new is installed. The file is git-ignored and
rebuilt by ``scripts/build_demo_store.py``.
"""

import sqlite3
from collections.abc import Iterable
from pathlib import Path

BATCH = 10_000


def build_store(docs: Iterable[dict[str, str]], path: str | Path) -> int:
    """Write ``{"docno", "title", "text"}`` dicts to a new SQLite file; returns the count.

    An existing file is replaced. Docnos must be unique (primary key).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.unlink(missing_ok=True)
    n = 0
    with sqlite3.connect(tmp) as db:
        db.execute("CREATE TABLE passages (docno TEXT PRIMARY KEY, title TEXT, text TEXT)")
        batch = []
        for doc in docs:
            batch.append((doc["docno"], doc["title"], doc["text"]))
            if len(batch) == BATCH:
                db.executemany("INSERT INTO passages VALUES (?, ?, ?)", batch)
                n += len(batch)
                batch = []
        db.executemany("INSERT INTO passages VALUES (?, ?, ?)", batch)
        n += len(batch)
    db.close()
    tmp.replace(path)   # only a complete file takes the final name
    return n


class PassageStore:
    """Read-only lookups in a store written by ``build_store``."""

    def __init__(self, path: str | Path) -> None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"no passage store at {path} (run scripts/build_demo_store.py)")
        # check_same_thread=False: Streamlit may call from another thread; reads only.
        self.db = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True, check_same_thread=False)

    def get(self, docnos: list[str]) -> dict[str, dict[str, str]]:
        """``{docno: {"title", "text"}}`` for the docnos found."""
        if not docnos:
            return {}
        marks = ",".join("?" * len(docnos))
        rows = self.db.execute(f"SELECT docno, title, text FROM passages WHERE docno IN ({marks})", docnos)
        return {docno: {"title": title, "text": text} for docno, title, text in rows}

    def __len__(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM passages").fetchone()[0]
