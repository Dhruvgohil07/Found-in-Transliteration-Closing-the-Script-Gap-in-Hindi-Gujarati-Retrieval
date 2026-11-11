"""Build the BM25 (Terrier) index of a corpus.

Run from the repo root inside the conda env `script-gap`:

    python -m scripts.build_bm25_index --config configs/index_bm25_hi.yaml

Each passage is indexed as title + text (``src.data.load.full_text``), or, if
the config has a ``docs_file``, as the text in that JSON-lines file (prepared
passages, e.g. the romanized corpus of Phase 6). If the config has a
``normalize`` list (Phase 6), the Devanagari normalization key with those rules
is applied to the text first; a ``roman_normalize`` list does the same with the
romanized key, and ``ngrams: n`` then cuts every word into character n-grams
(``src.data.normalize.text_key``). All of this is recorded in the index directory
(``normalization.json``), so searches can check that their queries use the same
rules. An existing index is never overwritten unless
``--force`` is given. At the end the script prints the collection statistics
and checks that every passage of the corpus was indexed.
"""

import argparse
import json
import sys
import time
from collections.abc import Iterator

import ir_datasets

from src.data.load import full_text, iter_docs
from src.data.normalize import DEVANAGARI_RULES, ROMAN_RULES, check_rules, text_key
from src.retrieval.bm25 import (
    build_index,
    index_stats,
    open_index,
    write_index_normalization,
)
from src.utils.config import load_config, resolve_path


def read_docs_file(path) -> Iterator[dict[str, str]]:
    """``{"docno", "text"}`` dicts from a JSON-lines file, one passage per line."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            yield {"docno": doc["docno"], "text": doc["text"]}


def build_corpus_index(cfg: dict, *, force: bool = False) -> bool:
    """Build the index described by ``cfg`` (keys as in configs/index_bm25_hi.yaml).

    Returns False, without building, if the index exists and ``force`` is off.
    """
    index_dir = resolve_path(cfg["index_dir"])
    if (index_dir / "data.properties").exists() and not force:
        return False

    chosen = check_rules(cfg.get("normalize", []), DEVANAGARI_RULES)
    rules = [rule for rule in DEVANAGARI_RULES if rule in chosen]  # canonical order
    chosen = check_rules(cfg.get("roman_normalize", []), ROMAN_RULES)
    roman_rules = [rule for rule in ROMAN_RULES if rule in chosen]
    ngrams = cfg.get("ngrams")

    if "docs_file" in cfg:      # prepared passages, e.g. romanized by scripts/romanize_corpus.py
        source = read_docs_file(resolve_path(cfg["docs_file"]))
    else:
        source = ({"docno": doc["docno"], "text": full_text(doc)} for doc in iter_docs(cfg["dataset"]))
    docs = ({"docno": d["docno"], "text": text_key(d["text"], rules, roman_rules, ngrams)} for d in source)
    start = time.perf_counter()
    build_index(
        docs, index_dir,
        tokeniser=cfg["tokeniser"], stemmer=cfg["stemmer"], stopwords=cfg["stopwords"],
        docno_length=cfg["docno_length"],
    )
    write_index_normalization(index_dir, rules, roman_rules, ngrams)
    minutes = (time.perf_counter() - start) / 60

    index = open_index(index_dir)
    stats = index_stats(index)
    index.close()
    if "docs_file" in cfg:
        with open(resolve_path(cfg["docs_file"]), encoding="utf-8") as f:
            expected = sum(1 for _ in f)
    else:
        expected = ir_datasets.load(cfg["dataset"]).docs_count()
    print(f"indexed in {minutes:.1f} min -> {cfg['index_dir']} (rules: {', '.join(rules) or 'none'}; "
          f"romanized rules: {', '.join(roman_rules) or 'none'}; n-grams: {ngrams or 'none'})")
    for name, value in stats.items():
        print(f"  {name}: {value:,.1f}" if isinstance(value, float) else f"  {name}: {value:,}")
    size_mb = sum(p.stat().st_size for p in index_dir.rglob("*") if p.is_file()) / 2**20
    print(f"  size on disk: {size_mb:,.1f} MB")
    if stats["documents"] != expected:
        raise RuntimeError(f"indexed {stats['documents']:,} documents, corpus has {expected:,}")
    print(f"all {expected:,} passages indexed")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--force", action="store_true", help="overwrite an existing index")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if not build_corpus_index(cfg, force=args.force):
        sys.exit(f"index exists, not overwriting: {cfg['index_dir']} (use --force)")


if __name__ == "__main__":
    main()
