"""Compute collection and split statistics and write them into the data card.

Run from the repo root inside the conda env `script-gap`, after prepare_data:

    python -m scripts.data_stats --config configs/data_hi.yaml

Every number is computed from the data: the corpus is streamed from
ir_datasets, and the splits are read from the files prepare_data wrote. The
result replaces the block between the ``stats`` markers in the data card, so
no statistic is ever copied by hand. Lengths are given in words (split on
whitespace) and in tokens of the main dense model's tokenizer, counted on the
raw text (without the e5 "query: "/"passage: " prefix or special tokens).
"""

import argparse
import re
from collections.abc import Iterable

import numpy as np
from transformers import AutoTokenizer

from src.data.load import full_text, iter_docs, read_qrels, read_queries
from src.utils.config import load_config, resolve_path

SPLITS = ["tune", "test", "train"]
START, END = "<!-- stats:start -->", "<!-- stats:end -->"
TOKEN_LIMIT = 256  # passage length the dense model is run with (Phase 1 benchmark)
BATCH = 4096
LATIN = re.compile(r"[A-Za-z]")


def count_tokens(tokenizer, texts: list[str]) -> list[int]:
    return [len(ids) for ids in tokenizer(texts, add_special_tokens=False)["input_ids"]]


def spread(values: Iterable[float]) -> str:
    """'mean / median / 95th percentile / max' of a list of lengths."""
    a = np.asarray(list(values))
    return f"{a.mean():.1f} / {np.median(a):.0f} / {np.percentile(a, 95):.0f} / {a.max():,}"


def collection_stats(dataset_id: str, tokenizer) -> tuple[list[str], set[str]]:
    words: list[int] = []
    tokens: list[int] = []
    docids: set[str] = set()
    batch: list[str] = []
    for doc in iter_docs(dataset_id):
        docids.add(doc["docno"])
        text = full_text(doc)
        words.append(len(text.split()))
        batch.append(text)
        if len(batch) == BATCH:
            tokens.extend(count_tokens(tokenizer, batch))
            batch = []
    tokens.extend(count_tokens(tokenizer, batch))

    articles = {docid.split("#")[0] for docid in docids}  # MIRACL docids are "article#passage"
    n_long = sum(t > TOKEN_LIMIT for t in tokens)
    rows = [
        "| | Value |",
        "|---|---|",
        f"| Passages | {len(docids):,} |",
        f"| Articles (distinct article ids) | {len(articles):,} |",
        f"| Passage length, words: mean / median / 95th pct / max | {spread(words)} |",
        f"| Passage length, e5 tokens: mean / median / 95th pct / max | {spread(tokens)} |",
        f"| Passages longer than {TOKEN_LIMIT} e5 tokens | {n_long:,} ({100 * n_long / len(tokens):.1f}%) |",
    ]
    return rows, docids


def split_stats(cfg: dict, tokenizer, docids: set[str]) -> list[str]:
    out = resolve_path(cfg["output_dir"])
    cells: dict[str, list[str]] = {}
    for name in SPLITS:
        queries = read_queries(out / f"{name}_queries.tsv")
        qrels = read_qrels(out / f"{name}_qrels.tsv")
        judged = [len(qrels.get(q, {})) for q in queries]
        relevant = [sum(r > 0 for r in qrels.get(q, {}).values()) for q in queries]
        missing = sum(d not in docids for q in qrels.values() for d in q)
        words = [len(t.split()) for t in queries.values()]
        tokens = count_tokens(tokenizer, list(queries.values()))
        latin = sum(bool(LATIN.search(t)) for t in queries.values())
        cells[name] = [
            f"{len(queries):,}",
            f"{sum(judged):,}",
            f"{np.mean(judged):.1f} ({min(judged)}–{max(judged)})",
            f"{np.mean(relevant):.1f} / {np.median(relevant):.0f} ({min(relevant)}–{max(relevant)})",
            f"{sum(r == 0 for r in relevant)}",
            f"{missing}",
            f"{np.mean(words):.1f} ({max(words)})",
            f"{np.mean(tokens):.1f} ({max(tokens)})",
            f"{latin}",
        ]
    labels = [
        "Queries",
        "Judgments",
        "Judged passages per query: mean (min–max)",
        "Relevant passages per query: mean / median (min–max)",
        "Queries with no relevant passage",
        "Judged passages missing from the corpus",
        "Query length, words: mean (max)",
        "Query length, e5 tokens: mean (max)",
        "Queries containing Latin letters",
    ]
    rows = ["| | " + " | ".join(SPLITS) + " |", "|---|" + "---|" * len(SPLITS)]
    rows += [f"| {label} | " + " | ".join(cells[s][i] for s in SPLITS) + " |" for i, label in enumerate(labels)]
    return rows


def write_block(card_path, lines: list[str]) -> None:
    text = card_path.read_text(encoding="utf-8")
    if START not in text or END not in text:
        raise ValueError(f"{card_path} has no '{START}' ... '{END}' block")
    head, rest = text.split(START, 1)
    _, tail = rest.split(END, 1)
    card_path.write_text(head + START + "\n" + "\n".join(lines) + "\n" + END + tail, encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    cfg = load_config(parser.parse_args().config)

    tokenizer = AutoTokenizer.from_pretrained(cfg["stats"]["tokenizer"])
    coll_rows, docids = collection_stats(cfg["dataset"]["docs"], tokenizer)
    lines = [
        f"Tokenizer: `{cfg['stats']['tokenizer']}`. Passages are title + text, as indexed.",
        "",
        "**Collection**",
        "",
        *coll_rows,
        "",
        "**Splits**",
        "",
        *split_stats(cfg, tokenizer, docids),
    ]
    write_block(resolve_path(cfg["stats"]["data_card"]), lines)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
