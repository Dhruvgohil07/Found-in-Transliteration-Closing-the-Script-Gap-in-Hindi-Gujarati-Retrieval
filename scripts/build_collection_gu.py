"""Build the Gujarati collection: IndicQA-gu judged passages + Wikipedia distractors.

Run from the repo root inside the conda env `script-gap`:

    python -m scripts.build_collection_gu --config configs/data_gu.yaml

Steps (docs/plans/10-gujarati.md, task 1):

1. Download the Gujarati files of IndicQA retrieval and the Gujarati Wikipedia
   dump from Hugging Face, each at a pinned revision (cached outside the repo).
2. Give every passage and query a short id (IndicQA uses 64-character hashes,
   too long for Terrier's docno field): judged passages ``indicqa-0001`` ...,
   queries ``gq-0001`` ..., both numbered in the sorted order of the original ids.
   ``id_map.tsv`` keeps the original ids.
3. Drop every Wikipedia article that shares a word 8-gram with a judged passage
   (it probably contains a copy of it), clean the rest and pack them into
   passages ``guwiki-<article id>#<k>`` (src/data/distractors.py). Passages
   that are mostly English, or exact repeats of an earlier passage (templated
   village stubs), are left out.
4. Write the corpus (judged passages first, then distractors in dump order),
   the labeled queries and qrels, and a meta file with every count, which
   scripts/data_stats.py puts into the data card.

Titles are left empty for every passage: the IndicQA passages have none, and
giving only the distractors a title would make the two kinds look different.
Running it again with the same config produces identical files.
"""

import argparse
import json

import pandas as pd
from huggingface_hub import hf_hub_download

from src.data.distractors import iter_distractors, latin_share, overlapping_articles
from src.data.load import write_docs, write_qrels, write_queries
from src.utils.config import load_config, resolve_path


def download(repo: str, revision: str, filename: str) -> pd.DataFrame:
    path = hf_hub_download(repo, filename, repo_type="dataset", revision=revision)
    return pd.read_parquet(path)


def short_ids(original: list[str], prefix: str) -> dict[str, str]:
    """``{original id: prefix-0001, ...}``, numbered in sorted order of the originals."""
    width = len(str(len(original)))
    return {orig: f"{prefix}-{n:0{width}d}" for n, orig in enumerate(sorted(original), start=1)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    cfg = load_config(parser.parse_args().config)
    judged_cfg = cfg["collection"]["judged"]
    wiki_cfg = cfg["collection"]["distractors"]
    out = cfg["collection"]["output"]

    # 1. judged passages, queries and qrels
    get = lambda part: download(judged_cfg["repo"], judged_cfg["revision"], judged_cfg["files"][part])
    corpus, queries, qrels = get("corpus"), get("queries"), get("qrels")
    doc_ids = short_ids(corpus["_id"].tolist(), judged_cfg["doc_prefix"])
    query_ids = short_ids(queries["_id"].tolist(), judged_cfg["query_prefix"])
    unknown = set(qrels["corpus-id"]) - set(doc_ids) | set(qrels["query-id"]) - set(query_ids)
    if unknown:
        raise ValueError(f"{len(unknown)} qrels ids are not in the corpus or queries")

    judged_docs = sorted(
        ({"docno": doc_ids[r["_id"]], "title": "", "text": r["text"].strip()} for r in corpus.to_dict("records")),
        key=lambda d: d["docno"],
    )
    labeled_queries = {query_ids[r["_id"]]: " ".join(r["text"].split()) for r in queries.to_dict("records")}
    labeled_qrels: dict[str, dict[str, int]] = {}
    for r in qrels.to_dict("records"):
        labeled_qrels.setdefault(query_ids[r["query-id"]], {})[doc_ids[r["corpus-id"]]] = int(r["score"])

    # 2. distractors
    wiki = download(wiki_cfg["repo"], wiki_cfg["revision"], wiki_cfg["file"])
    articles = dict(zip(wiki["id"].astype(str), wiki["text"]))
    titles = dict(zip(wiki["id"].astype(str), wiki["title"]))
    dropped = overlapping_articles(
        articles, (d["text"] for d in judged_docs), n=wiki_cfg["overlap_ngram"], min_shared=wiki_cfg["overlap_min_shared"]
    )
    pack = {k: wiki_cfg[k] for k in ("min_words", "max_words", "min_line_words", "min_last_words")}
    distractors = []
    used_articles = set()
    seen_texts = {d["text"] for d in judged_docs}
    n_latin = n_repeat = 0
    for article_id, k, text in iter_distractors(articles, dropped, **pack):
        if latin_share(text) > wiki_cfg["max_latin_share"]:
            n_latin += 1
        elif text in seen_texts:
            n_repeat += 1
        else:
            seen_texts.add(text)
            distractors.append({"docno": f"{wiki_cfg['doc_prefix']}-{article_id}#{k}", "title": "", "text": text})
            used_articles.add(article_id)

    # 3. write everything
    n_docs = write_docs(judged_docs + distractors, resolve_path(out["corpus"]))
    write_queries(labeled_queries, resolve_path(out["queries"]))
    write_qrels(labeled_qrels, resolve_path(out["qrels"]))
    with open(resolve_path(out["id_map"]), "w", encoding="utf-8", newline="\n") as f:
        f.write("kind\tshort_id\toriginal_id\n")
        f.writelines(f"passage\t{s}\t{o}\n" for o, s in sorted(doc_ids.items(), key=lambda x: x[1]))
        f.writelines(f"query\t{s}\t{o}\n" for o, s in sorted(query_ids.items(), key=lambda x: x[1]))

    meta = {
        "judged": {"repo": judged_cfg["repo"], "revision": judged_cfg["revision"],
                   "passages": len(judged_docs), "queries": len(labeled_queries),
                   "judgments": sum(len(v) for v in labeled_qrels.values())},
        "distractors": {"repo": wiki_cfg["repo"], "revision": wiki_cfg["revision"], "file": wiki_cfg["file"],
                        "articles_in_dump": len(articles),
                        "articles_dropped_overlap": len(dropped),
                        "articles_without_passage": len(articles) - len(dropped) - len(used_articles),
                        "articles_used": len(used_articles),
                        "passages_dropped_latin": n_latin,
                        "passages_dropped_repeat": n_repeat,
                        "passages": len(distractors),
                        "dropped_titles": sorted(titles[a] for a in dropped)},
        "corpus_passages": n_docs,
    }
    resolve_path(out["meta"]).write_text(json.dumps(meta, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"judged: {len(judged_docs)} passages, {len(labeled_queries)} queries, "
          f"{meta['judged']['judgments']} judgments")
    print(f"wikipedia: {len(articles):,} articles, {len(dropped)} dropped (overlap), "
          f"{len(used_articles):,} used -> {len(distractors):,} distractor passages "
          f"({n_latin} mostly-Latin and {n_repeat} repeated passages dropped)")
    print(f"corpus: {n_docs:,} passages -> {out['corpus']}")


if __name__ == "__main__":
    main()
