"""Write a random sample of queries with their relevant passages, for reading.

Run from the repo root inside the conda env `script-gap`, after prepare_data:

    python -m scripts.sample_queries --config configs/data_hi.yaml

The sample comes from the split named in the config (``tune``), never from
test, so looking at it cannot influence choices made on test queries. The
output is a UTF-8 Markdown file under data/ (git-ignored, since it holds
dataset text); open it in an editor rather than printing it to the console.
"""

import argparse
import random

from src.data.load import get_docs, read_qrels, read_queries
from src.utils.config import load_config, resolve_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    cfg = load_config(parser.parse_args().config)
    sample_cfg = cfg["sample"]
    if sample_cfg["split"] == "test":
        raise ValueError("sample from tune or train, never from test")

    out = resolve_path(cfg["output_dir"])
    queries = read_queries(out / f"{sample_cfg['split']}_queries.tsv")
    qrels = read_qrels(out / f"{sample_cfg['split']}_qrels.tsv")
    qids = sorted(random.Random(cfg["seed"]).sample(sorted(queries), sample_cfg["n_queries"]))

    relevant = {qid: sorted(d for d, r in qrels[qid].items() if r > 0) for qid in qids}
    docs = get_docs(cfg["dataset"]["docs"], {d for ds in relevant.values() for d in ds})

    lines = [
        f"# Sample of {len(qids)} {sample_cfg['split']} queries ({cfg['dataset']['labeled']})",
        "",
        f"Seed {cfg['seed']}. Relevant passages only (relevance > 0).",
    ]
    for qid in qids:
        n_judged = len(qrels[qid])
        lines += ["", f"## {qid}: {queries[qid]}", "", f"{len(relevant[qid])} relevant of {n_judged} judged"]
        for docno in relevant[qid]:
            doc = docs[docno]
            lines += ["", f"- **{docno}**, {doc['title']}: {doc['text']}"]

    path = resolve_path(sample_cfg["file"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {len(qids)} queries to {sample_cfg['file']}")


if __name__ == "__main__":
    main()
