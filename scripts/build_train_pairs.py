"""Write the dense fine-tuning examples of every configuration (N, 50, T).

Run from the repo root inside the conda env `script-gap`:

    python -m scripts.build_train_pairs --config configs/train_pairs_hi.yaml

First checks that the training queries share no id or text with tune or test
(``src.data.splits.check_no_overlap``). Then, for each configuration, builds the
examples (``src.train.build_pairs``), looks up the passage texts (title + text,
as indexed and encoded) with one pass over the corpus, and writes one JSON line
per example to ``{output_dir}/{lang}_train_{config}.jsonl`` (git-ignored,
rebuilt by this script). Prints the number of examples, queries per script and
distinct negatives of each configuration.
"""

import argparse
import json
from collections import Counter

from src.data.load import full_text, iter_docs, read_qrels, read_queries
from src.data.splits import check_no_overlap
from src.train.build_pairs import CONFIGS, attach_texts, build_examples
from src.utils.config import load_config, resolve_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    cfg = load_config(parser.parse_args().config)

    native = read_queries(resolve_path(cfg["native_queries"]))
    roman = read_queries(resolve_path(cfg["romanized_queries"]))
    qrels = read_qrels(resolve_path(cfg["qrels"]))
    check_no_overlap({"train": native, **{name: read_queries(resolve_path(path))
                                          for name, path in cfg["protected_queries"].items()}})
    print(f"no overlap between train ({len(native)} queries) and {', '.join(cfg['protected_queries'])}")

    examples = {c: build_examples(native, roman, qrels, c, cfg["seed"]) for c in CONFIGS}
    needed = {e[k] for ex in examples.values() for e in ex for k in ("pos_docno", "neg_docno")}
    passages = {d["docno"]: full_text(d) for d in iter_docs(cfg["dataset"]) if d["docno"] in needed}

    out_dir = resolve_path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    for config, ex in examples.items():
        rows = attach_texts(ex, passages)
        path = out_dir / f"{cfg['lang']}_train_{config}.jsonl"
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        scripts = Counter(e["script"] for e in {e["qid"]: e for e in rows}.values())
        print(f"{config:>2}: {len(rows):,} examples from {len({e['qid'] for e in rows}):,} queries "
              f"({dict(scripts)}), {len({e['neg_docno'] for e in rows}):,} distinct negatives -> {path.name}")


if __name__ == "__main__":
    main()
