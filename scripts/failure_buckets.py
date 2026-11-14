"""Group test queries into failure buckets and report each method's metrics per bucket (Phase 9).

Run from the repo root inside the conda env `script-gap`:

    python -m scripts.failure_buckets --config configs/failure_buckets_hi.yaml

Every query of the romanized set gets four labels, three by simple rules and
one by hand:

- **spelling** (rule): how many words of the native query (after the
  Devanagari key chosen in Phase 6) are missing from its top-1
  back-transliteration, in buckets 0 / 1 / 2+. It measures how far the typed
  spelling is from one the transliterator can map back.
- **code-mixed** (rule): the romanized query contains at least one common
  English word: a word whose lower-case form is frequent in IndicXlit's English
  dictionary (``configs/english_common_words_hi.yaml``; e.g. ``film``,
  ``missile``). Names such as ``bharat`` or ``gandhi`` are frequent in English
  text only when capitalised, so they do not count.
- **length** (rule): words in the romanized query, in the buckets of
  ``length_edges``.
- **named entity** (hand): whether the query names a person, place,
  organization, work, event or other proper name. Only a fixed sample of
  ``entity.n`` queries (seed ``seed``) is tagged. If the tag file does not
  exist yet, the script writes it with an empty ``has_entity`` column and stops,
  so the tags can be filled in by hand first.

For every bucket the table shows the number of queries and the mean of each
metric for each run in ``runs``. The per-query labels are saved too (used by
``scripts/qualitative_examples.py``). Test is used because every method choice
is already fixed; nothing here feeds back into a method.
"""

import argparse
import random
import re
import sys
from datetime import date

import pandas as pd

from src.data.backtranslit import load_english, tokenize
from src.data.load import read_queries
from src.data.normalize import normalize_devanagari
from src.eval.evaluate import read_per_query
from src.utils.config import load_config, resolve_path

_PUNCT = re.compile(r"[?,.!;:।'\"“”‘’()\[\]-]")


def native_words(text: str, rules: list[str]) -> list[str]:
    """Words of a Devanagari query, punctuation removed, normalized with the Devanagari key."""
    return [normalize_devanagari(w, rules) for w in _PUNCT.sub(" ", text).split()]


def missed_words(native: str, backtranslit: str, rules: list[str]) -> int:
    """How many words of the native query are not in the back-transliterated query."""
    bt = set(native_words(backtranslit, rules))
    return sum(w not in bt for w in native_words(native, rules))


def bucket(value: int, edges: list[int]) -> str:
    """Label of the bucket of ``value`` for upper bucket edges, e.g. edges [8, 12] -> '1-8', '9-12', '13+'."""
    low = 0 if edges and edges[0] == 0 else 1
    for edge in edges:
        if value <= edge:
            return f"{low}-{edge}" if low != edge else str(edge)
        low = edge + 1
    return f"{low}+"


def bucket_order(label: str) -> tuple[int, str]:
    """Sort key: numeric buckets by their lower bound ('1-8' < '9-12' < '13+'), others by name."""
    head = re.match(r"\d+", label)
    return (int(head.group()), label) if head else (10**9, label)


def write_sheet(cfg: dict, queries: dict[str, str], roman: dict[str, str]) -> None:
    """Write the named-entity sheet for a fixed random sample of queries, tags left empty."""
    ent = cfg["entity"]
    sample = sorted(random.Random(cfg["seed"]).sample(sorted(roman), ent["n"]))
    sheet = pd.DataFrame({"qid": sample, "has_entity": "", "entity": "",
                          "native": [queries[q] for q in sample], "romanized": [roman[q] for q in sample]})
    path = resolve_path(ent["file"])
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.to_csv(path, sep="\t", index=False, lineterminator="\n")
    sys.exit(f"wrote the empty entity sheet {ent['file']} ({ent['n']} queries): fill in has_entity "
             "(yes/no) and entity (the name), then run again")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    split, metrics = cfg["split"], cfg["metrics"]

    def path(key: str):
        return resolve_path(cfg[key].format(split=split))

    native, roman, bt = read_queries(path("native_queries")), read_queries(path("romanized_queries")), \
        read_queries(path("backtranslit_queries"))
    english = load_english(resolve_path(cfg["english_words"]))
    entity_file = resolve_path(cfg["entity"]["file"])
    if not entity_file.exists():
        write_sheet(cfg, native, roman)
    tags = pd.read_csv(entity_file, sep="\t", dtype=str, keep_default_na=False).set_index("qid")
    if not tags["has_entity"].isin(["yes", "no"]).all():
        sys.exit(f"{cfg['entity']['file']}: every has_entity must be 'yes' or 'no'")

    # 1. Labels per query
    labels = pd.DataFrame(index=sorted(roman))
    labels.index.name = "qid"
    labels["missed_words"] = [missed_words(native[q], bt[q], cfg["devanagari_rules"]) for q in labels.index]
    labels["spelling"] = labels["missed_words"].map(lambda n: bucket(n, cfg["missed_edges"]) + " missed")
    labels["english_words"] = [" ".join(w for w in tokenize(roman[q]) if w in english) for q in labels.index]
    labels["code_mixed"] = labels["english_words"].map(lambda s: "code-mixed" if s else "no English word")
    labels["n_words"] = [len(roman[q].split()) for q in labels.index]
    labels["length"] = labels["n_words"].map(lambda n: bucket(n, cfg["length_edges"]) + " words")
    labels["named_entity"] = [("entity" if tags.loc[q, "has_entity"] == "yes" else "no entity")
                              if q in tags.index else "" for q in labels.index]
    out_labels = resolve_path(cfg["output"]["labels"])
    out_labels.parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(out_labels, sep="\t", lineterminator="\n")

    # 2. Metrics per bucket
    scores = {label: read_per_query(resolve_path(cfg["per_query_dir"]) / f"{run}_{split}.csv").loc[labels.index]
              for label, run in cfg["runs"].items()}
    families = {"spelling": "Spelling: native words missed by top-1 back-transliteration",
                "code_mixed": "Code-mixing: an English word in the romanized query",
                "length": "Query length (romanized words)",
                "named_entity": f"Named entity (hand-tagged sample of {cfg['entity']['n']} queries)"}
    lines = [
        f"# {cfg['experiment']}: {cfg['lang']} {split} split, {cfg['query_set']} queries",
        "",
        f"Generated by `scripts/failure_buckets.py --config {args.config}` on {date.today().isoformat()}.",
        "",
        "Mean per-query scores of each run inside each bucket. Bucket rules are in the script's",
        f"docstring; per-query labels in `{cfg['output']['labels']}`.",
    ]
    for metric in metrics:
        lines += ["", f"## {metric}"]
        for family, title in families.items():
            groups = labels[family]
            lines += ["", f"### {title}", "",
                      "| Bucket | n | " + " | ".join(cfg["runs"]) + " |",
                      "|---|---|" + "---|" * len(cfg["runs"])]
            for value in sorted((v for v in groups.unique() if v), key=bucket_order):
                mask = groups == value
                means = [f"{scores[label].loc[mask, metric].mean():.4f}" for label in cfg["runs"]]
                lines.append(f"| {value} | {int(mask.sum())} | " + " | ".join(means) + " |")
    out_md = resolve_path(cfg["output"]["markdown"])
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
