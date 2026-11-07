"""Quality check of the romanized query sets (Phase 3 task 4).

Run from the repo root inside the conda env `script-gap`:

    python -m scripts.romanization_quality --config configs/romanize_hi.yaml

0. **Sanity checks** of the generated sets over all splits: queries with
   non-ASCII characters, with capital letters, or with a different number of
   words than the native query (a hint of translated, added or dropped words);
   for the natural set also the same check by request batch size, from the
   git-ignored progress file.
1. **CER table** on the hand-romanized subset: each generated set (rule,
   natural) against the human version A, as the reference. CER divides by the
   reference length, so the reference is always named.
2. **Review file** (``quality.review_file``, git-ignored):
   - ``n_review`` tune queries (fixed seed) with native / rule / natural side by
     side, for the manual check of the natural set (nonsense, untransliterated
     words, translated or dropped words);
   - the hand-romanized test queries with all four versions side by side, for
     the review gate.

Sets whose files do not exist yet are skipped, so the script can run part-way
through the phase.
"""

import argparse
import json
from pathlib import Path

from src.data.load import Queries, read_queries
from src.data.romanize import cer
from src.data.splits import split_ids
from src.utils.config import load_config, resolve_path


def load_if_exists(path: Path) -> Queries | None:
    return read_queries(path) if path.exists() else None


def share(flags: list[bool]) -> str:
    return f"{sum(flags)} ({sum(flags) / len(flags):.1%})"


def word_count_differs(native: str, roman: str) -> bool:
    return len(native.split()) != len(roman.split())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    cfg = load_config(parser.parse_args().config)
    in_dir, out_dir = resolve_path(cfg["input_dir"]), resolve_path(cfg["output_dir"])
    human_split = cfg["human"]["split"]

    def sets_for(split: str) -> dict[str, Queries | None]:
        return {
            "native": read_queries(in_dir / f"{split}_queries.tsv"),
            "rule": load_if_exists(out_dir / f"{split}_queries_roman_rule.tsv"),
            "natural": load_if_exists(out_dir / f"{split}_queries_roman_natural.tsv"),
            "human A": load_if_exists(out_dir / f"{split}_queries_roman_human_a.tsv"),
        }

    # 0. Sanity checks over all splits
    native_all: Queries = {}
    generated: dict[str, Queries] = {"rule": {}, "natural": {}}
    for split in cfg["splits"]:
        sets = sets_for(split)
        native_all.update(sets["native"])
        for method, queries in generated.items():
            if sets[method] is not None:
                queries.update(sets[method])
    print("| Set | Queries | Non-ASCII | Capital letters | Word count differs from native |")
    print("|---|---|---|---|---|")
    for method, queries in generated.items():
        if not queries:
            continue
        qids = sorted(queries)
        print(
            f"| {method} | {len(qids)} "
            f"| {share([not queries[q].isascii() for q in qids])} "
            f"| {share([queries[q] != queries[q].lower() for q in qids])} "
            f"| {share([word_count_differs(native_all[q], queries[q]) for q in qids])} |"
        )
    progress = in_dir / f"natural_progress_{cfg['lang']}.jsonl"
    if progress.exists():
        rows = [json.loads(line) for line in progress.read_text(encoding="utf-8").splitlines()]
        print("\n| Natural, batch size | Queries | Word count differs from native |")
        print("|---|---|---|")
        for size in sorted({row["batch_size"] for row in rows}):
            flags = [
                word_count_differs(r["native"], r["roman"]) for r in rows if r["batch_size"] == size
            ]
            print(f"| {size} | {len(flags)} | {share(flags)} |")
    if generated["rule"] and generated["natural"]:
        natural_vs_rule = cer(generated["natural"], generated["rule"])
        print(f"\nCER natural vs rule (rule as reference), all splits: {natural_vs_rule:.3f}\n")

    # 1. CER on the hand-romanized subset
    test = sets_for(human_split)
    pairs = [("rule", "human A"), ("natural", "human A")]
    print("| Compared | Reference | Queries | CER |\n|---|---|---|---|")
    for hyp, ref in pairs:
        if test[hyp] is None or test[ref] is None:
            print(f"| {hyp} | {ref} | - | (missing) |")
            continue
        n = len(set(test[hyp]) & set(test[ref]))
        print(f"| {hyp} | {ref} | {n} | {cer(test[hyp], test[ref]):.3f} |")

    # 2. Review file
    lines = ["# Romanization review (Phase 3)\n"]
    tune = sets_for("tune")
    if tune["natural"] is not None:
        sample, _ = split_ids(tune["native"], cfg["quality"]["n_review"], cfg["seed"])
        lines.append(
            f"## {len(sample)} tune queries: is the natural romanization realistic?\n\n"
            "Flag nonsense, Devanagari left in, or words translated, added or dropped.\n"
        )
        for qid in sample:
            lines.append(
                f"- `{qid}`\n  - native: {tune['native'][qid]}\n"
                f"  - rule: {tune['rule'][qid] if tune['rule'] else '-'}\n"
                f"  - natural: {tune['natural'][qid]}\n"
            )
    if test["human A"] is not None:
        lines.append(f"\n## Hand-romanized {human_split} queries, all versions\n")
        for qid in sorted(test["human A"]):
            versions = "".join(
                f"  - {name}: {queries[qid]}\n" for name, queries in test.items() if queries
            )
            lines.append(f"- `{qid}`\n{versions}")
    review = resolve_path(cfg["quality"]["review_file"])
    review.parent.mkdir(parents=True, exist_ok=True)
    review.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(f"\nreview file: {cfg['quality']['review_file']}")


if __name__ == "__main__":
    main()
