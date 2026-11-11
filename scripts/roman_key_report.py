"""What the romanized normalization key does to query words and to the corpus vocabulary.

Run from the repo root inside the conda env `script-gap`:

    python -m scripts.roman_key_report --config configs/docside_roman_key_report_hi.yaml

Two checks, before any retrieval run (Phase 6 task 5):

1. **Word matches (tune only).** The natural and the rule (Uroman) romanization
   of a tune query are two spellings of the same native query; with the same
   number of words they are aligned word by word. Reported: the share of
   aligned pairs with the same key, with no rules, with all rules, and with all
   rules but one (what each rule adds), plus the most common pairs still apart.
2. **Over-merging.** The romanized corpus (``docs_file``) is counted once; the
   key is applied to every distinct token. Reported: how much the vocabulary
   shrinks, and the largest groups of spellings that get one key, for a reader
   to check that they are variants of one word, not different words.

Tokens are lower-cased runs of ``[a-z0-9]``, as Terrier's ``utf`` tokeniser
cuts Latin text. The report goes to ``output`` (git-ignored) and is printed.
"""

import argparse
import json
import re
from collections import Counter, defaultdict

from src.data.load import read_queries
from src.data.normalize import ROMAN_RULES, normalize_romanized
from src.utils.config import load_config, resolve_path

TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


def aligned_pairs(natural: dict[str, str], rule: dict[str, str]) -> tuple[list[tuple[str, str]], int]:
    """Word pairs of the queries whose two romanizations have the same length."""
    pairs, aligned = [], 0
    for qid, text in natural.items():
        a, b = tokenize(text), tokenize(rule[qid])
        if len(a) == len(b):
            pairs += zip(a, b)
            aligned += 1
    return pairs, aligned


def match_rate(pairs, rules) -> float:
    return sum(normalize_romanized(a, rules) == normalize_romanized(b, rules) for a, b in pairs) / len(pairs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    cfg = load_config(parser.parse_args().config)
    if cfg["split"] != "tune":
        raise ValueError("the word check uses the tune split only")

    natural = read_queries(resolve_path(cfg["natural_queries"]))
    rule = read_queries(resolve_path(cfg["rule_queries"]))
    pairs, aligned = aligned_pairs(natural, rule)
    lines = [
        "# Romanized normalization key: word matches and vocabulary", "",
        (f"## 1. Tune word pairs (natural vs Uroman): {aligned} of {len(natural)} queries aligned, "
        f"{len(pairs)} word pairs"), "",
        "| Rules | same key | change when the rule is removed |", "|---|---|---|",
        f"| none | {match_rate(pairs, ()):.1%} | |",
    ]
    all_rate = match_rate(pairs, ROMAN_RULES)
    lines.append(f"| **all rules** | {all_rate:.1%} | |")
    for name in ROMAN_RULES:
        rate = match_rate(pairs, [r for r in ROMAN_RULES if r != name])
        lines.append(f"| all but `{name}` | {rate:.1%} | {rate - all_rate:+.1%} |")
    apart = Counter((normalize_romanized(a), normalize_romanized(b)) for a, b in pairs
                    if normalize_romanized(a) != normalize_romanized(b))
    lines += ["", "Most common pairs still apart with all rules (natural key / Uroman key): "
              + ", ".join(f"{a} / {b} ({c})" for (a, b), c in apart.most_common(cfg["top_apart"]))]

    vocab: Counter = Counter()
    with open(resolve_path(cfg["docs_file"]), encoding="utf-8") as f:
        for line in f:
            vocab.update(tokenize(json.loads(line)["text"]))
    groups: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for token, count in vocab.items():
        groups[normalize_romanized(token)].append((token, count))
    for forms in groups.values():
        forms.sort(key=lambda form: (-form[1], form[0]))
    occurrences = sum(vocab.values())
    changed = sum(c for t, c in vocab.items() if normalize_romanized(t) != t)
    merged = [forms for forms in groups.values() if len(forms) > 1]
    lines += [
        "", "## 2. Corpus vocabulary with all rules", "",
        (f"{occurrences:,} token occurrences, {len(vocab):,} distinct tokens -> {len(groups):,} keys "
        f"({(len(groups) - len(vocab)) / len(vocab):+.1%}); occurrences changed: {changed / occurrences:.1%}; "
        f"keys joining ≥ 2 spellings: {len(merged):,}."), "",
        ("Largest merges, by the occurrences of all but the most frequent spelling "
        "(up to 6 spellings, most frequent first):"), "",
        "| key | spellings (occurrences) |", "|---|---|",
    ]
    for forms in sorted(merged, key=lambda forms: (-sum(c for _, c in forms[1:]), forms[0][0]))[:cfg["top_merges"]]:
        key = normalize_romanized(forms[0][0])
        lines.append(f"| {key} | " + ", ".join(f"{t} ({c:,})" for t, c in forms[:6]) + " |")

    out = resolve_path(cfg["output"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
