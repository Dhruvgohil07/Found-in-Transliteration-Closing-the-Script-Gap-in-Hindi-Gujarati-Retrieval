"""What the Devanagari normalization key does to the corpus vocabulary and the queries.

Run from the repo root inside the conda env `script-gap`:

    python -m scripts.normalization_report --config configs/docside_normalization_report_hi.yaml

Three checks, before any retrieval run (Phase 6 task 1):

1. **Vocabulary.** One pass over the corpus counts every token. Each rule alone,
   and all rules together, are then applied to every *distinct* token (much
   faster than to every passage, and the same thing, since no rule looks across
   a word boundary). Reported: how many distinct tokens and token occurrences a
   rule changes, and how much the vocabulary shrinks.
2. **Over-merging.** The groups of spellings that end up with the same key,
   largest first (by the occurrences of all but the most frequent spelling), so
   a reader can check that each merge joins variants of one word, not different
   words.
3. **Queries.** On the tune split only: the words of the native query that the
   top-1 back-transliterated query misses (Phase 5), before and after the key is
   applied to both, i.e. how many of these mismatches the key resolves.

Tokens are maximal runs of Devanagari letters and marks, digits, Latin letters
and the zero-width (non-)joiner, after lower-casing. This is close to Terrier's
``utf`` tokeniser, except that Terrier splits at a zero-width joiner, so the
numbers describe the text the index is built from, not the exact index terms.
The report goes to ``output`` (git-ignored) and is printed.
"""

import argparse
import re
import time
from collections import Counter, defaultdict

from src.data.load import full_text, iter_docs, read_queries
from src.data.normalize import DEVANAGARI_RULES, normalize_devanagari
from src.utils.config import load_config, resolve_path

TOKEN = re.compile(r"[ऀ-ॣ०-९ॱ-ॿ‌‍a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


def count_vocabulary(dataset: str) -> tuple[Counter, int]:
    vocab: Counter = Counter()
    passages = 0
    for doc in iter_docs(dataset):
        vocab.update(tokenize(full_text(doc)))
        passages += 1
    return vocab, passages


def merge_groups(vocab: Counter, rules) -> dict[str, list[tuple[str, int]]]:
    """``{key: [(spelling, count), ...]}`` for every distinct key, spellings by count."""
    groups: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for token, count in vocab.items():
        groups[normalize_devanagari(token, rules)].append((token, count))
    for forms in groups.values():
        forms.sort(key=lambda form: (-form[1], form[0]))
    return groups


def minority(forms: list[tuple[str, int]]) -> int:
    """Occurrences of all spellings but the most frequent one."""
    return sum(count for _, count in forms[1:])


def unmatched_words(native: dict[str, str], other: dict[str, str], rules) -> Counter:
    """Native query words (with multiplicity) missing from the other query, after the key."""
    missed: Counter = Counter()
    for qid, text in native.items():
        native_tokens = Counter(normalize_devanagari(t, rules) for t in tokenize(text))
        other_tokens = Counter(normalize_devanagari(t, rules) for t in tokenize(other[qid]))
        missed.update(native_tokens - other_tokens)
    return missed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    cfg = load_config(parser.parse_args().config)
    if cfg["split"] != "tune":
        raise ValueError("the query check uses the tune split only")

    start = time.perf_counter()
    vocab, passages = count_vocabulary(cfg["dataset"])
    minutes = (time.perf_counter() - start) / 60
    occurrences = sum(vocab.values())

    settings = [(f"`{rule}` only", (rule,)) for rule in DEVANAGARI_RULES]
    settings.append(("**all rules**", DEVANAGARI_RULES))
    lines = [
        f"# Devanagari normalization key: vocabulary and query check ({cfg['dataset']})",
        "",
        (f"Corpus: {passages:,} passages, {occurrences:,} token occurrences, "
         f"{len(vocab):,} distinct tokens (counted in {minutes:.1f} min)."),
        "",
        "## 1. Vocabulary",
        "",
        ("| Rules | distinct tokens changed | occurrences changed | distinct after | "
         "vocabulary change | keys joining ≥ 2 spellings |"),
        "|---|---|---|---|---|---|",
    ]
    groups_by_setting = {}
    for label, rules in settings:
        groups = merge_groups(vocab, rules)
        groups_by_setting[label] = groups
        changed = [t for t in vocab if normalize_devanagari(t, rules) != t]
        changed_occ = sum(vocab[t] for t in changed)
        merged = sum(1 for forms in groups.values() if len(forms) > 1)
        shrink = (len(groups) - len(vocab)) / len(vocab)
        lines.append(f"| {label} | {len(changed):,} | {changed_occ:,} ({changed_occ / occurrences:.2%}) "
                     f"| {len(groups):,} | {shrink:+.2%} | {merged:,} |")

    n_all, n_rule = cfg["top_merges_all"], cfg["top_merges_per_rule"]
    lines += ["", "## 2. Largest merges",
              "", ("Ranked by the occurrences of all but the most frequent spelling; "
                   "up to 5 spellings shown, most frequent first."), ""]

    def merge_table(groups, n):
        top = sorted((forms for forms in groups.values() if len(forms) > 1),
                     key=lambda forms: (-minority(forms), forms[0][0]))[:n]
        rows = ["| key | spellings (occurrences) |", "|---|---|"]
        for forms in top:
            key = normalize_devanagari(forms[0][0], DEVANAGARI_RULES)
            rows.append(f"| {key} | " + ", ".join(f"{t} ({c:,})" for t, c in forms[:5]) + " |")
        return rows

    lines += ["### All rules", ""] + merge_table(groups_by_setting["**all rules**"], n_all)
    for label, rules in settings[:-1]:
        lines += ["", f"### {label}", ""] + merge_table(groups_by_setting[label], n_rule)

    native = read_queries(resolve_path(cfg["native_queries"]))
    bt = read_queries(resolve_path(cfg["backtranslit_queries"]))
    native_words = sum(len(tokenize(text)) for text in native.values())
    before = unmatched_words(native, bt, ())
    lines += [
        "", "## 3. Tune queries: native words missed by the top-1 back-transliterated query",
        "", f"{len(native)} tune queries, {native_words:,} native words.", "",
        "| Rules | native words missed | resolved by the key |", "|---|---|---|",
        f"| none | {sum(before.values())} | - |",
    ]
    for label, rules in settings:
        after = unmatched_words(native, bt, rules)
        lines.append(f"| {label} | {sum(after.values())} | "
                     f"{sum(before.values()) - sum(after.values())} |")

    after_all = unmatched_words(native, bt, DEVANAGARI_RULES)
    resolved_words = sorted(w for w in before
                            if normalize_devanagari(w, DEVANAGARI_RULES) not in after_all)
    lines += ["", "Native words resolved by all rules: " + (", ".join(resolved_words) or "-"), ""]

    out = resolve_path(cfg["output"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
