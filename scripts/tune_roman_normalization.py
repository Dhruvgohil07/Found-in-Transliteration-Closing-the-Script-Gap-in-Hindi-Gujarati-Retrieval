"""Leave-one-out ablation of the romanized normalization rules on the tune split.

Run from the repo root inside the conda env `script-gap`:

    python -m scripts.tune_roman_normalization --config configs/tune_docside_roman_normalization_hi.yaml

The same protocol as the Devanagari ablation (scripts/tune_normalization.py),
on the romanized (Uroman) index: settings ``none`` (the task 4 index), ``all``,
``no_<rule>`` for every rule, and any ``extra_settings`` from the config. Each
setting has its own index (a missing one is built first, a few minutes each)
and is searched with the raw romanized queries of every query set, normalized
with the same rules. BM25 parameters are fixed (``bm25_config``), so only the
rules differ. A rule is **kept** if removing it lowers ``metric`` on the
``decision_set``. Paired t-tests of ``all`` against each setting are printed.
Only the tune split is allowed; the table goes to ``output``, no rows go to
results/results.csv.
"""

import argparse

import pandas as pd

from scripts.build_bm25_index import build_corpus_index
from src.data.load import read_qrels, read_queries
from src.data.normalize import ROMAN_RULES, check_rules, normalize_romanized
from src.data.splits import subset_qrels
from src.eval.evaluate import mean_scores, per_query_scores
from src.eval.significance import compare
from src.retrieval.bm25 import check_normalization, open_index, retrieve
from src.utils.config import load_config, resolve_path


def settings(extra: dict[str, list[str]] | None = None) -> dict[str, list[str]]:
    """``{setting: rules}``: none, all, all-but-one for every rule, then ``extra``."""
    out = {"none": [], "all": list(ROMAN_RULES)}
    for rule in ROMAN_RULES:
        out[f"no_{rule}"] = [r for r in ROMAN_RULES if r != rule]
    for name, rules in (extra or {}).items():
        out[name] = [r for r in ROMAN_RULES if r in check_rules(rules, ROMAN_RULES)]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    cfg = load_config(parser.parse_args().config)
    if cfg["split"] != "tune":
        raise ValueError(f"tuning is only allowed on the tune split, not {cfg['split']!r}")

    index_cfg = load_config(cfg["index_config"])
    bm25_cfg = load_config(cfg["bm25_config"])
    params, top_k = bm25_cfg["bm25"], bm25_cfg["top_k"]
    all_qrels = read_qrels(resolve_path(cfg["qrels"]))
    query_sets = {name: read_queries(resolve_path(path)) for name, path in cfg["query_sets"].items()}
    metric, decision = cfg["metric"], cfg["decision_set"]
    all_settings = settings(cfg.get("extra_settings"))

    rows, per_query = [], {}
    for setting, rules in all_settings.items():
        if setting == "none":
            index_dir = index_cfg["index_dir"]          # the task 4 index: raw Uroman text
        else:
            index_dir = cfg["index_dir_pattern"].format(setting=setting)
            build_corpus_index({**index_cfg, "index_dir": index_dir, "roman_normalize": rules})
        check_normalization(resolve_path(index_dir), [], rules)
        index = open_index(resolve_path(index_dir))
        for name, queries in query_sets.items():
            normalized = {qid: normalize_romanized(text, rules) if rules else text
                          for qid, text in queries.items()}
            run = retrieve(index, normalized, k1=params["k1"], b=params["b"], top_k=top_k,
                           floor_idf=params["floor_idf"])
            pq = per_query_scores(run, subset_qrels(all_qrels, queries))
            per_query[setting, name] = pq
            rows.append({"setting": setting, "query_set": name, **mean_scores(pq)})
            print(f"{setting:18} {name:14} {metric}={rows[-1][metric]:.4f}")
        index.close()

    table = pd.DataFrame(rows)
    out = resolve_path(cfg["output"])
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False, float_format="%.4f", lineterminator="\n")

    pivot = table.pivot(index="setting", columns="query_set", values=metric)
    pivot = pivot.loc[list(all_settings), list(query_sets)]
    print(f"\n{metric} on {cfg['split']} (decision set: {decision}):")
    print(pivot.round(4).to_string())

    print("\npaired t-test, all vs each setting (p-values, uncorrected):")
    for setting in all_settings:
        if setting != "all":
            ps = {name: compare(per_query["all", name], per_query[setting, name], metric).p
                  for name in query_sets}
            print(f"  {setting:18} " + "  ".join(f"{n}={p:.3f}" for n, p in ps.items()))

    print(f"\nrule: {metric}(all) - {metric}(all but the rule), on {decision}")
    kept = []
    for rule in ROMAN_RULES:
        delta = pivot.loc["all", decision] - pivot.loc[f"no_{rule}", decision]
        if delta > 0:
            kept.append(rule)
        print(f"  {rule:12} {delta:+.4f}  {'keep' if delta > 0 else 'drop'}")
    print(f"\nkept rules: {kept}")


if __name__ == "__main__":
    main()
