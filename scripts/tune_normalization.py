"""Leave-one-out ablation of the Devanagari normalization rules on the tune split.

Run from the repo root inside the conda env `script-gap`:

    python -m scripts.tune_normalization --config configs/tune_docside_normalization_hi.yaml

Settings: ``none`` (the Phase 4 index, raw text), ``all`` (every rule) and, for
each rule, ``no_<rule>`` (every rule but that one). The rules change the
passages, so every setting has its own BM25 index; a missing index is built
first (about 4 minutes each). Each setting is searched with:

- the native queries (normalized with the same rules), and
- the weighted top-k back-transliterated queries of each romanized set
  (Phase 5), with their terms normalized (``normalize_weighted``).

BM25 uses the Phase 4 baseline parameters for every setting, so only the rules
differ. A rule is **kept** if removing it lowers the decision score: the mean
``metric`` over the ``decision_sets``, ``all`` minus ``no_<rule>`` > 0. Paired
t-tests (``all`` against each other setting) are printed to show how large the
differences are next to the noise. Only the tune split is allowed; the table goes
to ``output`` and no rows go to results/results.csv (which holds reported runs).
"""

import argparse

import pandas as pd

from scripts.build_bm25_index import build_corpus_index
from src.data.backtranslit import load_candidates, weighted_query
from src.data.load import read_qrels, read_queries
from src.data.normalize import (
    DEVANAGARI_RULES,
    check_rules,
    normalize_devanagari,
    normalize_weighted,
)
from src.data.splits import subset_qrels
from src.eval.evaluate import mean_scores, per_query_scores
from src.eval.significance import compare
from src.retrieval.bm25 import (
    check_normalization,
    open_index,
    retrieve,
    retrieve_weighted,
)
from src.utils.config import load_config, resolve_path


def settings(extra: dict[str, list[str]] | None = None) -> dict[str, list[str]]:
    """``{setting: rules}``: none, all, all-but-one for every rule, then any
    ``extra`` settings from the config (e.g. the set of kept rules, which
    leave-one-out alone never evaluates together)."""
    out = {"none": [], "all": list(DEVANAGARI_RULES)}
    for rule in DEVANAGARI_RULES:
        out[f"no_{rule}"] = [r for r in DEVANAGARI_RULES if r != rule]
    for name, rules in (extra or {}).items():
        out[name] = [r for r in DEVANAGARI_RULES if r in check_rules(rules, DEVANAGARI_RULES)]
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
    candidates = load_candidates(resolve_path(cfg["candidates_file"]))
    all_qrels = read_qrels(resolve_path(cfg["qrels"]))
    query_sets = {name: read_queries(resolve_path(path)) for name, path in cfg["query_sets"].items()}
    metric = cfg["metric"]
    all_settings = settings(cfg.get("extra_settings"))

    rows, per_query = [], {}
    for setting, rules in all_settings.items():
        if setting == "none":
            index_dir = index_cfg["index_dir"]          # the Phase 4 index: raw text
        else:
            index_dir = cfg["index_dir_pattern"].format(setting=setting)
            build_corpus_index({**index_cfg, "index_dir": index_dir, "normalize": rules})
        check_normalization(resolve_path(index_dir), rules)
        index = open_index(resolve_path(index_dir))

        for name, queries in query_sets.items():
            if name == "native":
                normalized = {qid: normalize_devanagari(text, rules) for qid, text in queries.items()}
                run = retrieve(index, normalized, k1=params["k1"], b=params["b"], top_k=top_k,
                               floor_idf=params["floor_idf"])
            else:
                weighted = {qid: normalize_weighted(weighted_query(text, candidates, cfg["k"]), rules)
                            for qid, text in queries.items()}
                run = retrieve_weighted(index, weighted, k1=params["k1"], b=params["b"],
                                        top_k=top_k, floor_idf=params["floor_idf"])
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
    pivot["decision"] = pivot[cfg["decision_sets"]].mean(axis=1)
    print(f"\n{metric} on {cfg['split']} (decision = mean over {cfg['decision_sets']}):")
    print(pivot.round(4).to_string())

    print("\npaired t-test, all vs each setting (p-values, uncorrected):")
    for setting in all_settings:
        if setting != "all":
            ps = {name: compare(per_query["all", name], per_query[setting, name], metric).p
                  for name in query_sets}
            print(f"  {setting:18} " + "  ".join(f"{n}={p:.3f}" for n, p in ps.items()))

    kept = [rule for rule in DEVANAGARI_RULES
            if pivot.loc["all", "decision"] - pivot.loc[f"no_{rule}", "decision"] > 0]
    print("\nrule: decision(all) - decision(all but the rule)")
    for rule in DEVANAGARI_RULES:
        delta = pivot.loc["all", "decision"] - pivot.loc[f"no_{rule}", "decision"]
        print(f"  {rule:14} {delta:+.4f}  {'keep' if rule in kept else 'drop'}")
    print(f"\nkept rules: {kept}")


if __name__ == "__main__":
    main()
