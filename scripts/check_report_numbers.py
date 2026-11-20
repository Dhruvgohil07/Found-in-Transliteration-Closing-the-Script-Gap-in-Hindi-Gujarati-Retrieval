"""Check that every metric value written in the report comes from the project's results (Phase 11).

Run from the repo root inside the conda env `script-gap`:

    python -m scripts.check_report_numbers --config configs/report.yaml

Every number with at least ``min_decimals`` decimals in the report's own text
(comments and ``\\input`` lines skipped: the input tables are generated) is
checked:

- a number with more decimals than ``min_decimals`` (a metric value such as
  0.6115) must appear exactly as written in one of the ``number_sources``: the
  results files, the reference paper's notes or the data cards;
- a shorter number (a p-value, a CER) is too likely to match some unrelated
  value, so it must be listed in the config: under ``pvalues``, which this
  script recomputes from the per-query scores (paired t-test), or under
  ``documented``, with the command that computes it.

The script lists every number it cannot trace and exits with an error if there
is one. Percentages and counts are not checked; they are read off the same
tables.
"""

import argparse
import glob
import re
import sys

from src.eval.evaluate import read_per_query
from src.eval.significance import compare
from src.utils.config import load_config, resolve_path

NUMBER = re.compile(r"(?<![\d.])\d+\.\d+(?![\d])")


def report_numbers(tex: str, min_decimals: int) -> list[tuple[int, str]]:
    """``(line number, number)`` for the numbers to check in the report's text."""
    found = []
    for n, line in enumerate(tex.splitlines(), start=1):
        code = re.sub(r"(?<!\\)%.*", "", line)            # drop a LaTeX comment
        if "\\input" in code:
            continue
        for m in NUMBER.finditer(code):
            if len(m.group().split(".")[1]) >= min_decimals:
                found.append((n, m.group()))
    return found


def pvalue_matches(text: str, p: float) -> bool:
    """True if the p-value ``p`` is correctly written as ``text`` ("0.003" or "< 0.001")."""
    if text.startswith("<"):
        return p < float(text[1:])
    return f"{p:.{len(text.split('.')[1])}f}" == text


def check_pvalues(entries: list[dict], per_query_dir: str = "results/per_query") -> tuple[set[str], list[str]]:
    """Recompute each listed p-value; returns the confirmed numbers and the failures."""
    confirmed, failures = set(), []
    for e in entries:
        a = read_per_query(resolve_path(f"{per_query_dir}/{e['a']}.csv"))
        b = read_per_query(resolve_path(f"{per_query_dir}/{e['b']}.csv"))
        p = compare(a, b, e["metric"]).p
        if pvalue_matches(e["text"], p):
            confirmed.add(e["text"].lstrip("< "))
        else:
            failures.append(f"{e['a']} vs {e['b']}: p = {p:.2e}, written as {e['text']!r}")
    return confirmed, failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    cfg = load_config(parser.parse_args().config)
    min_dec = cfg["min_decimals"]

    sources: set[str] = set()
    files = sorted({f for pattern in cfg["number_sources"] for f in glob.glob(str(resolve_path(pattern)))})
    for path in files:
        with open(path, encoding="utf-8") as f:
            sources.update(NUMBER.findall(f.read()))
    confirmed_p, p_failures = check_pvalues(cfg.get("pvalues", []))
    documented = set(cfg.get("documented", {}))

    tex = resolve_path(cfg["report"]).read_text(encoding="utf-8")
    numbers = report_numbers(tex, min_dec)
    missing = []
    for n, v in numbers:
        long_value = len(v.split(".")[1]) > min_dec
        ok = v in sources if long_value else (v in confirmed_p or v in documented)
        if not ok:
            missing.append((n, v))

    print(f"{len(numbers)} numbers in {cfg['report']}, checked against {len(files)} source files, "
          f"{len(confirmed_p)} recomputed p-values and {len(documented)} documented numbers")
    for failure in p_failures:
        print(f"  p-value wrong: {failure}")
    for n, v in missing:
        print(f"  line {n}: {v} cannot be traced")
    if missing or p_failures:
        sys.exit(f"{len(missing) + len(p_failures)} problems")
    print("every number is traced")


if __name__ == "__main__":
    main()
