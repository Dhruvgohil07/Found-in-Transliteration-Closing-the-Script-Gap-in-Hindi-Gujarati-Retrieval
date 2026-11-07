"""Hand-romanized queries: make the blank sheet, then collect the filled one.

Run from the repo root inside the conda env `script-gap`:

    python -m scripts.human_romanization --config configs/romanize_hi.yaml make
    (one person fills in sheet A)
    python -m scripts.human_romanization --config configs/romanize_hi.yaml collect

``make`` picks ``n_queries`` queries of the human split with the config seed and
writes the sheet ``human_sheet_a_{lang}.txt``. It shows only the native-script
query, never a generated romanization, so the typist is not influenced by it.
An existing sheet is never overwritten (it may hold typed work).

``collect`` reads the filled sheet, checks that every query has an answer
without Devanagari, and writes the frozen file
``{split}_queries_roman_human_a.tsv`` to ``output_dir``.

There is one version, A. The plan first had a second, independent version B
typed by another person, to measure how much human spellings vary; it was
dropped (docs/plans/03-romanized-queries.md, task 3). The "a" in the file names
is kept so a version B can still be added later without renaming anything.

Sheet format: each query is a block of three lines, and the typist fills in the
text after "> ":

    # 1004960#0
    किस यंत्र में मापन के समय अधिकतम पावर व्यय होती है?
    > kis yantra me mapan ke samay adhiktam power vyay hoti hai?
"""

import argparse
import sys
from pathlib import Path

from src.data.load import Queries, read_queries, write_queries
from src.data.romanize import check_romanized
from src.data.splits import split_ids, subset_queries
from src.utils.config import load_config, resolve_path

VERSIONS = ("a",)

INSTRUCTIONS = """\
## Hand romanization, version {version}
##
## Below are {n} Hindi search queries. Under each one, after "> ", type the query
## the way you would type it into a search box on your phone or laptop, in Latin
## letters (English keyboard). Spell it however feels natural to you; there is no
## right answer.
##
## - Work alone: do not look at any romanization tool or generated version.
## - Romanize every word; do not translate or shorten the query.
## - Do not change the "# id" lines or the Hindi lines.
"""


def sheet_path(cfg: dict, version: str) -> Path:
    return resolve_path(cfg["human"]["sheet_dir"]) / f"human_sheet_{version}_{cfg['lang']}.txt"


def output_path(cfg: dict, version: str) -> Path:
    split = cfg["human"]["split"]
    return resolve_path(cfg["output_dir"]) / f"{split}_queries_roman_human_{version}.tsv"


def write_sheet(queries: Queries, path: Path, version: str) -> None:
    lines = [INSTRUCTIONS.format(version=version.upper(), n=len(queries))]
    for qid in sorted(queries):
        lines.append(f"# {qid}\n{queries[qid]}\n> \n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def read_sheet(path: Path) -> Queries:
    """Parse a filled sheet into ``{qid: romanized}``; raises ``ValueError`` on a
    block with a missing or unusable answer."""
    answers: Queries = {}
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if not line.startswith("##")]
    qid = None
    for n, line in enumerate(lines, start=1):
        if line.startswith("# "):
            qid = line[2:].strip()
        elif line.startswith(">"):
            if qid is None:
                raise ValueError(f"{path.name}: answer line {n} has no '# id' line before it")
            answer = line[1:].strip()
            try:
                check_romanized(answer)
            except ValueError as e:
                raise ValueError(f"{path.name}: query {qid}: {e}") from e
            answers[qid] = answer
            qid = None
    return answers


def make(cfg: dict) -> None:
    hcfg = cfg["human"]
    queries = read_queries(resolve_path(cfg["input_dir"]) / f"{hcfg['split']}_queries.tsv")
    chosen, _ = split_ids(queries, hcfg["n_queries"], cfg["seed"])
    subset = subset_queries(queries, chosen)
    for version in VERSIONS:
        path = sheet_path(cfg, version)
        if path.exists():
            print(f"{path.name} exists, left as it is (it may already be filled in)")
            continue
        write_sheet(subset, path, version)
        print(f"{path.name}: {len(subset)} queries to romanize")


def collect(cfg: dict, force: bool) -> None:
    hcfg = cfg["human"]
    queries = read_queries(resolve_path(cfg["input_dir"]) / f"{hcfg['split']}_queries.tsv")
    chosen, _ = split_ids(queries, hcfg["n_queries"], cfg["seed"])
    for version in VERSIONS:
        out = output_path(cfg, version)
        if out.exists() and not force:
            sys.exit(f"frozen file exists, not overwriting: {out.name} (use --force)")
        answers = read_sheet(sheet_path(cfg, version))
        if sorted(answers) != chosen:
            missing = sorted(set(chosen) - set(answers))
            raise ValueError(f"version {version}: {len(missing)} queries unanswered, e.g. {missing[:3]}")
        write_queries(answers, out)
        print(f"version {version}: {len(answers)} queries -> {out.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("action", choices=["make", "collect"])
    parser.add_argument("--force", action="store_true", help="overwrite frozen output files")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.action == "make":
        make(cfg)
    else:
        collect(cfg, args.force)


if __name__ == "__main__":
    main()
