"""Collect the romanized words and generate their IndicXlit candidates (once, frozen).

Run from the repo root inside the conda env `script-gap`, with Docker Desktop running:

    docker build -t script-gap-indicxlit docker/indicxlit
    python -m scripts.backtranslit_candidates --config configs/backtranslit_candidates_hi.yaml

1. Every distinct transliterable word of the configured query files is written
   to ``words_file`` (``src.data.backtranslit.words_to_transliterate``).
2. ``scripts/indicxlit/generate_candidates.py`` runs inside the IndicXlit
   container, with the repo mounted at /work and ``model_cache`` mounted as the
   package's v1.0 model folder (the first run downloads the model and dictionaries).
3. The candidates file is checked: every word has ``beam_width`` candidates.

The candidates file is frozen: it is not overwritten unless ``--force`` is given.
"""

import argparse
import subprocess
import sys

import pandas as pd

from src.data.backtranslit import words_to_transliterate
from src.data.load import read_queries
from src.utils.config import REPO_ROOT, load_config, resolve_path

# Where IndicXlit downloads its v1.0 Latin->Indic model and dictionaries. Only this
# version folder is mounted: the package ships its own files one level up
# (models/en2indic/lang_list.txt), which a mount of the whole models folder would hide.
PACKAGE_MODELS = "/usr/local/lib/python3.8/site-packages/ai4bharat/transliteration/transformer/models/en2indic/v1.0"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--force", action="store_true", help="overwrite the frozen candidates")
    args = parser.parse_args()
    cfg = load_config(args.config)

    out = resolve_path(cfg["candidates_file"])
    if out.exists() and not args.force:
        sys.exit(f"frozen file exists, not overwriting: {cfg['candidates_file']} (use --force)")

    words = words_to_transliterate(read_queries(resolve_path(f)) for f in cfg["query_files"])
    words_file = resolve_path(cfg["words_file"])
    words_file.parent.mkdir(parents=True, exist_ok=True)
    words_file.write_text("".join(f"{w}\n" for w in words), encoding="utf-8", newline="\n")
    print(f"{len(words):,} distinct words -> {cfg['words_file']}")

    cache = resolve_path(cfg["model_cache"])
    cache.mkdir(parents=True, exist_ok=True)
    command = [
        "docker", "run", "--rm",
        "-v", f"{REPO_ROOT}:/work",
        "-v", f"{cache}:{PACKAGE_MODELS}",
        cfg["docker_image"],
        "python", "scripts/indicxlit/generate_candidates.py",
        "--words", cfg["words_file"], "--out", cfg["candidates_file"],
        "--meta", cfg["meta_file"], "--lang", cfg["lang"],
        "--beam", str(cfg["beam_width"]), "--chunk", str(cfg["chunk"]),
    ]
    print("running:", " ".join(command[:3]), "... (IndicXlit container)")
    subprocess.run(command, check=True)

    candidates = pd.read_csv(out, sep="\t", dtype={"word": str, "candidate": str},
                             keep_default_na=False)
    per_word = candidates.groupby("word").size()
    missing = sorted(set(words) - set(per_word.index))
    short = per_word[per_word < cfg["beam_width"]]
    print(f"{per_word.size:,} words with candidates; {len(missing)} missing; "
          f"{len(short)} with fewer than {cfg['beam_width']} candidates")
    if missing:
        raise RuntimeError(f"no candidates for {len(missing)} words, e.g. {missing[:5]}")


if __name__ == "__main__":
    main()
