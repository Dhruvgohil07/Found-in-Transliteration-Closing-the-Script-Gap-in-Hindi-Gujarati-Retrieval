"""Freeze the list of romanized query words that count as English (Phase 5 step 5).

Run from the repo root inside the conda env `script-gap`:

    python -m scripts.backtranslit_english_words --config configs/backtranslit_english_hi.yaml

A word from ``words_file`` is English if it has at least ``min_length`` letters
and its frequency in IndicXlit's English word dictionary (the highest of the
lower-case, capitalised and upper-case forms) is at least ``min_prob``. The
list is written to ``output`` (versioned), so query building does not need the
dictionary.
"""

import argparse
import json
import zipfile

from src.utils.config import load_config, resolve_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    cfg = load_config(parser.parse_args().config)

    with zipfile.ZipFile(resolve_path(cfg["english_dict"])) as archive:
        prob = json.loads(archive.read(cfg["english_dict_member"]))
    words = resolve_path(cfg["words_file"]).read_text(encoding="utf-8").split()

    def english_prob(word: str) -> float:
        return max(prob.get(form, 0.0) for form in (word, word.capitalize(), word.upper()))

    english = [w for w in words if len(w) >= cfg["min_length"] and english_prob(w) >= cfg["min_prob"]]
    out = resolve_path(cfg["output"])
    out.write_text("".join(f"{w}\n" for w in english), encoding="utf-8", newline="\n")
    print(f"{len(english)} of {len(words)} words judged English -> {cfg['output']}")


if __name__ == "__main__":
    main()
