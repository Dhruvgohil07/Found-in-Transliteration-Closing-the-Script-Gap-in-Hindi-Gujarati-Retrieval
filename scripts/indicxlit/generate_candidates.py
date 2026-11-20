"""Generate IndicXlit back-transliteration candidates, with scores, for a word list.

Runs INSIDE the IndicXlit container (Python 3.8, docker/indicxlit/); started by
scripts/backtranslit_candidates.py, which mounts the repo at /work. The scoring
(IndicXlit's rescore formula, recorded per candidate) lives in xlit_scoring.py,
shared with the demo's live service (serve.py).
"""

from __future__ import annotations

import argparse
import json
import time

from xlit_scoring import ALPHA, ScoredXlit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--words", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--meta", required=True)
    parser.add_argument("--lang", required=True)
    parser.add_argument("--beam", type=int, required=True)
    parser.add_argument("--chunk", type=int, default=64)
    args = parser.parse_args()

    with open(args.words, encoding="utf-8") as f:
        words = [w for w in f.read().split("\n") if w]
    xlit = ScoredXlit([args.lang], args.beam)

    start = time.time()
    mismatches = 0
    scored = {}
    for i in range(0, len(words), args.chunk):
        chunk_scored, chunk_mismatches = xlit.candidates(words[i:i + args.chunk], args.lang)
        scored.update(chunk_scored)
        mismatches += chunk_mismatches
        print(f"{min(i + args.chunk, len(words))}/{len(words)} words", flush=True)

    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        f.write("word\trank\tcandidate\tp_model\tp_word\tscore\n")
        for w in words:
            f.writelines(f"{w}\t{rank}\t{c}\t{p_model!r}\t{p_word!r}\t{score!r}\n"
                         for rank, (c, p_model, p_word, score) in enumerate(scored[w], start=1))

    meta = {
        "tool": "ai4bharat-transliteration==1.1.3 (IndicXlit v1.0, en->indic transformer)",
        "lang": args.lang, "beam_width": args.beam, "rescore": True, "alpha": ALPHA,
        "words": len(words), "ranking_mismatches_vs_indicxlit": mismatches,
        "seconds": round(time.time() - start, 1),
    }
    with open(args.meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(json.dumps(meta, indent=2))
    if mismatches:
        raise SystemExit(f"{mismatches} candidates ranked differently from IndicXlit's own output")


if __name__ == "__main__":
    main()
