"""Generate IndicXlit back-transliteration candidates, with scores, for a word list.

Runs INSIDE the IndicXlit container (Python 3.8, docker/indicxlit/); started by
scripts/backtranslit_candidates.py, which mounts the repo at /work.

IndicXlit's public API returns only ranked candidate strings. Its ranking is,
per source word (ai4bharat.transliteration, transformer/base_engine.py, rescore):

    score(c) = alpha * p_model(c) + (1 - alpha) * p_word(c)     (alpha = 0.9)

where p_model is the beam probability normalised over the beam's candidates and
p_word the corpus word frequency normalised over the candidates found in the
frequency dictionary; a candidate missing from the dictionary gets score 0.
This script records those numbers by wrapping the engine's rescore step, and
checks that its own ranking equals the one IndicXlit returns.
"""

from __future__ import annotations

import argparse
import json
import time

from ai4bharat.transliteration import XlitEngine
from ai4bharat.transliteration.utils import hardfix_wordfinal_virama

ALPHA = 0.9  # fixed inside IndicXlit's post_process (rescore alpha)


def source_word(fairseq_source: str) -> str:
    """'__hi__ k i s a n' -> 'kisan' (drop the language token and the spaces)."""
    return "".join(t for t in fairseq_source.split(" ") if not t.startswith("__"))


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
    engine = XlitEngine(args.lang, beam_width=args.beam, rescore=True)
    word_prob = engine.word_prob_dicts[args.lang]
    scored: dict[str, list[tuple[str, float, float, float]]] = {}
    official_rescore = engine.rescore

    def recording_rescore(res_dict, result_dict, tgt_lang, alpha):
        assert alpha == ALPHA, alpha
        for item in res_dict.values():
            hyps = [hyp for hyp, _ in item["H"]]                  # spaced chars, beam order
            model = result_dict[item["S"]]                       # hyp -> beam probability
            model_total = sum(model[h] for h in hyps)
            joined = ["".join(h.split(" ")) for h in hyps]
            freq_total = sum(word_prob[c] for c in joined if c in word_prob)
            rows = []
            for h, c in zip(hyps, joined):
                p_model = model[h] / model_total
                p_word = word_prob[c] / freq_total if c in word_prob else 0.0
                score = alpha * p_model + (1 - alpha) * p_word if c in word_prob else 0.0
                rows.append((c, p_model, p_word, score))
            rows.sort(key=lambda r: r[3], reverse=True)          # stable, like IndicXlit
            scored[source_word(item["S"])] = rows
        return official_rescore(res_dict, result_dict, tgt_lang, alpha)

    engine.rescore = recording_rescore

    start = time.time()
    mismatches = 0
    for i in range(0, len(words), args.chunk):
        chunk = words[i:i + args.chunk]
        official = engine.batch_transliterate_words(chunk, "en", args.lang, topk=args.beam)[0]
        ours = [c for w in chunk for c, _, _, _ in scored[w]]
        mismatches += sum(a != b for a, b in zip(official, ours)) + abs(len(official) - len(ours))
        print(f"{min(i + args.chunk, len(words))}/{len(words)} words", flush=True)

    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        f.write("word\trank\tcandidate\tp_model\tp_word\tscore\n")
        for w in words:
            for rank, (c, p_model, p_word, score) in enumerate(scored[w], start=1):
                # Same final touch as XlitEngine.translit_word applies to each candidate.
                c = hardfix_wordfinal_virama(c)
                f.write(f"{w}\t{rank}\t{c}\t{p_model!r}\t{p_word!r}\t{score!r}\n")

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
