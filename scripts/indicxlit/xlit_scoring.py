"""IndicXlit candidates with their scores; shared by the frozen generator and the demo service.

Runs INSIDE the IndicXlit container (Python 3.8, docker/indicxlit/), imported by
generate_candidates.py (Phase 5, frozen candidate files) and serve.py (Phase 11,
live back-transliteration for the demo), so both score candidates identically.

IndicXlit's public API returns only ranked candidate strings. Its ranking is,
per source word (ai4bharat.transliteration, transformer/base_engine.py, rescore):

    score(c) = alpha * p_model(c) + (1 - alpha) * p_word(c)     (alpha = 0.9)

where p_model is the beam probability normalised over the beam's candidates and
p_word the corpus word frequency normalised over the candidates found in the
frequency dictionary; a candidate missing from the dictionary gets score 0.
``ScoredXlit`` records those numbers by wrapping the engine's rescore step, and
counts any difference between its own ranking and the one IndicXlit returns.
"""

from __future__ import annotations

from collections.abc import Sequence

from ai4bharat.transliteration import XlitEngine
from ai4bharat.transliteration.utils import hardfix_wordfinal_virama

ALPHA = 0.9  # fixed inside IndicXlit's post_process (rescore alpha)

# A scored candidate: (candidate, p_model, p_word, score). Written out in the annotations
# below, which `from __future__ import annotations` keeps unevaluated on Python 3.8.


def source_word(fairseq_source: str) -> str:
    """'__hi__ k i s a n' -> 'kisan' (drop the language token and the spaces)."""
    return "".join(t for t in fairseq_source.split(" ") if not t.startswith("__"))


class ScoredXlit:
    """An IndicXlit engine whose rescore step also records each candidate's scores."""

    def __init__(self, langs: Sequence[str], beam: int) -> None:
        self.beam = beam
        self.engine = XlitEngine(list(langs), beam_width=beam, rescore=True)
        self._scored: dict[str, list[tuple[str, float, float, float]]] = {}
        official_rescore = self.engine.rescore

        def recording_rescore(res_dict, result_dict, tgt_lang, alpha):
            assert alpha == ALPHA, alpha
            word_prob = self.engine.word_prob_dicts[tgt_lang]
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
                self._scored[source_word(item["S"])] = rows
            return official_rescore(res_dict, result_dict, tgt_lang, alpha)

        self.engine.rescore = recording_rescore

    def candidates(self, words: Sequence[str],
                   lang: str) -> tuple[dict[str, list[tuple[str, float, float, float]]], int]:
        """Scored candidates per word, in IndicXlit's rank order, and the number of
        ranking mismatches against IndicXlit's own output (expected 0).

        Each candidate gets the same final touch as ``XlitEngine.translit_word``
        (``hardfix_wordfinal_virama``); the mismatch check compares before it.
        """
        self._scored.clear()
        official = self.engine.batch_transliterate_words(list(words), "en", lang, topk=self.beam)[0]
        ours = [c for w in words for c, _, _, _ in self._scored[w]]
        mismatches = sum(a != b for a, b in zip(official, ours)) + abs(len(official) - len(ours))
        out = {w: [(hardfix_wordfinal_virama(c), pm, pw, s) for c, pm, pw, s in self._scored[w]]
               for w in words}
        return out, mismatches
