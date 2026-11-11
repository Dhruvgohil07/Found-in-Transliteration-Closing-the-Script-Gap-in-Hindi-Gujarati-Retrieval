"""Normalization keys for spelling variants (Phase 6).

BM25 matches exact strings, so two valid spellings of one word never match each
other. The Phase 5 failure analysis found this was the largest cause of the
remaining gap: IndicXlit returns आरंभिक, the passage says आरम्भिक. A
**normalization key** maps such variants to one form. It is applied the same way
to the passages (when the index is built) and to the queries (before search), so
a variant on either side meets the same key. This is the same idea as the
normalizer in Lucene's Hindi analyzer, used by MIRACL's own BM25 baseline.

**Devanagari rules** (``normalize_devanagari``). Each rule is named, so the
ablation can switch it off; they always run in the order below:

- ``zero_width``: delete the zero-width joiner / non-joiner (U+200D / U+200C).
  They only change how a cluster is drawn (क्‍या vs क्या), but Terrier's ``utf``
  tokeniser splits a word at them (``क्`` + ``या``).
- ``nukta``: Unicode NFC, then drop the nukta dot (ज़ -> ज, ड़ -> ड). NFC first
  brings both encodings of a nukta letter (one code point, or letter + dot) to
  one form; then the dot is removed, so ज़मीन and जमीन meet.
- ``chandrabindu``: chandrabindu -> anusvara (आँख -> आंख). Both mark a nasal
  vowel and both spellings are common.
- ``nasal``: a nasal consonant + virama before a consonant of its own class
  becomes the anusvara (आरम्भिक -> आरंभिक, पम्प -> पंप, हिन्दी -> हिंदी). This
  is the spelling the Central Hindi Directorate's standard prescribes. Only the
  *same class* is merged (ङ before क-घ, ञ before च-झ, ण before ट-ढ, न before
  त-ध, म before प-भ); clusters such as सम्मान, अन्य, जन्म, उन्होंने have no
  anusvara spelling and stay as they are.
- ``final_virama``: drop a virama at the end of a word (सम्राट् -> सम्राट).
  Sanskrit-style spelling writes the final virama, everyday Hindi does not.
- ``digits``: Devanagari digits -> ASCII digits (२०२३ -> 2023).

Non-Devanagari text passes through unchanged (except NFC under ``nukta``).
"""

import re
import unicodedata
from collections.abc import Callable, Iterable

VIRAMA = "्"        # ्  (halant)
NUKTA = "़"         # ़
CHANDRABINDU = "ँ"  # ँ
ANUSVARA = "ं"      # ं
ZERO_WIDTH = "‌‍"  # zero-width non-joiner, zero-width joiner

# Each nasal consonant and the consonants of its own class (varga), as ranges.
NASAL_CLASSES = {
    "ङ": "क-घ",  # ङ before क ख ग घ (velar)
    "ञ": "च-झ",  # ञ before च छ ज झ (palatal)
    "ण": "ट-ढ",  # ण before ट ठ ड ढ (retroflex)
    "न": "त-ध",  # न before त थ द ध (dental)
    "म": "प-भ",  # म before प फ ब भ (labial)
}
# nasal + virama, only when the next character is a consonant of the same class
# (a lookahead, so the following consonant itself is kept)
_NASAL = re.compile("|".join(f"{nasal}{VIRAMA}(?=[{cls}])" for nasal, cls in NASAL_CLASSES.items()))

# Characters that continue a Devanagari word: letters, vowel signs and other
# marks (U+0900-U+0963, U+0971-U+097F) and the zero-width (non-)joiner. The
# danda (। ॥), the digits and the abbreviation sign ॰ (U+0964-U+0970) end a word.
_WORD_CHARS = "ऀ-ॣॱ-ॿ" + ZERO_WIDTH
_FINAL_VIRAMA = re.compile(f"{VIRAMA}(?![{_WORD_CHARS}])")

_ZERO_WIDTH = re.compile(f"[{ZERO_WIDTH}]")
_DIGITS = str.maketrans({chr(0x0966 + i): str(i) for i in range(10)})  # ० .. ९ -> 0 .. 9


def _drop_nukta(text: str) -> str:
    # NFD splits every nukta letter into letter + dot (the precomposed ones such as
    # U+095B ज़ are excluded from NFC composition, so NFD is the reliable way);
    # after the dot is removed, NFC recomposes everything else as before.
    return unicodedata.normalize("NFC", unicodedata.normalize("NFD", text).replace(NUKTA, ""))


DEVANAGARI_RULE_FUNCTIONS: dict[str, Callable[[str], str]] = {
    "zero_width": lambda text: _ZERO_WIDTH.sub("", text),
    "nukta": _drop_nukta,
    "chandrabindu": lambda text: text.replace(CHANDRABINDU, ANUSVARA),
    "nasal": lambda text: _NASAL.sub(ANUSVARA, text),
    "final_virama": lambda text: _FINAL_VIRAMA.sub("", text),
    "digits": lambda text: text.translate(_DIGITS),
}
DEVANAGARI_RULES = tuple(DEVANAGARI_RULE_FUNCTIONS)  # all rules, in the order they run


def check_rules(rules: Iterable[str], known: tuple[str, ...]) -> set[str]:
    """The chosen rule names as a set; unknown names raise ``ValueError``."""
    chosen = set(rules)
    unknown = chosen - set(known)
    if unknown:
        raise ValueError(f"unknown rules {sorted(unknown)}; known: {list(known)}")
    return chosen


def normalize_devanagari(text: str, rules: Iterable[str] = DEVANAGARI_RULES) -> str:
    """Apply the chosen Devanagari rules (all by default) to ``text``.

    The rules always run in the order of ``DEVANAGARI_RULES``, whatever order
    ``rules`` lists them in, so a set of rules always gives the same key.
    """
    chosen = check_rules(rules, DEVANAGARI_RULES)
    for name in DEVANAGARI_RULES:
        if name in chosen:
            text = DEVANAGARI_RULE_FUNCTIONS[name](text)
    return text


def normalize_weighted(query: dict[str, float], rules: Iterable[str] = DEVANAGARI_RULES) -> dict[str, float]:
    """Apply the Devanagari key to every term of a weighted query (Phase 5).

    Terms that get the same key are merged and their weights added: if two of a
    word's IndicXlit candidates are spelling variants (आंख 0.6, आँख 0.3), the
    query gets one term आंख with weight 0.9, which the normalized index stores
    for both spellings.
    """
    merged: dict[str, float] = {}
    for term, weight in query.items():
        key = normalize_devanagari(term, rules)
        merged[key] = merged.get(key, 0.0) + weight
    return merged


# --- romanized (Latin-script) key --------------------------------------------
#
# Uroman writes the passages one way (``kaa kii raassttrapati mem``), users type
# another (``ka ki rashtrapati mein``). The romanized key maps both to one form
# (``k ki rastrapati men``). Each rule targets a difference seen between the
# natural and the Uroman romanizations of the same tune queries; they always run
# in the order below, on lower-cased text:
#
# - ``ch``: ch -> c. Uroman writes च as ``c`` (``cunaav``), users ``ch`` (``chunav``).
# - ``sh``: sh -> s. Users write both श and ष as ``sh``, Uroman ष as ``ss``.
# - ``w_v``: w -> v (``dwara`` / ``dvaaraa``).
# - ``ph_f``: ph -> f (फ typed as ``ph`` or ``f``).
# - ``z_j``: z -> j (ज़ typed as ``z`` or ``j``).
# - ``anusvara``: an ``m`` after a vowel becomes ``n`` at the end of a word or
#   before a consonant other than p, b, m, v. Uroman writes the anusvara as ``m``
#   everywhere (``mem``, ``haim``, ``baamdh``), users mostly as ``n`` (``mein``,
#   ``hain``, ``bandh``); before p, b, m, v both write ``m`` (``samvidhan``).
# - ``long_vowels``: aa -> a, ii / ee -> i, uu / oo -> u (``kaa`` / ``ka``,
#   ``ooncha`` / ``uumcaa``).
# - ``doubled``: a letter repeated -> once. Uroman doubles the retroflex letters
#   (``tt dd nn ss`` for ट ड ण ष: ``raassttrapati``).
# - ``ei_e``: ei -> e. Users write में as ``mein`` (Uroman ``mem`` -> ``men``).
# - ``eh_ah``: eh -> ah. Users write अह as ``eh`` (``pehle``, ``kehte``), Uroman
#   as ``ah`` (``pahale``, ``kahate``).
# - ``y_glide``: delete a ``y`` between two vowels. Users insert it (``liye``,
#   ``gayi``), Uroman does not (``lie``, ``gaii``).
# - ``schwa``: delete an ``a`` between two consonants when a vowel comes before
#   and after (V C a C V -> V C C V), the Hindi schwa-deletion pattern: Uroman
#   keeps the vowel (``kitane``, ``sabase``), users drop it (``kitne``,
#   ``sabse``). A consonant may be a letter + ``h`` (``bh``, ``th``, ...).
# - ``final_a``: delete a word-final ``a`` after a consonant (``yuddha`` /
#   ``yuddh``, ``kisana`` / ``kisan``).

_VOWELS = "aeiou"
_CONS = "bcdfghjklmnpqrstvwxyz"
_C = f"[{_CONS}]h?"  # one consonant, possibly aspirated (letter + h)

ROMAN_RULE_FUNCTIONS: dict[str, Callable[[str], str]] = {
    "ch": lambda text: text.replace("ch", "c"),
    "sh": lambda text: text.replace("sh", "s"),
    "w_v": lambda text: text.replace("w", "v"),
    "ph_f": lambda text: text.replace("ph", "f"),
    "z_j": lambda text: text.replace("z", "j"),
    "anusvara": lambda text: re.sub(r"(?<=[aeiou])m(?=[cdfghjklnqrstwxyz]|\b)", "n", text),
    "long_vowels": lambda text: re.sub(
        r"aa|ii|ee|uu|oo", lambda m: {"aa": "a", "ii": "i", "ee": "i", "uu": "u", "oo": "u"}[m.group()], text),
    "doubled": lambda text: re.sub(r"([a-z])\1+", r"\1", text),
    "ei_e": lambda text: text.replace("ei", "e"),
    "eh_ah": lambda text: text.replace("eh", "ah"),
    "y_glide": lambda text: re.sub(r"(?<=[aeiou])y(?=[aeiou])", "", text),
    "schwa": lambda text: re.sub(f"([{_VOWELS}]{_C})a(?={_C}[{_VOWELS}])", r"\1", text),
    "final_a": lambda text: re.sub(f"(?<=[{_CONS}])a\\b", "", text),
}
ROMAN_RULES = tuple(ROMAN_RULE_FUNCTIONS)  # all rules, in the order they run


def normalize_romanized(text: str, rules: Iterable[str] = ROMAN_RULES) -> str:
    """Lower-case ``text`` and apply the chosen romanized rules (all by default),
    always in the order of ``ROMAN_RULES``."""
    chosen = check_rules(rules, ROMAN_RULES)
    text = text.lower()
    for name in ROMAN_RULES:
        if name in chosen:
            text = ROMAN_RULE_FUNCTIONS[name](text)
    return text


# --- character n-grams -------------------------------------------------------

_LATIN_TOKEN = re.compile(r"[a-z0-9]+")
BOUNDARY = "0"  # marks the start and end of a word inside an n-gram


def char_ngrams(text: str, n: int) -> str:
    """Replace every Latin word by its overlapping character ``n``-grams.

    Two spellings that differ in a few places still share some n-grams
    (as 3-grams, ``olympic`` / ``olampik`` share ``0ol`` and ``mpi``;
    ``bhartia`` / ``bhartiy`` share 5 of 7), so BM25 can match near-misses
    that whole-word matching cannot. Each word is padded with
    ``0`` at both ends first, so n-grams at the start and end of a word differ
    from ones inside it (``0ki`` vs ``ki``): Terrier's tokeniser keeps digits but
    splits on ``_``, the usual marker. A word shorter than ``n`` after padding
    becomes one token. Tokens with a digit (``2016``, ``co2``) are kept whole:
    numbers should match exactly. Only ``[a-z0-9]`` text is kept (lower-cased).
    """
    if n < 2:
        raise ValueError(f"n must be at least 2, got {n}")
    return " ".join(_ngram_tokens(text, n))


def text_key(text: str, rules: Iterable[str] = (), roman_rules: Iterable[str] = (),
             ngrams: int | None = None) -> str:
    """The full text treatment of Phase 6, in one place for passages and queries:
    the Devanagari key, then the romanized key, then character n-grams (each
    step only if asked for). The index build and every search call this, so
    the two sides cannot drift apart."""
    rules, roman_rules = list(rules), list(roman_rules)
    if rules:
        text = normalize_devanagari(text, rules)
    if roman_rules:
        text = normalize_romanized(text, roman_rules)
    if ngrams:
        text = char_ngrams(text, ngrams)
    return text


def _ngram_tokens(text: str, n: int) -> list[str]:
    """The n-gram tokens of ``char_ngrams``, as a list."""
    out = []
    for token in _LATIN_TOKEN.findall(text.lower()):
        if not token.isalpha():
            out.append(token)
            continue
        padded = f"{BOUNDARY}{token}{BOUNDARY}"
        if len(padded) <= n:
            out.append(padded)
        else:
            out.extend(padded[i:i + n] for i in range(len(padded) - n + 1))
    return out
