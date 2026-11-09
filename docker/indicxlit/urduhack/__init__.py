"""Stand-in for the `urduhack` package inside the IndicXlit container.

IndicXlit imports ``urduhack.normalize`` at module level but calls it only for
Urdu (``lang_code == "ur"``). The real package imports TensorFlow on import, a
very large download for a code path this project never runs (Hindi and Gujarati
only), so this stub takes its place and fails loudly if Urdu is ever requested.
"""


def normalize(text: str) -> str:
    raise NotImplementedError("Urdu is not used in this project; the urduhack stub cannot normalize it")
