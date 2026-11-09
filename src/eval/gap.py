"""The script gap: how much a metric drops when native queries are replaced by
romanized versions of the same queries (SPECS §7.2, Phase 4 task 4).

For a metric M, the relative drop is

    Delta = (M_native - M_roman) / M_native

so 0 means no loss and 1 (100%) means everything is lost. Both runs must be
scored on the *same* queries: the hand-romanized set covers only 50 test
queries, so its native comparison uses the native scores of those 50 queries,
not the average over all 250.

Significance: a paired t-test of native vs romanized per-query scores; with m
romanized sets compared to the same native run, a drop counts as significant
only if p < alpha / m (Bonferroni).
"""

from dataclasses import dataclass

import pandas as pd

from src.eval.significance import bonferroni_significant, compare


@dataclass(frozen=True)
class GapRow:
    """Native vs one romanized query set, for one metric."""

    query_set: str
    metric: str
    n_queries: int
    native: float
    roman: float
    p: float
    significant: bool

    @property
    def drop(self) -> float:
        """Relative drop (M_native - M_roman) / M_native; NaN if M_native is 0."""
        return (self.native - self.roman) / self.native if self.native else float("nan")


def gap_rows(native: pd.DataFrame, romanized: dict[str, pd.DataFrame], metrics: list[str],
             alpha: float = 0.05) -> list[GapRow]:
    """Compare a native run's per-query scores with each romanized run's.

    ``native`` and every value of ``romanized`` are per-query score tables
    (index: qid). Each comparison is restricted to the romanized set's queries,
    which must all be in the native table. The Bonferroni family is the
    romanized sets compared with this native run (m = len(romanized)), per metric.
    """
    rows = []
    for query_set, roman in romanized.items():
        missing = set(roman.index) - set(native.index)
        if missing:
            raise ValueError(f"{query_set}: {len(missing)} queries are not in the native run")
        same_queries = native.loc[roman.index]
        for metric in metrics:
            result = compare(same_queries, roman, metric)
            rows.append(GapRow(
                query_set=query_set, metric=metric, n_queries=result.n_queries,
                native=result.mean_a, roman=result.mean_b, p=result.p,
                significant=bonferroni_significant(result.p, len(romanized), alpha),
            ))
    return rows


def gap_recovered(native: float, roman: float, method: float) -> float:
    """Share of the native-vs-romanized gap that a method closes (SPECS §7.2):

        GapRecovered = (M_method - M_roman) / (M_native - M_roman)

    0 means no improvement over the romanized baseline, 1 (100%) means the
    method reaches native effectiveness, above 1 means it beats native.
    NaN if there is no gap to close.
    """
    gap = native - roman
    return (method - roman) / gap if gap else float("nan")
