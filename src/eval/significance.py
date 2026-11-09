"""Is the difference between two runs real, or could it be chance? (SPECS §7.2)

**Paired t-test.** Two runs are scored on the *same* queries, so we compare them
query by query: for each query $q$ take the difference $d_q = a_q - b_q$ of a
metric. If the runs were equally good, the mean difference $\bar d$ would be
close to 0. The test statistic is

    t = mean(d) / (std(d) / sqrt(n))

and the p-value is the probability of a |t| at least this large if the true
mean difference were 0. Pairing removes the large query-to-query variation
(some queries are easy for every run), which makes the test much more
sensitive than comparing two unpaired averages.

**Bonferroni correction.** Each test at level alpha has an alpha chance of a
false "significant". With m comparisons against one baseline, a result counts
as significant only if p < alpha / m, which keeps the chance of *any* false
positive among the m tests at most alpha. It is the correction the reference
paper uses.
"""

import math
from dataclasses import dataclass

import pandas as pd
from scipy import stats

ALPHA = 0.05


@dataclass(frozen=True)
class Comparison:
    """Result of comparing run A with run B on one metric."""

    metric: str
    n_queries: int
    mean_a: float
    mean_b: float
    t: float
    p: float

    @property
    def diff(self) -> float:
        return self.mean_a - self.mean_b


def paired_ttest(a: pd.Series, b: pd.Series) -> tuple[float, float]:
    """Two-sided paired t-test of per-query scores ``a`` and ``b`` (indexed by qid).

    Both must cover exactly the same queries, otherwise ``ValueError``: a
    missing query would silently change what is being compared. If every
    per-query difference is 0 the runs are identical and ``(0.0, 1.0)`` is
    returned (scipy would give NaN there).
    """
    if set(a.index) != set(b.index):
        raise ValueError(f"runs cover different queries ({len(a)} vs {len(b)} queries)")
    b = b.loc[a.index]  # same query order on both sides
    if (a.to_numpy() == b.to_numpy()).all():
        return 0.0, 1.0
    result = stats.ttest_rel(a.to_numpy(), b.to_numpy())
    return float(result.statistic), float(result.pvalue)


def compare(per_query_a: pd.DataFrame, per_query_b: pd.DataFrame, metric: str) -> Comparison:
    """Compare two runs' per-query scores (from ``src.eval.evaluate``) on one metric."""
    a, b = per_query_a[metric], per_query_b[metric]
    t, p = paired_ttest(a, b)
    return Comparison(metric, len(a), float(a.mean()), float(b.mean()), t, p)


def bonferroni_significant(p: float, n_comparisons: int, alpha: float = ALPHA) -> bool:
    """True if ``p`` is below the Bonferroni-corrected threshold ``alpha / n_comparisons``."""
    if n_comparisons < 1:
        raise ValueError("n_comparisons must be at least 1")
    return not math.isnan(p) and p < alpha / n_comparisons
