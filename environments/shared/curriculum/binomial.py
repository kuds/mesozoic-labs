"""The exact binomial lower confidence bound the certification gates share.

``binomial_lcb`` is the one-sided Clopper-Pearson lower bound, computed
scipy-free by bisection on the binomial tail.  The recovery gate
(``recovery_quality/v1``), the task-success gate and the gait gate
(``locomotion_gait/v2``) judge their success counts with it, so this module
is part of the gait measurement identity (``gait/identity.py``): an edit here
changes which gait panels certify, and makes every planned gait hash stale.
``recovery_gate`` re-exports it, its home before it moved here.
"""

from __future__ import annotations

import math


def _log_binom_pmf(k: int, n: int, p: float) -> float:
    if p <= 0.0:
        return 0.0 if k == 0 else -math.inf
    if p >= 1.0:
        return 0.0 if k == n else -math.inf
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) + k * math.log(p) + (n - k) * math.log1p(-p)


def _binom_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k) for X ~ Binomial(n, p), in a numerically safe direct sum."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    total = 0.0
    for i in range(0, k + 1):
        total += math.exp(_log_binom_pmf(i, n, p))
    return min(total, 1.0)


def binomial_lcb(successes: int, trials: int, *, alpha: float = 0.05) -> float:
    """Exact one-sided Clopper-Pearson LOWER bound on a success probability.

    The smallest p that the data cannot reject at level *alpha*: solves
    ``P(X >= successes | p) = alpha``.  0.0 when there are no successes.
    """
    if trials <= 0:
        raise ValueError("trials must be positive")
    if not 0 <= successes <= trials:
        raise ValueError("successes must be in [0, trials]")
    if successes == 0:
        return 0.0
    lo, hi = 0.0, 1.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if 1.0 - _binom_cdf(successes - 1, trials, mid) < alpha:
            lo = mid
        else:
            hi = mid
    return lo
