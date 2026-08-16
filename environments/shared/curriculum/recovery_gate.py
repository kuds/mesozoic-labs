"""The recovery gate statistic: ``recovery_quality/v1`` (stage 1b, W4).

What it certifies (STAGE1_SPLIT_PLAN §3.4): not survival under pushes but
**per-shove recovery** — after each scheduled push the body re-enters a
safe pose/velocity/contact set within a bounded number of control steps
and *stays* there for a dwell window.  An episode succeeds when it reaches
the full horizon AND every push in it was recovered; the gate certifies a
one-sided 95% lower confidence bound on the episode success probability.

Why an exact binomial bound: recovery success is a genuine binary event
per episode (unlike stance duty, which is continuous), so the Student-t
machinery the stance gate uses is the wrong shape here.  The bounds are
exact Clopper-Pearson, computed scipy-free by bisection on the binomial
tail; the split plan's own pinned figures are the regression tests
(LCB95 of 34/40 = 0.72526; one-sided 95% upper bound of 0/40 = 7.216%).

The paired null-superiority statistic lives here too: the same seeds and
the same push schedules are run under the policy and under each null
controller, and the gate can additionally require a positive lower bound
on the mean per-seed success difference.  Pairing is the point — the
schedule PRF guarantees the nulls saw identical pushes, so the difference
isolates control.

All thresholds are PROVISIONAL until the calibration runs (plan P3/P5)
measure the finalized pushed task; the recovery stage's config declares
``none/v1`` until then, and the resolver (W5) freezes the null baselines
this gate's paired criterion consumes.  Everything fails closed: NaN or
missing inputs are gate failures, never passes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from .stance_gate import one_sided_t95

RECOVERY_GATE_KIND = "recovery_quality/v1"


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


def binomial_ucb(failures_complement_successes: int, trials: int, *, alpha: float = 0.05) -> float:
    """Exact one-sided Clopper-Pearson UPPER bound on a success probability.

    Solves ``P(X <= successes | p) = alpha``.  1.0 when every trial
    succeeded.  For 0/40 this is the split plan's 7.216% — the honest
    ceiling on a null that never succeeded, which is a bound, not zero.
    """
    successes = failures_complement_successes
    if trials <= 0:
        raise ValueError("trials must be positive")
    if not 0 <= successes <= trials:
        raise ValueError("successes must be in [0, trials]")
    if successes == trials:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if _binom_cdf(successes, trials, mid) > alpha:
            lo = mid
        else:
            hi = mid
    return hi


def paired_difference_lcb(differences: Sequence[float]) -> float:
    """One-sided 95% lower bound on the mean of paired differences.

    The paired policy-minus-null success differences share seeds and push
    schedules, so this bounds the CONTROL contribution.  Uses the same
    tabulated one-sided t95 quantile as the stance gate; NaN (fail-closed)
    when fewer than two pairs exist or any difference is non-finite.
    """
    n = len(differences)
    if n < 2 or any(not math.isfinite(d) for d in differences):
        return math.nan
    mean = sum(differences) / n
    variance = sum((d - mean) ** 2 for d in differences) / (n - 1)
    return mean - one_sided_t95(n - 1) * math.sqrt(variance / n)


def per_push_recovery(
    safe_mask: Sequence[bool],
    *,
    push_end_step: int,
    t_recover_steps: int,
    dwell_steps: int,
) -> tuple[bool, int]:
    """Judge one push: did the body re-enter the safe set in time and stay?

    ``safe_mask[t]`` is the per-control-step safe-set predicate (the harness
    computes it from pose/velocity/contact signals; this function is pure
    event logic).  Recovery requires some step ``s`` with
    ``push_end_step <= s <= push_end_step + t_recover_steps`` where the mask
    holds for ``dwell_steps`` consecutive steps starting at ``s`` — re-entry
    alone is not recovery; a body that touches the safe set mid-fall and
    leaves again has not recovered.

    Returns ``(recovered, recovery_step)`` with ``recovery_step = -1`` when
    no qualifying entry exists (including when the episode ended first —
    a truncated mask simply has no room for the dwell, which is the
    correct fail-closed reading).
    """
    if dwell_steps < 1 or t_recover_steps < 0:
        raise ValueError("dwell_steps must be >= 1 and t_recover_steps >= 0")
    horizon = len(safe_mask)
    for start in range(push_end_step, min(push_end_step + t_recover_steps, horizon - dwell_steps) + 1):
        if start < 0 or start + dwell_steps > horizon:
            continue
        if all(safe_mask[start : start + dwell_steps]):
            return True, start
    return False, -1


def episode_recovery_success(full_horizon: bool, per_push_recovered: Sequence[bool]) -> bool:
    """The episode-level event the gate counts: survive AND recover every push.

    An episode with no in-horizon pushes counts only through full_horizon —
    the schedule guarantees pushes for any full-length episode, so a
    push-free episode is a short one and has already failed.
    """
    return bool(full_horizon) and all(bool(r) for r in per_push_recovered)


@dataclass(frozen=True)
class RecoveryGateThresholds:
    """The ``recovery_quality/v1`` criteria.  All PROVISIONAL until P5.

    ``min_recovery_success_lcb`` is the load-bearing bound.
    ``min_paired_success_delta_lcb`` is optional (None disables) because it
    needs the frozen null panels the resolver (W5) provides; once those
    exist it becomes the authoritative criterion per the split plan.
    """

    min_recovery_success_lcb: float
    t_recover_steps: int
    dwell_steps: int
    min_eval_episodes: int = 40
    min_paired_success_delta_lcb: "float | None" = None


@dataclass(frozen=True)
class RecoveryPanel:
    """One evaluation panel's episode outcomes (and optional paired nulls)."""

    episode_successes: tuple[bool, ...]
    #: Per-seed policy-minus-null success differences (same seeds, same
    #: schedules), or None when no null panel was run.
    paired_null_differences: "tuple[float, ...] | None" = None


@dataclass(frozen=True)
class RecoveryGateResult:
    passed: bool
    failures: tuple[str, ...]
    success_fraction: float
    success_lcb: float
    paired_delta_lcb: "float | None"


def evaluate_recovery_gate(panel: RecoveryPanel, thresholds: RecoveryGateThresholds) -> RecoveryGateResult:
    """Judge a panel against the recovery criteria, naming every failure."""
    failures: list[str] = []
    n = len(panel.episode_successes)
    successes = sum(1 for s in panel.episode_successes if s)
    if n < thresholds.min_eval_episodes:
        failures.append(f"n_episodes {n} < min_eval_episodes {thresholds.min_eval_episodes}")
    success_fraction = successes / n if n else math.nan
    success_lcb = binomial_lcb(successes, n) if n else math.nan
    if not math.isfinite(success_lcb) or success_lcb < thresholds.min_recovery_success_lcb:
        failures.append(
            f"recovery_success_lcb {success_lcb:.4f} < {thresholds.min_recovery_success_lcb:.4f} "
            f"({successes}/{n} episodes recovered every push to the horizon)"
        )
    paired_delta_lcb: "float | None" = None
    if thresholds.min_paired_success_delta_lcb is not None:
        if panel.paired_null_differences is None:
            # The criterion is declared but its evidence was not produced:
            # fail closed, never skip.  Missing baselines block advancement
            # (split plan §5); the resolver supplies them.
            failures.append("paired null panel missing while min_paired_success_delta_lcb is declared")
        else:
            paired_delta_lcb = paired_difference_lcb(panel.paired_null_differences)
            if not math.isfinite(paired_delta_lcb) or paired_delta_lcb < thresholds.min_paired_success_delta_lcb:
                failures.append(
                    f"paired_success_delta_lcb {paired_delta_lcb:.4f} < {thresholds.min_paired_success_delta_lcb:.4f}"
                )
    return RecoveryGateResult(
        passed=not failures,
        failures=tuple(failures),
        success_fraction=success_fraction,
        success_lcb=success_lcb,
        paired_delta_lcb=paired_delta_lcb,
    )
