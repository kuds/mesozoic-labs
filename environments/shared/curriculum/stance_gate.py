"""The stance-quality advancement gate: per-episode metrics and its bound.

Stage 1a cannot be gated on return.  On stage 1 as configured the zero-action
statue is the *global optimum* -- it collects 3271.8 of a theoretical 3350
maximum, with ``energy`` and ``smoothness`` at exactly zero, while any active
policy pays both.  So a policy's realistic ceiling sits *below* the statue's
score and no reward threshold admits a competent policy while excluding a
passive one.  See docs/STAGE1_SPLIT_PLAN.md section 2.3.1.

This module supplies the replacement: a gate on **physical stance quality**,
which does not rescale when a reward weight changes.

What the gate is for
--------------------
Not for beating the statue.  Section 2.3.1 is explicit that the statue's
numbers are "what a 1a policy must match, not beat" -- the statue defines the
quality ceiling, and 1a's job is to certify that a policy has *not bought
stability with actuation*.  Robustness is stage 1b's job, delivered through
declared ``xfrc_applied`` perturbations rather than through reset noise, where
a lucky draw and a good controller are indistinguishable.

The policy this gate exists to reject is therefore the **chatterer**.  Run
``20260801_021545``'s best checkpoint reached the full horizon in every rolled
out episode and scored 2347.67 -- clearing ``min_avg_reward = 1950`` -- while
spending **28.4%** of its steps with neither foot bearing load.  Against the
statue's 0.000 that is the defect stage 1a is supposed to catch, and the
reward gate waved it through.

The statistic
-------------
Three parts, adopted 2026-08-01:

1. **Screening, every evaluation** -- raw fractions: ``full_horizon_fraction``
   and mean unsupported duty, with ``required_consecutive`` acting as
   scheduler hysteresis ONLY.  Re-running a deterministic panel is not
   statistical replication.
2. **The load-bearing bound** -- a one-sided 95% *upper* confidence bound on
   mean unsupported duty.  This is what actually certifies stance quality.
3. **Confirmation** -- one predeclared held-out panel at n ~ 100-180 for the
   binary full-horizon event, run once a candidate qualifies.  That step is
   offline and deliberately lives outside this module.

Why a bound on duty rather than on a pass/fail count: binarising costs almost
all the power available at this panel size.  Measured, a binomial
``LCB95 >= 0.90`` at n=40 requires a clean 40/40 and **would reject the
statue 28% of the time** -- pooled over three independent 40-seed blocks the
zero-action policy scores 119/120 = 99.17%, and at that true rate the chance
of scoring 40/40 is only 71.6%.  A gate that rejects the theoretical optimum
more than a quarter of the time is not measuring the policy.  Duty is
continuous with tiny measured variance, so an interval on it has real power
at 40 episodes where the count has almost none.

**Direction.** Section 2.3 of STAGE1_SPLIT_PLAN says "one-sided LCB on mean
unsupported duty".  That is a slip: duty is a quantity we require to stay
*below* a ceiling, so the conservative direction is the UPPER bound.  A lower
bound would pass a policy whose mean already exceeded the ceiling whenever the
panel was noisy enough, which inverts the gate.  This module computes the
upper bound and the doc has been corrected to match.

Two semantics that had to be decided rather than inferred
---------------------------------------------------------
*Failed episodes are excluded from duty*, not scored as duty 1.0.  Survival is
already gated by ``min_full_horizon_fraction``; folding it into duty as well
double-counts it.  The alternative -- averaging duty over whatever steps a
failed episode happened to run -- is actively perverse: a policy that
faceplants at step 50 would earn a flattering duty score from its 50 cleanest
steps.  Excluding them keeps the two criteria independent, which is what makes
the pair readable.

*Duty is measured after a settling window* (``settle_steps``), because a
policy that corrects reset randomisation may legitimately move a foot doing
it, and settling is the capability 1a exists to certify.  The 200-step default
is ``[inferred]``, not measured -- it is 20% of the horizon and should be
calibrated against rollout data before it is treated as evidence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

#: The gate kind this module evaluates.  Imported by ``gate_schema`` so the
#: registry and the implementation cannot drift apart.
STANCE_GATE_KIND = "stance_quality/v1"

#: One-sided Student-t quantiles at 95%, by degrees of freedom.  Tabulated
#: rather than taken from ``scipy.stats``: scipy is declared only in the
#: ``test`` extra (pyproject.toml), and a gate that decides advancement during
#: a Colab training run must not depend on a package the training extra does
#: not install.  Interpolation is linear in ``1/dof``, which reproduces the
#: exact quantile to within 2e-4 over dof 1-5000 -- verified against
#: ``scipy.stats.t.ppf`` in the tests, where scipy *is* available.  At the
#: gate's own operating point, dof 39, the table is exact to 5 decimals.  The
#: residual is not signed (interpolation both over- and under-shoots by up to
#: 1.4e-4), but it is three orders of magnitude below the duty ceiling this
#: bound is compared against, so it cannot change a gate decision.
_T95_BY_DOF: tuple[tuple[float, float], ...] = (
    (1, 6.3138),
    (2, 2.9200),
    (3, 2.3534),
    (4, 2.1318),
    (5, 2.0150),
    (6, 1.9432),
    (7, 1.8946),
    (8, 1.8595),
    (9, 1.8331),
    (10, 1.8125),
    (12, 1.7823),
    (15, 1.7531),
    (20, 1.7247),
    (25, 1.7081),
    (29, 1.6991),
    (39, 1.6849),
    (49, 1.6766),
    (59, 1.6711),
    (99, 1.6604),
    (199, 1.6525),
)

#: The dof -> infinity limit of the one-sided 95% t quantile (the normal z).
_Z95 = 1.6449


@dataclass(frozen=True)
class StanceGateThresholds:
    """The ``stance_quality/v1`` thresholds, as declared in ``[curriculum]``."""

    min_full_horizon_fraction: float
    max_unsupported_duty: float
    max_unsupported_duty_ucb: float
    settle_steps: int = 0
    min_eval_episodes: int = 40
    min_avg_reward: float = -math.inf
    required_consecutive: int = 3


def one_sided_t95(dof: int) -> float:
    """Return the one-sided 95% Student-t quantile for ``dof``.

    Interpolates linearly in ``1/dof`` between tabulated values.  Below the
    table the smallest tabulated dof is used (the most conservative, i.e.
    largest, quantile); above it the normal limit.
    """
    if dof < 1:
        return math.inf
    if dof >= _T95_BY_DOF[-1][0]:
        # Past the table, interpolate toward the normal limit the same way.
        last_dof, last_t = _T95_BY_DOF[-1]
        if dof == last_dof:
            return last_t
        weight = (1.0 / dof) / (1.0 / last_dof)
        return _Z95 + weight * (last_t - _Z95)
    for (lo_dof, lo_t), (hi_dof, hi_t) in zip(_T95_BY_DOF, _T95_BY_DOF[1:]):
        if lo_dof <= dof <= hi_dof:
            if dof == lo_dof:
                return lo_t
            span = (1.0 / lo_dof) - (1.0 / hi_dof)
            weight = ((1.0 / dof) - (1.0 / hi_dof)) / span
            return hi_t + weight * (lo_t - hi_t)
    return _T95_BY_DOF[0][1]


def mean_upper_confidence_bound(values: Sequence[float]) -> float:
    """One-sided 95% upper confidence bound on the mean of ``values``.

    Returns ``inf`` for fewer than two observations, so an under-powered panel
    fails the gate rather than passing it on no evidence.

    A zero-variance panel returns the sample mean exactly.  That is the
    statue's case -- every episode measures 0.000 -- and it is accepted here
    rather than floored: duty is a physical occupancy fraction measured on a
    deterministic panel, not the rate of a rare event, so identical
    observations really are evidence of an identical underlying quantity.
    """
    sample = np.asarray(list(values), dtype=float)
    n = sample.size
    if n < 2:
        return math.inf
    mean = float(sample.mean())
    # ddof=1: this is an inference about the population mean, not a
    # description of the panel.
    std_err = float(sample.std(ddof=1)) / math.sqrt(n)
    return mean + one_sided_t95(n - 1) * std_err


def episode_unsupported_duty(
    unsupported_flags: Sequence[float] | Sequence[bool],
    *,
    settle_steps: int = 0,
) -> float | None:
    """Fraction of post-settling steps in which neither foot bore load.

    Args:
        unsupported_flags: Per-step truthy values, one per environment step of
            a single episode, true when neither foot is bearing load.
        settle_steps: Leading steps to discard before measuring.

    Returns:
        The duty fraction, or ``None`` when the episode is shorter than the
        settling window and therefore carries no measurable tail.
    """
    flags = np.asarray(list(unsupported_flags), dtype=float)
    if flags.size <= settle_steps:
        return None
    return float(flags[settle_steps:].mean())


@dataclass(frozen=True)
class StancePanel:
    """The per-evaluation summary the gate consumes.

    Deliberately a flat value object of scalars rather than a holder of
    per-episode traces: it is recorded into ``CurriculumManager``'s evaluation
    history, which is serialized into run summaries and result bundles, and
    per-step traces have no business there.  It also means the panel a gate
    decision was made from round-trips exactly through that history, so the
    decision can be re-derived after the fact.

    ``mean_unsupported_duty`` and ``unsupported_duty_ucb`` are ``inf`` when
    the panel could not measure them, which fails the gate rather than
    passing it on absent evidence.
    """

    n_episodes: int
    full_horizon_fraction: float
    mean_reward: float
    #: Number of full-horizon episodes that supplied a measurable duty.
    n_duty_episodes: int
    mean_unsupported_duty: float
    unsupported_duty_ucb: float


def stance_panel_from_episode_duties(
    *,
    episode_lengths: Sequence[float],
    episode_duties: Sequence[float | None],
    episode_rewards: Sequence[float],
    horizon: int,
) -> StancePanel:
    """Build the panel from per-episode duties that a caller already reduced.

    The streaming callers -- SB3's evaluation callback and the MJX rollout --
    accumulate duty per episode as steps arrive rather than retaining a
    per-step trace for every episode of the panel.  They hand the reduced
    scalars here so the rules that actually define the gate stay in one place:
    which episodes count (full-horizon only), how their duties combine, and
    how the bound is formed.  Only the mechanical act of applying the settling
    window while streaming happens outside this module.

    ``episode_duties`` is positionally aligned with ``episode_lengths``; use
    ``None`` for an episode whose duty could not be measured.
    """
    lengths = np.asarray(list(episode_lengths), dtype=float)
    n = lengths.size
    reached = lengths >= horizon

    duties = [
        float(duty)
        for index, duty in enumerate(episode_duties)
        if index < n and reached[index] and duty is not None and math.isfinite(float(duty))
    ]

    return StancePanel(
        n_episodes=int(n),
        full_horizon_fraction=float(reached.mean()) if n else 0.0,
        mean_reward=float(np.mean(episode_rewards)) if len(episode_rewards) else -math.inf,
        n_duty_episodes=len(duties),
        mean_unsupported_duty=float(np.mean(duties)) if duties else math.inf,
        unsupported_duty_ucb=mean_upper_confidence_bound(duties) if duties else math.inf,
    )


def summarize_stance_panel(
    *,
    episode_lengths: Sequence[float],
    episode_unsupported: Iterable[Sequence[float] | Sequence[bool]],
    episode_rewards: Sequence[float],
    horizon: int,
    settle_steps: int = 0,
) -> StancePanel:
    """Reduce one evaluation's per-episode traces to the gate's panel summary.

    ``episode_unsupported`` yields one per-step flag sequence per episode, in
    the same order as ``episode_lengths``.  Only episodes that reached
    ``horizon`` contribute duty -- see the module docstring for why failures
    are excluded rather than scored as 1.0.

    Both backends must funnel through this function.  It is where the
    settling window and the failed-episode rule are applied, and a backend
    that reduced traces itself would be free to disagree with the other one
    about what the gate means.
    """
    return stance_panel_from_episode_duties(
        episode_lengths=episode_lengths,
        episode_duties=[episode_unsupported_duty(flags, settle_steps=settle_steps) for flags in episode_unsupported],
        episode_rewards=episode_rewards,
        horizon=horizon,
    )


def evaluate_stance_gate(
    panel: StancePanel,
    thresholds: StanceGateThresholds,
) -> tuple[bool, list[str]]:
    """Check one evaluation panel against the gate.

    Returns:
        ``(passed, failures)`` where *failures* names every criterion that did
        not hold, in declaration order, so a near-miss is diagnosable from the
        log without re-running the evaluation.
    """
    failures: list[str] = []

    if panel.n_episodes < thresholds.min_eval_episodes:
        failures.append(
            f"n_episodes {panel.n_episodes} < {thresholds.min_eval_episodes} "
            "(the bound's power is specified at this panel size)"
        )

    if panel.full_horizon_fraction < thresholds.min_full_horizon_fraction:
        failures.append(
            f"full_horizon_fraction {panel.full_horizon_fraction:.4f} < {thresholds.min_full_horizon_fraction:.4f}"
        )

    if panel.n_duty_episodes == 0:
        failures.append("no full-horizon episode supplied a measurable unsupported duty")
    else:
        if panel.mean_unsupported_duty > thresholds.max_unsupported_duty:
            failures.append(
                f"mean_unsupported_duty {panel.mean_unsupported_duty:.4f} > {thresholds.max_unsupported_duty:.4f}"
            )
        # The certifying criterion.  Note UCB >= mean by construction, so this
        # subsumes the screening check whenever both ceilings are equal; both
        # are kept configurable because a config may legitimately want a tight
        # screen with a looser certified bound.
        if panel.unsupported_duty_ucb > thresholds.max_unsupported_duty_ucb:
            failures.append(
                f"unsupported_duty_ucb {panel.unsupported_duty_ucb:.4f} > {thresholds.max_unsupported_duty_ucb:.4f}"
            )

    if panel.mean_reward < thresholds.min_avg_reward:
        # A rail, not the gate: set well below the statue on purpose, its only
        # job is rejecting a policy that threw away most of the return.
        failures.append(f"mean_reward {panel.mean_reward:.1f} < {thresholds.min_avg_reward:.1f} (rail)")

    return (not failures), failures
