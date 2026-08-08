"""Curriculum-gate evaluation.

Two entry points, for two different questions:

* :func:`evaluate_recorded_gate` — "does this *history* of evaluations show a
  pass?", used by the catalog and sweep reporting to describe finished runs.
* :func:`evaluate_stage_gate` — "may this completed stage advance, and may its
  artifacts claim it passed?", used by the trainers.

The second exists because it did not, and every trainer carried its own copy
of the rule.  ``notebooks/sb3_training.ipynb`` checked ``min_avg_reward`` /
``min_avg_episode_length`` / ``min_avg_forward_vel`` / ``min_success_rate``
inline and knew nothing about ``gate_kind``; when T-Rex stage 1 moved to
``stance_quality/v1`` and retired ``min_avg_episode_length``, that checklist
silently degraded to *reward alone* — the one criterion the zero-action statue
clears by 68%, and the exact reading the stance gate exists to refute.  Run
``20260802_203215`` advanced to stage 2 with ``publication_gate_passed = True``
recorded beside a ``stance_gate_report.txt`` reading ``GATE: FAIL`` at 10.6x
the duty ceiling.

So this function is the only implementation, it dispatches on the declared
``gate_kind``, and it is fail-closed at every branch: an undeclared kind, an
unknown kind, a missing stance panel and an unreadable verdict all return
*False* with a reason, because "we could not check" must never read as "it
passed".
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def evaluate_recorded_gate(
    curriculum: dict[str, Any],
    evaluations: list[dict[str, Any]],
) -> bool | None:
    """Evaluate a curriculum gate only when every enabled metric is recorded.

    Evaluations must be chronological. ``None`` means the available records
    cannot prove either a pass or a failure—for example, when a velocity or
    success-rate gate is enabled but that metric was not saved.
    """
    criteria: list[tuple[str, float]] = []
    ceilings: list[tuple[str, float]] = []
    if curriculum.get("min_avg_reward") is not None:
        criteria.append(("mean_reward", float(curriculum["min_avg_reward"])))
    if curriculum.get("min_avg_episode_length") is not None:
        criteria.append(("mean_episode_length", float(curriculum["min_avg_episode_length"])))
    if float(curriculum.get("min_avg_forward_vel") or 0.0) > 0.0:
        criteria.append(("mean_forward_vel", float(curriculum["min_avg_forward_vel"])))
    if float(curriculum.get("min_success_rate") or 0.0) > 0.0:
        criteria.append(("mean_success_rate", float(curriculum["min_success_rate"])))
    # stance_quality/v1. Without these a stance-gated stage would be reported
    # as gated on its reward RAIL alone -- which is the claim that gate exists
    # to refute, since the statue clears the rail by 68%.
    if curriculum.get("min_full_horizon_fraction") is not None:
        criteria.append(("full_horizon_fraction", float(curriculum["min_full_horizon_fraction"])))
    if curriculum.get("max_unsupported_duty") is not None:
        ceilings.append(("mean_unsupported_duty", float(curriculum["max_unsupported_duty"])))
    if curriculum.get("max_unsupported_duty_ucb") is not None:
        ceilings.append(("unsupported_duty_ucb", float(curriculum["max_unsupported_duty_ucb"])))
    if not (criteria or ceilings) or not evaluations:
        return None

    min_eval_episodes = int(curriculum.get("min_eval_episodes", 10))
    required_consecutive = int(curriculum.get("required_consecutive", 3))
    consecutive = 0
    incomplete = False
    for evaluation in evaluations:
        required_values = [evaluation.get(key) for key, _ in (*criteria, *ceilings)]
        n_episodes = evaluation.get("n_episodes")
        if any(value is None for value in required_values) or n_episodes is None:
            incomplete = True
            consecutive = 0
            continue

        passes = (
            int(n_episodes) >= min_eval_episodes
            and all(float(evaluation[key]) >= threshold for key, threshold in criteria)
            and all(float(evaluation[key]) <= ceiling for key, ceiling in ceilings)
        )
        consecutive = consecutive + 1 if passes else 0
        if consecutive >= required_consecutive:
            return True

    return None if incomplete else False


def _gate_metric(stage_results: Mapping[str, Any], *keys: str) -> float | None:
    """First finite float among *keys*, or ``None`` when none is present.

    The trainers write ``""`` rather than ``None`` for a metric they could not
    measure, so both sentinels have to fall through to the next candidate.

    Non-finite values fall through too, and that is load-bearing rather than
    tidiness: ``nan < threshold`` is ``False``, so an unfiltered NaN reward
    *clears* every floor below.  Measured on this implementation before the
    guard existed — ``min_avg_reward = 100`` against a NaN best-model reward
    returned ``(True, [])``.  Falling through reaches ``None``, which the
    callers report as an unmeasurable criterion and fail.

    The per-value rule itself is :func:`~environments.shared.curriculum.
    gate_schema.finite_gate_metric`, shared with the JAX backend so the two
    cannot drift on what counts as measured.
    """
    from environments.shared.curriculum.gate_schema import finite_gate_metric

    for key in keys:
        number = finite_gate_metric(stage_results.get(key))
        if number is not None:
            return number
    return None


def _stance_stage_gate(
    gate_kind: str,
    stance_report: Mapping[str, Any] | None,
    stage: int | str,
) -> tuple[bool, list[str]]:
    """Read the verdict off a stance gate report, refusing every substitute.

    Deliberately does not re-derive anything.  ``evaluate_stance_gate`` already
    checks the panel size, the full-horizon floor, both duty ceilings and the
    reward rail; re-implementing any of them here would recreate exactly the
    divergence this module was written to end.
    """
    if stance_report is None:
        return False, [
            f"stage {stage} declares {gate_kind} but no stance panel was measured, so its "
            "duty criteria are unproven. The gate does not fall back to the reward rail: "
            "the zero-action statue clears that rail by 68%, which is what this gate kind "
            "exists to reject."
        ]
    reported_kind = stance_report.get("gate_kind")
    if reported_kind != gate_kind:
        return False, [
            f"stage {stage} declares {gate_kind} but the stance report scored "
            f"{reported_kind!r}; a verdict for a different gate cannot certify this one"
        ]
    passed = stance_report.get("passed")
    if not isinstance(passed, bool):
        return False, [
            f"stage {stage} stance report carries no boolean verdict (passed={passed!r}), "
            "so it proves neither a pass nor a failure"
        ]
    if passed:
        return True, []
    # A bare string is iterable, and comprehending over one shreds the reason
    # into single characters — the raised message then says nothing usable.
    reported = stance_report.get("failures", ())
    failures = [reported] if isinstance(reported, str) else [str(failure) for failure in reported]
    return False, failures or [f"stage {stage} failed {gate_kind} without naming a criterion"]


def _reward_and_length_stage_gate(
    curriculum: Mapping[str, Any],
    stage_results: Mapping[str, Any],
) -> tuple[bool, list[str]]:
    """The historical conjunction, over the selected checkpoint's metrics.

    Thresholds are applied only when the stage declares them, and the velocity
    and success floors only when positive — the same semantics the notebook
    applied inline, preserved exactly so no existing stage changes verdict.

    A block that declares this kind and then sets no threshold at all is the
    degenerate case of those semantics: an empty conjunction is vacuously true,
    so it passed everything.  ``gate_schema`` requires ``min_avg_reward`` for
    this kind, but that check runs at config load and this function is reachable
    without it, so the emptiness is caught here too rather than assumed away.
    """
    failures: list[str] = []
    checked = 0

    target_reward = curriculum.get("min_avg_reward")
    if target_reward is not None:
        checked += 1
        reward = _gate_metric(stage_results, "best_model_reward", "best_eval_reward", "mean_reward")
        if reward is None:
            failures.append(f"no reward measurement available to check min_avg_reward {float(target_reward):.2f}")
        elif reward < float(target_reward):
            failures.append(f"best model reward {reward:.2f} < {float(target_reward):.2f}")

    target_length = curriculum.get("min_avg_episode_length")
    if target_length is not None:
        checked += 1
        length = _gate_metric(stage_results, "best_model_length", "best_eval_length", "mean_episode_length")
        if length is None:
            failures.append(
                f"no episode-length measurement available to check min_avg_episode_length {float(target_length):.1f}"
            )
        elif length < float(target_length):
            failures.append(f"best model episode length {length:.1f} < {float(target_length):.1f}")

    target_fwd_vel = float(curriculum.get("min_avg_forward_vel") or 0.0)
    if target_fwd_vel > 0.0:
        checked += 1
        fwd_vel = _gate_metric(stage_results, "best_model_fwd_vel", "mean_forward_vel")
        if fwd_vel is None:
            failures.append(
                f"no forward-velocity measurement available to check min_avg_forward_vel {target_fwd_vel:.2f}"
            )
        elif fwd_vel < target_fwd_vel:
            failures.append(f"best model forward vel {fwd_vel:.2f} m/s < {target_fwd_vel:.2f} m/s")

    target_success = float(curriculum.get("min_success_rate") or 0.0)
    if target_success > 0.0:
        checked += 1
        success = _gate_metric(stage_results, "best_model_success_rate", "mean_success_rate")
        if success is None:
            failures.append(f"no success-rate measurement available to check min_success_rate {target_success:.0%}")
        elif success < target_success:
            failures.append(f"best model success rate {success:.0%} < {target_success:.0%}")

    if not checked:
        return False, [
            'gate_kind "reward_and_length/v1" is declared with no threshold set, so it '
            "checks nothing and would pass any policy; declare min_avg_reward, or "
            'gate_kind = "none/v1" for a non-advancing pilot'
        ]
    return not failures, failures


def evaluate_stage_gate(
    curriculum: Mapping[str, Any],
    stage_results: Mapping[str, Any],
    *,
    stage: int | str,
    stance_report: Mapping[str, Any] | None = None,
) -> tuple[bool, list[str]]:
    """Decide whether a completed stage passes its declared curriculum gate.

    Args:
        curriculum: The stage's ``curriculum_kwargs`` block.
        stage_results: The stage result dict, read for the selected
            checkpoint's metrics (``best_model_*``, falling back to
            ``best_eval_*`` and then the live-eval means).
        stage: Stage identifier, used only in failure messages.
        stance_report: The ``mesozoic.stance-gate-report/v2`` dict produced for
            this stage, required by ``stance_quality/v1`` and ignored by every
            other kind.

    Returns:
        ``(passed, failures)``.  *failures* names every criterion that did not
        hold, so the reason survives into ``gate_failures`` and the raised
        message without re-running anything.
    """
    # Imported here rather than at module scope to break the import cycle:
    # `environments.shared.curriculum` re-exports the SB3 callbacks, several of
    # which reach back into `reporting`, so a module-scope import here would
    # close the loop at import time.
    #
    # It is NOT what keeps `reporting` importable without stable-baselines3 --
    # importing the `gate_schema` SUBMODULE executes the `curriculum` package
    # `__init__` regardless, so deferring changes only when that happens.
    # SB3-optionality comes from `curriculum.sb3_compat`, which makes the
    # callbacks raise at construction rather than at import; the whole test
    # suite passes with stable-baselines3 absent because of that, not this.
    from environments.shared.curriculum.gate_schema import GATE_KINDS
    from environments.shared.curriculum.stance_gate import STANCE_GATE_KIND

    gate_kind = curriculum.get("gate_kind")
    if gate_kind is None:
        return False, [
            f"stage {stage} declares no gate_kind, so nothing certifies it. A stage with "
            'no gate must say so explicitly with gate_kind = "none/v1".'
        ]
    if gate_kind not in GATE_KINDS:
        return False, [f"stage {stage} declares unknown gate_kind {gate_kind!r}; known kinds: {sorted(GATE_KINDS)}"]
    if gate_kind == "none/v1":
        return False, [
            f'stage {stage} declares gate_kind "none/v1", a non-advancing pilot; it refuses '
            "to advance rather than passing by default"
        ]
    if gate_kind == STANCE_GATE_KIND:
        return _stance_stage_gate(gate_kind, stance_report, stage)
    return _reward_and_length_stage_gate(curriculum, stage_results)
