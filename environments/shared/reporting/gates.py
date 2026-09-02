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

``recovery_quality/v1`` (stage 1b, plan P5) is dispatched here too, and it is
the one kind whose verdict this module does not compute at all: it delegates
to :func:`~environments.shared.curriculum.gate_resolver.evaluate_recovery_gate_from_resolution`,
which reads the stage's frozen ``gate_resolution.json`` — capability spec,
null manifest, and decision procedure, hashed together and pinned to a task
fingerprint.  Every input that path needs but this one was not given (the
stage directory, a recorded task fingerprint, the pushed panel's per-seed
successes) is a *refusal* naming what is missing, never a fall-through to the
reward gate: a pushed stage certified on return alone is precisely the
advance-on-unmeasured-evidence failure the gate architecture exists to stop.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
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
    measure, so both sentinels have to fall through to the next candidate — as
    does an absent key: ``build_stage_results_from_eval_data`` leaves the
    velocity/success keys out entirely when the post-training panel did not
    run, and every caller reports the resulting ``None`` as an unmeasured
    criterion by name (review ER4).

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


#: Stage-directory artifacts that record the task the stage actually ran
#: under.  ``config.save_stage_config`` writes both whenever a fingerprint
#: exists — the sidecar verbatim, the snapshot under ``task_fingerprint`` — so
#: either answers "what is the current task?" without this module re-deriving
#: it.  Re-deriving would answer a *different* question (the task as configured
#: now, not the one this stage ran) and would need the plant model loaded.
_TASK_FINGERPRINT_ARTIFACTS = ("task_fingerprint.json", "stage_config.json")

#: ``curriculum_kwargs`` keys the frozen capability spec records under the same
#: name.  The frozen record is authoritative for the verdict; these are checked
#: for *agreement* so a config edited without re-resolving cannot leave the
#: stage gated on a criterion nobody measured — the same "declared but not
#: enforced" hole the gate schema exists to close.
_RECOVERY_SPEC_KEYS = (
    "min_recovery_success_lcb",
    "recovery_t_recover_steps",
    "recovery_dwell_steps",
    "min_paired_success_delta_lcb",
    "min_eval_episodes",
)


def _current_task_sha256(stage_dir: Path) -> str | None:
    """The task fingerprint this stage ran under, from its own artifacts.

    ``None`` when neither artifact records one.  The recovery arm reads that
    as an unprovable staleness check and refuses: a frozen resolution whose
    task cannot be compared against the current one is indistinguishable from
    a stale one, and stale baselines block.
    """
    for name in _TASK_FINGERPRINT_ARTIFACTS:
        path = stage_dir / name
        if not path.is_file():
            continue
        try:
            record: Any = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            # A truncated or hand-edited artifact proves nothing.  Try the
            # next one; refusing is what happens if neither answers.
            continue
        if name == "stage_config.json" and isinstance(record, Mapping):
            record = record.get("task_fingerprint")
        if isinstance(record, Mapping):
            recorded = record.get("task_sha256")
            if isinstance(recorded, str) and recorded:
                return recorded
    return None


def _same_threshold(declared: Any, frozen: Any) -> bool:
    """Whether a declared threshold and a frozen one state the same criterion.

    Anything that will not compare as a number counts as a disagreement: the
    fail-closed reading of "these two records cannot be shown to agree".
    """
    if declared is None or frozen is None:
        return declared is None and frozen is None
    try:
        return float(declared) == float(frozen)
    except (TypeError, ValueError):
        return False


def _recovery_spec_disagreements(
    curriculum: Mapping[str, Any],
    resolution: Mapping[str, Any],
    *,
    stage: int | str,
) -> list[str]:
    """Every declared recovery threshold the frozen spec does not match."""
    spec = resolution.get("capability_spec")
    if not isinstance(spec, Mapping):
        return [
            f"stage {stage}'s gate_resolution.json carries no capability_spec, so the "
            "criteria it would be judged against are unreadable; re-resolve the gate"
        ]
    failures: list[str] = []
    for key in _RECOVERY_SPEC_KEYS:
        if key not in curriculum:
            continue
        if not _same_threshold(curriculum[key], spec.get(key)):
            failures.append(
                f"stage {stage} declares {key} = {curriculum[key]!r} but its frozen gate "
                f"resolution was resolved at {spec.get(key)!r}. The verdict comes from the "
                "frozen record, so a config edited without re-resolving would gate on a "
                "criterion nobody measured. Re-resolve the gate, or restore the declaration."
            )
    return failures


def _recovery_reward_rail(
    curriculum: Mapping[str, Any],
    stage_results: Mapping[str, Any],
    *,
    stage: int | str,
) -> list[str]:
    """Enforce the optional reward RAIL a recovery config may declare.

    ``recovery_quality/v1`` may carry ``min_avg_reward`` in the same role
    ``stance_quality/v1`` gives it — a rail well below the null, not the gate
    — but the frozen capability spec records no rail, so the resolver cannot
    enforce one.  A declared criterion nobody evaluates is the half-enforced
    gate this module exists to prevent, so it is checked here as an additional
    *conjunct*: it can only refuse, never advance anything the frozen verdict
    did not already pass.
    """
    target = curriculum.get("min_avg_reward")
    if target is None:
        return []
    reward = _gate_metric(stage_results, "best_model_reward", "best_eval_reward", "mean_reward")
    if reward is None:
        return [
            f"stage {stage} declares min_avg_reward {float(target):.2f} as a recovery rail, "
            "but no reward measurement is available to check it"
        ]
    if reward < float(target):
        return [f"stage {stage} best model reward {reward:.2f} < recovery rail {float(target):.2f}"]
    return []


def _recovery_stage_gate(
    curriculum: Mapping[str, Any],
    stage_results: Mapping[str, Any],
    *,
    stage: int | str,
    stage_dir: "str | Path | None",
    recovery_successes_by_seed: "Mapping[int, bool] | None",
) -> tuple[bool, list[str]]:
    """Judge ``recovery_quality/v1`` through the stage's FROZEN resolution.

    The verdict itself is produced by
    :func:`~environments.shared.curriculum.gate_resolver.evaluate_recovery_gate_from_resolution`
    and by nothing here: thresholds come from the frozen capability spec, the
    paired differences from the frozen null manifest (same seeds, same push
    schedules), and absence, tampering, and staleness have already blocked
    inside :func:`~environments.shared.curriculum.gate_resolver.require_gate_resolution`.
    This function's whole job is to hand that path its three inputs — the
    stage directory, the current task fingerprint, the pushed panel's per-seed
    successes — or to say precisely which one it does not have.

    ``GateResolutionError`` becomes a refusal with its message attached; it is
    never allowed to read as a pass, and it is never swallowed silently.
    """
    # Deferred for the same import-cycle reason as the dispatch below.
    from environments.shared.curriculum.gate_resolver import (
        GateResolutionError,
        evaluate_recovery_gate_from_resolution,
        require_gate_resolution,
    )

    if stage_dir is None:
        return False, [
            f"stage {stage} declares recovery_quality/v1 but this call carried no stage_dir, "
            "so its frozen gate_resolution.json cannot be read. A recovery verdict comes only "
            "from curriculum.gate_resolver.evaluate_recovery_gate_from_resolution, and no "
            "resolution means no advancement: missing baselines block, they are never skipped."
        ]
    if recovery_successes_by_seed is None:
        return False, [
            f"stage {stage} declares recovery_quality/v1 but no pushed-panel evidence was "
            "supplied (recovery_successes_by_seed). The gate's estimand is per-seed episode "
            "success on the registered panel seeds, paired against the frozen null manifest; "
            "roll it with recovery_evaluation.roll_recovery_panel and pass "
            "RecoveryPanelEvidence.successes_by_seed(). No evidence is a blocked gate, never "
            "a pass."
        ]
    if not recovery_successes_by_seed:
        return False, [
            f"stage {stage} declares recovery_quality/v1 and its pushed panel carried no "
            "episodes, so nothing was measured; an empty panel is a blocked gate, not a pass"
        ]

    resolved_dir = Path(stage_dir)
    current_task_sha256 = _current_task_sha256(resolved_dir)
    if current_task_sha256 is None:
        return False, [
            f"stage {stage} declares recovery_quality/v1 but {resolved_dir} records no task "
            f"fingerprint ({' or '.join(_TASK_FINGERPRINT_ARTIFACTS)}), so the frozen "
            "resolution's staleness check cannot run. A resolution that cannot be compared "
            "against the current task is treated as stale: recalibrate rather than proceeding."
        ]

    try:
        # Loaded once for the config-vs-frozen agreement check below; the
        # VERDICT still comes from the resolver's own entry point, which
        # re-reads and re-validates the same record.
        resolution = require_gate_resolution(resolved_dir, current_task_sha256=current_task_sha256)
        disagreements = _recovery_spec_disagreements(curriculum, resolution, stage=stage)
        if disagreements:
            return False, disagreements
        result = evaluate_recovery_gate_from_resolution(
            resolved_dir,
            current_task_sha256=current_task_sha256,
            policy_successes_by_seed=recovery_successes_by_seed,
        )
    except GateResolutionError as exc:
        return False, [f"stage {stage} recovery gate could not be resolved: {exc}"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        # A truncated, non-JSON, or structurally incomplete resolution is
        # unreadable rather than absent, so it arrives as JSONDecodeError /
        # OSError / KeyError instead of GateResolutionError.  "We could not
        # read the frozen record" is a refusal like any other; raising would
        # leave the verdict to whatever each caller does with an exception.
        return False, [
            f"stage {stage}'s frozen gate resolution could not be read "
            f"({type(exc).__name__}: {exc}); re-resolve the gate rather than trusting it"
        ]

    rail_failures = _recovery_reward_rail(curriculum, stage_results, stage=stage)
    failures = [*result.failures, *rail_failures]
    return bool(result.passed and not rail_failures), failures


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
    stage_dir: "str | Path | None" = None,
    recovery_successes_by_seed: "Mapping[int, bool] | None" = None,
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
        stage_dir: The stage's own directory, required by
            ``recovery_quality/v1`` and ignored by every other kind: it holds
            the frozen ``gate_resolution.json`` and the task fingerprint that
            resolution is checked against.  Omitting it does not soften the
            recovery gate — it refuses.
        recovery_successes_by_seed: The pushed panel's per-episode successes
            keyed by panel seed (``RecoveryPanelEvidence.successes_by_seed()``),
            required by ``recovery_quality/v1`` and ignored by every other
            kind.  The seeds must be the ones the frozen null manifest was
            measured on; the resolver refuses any other pairing.

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
    from environments.shared.curriculum.recovery_gate import RECOVERY_GATE_KIND
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
    if gate_kind == RECOVERY_GATE_KIND:
        # The recovery verdict comes ONLY from the gate resolver
        # (evaluate_recovery_gate_from_resolution): frozen thresholds, frozen
        # null pairings.  Falling through to the reward gate would certify a
        # pushed stage on return alone, so every input the resolver needs and
        # this call did not get is a refusal inside _recovery_stage_gate.
        return _recovery_stage_gate(
            curriculum,
            stage_results,
            stage=stage,
            stage_dir=stage_dir,
            recovery_successes_by_seed=recovery_successes_by_seed,
        )
    if gate_kind == STANCE_GATE_KIND:
        return _stance_stage_gate(gate_kind, stance_report, stage)
    return _reward_and_length_stage_gate(curriculum, stage_results)
