"""Curriculum-gate evaluation.

Two entry points, for two different questions:

* :func:`evaluate_recorded_gate` — "does this *history* of evaluations show a
  pass?", used by the Drive summary notebook to describe finished runs.
* :func:`evaluate_stage_gate` — "may this completed stage advance, and may its
  artifacts claim it passed?", used by the training notebook (through
  ``generate_stage_artifacts``) and the gate backfill tool.

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

``task_success/v1`` (the hunting deliverable, plan §4.4) is judged here from
the stage directory's ``evaluation_selected.csv`` — the selected checkpoint's
per-episode task successes, hash-bound to the handoff pair
``select_handoff_checkpoint`` picks now — through
:func:`~environments.shared.curriculum.task_success_gate.evaluate_task_success_gate`
(:func:`task_success_statistics` reads and binds the evidence).  A rounded
mean cannot recover ``k/n``, so the CSV is the ONLY input the kind accepts:
an absent file, an unbound one, or one for another checkpoint is a refusal.
The dispatch is closed at the end: a kind registered in ``GATE_KINDS`` with
no arm here is refused by name, never routed to the reward conjunction.
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..config import STAGE_CONFIG_FILENAME
from ..curriculum.manager import DEFAULT_MIN_EVAL_EPISODES
from ..file_io import read_json_object


def evaluate_recorded_gate(
    curriculum: dict[str, Any],
    evaluations: list[dict[str, Any]],
) -> bool | None:
    """Evaluate a curriculum gate only when every enabled metric is recorded.

    Evaluations must be chronological. ``None`` means the available records
    cannot prove either a pass or a failure—for example, when a velocity or
    success-rate gate is enabled but that metric was not saved.

    A ``task_success/v1`` history is judged on each evaluation's
    ``success_count`` / ``n_success_samples`` (the keys
    ``CurriculumManager.record_eval`` stores) through the same exact
    binomial bound the gate certifies with; a history that records no
    count — ``evaluations.npz`` carries none — is incomplete (``None``),
    never a pass on the reward rail the hunting statue clears.
    """
    from environments.shared.curriculum.gait_gate import GAIT_GATE_KIND
    from environments.shared.curriculum.task_success_gate import TASK_SUCCESS_GATE_KIND

    if curriculum.get("gate_kind") == GAIT_GATE_KIND:
        # Aggregate eval history has neither selected-pair binding nor the
        # physics-substep traces needed to reconstruct physical gait.
        return None
    if curriculum.get("gate_kind") == TASK_SUCCESS_GATE_KIND:
        return _recorded_task_success_gate(curriculum, evaluations)
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

    min_eval_episodes = int(curriculum.get("min_eval_episodes", DEFAULT_MIN_EVAL_EPISODES))
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


def _recorded_task_success_gate(
    curriculum: Mapping[str, Any],
    evaluations: list[dict[str, Any]],
) -> bool | None:
    """The ``task_success/v1`` reading of a recorded evaluation history.

    Each evaluation needs ``success_count`` and ``n_success_samples``; the
    bound over them must clear ``min_success_lcb`` at ``n_success_samples
    >= min_eval_episodes``, together with the reward rail and the optional
    length floor over the same evaluation, ``required_consecutive`` times
    in a row.  An evaluation missing any of those is incomplete and resets
    the streak; a history that never proves a pass reads ``None`` when any
    evaluation was incomplete and ``False`` otherwise.  An undeclared bar
    or panel size is ``None``: nothing can be proven against a gate nobody
    stated.
    """
    from environments.shared.curriculum.task_success_gate import (
        TaskSuccessGateThresholds,
        evaluate_task_success_gate,
    )

    try:
        thresholds = TaskSuccessGateThresholds.from_curriculum(curriculum)
    except ValueError:
        return None
    if not evaluations:
        return None
    required_consecutive = int(curriculum.get("required_consecutive", 3))
    consecutive = 0
    incomplete = False
    for evaluation in evaluations:
        count = evaluation.get("success_count")
        n_samples = evaluation.get("n_success_samples")
        reward = evaluation.get("mean_reward") if thresholds.min_avg_reward is not None else 0.0
        length = evaluation.get("mean_episode_length") if thresholds.min_avg_episode_length is not None else 0.0
        if count is None or n_samples is None or reward is None or length is None:
            incomplete = True
            consecutive = 0
            continue
        k, n = int(count), int(n_samples)
        if n < 0 or k < 0 or k > n:
            incomplete = True
            consecutive = 0
            continue
        result = evaluate_task_success_gate([True] * k + [False] * (n - k), thresholds)
        passes = (
            result.passed
            and (thresholds.min_avg_reward is None or float(reward) >= thresholds.min_avg_reward)
            and (thresholds.min_avg_episode_length is None or float(length) >= thresholds.min_avg_episode_length)
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
    gate_schema.finite_gate_metric`, the one rule every gate path shares, so
    no two paths can drift on what counts as measured.
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
_TASK_FINGERPRINT_ARTIFACTS = ("task_fingerprint.json", STAGE_CONFIG_FILENAME)

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
        # A truncated or hand-edited artifact proves nothing (None).  Try the
        # next one; refusing is what happens if neither answers.
        record: Any = read_json_object(stage_dir / name)
        if name == STAGE_CONFIG_FILENAME and record is not None:
            record = record.get("task_fingerprint")
        if isinstance(record, Mapping):
            recorded = record.get("task_sha256")
            if isinstance(recorded, str) and recorded:
                return recorded
    return None


def _same_threshold(declared: Any, frozen: Any) -> bool:
    """Whether a declared threshold and a frozen one state the same criterion.

    The comparison lives in ``curriculum.gate_schema.same_threshold`` since
    the gate-configuration digest (decision D-A22) needs it too; this is the
    same function, reached lazily for the import-cycle reason
    :func:`evaluate_stage_gate` documents.
    """
    from ..curriculum.gate_schema import same_threshold

    return same_threshold(declared, frozen)


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


def _reward_rail(
    curriculum: Mapping[str, Any],
    stage_results: Mapping[str, Any],
    *,
    stage: int | str,
    kind_label: str,
) -> list[str]:
    """Enforce the optional reward RAIL a composite-gate config may declare.

    ``recovery_quality/v1`` and ``task_success/v1`` may carry
    ``min_avg_reward`` in the same role ``stance_quality/v1`` gives it — a
    rail well below the null, not the gate — but neither the frozen recovery
    capability spec nor the task-success bound records a rail, so their own
    evaluators cannot enforce one.  A declared criterion nobody evaluates is
    the half-enforced gate this module exists to prevent, so it is checked
    here as an additional *conjunct*: it can only refuse, never advance
    anything the kind's own verdict did not already pass.  *kind_label*
    names the rail in the failure text (``recovery rail`` / ``task_success
    rail``).
    """
    target = curriculum.get("min_avg_reward")
    if target is None:
        return []
    reward = _gate_metric(stage_results, "best_model_reward", "best_eval_reward", "mean_reward")
    if reward is None:
        return [
            f"stage {stage} declares min_avg_reward {float(target):.2f} as a {kind_label} rail, "
            "but no reward measurement is available to check it"
        ]
    if reward < float(target):
        return [f"stage {stage} best model reward {reward:.2f} < {kind_label} rail {float(target):.2f}"]
    return []


def _recovery_reward_rail(
    curriculum: Mapping[str, Any],
    stage_results: Mapping[str, Any],
    *,
    stage: int | str,
) -> list[str]:
    """The recovery spelling of :func:`_reward_rail` (kept for its callers and tests)."""
    return _reward_rail(curriculum, stage_results, stage=stage, kind_label="recovery")


def task_success_statistics(stage_dir: "str | Path") -> "tuple[dict[str, Any] | None, list[str]]":
    """The selected checkpoint's task-success count, bound to the handoff pair.

    Reads ``<stage_dir>/evaluation_selected.csv`` — the per-episode evidence
    the notebook and the trainers' post-training panel write (and the JAX
    saver D-D17 retired wrote) for the SELECTED checkpoint — and binds it to the handoff pair
    ``select_handoff_checkpoint`` picks NOW through the rows'
    ``checkpoint_sha256`` column — and, when the rows record one, their
    ``normalization_sha256`` against the pair's ``_vecnorm.pkl``: an SB3
    handoff is the pair, and a panel rolled under other observation
    statistics is a panel of a different policy (the backfill tool and
    publication refuse the same file).  Returns ``(stats, failures)``:
    *stats* carries ``best_model_success_count``, ``best_model_n_episodes``,
    ``best_model_success_lcb`` (exact one-sided 95% Clopper-Pearson), the
    bound ``checkpoint_sha256`` and — when the rows carry the columns —
    ``best_model_reward`` / ``best_model_length``, the same panel's means
    the rail and the length floor are judged on; or is ``None`` with
    *failures* naming what is missing.  Every missing input is a refusal:
    a rounded mean cannot recover ``k/n``, and evidence that cannot be
    shown to describe the handoff is evidence for some other policy.
    """
    from environments.shared.curriculum.checkpoints import select_handoff_checkpoint
    from environments.shared.curriculum.recovery_gate import binomial_lcb
    from environments.shared.curriculum.task_success_gate import read_task_successes
    from environments.shared.result_bundle.hashing import sha256_file

    root = Path(stage_dir)
    handoff = select_handoff_checkpoint(root / "models")
    if handoff is None:
        return None, [
            f"{root / 'models'} has no complete handoff pair (a checkpoint with its matched _vecnorm.pkl), "
            "so there is no selected checkpoint whose task_success evidence could be judged"
        ]
    handoff_name, handoff_stem, handoff_vecnorm = handoff
    handoff_zip = Path(handoff_stem + ".zip")
    path = root / "evaluation_selected.csv"
    if not path.is_file():
        return None, [
            f"{path} is absent: the selected checkpoint's per-episode task_success evidence is the only "
            "input task_success/v1 accepts (a rounded mean cannot recover k/n)"
        ]
    try:
        evidence = read_task_successes(path)
    except (OSError, ValueError) as exc:
        return None, [f"{path} cannot be read as task_success evidence: {exc}"]
    if evidence.checkpoint_sha256 is None:
        return None, [
            f"{path} records no checkpoint_sha256 column, so it cannot be shown to describe the handoff {handoff_name}"
        ]
    handoff_digest = sha256_file(handoff_zip)
    if evidence.checkpoint_sha256 != handoff_digest:
        return None, [
            f"{path} describes checkpoint {evidence.checkpoint_sha256}, not the handoff {handoff_name} ({handoff_digest})"
        ]
    if evidence.normalization_sha256 is not None:
        vecnorm_digest = sha256_file(Path(handoff_vecnorm))
        if evidence.normalization_sha256 != vecnorm_digest:
            return None, [
                f"{path} ran under VecNormalize statistics {evidence.normalization_sha256}, not the handoff "
                f"{handoff_name}'s sidecar ({vecnorm_digest}); a panel rolled under other observation "
                "statistics is evidence for a different policy"
            ]
    n = len(evidence.successes)
    k = sum(1 for success in evidence.successes if success)
    stats: dict[str, Any] = {
        "best_model_success_count": k,
        "best_model_n_episodes": n,
        "best_model_success_lcb": binomial_lcb(k, n),
        "checkpoint_sha256": handoff_digest,
    }
    if evidence.mean_reward is not None:
        stats["best_model_reward"] = evidence.mean_reward
    if evidence.mean_length is not None:
        stats["best_model_length"] = evidence.mean_length
    return stats, []


def _task_success_stage_gate(
    curriculum: Mapping[str, Any],
    stage_results: Mapping[str, Any],
    *,
    stage: int | str,
    stage_dir: "str | Path | None",
) -> tuple[bool, list[str]]:
    """Judge ``task_success/v1`` from the stage directory's selected-checkpoint evidence.

    The verdict is :func:`~environments.shared.curriculum.task_success_gate.evaluate_task_success_gate`
    over the per-episode successes :func:`task_success_statistics` reads and
    hash-binds, then the reward rail and the optional length floor as
    conjuncts over the SAME panel's mean reward / length (the CSV's own
    rows; *stage_results* is consulted only when the file carries no such
    column).  Judging the rail on *stage_results* alone would read the
    ``best_eval_reward`` of a results dict built from disk — the argmax
    EvalCallback panel of the mean-reward best_model — beside a bound from
    the handoff pair's panel,
    and publication (which rails on the CSV) would disagree with the
    verdict on the same directory.  Every input this call lacks is a
    refusal naming it — no stage directory, no evidence file, evidence for
    another checkpoint, an undeclared bar — never a fall-through to the
    reward arm, which the ~600 hunting statue clears.
    """
    from environments.shared.curriculum.task_success_gate import (
        TaskSuccessGateThresholds,
        evaluate_task_success_gate,
        read_task_successes,
    )

    if stage_dir is None:
        return False, [
            f"stage {stage} declares task_success/v1, which is judged from the stage directory's "
            "evaluation_selected.csv; no stage_dir was given"
        ]
    try:
        thresholds = TaskSuccessGateThresholds.from_curriculum(curriculum)
    except ValueError as exc:
        return False, [f"stage {stage} declares task_success/v1 without a judgeable gate: {exc}"]
    stats, failures = task_success_statistics(Path(stage_dir))
    if stats is None:
        return False, [f"stage {stage} task_success/v1 evidence: {failure}" for failure in failures]
    # The statistics above already bound the file to the handoff; the
    # verdict itself comes from the shared evaluator over the same rows.
    evidence = read_task_successes(Path(stage_dir) / "evaluation_selected.csv")
    result = evaluate_task_success_gate(evidence.successes, thresholds)
    failures = [f"stage {stage} {failure}" for failure in result.failures]
    # The panel's own aggregates shadow whatever the caller passed.
    judged: dict[str, Any] = dict(stage_results)
    judged.update({key: stats[key] for key in ("best_model_reward", "best_model_length") if key in stats})
    failures.extend(_reward_rail(curriculum, judged, stage=stage, kind_label="task_success"))
    if thresholds.min_avg_episode_length is not None:
        length = _gate_metric(judged, "best_model_length", "best_eval_length", "mean_episode_length")
        if length is None:
            failures.append(
                f"stage {stage} declares min_avg_episode_length {thresholds.min_avg_episode_length:.1f}, "
                "but no episode-length measurement is available to check it"
            )
        elif length < thresholds.min_avg_episode_length:
            failures.append(
                f"stage {stage} best model episode length {length:.1f} < {thresholds.min_avg_episode_length:.1f}"
            )
    return not failures, failures


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


#: Replayed metrics agree with the stored ones within one storage rounding
#: quantum (metrics are stored to six decimals, so a last-bit difference in
#: an unrounded value can move a stored one by 1e-6) plus a relative epsilon
#: for unrounded values. Integers, booleans, strings and structure are exact.
_REPLAY_ABS_TOLERANCE = 2e-6
_REPLAY_REL_TOLERANCE = 1e-9
#: A trace's stored reset state, and its first sample, must match a fresh
#: reset of its seed this closely (metres, radians, metres per second); seeds
#: differ by the reset noise, orders of magnitude more.
_RESET_TOLERANCE = 1e-9


def _replay_mismatch(stored: Any, measured: Any, path: str = "metrics") -> str | None:
    """The first path where a stored metric disagrees with its replay, or ``None``."""
    if isinstance(measured, Mapping):
        if not isinstance(stored, Mapping) or set(stored) != set(measured):
            return path
        for key in sorted(measured):
            mismatch = _replay_mismatch(stored[key], measured[key], f"{path}.{key}")
            if mismatch is not None:
                return mismatch
        return None
    if isinstance(measured, list):
        if not isinstance(stored, list) or len(stored) != len(measured):
            return path
        for index, (left, right) in enumerate(zip(stored, measured, strict=True)):
            mismatch = _replay_mismatch(left, right, f"{path}[{index}]")
            if mismatch is not None:
                return mismatch
        return None
    if isinstance(measured, float) and not isinstance(measured, bool):
        if isinstance(stored, bool) or not isinstance(stored, (int, float)):
            return path
        if not math.isclose(stored, measured, rel_tol=_REPLAY_REL_TOLERANCE, abs_tol=_REPLAY_ABS_TOLERANCE):
            return path
        return None
    return None if type(stored) is type(measured) and stored == measured else path


def _trace_content_sha256(trace: Mapping[str, Any]) -> str:
    """Digest of a trace's arrays (names, dtypes, shapes and bytes), independent of the archive's bytes."""
    import hashlib

    import numpy as np

    digest = hashlib.sha256()
    for key in sorted(trace):
        array = np.ascontiguousarray(trace[key])
        digest.update(f"{key}|{array.dtype.str}|{array.shape}|".encode())
        digest.update(array.tobytes())
    return f"sha256:{digest.hexdigest()}"


def _replay_gait_traces(
    root: Path,
    report: Mapping[str, Any],
    protocol_payload: Mapping[str, Any],
    episodes: list[Mapping[str, Any]],
    foot_names: list[str],
) -> list[dict[str, Any]]:
    """Require reproducible raw physics evidence tied to its seeds, its authored scales, and timing.

    Each trace must start from a fresh reset of its own seed (its stored
    reset state and first sample are compared with one), no two traces may
    hold the same arrays, and its metrics are recomputed. Returns the panel's
    episodes with every metric replaced by its replayed value: the caller
    judges those, never the stored numbers, which only have to agree within
    ``_REPLAY_ABS_TOLERANCE`` (portable across BLAS kernels and machines).
    """
    import numpy as np

    from environments.shared.gait.metrics import episode_gait_metrics
    from environments.shared.gait.morphology import GaitMorphology
    from environments.shared.gait.report import RESET_STATE_FIELDS, TRACE_DIRECTORY, json_safe, reset_state_digest
    from environments.shared.gait.types import GaitProtocol
    from environments.shared.plant_contract import current_plant_identity, validate_environment_plant
    from environments.shared.record_fields import is_sha256_digest
    from environments.shared.result_bundle.hashing import canonical_json_sha256, sha256_file
    from environments.shared.species_registry import get_species_config
    from environments.shared.task_fingerprint import constructor_task_differences, stage_task_fingerprint

    detector = protocol_payload.get("detector")
    panel = protocol_payload.get("panel")
    sampling = protocol_payload.get("sampling")
    morphology = report.get("morphology")
    if not all(isinstance(block, Mapping) for block in (detector, panel, sampling, morphology)):
        raise ValueError("gait replay requires detector, panel, sampling, and morphology records")
    assert isinstance(detector, Mapping) and isinstance(panel, Mapping)
    assert isinstance(sampling, Mapping) and isinstance(morphology, Mapping)
    settings = GaitProtocol(
        **{key: value for key, value in detector.items() if key not in {"schema", "default_status"}}
    )
    horizon = panel.get("horizon_control_steps")
    settle = panel.get("settle_s")
    direction = panel.get("direction_xy")
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        raise ValueError("gait replay has no positive integer horizon")
    if isinstance(settle, bool) or not isinstance(settle, (int, float)) or not math.isfinite(settle) or settle < 0:
        raise ValueError("gait replay has no finite settle time")
    if not isinstance(direction, list) or len(direction) != 2:
        raise ValueError("gait replay has no two-dimensional direction")
    traces = report.get("traces")
    if not isinstance(traces, list) or len(traces) != len(episodes):
        raise ValueError("gait report requires one raw NPZ trace per declared episode")
    expected_paths = {f"{TRACE_DIRECTORY}/episode_{index:04d}.npz" for index in range(len(episodes))}
    actual_paths = {path.relative_to(root).as_posix() for path in (root / TRACE_DIRECTORY).glob("*.npz")}
    if actual_paths != expected_paths:
        raise ValueError("gait raw trace files do not match the declared fixed panel")
    reset_hashes = [episode.get("reset_state_sha256") for episode in episodes]
    if any(not is_sha256_digest(digest) for digest in reset_hashes) or len(set(reset_hashes)) != len(episodes):
        raise ValueError("gait panel reset states are missing or duplicated")
    saved = read_json_object(root / "stage_config.json")
    env_kwargs = None if saved is None else saved.get("reward_weights", saved.get("env_kwargs", {}))
    if not isinstance(env_kwargs, Mapping):
        raise ValueError("gait replay requires the recorded environment constructor")
    env = get_species_config(report["species"]).env_class(**env_kwargs)
    replayed: list[dict[str, Any]] = []
    contents: set[str] = set()
    try:
        plant = current_plant_identity(report["species"])
        validate_environment_plant(env, plant, artifact="gait raw evidence replay")
        if report.get("plant_identity") != plant.to_dict():
            raise ValueError("gait report plant identity does not match the current compiled plant")
        # The recorded task must be what today's code derives from its own
        # env section on the current plant, and the recorded constructor
        # must build it.  The constructor is compared key by key, never
        # re-hashed: save_stage_config writes every default out, and a
        # re-derived compsognathus task then keeps the quiet push keys its
        # trainer carved out.
        recorded_task = report.get("task_fingerprint")
        recorded_env = recorded_task.get("env") if isinstance(recorded_task, Mapping) else None
        if not isinstance(recorded_env, Mapping):
            raise ValueError("gait report records no task fingerprint environment")
        try:
            task = stage_task_fingerprint(
                report["species"], report["stage"], env_kwargs=dict(recorded_env), plant_identity=plant
            )
            differing = constructor_task_differences(report["species"], env_kwargs, task)
        except RuntimeError as exc:
            raise ValueError(f"gait task cannot be re-derived: {exc}") from exc
        if recorded_task != task or report.get("task_sha256") != task["task_sha256"]:
            raise ValueError("gait report task fingerprint does not re-derive on the current code and plant")
        if differing:
            raise ValueError(
                "the recorded environment constructor does not build the gait report's task: " + ", ".join(differing)
            )
        actual_morphology = GaitMorphology.from_env(env, report["species"])
        if canonical_json_sha256(actual_morphology.describe()) != canonical_json_sha256(morphology):
            raise ValueError("gait morphology scales or geometry differ from the current recorded plant")
        physical_dt = float(env.model.opt.timestep)
        control_dt = float(env.dt)
        if sampling.get("physics_dt_s") != physical_dt or sampling.get("control_dt_s") != control_dt:
            raise ValueError("gait sampling timesteps differ from the current recorded plant")
        root_address = actual_morphology.root_qpos_address
        for index, (entry, episode) in enumerate(zip(traces, episodes, strict=True)):
            relative = f"{TRACE_DIRECTORY}/episode_{index:04d}.npz"
            if not isinstance(entry, Mapping) or entry.get("path") != relative:
                raise ValueError(f"gait trace {index} has an unexpected local path")
            path = root / relative
            path.resolve().relative_to(root.resolve())
            if not path.is_file() or entry.get("sha256") != sha256_file(path):
                raise ValueError(f"gait trace {index} is missing or its digest changed")
            with np.load(path, allow_pickle=False) as archive:
                trace = {key: archive[key] for key in archive.files}
            content = _trace_content_sha256(trace)
            if content in contents:
                raise ValueError(f"gait trace {index} duplicates another episode's trace")
            contents.add(content)
            # The trace belongs to its seed: a fresh reset of that seed gives
            # its stored reset state, which its digest and first sample match.
            seed = episode.get("seed")
            if isinstance(seed, bool) or not isinstance(seed, int):
                raise ValueError(f"gait episode {index} has no integer seed")
            if any(key not in trace for key in RESET_STATE_FIELDS) or "physics_diverged" not in trace:
                raise ValueError(f"gait trace {index} lacks its reset state or divergence record")
            stored_reset = [np.asarray(trace[key], dtype=np.float64) for key in RESET_STATE_FIELDS]
            if reset_state_digest(*stored_reset) != episode["reset_state_sha256"]:
                raise ValueError(f"gait episode {index} reset digest does not describe its trace's reset state")
            env.reset(seed=seed)
            data = env.unwrapped.data
            fresh = (data.qpos, data.qvel, data.mocap_pos)
            if any(
                np.shape(state) != np.shape(expected)
                or not np.allclose(state, expected, rtol=0.0, atol=_RESET_TOLERANCE)
                for state, expected in zip(stored_reset, fresh, strict=True)
            ):
                raise ValueError(f"gait trace {index} does not start from a fresh reset of seed {seed}")
            first = {
                "root_position_m": stored_reset[0][root_address : root_address + 3],
                "root_quat_wxyz": stored_reset[0][root_address + 3 : root_address + 7],
                "foot_position_m": np.asarray(data.site_xpos[list(actual_morphology.foot_site_ids)]),
            }
            for key, expected in first.items():
                sample = np.asarray(trace.get(key, ()), dtype=np.float64)
                if (
                    sample.ndim < 1
                    or len(sample) < 1
                    or sample[0].shape != expected.shape
                    or not np.allclose(sample[0], expected, rtol=0.0, atol=_RESET_TOLERANCE)
                ):
                    raise ValueError(f"gait trace {index} first sample is not seed {seed}'s reset state")
            diverged = trace["physics_diverged"]
            if (
                diverged.shape != ()
                or diverged.dtype != np.bool_
                or episode.get("physics_diverged") is not bool(diverged)
            ):
                raise ValueError(f"gait episode {index} divergence record disagrees with its trace")
            time = np.asarray(trace["time_s"], dtype=float)
            length = episode.get("length")
            if isinstance(length, bool) or not isinstance(length, int) or not 1 <= length <= horizon:
                raise ValueError(f"gait episode {index} has an invalid control-step length")
            span = float(time[-1] - time[0]) if time.ndim == 1 and len(time) else math.nan
            expected_span = length * control_dt
            if (
                time.ndim != 1
                or len(time) < (1 if bool(diverged) else 2)
                or not np.all(np.isfinite(time))
                or not np.allclose(np.diff(time), physical_dt, rtol=1e-6, atol=1e-9)
                or not (
                    span <= expected_span + 1e-9
                    if bool(diverged)
                    else math.isclose(span, expected_span, rel_tol=1e-6, abs_tol=1e-9)
                )
            ):
                raise ValueError(f"gait episode {index} raw trace duration/timesteps do not match its recorded length")
            if not isinstance(episode.get("terminated"), bool) or not isinstance(episode.get("truncated"), bool):
                raise ValueError(f"gait episode {index} has no termination evidence")
            completed = length == horizon and not episode["terminated"] and not bool(diverged)
            if episode.get("completed_horizon") is not completed:
                raise ValueError(f"gait episode {index} completion contradicts its length/termination")
            measured = episode_gait_metrics(
                trace,
                body_weight_n=actual_morphology.body_weight_n,
                leg_length_m=actual_morphology.leg_length_m,
                foot_names=tuple(foot_names),
                protocol=settings,
                settle_s=float(settle),
                direction_xy=(float(direction[0]), float(direction[1])),
            )
            measured = json_safe(measured)
            mismatch = _replay_mismatch({key: episode.get(key) for key in measured}, measured)
            if mismatch is not None:
                raise ValueError(
                    f"gait episode {index} metrics do not reproduce from its raw physics trace ({mismatch})"
                )
            replayed.append({**episode, **measured})
    finally:
        env.close()
    return replayed


def _read_gait_panel_episodes(path: Path, bindings: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Decode strict per-row metric objects; malformed rows raise to the judge."""
    episodes: list[Mapping[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            if any(row.get(key) != expected for key, expected in bindings.items()):
                raise ValueError(f"gait_panel.csv row {index} describes a different checkpoint/task/protocol")
            episode = json.loads(row.get("metrics_json", ""))
            if not isinstance(episode, Mapping):
                raise ValueError(f"gait_panel.csv row {index} has no episode metrics object")
            episodes.append(episode)
    return episodes


def gait_statistics(stage_dir: str | Path, curriculum: Mapping[str, Any]) -> tuple[dict[str, Any] | None, list[str]]:
    """Read and rejudge the selected handoff's hash-bound physical gait panel.

    A recorded ``passed`` flag is never evidence. The hashed CSV's complete
    episode payloads must agree with the report, every raw trace must start
    from a fresh reset of its own seed, and the shared classifier judges the
    metrics replayed from those traces (not the stored ones) and the
    confidence bound under today's declared thresholds. Old checkpoints,
    tasks, protocols, duplicated seeds and duplicated traces refuse.
    """
    from environments.shared.curriculum.checkpoints import select_handoff_checkpoint
    from environments.shared.curriculum.gait_gate import GAIT_GATE_KIND, GaitGateThresholds, evaluate_gait_gate
    from environments.shared.curriculum.gate_schema import validate_gate_config
    from environments.shared.gait.identity import validate_measurement_protocol_identity
    from environments.shared.gait.report import REPORT_SCHEMA
    from environments.shared.gait.seeds import checkpoint_seed_provenance
    from environments.shared.result_bundle.hashing import canonical_json_sha256, sha256_file
    from environments.shared.task_fingerprint import read_checkpoint_task_fingerprint

    root = Path(stage_dir)
    try:
        if validate_gate_config("gait-panel", curriculum) != GAIT_GATE_KIND:
            return None, [f"gait evidence can certify only an explicit {GAIT_GATE_KIND} gate"]
        thresholds = GaitGateThresholds.from_curriculum(curriculum)
        report = read_json_object(root / "gait_report.json")
        if report is None or report.get("schema") != REPORT_SCHEMA:
            return None, ["missing or unsupported gait_report.json"]
        if (
            report.get("status") != "complete"
            or any(
                report.get(key) is not True for key in ("plant_validated", "task_validated", "certification_eligible")
            )
            or report.get("report_only") is not False
        ):
            return None, ["gait report is incomplete, unvalidated, or report-only and cannot certify advancement"]
        handoff = select_handoff_checkpoint(root / "models")
        if handoff is None:
            return None, ["no selected checkpoint with matching normalization statistics"]
        _, model_path, vecnorm_path = handoff
        current = {
            "checkpoint_sha256": sha256_file(f"{model_path}.zip"),
            "normalization_sha256": sha256_file(vecnorm_path),
            "task_sha256": _current_task_sha256(root),
            "measurement_protocol_sha256": thresholds.measurement_protocol_sha256,
        }
        if current["task_sha256"] is None:
            return None, ["stage task fingerprint is missing"]
        for key, expected in current.items():
            if report.get(key) != expected:
                return None, [f"gait report {key} does not match the selected handoff/current task/protocol"]
        stage = report.get("stage")
        if isinstance(stage, bool) or not isinstance(stage, (int, str)):
            return None, ["gait report has no stage reference"]
        expected_seed_provenance = checkpoint_seed_provenance(
            f"{model_path}.zip",
            seed_start=thresholds.gait_panel_seed_start,
            episodes=thresholds.min_eval_episodes,
            stage=stage,
            species=report.get("species") if isinstance(report.get("species"), str) else None,
        )
        if report.get("seed_provenance") != expected_seed_provenance:
            return None, ["gait seed provenance is missing or does not match the selected checkpoint"]
        if read_checkpoint_task_fingerprint(f"{model_path}.zip") != report.get("task_fingerprint"):
            return None, ["gait task fingerprint does not match the selected checkpoint's recorded task"]
        if report.get("gait_profile") != thresholds.gait_profile:
            return None, ["gait report gait_profile does not match the declared profile"]
        protocol = report.get("measurement_protocol")
        if (
            not isinstance(protocol, Mapping)
            or canonical_json_sha256(protocol) != thresholds.measurement_protocol_sha256
        ):
            return None, ["gait report measurement protocol payload does not match its declared digest"]
        species = report.get("species")
        if not isinstance(species, str) or not validate_measurement_protocol_identity(dict(protocol), species):
            return None, ["gait measurement implementation or foot registry is stale or unknown"]
        declared_panel = protocol.get("panel")
        if not isinstance(declared_panel, Mapping) or (
            declared_panel.get("episodes") != thresholds.min_eval_episodes
            or declared_panel.get("seed_start") != thresholds.gait_panel_seed_start
        ):
            return None, ["gait measurement protocol does not describe the declared fixed panel"]
        panel = report.get("panel_csv")
        if not isinstance(panel, Mapping) or panel.get("path") != "gait_panel.csv":
            return None, ["gait report has no local gait_panel.csv evidence binding"]
        path = root / "gait_panel.csv"
        if not path.is_file() or panel.get("sha256") != sha256_file(path):
            return None, ["gait_panel.csv is missing or its digest changed"]
        episodes = _read_gait_panel_episodes(path, current)
        if not episodes or episodes != report.get("episodes"):
            return None, ["gait report episodes do not agree with the hashed CSV panel"]
        seeds = [episode.get("seed") for episode in episodes]
        if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds) or len(set(seeds)) != len(seeds):
            return None, ["gait panel episode seeds are missing or duplicated"]
        if seeds != list(
            range(thresholds.gait_panel_seed_start, thresholds.gait_panel_seed_start + thresholds.min_eval_episodes)
        ):
            return None, ["gait panel seeds do not match the fixed declared panel"]
        if report.get("seed_start") != thresholds.gait_panel_seed_start:
            return None, ["gait report seed_start does not match the fixed declared panel"]
        foot_names = report.get("foot_names")
        if not isinstance(foot_names, list) or any(not isinstance(name, str) for name in foot_names):
            return None, ["gait report foot registry is missing"]
        if foot_names != protocol.get("foot_names"):
            return None, ["gait report foot order differs from its measurement registry"]
        # Judged on the replayed metrics, never the stored numbers.
        replayed = _replay_gait_traces(root, report, protocol, episodes, foot_names)
        result = evaluate_gait_gate(replayed, thresholds, foot_names=foot_names)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        return None, [f"gait evidence could not be read: {type(exc).__name__}: {exc}"]
    return {
        **current,
        "selected_gait_success_count": result.success_count,
        "selected_gait_n_episodes": result.n_episodes,
        "selected_gait_success_lcb": result.success_lcb,
        "passed": result.passed,
        "failures": list(result.failures),
        "episode_failures": [list(failures) for failures in result.episode_failures],
    }, []


def gait_stage_verdict(
    stats: Mapping[str, Any] | None, failures: list[str], *, stage: int | str
) -> tuple[bool, list[str]]:
    """The stage verdict from one :func:`gait_statistics` reading (no second replay)."""
    if stats is None:
        return False, [f"stage {stage} gait evidence: {failure}" for failure in failures]
    return bool(stats["passed"]), [f"stage {stage} {failure}" for failure in stats["failures"]]


def _gait_stage_gate(
    curriculum: Mapping[str, Any], *, stage: int | str, stage_dir: str | Path | None
) -> tuple[bool, list[str]]:
    from environments.shared.curriculum.gait_gate import GAIT_GATE_KIND

    if stage_dir is None:
        return False, [f"stage {stage} {GAIT_GATE_KIND} requires its selected-handoff gait panel and stage_dir"]
    stats, failures = gait_statistics(stage_dir, curriculum)
    return gait_stage_verdict(stats, failures, stage=stage)


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
            ``recovery_quality/v1`` (it holds the frozen
            ``gate_resolution.json`` and the task fingerprint that resolution
            is checked against) and by ``task_success/v1`` (it holds the
            selected checkpoint's ``evaluation_selected.csv`` and the handoff
            pair it is bound to), ignored by every other kind.  Omitting it
            does not soften either gate — both refuse.
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
    from environments.shared.curriculum.gait_gate import GAIT_GATE_KIND
    from environments.shared.curriculum.gate_schema import GATE_KINDS
    from environments.shared.curriculum.recovery_gate import RECOVERY_GATE_KIND
    from environments.shared.curriculum.stance_gate import STANCE_GATE_KIND
    from environments.shared.curriculum.task_success_gate import TASK_SUCCESS_GATE_KIND

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
    if gate_kind == GAIT_GATE_KIND:
        return _gait_stage_gate(curriculum, stage=stage, stage_dir=stage_dir)
    if gate_kind == TASK_SUCCESS_GATE_KIND:
        # The hunting verdict comes ONLY from the selected checkpoint's
        # per-episode evidence (plan §4.4): the reward conjunction below
        # would certify a hunt on min_avg_reward, which is a collapse rail
        # the statue clears, so every missing input refuses inside the arm.
        return _task_success_stage_gate(curriculum, stage_results, stage=stage, stage_dir=stage_dir)
    if gate_kind == "reward_and_length/v1":
        return _reward_and_length_stage_gate(curriculum, stage_results)
    # A kind registered in GATE_KINDS with no arm above: the NEXT kind
    # (command_tracking/v1, Phase D) lands here first, and must fail closed
    # by construction rather than certify on the reward conjunction.
    return False, [
        f"stage {stage} declares gate_kind {gate_kind!r}, which is registered but has no evaluator in "
        "reporting.gates.evaluate_stage_gate; refusing rather than judging it on the reward conjunction"
    ]
