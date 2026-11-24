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

``stance_quality/v2`` (the floor-truth stance gate, decision D-D23) is
judged from the stage's v2 stance gate report, and unlike v1 the arm trusts
none of it: :func:`stance_v2_report_refusals` refuses a report not scored by
the v2 code (``scored_gate_kind``), scored under other thresholds or another
measurement definition, for another checkpoint than the handoff pair
``select_handoff_checkpoint`` picks now, or on another task than the one the
stage's own records say it ran (``task_sha256``: the ``[env]`` block and
plant), and the verdict is then RE-DERIVED from the report's per-episode
rows through
:func:`~environments.shared.curriculum.stance_gate_v2.evaluate_stance_v2_gate`
and must agree with the recorded one -- a report that cannot be re-derived
is refused, never judged a FAIL.
``stance_quality_heading/v1`` (the six-heading certificate, decision D-D32)
is judged from the stage's ``stance_heading_report.json`` the same way --
:func:`stance_heading_report_refusals` binds it like a v2 report and to its
protocol, grid, seed list, determinism, frozen normalization and gate
digest, and the verdict is re-derived from every (seed, heading) row through
:func:`~environments.shared.curriculum.stance_heading_gate.evaluate_stance_heading_gate`.
A v2 report, the report-only heading probe or no certificate at all is a
refusal: the arm never falls back to the yaw-zero panel.

``command_tracking/v1`` (decision D-D61) is judged from the stage's
``command_tracking_report.json`` the same way:
:func:`command_tracking_report_refusals` binds it to its schema, protocol,
gate digest, frozen normalization, the certification seed, the handoff pair
AND the stage's handoff choice (``models/handoff_choice.json``: the pair
chosen by the gated statistic on the selection panel), the task, horizon,
control step and command config the stage recorded, and the verdict is
re-derived from every episode record through
:func:`~environments.shared.curriculum.command_tracking_gate.evaluate_command_tracking_gate`.
A history (:func:`evaluate_recorded_gate`) always reads ``None`` for it.

The dispatch is closed at the end: a kind registered in ``GATE_KINDS`` with
no arm here is refused by name, never routed to the reward conjunction.  It
is closed at the start too: a curriculum block that declares a retired gate
kind or key (``gate_schema.RETIRED_GATE_KINDS`` and
``RETIRED_CURRICULUM_KEYS``: ``task_success/v1``, ``min_success_rate``,
``min_success_lcb`` and ``task_success_panel_episodes``, removed with the
hunt nodes by decision D-D59) is refused naming the decision, so a frozen
record of a hunt stage is never judged on the thresholds it still has.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..config import STAGE_CONFIG_FILENAME
from ..curriculum.manager import DEFAULT_MIN_EVAL_EPISODES
from ..file_io import read_json_object

if TYPE_CHECKING:
    from ..curriculum.command_tracking_gate import CommandTrackingResult
    from ..curriculum.stance_gate_v2 import StanceV2Result
    from ..curriculum.stance_heading_gate import StanceHeadingResult


def evaluate_recorded_gate(
    curriculum: dict[str, Any],
    evaluations: list[dict[str, Any]],
) -> bool | None:
    """Evaluate a curriculum gate only when every enabled metric is recorded.

    Evaluations must be chronological. ``None`` means the available records
    cannot prove either a pass or a failure—for example, when a velocity
    gate is enabled but that metric was not saved.

    A history whose curriculum declares a retired gate kind or key
    (``task_success/v1`` or ``min_success_rate``, among others; decision
    D-D59) is always ``None``: no reader evaluates the criterion it was
    judged on, and judging it on the thresholds it has left would prove a
    pass of a gate nobody declared.

    A ``stance_quality/v2`` history is read from each evaluation's
    ``stance_clean_count`` / ``n_stance_samples`` through the exact binomial
    bound the gate certifies with — and since
    the in-training manager refuses the kind and records neither today, it
    reads ``None``: never the generic arm below, which would judge a v2
    stage on the reward rail its statue clears.

    A ``stance_quality_heading/v1`` history is always ``None``: no recorded
    count can stand for every seed at every heading of the grid, so a
    history can prove neither a pass nor a failure of the certificate.

    A ``command_tracking/v1`` history (decision D-D61) is always ``None``
    too: its verdict is a per-episode classification of the stage's own
    command panel (seeds planned from the certification seed, each
    episode's floor-truth gait clause included) on its chosen handoff pair,
    and an in-training evaluation rolls other seeds on other checkpoints
    without the gait recorder, so no recorded scalar can stand for it.
    """
    from environments.shared.curriculum.command_tracking_gate import COMMAND_TRACKING_GATE_KIND
    from environments.shared.curriculum.gate_schema import retired_gate_declarations
    from environments.shared.curriculum.stance_gate_v2 import STANCE_GATE_V2_KIND
    from environments.shared.curriculum.stance_heading_gate import STANCE_HEADING_GATE_KIND

    if retired_gate_declarations(curriculum):
        return None
    if curriculum.get("gate_kind") == STANCE_GATE_V2_KIND:
        return _recorded_stance_v2_gate(curriculum, evaluations)
    if curriculum.get("gate_kind") == STANCE_HEADING_GATE_KIND:
        return None
    if curriculum.get("gate_kind") == COMMAND_TRACKING_GATE_KIND:
        return None
    criteria: list[tuple[str, float]] = []
    ceilings: list[tuple[str, float]] = []
    if curriculum.get("min_avg_reward") is not None:
        criteria.append(("mean_reward", float(curriculum["min_avg_reward"])))
    if curriculum.get("min_avg_episode_length") is not None:
        criteria.append(("mean_episode_length", float(curriculum["min_avg_episode_length"])))
    if float(curriculum.get("min_avg_forward_vel") or 0.0) > 0.0:
        criteria.append(("mean_forward_vel", float(curriculum["min_avg_forward_vel"])))
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


def _recorded_stance_v2_gate(
    curriculum: Mapping[str, Any],
    evaluations: list[dict[str, Any]],
) -> bool | None:
    """The ``stance_quality/v2`` reading of a recorded evaluation history.

    Each evaluation needs ``stance_clean_count`` and ``n_stance_samples``
    (the scalars an in-training screen would record); the binomial bound
    over them must clear ``min_clean_stance_lcb`` at ``n_stance_samples >=
    min_eval_episodes``, ``required_consecutive`` times in a row.  This is a
    reading of the RECORDED history only, so it is never a certificate: the
    v2 verdict is the post-stage report's, and the statue-relative criteria
    cannot be read from a history at all.  An evaluation missing a count is
    incomplete and resets the streak; a history that never proves a pass
    reads ``None`` when any evaluation was incomplete (every history today:
    the manager refuses the kind and records no count) and ``False``
    otherwise.  An undeclared or unreadable bar is ``None``.
    """
    from environments.shared.curriculum.recovery_gate import binomial_lcb
    from environments.shared.curriculum.stance_gate_v2 import StanceV2Thresholds

    try:
        thresholds = StanceV2Thresholds.from_curriculum(curriculum)
    except ValueError:
        return None
    if not evaluations:
        return None
    consecutive = 0
    incomplete = False
    for evaluation in evaluations:
        count = evaluation.get("stance_clean_count")
        n_samples = evaluation.get("n_stance_samples")
        if count is None or n_samples is None:
            incomplete = True
            consecutive = 0
            continue
        k, n = int(count), int(n_samples)
        if n <= 0 or k < 0 or k > n:
            incomplete = True
            consecutive = 0
            continue
        passes = n >= thresholds.min_eval_episodes and binomial_lcb(k, n) >= thresholds.min_clean_stance_lcb
        consecutive = consecutive + 1 if passes else 0
        if consecutive >= thresholds.required_consecutive:
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


def _same_reading(a: Any, b: Any) -> bool:
    """Equality that reads NaN and ``None`` (JSON's spelling of NaN) as the same unmeasured value."""
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        return set(a) == set(b) and all(_same_reading(a[key], b[key]) for key in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same_reading(x, y) for x, y in zip(a, b))
    unmeasured_a = a is None or (isinstance(a, float) and math.isnan(a))
    unmeasured_b = b is None or (isinstance(b, float) and math.isnan(b))
    if unmeasured_a or unmeasured_b:
        return unmeasured_a and unmeasured_b
    return bool(a == b)


def _measurement_refusals(report: Mapping[str, Any], *, stage: int | str) -> list[str]:
    """Every way *report*'s ``measurement`` block fails to be the one this checkout measures with.

    The manifest's constants must be exactly the library's
    (``gait.constants.measurement_constants``), its registry entry exactly
    the model's current registry entry (its env class's ``FOOT_SUPPORT``,
    ``gait.morphology.measurement_definition``), and its recorded sha256
    the digest of the block as recorded: a report measured under other
    definitions scores other keys under the same names, and changing any of
    them after adoption needs a new ``MEASUREMENT_VERSION``.  The
    model-read entries (``dt``, ``frame_skip``, ``body_weight_n``,
    ``feet``) are bound elsewhere: the step against the stage's recorded
    ``[env]`` block (:func:`_admit_stance_v2_report`), the rest through the
    plant hashes the task fingerprint carries.
    """
    from environments.shared.gait.constants import (
        MEASUREMENT_MANIFEST_SCHEMA,
        measurement_constants,
        measurement_sha256,
    )
    from environments.shared.gait.morphology import measurement_definition
    from environments.shared.model_id import ModelId

    manifest = report.get("measurement")
    if not isinstance(manifest, Mapping):
        return [f"stage {stage} stance report records no measurement manifest, so what its metrics mean is unknown"]
    refusals: list[str] = []
    if manifest.get("schema") != MEASUREMENT_MANIFEST_SCHEMA:
        refusals.append(
            f"stage {stage} stance report's measurement manifest is {manifest.get('schema')!r}, not "
            f"{MEASUREMENT_MANIFEST_SCHEMA!r}"
        )
    if not _same_reading(manifest.get("constants"), measurement_constants()):
        refusals.append(
            f"stage {stage} stance report was measured under constants {manifest.get('constants')!r}, not this "
            f"checkout's {measurement_constants()!r}; a floor-truth key means what its constants make it mean"
        )
    try:
        recorded_digest = measurement_sha256(dict(manifest))
    except (TypeError, ValueError) as exc:
        recorded_digest = f"unhashable ({exc})"
    if report.get("measurement_sha256") != recorded_digest:
        refusals.append(
            f"stage {stage} stance report's measurement_sha256 {report.get('measurement_sha256')!r} is not the "
            f"digest of its own manifest ({recorded_digest}); the block was edited after it was measured"
        )
    # The report and its manifest each name the model whole (decision D-D33);
    # either one naming it by half is refused, never matched on the species.
    try:
        model_id = ModelId.from_record(report)
    except ValueError as exc:
        return [*refusals, f"stage {stage} stance report does not name its model: {exc}"]
    try:
        measured_on = ModelId.from_record(manifest)
    except ValueError as exc:
        return [*refusals, f"stage {stage} stance report's measurement manifest does not name its model: {exc}"]
    if measured_on != model_id:
        refusals.append(f"stage {stage} stance report was measured on {measured_on}, not {model_id}")
    expected = measurement_definition(model_id)["registry"]
    if not _same_reading(manifest.get("registry"), expected):
        refusals.append(
            f"stage {stage} stance report used the support registry entry {manifest.get('registry')!r}, not the "
            f"current {expected!r} for {model_id}"
        )
    return refusals


def _recorded_stage_env(stage_dir: Path) -> "tuple[dict[str, Any] | None, str | None]":
    """The ``[env]`` block *stage_dir*'s ``stage_config.json`` records, else ``None`` and why it records none.

    ``reward_weights`` is the name :func:`~environments.shared.config.save_stage_config`
    writes (the constructor defaults overlaid with the stage's kwargs);
    ``env_kwargs`` is the in-memory name the same dict carries, which a
    hand-built record may use instead.  A record carrying both must agree on
    every key they share -- otherwise which block the stage ran is unknown --
    and is read as their union, so a key one block omits (a horizon among
    them) is never read as absent while the other records it.
    """
    from ..config import read_recorded_stage_config

    record = read_recorded_stage_config(stage_dir)
    if record is None:
        return None, f"{stage_dir / STAGE_CONFIG_FILENAME} is absent or unreadable"
    blocks = [record[name] for name in ("reward_weights", "env_kwargs") if isinstance(record.get(name), Mapping)]
    if not blocks:
        return None, f"{stage_dir / STAGE_CONFIG_FILENAME} records no [env] block (reward_weights)"
    if len(blocks) == 2:
        differing = sorted(key for key in set(blocks[0]) & set(blocks[1]) if blocks[0][key] != blocks[1][key])
        if differing:
            return None, (
                f"{stage_dir / STAGE_CONFIG_FILENAME} records reward_weights and env_kwargs that disagree on "
                f"{', '.join(differing)}"
            )
    return {key: value for block in reversed(blocks) for key, value in block.items()}, None


def stance_v2_report_refusals(
    curriculum: Mapping[str, Any],
    stance_report: Mapping[str, Any] | None,
    *,
    stage: int | str,
    stage_dir: "str | Path | None",
) -> list[str]:
    """Every reason *stance_report* cannot be the ``stance_quality/v2`` evidence for this stage and handoff.

    Empty exactly when the judge would re-derive a verdict from it
    (:func:`_stance_v2_stage_gate`, which reads the verdict off the same
    :func:`_admit_stance_v2_report`).  Shared with the backfill tool, which
    refuses a directory on these reasons rather than writing a FAIL -- so
    every way a report can fail to be re-derived is HERE, and a FAIL is only
    ever a panel that re-derives cleanly to a failing verdict.  Refused,
    each by name:

    * no report, or one whose ``schema`` is not the v3 report a v2 panel writes;
    * ``scored_gate_kind`` other than ``stance_quality/v2`` -- the report
      echoes the DECLARED kind in ``gate_kind``, so a v1-scored report
      labelled v2 would otherwise pass on full horizon and the rail -- and a
      declared kind or a probe marker that says it is not this verdict;
    * ``thresholds`` that differ from the gate view of the block judged
      under on any key (:func:`~environments.shared.curriculum.gate_schema.gate_config_differences`;
      ``settle_steps`` included);
    * a checkpoint scored with the plant contract waived
      (``checkpoint_plant_validated`` not true: ``--allow-legacy-plant``);
    * a ``measurement`` block that is not this checkout's
      (:func:`_measurement_refusals`);
    * a statue block that is not a separately rolled zero-action panel
      (``reused_policy_panel`` not false, or another ``policy``): a
      checkpoint's statue-relative ratios to its own panel are 1 by
      construction;
    * no stage directory, no handoff pair in it, or handoff digests that are
      not the pair's -- including the statue's ``None`` digests: a report
      that scored no checkpoint certifies none;
    * a stage directory whose ``stage_config.json`` is absent or records no
      ``[env]`` block or horizon, or no task fingerprint; a report whose
      ``task_sha256`` is not the task the stage ran (another ``[env]``
      block -- reward weights, reset noise, terrain, horizon -- or another
      plant), whose horizon is not the stage's, or whose manifest's control
      step or ``frame_skip`` is not the stage's;
    * a non-boolean ``passed``;
    * and, once all of that holds, a report whose verdict cannot be
      re-derived from its own rows -- no rows, a row off the certification
      panel (seed ``PUBLICATION_SEED_START + i``), a declared criterion
      whose metric its rows were recorded without (a metric added after
      the panel was rolled; re-rolling it measures the metric), a
      statue-relative criterion declared with no statue rows, or a recorded
      statue reference its rows do not reduce to -- or whose recorded
      ``passed``, clean count or per-episode clean flags disagree with the
      re-derivation.
    """
    refusals, _ = _admit_stance_v2_report(curriculum, stance_report, stage=stage, stage_dir=stage_dir)
    return refusals


def _admit_stance_v2_report(
    curriculum: Mapping[str, Any],
    stance_report: Mapping[str, Any] | None,
    *,
    stage: int | str,
    stage_dir: "str | Path | None",
) -> "tuple[list[str], StanceV2Result | None]":
    """``(refusals, result)``: the :func:`stance_v2_report_refusals` reasons, and the re-derived verdict.

    ``result`` is the :class:`~environments.shared.curriculum.stance_gate_v2.StanceV2Result`
    re-derived from the report's rows exactly when ``refusals`` is empty,
    else ``None``.  The binding checks run first and the re-derivation only
    on a report that passes them all: rows of a report for another
    checkpoint, task or gate are not worth reading.  The binding checks the
    six-heading certificate shares (:func:`_stance_report_identity_refusals`,
    :func:`_stance_report_stage_refusals`) frame the v2 statue block's own.
    """
    from environments.shared.curriculum.stance_gate_v2 import (
        STANCE_GATE_V2_KIND,
        STANCE_V2_REPORT_SCHEMA,
    )

    kind = STANCE_GATE_V2_KIND
    if stance_report is None:
        return [
            f"stage {stage} declares {kind} but no stance gate report was produced, so no episode was classified. "
            "The gate does not fall back to the reward rail, which the zero-action statue clears."
        ], None
    refusals = _stance_report_identity_refusals(
        curriculum, stance_report, kind=kind, schema=STANCE_V2_REPORT_SCHEMA, stage=stage
    )
    refusals.extend(_statue_block_refusals(stance_report.get("statue"), stage=stage, where="stance report's"))
    stage_refusals, horizon, control_dt = _stance_report_stage_refusals(
        stance_report, kind=kind, stage=stage, stage_dir=stage_dir
    )
    refusals.extend(stage_refusals)
    if refusals or horizon is None or control_dt is None:
        return refusals, None
    return _rederive_stance_v2_verdict(curriculum, stance_report, stage=stage, horizon=horizon, control_dt=control_dt)


def _stance_report_identity_refusals(
    curriculum: Mapping[str, Any],
    report: Mapping[str, Any],
    *,
    kind: str,
    schema: str,
    stage: int | str,
) -> list[str]:
    """What a floor-truth stance report must say about itself to be *kind*'s evidence for this gate.

    Shared by ``stance_quality/v2`` and the six-heading certificate, each
    by name: the ``schema``; ``scored_gate_kind`` -- the code that scored it,
    since the report echoes the DECLARED kind in ``gate_kind`` -- and the
    declared kind; no probe marker (every one the report writer knows);
    ``thresholds`` equal to the gate view of the block judged under on every
    key (``settle_steps`` included); ``checkpoint_plant_validated`` true;
    and a ``measurement`` block that is this checkout's.
    """
    from environments.shared.curriculum.gate_schema import gate_config_differences, gate_config_view
    from environments.shared.reporting.stance_report import _PROBE_MARKERS

    refusals: list[str] = []
    if report.get("schema") != schema:
        refusals.append(f"stage {stage} stance report has schema {report.get('schema')!r}, not {schema!r}")
    scored = report.get("scored_gate_kind")
    if scored != kind:
        refusals.append(
            f"stage {stage} declares {kind} but the stance report was scored by {scored!r}; a report the "
            f"{kind} code did not score cannot certify it, whatever kind it echoes"
        )
    if report.get("gate_kind") != kind:
        refusals.append(
            f"stage {stage} stance report records declared gate_kind {report.get('gate_kind')!r}, not {kind!r}"
        )
    # Every probe marker the report writer knows, so a probe added there is refused here without an edit.
    probes = [key for key in _PROBE_MARKERS if report.get(key) is not None]
    if probes:
        refusals.append(f"stage {stage} stance report is a probe ({', '.join(probes)}), which never certifies")
    recorded = report.get("thresholds")
    if not isinstance(recorded, Mapping):
        refusals.append(f"stage {stage} stance report records no thresholds, so the gate it scored is unknown")
    else:
        differences = gate_config_differences(recorded, gate_config_view(curriculum))
        if differences:
            refusals.append(
                f"stage {stage} stance report was scored under other thresholds than the gate judged under "
                f"({'; '.join(differences)}); its verdict certifies only the gate it scored"
            )
    # A zero-action report records None here and is refused below for its
    # None digests; a checkpoint report records False only when it was
    # scored with --allow-legacy-plant, which judges a checkpoint the plant
    # contract would have refused -- a verdict about another plant.
    if report.get("checkpoint_plant_validated") is not True:
        refusals.append(
            f"stage {stage} stance report records checkpoint_plant_validated="
            f"{report.get('checkpoint_plant_validated')!r}: the checkpoint was not validated against the "
            "current plant (--allow-legacy-plant), so its panel is not a verdict about this plant's policy"
        )
    refusals.extend(_measurement_refusals(report, stage=stage))
    return refusals


def _statue_block_refusals(statue: Any, *, stage: int | str, where: str) -> list[str]:
    """Why a report's statue block is not a separately rolled zero-action panel; empty when it is (or is absent).

    The builder reuses the policy's panel as the statue only when the
    policy IS the zero command; on a checkpoint report that would divide
    each episode by its own panel's means, so ``reused_policy_panel`` must
    be false and the ``policy`` the statue the ratios are defined against.
    """
    from environments.shared.curriculum.stance_gate_v2 import STATUE_POLICY

    if statue is None:
        return []
    if not isinstance(statue, Mapping):
        return [f"stage {stage} {where} statue block is {type(statue).__name__}, not an object"]
    refusals: list[str] = []
    if statue.get("reused_policy_panel") is not False:
        refusals.append(
            f"stage {stage} {where} statue panel is not a separately rolled one "
            f"(reused_policy_panel={statue.get('reused_policy_panel')!r}); a checkpoint's ratios to its "
            "own panel are 1 by construction"
        )
    if statue.get("policy") != STATUE_POLICY:
        refusals.append(
            f"stage {stage} {where} statue panel scored {statue.get('policy')!r}, not the "
            f"{STATUE_POLICY!r} statue the ratios are defined against"
        )
    return refusals


def _stance_report_stage_refusals(
    report: Mapping[str, Any],
    *,
    kind: str,
    stage: int | str,
    stage_dir: "str | Path | None",
) -> "tuple[list[str], int | None, float | None]":
    """``(refusals, horizon, control_dt)``: how a floor-truth stance report is bound to *stage_dir*.

    Shared by ``stance_quality/v2`` and the six-heading certificate: the
    handoff digests must be the pair ``select_handoff_checkpoint`` picks now
    (a report that scored no checkpoint certifies none); the task the report
    rolled must be the one the stage recorded it ran; its horizon must be
    the stage's, and its manifest's control step and ``frame_skip`` the
    stage's; and ``passed`` must be a boolean.  ``horizon`` and
    ``control_dt`` are the STAGE's, for the re-derivation, and ``None`` when
    the stage records none.
    """
    from environments.shared.curriculum.checkpoints import select_handoff_checkpoint
    from environments.shared.result_bundle.hashing import sha256_file

    refusals: list[str] = []
    horizon: int | None = None
    control_dt: float | None = None
    if stage_dir is None:
        refusals.append(
            f"stage {stage} declares {kind}, whose report is bound to the stage directory's handoff pair and "
            "recorded task; no stage_dir was given"
        )
    else:
        root = Path(stage_dir)
        handoff = select_handoff_checkpoint(root / "models")
        recorded_handoff = report.get("handoff")
        recorded_handoff = recorded_handoff if isinstance(recorded_handoff, Mapping) else {}
        if handoff is None:
            refusals.append(
                f"{root / 'models'} has no complete handoff pair, so there is no checkpoint the stance report "
                "could be shown to have scored"
            )
        else:
            name, stem, vecnorm = handoff
            for label, path, recorded_digest in (
                ("checkpoint", Path(stem + ".zip"), recorded_handoff.get("checkpoint_sha256")),
                ("VecNormalize sidecar", Path(vecnorm), recorded_handoff.get("normalization_sha256")),
            ):
                digest = sha256_file(path)
                if recorded_digest != digest:
                    refusals.append(
                        f"stage {stage} stance report scored {label} {recorded_digest!r}, not the handoff "
                        f"{name}'s {path.name} ({digest}); a panel of another policy certifies nothing here"
                    )
        # The task: the same policy scores another number under another
        # [env] block, and only the horizon was ever compared -- a panel
        # rolled with tripled alive_bonus and no reset noise certified
        # against a rail its own stage's panel fails.  The fingerprint is
        # the one training recorded (task_fingerprint.json / stage_config.json),
        # and absence refuses: a stage whose task is unknown cannot be shown
        # to be the task the panel rolled.
        task = _current_task_sha256(root)
        if task is None:
            refusals.append(
                f"{root} records no task fingerprint (task_fingerprint.json / stage_config.json), so the task the "
                "stance panel rolled cannot be shown to be the one the stage ran"
            )
        elif report.get("task_sha256") != task:
            refusals.append(
                f"stage {stage} stance report rolled task {report.get('task_sha256')!r}, but the stage ran "
                f"task {task}; a panel under another [env] block or plant scores another task"
            )
        env, missing = _recorded_stage_env(root)
        if env is None:
            refusals.append(f"{missing}, so the horizon and control step the stage ran are unknown")
        else:
            value = env.get("max_episode_steps")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                refusals.append(
                    f"{root / STAGE_CONFIG_FILENAME} records no integer max_episode_steps ({value!r}), so which "
                    "episodes reached the horizon is unknown"
                )
            else:
                horizon = int(value)
                if report.get("horizon") != horizon:
                    refusals.append(
                        f"stage {stage} stance report scored a {report.get('horizon')!r}-step horizon, but "
                        f"the stage ran {horizon} steps (stage_config.json)"
                    )
            # The defaults stance_report and publication read the step with.
            try:
                frame_skip = int(env.get("frame_skip", 5))
                control_dt = float(env.get("timestep", 0.002)) * frame_skip
            except (TypeError, ValueError):
                refusals.append(f"{root / STAGE_CONFIG_FILENAME} records an unreadable timestep or frame_skip")
            else:
                manifest = report.get("measurement")
                manifest = manifest if isinstance(manifest, Mapping) else {}
                measured_dt = manifest.get("dt")
                if (
                    manifest.get("frame_skip") != frame_skip
                    or isinstance(measured_dt, bool)
                    or not isinstance(measured_dt, (int, float))
                    or abs(float(measured_dt) - control_dt) > 1e-9
                ):
                    refusals.append(
                        f"stage {stage} stance report was measured at dt {measured_dt!r} / frame_skip "
                        f"{manifest.get('frame_skip')!r}, but the stage ran dt {control_dt:g} / frame_skip "
                        f"{frame_skip} (stage_config.json); every seconds-defined window is another length"
                    )
    if not isinstance(report.get("passed"), bool):
        refusals.append(f"stage {stage} stance report carries no boolean verdict (passed={report.get('passed')!r})")
    return refusals, horizon, control_dt


def _rederive_stance_v2_verdict(
    curriculum: Mapping[str, Any],
    stance_report: Mapping[str, Any],
    *,
    stage: int | str,
    horizon: int,
    control_dt: float,
) -> "tuple[list[str], StanceV2Result | None]":
    """``(refusals, result)``: the verdict re-derived from an otherwise admissible report's own rows.

    Recomputed -- on the registered certification panel (seed
    ``PUBLICATION_SEED_START + i``), with the statue reference re-reduced
    from the statue's own rows when a statue-relative criterion is
    declared, at the STAGE's horizon and control step -- through
    :func:`~environments.shared.curriculum.stance_gate_v2.evaluate_stance_v2_gate`,
    and refused unless it agrees with the recorded verdict, clean count and
    per-episode classification.  A report's ``passed`` is therefore never
    what certifies the stage: the rows are.  Rows recorded before a metric a
    declared criterion reads existed are refused by name rather than scored:
    every episode would fail that criterion as unmeasured, a verdict about
    the panel's age rather than the policy.
    """
    from environments.shared.constants import PUBLICATION_SEED_START
    from environments.shared.curriculum.stance_gate_v2 import (
        STANCE_GATE_V2_KIND,
        StanceV2Thresholds,
        StatueReference,
        evaluate_stance_v2_gate,
        statue_reference,
        unrecorded_criteria,
    )
    from environments.shared.gait.stance_metrics import StanceEpisodeMetrics

    try:
        thresholds = StanceV2Thresholds.from_curriculum(curriculum)
    except ValueError as exc:
        return [f"stage {stage} declares {STANCE_GATE_V2_KIND} without a judgeable gate: {exc}"], None
    evidence_rows = stance_report.get("episode_evidence")
    if isinstance(evidence_rows, list) and evidence_rows and all(isinstance(row, Mapping) for row in evidence_rows):
        unrecorded = unrecorded_criteria(thresholds, set.intersection(*(set(row) for row in evidence_rows)))
        if unrecorded:
            named = ", ".join(key if key == metric else f"{key} ({metric})" for key, metric in unrecorded)
            return [
                f"stage {stage} stance report's episode rows record no metric for the declared {named}: the panel "
                "was rolled before the metric existed, so it was never measured on it; re-roll the panel on the "
                "handoff pair (the notebook JUDGE branch / generate_stage_artifacts) to measure it"
            ], None

    def episodes_of(rows: Any, *, what: str) -> list[StanceEpisodeMetrics]:
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"the {what} records no episode rows")
        episodes = []
        for index, row in enumerate(rows):
            if not isinstance(row, Mapping) or row.get("seed") != PUBLICATION_SEED_START + index:
                seed = row.get("seed") if isinstance(row, Mapping) else row
                raise ValueError(
                    f"{what} row {index} ran on seed {seed!r}, not the certification panel seed "
                    f"{PUBLICATION_SEED_START + index}"
                )
            episodes.append(StanceEpisodeMetrics.from_row(row))
        return episodes

    try:
        rows = stance_report["episode_evidence"]
        episodes = episodes_of(rows, what="stance report")
        statue: StatueReference | None = None
        if thresholds.declares_statue_criteria():
            block = stance_report.get("statue")
            if not isinstance(block, Mapping):
                raise ValueError("a statue-relative criterion is declared but the report rolled no statue panel")
            statue = statue_reference(episodes_of(block.get("episode_evidence"), what="statue panel"), horizon=horizon)
            recorded_statue = StatueReference.from_dict(block.get("reference") or {})
            if not _same_reading(statue.as_dict(), recorded_statue.as_dict()):
                raise ValueError(
                    f"the recorded statue reference {recorded_statue.as_dict()} is not the one its own rows "
                    f"reduce to ({statue.as_dict()})"
                )
        result = evaluate_stance_v2_gate(episodes, thresholds, horizon=horizon, statue=statue, control_dt=control_dt)
    except (KeyError, TypeError, ValueError) as exc:
        return [
            f"stage {stage} stance report cannot be re-derived ({type(exc).__name__}: {exc}); a verdict that "
            "cannot be recomputed from its rows proves nothing"
        ], None
    recorded_result = stance_report.get("result")
    recorded_clean = [row.get("clean") for row in rows]
    disagreements = []
    if result.passed != stance_report["passed"]:
        disagreements.append(f"passed {stance_report['passed']} recorded, {result.passed} re-derived")
    if not isinstance(recorded_result, Mapping) or recorded_result.get("n_clean") != result.n_clean:
        recorded_n = recorded_result.get("n_clean") if isinstance(recorded_result, Mapping) else None
        disagreements.append(f"{recorded_n!r} clean episodes recorded, {result.n_clean} re-derived")
    if recorded_clean != [not reasons for reasons in result.episode_reasons]:
        disagreements.append("the per-episode clean flags differ from the re-derived classification")
    if disagreements:
        return [
            f"stage {stage} stance report's verdict disagrees with the one re-derived from its own rows "
            f"({'; '.join(disagreements)}); the report was edited or scored by other code"
        ], None
    return [], result


def _stance_v2_stage_gate(
    curriculum: Mapping[str, Any],
    stance_report: Mapping[str, Any] | None,
    *,
    stage: int | str,
    stage_dir: "str | Path | None",
) -> tuple[bool, list[str]]:
    """Judge ``stance_quality/v2`` by re-deriving the report's verdict, refusing every substitute.

    :func:`_admit_stance_v2_report` -- the :func:`stance_v2_report_refusals`
    binding checks, then the re-derivation from the report's own rows
    (:func:`_rederive_stance_v2_verdict`) -- and the verdict is the
    re-derived one.  A refused report fails with its refusals; an admitted
    one passes or fails on the re-derived reasons.
    """
    refusals, result = _admit_stance_v2_report(curriculum, stance_report, stage=stage, stage_dir=stage_dir)
    if refusals or result is None:
        return False, refusals or [f"stage {stage} stance report could not be judged"]
    if result.passed:
        return True, []
    return False, [f"stage {stage} {failure}" for failure in result.failures]


def stance_heading_report_refusals(
    curriculum: Mapping[str, Any],
    report: Mapping[str, Any] | None,
    *,
    stage: int | str,
    stage_dir: "str | Path | None",
) -> list[str]:
    """Every reason *report* cannot be the six-heading certificate for this stage and handoff (decision D-D32).

    Empty exactly when the judge would re-derive a verdict from it
    (:func:`_stance_heading_stage_gate`, through the same
    :func:`_admit_stance_heading_report`).  Shared with the backfill tool and
    ancestor reuse (rule 8), which refuse on these reasons -- so a FAIL is
    only ever a certificate whose rows re-derive cleanly to a failing
    verdict.  Refused, each by name:

    * no certificate; a ``stance_quality/v2`` report (``mesozoic.stance-gate-report/v5``),
      which certifies yaw zero only and cannot be promoted; the report-only
      heading probe (``mesozoic.stance-heading-probe/v1``); any other schema;
    * everything a v2 report is refused for
      (:func:`_stance_report_identity_refusals`,
      :func:`_stance_report_stage_refusals`): the scored and declared kind,
      a probe marker, other thresholds, a checkpoint scored with the plant
      contract waived, another measurement definition, handoff digests that
      are not the pair's, another task, horizon or control step, a
      non-boolean ``passed``;
    * a protocol other than ``whole_scene_yaw/v1``, a grid other than its
      six headings in order (a missing heading, a synonym such as -180 beside
      180, a reorder), seeds other than the certification panel
      ``PUBLICATION_SEED_START + i`` (the 7042 fresh-audit block among them:
      a recorded manual audit, never this gate's evidence);
    * a policy not rolled deterministically, or observations normalised any
      way but the checkpoint's frozen VecNormalize statistics;
    * a ``gate_sha256`` that is not the digest of the gate judged under;
    * per heading, a statue block that is not a separately rolled
      zero-action panel, no statue rows where a statue-relative criterion is
      declared, or a recorded statue reference its rows do not reduce to;
    * and, once all of that holds, rows that are not exactly the seed list
      at their heading, or a recorded ``passed``, clean-seed count,
      per-heading clean count or per-cell ``clean`` flag the re-derivation
      contradicts.
    """
    refusals, _ = _admit_stance_heading_report(curriculum, report, stage=stage, stage_dir=stage_dir)
    return refusals


def _admit_stance_heading_report(
    curriculum: Mapping[str, Any],
    report: Mapping[str, Any] | None,
    *,
    stage: int | str,
    stage_dir: "str | Path | None",
) -> "tuple[list[str], StanceHeadingResult | None]":
    """``(refusals, result)``: the :func:`stance_heading_report_refusals` reasons, and the re-derived verdict."""
    from environments.shared.constants import PUBLICATION_SEED_START
    from environments.shared.curriculum.gate_schema import gate_config_sha256, gate_config_view
    from environments.shared.curriculum.stance_gate_v2 import STANCE_V2_REPORT_SCHEMA
    from environments.shared.curriculum.stance_heading_gate import (
        FROZEN_VECNORMALIZE_MODE,
        HEADING_GRID_DEG,
        HEADING_PROTOCOL,
        STANCE_HEADING_GATE_KIND,
        STANCE_HEADING_REPORT_JSON,
        STANCE_HEADING_REPORT_SCHEMA,
        validate_heading_grid,
    )
    from environments.shared.reporting.stance_report import HEADING_PROBE_SCHEMA

    kind = STANCE_HEADING_GATE_KIND
    if report is None:
        return [
            f"stage {stage} declares {kind} but no six-heading certificate ({STANCE_HEADING_REPORT_JSON}) was "
            "produced, so no heading was measured. The gate never falls back to the yaw-zero stance_quality/v2 "
            "panel, nor to the reward rail the zero-action statue clears."
        ], None
    schema = report.get("schema")
    if schema == STANCE_V2_REPORT_SCHEMA:
        return [
            f"stage {stage} declares {kind} but its stance report is a stance_quality/v2 report ({schema}), which "
            "certifies yaw zero only and cannot be promoted to the six-heading certificate, whatever is attached "
            "beside it"
        ], None
    if schema == HEADING_PROBE_SCHEMA:
        return [
            f"stage {stage} declares {kind} but its stance report is the report-only heading probe ({schema}): "
            "eight seeds at four headings, no floor-truth classification and no binding, which never certifies"
        ], None
    refusals = _stance_report_identity_refusals(
        curriculum, report, kind=kind, schema=STANCE_HEADING_REPORT_SCHEMA, stage=stage
    )
    if report.get("protocol") != HEADING_PROTOCOL:
        refusals.append(
            f"stage {stage} certificate was rolled under protocol {report.get('protocol')!r}, not {HEADING_PROTOCOL!r}"
        )
    try:
        validate_heading_grid(report.get("heading_grid_deg"))  # type: ignore[arg-type]
    except ValueError as exc:
        refusals.append(f"stage {stage} certificate's heading grid is refused: {exc}")
    seeds = report.get("seeds")
    if (
        not isinstance(seeds, list)
        or not seeds
        or seeds != [PUBLICATION_SEED_START + index for index in range(len(seeds))]
        or any(isinstance(seed, bool) for seed in seeds)
    ):
        shown = seeds if not isinstance(seeds, list) or len(seeds) <= 4 else [*seeds[:3], "...", seeds[-1]]
        refusals.append(
            f"stage {stage} certificate rolled seeds {shown!r}, not the certification panel "
            f"{PUBLICATION_SEED_START}, {PUBLICATION_SEED_START + 1}, ...: another seed block (the fresh audit at "
            "7042 among them) is not this gate's evidence"
        )
    if report.get("policy_deterministic") is not True:
        refusals.append(
            f"stage {stage} certificate records policy_deterministic={report.get('policy_deterministic')!r}: a "
            "stochastic policy draws other actions at each heading, so its rows do not pair by seed"
        )
    normalization = report.get("normalization")
    mode = normalization.get("mode") if isinstance(normalization, Mapping) else None
    if mode != FROZEN_VECNORMALIZE_MODE:
        refusals.append(
            f"stage {stage} certificate normalised observations with {mode!r}, not the checkpoint's frozen "
            f"VecNormalize statistics ({FROZEN_VECNORMALIZE_MODE!r}); a policy fed other observations is another "
            "policy"
        )
    current_gate = gate_config_sha256(gate_config_view(curriculum))
    if report.get("gate_sha256") != current_gate:
        refusals.append(
            f"stage {stage} certificate records gate_sha256 {report.get('gate_sha256')!r}, not the digest of the "
            f"gate judged under ({current_gate})"
        )
    raw_headings = report.get("headings")
    headings = (
        [heading for heading in raw_headings if isinstance(heading, Mapping)] if isinstance(raw_headings, list) else []
    )
    recorded_grid = [heading.get("yaw_deg") for heading in headings] if isinstance(raw_headings, list) else None
    if recorded_grid != list(HEADING_GRID_DEG) or len(raw_headings or ()) != len(HEADING_GRID_DEG):
        refusals.append(
            f"stage {stage} certificate's heading cells are {recorded_grid!r}, not one per heading of "
            f"{list(HEADING_GRID_DEG)} in order"
        )
    else:
        for heading in headings:
            refusals.extend(
                _statue_block_refusals(
                    heading.get("statue"), stage=stage, where=f"certificate's {float(heading['yaw_deg']):+g} deg"
                )
            )
    stage_refusals, horizon, control_dt = _stance_report_stage_refusals(
        report, kind=kind, stage=stage, stage_dir=stage_dir
    )
    refusals.extend(stage_refusals)
    if refusals or horizon is None or control_dt is None:
        return refusals, None
    return _rederive_stance_heading_verdict(curriculum, report, stage=stage, horizon=horizon, control_dt=control_dt)


def _rederive_stance_heading_verdict(
    curriculum: Mapping[str, Any],
    report: Mapping[str, Any],
    *,
    stage: int | str,
    horizon: int,
    control_dt: float,
) -> "tuple[list[str], StanceHeadingResult | None]":
    """``(refusals, result)``: the certificate re-derived from an otherwise admissible report's own rows.

    Every heading's rows must be exactly the recorded seed list (the
    certification panel) at that heading -- which makes the grid complete
    and unique -- and, where a statue-relative criterion is declared, its
    statue rows too, re-reduced to the reference that heading is judged
    against.  Recomputed at the STAGE's horizon and control step through
    :func:`~environments.shared.curriculum.stance_heading_gate.evaluate_stance_heading_gate`,
    and refused unless it agrees with the recorded verdict, clean-seed
    count, each heading's clean count and every cell's clean flag.  Rows
    recorded before a metric a declared criterion reads are refused by name,
    as v2's are.
    """
    from environments.shared.curriculum.stance_gate_v2 import (
        StanceV2Thresholds,
        StatueReference,
        statue_reference,
        unrecorded_criteria,
    )
    from environments.shared.curriculum.stance_heading_gate import (
        STANCE_HEADING_GATE_KIND,
        HeadingEpisode,
        canonical_heading_deg,
        evaluate_stance_heading_gate,
    )
    from environments.shared.gait.stance_metrics import StanceEpisodeMetrics

    try:
        thresholds = StanceV2Thresholds.from_curriculum(curriculum)
    except ValueError as exc:
        return [f"stage {stage} declares {STANCE_HEADING_GATE_KIND} without a judgeable gate: {exc}"], None
    seeds: list[int] = list(report["seeds"])
    headings: list[Mapping[str, Any]] = list(report["headings"])
    # Only list-valued cells: any other value is refused by name below, never iterated here.
    all_rows = [
        row
        for heading in headings
        if isinstance(heading.get("episode_evidence"), list)
        for row in heading["episode_evidence"]
    ]
    if all_rows and all(isinstance(row, Mapping) for row in all_rows):
        unrecorded = unrecorded_criteria(thresholds, set.intersection(*(set(row) for row in all_rows)))
        if unrecorded:
            named = ", ".join(key if key == metric else f"{key} ({metric})" for key, metric in unrecorded)
            return [
                f"stage {stage} certificate's episode rows record no metric for the declared {named}: the grid "
                "was rolled before the metric existed; re-roll the certificate on the handoff pair (the notebook "
                "JUDGE branch / generate_stage_artifacts) to measure it"
            ], None

    def episodes_of(rows: Any, *, yaw: float, what: str) -> list[StanceEpisodeMetrics]:
        if not isinstance(rows, list) or len(rows) != len(seeds):
            count = len(rows) if isinstance(rows, list) else rows
            raise ValueError(f"the {what} at {yaw:+g} deg records {count!r} rows, not one per seed ({len(seeds)})")
        episodes = []
        for index, row in enumerate(rows):
            if not isinstance(row, Mapping) or row.get("seed") != seeds[index]:
                seed = row.get("seed") if isinstance(row, Mapping) else row
                raise ValueError(f"{what} row {index} at {yaw:+g} deg ran on seed {seed!r}, not {seeds[index]}")
            if canonical_heading_deg(row.get("yaw_deg")) != yaw:
                raise ValueError(f"{what} row {index} of the {yaw:+g} deg cell records heading {row.get('yaw_deg')!r}")
            episodes.append(StanceEpisodeMetrics.from_row(row))
        return episodes

    try:
        policy: list[HeadingEpisode] = []
        statue_by_heading: dict[float, StatueReference | None] = {}
        for heading in headings:
            yaw = canonical_heading_deg(heading["yaw_deg"])
            metrics = episodes_of(heading.get("episode_evidence"), yaw=yaw, what="certificate")
            policy.extend(HeadingEpisode(seed, yaw, row) for seed, row in zip(seeds, metrics))
            statue_by_heading[yaw] = None
            if thresholds.declares_statue_criteria():
                block = heading.get("statue")
                if not isinstance(block, Mapping):
                    raise ValueError(
                        f"a statue-relative criterion is declared but the {yaw:+g} deg cell rolled no statue panel"
                    )
                reference = statue_reference(
                    episodes_of(block.get("episode_evidence"), yaw=yaw, what="statue panel"), horizon=horizon
                )
                recorded_reference = StatueReference.from_dict(block.get("reference") or {})
                if not _same_reading(reference.as_dict(), recorded_reference.as_dict()):
                    raise ValueError(
                        f"the {yaw:+g} deg cell's recorded statue reference {recorded_reference.as_dict()} is not "
                        f"the one its own rows reduce to ({reference.as_dict()})"
                    )
                statue_by_heading[yaw] = reference
        result = evaluate_stance_heading_gate(
            policy,
            thresholds,
            horizon=horizon,
            seeds=seeds,
            statue_by_heading=statue_by_heading,
            control_dt=control_dt,
        )
    except (KeyError, TypeError, ValueError) as exc:
        return [
            f"stage {stage} certificate cannot be re-derived ({type(exc).__name__}: {exc}); a verdict that cannot "
            "be recomputed from its rows proves nothing"
        ], None
    disagreements: list[str] = []
    if result.passed != report["passed"]:
        disagreements.append(f"passed {report['passed']} recorded, {result.passed} re-derived")
    recorded_result = report.get("result")
    recorded_k = recorded_result.get("n_clean_seeds") if isinstance(recorded_result, Mapping) else None
    if recorded_k != result.n_clean_seeds:
        disagreements.append(f"{recorded_k!r} clean seeds recorded, {result.n_clean_seeds} re-derived")
    for heading in headings:
        yaw = canonical_heading_deg(heading["yaw_deg"])
        rederived = result.heading_result(yaw)
        if rederived is None:
            disagreements.append(f"the {yaw:+g} deg cell was not re-derived")
            continue
        cell = heading.get("result")
        recorded_n = cell.get("n_clean") if isinstance(cell, Mapping) else None
        if recorded_n != rederived.n_clean:
            disagreements.append(
                f"{recorded_n!r} clean episodes recorded at {yaw:+g} deg, {rederived.n_clean} re-derived"
            )
        flags = [row.get("clean") for row in heading["episode_evidence"]]
        if flags != [not reasons for reasons in rederived.episode_reasons]:
            disagreements.append(f"the {yaw:+g} deg cell's clean flags differ from the re-derived classification")
    if disagreements:
        return [
            f"stage {stage} certificate's verdict disagrees with the one re-derived from its own rows "
            f"({'; '.join(disagreements)}); the report was edited or scored by other code"
        ], None
    return [], result


def _stance_heading_stage_gate(
    curriculum: Mapping[str, Any],
    report: Mapping[str, Any] | None,
    *,
    stage: int | str,
    stage_dir: "str | Path | None",
) -> tuple[bool, list[str]]:
    """Judge ``stance_quality_heading/v1`` by re-deriving the certificate, refusing every substitute.

    :func:`_admit_stance_heading_report` -- the binding checks, then the
    re-derivation from every (seed, heading) row -- and the verdict is the
    re-derived one.  A refused certificate fails with its refusals; an
    admitted one passes or fails on the re-derived reasons.  There is no
    fallback: a stage with no certificate, or one that cannot be admitted,
    fails, whatever a yaw-zero panel beside it says.
    """
    refusals, result = _admit_stance_heading_report(curriculum, report, stage=stage, stage_dir=stage_dir)
    if refusals or result is None:
        return False, refusals or [f"stage {stage} certificate could not be judged"]
    if result.passed:
        return True, []
    return False, [f"stage {stage} {failure}" for failure in result.failures]


def command_tracking_report_refusals(
    curriculum: Mapping[str, Any],
    report: Mapping[str, Any] | None,
    *,
    stage: int | str,
    stage_dir: "str | Path | None",
) -> list[str]:
    """Every reason *report* cannot be the ``command_tracking/v1`` evidence for this stage and handoff (D-D61).

    Empty exactly when the judge would re-derive a verdict from it
    (:func:`_command_tracking_stage_gate`, through the same
    :func:`_admit_command_tracking_report`).  Shared with the backfill tool,
    which refuses on these reasons -- so a FAIL is only ever a panel that
    re-derives cleanly to a failing verdict.  Refused, each by name:

    * no report, another ``schema``, a ``scored_gate_kind`` or declared
      ``gate_kind`` other than ``command_tracking/v1``, another ``protocol``;
    * ``thresholds`` that differ from the gate view of the block judged
      under, or a ``gate_sha256`` that is not its digest;
    * a checkpoint scored with the plant contract waived, a policy not
      rolled deterministically, or observations normalised any way but the
      checkpoint's frozen VecNormalize statistics;
    * a panel that does not start at the certification seed
      ``PUBLICATION_SEED_START``;
    * no stage directory, no handoff pair in it, handoff digests that are not
      the pair's, and no valid handoff choice (``models/handoff_choice.json``)
      naming that pair under this gate and the stage's task, among exactly
      the pairs the stage holds (none rewritten or added since the choice):
      a ``command_tracking/v1`` handoff is chosen by the gated statistic,
      and a pair chosen by reward is not this gate's handoff;
    * a stage directory that records no task fingerprint, ``[env]`` block,
      horizon or live command config; a report rolled on another task,
      horizon, control step or command config than the stage ran;
    * a reference panel that is not the stage's entry edge (the parent
      checkpoint ``stage_config.json`` records it loaded);
    * a non-boolean ``passed``;
    * and, once all of that holds, records that cannot be read back
      (``CommandTrackingEpisode.from_dict``), seeds or families that are not
      the panel the stage's thresholds and command plan from the
      certification seed, or a recorded ``passed``, success count or
      per-episode success flag the re-derivation contradicts.
    """
    refusals, _ = _admit_command_tracking_report(curriculum, report, stage=stage, stage_dir=stage_dir)
    return refusals


def _admit_command_tracking_report(
    curriculum: Mapping[str, Any],
    report: Mapping[str, Any] | None,
    *,
    stage: int | str,
    stage_dir: "str | Path | None",
) -> "tuple[list[str], CommandTrackingResult | None]":
    """``(refusals, result)``: the :func:`command_tracking_report_refusals` reasons, and the re-derived verdict."""
    from environments.shared.constants import PUBLICATION_SEED_START
    from environments.shared.curriculum.command_tracking_evidence import (
        COMMAND_TRACKING_REPORT_JSON,
        COMMAND_TRACKING_REPORT_SCHEMA,
    )
    from environments.shared.curriculum.command_tracking_gate import (
        COMMAND_PANEL_PROTOCOL,
        COMMAND_TRACKING_GATE_KIND,
    )
    from environments.shared.curriculum.gate_schema import (
        gate_config_differences,
        gate_config_sha256,
        gate_config_view,
    )
    from environments.shared.curriculum.stance_heading_gate import FROZEN_VECNORMALIZE_MODE

    kind = COMMAND_TRACKING_GATE_KIND
    if report is None:
        return [
            f"stage {stage} declares {kind} but no command-tracking report ({COMMAND_TRACKING_REPORT_JSON}) was "
            "produced, so no command episode was judged. The gate never falls back to the reward return, which "
            "mixes the tracking terms with every other term."
        ], None
    refusals: list[str] = []
    if report.get("schema") != COMMAND_TRACKING_REPORT_SCHEMA:
        refusals.append(
            f"stage {stage} command-tracking report has schema {report.get('schema')!r}, not "
            f"{COMMAND_TRACKING_REPORT_SCHEMA!r}"
        )
    if report.get("scored_gate_kind") != kind:
        refusals.append(
            f"stage {stage} declares {kind} but the report was scored by {report.get('scored_gate_kind')!r}; a "
            f"report the {kind} code did not score cannot certify it"
        )
    if report.get("gate_kind") != kind:
        refusals.append(f"stage {stage} report records declared gate_kind {report.get('gate_kind')!r}, not {kind!r}")
    if report.get("protocol") != COMMAND_PANEL_PROTOCOL:
        refusals.append(
            f"stage {stage} report was rolled under protocol {report.get('protocol')!r}, not {COMMAND_PANEL_PROTOCOL!r}"
        )
    recorded = report.get("thresholds")
    view = gate_config_view(curriculum)
    if not isinstance(recorded, Mapping):
        refusals.append(f"stage {stage} report records no thresholds, so the gate it scored is unknown")
    else:
        differences = gate_config_differences(recorded, view)
        if differences:
            refusals.append(
                f"stage {stage} report was scored under other thresholds than the gate judged under "
                f"({'; '.join(differences)}); its verdict certifies only the gate it scored"
            )
    current_gate = gate_config_sha256(view)
    if report.get("gate_sha256") != current_gate:
        refusals.append(
            f"stage {stage} report records gate_sha256 {report.get('gate_sha256')!r}, not the digest of the gate "
            f"judged under ({current_gate})"
        )
    if report.get("checkpoint_plant_validated") is not True:
        refusals.append(
            f"stage {stage} report records checkpoint_plant_validated={report.get('checkpoint_plant_validated')!r}: "
            "the checkpoint was not validated against the current plant, so its panel is not a verdict about this "
            "plant's policy"
        )
    if report.get("policy_deterministic") is not True:
        refusals.append(
            f"stage {stage} report records policy_deterministic={report.get('policy_deterministic')!r}: a stochastic "
            "rollout is another panel on every roll"
        )
    normalization = report.get("normalization")
    mode = normalization.get("mode") if isinstance(normalization, Mapping) else None
    if mode != FROZEN_VECNORMALIZE_MODE:
        refusals.append(
            f"stage {stage} report normalised observations with {mode!r}, not the checkpoint's frozen VecNormalize "
            f"statistics ({FROZEN_VECNORMALIZE_MODE!r}); a policy fed other observations is another policy"
        )
    if report.get("seed_start") != PUBLICATION_SEED_START or isinstance(report.get("seed_start"), bool):
        refusals.append(
            f"stage {stage} report's panel starts at seed {report.get('seed_start')!r}, not the certification seed "
            f"{PUBLICATION_SEED_START}: another seed block (the handoff choice's selection panel among them) is not "
            "this gate's evidence"
        )
    stage_refusals, binding = _command_report_stage_refusals(
        report, stage=stage, stage_dir=stage_dir, gate=current_gate
    )
    refusals.extend(stage_refusals)
    if refusals or binding is None:
        return refusals, None
    return _rederive_command_tracking_verdict(curriculum, report, stage=stage, binding=binding)


def _selection_overlap_problem(selection_plan: Any, certification_plan: Any) -> str | None:
    """Why a handoff choice's selection panel is not disjoint from the certification panel, or ``None``.

    Both plans are ``CommandPanelPlan.as_dict()`` records; a plan whose
    seeds cannot be read is a problem too, since disjointness cannot then be
    shown.
    """
    from .command_tracking_report import shared_panel_seeds

    seeds: list[list[int]] = []
    for label, plan in (("selection", selection_plan), ("certification", certification_plan)):
        listed = plan.get("seeds") if isinstance(plan, Mapping) else None
        if not isinstance(listed, list) or not all(
            isinstance(seed, int) and not isinstance(seed, bool) for seed in listed
        ):
            return f"records no readable {label} plan, so its selection draws cannot be shown disjoint from the certificate"
        seeds.append(listed)
    shared = shared_panel_seeds(seeds[0], seeds[1])
    if shared:
        return (
            f"was made on a selection panel sharing {len(shared)} seed(s) with the certification panel (first "
            f"{shared[0]}); the draws that chose the handoff must not certify it"
        )
    return None


def _command_report_stage_refusals(
    report: Mapping[str, Any],
    *,
    stage: int | str,
    stage_dir: "str | Path | None",
    gate: str,
) -> "tuple[list[str], dict[str, Any] | None]":
    """``(refusals, binding)``: how a command-tracking report is bound to *stage_dir*.

    ``binding`` holds what the re-derivation needs from the STAGE's own
    records (its command config, horizon and control step, and the parent
    checkpoint digest its entry edge recorded), and is ``None`` when any of
    them is unknown.
    """
    from environments.shared.curriculum.checkpoints import (
        handoff_choice_candidates_problem,
        recorded_handoff_choice,
        select_handoff_checkpoint,
    )
    from environments.shared.curriculum.command_tracking_gate import COMMAND_TRACKING_GATE_KIND, command_config_from
    from environments.shared.result_bundle.hashing import sha256_file

    from ..config import env_kwargs_record, read_recorded_stage_config

    kind = COMMAND_TRACKING_GATE_KIND
    if stage_dir is None:
        return [
            f"stage {stage} declares {kind}, whose report is bound to the stage directory's handoff pair, its "
            "handoff choice and its recorded task; no stage_dir was given"
        ], None
    root = Path(stage_dir)
    refusals: list[str] = []
    binding: dict[str, Any] = {}
    handoff = select_handoff_checkpoint(root / "models")
    recorded_handoff = report.get("handoff")
    recorded_handoff = recorded_handoff if isinstance(recorded_handoff, Mapping) else {}
    if handoff is None:
        refusals.append(
            f"{root / 'models'} has no complete handoff pair, so there is no checkpoint the report could be shown "
            "to have scored"
        )
    else:
        name, stem, vecnorm = handoff
        digests = {}
        for label, path, key in (
            ("checkpoint", Path(stem + ".zip"), "checkpoint_sha256"),
            ("VecNormalize sidecar", Path(vecnorm), "normalization_sha256"),
        ):
            digests[key] = sha256_file(path)
            if recorded_handoff.get(key) != digests[key]:
                refusals.append(
                    f"stage {stage} report scored {label} {recorded_handoff.get(key)!r}, not the handoff {name}'s "
                    f"{path.name} ({digests[key]}); a panel of another policy certifies nothing here"
                )
        choice, problem = recorded_handoff_choice(root / "models")
        if choice is None:
            refusals.append(
                f"stage {stage} records no valid handoff choice in {root / 'models'} "
                f"({problem or 'handoff_choice.json is absent'}): a {kind} handoff is chosen by the gated statistic "
                "on the selection panel (reporting.command_tracking_report.choose_command_handoff, run at the end of "
                "training and by evaluate_stage_checkpoints), never by reward"
            )
        else:
            if choice.get("gate_kind") != kind or choice.get("gate_sha256") != gate:
                refusals.append(
                    f"stage {stage}'s handoff was chosen under gate {choice.get('gate_kind')!r} "
                    f"{choice.get('gate_sha256')!r}, not the gate judged under ({gate}); choose it again under this "
                    "gate (evaluate_stage_checkpoints does on the JUDGE branch)"
                )
            if (choice.get("checkpoint_sha256"), choice.get("normalization_sha256")) != (
                digests["checkpoint_sha256"],
                digests["normalization_sha256"],
            ):
                refusals.append(f"stage {stage}'s handoff choice names another pair than the handoff {name}")
            stale = handoff_choice_candidates_problem(choice, root / "models")
            if stale is not None:
                refusals.append(
                    f"stage {stage}'s handoff choice is no longer a choice among the stage's checkpoints ({stale}); "
                    "choose it again (evaluate_stage_checkpoints does on the JUDGE branch)"
                )
            overlap = _selection_overlap_problem(choice.get("plan"), report.get("plan"))
            if overlap is not None:
                refusals.append(f"stage {stage}'s handoff choice {overlap}")
    task = _current_task_sha256(root)
    if task is None:
        refusals.append(
            f"{root} records no task fingerprint (task_fingerprint.json / stage_config.json), so the task the "
            "command panel rolled cannot be shown to be the one the stage ran"
        )
    else:
        if report.get("task_sha256") != task:
            refusals.append(
                f"stage {stage} report rolled task {report.get('task_sha256')!r}, but the stage ran task {task}; a "
                "panel under another [env] block, command or plant scores another task"
            )
        choice, _ = recorded_handoff_choice(root / "models")
        if choice is not None and choice.get("task_sha256") != task:
            refusals.append(
                f"stage {stage}'s handoff was chosen on task {choice.get('task_sha256')!r}, not the task the stage "
                f"ran ({task})"
            )
    env, missing = _recorded_stage_env(root)
    if env is None:
        refusals.append(f"{missing}, so the horizon, control step and command the stage ran are unknown")
    else:
        value = env.get("max_episode_steps")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            refusals.append(f"{root / STAGE_CONFIG_FILENAME} records no integer max_episode_steps ({value!r})")
        else:
            binding["horizon"] = int(value)
            if report.get("horizon") != int(value):
                refusals.append(
                    f"stage {stage} report scored a {report.get('horizon')!r}-step horizon, but the stage ran "
                    f"{int(value)} steps (stage_config.json)"
                )
        try:
            control_dt = float(env.get("timestep", 0.002)) * int(env.get("frame_skip", 5))
        except (TypeError, ValueError):
            refusals.append(f"{root / STAGE_CONFIG_FILENAME} records an unreadable timestep or frame_skip")
        else:
            binding["control_dt"] = control_dt
            measured = report.get("control_dt")
            if (
                isinstance(measured, bool)
                or not isinstance(measured, (int, float))
                or abs(float(measured) - control_dt) > 1e-9
            ):
                refusals.append(
                    f"stage {stage} report was rolled at control step {measured!r}, but the stage ran {control_dt:g} s "
                    "(stage_config.json); every seconds-defined window is another length"
                )
        try:
            command_config = command_config_from(env.get("command_config"))
        except (TypeError, ValueError) as exc:
            refusals.append(f"{root / STAGE_CONFIG_FILENAME} records an unreadable command_config ({exc})")
        else:
            if command_config is None:
                refusals.append(
                    f"{root / STAGE_CONFIG_FILENAME} records no live command_config: the stage ran command_mode "
                    f"'none', and {kind} judges a live command"
                )
            else:
                binding["command_config"] = command_config
                recorded_command = env_kwargs_record({"command_config": command_config})["command_config"]
                if not _same_reading(report.get("command_config"), recorded_command):
                    refusals.append(
                        f"stage {stage} report rolled command config {report.get('command_config')!r}, not the one "
                        f"the stage ran ({recorded_command})"
                    )
    record = read_recorded_stage_config(root) or {}
    run = record.get("run") if isinstance(record.get("run"), Mapping) else {}
    binding["parent_checkpoint_sha256"] = run.get("parent_checkpoint_sha256") if isinstance(run, Mapping) else None
    reference = report.get("reference")
    if reference is not None:
        reference_handoff = reference.get("handoff") if isinstance(reference, Mapping) else None
        reference_digest = (
            reference_handoff.get("checkpoint_sha256") if isinstance(reference_handoff, Mapping) else None
        )
        if binding["parent_checkpoint_sha256"] is None or reference_digest != binding["parent_checkpoint_sha256"]:
            refusals.append(
                f"stage {stage} report's reference panel scored checkpoint {reference_digest!r}, not the parent "
                f"checkpoint the stage entered on ({binding['parent_checkpoint_sha256']!r}, stage_config.json's "
                "run.parent_checkpoint_sha256): the paired reference is the policy the node continues"
            )
    if not isinstance(report.get("passed"), bool):
        refusals.append(f"stage {stage} report carries no boolean verdict (passed={report.get('passed')!r})")
    if refusals or {"horizon", "control_dt", "command_config"} - set(binding):
        return refusals, None
    return refusals, binding


def _rederive_command_tracking_verdict(
    curriculum: Mapping[str, Any],
    report: Mapping[str, Any],
    *,
    stage: int | str,
    binding: Mapping[str, Any],
) -> "tuple[list[str], CommandTrackingResult | None]":
    """``(refusals, result)``: the verdict re-derived from an otherwise admissible report's own records.

    The records are read back strictly, must be exactly the panel the
    stage's thresholds and command config plan from the certification seed
    (the same seeds and families, in order) with the reference, when
    present, on the same seeds, and are judged through
    :func:`~environments.shared.curriculum.command_tracking_gate.evaluate_command_tracking_gate`
    under the STAGE's command config; the result must agree with the
    recorded verdict, each family's success count and every episode's
    success flag.
    """
    from environments.shared.command_tracking_panel import plan_command_panel
    from environments.shared.constants import PUBLICATION_SEED_START
    from environments.shared.curriculum.command_tracking_gate import (
        COMMAND_TRACKING_GATE_KIND,
        CommandTrackingEpisode,
        evaluate_command_tracking_gate,
        thresholds_from_curriculum,
    )

    try:
        thresholds = thresholds_from_curriculum(curriculum)
    except ValueError as exc:
        return [f"stage {stage} declares {COMMAND_TRACKING_GATE_KIND} without a judgeable gate: {exc}"], None
    command_config = binding["command_config"]
    try:
        entries = report.get("episode_evidence")
        if not isinstance(entries, list) or not entries:
            raise ValueError("the report records no episode rows")
        episodes = [CommandTrackingEpisode.from_dict(entry["record"]) for entry in entries]
        recorded_success = [entry["success"] for entry in entries]
        if not all(isinstance(flag, bool) for flag in recorded_success):
            raise ValueError("an episode row records a non-boolean success flag")
        plan = plan_command_panel(thresholds, command_config, seed_start=PUBLICATION_SEED_START)
        if [episode.seed for episode in episodes] != list(plan.seeds):
            raise ValueError(
                f"the episodes ran on seeds {[episode.seed for episode in episodes][:6]}..., not the panel the stage's "
                f"thresholds and command plan from {PUBLICATION_SEED_START} ({list(plan.seeds)[:6]}...)"
            )
        if [episode.family for episode in episodes] != list(plan.families):
            raise ValueError("the episodes' families are not the planned panel's")
        if any(episode.horizon != binding["horizon"] for episode in episodes) or any(
            abs(episode.dt - binding["control_dt"]) > 1e-9 for episode in episodes
        ):
            raise ValueError("an episode was rolled at another horizon or control step than the stage ran")
        reference = None
        block = report.get("reference")
        if block is not None:
            reference = [CommandTrackingEpisode.from_dict(entry["record"]) for entry in block["episode_evidence"]]
            if [episode.seed for episode in reference] != list(plan.seeds):
                raise ValueError("the reference panel ran on other seeds than the policy's")
            if [episode.family for episode in reference] != list(plan.families):
                raise ValueError("the reference panel's families are not the planned panel's")
        result = evaluate_command_tracking_gate(
            episodes, thresholds, command_config=command_config, reference=reference
        )
    except (KeyError, TypeError, ValueError) as exc:
        return [
            f"stage {stage} command-tracking report cannot be re-derived ({type(exc).__name__}: {exc}); a verdict "
            "that cannot be recomputed from its records proves nothing"
        ], None
    disagreements: list[str] = []
    if result.passed != report["passed"]:
        disagreements.append(f"passed {report['passed']} recorded, {result.passed} re-derived")
    if recorded_success != [not reasons for reasons in result.episode_reasons]:
        disagreements.append("the per-episode success flags differ from the re-derived classification")
    recorded_result = report.get("result")
    recorded_families = recorded_result.get("families") if isinstance(recorded_result, Mapping) else None
    recorded_counts = (
        {family.get("family"): family.get("n_success") for family in recorded_families if isinstance(family, Mapping)}
        if isinstance(recorded_families, list)
        else None
    )
    if recorded_counts != {family.family: family.n_success for family in result.families}:
        disagreements.append(
            f"success counts {recorded_counts!r} recorded, "
            f"{ {family.family: family.n_success for family in result.families} } re-derived"
        )
    if disagreements:
        return [
            f"stage {stage} command-tracking report's verdict disagrees with the one re-derived from its own records "
            f"({'; '.join(disagreements)}); the report was edited or scored by other code"
        ], None
    return [], result


def _command_tracking_stage_gate(
    curriculum: Mapping[str, Any],
    report: Mapping[str, Any] | None,
    *,
    stage: int | str,
    stage_dir: "str | Path | None",
) -> tuple[bool, list[str]]:
    """Judge ``command_tracking/v1`` by re-deriving the report's verdict, refusing every substitute (D-D61).

    :func:`_admit_command_tracking_report` -- the binding checks, then the
    re-derivation from the report's records -- and the verdict is the
    re-derived one.  There is no fallback: a stage with no report, or one
    that cannot be admitted, fails, whatever its reward says.
    """
    refusals, result = _admit_command_tracking_report(curriculum, report, stage=stage, stage_dir=stage_dir)
    if refusals or result is None:
        return False, refusals or [f"stage {stage} command-tracking report could not be judged"]
    if result.passed:
        return True, []
    return False, [f"stage {stage} {failure}" for failure in result.failures]


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

    ``recovery_quality/v1`` may carry ``min_avg_reward`` in the same role
    ``stance_quality/v1`` gives it — a rail well below the null, not the
    gate — but the frozen recovery capability spec records no rail, so its
    own evaluator cannot enforce one.  A declared criterion nobody evaluates
    is the half-enforced gate this module exists to prevent, so it is
    checked here as an additional *conjunct*: it can only refuse, never
    advance anything the kind's own verdict did not already pass.
    *kind_label* names the rail in the failure text (``recovery rail``).
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
    floor only when positive — the same semantics the notebook applied
    inline, preserved exactly so no existing stage changes verdict.  The
    success-rate floor ``min_success_rate`` is retired (decision D-D59), and
    :func:`evaluate_stage_gate` refuses a block that still sets it before
    this conjunction is reached.

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
    command_tracking_report: Mapping[str, Any] | None = None,
) -> tuple[bool, list[str]]:
    """Decide whether a completed stage passes its declared curriculum gate.

    Args:
        curriculum: The stage's ``curriculum_kwargs`` block.
        stage_results: The stage result dict, read for the selected
            checkpoint's metrics (``best_model_*``, falling back to
            ``best_eval_*`` and then the live-eval means).
        stage: Stage identifier, used only in failure messages.
        stance_report: The stance gate report dict produced for this stage
            (``mesozoic.stance-gate-report/v4`` for ``stance_quality/v1``,
            ``/v5`` for ``stance_quality/v2`` -- the report revisions, bumped
            from /v2 and /v3 by decision D-D33, not the kinds -- the
            ``mesozoic.stance-heading-report/v2`` certificate for
            ``stance_quality_heading/v1``), required by every stance kind
            and ignored by every other kind.
        stage_dir: The stage's own directory, required by
            ``recovery_quality/v1`` (it holds the frozen
            ``gate_resolution.json`` and the task fingerprint that resolution
            is checked against) and by ``stance_quality/v2`` and
            ``stance_quality_heading/v1`` (the handoff pair their report
            must have scored), ignored by every other kind.  Omitting it
            does not soften any of them — all three refuse.
        recovery_successes_by_seed: The pushed panel's per-episode successes
            keyed by panel seed (``RecoveryPanelEvidence.successes_by_seed()``),
            required by ``recovery_quality/v1`` and ignored by every other
            kind.  The seeds must be the ones the frozen null manifest was
            measured on; the resolver refuses any other pairing.
        command_tracking_report: The ``mesozoic.command-tracking-report/v1``
            report rolled on the stage's handoff pair
            (``reporting.command_tracking_report``), required by
            ``command_tracking/v1`` together with *stage_dir* (its handoff
            pair, handoff choice and recorded task) and ignored by every
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
    from environments.shared.curriculum.command_tracking_gate import COMMAND_TRACKING_GATE_KIND
    from environments.shared.curriculum.gate_schema import GATE_KINDS, retired_gate_declarations
    from environments.shared.curriculum.recovery_gate import RECOVERY_GATE_KIND
    from environments.shared.curriculum.stance_gate import STANCE_GATE_KIND
    from environments.shared.curriculum.stance_gate_v2 import STANCE_GATE_V2_KIND
    from environments.shared.curriculum.stance_heading_gate import STANCE_HEADING_GATE_KIND

    # A retired kind or key (decision D-D59: task_success/v1, min_success_rate
    # and the rest) is refused before any arm: the kind would otherwise read
    # as merely unknown, and a reward_and_length/v1 block still setting
    # min_success_rate would be judged on its reward floor alone.
    retired = retired_gate_declarations(curriculum)
    if retired:
        return False, [f"stage {stage}: {message}" for message in retired]
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
    if gate_kind == STANCE_GATE_V2_KIND:
        # The floor-truth stance verdict comes ONLY from a v2-scored report
        # bound to this stage's handoff pair, re-derived from its rows; never
        # the v1 arm above (which reads `passed` and re-derives nothing) and
        # never the reward conjunction below, which the statue clears.
        return _stance_v2_stage_gate(curriculum, stance_report, stage=stage, stage_dir=stage_dir)
    if gate_kind == STANCE_HEADING_GATE_KIND:
        # The six-heading certificate (D-D32) comes ONLY from a heading
        # report bound to this stage's handoff pair, re-derived from every
        # (seed, heading) row; never the v2 arm above (a yaw-zero panel is the
        # evidence this kind exists to refuse) and never the reward return.
        return _stance_heading_stage_gate(curriculum, stance_report, stage=stage, stage_dir=stage_dir)
    if gate_kind == COMMAND_TRACKING_GATE_KIND:
        # The command-tracking verdict (D-D61) comes ONLY from a report rolled
        # on this stage's chosen handoff pair, bound to its handoff choice,
        # task and command, and re-derived from every episode record; never
        # the reward conjunction below, whose return mixes the tracking terms
        # with every other term.
        return _command_tracking_stage_gate(curriculum, command_tracking_report, stage=stage, stage_dir=stage_dir)
    if gate_kind == "reward_and_length/v1":
        return _reward_and_length_stage_gate(curriculum, stage_results)
    # A kind registered in GATE_KINDS with no arm above lands here, and must
    # fail closed by construction rather than certify on the reward
    # conjunction.
    return False, [
        f"stage {stage} declares gate_kind {gate_kind!r}, which is registered but has no evaluator in "
        "reporting.gates.evaluate_stage_gate; refusing rather than judging it on the reward conjunction"
    ]
