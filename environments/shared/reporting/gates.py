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

The dispatch is closed at the end: a kind registered in ``GATE_KINDS`` with
no arm here is refused by name, never routed to the reward conjunction.
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
    from ..curriculum.stance_gate_v2 import StanceV2Result


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

    A ``stance_quality/v2`` history is read the same way, from each
    evaluation's ``stance_clean_count`` / ``n_stance_samples`` — and since
    the in-training manager refuses the kind and records neither today, it
    reads ``None``: never the generic arm below, which would judge a v2
    stage on the reward rail its statue clears.
    """
    from environments.shared.curriculum.stance_gate_v2 import STANCE_GATE_V2_KIND
    from environments.shared.curriculum.task_success_gate import TASK_SUCCESS_GATE_KIND

    if curriculum.get("gate_kind") == TASK_SUCCESS_GATE_KIND:
        return _recorded_task_success_gate(curriculum, evaluations)
    if curriculum.get("gate_kind") == STANCE_GATE_V2_KIND:
        return _recorded_stance_v2_gate(curriculum, evaluations)
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
    the species' current ``SUPPORT_REGISTRY`` entry
    (``gait.morphology.measurement_definition``), and its recorded sha256
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
    from environments.shared.species_names import resolve_species_id

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
    try:
        species = resolve_species_id(str(report.get("species")))
    except ValueError:
        species = str(report.get("species"))
    if manifest.get("species") != species:
        refusals.append(f"stage {stage} stance report was measured on {manifest.get('species')!r}, not {species!r}")
    expected = measurement_definition(species)["registry"]
    if not _same_reading(manifest.get("registry"), expected):
        refusals.append(
            f"stage {stage} stance report used the support registry entry {manifest.get('registry')!r}, not the "
            f"current {expected!r} for {species}"
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
      panel (seed ``PUBLICATION_SEED_START + i``), a statue-relative
      criterion declared with no statue rows, or a recorded statue reference
      its rows do not reduce to -- or whose recorded ``passed``, clean count
      or per-episode clean flags disagree with the re-derivation.
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
    checkpoint, task or gate are not worth reading.
    """
    from environments.shared.curriculum.checkpoints import select_handoff_checkpoint
    from environments.shared.curriculum.gate_schema import gate_config_differences, gate_config_view
    from environments.shared.curriculum.stance_gate_v2 import (
        STANCE_GATE_V2_KIND,
        STANCE_V2_REPORT_SCHEMA,
        STATUE_POLICY,
    )
    from environments.shared.reporting.stance_report import _PROBE_MARKERS
    from environments.shared.result_bundle.hashing import sha256_file

    kind = STANCE_GATE_V2_KIND
    if stance_report is None:
        return [
            f"stage {stage} declares {kind} but no stance gate report was produced, so no episode was classified. "
            "The gate does not fall back to the reward rail, which the zero-action statue clears."
        ], None
    refusals: list[str] = []
    if stance_report.get("schema") != STANCE_V2_REPORT_SCHEMA:
        refusals.append(
            f"stage {stage} stance report has schema {stance_report.get('schema')!r}, not {STANCE_V2_REPORT_SCHEMA!r}"
        )
    scored = stance_report.get("scored_gate_kind")
    if scored != kind:
        refusals.append(
            f"stage {stage} declares {kind} but the stance report was scored by {scored!r}; a report the "
            f"{kind} code did not score cannot certify it, whatever kind it echoes"
        )
    if stance_report.get("gate_kind") != kind:
        refusals.append(
            f"stage {stage} stance report records declared gate_kind {stance_report.get('gate_kind')!r}, not {kind!r}"
        )
    # Every probe marker the report writer knows, so a probe added there is refused here without an edit.
    probes = [key for key in _PROBE_MARKERS if stance_report.get(key) is not None]
    if probes:
        refusals.append(f"stage {stage} stance report is a probe ({', '.join(probes)}), which never certifies")
    recorded = stance_report.get("thresholds")
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
    if stance_report.get("checkpoint_plant_validated") is not True:
        refusals.append(
            f"stage {stage} stance report records checkpoint_plant_validated="
            f"{stance_report.get('checkpoint_plant_validated')!r}: the checkpoint was not validated against the "
            "current plant (--allow-legacy-plant), so its panel is not a verdict about this plant's policy"
        )
    refusals.extend(_measurement_refusals(stance_report, stage=stage))
    statue = stance_report.get("statue")
    if statue is not None:
        if not isinstance(statue, Mapping):
            refusals.append(f"stage {stage} stance report's statue block is {type(statue).__name__}, not an object")
        else:
            # The builder reuses the policy's panel as the statue only when
            # the policy IS the zero command; on a checkpoint report that
            # would divide each episode by its own panel's means.
            if statue.get("reused_policy_panel") is not False:
                refusals.append(
                    f"stage {stage} stance report's statue panel is not a separately rolled one "
                    f"(reused_policy_panel={statue.get('reused_policy_panel')!r}); a checkpoint's ratios to its "
                    "own panel are 1 by construction"
                )
            if statue.get("policy") != STATUE_POLICY:
                refusals.append(
                    f"stage {stage} stance report's statue panel scored {statue.get('policy')!r}, not the "
                    f"{STATUE_POLICY!r} statue the ratios are defined against"
                )
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
        recorded_handoff = stance_report.get("handoff")
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
        elif stance_report.get("task_sha256") != task:
            refusals.append(
                f"stage {stage} stance report rolled task {stance_report.get('task_sha256')!r}, but the stage ran "
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
                if stance_report.get("horizon") != horizon:
                    refusals.append(
                        f"stage {stage} stance report scored a {stance_report.get('horizon')!r}-step horizon, but "
                        f"the stage ran {horizon} steps (stage_config.json)"
                    )
            # The defaults stance_report and publication read the step with.
            try:
                frame_skip = int(env.get("frame_skip", 5))
                control_dt = float(env.get("timestep", 0.002)) * frame_skip
            except (TypeError, ValueError):
                refusals.append(f"{root / STAGE_CONFIG_FILENAME} records an unreadable timestep or frame_skip")
            else:
                manifest = stance_report.get("measurement")
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
    if not isinstance(stance_report.get("passed"), bool):
        refusals.append(
            f"stage {stage} stance report carries no boolean verdict (passed={stance_report.get('passed')!r})"
        )
    if refusals or horizon is None or control_dt is None:
        return refusals, None
    return _rederive_stance_v2_verdict(curriculum, stance_report, stage=stage, horizon=horizon, control_dt=control_dt)


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
    what certifies the stage: the rows are.
    """
    from environments.shared.constants import PUBLICATION_SEED_START
    from environments.shared.curriculum.stance_gate_v2 import (
        STANCE_GATE_V2_KIND,
        StanceV2Thresholds,
        StatueReference,
        evaluate_stance_v2_gate,
        statue_reference,
    )
    from environments.shared.gait.stance_metrics import StanceEpisodeMetrics

    try:
        thresholds = StanceV2Thresholds.from_curriculum(curriculum)
    except ValueError as exc:
        return [f"stage {stage} declares {STANCE_GATE_V2_KIND} without a judgeable gate: {exc}"], None

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
        stance_report: The stance gate report dict produced for this stage
            (``mesozoic.stance-gate-report/v2`` for ``stance_quality/v1``,
            ``/v3`` for ``stance_quality/v2``), required by both stance kinds
            and ignored by every other kind.
        stage_dir: The stage's own directory, required by
            ``recovery_quality/v1`` (it holds the frozen
            ``gate_resolution.json`` and the task fingerprint that resolution
            is checked against) and by ``task_success/v1`` (it holds the
            selected checkpoint's ``evaluation_selected.csv`` and the handoff
            pair it is bound to) and by ``stance_quality/v2`` (the handoff
            pair its report must have scored), ignored by every other kind.
            Omitting it does not soften any of them — all three refuse.
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
    from environments.shared.curriculum.stance_gate_v2 import STANCE_GATE_V2_KIND
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
    if gate_kind == STANCE_GATE_V2_KIND:
        # The floor-truth stance verdict comes ONLY from a v2-scored report
        # bound to this stage's handoff pair, re-derived from its rows; never
        # the v1 arm above (which reads `passed` and re-derives nothing) and
        # never the reward conjunction below, which the statue clears.
        return _stance_v2_stage_gate(curriculum, stance_report, stage=stage, stage_dir=stage_dir)
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
