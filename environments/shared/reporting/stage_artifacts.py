"""Post-training stage artifact generation.

The shared entry-points that the training notebook and the gate backfill tool
call, so that every stage they judge records the same artifacts."""

from __future__ import annotations

import json as _json
import logging
from pathlib import Path
from typing import Any, Mapping

from ..constants import PUBLICATION_SEED_START
from ..curriculum.checkpoints import select_handoff_checkpoint
from ..curriculum.stance_gate import DEFAULT_MIN_EVAL_EPISODES_STANCE
from ..policy_loading import load_sb3_model
from . import csv_output, stage_layout, text_summaries

logger = logging.getLogger(__name__)

#: Episodes per cutoff in the action-filter probe sweep. Far below the gate's
#: ``min_eval_episodes`` on purpose: the probe certifies nothing, and the
#: effect it measures is enormous -- 96 steps against a 1000-step horizon on
#: T-Rex stage 1 (issue #491) -- so it does not need the sample size the
#: gate's bound has its power specified at. Keeping it small is what makes a
#: multi-cutoff sweep cost about what the old single 40-episode probe did.
_PROBE_EPISODES = 10

#: Episodes per row in the two sweep probes. Smaller than ``_PROBE_EPISODES``
#: because both roll far more rows -- 13 for the ablation, 14 for the impulse
#: sweep once the statue control doubles it -- and both measure effects that are
#: a fall or a full horizon rather than a shift in a mean.
_ABLATION_EPISODES = 8
_IMPULSE_EPISODES = 8


def build_stage_results_from_eval_data(
    stage_dir: "str | Path",
    stage: int,
    stage_config: dict[str, Any],
    timesteps: int,
    duration_seconds: float = 0.0,
    sim_dt: float | None = None,
) -> dict[str, Any]:
    """Build a ``stage_results`` dict from on-disk evaluation artifacts.

    Reads ``evaluations.npz`` (written by SB3's ``EvalCallback``) and
    ``metrics.json`` to reconstruct the same results dict that the
    training notebook's ``train_stage`` produces.  This lets the gate
    backfill tool and any other post-hoc consumer build a consistent
    results dict without re-running evaluation.

    If *duration_seconds* is 0 and a ``metrics.json`` exists, the duration
    is read from ``training_duration_seconds`` in that file.

    The post-training velocity/success panel's numbers (``mean_forward_vel``,
    ``std_forward_vel``, ``mean_distance_traveled``, ``mean_success_rate``)
    are read from ``metrics.json`` too.  When the panel was skipped or failed
    the keys are OMITTED rather than zeroed, so the gate reads them as
    unmeasured; ``best_model_*`` default to ``""`` and can be filled by the
    caller after running ``eval_policy``.

    *sim_dt* is the env's control step, which the summaries multiply episode
    lengths by to print sim time. Pass the env's ``dt``, as
    :func:`generate_stage_artifacts` and :func:`evaluate_stage_checkpoints`
    do; without it the value falls back to ``env_kwargs["sim_dt"]`` or 0.01 s,
    which is wrong for the compsognathus pair (0.02 s).
    """
    import numpy as _np

    stage_dir = Path(stage_dir)
    model_dir = stage_dir / "models"

    # ── Parse evaluations.npz ───────────────────────────────────────────
    eval_npz = stage_dir / "evaluations.npz"
    mean_reward = 0.0
    std_reward = 0.0
    mean_length = 0.0
    std_length = 0.0
    best_eval_reward: float | str = ""
    best_eval_std: float | str = ""
    best_eval_length: float | str = ""
    best_eval_std_length: float | str = ""
    best_eval_timestep: int | str = ""

    if eval_npz.exists():
        eval_data = _np.load(str(eval_npz))
        eval_rewards = eval_data["results"]
        eval_lengths = eval_data["ep_lengths"]
        eval_timesteps = eval_data["timesteps"]

        mean_per_eval = eval_rewards.mean(axis=1)
        best_idx = int(mean_per_eval.argmax())

        best_eval_reward = round(float(mean_per_eval[best_idx]), 2)
        best_eval_std = round(float(eval_rewards[best_idx].std()), 2)
        best_eval_length = round(float(eval_lengths[best_idx].mean()), 1)
        best_eval_std_length = round(float(eval_lengths[best_idx].std()), 1)
        best_eval_timestep = int(eval_timesteps[best_idx])

        # Use last eval as "final" metrics
        mean_reward = float(mean_per_eval[-1])
        std_reward = float(eval_rewards[-1].std())
        mean_length = float(eval_lengths[-1].mean())
        std_length = float(eval_lengths[-1].std())

    # ── Duration and provenance from sidecars ───────────────────────────
    metrics: dict[str, Any] = {}
    metrics_path = stage_dir / "metrics.json"
    if metrics_path.exists():
        metrics = _json.loads(metrics_path.read_text())
        if duration_seconds == 0.0:
            duration_seconds = metrics.get("training_duration_seconds", 0.0)
    # The velocity/success panel (train_base._post_training_eval_panels)
    # writes these for the handoff checkpoint.  They used to be hardcoded to
    # 0.0 here although this file was already open, so every stage-2/3 sweep
    # trial failed its velocity gate at "0.00 m/s" whatever it had learned
    # (review ER4).  A skipped (post_eval_skipped) or failed panel leaves a
    # key absent or null; it is then left OUT so gates._gate_metric reads it
    # as unmeasured rather than as a fabricated zero.  Deferred import: the
    # curriculum package reaches back into reporting at import time.
    from environments.shared.curriculum.gate_schema import finite_gate_metric

    measured_panel = {
        key: value
        for key in ("mean_forward_vel", "std_forward_vel", "mean_distance_traveled", "mean_success_rate")
        if (value := finite_gate_metric(metrics.get(key))) is not None
    }
    plant_identity = metrics.get("plant_identity")
    if not isinstance(plant_identity, Mapping):
        saved_config_path = stage_dir / "stage_config.json"
        if saved_config_path.exists():
            saved_config = _json.loads(saved_config_path.read_text())
            plant_identity = saved_config.get("plant_identity")

    # The SELECTED checkpoint, so the `model_path` this records is the one the
    # replay shows and the evidence CSV is evidence for. It used to hardcode
    # `best_model` while the replay, the stance gate report and the next-stage
    # handoff all resolved through `select_handoff_checkpoint`, which prefers
    # the risk-adjusted `robust_best_model`. This path feeds the gate backfill
    # tool and `generate_stage_artifacts` called without `stage_results`,
    # where nothing else re-derives it.
    #
    # Falls back to `best_model` rather than skipping when no candidate is
    # complete: unlike the replay and the gate report, this function only
    # *describes* a run, and an empty model_path would lose information the
    # caller can still check for itself.
    handoff = select_handoff_checkpoint(model_dir)
    if handoff is None:
        best_model_path = model_dir / "best_model"
        vecnorm_path = str(model_dir / "best_model_vecnorm.pkl")
    else:
        _, selected_path, vecnorm_path = handoff
        best_model_path = Path(selected_path)
    if sim_dt is None:
        sim_dt = stage_config.get("env_kwargs", {}).get("sim_dt", 0.01)

    result = {
        "stage": stage,
        "name": stage_config["name"],
        "description": stage_config["description"],
        "timesteps": timesteps,
        "duration_seconds": duration_seconds,
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "mean_episode_length": mean_length,
        "std_episode_length": std_length,
        **measured_panel,
        "best_eval_reward": best_eval_reward,
        "best_eval_std": best_eval_std,
        "best_eval_length": best_eval_length,
        "best_eval_std_length": best_eval_std_length,
        "best_eval_timestep": best_eval_timestep,
        "sim_dt": sim_dt,
        "model_path": str(best_model_path),
        "vecnorm_path": vecnorm_path,
    }
    if isinstance(plant_identity, Mapping):
        result["plant_identity"] = dict(plant_identity)
    return result


def _write_stance_gate_report(
    *,
    species: str,
    stage: int,
    stage_config: dict[str, Any],
    stage_dir: Path,
    model_dir: Path,
) -> dict[str, Any] | None:
    """Score the selected checkpoint against the stance gate, into the run dir.

    Returns the report dict, or ``None`` when no panel was rolled.  The return
    value is what :func:`_apply_stage_gate` certifies a ``stance_quality/v1``
    stage from, so ``None`` fails that stage closed.

    Only for stages that actually declare ``stance_quality/v1``: rolling a
    40-episode panel costs a few minutes, which is nothing beside a multi-hour
    stage but is pure waste for a stage the criteria do not govern.

    Runs here rather than in the notebook so it happens for every SB3 run
    without anyone remembering, and lands beside
    ``stage_summary.txt`` in the run directory -- which is on Drive, so the
    verdict survives a lost runtime.

    ``stance_report_episodes`` in ``[curriculum]`` overrides the panel size;
    ``0`` skips the report entirely. It exists because "a few minutes" is per
    stage, which a smoke or debug run need not pay (the tuning sweeps,
    retired by D-D17, paid it once per trial). Overriding
    downward makes the bound weaker than the gate claims -- the panel size is
    what its power is specified at -- so the log says so when it happens.

    Deliberately non-fatal to *artifact generation*: losing it must never cost
    a completed training run its summary, replays and graphs, and the
    checkpoint may legitimately be absent (a stage stopped before its first
    evaluation produced one).  It is emphatically fatal to *certification* —
    a stance-gated stage with no panel fails, because the alternative is
    certifying stance quality nobody measured.
    """
    from environments.shared.curriculum.stance_gate import STANCE_GATE_KIND

    curriculum = stage_config.get("curriculum_kwargs", {})
    if curriculum.get("gate_kind") != STANCE_GATE_KIND:
        return None

    declared_episodes = int(curriculum.get("min_eval_episodes", DEFAULT_MIN_EVAL_EPISODES_STANCE))
    report_episodes = curriculum.get("stance_report_episodes")
    report_episodes = declared_episodes if report_episodes is None else int(report_episodes)
    if report_episodes < 1:
        logger.info(
            "Stance gate report skipped for stage %s: stance_report_episodes = %d",
            stage,
            report_episodes,
        )
        return None
    if report_episodes != declared_episodes:
        logger.warning(
            "Stance gate report for stage %s rolls %d episodes, not the stage's min_eval_episodes "
            "%d. The bound's power is specified at the latter; this panel does not certify what "
            "the gate claims.",
            stage,
            report_episodes,
            declared_episodes,
        )

    # The SELECTED checkpoint, through the one selector — the same call the
    # replay and the next-stage handoff make. This used to be a third private
    # copy of the preference order that also passed `vecnorm_path=None` when
    # the statistics were missing, silently scoring the policy on unnormalised
    # observations: a different policy, reported as this one's gate verdict.
    handoff = select_handoff_checkpoint(model_dir)
    if handoff is None:
        logger.warning(
            "Stance gate report skipped for stage %s: no checkpoint in %s has its "
            "matched _vecnorm.pkl, and scoring without the observation statistics "
            "would report a verdict for a different policy.",
            stage,
            model_dir,
        )
        return None

    selected_name, selected_path, selected_vecnorm = handoff
    try:
        from environments.shared.reporting.stance_report import (
            build_stance_gate_report,
            write_stance_gate_report,
        )

        logger.info("Stance gate report scoring stage %s checkpoint: %s", stage, selected_name)
        report = build_stance_gate_report(
            species,
            stage,
            stage_config=stage_config,
            model_path=f"{selected_path}.zip",
            vecnorm_path=selected_vecnorm,
            episodes=report_episodes,
        )
        written = write_stance_gate_report(stage_dir, report)
        logger.info(
            "Stance gate report: %s (duty %.4f, bilateral %.4f) -> %s",
            "PASS" if report["passed"] else "FAIL",
            report["metrics"]["mean_unsupported_duty"],
            report["metrics"]["bilateral_support_duty"],
            written["stance_gate_report_txt"],
        )
        return report
    except Exception:  # noqa: BLE001 - a diagnostic must not sink the run
        logger.warning("Stance gate report failed for stage %s", stage, exc_info=True)
    return None


def _write_task_success_evidence(
    *,
    species_cfg: Any,
    stage: int,
    stage_config: dict[str, Any],
    stage_dir: Path,
    model_dir: Path,
    algorithm: str,
    allow_legacy_plant: bool = False,
) -> "Path | None":
    """Make sure a ``task_success/v1`` stage holds the evidence its gate is judged on.

    The verdict for this kind comes ONLY from ``evaluation_selected.csv`` —
    the selected checkpoint's per-episode task successes, hash-bound to the
    handoff pair — and the notebook writes that file itself
    (``evaluate_stage_checkpoints``).  A caller that rolled no bound panel
    would otherwise record a FAILED verdict for want of evidence (decision
    D-B12).  This is the stance treatment (:func:`_write_stance_gate_report`):
    for every SB3 run, when the file is absent or is not
    bound to the handoff ``select_handoff_checkpoint`` picks NOW, roll
    ``min_eval_episodes`` episodes from that pair on the publication seed
    and write it through ``csv_output.save_evaluation_episodes``.  A file
    already bound to the handoff (the trainer's post-training panel wrote
    it) is kept, so the on-disk verdict and the count ``metrics.json``
    records agree by construction — "bound" meaning the checkpoint digest
    AND, when the rows record one, the VecNormalize sidecar digest (the
    judge refuses a panel rolled under other observation statistics), and
    the file must hold at least ``min_eval_episodes`` rows: a smaller
    trainer panel would leave the stage unjudgeable where a fresh roll
    makes it judgeable.

    ``task_success_panel_episodes`` in ``[curriculum]`` overrides the panel
    size (``0`` skips the roll; a smaller panel is warned about, since the
    bound's power is specified at ``min_eval_episodes`` and the judge then
    refuses it by its own n).  Non-fatal to artifact generation, like the
    stance report: a failed roll leaves no file, and the judge refuses.

    Returns the evidence path, or ``None`` when nothing was written.
    """
    from environments.shared.curriculum.task_success_gate import TASK_SUCCESS_GATE_KIND, read_task_successes

    curriculum = stage_config.get("curriculum_kwargs", {})
    if curriculum.get("gate_kind") != TASK_SUCCESS_GATE_KIND:
        return None
    handoff = select_handoff_checkpoint(model_dir)
    if handoff is None:
        logger.warning(
            "Task-success evidence skipped for stage %s: no checkpoint in %s has its matched _vecnorm.pkl, "
            "and rolling without the observation statistics would record evidence for a different policy.",
            stage,
            model_dir,
        )
        return None
    selected_name, selected_path, selected_vecnorm = handoff
    selected_zip = Path(f"{selected_path}.zip")
    declared = curriculum.get("min_eval_episodes")
    declared_episodes = None if declared is None else int(declared)
    evidence_path = stage_dir / "evaluation_selected.csv"
    if evidence_path.is_file():
        from ..result_bundle.hashing import sha256_file

        try:
            recorded = read_task_successes(evidence_path)
        except (OSError, ValueError):
            recorded = None
        bound = (
            recorded is not None
            and recorded.checkpoint_sha256 == sha256_file(selected_zip)
            and (
                recorded.normalization_sha256 is None or recorded.normalization_sha256 == sha256_file(selected_vecnorm)
            )
        )
        if (
            bound
            and recorded is not None
            and declared_episodes is not None
            and len(recorded.successes) < declared_episodes
        ):
            logger.warning(
                "Stage %s %s is bound to the handoff %s but holds %d episodes, fewer than min_eval_episodes %d; "
                "re-rolling the selected panel at the declared size so the stage can be judged.",
                stage,
                evidence_path.name,
                selected_name,
                len(recorded.successes),
                declared_episodes,
            )
        elif bound:
            logger.info(
                "Task-success evidence for stage %s already bound to %s: %s", stage, selected_name, evidence_path
            )
            return evidence_path
        else:
            logger.warning(
                "Stage %s %s is not bound to the handoff %s; re-rolling the selected panel so the verdict "
                "describes the checkpoint it certifies.",
                stage,
                evidence_path.name,
                selected_name,
            )

    if declared_episodes is None:
        logger.warning("Task-success evidence skipped for stage %s: no min_eval_episodes declared", stage)
        return None
    panel_episodes = curriculum.get("task_success_panel_episodes")
    panel_episodes = declared_episodes if panel_episodes is None else int(panel_episodes)
    if panel_episodes < 1:
        logger.info(
            "Task-success evidence skipped for stage %s: task_success_panel_episodes = %d", stage, panel_episodes
        )
        return None
    if panel_episodes != declared_episodes:
        logger.warning(
            "Task-success evidence for stage %s rolls %d episodes, not the stage's min_eval_episodes %d. "
            "The bound's power is specified at the latter; this panel does not certify what the gate claims.",
            stage,
            panel_episodes,
            declared_episodes,
        )

    try:
        from ..curriculum import load_vecnorm_stats
        from ..evaluation import eval_policy
        from ..plant_contract import current_plant_identity, validate_model_plant
        from ..policy_loading import _ensure_sb3
        from ..train_base import create_vec_env

        sb3 = _ensure_sb3()
        alg_cls = sb3["SAC"] if algorithm == "sac" else sb3["PPO"]
        plant_identity = current_plant_identity(species_cfg.species)
        algo_kwargs = stage_config.get(f"{algorithm}_kwargs", {})
        eval_env = create_vec_env(
            species_cfg,
            {stage: stage_config},
            stage,
            1,
            PUBLICATION_SEED_START,
            algorithm=algorithm,
            gamma=algo_kwargs.get("gamma"),
            plant_identity=plant_identity,
        )
        try:
            # seed=None: a seeded archive would re-seed eval_env with its TRAINING seed on
            # load, and the rows below record PUBLICATION_SEED_START.
            model = load_sb3_model(selected_path, algorithm=alg_cls, env=eval_env, seed=None)
            validate_model_plant(model, plant_identity, artifact=str(selected_zip), allow_legacy=allow_legacy_plant)
            load_vecnorm_stats(
                selected_vecnorm,
                eval_env,
                current_plant=plant_identity,
                allow_legacy_plant=allow_legacy_plant,
            )
            eval_env.training = False
            eval_env.norm_reward = False
            logger.info(
                "Task-success evidence rolling stage %s checkpoint %s (%d episodes)",
                stage,
                selected_name,
                panel_episodes,
            )
            rewards, lengths, fwd_vels, successes, distances = eval_policy(
                model,
                eval_env,
                species_cfg.success_keys,
                n_episodes=panel_episodes,
            )
        finally:
            eval_env.close()
        written = csv_output.save_evaluation_episodes(
            stage_dir,
            rewards=rewards,
            lengths=lengths,
            forward_velocities=fwd_vels,
            distances=distances,
            successes=successes,
            evaluation_seed=PUBLICATION_SEED_START,
            checkpoint_label="selected",
            checkpoint_path=selected_zip,
            normalization_path=selected_vecnorm,
        )
        logger.info(
            "Task-success evidence: %d/%d episodes succeeded -> %s",
            sum(1 for success in successes if success),
            len(successes),
            written,
        )
        return written
    except Exception:  # noqa: BLE001 - evidence rolling must not sink the run; the judge refuses without it
        logger.warning("Task-success evidence could not be rolled for stage %s", stage, exc_info=True)
    return None


def _run_stance_probes(
    *,
    species: str,
    stage: int,
    stage_config: dict[str, Any],
    stage_dir: Path,
    model_dir: Path,
    report: dict[str, Any] | None,
) -> None:
    """Run the stance probe battery against the selected checkpoint.

    Pure diagnostics, so they run LAST in :func:`generate_stage_artifacts` --
    after the gate verdict is recorded and the summary, graphs and replays are
    written. They used to run inside :func:`_write_stance_gate_report`'s
    ``try``, between "report written" and "verdict recorded", which put ~300
    probe episodes of exposure in front of everything a finished run cannot
    afford to lose -- and meant a probe-wiring exception (one the helpers'
    internal handlers cannot see, e.g. a signature drift at a call site) was
    caught by the report's handler and turned an already-written PASS into a
    recorded FAIL.

    *report* is the certification report the probes annotate; ``None`` (no
    panel was measured) means there is nothing to probe. Each probe call
    carries its own handler so a failure costs that probe alone -- including
    caller-side failures, which is what makes the CHANGELOG's "individually
    non-fatal" claim true at this level too.
    """
    if report is None:
        return

    curriculum = stage_config.get("curriculum_kwargs", {})
    declared_episodes = int(curriculum.get("min_eval_episodes", DEFAULT_MIN_EVAL_EPISODES_STANCE))
    report_episodes = curriculum.get("stance_report_episodes")
    report_episodes = declared_episodes if report_episodes is None else int(report_episodes)

    # The same selector the report itself used; re-resolved here so the probes
    # keep describing the one policy every other artifact describes.
    handoff = select_handoff_checkpoint(model_dir)
    if handoff is None:
        logger.warning(
            "Stance probes skipped for stage %s: no checkpoint in %s has its matched _vecnorm.pkl",
            stage,
            model_dir,
        )
        return
    _selected_name, selected_path, selected_vecnorm = handoff

    probe_calls = (
        (
            "filtered action",
            lambda: _write_filtered_action_probe(
                species=species,
                stage=stage,
                stage_config=stage_config,
                stage_dir=stage_dir,
                model_path=f"{selected_path}.zip",
                vecnorm_path=selected_vecnorm,
                episodes=report_episodes,
            ),
        ),
        (
            "constant hold",
            lambda: _write_constant_hold_probe(
                species=species,
                stage=stage,
                stage_config=stage_config,
                stage_dir=stage_dir,
                model_path=f"{selected_path}.zip",
                vecnorm_path=selected_vecnorm,
                episodes=report_episodes,
                measured=report,
            ),
        ),
        (
            "release ablation",
            lambda: _write_constant_hold_ablation(
                species=species,
                stage=stage,
                stage_config=stage_config,
                stage_dir=stage_dir,
                model_path=f"{selected_path}.zip",
                vecnorm_path=selected_vecnorm,
                measured=report,
            ),
        ),
        (
            "impulse",
            lambda: _write_impulse_probe(
                species=species,
                stage=stage,
                stage_config=stage_config,
                stage_dir=stage_dir,
                model_path=f"{selected_path}.zip",
                vecnorm_path=selected_vecnorm,
                settle_steps=int(report["settle_steps"]),
            ),
        ),
    )
    for label, run_probe in probe_calls:
        try:
            run_probe()
        except Exception:  # noqa: BLE001 - a diagnostic must not sink the run
            logger.warning("Stance probe (%s) failed for stage %s", label, stage, exc_info=True)


def _probe_cutoffs(raw: Any) -> list[float]:
    """Normalise ``stance_probe_filter_hz`` to a sorted list of cutoffs.

    Accepts a single number or a list, because the useful reading is a curve
    rather than a point. A single cutoff answers a yes/no that is already
    known to be "no" on this plant -- the checkpoint that PASSED the gate
    falls at every cutoff from 5 to 35 Hz against a 100 Hz control rate
    (issue #491) -- so its PASS/FAIL carries no information. The scalar that
    can actually move is how long the filtered policy survives, and reading
    that against cutoff shows *how much* high-frequency content the policy
    depends on rather than merely that it depends on some.
    """
    values = raw if isinstance(raw, (list, tuple)) else [raw]
    cutoffs: list[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            logger.warning("stance_probe_filter_hz entry is not a number: %r; ignoring it", value)
            continue
        if number > 0:
            cutoffs.append(number)
    return sorted(set(cutoffs))


def _write_filtered_action_probe(
    *,
    species: str,
    stage: int,
    stage_config: dict[str, Any],
    stage_dir: Path,
    model_path: str,
    vecnorm_path: str | None,
    episodes: int,
) -> None:
    """Re-score the selected checkpoint with its actions low-passed, over a sweep.

    Measures how much of its own high-frequency command content the policy
    needs in order to stand. A first-order low-pass between policy and plant
    attenuates the tremor while leaving balance correction -- a ~1.1-1.4 Hz
    phenomenon on this plant -- essentially untouched, so survival under the
    filter is a direct read on whether the tremor is load-bearing.

    That question is settled for today's policies and the answer is "yes"
    (issue #491), which is why this logs a CURVE. Reported per cutoff, the
    survival length is a regression metric: 96 steps at 5 Hz is the current
    T-Rex stage-1 baseline, and a policy reaching the horizon there would be
    one that could survive a real actuator's bandwidth limit.

    Off unless ``stance_probe_filter_hz`` is set. Each cutoff costs a panel,
    so the probe deliberately rolls far fewer episodes than the gate report:
    it certifies nothing, and the effect it measures is enormous (96 steps
    against 1000), so it does not need the sample size the bound's power is
    specified at.

    Writes ``stance_gate_probe_filtered.{txt,json}`` and deliberately NOT
    ``stance_panel_selected.csv``; the probe scored a modified policy and must
    never supply the evidence a bundle is certified from.
    """
    cutoffs = _probe_cutoffs(stage_config.get("curriculum_kwargs", {}).get("stance_probe_filter_hz"))
    if not cutoffs:
        return
    probe_episodes = max(1, min(episodes, _PROBE_EPISODES))
    entries: list[dict[str, Any]] = []
    try:
        from environments.shared.reporting.stance_report import (
            build_stance_gate_report,
            write_action_filter_sweep,
        )

        for cutoff in cutoffs:
            probe = build_stance_gate_report(
                species,
                stage,
                stage_config=stage_config,
                model_path=model_path,
                vecnorm_path=vecnorm_path,
                episodes=probe_episodes,
                filter_actions_hz=cutoff,
            )
            entries.append(probe)
            logger.info(
                "Filtered action probe %.4g Hz: episode length %.1f, full-horizon %.4f, reward %.1f. "
                "MODIFIED policy -- not a gate verdict.",
                cutoff,
                probe["metrics"]["episode_length_mean"],
                probe["metrics"]["full_horizon_fraction"],
                probe["metrics"]["reward_mean"],
            )
    except Exception:  # noqa: BLE001 - a diagnostic must not sink the run
        logger.warning("Filtered action probe failed for stage %s", stage, exc_info=True)
    # Written even if a later cutoff raised: a partial curve is still a curve,
    # and discarding the cutoffs that succeeded would lose the measurement to
    # a failure in one of them.
    if entries:
        try:
            from environments.shared.reporting.stance_report import write_action_filter_sweep

            written = write_action_filter_sweep(stage_dir, entries, probe_episodes=probe_episodes)
            logger.info("Filtered action probe sweep -> %s", written["action_filter_sweep_txt"])
        except Exception:  # noqa: BLE001 - a diagnostic must not sink the run
            logger.warning("Could not write the filtered action probe sweep", exc_info=True)


def _write_constant_hold_probe(
    *,
    species: str,
    stage: int,
    stage_config: dict[str, Any],
    stage_dir: Path,
    model_path: str,
    vecnorm_path: str | None,
    episodes: int,
    measured: dict[str, Any],
) -> None:
    """Re-score the checkpoint with its action frozen to the constant it averages.

    The question the filtered probe leaves open. Low-passing a policy still
    lets it respond, just slowly, so a fall under the filter proves the policy
    needs *bandwidth* without saying whether it needs *feedback*. Cutting the
    feedback outright and commanding the policy's own post-settle mean
    separates the two, and the two have opposite fixes: a pose that stands
    under a constant means the tremor is waste the action penalties failed to
    suppress, while a pose that falls means the tremor is the only thing
    holding the animal up and penalising it harder would be actively wrong.

    Off unless ``stance_probe_hold_constant`` is set. Reuses the gate report's
    already-measured per-actuator DC rather than rolling a measurement panel of
    its own, so the whole probe costs the variant panels and nothing else.

    Writes ``stance_gate_probe_constant.{txt,json}`` and deliberately NOT
    ``stance_panel_selected.csv``.
    """
    curriculum = stage_config.get("curriculum_kwargs", {})
    if not curriculum.get("stance_probe_hold_constant"):
        return
    probe_episodes = max(1, min(episodes, _PROBE_EPISODES))
    entries: list[dict[str, Any]] = []
    try:
        from environments.shared.reporting.stance_report import (
            build_stance_gate_report,
            constant_hold_actions,
            constant_hold_variants,
        )

        hold = constant_hold_actions(measured)
        variants = constant_hold_variants(
            hold,
            settle_steps=int(measured["settle_steps"]),
            horizon=int(measured["horizon"]),
        )
        for variant in variants:
            probe = build_stance_gate_report(
                species,
                stage,
                stage_config=stage_config,
                model_path=model_path,
                vecnorm_path=vecnorm_path,
                episodes=probe_episodes,
                hold_constant=variant,
            )
            entries.append(probe)
            logger.info(
                "Constant-hold probe %s: episode length %.1f, full-horizon %.4f, reward %.1f. "
                "MODIFIED policy -- not a gate verdict.",
                variant.label,
                probe["metrics"]["episode_length_mean"],
                probe["metrics"]["full_horizon_fraction"],
                probe["metrics"]["reward_mean"],
            )
    except Exception:  # noqa: BLE001 - a diagnostic must not sink the run
        logger.warning("Constant-hold probe failed for stage %s", stage, exc_info=True)
    # Written even if a later variant raised, for the same reason the filter
    # sweep is: the variants that succeeded are still a measurement, and the
    # controls are what make the others readable.
    if entries:
        try:
            from environments.shared.reporting.stance_report import write_constant_hold_probe

            written = write_constant_hold_probe(stage_dir, entries, probe_episodes=probe_episodes)
            logger.info("Constant-hold probe -> %s", written["constant_hold_probe_txt"])
        except Exception:  # noqa: BLE001 - a diagnostic must not sink the run
            logger.warning("Could not write the constant-hold probe", exc_info=True)


def _write_constant_hold_ablation(
    *,
    species: str,
    stage: int,
    stage_config: dict[str, Any],
    stage_dir: Path,
    model_path: str,
    vecnorm_path: str | None,
    measured: dict[str, Any],
) -> None:
    """Ablate the held pose one actuator group at a time, into the run dir.

    Answers *which* joints make the pose unholdable, where the constant-hold
    probe only answers *whether* it is. Each group is tested twice -- released
    (is it necessary?) and held alone (is it sufficient?) -- because either
    side alone lets a conspicuous group masquerade as a cause. On the T-Rex
    that is not hypothetical: the tail is the most extreme thing in the DC
    table and is provably inert, while the toes carry the whole effect.

    Off unless ``stance_probe_release_ablation`` is set. Costs 2 panels per
    actuator group plus 3 fixed rows -- 13 on the T-Rex -- so it is the most
    expensive of the three probes and is opt-in per stage.
    """
    curriculum = stage_config.get("curriculum_kwargs", {})
    if not curriculum.get("stance_probe_release_ablation"):
        return
    entries: list[dict[str, Any]] = []
    try:
        from environments.shared.reporting.stance_report import (
            build_stance_gate_report,
            constant_hold_actions,
            constant_hold_release_variants,
        )

        hold = constant_hold_actions(measured)
        for variant in constant_hold_release_variants(hold, measured, horizon=int(measured["horizon"])):
            probe = build_stance_gate_report(
                species,
                stage,
                stage_config=stage_config,
                model_path=model_path,
                vecnorm_path=vecnorm_path,
                episodes=_ABLATION_EPISODES,
                hold_constant=variant,
            )
            entries.append(probe)
            logger.info(
                "Release ablation %s: episode length %.1f, full-horizon %.4f. MODIFIED policy -- not a gate verdict.",
                variant.label,
                probe["metrics"]["episode_length_mean"],
                probe["metrics"]["full_horizon_fraction"],
            )
    except Exception:  # noqa: BLE001 - a diagnostic must not sink the run
        logger.warning("Release ablation failed for stage %s", stage, exc_info=True)
    if entries:
        try:
            from environments.shared.reporting.stance_report import write_constant_hold_ablation

            written = write_constant_hold_ablation(stage_dir, entries, probe_episodes=_ABLATION_EPISODES)
            logger.info("Release ablation -> %s", written["constant_hold_ablation_txt"])
        except Exception:  # noqa: BLE001 - a diagnostic must not sink the run
            logger.warning("Could not write the release ablation", exc_info=True)


def _write_impulse_probe(
    *,
    species: str,
    stage: int,
    stage_config: dict[str, Any],
    stage_dir: Path,
    model_path: str,
    vecnorm_path: str | None,
    settle_steps: int,
) -> None:
    """Shove the animal mid-episode and record whether it recovers.

    The only measurement in the pipeline that tracks *balance* rather than
    *not falling over*. Stage 1 declares no in-episode disturbance, so every
    other criterion -- duty, full-horizon share, reward -- is satisfied by a
    statue, and nothing else can tell a policy that learned to recover from one
    that learned to stand still.

    Rolls the zero-action statue over the same sweep as the control. That is
    not optional: the statue cannot respond to anything, so its survival is the
    plant's *passive* robustness and only the policy's margin above it is
    attributable to control. Reporting the policy's numbers alone would credit
    the plant's own stability to the policy.

    Off unless ``stance_probe_impulse_speeds`` is set. Costs 2 panels per
    (speed, direction) plus the two zero controls -- 14 on the default sweep --
    because the statue side doubles it.
    """
    curriculum = stage_config.get("curriculum_kwargs", {})
    raw = curriculum.get("stance_probe_impulse_speeds")
    if not raw:
        return
    values = raw if isinstance(raw, (list, tuple)) else [raw]
    speeds: list[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            logger.warning("stance_probe_impulse_speeds entry is not a number: %r; ignoring it", value)
            continue
        if number > 0:
            speeds.append(number)
    if not speeds:
        return
    try:
        from environments.shared.reporting.stance_report import (
            build_stance_gate_report,
            impulse_variants,
            write_impulse_probe,
        )

        variants = impulse_variants(speeds, step=settle_steps)

        def _sweep(zero_action: bool) -> list[dict[str, Any]]:
            return [
                build_stance_gate_report(
                    species,
                    stage,
                    stage_config=stage_config,
                    model_path=None if zero_action else model_path,
                    vecnorm_path=None if zero_action else vecnorm_path,
                    zero_action=zero_action,
                    episodes=_IMPULSE_EPISODES,
                    impulse=variant,
                )
                for variant in variants
            ]

        policy_sweep = _sweep(zero_action=False)
        statue_sweep = _sweep(zero_action=True)
        written = write_impulse_probe(stage_dir, policy_sweep, statue_sweep, probe_episodes=_IMPULSE_EPISODES)
        envelopes = {
            row["impulse"]["axis_label"]: row["metrics"]["full_horizon_fraction"]
            for row in policy_sweep
            if row["impulse"]["speed"] > 0
        }
        logger.info(
            "Impulse recovery probe -> %s (policy full-horizon by direction: %s). MODIFIED task -- not a gate verdict.",
            written["impulse_probe_txt"],
            envelopes,
        )
    except Exception:  # noqa: BLE001 - a diagnostic must not sink the run
        logger.warning("Impulse recovery probe failed for stage %s", stage, exc_info=True)


#: What ``generate_stage_artifacts``' verdict records as ``judged_by``
#: (decision D-A5): the evidence-backed post-stage judgement.
GATE_VERDICT_JUDGED_BY = "reporting.stage_artifacts.generate_stage_artifacts"


def _write_stage_gate_verdict(
    *,
    stage_dir: Path,
    species: "str | None",
    stage: "int | str",
    curriculum: Mapping[str, Any],
    passed: bool,
    failures: list[str],
    stage_results: dict[str, Any],
) -> None:
    """Write the per-node ``gate_verdict.json`` beside the handoff pair it judged.

    The handoff resolved here is the same one ``_write_stance_gate_report``
    scored and the notebook's evidence CSVs bind (all through
    ``select_handoff_checkpoint``), so the verdict is hash-bound to the
    checkpoint the evidence describes.  ``stage_id`` is the manifest id when
    the species is known, else the stage reference spelled out.
    """
    from ..curriculum.gate_schema import gate_config_view
    from ..result_bundle import write_gate_verdict
    from ..stage_manifest import StageManifestError, load_stage_manifest
    from .gates import _current_task_sha256

    stage_id = str(stage)
    if species:
        try:
            stage_id = load_stage_manifest(species).resolve(stage).id
        except StageManifestError:
            logger.warning("Stage %s is not in the %s manifest; recording the verdict by reference", stage, species)
    handoff = select_handoff_checkpoint(stage_dir / "models")
    # Recorded as declared: null for a stage that declares no gate kind.
    gate_kind: Any = curriculum.get("gate_kind")
    write_gate_verdict(
        stage_dir,
        species=species or "",
        stage=stage,
        stage_id=stage_id,
        gate_kind=gate_kind,
        gate_schema_version=curriculum.get("gate_schema_version"),
        passed=passed,
        failures=failures,
        task_sha256=_current_task_sha256(stage_dir),
        judged_by=GATE_VERDICT_JUDGED_BY,
        checkpoint=Path(handoff[1] + ".zip") if handoff is not None else None,
        normalization=Path(handoff[2]) if handoff is not None else None,
        # D-A22: the verdict records the gate it is judged under — the block
        # handed in, which is the CURRENT config (the notebook's JUDGE branch
        # judges a directory under the config of the session, not the one
        # its stage_config.json recorded; see _warn_if_judged_under_another_gate).
        gate_config=gate_config_view(curriculum),
        stage_result=stage_results,
    )


def _recorded_curriculum_block(stage_dir: Path) -> "Mapping[str, Any] | None":
    """The ``[curriculum]`` block *stage_dir*'s ``stage_config.json`` recorded, else None.

    Read under either spelling (``curriculum`` as ``save_stage_config``
    writes it, ``curriculum_kwargs`` as the loaded config carries it), the
    way the bundle's evidence reader and the backfill tool read it.  None
    for an absent, unreadable or block-less file.
    """
    path = stage_dir / "stage_config.json"
    if not path.is_file():
        return None
    try:
        record: Any = _json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(record, Mapping):
        return None
    block = record.get("curriculum", record.get("curriculum_kwargs"))
    return block if isinstance(block, Mapping) else None


def _warn_if_judged_under_another_gate(stage: "int | str", stage_dir: Path, curriculum: Mapping[str, Any]) -> None:
    """Decision D-B8: say so when the gate judged under is not the one the stage trained under.

    Re-judging a directory under an edited gate is the intended path (edit
    a threshold, re-judge, never retrain), so this is a WARNING and never a
    refusal: the verdict records the gate it is judged under (D-A22) and
    reuse rule 7 compares against that, not against the recorded block.
    Names the thresholds that differ, so the log says what changed.
    """
    from ..curriculum.gate_schema import gate_config_differences, gate_config_sha256, gate_config_view

    recorded = _recorded_curriculum_block(stage_dir)
    if recorded is None:
        return
    recorded_view = gate_config_view(recorded)
    current_view = gate_config_view(curriculum)
    if gate_config_sha256(recorded_view) == gate_config_sha256(current_view):
        return
    differences = gate_config_differences(recorded_view["thresholds"], current_view)
    logger.warning(
        "Stage %s is judged under a gate configuration that differs from the block its stage_config.json "
        "recorded (%s): the verdict certifies the gate it is judged under (decision D-A22), not the one the "
        "stage trained under, and reuse rule 7 compares against this gate",
        stage,
        "; ".join(differences) if differences else "the gate kind or schema version differs",
    )


def _apply_stage_gate(
    *,
    stage: int,
    stage_config: dict[str, Any],
    stage_results: dict[str, Any],
    stance_report: dict[str, Any] | None,
    stage_dir: "str | Path | None" = None,
    recovery_successes_by_seed: "dict[int, bool] | None" = None,
    species: "str | None" = None,
) -> None:
    """Record this stage's gate verdict onto *stage_results*, in place.

    Runs here, in the one entry point the notebook calls for every node it
    trains or judges,
    because the alternative is what actually happened:
    each caller kept a private checklist, the notebook's drifted out of step
    with ``gate_kind``, and run ``20260802_203215`` recorded
    ``publication_gate_passed = True`` beside a ``GATE: FAIL`` stance report
    and advanced to stage 2 on it.

    Sets ``gate_passed``, ``publication_gate_passed`` and ``gate_failures``;
    callers enforce them.  When *stage_dir* is given the same verdict is
    also written as the stage's ``gate_verdict.json`` (decision D-A5),
    hash-bound to the handoff pair, which is what lets another run reuse the
    node.  Never raises: a stage that cannot be certified is recorded as
    failing, which is the fail-closed reading, and an exception here — the
    verdict file included — would instead cost the run the artifacts written
    around it.
    """
    from .gates import evaluate_stage_gate

    try:
        passed, failures = evaluate_stage_gate(
            stage_config.get("curriculum_kwargs", {}),
            stage_results,
            stage=stage,
            stance_report=stance_report,
            # recovery_quality/v1 judges solely through the stage directory's
            # frozen gate_resolution.json plus the pushed panel's per-seed
            # successes; forwarding both is what makes the frozen gate
            # REACHABLE from this shared entry point (gap review EE2 — before
            # this, every recovery verdict was the fail-closed no-stage_dir
            # refusal, regardless of any frozen resolution on disk).
            stage_dir=stage_dir,
            recovery_successes_by_seed=recovery_successes_by_seed,
        )
    except Exception as exc:  # noqa: BLE001 - the docstring's promise, kept
        # "Never raises" has to be enforced, not asserted. `evaluate_stage_gate`
        # coerces config values (`float(min_avg_reward)`) and iterates a
        # report's `failures`, so a malformed TOML value or a hand-edited
        # report reaches here as TypeError/ValueError. Letting it propagate
        # would cost a completed multi-hour run the artifacts written around
        # this call; recording it as a failure keeps the fail-closed reading
        # AND the artifacts.
        logger.warning("Stage %s curriculum gate could not be evaluated", stage, exc_info=True)
        passed, failures = False, [f"stage {stage} gate evaluation raised {type(exc).__name__}: {exc}"]
    curriculum = stage_config.get("curriculum_kwargs", {})
    if curriculum.get("gate_kind") == "task_success/v1" and stage_dir is not None:
        # The numbers the verdict was judged on travel with it: the count,
        # the panel size and the bound go onto stage_results, hence into
        # gate_verdict.json's stage_result and the summary's stage row
        # (selected_model_success_count / _n_episodes / _success_lcb) —
        # and the same panel's mean reward / length, which the rail and
        # the length floor were judged on, so the verdict describes one
        # panel rather than a bound from the CSV beside a best_eval_reward
        # from the argmax EvalCallback panel.  Read through the same
        # binding the judge used; absent evidence copies nothing, and the
        # verdict above already says why.
        try:
            from .gates import task_success_statistics

            stats, _ = task_success_statistics(Path(stage_dir))
        except Exception:  # noqa: BLE001 - a copy of the numbers must never cost the artifacts
            logger.warning("Stage %s task-success statistics could not be read", stage, exc_info=True)
            stats = None
        if stats:
            for key in (
                "best_model_success_count",
                "best_model_n_episodes",
                "best_model_success_lcb",
                "best_model_reward",
                "best_model_length",
            ):
                if key in stats:
                    stage_results[key] = stats[key]
    # The verdict travels with the gate it was earned under, so a summary
    # re-served after the gate changes can say so instead of re-serving a
    # bare boolean beneath the current gate's description (review SS5).
    stage_results["gate_kind"] = curriculum.get("gate_kind")
    stage_results["gate_schema_version"] = curriculum.get("gate_schema_version")
    stage_results["gate_passed"] = passed
    stage_results["publication_gate_passed"] = passed
    stage_results["gate_failures"] = failures
    if passed:
        logger.info("Stage %s curriculum gate: PASS", stage)
    else:
        logger.warning("Stage %s curriculum gate: FAIL — %s", stage, "; ".join(failures))
    if stage_dir is not None:
        try:
            _warn_if_judged_under_another_gate(stage, Path(stage_dir), curriculum)
        except Exception:  # noqa: BLE001 - a diagnostic must never cost the artifacts
            logger.warning("Stage %s: could not compare the judged gate with the recorded one", stage, exc_info=True)
        try:
            _write_stage_gate_verdict(
                stage_dir=Path(stage_dir),
                species=species,
                stage=stage,
                curriculum=curriculum,
                passed=passed,
                failures=failures,
                stage_results=stage_results,
            )
        except Exception:  # noqa: BLE001 - the verdict file must never cost the artifacts
            logger.warning("Stage %s gate verdict could not be written to %s", stage, stage_dir, exc_info=True)


def evaluate_stage_checkpoints(
    species_cfg,
    stage_config,
    stage,
    algorithm,
    stage_dir,
    *,
    final_path,
    final_vecnorm_path,
    timesteps,
    duration_seconds,
    plant_identity,
    evaluation_seed,
    model=None,
):
    """Evaluate a trained node's final and selected checkpoints and build its stage_results.

    The notebook's evaluation of a node, after ``train_stage`` trains it or
    when the chain loop's JUDGE branch judges a node whose budget the
    RESUME cell spent but whose gate was never judged (no
    ``gate_verdict.json``). ``model`` is the in-memory final model right
    after training; ``None`` loads ``<final_path>.zip`` back, validates its
    plant and takes its cumulative step counter as ``timesteps``. Each
    checkpoint is rolled for 30 episodes on ``evaluation_seed`` under the
    VecNormalize sidecar it is paired with, and each panel is written as
    evidence bound to that checkpoint and sidecar; the handoff is the shared
    selector's (:func:`select_handoff_checkpoint`). Prints the report the
    notebook shows. Moved verbatim from the notebook's infrastructure cell
    with its globals as parameters, and unannotated like it (its ``""``
    placeholders become floats), so mypy leaves its body alone.

    Returns (model, handoff_stem, final_model_path, handoff_vecnorm_path, stage_results).
    """
    import numpy as np

    from ..curriculum import load_vecnorm_stats
    from ..evaluation import eval_policy
    from ..plant_contract import validate_model_plant
    from ..train_base import create_vec_env

    algorithm = algorithm.lower()

    def _eval_forward_vel(model, vecnorm_path, n_episodes):
        """Roll ``model`` on a fresh evaluation env under ``vecnorm_path``'s statistics."""
        eval_env = create_vec_env(
            species_cfg,
            {stage: stage_config},
            stage,
            1,
            evaluation_seed,
            algorithm=algorithm,
            gamma=stage_config.get(f"{algorithm}_kwargs", {}).get("gamma"),
            plant_identity=plant_identity,
        )
        if vecnorm_path and Path(vecnorm_path).exists():
            load_vecnorm_stats(vecnorm_path, eval_env, current_plant=plant_identity)
        eval_env.training = False
        eval_env.norm_reward = False
        try:
            return eval_policy(model, eval_env, species_cfg.success_keys, n_episodes=n_episodes)
        finally:
            eval_env.close()

    stage_dir = Path(stage_dir)
    model_dir = stage_dir / "models"
    final_path = Path(final_path)
    final_vecnorm_path = str(final_vecnorm_path)
    vecnorm_save_path = final_vecnorm_path
    if model is None:
        model = load_sb3_model(str(final_path), algorithm=algorithm)
        validate_model_plant(model, plant_identity, artifact=f"{final_path}.zip")
        timesteps = int(getattr(model, "num_timesteps", 0)) or timesteps

    # Evaluate the final model with its matching VecNormalize stats.
    episode_rewards, episode_lengths, episode_fwd_vels, episode_successes, episode_distances = _eval_forward_vel(
        model,
        final_vecnorm_path,
        n_episodes=30,
    )
    final_eval_path = csv_output.save_evaluation_episodes(
        stage_dir,
        rewards=episode_rewards,
        lengths=episode_lengths,
        forward_velocities=episode_fwd_vels,
        distances=episode_distances,
        successes=episode_successes,
        evaluation_seed=evaluation_seed,
        checkpoint_label="final",
        checkpoint_path=f"{final_path}.zip",
        normalization_path=final_vecnorm_path,
    )
    print(f"Final-model episode evidence saved to: {final_eval_path}")
    mean_reward = float(np.mean(episode_rewards))
    std_reward = float(np.std(episode_rewards))
    mean_length = float(np.mean(episode_lengths))
    std_length = float(np.std(episode_lengths))
    mean_fwd_vel = float(np.mean(episode_fwd_vels))
    std_fwd_vel = float(np.std(episode_fwd_vels))
    mean_distance = float(np.mean(episode_distances))
    # The training envs are closed by now: a bare env of this node's task
    # answers for its control timestep.
    _probe_env = species_cfg.env_class(**stage_config.get("env_kwargs", {}))
    sim_dt = float(_probe_env.dt)
    _probe_env.close()
    mean_success_rate = float(np.mean(episode_successes))
    print(f"Eval (final model): mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"Eval: mean_length={mean_length:.1f} +/- {std_length:.1f} steps ({mean_length * sim_dt:.2f}s sim time)")
    print(f"Eval: mean_forward_vel={mean_fwd_vel:.2f} +/- {std_fwd_vel:.2f} m/s")
    print(f"Eval: mean_distance_traveled={mean_distance:.2f} m")
    print(f"Eval: success_rate={mean_success_rate:.0%}")

    # Build base results dict from on-disk eval data (evaluations.npz),
    # then enrich with the live 30-episode evaluation metrics above.
    stage_results = build_stage_results_from_eval_data(
        stage_dir,
        stage,
        stage_config,
        timesteps=timesteps,
        duration_seconds=duration_seconds,
        sim_dt=sim_dt,
    )
    best_eval_reward = stage_results["best_eval_reward"]
    best_eval_std = stage_results["best_eval_std"]
    best_eval_timestep = stage_results["best_eval_timestep"]
    if best_eval_reward != "":
        print(f"Best model eval:  mean_reward={best_eval_reward} +/- {best_eval_std} (at {best_eval_timestep:,} steps)")

    # Override with richer live-eval metrics
    stage_results.update(
        {
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "mean_episode_length": mean_length,
            "std_episode_length": std_length,
            "mean_forward_vel": mean_fwd_vel,
            "std_forward_vel": std_fwd_vel,
            "mean_distance_traveled": mean_distance,
            "mean_success_rate": mean_success_rate,
            "sim_dt": sim_dt,
        }
    )

    # The SELECTED checkpoint: next-stage loading, the evidence CSV below, the
    # replay video, and the stance gate report must all describe the same
    # policy. `select_handoff_checkpoint` is the one selector they share --
    # the notebook used to carry a private copy of the preference order, which
    # is how the replay ended up showing `best_model` while
    # `evaluation_selected.csv` was evidence for `robust_best_model`.
    #
    # It prefers the risk-adjusted robust_best_model (highest mean - std eval)
    # over SB3's mean-reward best_model: a high mean can be propped up by a good
    # mode while a fat failure tail is already growing (run 20260709_185946).
    # It requires the matched VecNormalize stats, so it cannot return a hybrid.
    _handoff = select_handoff_checkpoint(model_dir)
    if _handoff is None:
        best_model_zip = model_dir / "best_model.zip"
        best_vecnorm_candidate = str(model_dir / "best_model_vecnorm.pkl")
        if best_model_zip.exists():
            raise FileNotFoundError(
                f"Selected checkpoint {best_model_zip} is missing its matched VecNormalize state "
                f"{best_vecnorm_candidate}; refusing to evaluate or export a hybrid checkpoint."
            )
    else:
        _selected_name, _selected_path, best_vecnorm_candidate = _handoff
        best_model_zip = model_dir / f"{_selected_name}.zip"
        print(f"Selected checkpoint: {_selected_name}")
    best_model_reward, best_model_std_reward = "", ""
    best_model_length, best_model_std_length = "", ""
    best_model_fwd_vel, best_model_std_fwd_vel = "", ""
    best_model_distance = ""
    best_model_success_rate = ""
    if best_model_zip.exists():
        best_path = model_dir / best_model_zip.stem
        model = load_sb3_model(str(best_path), algorithm=algorithm)
        validate_model_plant(model, plant_identity, artifact=str(best_model_zip))
        output_path = str(best_path)
        print(f"Loaded best model for next-stage: {best_path}.zip")

        # Guaranteed by select_handoff_checkpoint, which only returns a
        # candidate whose matched statistics exist. Kept as an assertion
        # because exporting a hybrid checkpoint is silent and unrecoverable.
        if not Path(best_vecnorm_candidate).exists():
            raise FileNotFoundError(
                f"Selected checkpoint {best_model_zip} is missing its matched VecNormalize state "
                f"{best_vecnorm_candidate}; refusing to evaluate or export a hybrid checkpoint."
            )
        vecnorm_save_path = best_vecnorm_candidate
        print(f"Using matched VecNormalize for selected checkpoint: {vecnorm_save_path}")

        # Evaluate the best model over 30 episodes
        print("Evaluating best model (30 episodes)...")
        bm_rewards, bm_lengths, bm_fwd_vels, bm_successes, bm_distances = _eval_forward_vel(
            model,
            vecnorm_save_path,
            n_episodes=30,
        )
        selected_eval_path = csv_output.save_evaluation_episodes(
            stage_dir,
            rewards=bm_rewards,
            lengths=bm_lengths,
            forward_velocities=bm_fwd_vels,
            distances=bm_distances,
            successes=bm_successes,
            evaluation_seed=evaluation_seed,
            checkpoint_label="selected",
            checkpoint_path=best_model_zip,
            normalization_path=vecnorm_save_path,
        )
        print(f"Selected-model episode evidence saved to: {selected_eval_path}")
        best_model_reward = round(float(np.mean(bm_rewards)), 2)
        best_model_std_reward = round(float(np.std(bm_rewards)), 2)
        best_model_length = round(float(np.mean(bm_lengths)), 1)
        best_model_std_length = round(float(np.std(bm_lengths)), 1)
        best_model_fwd_vel = round(float(np.mean(bm_fwd_vels)), 2)
        best_model_std_fwd_vel = round(float(np.std(bm_fwd_vels)), 2)
        best_model_distance = round(float(np.mean(bm_distances)), 2)
        best_model_success_rate = round(float(np.mean(bm_successes)), 2)
        print(f"Best model eval:  mean_reward={best_model_reward} +/- {best_model_std_reward}")
        print(f"Best model eval:  mean_length={best_model_length} +/- {best_model_std_length}")
        print(f"Best model eval:  mean_fwd_vel={best_model_fwd_vel} +/- {best_model_std_fwd_vel} m/s")
        print(f"Best model eval:  mean_distance={best_model_distance} m")
        print(f"Best model eval:  success_rate={best_model_success_rate:.0%}")
    else:
        output_path = str(final_path)
        selected_eval_path = csv_output.save_evaluation_episodes(
            stage_dir,
            rewards=episode_rewards,
            lengths=episode_lengths,
            forward_velocities=episode_fwd_vels,
            distances=episode_distances,
            successes=episode_successes,
            evaluation_seed=evaluation_seed,
            checkpoint_label="selected",
            checkpoint_path=f"{final_path}.zip",
            normalization_path=final_vecnorm_path,
        )
        print(f"Selected-model episode evidence saved to: {selected_eval_path}")
        best_model_reward = round(float(np.mean(episode_rewards)), 2)
        best_model_std_reward = round(float(np.std(episode_rewards)), 2)
        best_model_length = round(float(np.mean(episode_lengths)), 1)
        best_model_std_length = round(float(np.std(episode_lengths)), 1)
        best_model_fwd_vel = round(float(np.mean(episode_fwd_vels)), 3)
        best_model_std_fwd_vel = round(float(np.std(episode_fwd_vels)), 3)
        best_model_distance = round(float(np.mean(episode_distances)), 3)
        best_model_success_rate = round(float(np.mean(episode_successes)), 4)

    # The curriculum gate is NOT evaluated here: `generate_stage_artifacts`
    # judges it through the one shared `reporting.gates.evaluate_stage_gate`
    # and records `gate_passed` / `publication_gate_passed` / `gate_failures`
    # onto this dict; the chain loop enforces it. The notebook's own checklist
    # knew nothing about `gate_kind`, so when Tyrannosaurus Rex stage 1 moved
    # to stance_quality/v1 it quietly degraded to a reward comparison, which
    # the zero-action statue clears by 68% (run 20260802_203215 advanced to
    # stage 2 beside a stance_gate_report.txt reading GATE: FAIL).

    # Update stage_results with model paths and best-model eval
    stage_results.update(
        {
            "model_path": output_path,
            "final_model_path": str(final_path),
            "vecnorm_path": vecnorm_save_path,
            "final_vecnorm_path": final_vecnorm_path,
            "best_model_reward": best_model_reward,
            "best_model_std_reward": best_model_std_reward,
            "best_model_length": best_model_length,
            "best_model_std_length": best_model_std_length,
            "best_model_fwd_vel": best_model_fwd_vel,
            "best_model_std_fwd_vel": best_model_std_fwd_vel,
            "best_model_distance": best_model_distance,
            "best_model_success_rate": best_model_success_rate,
        }
    )

    return model, output_path, str(final_path), vecnorm_save_path, stage_results


def generate_stage_artifacts(
    species_cfg,
    stage_config: dict[str, Any],
    stage: int,
    algorithm: str,
    stage_dir: "str | Path",
    seed: int,
    stage_results: dict[str, Any] | None = None,
    timesteps: int = 0,
    record_videos: bool = True,
    generate_graphs: bool = True,
    allow_legacy_plant: bool = False,
    recovery_successes_by_seed: "dict[int, bool] | None" = None,
) -> dict[str, Any]:
    """Write stage summary, record replay videos, and generate training graphs.

    This is the single shared entry-point for generating post-training
    artifacts.  The training notebook calls it for every node it trains or
    judges, so the artifacts are always consistent.

    When *stage_results* is ``None``, a results dict is built from on-disk
    eval data via :func:`build_stage_results_from_eval_data`.  Callers
    that already have richer metrics (e.g. the notebook, which runs a
    full 30-episode eval) should pass their own *stage_results*.

    When *generate_graphs* is ``True`` (the default), training curves and
    diagnostic graphs are saved to the stage directory.  Requires
    ``matplotlib``.

    For a ``recovery_quality/v1`` stage, pass *recovery_successes_by_seed*
    (``RecoveryPanelEvidence.successes_by_seed()`` from the post-training
    pushed panel); the gate is judged against the stage directory's frozen
    ``gate_resolution.json`` and refuses without both.

    Also evaluates the stage's declared curriculum gate and records the
    verdict onto *stage_results* as ``gate_passed`` /
    ``publication_gate_passed`` / ``gate_failures``.  It happens here, not in
    each caller, so no trainer can advance on a checklist that has drifted
    away from ``gate_kind`` — see :func:`_apply_stage_gate`.  Callers must
    enforce the verdict; this function records it.

    Returns the (possibly enriched) *stage_results* dict.
    """
    stage_dir = Path(stage_dir)
    model_dir = stage_dir / "models"
    species = species_cfg.species

    if stage_results is None:
        # The summaries print episode length × sim_dt as sim time, so take the
        # node's own control step from a bare env of its task (the compsognathus
        # pair steps at 0.02 s, not the 0.01 s default).
        probe_env = species_cfg.env_class(**stage_config.get("env_kwargs", {}))
        try:
            sim_dt = float(probe_env.dt)
        finally:
            probe_env.close()
        stage_results = build_stage_results_from_eval_data(
            stage_dir,
            stage,
            stage_config,
            timesteps=timesteps,
            sim_dt=sim_dt,
        )

    stance_report = _write_stance_gate_report(
        species=species,
        stage=stage,
        stage_config=stage_config,
        stage_dir=stage_dir,
        model_dir=model_dir,
    )
    # A task_success/v1 stage is judged from evaluation_selected.csv; make
    # sure the directory holds one bound to the handoff before the gate
    # (a caller that rolled no bound panel wrote none — decision D-B12).
    _write_task_success_evidence(
        species_cfg=species_cfg,
        stage=stage,
        stage_config=stage_config,
        stage_dir=stage_dir,
        model_dir=model_dir,
        algorithm=algorithm,
        allow_legacy_plant=allow_legacy_plant,
    )
    # Before the replays and graphs below, which are best-effort and can be
    # skipped: the verdict must not depend on whether matplotlib imported.
    # This also writes the stage's gate_verdict.json (species names the id).
    _apply_stage_gate(
        stage=stage,
        stage_config=stage_config,
        stage_results=stage_results,
        stance_report=stance_report,
        stage_dir=stage_dir,
        recovery_successes_by_seed=recovery_successes_by_seed,
        species=species,
    )

    # After the gate, not before it: the summary now states the verdict and
    # every criterion that failed, and writing it first would print a stage
    # summary that says nothing about the decision the stage turns on.
    text_summaries.write_stage_summary(stage_dir, stage_results, species, algorithm)
    logger.info("Stage summary written to: %s", stage_dir / "stage_summary.txt")

    # Figures and replays render into local scratch and publish to the stage
    # directory in one pass on exit.  The plots and videos below read their
    # inputs from `stage_dir` (evaluations.npz, diagnostics.npz, models/) and
    # only their *outputs* are staged — see stage_layout.staged_artifacts for
    # why writing an mp4 straight onto a Drive mount is the expensive part.
    with stage_layout.staged_artifacts(stage_dir) as staging:
        figures_out = stage_layout.figures_dir(staging, create=True)
        replays_out = stage_layout.replays_dir(staging, create=True)

        # ── Generate training graphs ────────────────────────────────────
        if generate_graphs:
            try:
                from environments.shared.visualization import (
                    plot_diagnostics_graphs,
                    plot_foot_contacts,
                    plot_stance_diagnostics,
                    plot_training_curves,
                )

                stage_dirs = [(stage, stage_dir)]
                stage_configs: dict[int | str, dict[str, Any]] = {stage: stage_config}

                plot_training_curves(
                    stage_dirs,
                    stage_configs,
                    species,
                    algorithm,
                    save_path=figures_out / "training_curves.png",
                    show=False,
                )
                plot_diagnostics_graphs(
                    stage_dirs,
                    stage_configs,
                    species,
                    algorithm,
                    save_dir=figures_out,
                    show=False,
                )
                plot_foot_contacts(
                    stage_dirs,
                    stage_configs,
                    species,
                    algorithm,
                    save_path=figures_out / "foot_contacts.png",
                    show=False,
                )
                plot_stance_diagnostics(
                    stage_dirs,
                    stage_configs,
                    species,
                    algorithm,
                    save_path=figures_out / "stance_diagnostics.png",
                    show=False,
                )
            except ImportError:
                logger.warning("Skipping graph generation (matplotlib not installed).")
            except Exception:
                logger.warning("Graph generation failed.", exc_info=True)

        if record_videos:
            stage_results = _record_stage_replays(
                species_cfg=species_cfg,
                stage_config=stage_config,
                stage=stage,
                algorithm=algorithm,
                stage_dir=stage_dir,
                replays_out=replays_out,
                model_dir=model_dir,
                seed=seed,
                stage_results=stage_results,
                allow_legacy_plant=allow_legacy_plant,
            )

    # Probes run dead last, outside the staging context: they are the most
    # expensive artifact step (~300 episodes at the trex stage-1 settings) and
    # nothing downstream reads them, so a runtime lost mid-probe costs the
    # probes alone -- never the recorded verdict, summary, graphs or replays
    # above. Ordering is pinned by the wiring tests.
    _run_stance_probes(
        species=species,
        stage=stage,
        stage_config=stage_config,
        stage_dir=stage_dir,
        model_dir=model_dir,
        report=stance_report,
    )

    return stage_results


def _record_stage_replays(
    *,
    species_cfg,
    stage_config: dict[str, Any],
    stage: int,
    algorithm: str,
    stage_dir: Path,
    replays_out: Path,
    model_dir: Path,
    seed: int,
    stage_results: dict[str, Any],
    allow_legacy_plant: bool,
) -> dict[str, Any]:
    """Record the best and final replays into *replays_out*.

    Split out of :func:`generate_stage_artifacts` only so the staging
    context there stays readable; the behaviour is unchanged apart from
    the videos landing under ``replays/`` instead of the stage root.
    """
    species = species_cfg.species

    # ── Record replay videos for the selected and final checkpoints ──────
    from ..plant_contract import PlantCompatibilityError, current_plant_identity, validate_model_plant
    from ..stage_manifest import stage_label

    try:
        from environments.shared.evaluation import TREX_STAGE1_CAMERA_VIEWS, record_stage_video
        from environments.shared.policy_loading import _ensure_sb3

        sb3 = _ensure_sb3()
        env_kwargs = stage_config["env_kwargs"].copy()
        alg_cls = sb3["SAC"] if algorithm == "sac" else sb3["PPO"]
        plant_identity = current_plant_identity(species)

        final_path = model_dir / f"{stage_label(stage)}_final"
        final_vecnorm_path = str(final_path) + "_vecnorm.pkl"
        # Recovery is stance plus scheduled pushes, so its replays carry the
        # same side/front camera views and per-frame stance CSV — the side
        # view is where a shove and the response are actually visible.
        replay_diagnostics = species.lower() == "trex" and stage in (1, "recovery")
        replay_camera_views = TREX_STAGE1_CAMERA_VIEWS if replay_diagnostics else None

        # The SELECTED checkpoint, via the same selector that decides the
        # next-stage handoff and that `evaluation_selected.csv` is evidence
        # for. This used to hardcode `best_model` and label the replay
        # "best", while the selector prefers the risk-adjusted
        # `robust_best_model` — so on any run where both exist (they
        # normally do) the video and the evidence CSV in the same folder
        # described DIFFERENT POLICIES, with nothing in either name saying
        # so. `stance_gate_report` picks the risk-adjusted one too. One
        # selector, one label.
        #
        # Returning None means no candidate has its matched VecNormalize
        # statistics. Recording anyway would replay the policy on
        # unnormalised observations — a different policy again — so the
        # replay is skipped and says why, rather than producing footage
        # that misrepresents the checkpoint.
        handoff = select_handoff_checkpoint(model_dir)
        if handoff is None:
            logger.warning(
                "Stage %s selected-checkpoint replay skipped: neither robust_best_model nor "
                "best_model in %s has its matched _vecnorm.pkl, and replaying without the "
                "observation statistics would show a different policy.",
                stage,
                model_dir,
            )
        else:
            selected_name, selected_path, selected_vecnorm = handoff
            logger.info("Stage %s selected checkpoint for replay: %s", stage, selected_name)
            selected_model = load_sb3_model(selected_path, algorithm=alg_cls)
            validate_model_plant(
                selected_model,
                plant_identity,
                artifact=f"{selected_path}.zip",
                allow_legacy=allow_legacy_plant,
            )
            record_stage_video(
                selected_model,
                env_class=species_cfg.env_class,
                env_kwargs=env_kwargs,
                stage=stage,
                stage_dir=stage_dir,
                output_dir=replays_out,
                species=species,
                algorithm=algorithm,
                seed=seed,
                vecnorm_path=selected_vecnorm,
                label="selected",
                plant_identity=plant_identity,
                allow_legacy_plant=allow_legacy_plant,
                camera_views=replay_camera_views,
                collect_stance_diagnostics=replay_diagnostics,
            )

        if (Path(str(final_path) + ".zip")).exists():
            final_model = load_sb3_model(str(final_path), algorithm=alg_cls)
            validate_model_plant(
                final_model,
                plant_identity,
                artifact=str(final_path) + ".zip",
                allow_legacy=allow_legacy_plant,
            )
            record_stage_video(
                final_model,
                env_class=species_cfg.env_class,
                env_kwargs=env_kwargs,
                stage=stage,
                stage_dir=stage_dir,
                output_dir=replays_out,
                species=species,
                algorithm=algorithm,
                seed=seed,
                vecnorm_path=final_vecnorm_path,
                label="final",
                plant_identity=plant_identity,
                allow_legacy_plant=allow_legacy_plant,
                camera_views=replay_camera_views,
                collect_stance_diagnostics=replay_diagnostics,
            )
    except PlantCompatibilityError:
        raise
    except Exception:
        logger.warning("Video recording failed.", exc_info=True)

    return stage_results
