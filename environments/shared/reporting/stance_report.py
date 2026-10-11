"""Build, render and write the stance gate report for a checkpoint (``stance_quality/v1``, ``/v2`` and the heading certificate).

The library half of ``environments/shared/scripts/stance_gate_report.py``:
everything that rolls a policy on the stage's own configured environment,
reduces the episodes with the same
:mod:`~environments.shared.curriculum.stance_gate` code the training loop
runs, and renders the verdict criterion by criterion -- plus the three probes
built on the same rollout (the low-pass action filter, the constant-hold
family and the root impulse) and the per-episode panel evidence.  The script
keeps only its CLI and re-imports these names.

A stage that declares ``stance_quality/v2`` (decision D-D23) is scored by
:mod:`~environments.shared.curriculum.stance_gate_v2` instead: the same
rollout with the floor-truth recorder (``environments.shared.gait``)
attached, a statue panel on the same seeds when the gate's statue-relative
criteria need one, and a ``mesozoic.stance-gate-report/v3`` report that
records what scored it, under which thresholds and measurement, for which
handoff pair -- everything ``reporting.gates`` checks before it re-derives
the verdict.  Probes on such a stage stay v1-shaped and unrecorded.

A stage that declares ``stance_quality_heading/v1`` (decision D-D32) is
certified by :func:`build_stance_heading_report` instead: the v2 panel and
its statue panel rolled at each heading of the fixed whole-scene grid
(:data:`~environments.shared.curriculum.stance_heading_gate.HEADING_GRID_DEG`),
scored by :mod:`~environments.shared.curriculum.stance_heading_gate` into a
``mesozoic.stance-heading-report/v1`` certificate and its
``stance_heading_panel_selected.csv``.  Such a stage writes no
``stance_gate_report.*``; its probes annotate the certificate's 0 degree
cell (:func:`heading_zero_panel`).

It lives in the reporting package because the training pipeline emits the
same report automatically: :mod:`~environments.shared.reporting.stage_artifacts`
calls :func:`build_stance_gate_report` and the probe writers for every
stance-gated stage, and a library must not import from a script -- the
scripts tree sits outside the package boundary, and until cleanup CU-9 it
sat outside coverage too, so the code the certification path runs was
hidden from the coverage gate.

Why one implementation rather than a notebook cell: the gate's verdict is the
thing runs are judged on, and a verdict computed by hand-rolled
reimplementation is not the same verdict.  Everything here funnels through
``stance_panel_from_episode_duties`` and ``evaluate_stance_gate``, so what
this reports is what the curriculum would decide.

Deliberately not re-exported from :mod:`environments.shared.reporting`: the
species registry imported here pulls ``train_base`` at module level, which
every importer of the reporting package would then pay for.  Import this
module by its full path, lazily where the importer must stay light.
"""

from __future__ import annotations

import logging
import math
import textwrap
from collections.abc import Mapping
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from ..action_filter import low_pass_alpha
from ..config import read_recorded_stage_config
from ..constants import PUBLICATION_SEED_START
from ..curriculum.gate_schema import STANCE_GATE_KINDS, gate_config_sha256, gate_config_view
from ..curriculum.stance_gate import (
    STANCE_GATE_KIND,
    StanceGateThresholds,
    evaluate_stance_gate,
    stance_panel_from_episode_duties,
)
from ..curriculum.stance_gate_v2 import (
    EPISODE_CRITERIA,
    HOP_OR_FALL_KEYS,
    HORIZON_REASON,
    STANCE_GATE_V2_KIND,
    STANCE_V2_PANEL_FIELDNAMES,
    STANCE_V2_REPORT_SCHEMA,
    STATUE_POLICY,
    StanceV2Thresholds,
    StatueReference,
    evaluate_stance_v2_gate,
    statue_reference,
)
from ..curriculum.stance_heading_gate import (
    FROZEN_VECNORMALIZE_MODE,
    HEADING_GRID_DEG,
    HEADING_PROTOCOL,
    POLICY_ROLE,
    STANCE_HEADING_GATE_KIND,
    STANCE_HEADING_PANEL_CSV,
    STANCE_HEADING_PANEL_FIELDNAMES,
    STANCE_HEADING_REPORT_JSON,
    STANCE_HEADING_REPORT_SCHEMA,
    STANCE_HEADING_REPORT_TXT,
    STATUE_ROLE,
    HeadingEpisode,
    evaluate_stance_heading_gate,
    heading_grid_cell,
    validate_heading_grid,
)
from ..file_io import atomic_write_csv, atomic_write_json, atomic_write_text, read_json_object
from ..gait.stance_metrics import (
    STANCE_METRIC_FIELDS,
    STANCE_METRIC_FOOT_FIELDS,
    StanceEpisodeMetrics,
    episode_stance_metrics,
)
from ..policy_loading import PolicyLoadError, load_sb3_checkpoint
from ..species_names import species_display_name
from ..species_registry import SPECIES_FACTORIES
from ..stage_manifest import stage_display_name
from ..stance_diagnostics import derive_stance_info

logger = logging.getLogger(__name__)


class StanceGateReportError(RuntimeError):
    """The report could not be produced for a reason the caller should see.

    Deliberately a plain ``Exception`` subclass rather than ``SystemExit``,
    which is what these paths used to raise.  ``SystemExit`` derives from
    ``BaseException``, so it sailed straight through the ``except Exception``
    guard in ``reporting.stage_artifacts._write_stance_gate_report`` that
    exists precisely so a diagnostic cannot sink a finished training run --
    a truncated VecNormalize ``.pkl`` aborted artifact generation before the
    graphs and videos were written.  ``main`` converts this to ``SystemExit``
    for CLI use, which is where that behaviour belongs.
    """


def _load_policy(
    model_path: str,
    vecnorm_path: str | None,
    env_factory: Any,
    *,
    plant_identity: Any = None,
    allow_legacy_plant: bool = False,
):
    """Return ``(predict_fn, describe)`` for a saved SB3 checkpoint.

    Observation normalisation is applied from the saved ``VecNormalize``
    statistics rather than re-estimated: a policy evaluated on unnormalised
    observations is a *different policy*, and would score arbitrarily badly
    for reasons that have nothing to do with stance.  That is why a missing
    or unreadable file is fatal here instead of a warning.

    Loaded through the shared :func:`~environments.shared.policy_loading.load_sb3_checkpoint`
    -- the same ``VecNormalize.load``-over-a-throwaway-``DummyVecEnv`` path
    the sibling report scripts use -- with no sidecar guessing: the caller
    names the statistics (the artifact path hands over the handoff
    checkpoint's own), and ``None`` is honoured as a deliberate, labelled
    unnormalised run.  Both artifacts are checked against *plant_identity*
    as they load.
    """
    try:
        model, normalizer, vecnorm_path = load_sb3_checkpoint(
            model_path,
            vecnorm_path,
            env_factory,
            guess_sidecar=False,
            allow_unnormalized=True,
            plant_identity=plant_identity,
            allow_legacy_plant=allow_legacy_plant,
        )
    except PolicyLoadError as exc:
        # Same message, this module's exception: it is the type callers and
        # tests pin, and like PolicyLoadError it is an Exception rather than a
        # SystemExit, so the artifact guard in reporting.stage_artifacts sees it.
        raise StanceGateReportError(str(exc)) from exc

    if normalizer is None:

        def unnormalised(obs: np.ndarray) -> np.ndarray:
            return np.asarray(model.predict(obs, deterministic=True)[0])

        # The record a heading certificate binds (D-D32): its judge admits only the frozen statistics below.
        unnormalised.normalization = {"mode": "none"}  # type: ignore[attr-defined]
        return unnormalised, "no obs normalisation"

    obs_rms = normalizer.obs_rms
    if isinstance(obs_rms, dict):
        # SB3 keeps a per-key RunningMeanStd for Dict observation spaces. Every
        # species here exposes a flat Box, so a dict means the checkpoint was
        # trained against a different observation contract than the stage
        # config builds — normalising with the wrong statistics would silently
        # score a different policy.
        raise StanceGateReportError(
            f"{vecnorm_path} holds per-key statistics for a Dict observation space "
            f"(keys: {sorted(obs_rms)}), but this stage builds a flat Box observation."
        )
    mean = np.asarray(obs_rms.mean, dtype=np.float64)
    var = np.asarray(obs_rms.var, dtype=np.float64)
    epsilon = float(normalizer.epsilon)
    clip = float(normalizer.clip_obs)

    def predict(obs: np.ndarray) -> np.ndarray:
        normalized = np.clip((obs - mean) / np.sqrt(var + epsilon), -clip, clip).astype(np.float32)
        # np.asarray, not a bare return: SB3's predict() is untyped when
        # stable-baselines3 is absent (as it is in the lint job), so the value
        # is Any there and typed here. Coercing makes the annotation true in
        # both environments instead of only the one that happens to run mypy.
        return np.asarray(model.predict(normalized, deterministic=True)[0])

    # What the heading certificate records as its normalization (D-D32): the saved statistics, applied frozen.
    predict.normalization = {"mode": FROZEN_VECNORMALIZE_MODE, "clip_obs": clip, "epsilon": epsilon}  # type: ignore[attr-defined]
    return predict, f"VecNormalize stats applied (clip {clip:g})"


def run_panel(
    species: str,
    *,
    predict: Any,
    episodes: int,
    seed: int,
    settle_steps: int,
    horizon: int,
    env_kwargs: dict[str, Any],
    plant_identity: Any = None,
    control_dt: float = 0.01,
    impulse: "RootImpulse | None" = None,
    floor_truth: bool = False,
    spawn_yaw: "SpawnYaw | None" = None,
) -> dict[str, Any]:
    """Roll ``episodes`` deterministic episodes and reduce them for the gate.

    The rollout environment is validated against *plant_identity* when one is
    supplied, and always closed -- a MuJoCo env holds native handles, and the
    artifact path builds one of these per stage.

    Returns the panel plus the per-episode evidence it was reduced from, so a
    caller can serialize the measurements rather than only their summary.

    ``floor_truth=True`` (the ``stance_quality/v2`` panel) also attaches a
    :class:`~environments.shared.gait.recorder.SubstepContactRecorder` to the
    rollout env for the whole panel -- attached after the env is built and
    detached in the same ``finally`` that closes it, so a rollout that raises
    never leaves the hook installed -- and adds ``stance_metrics`` (one
    :class:`~environments.shared.gait.stance_metrics.StanceEpisodeMetrics`
    per episode, measured with ``settle_steps``) and ``measurement`` (the
    :func:`~environments.shared.gait.constants.measurement_manifest` the
    metrics mean what they mean under) to the result.  The recorder reads
    the physics and never writes it (the trajectory is bit-identical,
    ``test_gait_recorder.py``), so every other number in the result is the
    one an unrecorded panel would produce.  Without it the rollout is
    exactly the v1 one.
    """
    env_class = SPECIES_FACTORIES[species]().env_class
    env = env_class(**env_kwargs)
    if plant_identity is not None:
        from environments.shared.plant_contract import validate_environment_plant

        try:
            validate_environment_plant(env, plant_identity, artifact=f"{species} stance gate report environment")
        except BaseException:
            env.close()
            raise

    lengths: list[float] = []
    rewards: list[float] = []
    duties: list[float] = []
    bilateral_duties: list[float] = []
    single_duties: list[float] = []
    terminations: dict[str, int] = {}
    # Per-term totals, so a failing panel can be read for *why*. A policy that
    # keeps paying the airborne penalty is being funded by some other term;
    # which one is not guessable from the aggregate score.
    components: dict[str, float] = {}
    # Per-actuator action statistics. Which actuators carry a static offset
    # decides whether a leg-pose weight can reach it at all: `leg_home_pose`
    # covers only the leg joints, so a displacement living in the tail or neck
    # is invisible to it and unfixable by it (issue #489).
    action_stats = _new_action_stats()
    joint_names: list[str] = []
    pose_mapping: list[dict[str, Any]] = []
    recorder: Any = None
    measurement: dict[str, Any] | None = None
    stance_metrics: list[StanceEpisodeMetrics] = []
    yaw_changes: list[float] = []

    try:
        joint_names = _actuator_joint_names(env)
        pose_mapping = _actuator_pose_mapping(env)
        if floor_truth:
            # Imported here: the recorder and the morphology import mujoco
            # and walk the model, which only a floor-truth panel needs.
            from ..gait.constants import measurement_manifest
            from ..gait.morphology import Morphology
            from ..gait.recorder import SubstepContactRecorder

            morphology = Morphology.from_env(env, species)
            measurement = measurement_manifest(morphology, env.unwrapped.model)
            recorder = SubstepContactRecorder(env, morphology).attach()
        _roll_episodes(
            env,
            predict=predict,
            episodes=episodes,
            seed=seed,
            settle_steps=settle_steps,
            lengths=lengths,
            rewards=rewards,
            duties=duties,
            bilateral_duties=bilateral_duties,
            single_duties=single_duties,
            terminations=terminations,
            components=components,
            action_stats=action_stats,
            impulse=impulse,
            recorder=recorder,
            stance_metrics=stance_metrics,
            spawn_yaw=spawn_yaw,
            # Only the heading probe reads the yaw; every other panel rolls exactly as before.
            yaw_changes=yaw_changes if spawn_yaw is not None else None,
        )
    finally:
        try:
            if recorder is not None:
                recorder.detach()
        finally:
            env.close()

    panel = stance_panel_from_episode_duties(
        episode_lengths=lengths,
        episode_duties=duties,
        episode_rewards=rewards,
        horizon=horizon,
    )

    def _full_horizon_mean(values: list[float]) -> float:
        """Mean over the same episodes the gate measures duty on.

        The ungated shares used to be averaged over every episode that
        measured anything, while the gated duty uses full-horizon episodes
        only. The three then did not sum to 1 whenever an episode failed
        early -- measured 0.90 with 5 failures in 40 -- which is exactly the
        identity the report cites as the reason for printing bilateral at
        all. Worse, the drift ran in the direction that hides the problem:
        the flailing episodes were folded into bilateral and single but
        excluded from unsupported.
        """
        kept = [value for value, length in zip(values, lengths) if length >= horizon and not np.isnan(value)]
        return float(np.mean(kept)) if kept else float("nan")

    result: dict[str, Any] = {
        "panel": panel,
        "lengths": np.asarray(lengths),
        "rewards": np.asarray(rewards),
        "terminations": terminations,
        "components": {key: value / episodes for key, value in components.items()},
        "bilateral_duty": _full_horizon_mean(bilateral_duties),
        "single_duty": _full_horizon_mean(single_duties),
        "action": _reduce_action_stats(action_stats, joint_names, control_dt, pose_mapping),
        # The measurements the panel was reduced from. Kept rather than
        # discarded so the report can serialize the evidence, not only its
        # summary: an aggregate cannot be re-checked, and duty is the one
        # column the publication evidence CSV does not carry.
        "episodes": [
            {
                "episode": index,
                "seed": seed + index,
                "length": length,
                "reward": reward,
                "reached_horizon": length >= horizon,
                "unsupported_duty": duty,
                "bilateral_support_duty": bilateral,
                "single_support_duty": single,
            }
            for index, (length, reward, duty, bilateral, single) in enumerate(
                zip(lengths, rewards, duties, bilateral_duties, single_duties)
            )
        ],
    }
    if floor_truth:
        result["stance_metrics"] = stance_metrics
        result["measurement"] = measurement
    if spawn_yaw is not None:
        result["yaw_changes"] = yaw_changes
    return result


def _new_action_stats() -> dict[str, Any]:
    return {
        "sum": None,
        "sq_sum": None,
        "count": 0,
        # Physical controls actually applied by the environment.  These stay
        # separate from normalized actions because the species mapping can be
        # piecewise about its home control; in that case E[f(a)] != f(E[a])
        # and a normalized variance cannot be converted with one range-wide
        # scale factor.
        "ctrl_sum": None,
        "ctrl_sq_sum": None,
        "ctrl_count": 0,
        "delta": 0.0,
        "jerk": 0.0,
        # Counted separately: a difference needs two samples and a second
        # difference three, and episode boundaries reset the history, so
        # dividing all three by the sample count biases the ratio -- and the
        # ratio is what becomes a frequency.
        "delta_count": 0,
        "jerk_count": 0,
        "prev": None,
        "prev_prev": None,
    }


def _accumulate_action(
    stats: dict[str, Any],
    action: np.ndarray,
    applied_ctrl: "np.ndarray | None" = None,
) -> None:
    """Accumulate one post-settle action and its applied physical control."""
    if stats["sum"] is None or stats["sum"].shape != action.shape:
        stats["sum"] = np.zeros_like(action)
        stats["sq_sum"] = np.zeros_like(action)
    stats["sum"] += action
    stats["sq_sum"] += action * action
    stats["count"] += 1
    prev, prev_prev = stats["prev"], stats["prev_prev"]
    if prev is not None and prev.shape == action.shape:
        # Summed over actuators, matching the `Sum` in the reward terms so
        # these invert straight back through `reward_action_smoothness` and
        # `reward_action_jerk`.
        stats["delta"] += float(np.sum((action - prev) ** 2))
        stats["delta_count"] += 1
        if prev_prev is not None and prev_prev.shape == action.shape:
            stats["jerk"] += float(np.sum((action - 2.0 * prev + prev_prev) ** 2))
            stats["jerk_count"] += 1
    stats["prev_prev"] = prev
    stats["prev"] = action

    if applied_ctrl is not None:
        ctrl = np.asarray(applied_ctrl, dtype=np.float64).ravel()
        if ctrl.shape == action.shape:
            if stats["ctrl_sum"] is None or stats["ctrl_sum"].shape != ctrl.shape:
                stats["ctrl_sum"] = np.zeros_like(ctrl)
                stats["ctrl_sq_sum"] = np.zeros_like(ctrl)
                stats["ctrl_count"] = 0
            stats["ctrl_sum"] += ctrl
            stats["ctrl_sq_sum"] += ctrl * ctrl
            stats["ctrl_count"] += 1


def _actuator_joint_names(env: Any) -> list[str]:
    """Names of the joint each actuator drives, or ``[]`` if unavailable.

    Unlike the training callback -- which reaches the environment only through
    VecEnv wrappers and so cannot resolve these without a best-effort lookup
    that silently returns nothing -- this script holds the env directly, so the
    mapping is exact and the per-actuator report can be read by joint rather
    than by index.
    """
    try:
        import mujoco

        model = env.unwrapped.model
        names = []
        for index in range(model.nu):
            joint_id = int(model.actuator_trnid[index, 0])
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            names.append(name or f"actuator_{index}")
        return names
    except Exception:  # noqa: BLE001 - names are a convenience, not the measurement
        return []


def _is_angular_position_control(model: Any, index: int, joint_id: int) -> bool:
    """Whether actuator control values are directly interpretable as radians.

    A joint transmission alone is not enough: Velociraptor's claw actuators
    are geared motors, where ``ctrl=1`` is a dimensionless motor command rather
    than a one-radian target.  Restrict degree fields to the canonical direct,
    unit-gear hinge position servo compiled by MuJoCo's ``<position>`` shortcut.
    Anything else keeps normalized and native-control statistics only.
    """
    import mujoco

    gain = float(model.actuator_gainprm[index, 0])
    bias = np.asarray(model.actuator_biasprm[index], dtype=np.float64)
    gear = np.asarray(model.actuator_gear[index], dtype=np.float64)
    return bool(
        int(model.actuator_trntype[index]) == int(mujoco.mjtTrn.mjTRN_JOINT)
        and int(model.jnt_type[joint_id]) == int(mujoco.mjtJoint.mjJNT_HINGE)
        and int(model.actuator_gaintype[index]) == int(mujoco.mjtGain.mjGAIN_FIXED)
        and int(model.actuator_biastype[index]) == int(mujoco.mjtBias.mjBIAS_AFFINE)
        and np.isfinite(gain)
        and gain > 0.0
        and np.isclose(bias[0], 0.0)
        and np.isclose(bias[1], -gain)
        and np.isclose(gear[0], 1.0)
        and np.allclose(gear[1:], 0.0)
    )


def _actuator_pose_mapping(env: Any) -> list[dict[str, Any]]:
    """Per actuator, separate action origin, home control, and home pose.

    Returns the physical control at normalized actions -1, 0, and +1, the
    named home keyframe's control target, and the joint pose in that keyframe.
    These are deliberately distinct: T-Rex maps residuals piecewise around
    ``key_ctrl['home']``, while its ankles use an intentional control preload
    relative to ``key_qpos['home']``.

    This exists because ``|action| = 1`` is **not one physical event**.
    ``_scale_action`` maps ``[-1, 1]`` onto each actuator's own physical
    control targets, and those spans differ per actuator -- on the physics-r6
    T-Rex by a factor of six: a saturated tail target was 8-12 degrees from
    its origin while a saturated toe target was 37.5 degrees away (the toe
    actuators were deleted in r7; plant_versions.toml note 9). Counting both
    as "saturated" pools them into one number, which is the same mistake the
    DC/AC split was introduced to fix one level up -- a normalized-unit
    summary hiding a physical distinction. Measured on r6, it hid the answer:
    the tail looked like the most extreme thing in the DC table and was
    provably inert, while the toes carried the whole effect.
    """
    try:
        # Probe the environment's own mapping instead of duplicating a shared
        # midpoint formula.  T-Rex overrides that formula with a piecewise
        # home-control residual mapping, and future action interfaces may do
        # the same.
        unwrapped = env.unwrapped
        model = unwrapped.model
        action_shape = tuple(unwrapped.action_space.shape)
        negative_ctrl = np.asarray(
            unwrapped._scale_action(-np.ones(action_shape, dtype=np.float64)),
            dtype=np.float64,
        ).ravel()
        action_zero_ctrl = np.asarray(
            unwrapped._scale_action(np.zeros(action_shape, dtype=np.float64)),
            dtype=np.float64,
        ).ravel()
        positive_ctrl = np.asarray(
            unwrapped._scale_action(np.ones(action_shape, dtype=np.float64)),
            dtype=np.float64,
        ).ravel()
        if any(values.shape != (model.nu,) for values in (negative_ctrl, action_zero_ctrl, positive_ctrl)):
            return []

        keyframe = int(getattr(unwrapped, "_reset_keyframe_id", 0))
        if not 0 <= keyframe < model.nkey:
            keyframe = 0
        home_qpos = model.key_qpos[keyframe] if model.nkey else None
        home_ctrl = model.key_ctrl[keyframe] if model.nkey else action_zero_ctrl
        mapping = []
        for index in range(model.nu):
            joint_id = int(model.actuator_trnid[index, 0])
            address = int(model.jnt_qposadr[joint_id])
            low, high = (float(value) for value in model.actuator_ctrlrange[index])
            mapping.append(
                {
                    "ctrl_min": low,
                    "ctrl_max": high,
                    "action_negative_ctrl": float(negative_ctrl[index]),
                    "action_zero_ctrl": float(action_zero_ctrl[index]),
                    "action_positive_ctrl": float(positive_ctrl[index]),
                    "angular_position_ctrl": _is_angular_position_control(model, index, joint_id),
                    # With no keyframe, use action zero for both anchors so the
                    # optional offsets read as unavailable/zero rather than
                    # inventing a nominal posture.
                    "home_ctrl": float(home_ctrl[index]),
                    "home_qpos": (
                        float(home_qpos[address]) if home_qpos is not None else float(action_zero_ctrl[index])
                    ),
                }
            )
        return mapping
    except Exception:  # noqa: BLE001 - the angles are a convenience, not the measurement
        return []


def _commanded_angle(entry: dict[str, Any], action: float) -> float:
    """The control target for an action, using the probed piecewise mapping.

    Real reports use the applied-control moments recorded during rollout.  The
    helper remains useful for endpoint display, fixtures, and legacy reports,
    but must still mirror ``home-keyframe-residual/v1`` on both sides of zero.
    Under ``home-keyframe-residual-softcubic/v1`` (the anatomical compsognathus
    legs) it is exact at zero and at both endpoints only: the shaped residual
    ``0.1 a + 0.9 a**3`` lies closer to home than this line between them.
    """
    clipped = float(np.clip(action, -1.0, 1.0))
    zero = float(
        entry.get(
            "action_zero_ctrl",
            entry.get("home_ctrl", entry.get("home", 0.5 * (entry["ctrl_min"] + entry["ctrl_max"]))),
        )
    )
    negative = float(entry.get("action_negative_ctrl", entry["ctrl_min"]))
    positive = float(entry.get("action_positive_ctrl", entry["ctrl_max"]))
    return zero + clipped * (zero - negative if clipped < 0.0 else positive - zero)


def _reduce_action_stats(
    stats: dict[str, Any],
    names: list[str],
    control_dt: float,
    pose_mapping: "list[dict[str, Any]] | None" = None,
) -> dict[str, Any]:
    """Reduce the accumulators into the report's ``action`` block.

    The DC/AC split is the point. ``mean(a)`` per actuator is the policy's
    static normalized residual and the spread about it is what the policy does
    on top. Applied-control moments are reduced independently, because the
    residual-to-control mapping can have different slopes on either side of
    zero. Static offset and variation have different causes and fixes, and a
    single pooled number cannot tell them apart (issue #489).
    """
    count = stats["count"]
    if not count or stats["sum"] is None:
        return {}
    mean = stats["sum"] / count
    var = np.maximum(stats["sq_sum"] / count - mean * mean, 0.0)
    ac = np.sqrt(var)
    delta = stats["delta"] / stats["delta_count"] if stats["delta_count"] else 0.0
    jerk = stats["jerk"] / stats["jerk_count"] if stats["jerk_count"] else 0.0
    # For a narrowband action signal `jerk/delta = (2 sin(pi f dt))^2`, blind
    # to any constant offset. At the white-noise limit the ratio saturates,
    # which means broadband rather than a tone at that frequency.
    freq = float("nan")
    if delta > 0.0:
        freq = math.asin(min(1.0, math.sqrt(jerk / delta) / 2.0)) / (math.pi * control_dt)
    mapping = pose_mapping or []
    ctrl_count = int(stats.get("ctrl_count", 0))
    ctrl_mean: "np.ndarray | None" = None
    ctrl_ac: "np.ndarray | None" = None
    if (
        ctrl_count == count
        and stats.get("ctrl_sum") is not None
        and stats.get("ctrl_sq_sum") is not None
        and stats["ctrl_sum"].shape == mean.shape
        and stats["ctrl_sq_sum"].shape == mean.shape
    ):
        ctrl_mean = stats["ctrl_sum"] / ctrl_count
        ctrl_var = np.maximum(stats["ctrl_sq_sum"] / ctrl_count - ctrl_mean * ctrl_mean, 0.0)
        ctrl_ac = np.sqrt(ctrl_var)
    per_actuator = []
    for index in range(mean.size):
        entry: dict[str, Any] = {
            "index": index,
            "joint": names[index] if index < len(names) else f"actuator_{index}",
            "dc": float(mean[index]),
            "ac_rms": float(ac[index]),
        }
        if index < len(mapping):
            scale = mapping[index]
            action_zero_ctrl = float(
                scale.get(
                    "action_zero_ctrl",
                    scale.get("home_ctrl", scale.get("home", 0.5 * (scale["ctrl_min"] + scale["ctrl_max"]))),
                )
            )
            home_ctrl = float(scale.get("home_ctrl", action_zero_ctrl))
            home_qpos = float(scale.get("home_qpos", scale.get("home", home_ctrl)))
            angular_position_ctrl = bool(scale.get("angular_position_ctrl", True))
            # Preserve control anchors in native units so the report is
            # auditable without reloading the exact plant model.  Control and
            # qpos units are comparable only for the verified position-servo
            # case below.
            entry["angular_position_ctrl"] = angular_position_ctrl
            entry["action_zero_ctrl"] = action_zero_ctrl
            entry["home_ctrl"] = home_ctrl
            # Applied-control moments are authoritative.  Mapping the action
            # mean and variance after reduction is not equivalent when the
            # negative and positive residual slopes differ, so omit physical
            # DC/AC rather than fabricate them when controls were unavailable.
            if ctrl_mean is not None and ctrl_ac is not None:
                entry["ctrl_mean"] = float(ctrl_mean[index])
                entry["ctrl_ac_rms"] = float(ctrl_ac[index])
            if angular_position_ctrl:
                entry["home_qpos"] = home_qpos
                entry["range_deg"] = math.degrees(scale["ctrl_max"] - scale["ctrl_min"])
                if ctrl_mean is not None and ctrl_ac is not None:
                    entry["dc_deg"] = math.degrees(float(ctrl_mean[index]) - action_zero_ctrl)
                    entry["ac_rms_deg"] = math.degrees(float(ctrl_ac[index]))
                # A mapping-contract check, not a posture offset.  Under
                # home-keyframe-residual/v1 this must be zero even when the
                # home actuator target intentionally preloads a joint against
                # gravity.
                entry["zero_offset_deg"] = math.degrees(action_zero_ctrl - home_ctrl)
                entry["home_preload_deg"] = math.degrees(home_ctrl - home_qpos)
        per_actuator.append(entry)
    reduced = {
        "dc_rms": float(np.sqrt(np.mean(mean * mean))),
        "ac_rms": float(np.sqrt(np.mean(var))),
        "delta_per_step": float(delta),
        "jerk_per_step": float(jerk),
        "effective_freq_hz": freq,
        "per_actuator": per_actuator,
    }
    degrees = [entry["dc_deg"] for entry in per_actuator if "dc_deg" in entry]
    if degrees:
        # Reported alongside `dc_rms` rather than replacing it: the normalized
        # figure is what inverts through the action penalties, and the degrees
        # say how far the physical control targets moved.
        reduced["dc_rms_deg"] = float(np.sqrt(np.mean(np.square(degrees))))
        reduced["max_abs_dc_deg"] = float(np.max(np.abs(degrees)))
    return reduced


def _roll_episodes(
    env: Any,
    *,
    predict: Any,
    episodes: int,
    seed: int,
    settle_steps: int,
    lengths: list[float],
    rewards: list[float],
    duties: list[float],
    bilateral_duties: list[float],
    single_duties: list[float],
    terminations: dict[str, int],
    components: dict[str, float],
    action_stats: dict[str, Any],
    impulse: "RootImpulse | None" = None,
    recorder: Any = None,
    stance_metrics: "list[StanceEpisodeMetrics] | None" = None,
    spawn_yaw: "SpawnYaw | None" = None,
    yaw_changes: "list[float] | None" = None,
) -> None:
    """Roll the panel, appending per-episode measurements to the given lists.

    Split out only so ``run_panel`` can wrap it in the ``finally`` that closes
    the environment; the accumulation is unchanged.

    With an attached *recorder* every step is closed with the policy's
    action and the env's reward right after ``env.step``, and every episode
    is reduced to its floor-truth metrics (appended to *stance_metrics*)
    once it ends.  The recorder learns of a new episode from its own hook
    (the first substep after ``env.reset``), so nothing is called at reset.
    """
    for index in range(episodes):
        obs, _ = env.reset(seed=seed + index)
        if spawn_yaw is not None:
            obs = _apply_spawn_yaw(env, spawn_yaw)
        # The heading probe's reading: the root's unwrapped yaw change over
        # the episode, accumulated step by step so a turn past 180 degrees
        # is not folded back.
        yaw_previous = _root_yaw(env) if yaw_changes is not None else 0.0
        yaw_turned = 0.0
        # A stateful predict (the low-pass probe) must not carry one episode's
        # tail into the next; plain policies have no reset and are untouched.
        reset_predict = getattr(predict, "reset", None)
        if callable(reset_predict):
            reset_predict()
        total = 0.0
        steps = 0
        unsupported = 0
        bilateral = 0
        single = 0
        measured = 0
        while True:
            # Before the action is commanded, so the policy's first response to
            # the shove is the step that follows it rather than the one it was
            # already committed to.
            if impulse is not None and steps == impulse.step:
                _apply_root_impulse(env, impulse)
            action = np.asarray(predict(obs), dtype=np.float64).ravel()
            # Post-settle only, the same window the duty uses, so the reset
            # transient does not read as tremor.
            measure_step = steps >= settle_steps
            obs, reward, terminated, truncated, info = env.step(action)
            if recorder is not None:
                recorder.end_step(action, float(reward))
            if yaw_changes is not None:
                yaw_now = _root_yaw(env)
                yaw_turned += math.remainder(yaw_now - yaw_previous, 2.0 * math.pi)
                yaw_previous = yaw_now
            if measure_step:
                # ``data.ctrl`` is the target the environment actually
                # applied.  Reading it after ``step`` avoids a second call to
                # a potentially species-specific mapping and also remains
                # correct if plant-side action dynamics are added later.
                try:
                    applied_ctrl = np.asarray(env.unwrapped.data.ctrl, dtype=np.float64).copy()
                except Exception:  # noqa: BLE001 - normalized stats remain useful without it
                    applied_ctrl = None
                _accumulate_action(action_stats, action, applied_ctrl)
            total += float(reward)
            for key, value in info.items():
                if key.startswith("reward_") and key != "reward_total":
                    components[key] = components.get(key, 0.0) + float(value)
            stance = derive_stance_info(info)
            if stance and measure_step:
                measured += 1
                unsupported += int(stance["unsupported_duty"])
                # The gate bounds unsupported duty only. Reporting the full
                # three-way split shows where recovered airborne time actually
                # went: a policy that trades flight for SINGLE support drives
                # the gated number down without ever planting both feet.
                bilateral += int(stance["bilateral_support_duty"])
                single += int(stance["single_support_duty"])
            steps += 1
            if terminated or truncated:
                reason = info.get("termination_reason", "terminated" if terminated else "truncated")
                terminations[reason] = terminations.get(reason, 0) + 1
                if recorder is not None and stance_metrics is not None:
                    trace = recorder.end_episode(
                        terminated=bool(terminated), truncated=bool(truncated), termination_reason=str(reason)
                    )
                    stance_metrics.append(episode_stance_metrics(trace, settle_steps=settle_steps))
                break
        # An episode boundary is not an action difference.
        action_stats["prev"] = None
        action_stats["prev_prev"] = None
        if yaw_changes is not None:
            yaw_changes.append(float(np.degrees(abs(yaw_turned))))
        lengths.append(float(steps))
        rewards.append(total)
        # All three shares stay positionally aligned with `lengths`, so the
        # full-horizon filter below applies to them identically.
        duties.append(unsupported / measured if measured else float("nan"))
        bilateral_duties.append(bilateral / measured if measured else float("nan"))
        single_duties.append(single / measured if measured else float("nan"))


#: Bumped when the JSON report's field meanings change.  This is the schema
#: of a report that scored the ``stance_quality/v1`` criteria (every probe
#: report among them); "v2" here is the REPORT's revision, not the gate kind's.
REPORT_SCHEMA = "mesozoic.stance-gate-report/v2"

#: A report that scored ``stance_quality/v2`` (decision D-D23) has another
#: shape (per-episode floor-truth rows, the measurement manifest, the handoff
#: digests, the statue block) and its own revision,
#: ``stance_gate_v2.STANCE_V2_REPORT_SCHEMA`` = ``mesozoic.stance-gate-report/v3``.


def _low_pass_predict(predict: Any, cutoff_hz: float, control_dt: float) -> Any:
    """Wrap *predict* so its action is low-passed before reaching the plant.

    A probe, not a feature: it answers whether a policy's high-frequency
    action content is load-bearing or waste. Run ``20260805_011234`` passed
    the stance gate carrying a tremor of rms 0.277 at an effective 22.6 Hz,
    and whether that tremor is closed-loop stabilisation or a free-running
    buzz decides whether the fix belongs on the pose or on the action path
    (issue #489). Open-loop experiments on the statue bound what a tremor
    *can* do; only filtering the real policy answers it for that policy.

    First-order IIR, ``y += alpha * (x - y)`` with
    ``alpha = dt / (RC + dt)`` and ``RC = 1 / (2*pi*fc)`` -- the discrete
    equivalent of an RC filter, chosen because it is the one a plant-side
    implementation would most plausibly use. Filtering happens between the
    policy and the environment, so the policy still sees unfiltered
    observations and is genuinely being asked to stand without its own
    high-frequency component.

    The state is per-call-sequence and reset by the caller between episodes
    via :func:`reset`, so one episode's tail cannot leak into the next.

    The discretization is shared with the training-time plant filter
    (environments/shared/action_filter.py) so probe cutoffs and plant
    cutoffs are directly comparable.  When probing a plant that itself
    filters (action_filter_cutoff_hz > 0), the two filters stack.
    """
    alpha = low_pass_alpha(cutoff_hz, control_dt)
    state: dict[str, Any] = {"y": None}

    def filtered(obs: np.ndarray) -> np.ndarray:
        action = np.asarray(predict(obs), dtype=np.float64)
        previous = state["y"]
        # Seeded with the first action rather than zeros: starting from zero
        # would inject a step transient that is an artefact of the probe, not
        # a property of the policy.
        state["y"] = (
            action if previous is None or previous.shape != action.shape else previous + alpha * (action - previous)
        )
        return np.asarray(state["y"], dtype=np.float32)

    filtered.reset = lambda: state.update(y=None)  # type: ignore[attr-defined]
    return filtered


@dataclass(frozen=True)
class ConstantHold:
    """A constant command to substitute for the policy, and when to substitute it.

    The probe behind ``docs/investigations/TREX_STAGE1_BOUNCE_2026_08.md`` §5:
    the passing T-Rex stage-1 policy stands with a 0.366 rms tremor at ~22.7 Hz,
    and the low-pass probe shows that tremor is load-bearing. That leaves two
    readings, which the filter probe cannot separate:

    * the *pose* the policy holds needs continuous active stabilisation, or
    * the pose is holdable by a constant and the tremor is waste the action
      penalties failed to suppress.

    Replacing the policy with the constant it commands **on average** decides
    it. If the animal keeps standing, the smooth solution existed and the
    optimiser did not find it. If it falls, the pose genuinely requires
    feedback, and the question moves to why the policy chose a pose it then has
    to fight -- which pointed at the actuators no stage-1 term constrains (13
    of 21 on the physics-r6 plant that measurement was made on; the six toe
    actuators are gone as of r7).

    Attributes:
        actions: The constant command, one entry per actuator.
        handoff_steps: Steps of real policy before the substitution. ``0``
            commands the constant from reset; a value at or above the horizon
            never substitutes at all, which is how the probe rolls its own
            unmodified control panel without producing a report that could be
            mistaken for a gate verdict.
        ramp_steps: Steps to blend from the last commanded action into
            *actions*. ``0`` switches in one step. The ramp exists to answer
            the obvious objection to a hard switch -- that the animal fell from
            the step transient rather than from losing feedback -- with a
            second measurement instead of an argument.
        label: Short name for this variant in the probe artifact.
        source: Where *actions* came from, recorded so a reader can tell a
            measured hold from a hand-supplied one.
    """

    actions: tuple[float, ...]
    handoff_steps: int
    ramp_steps: int = 0
    label: str = "hold"
    source: str = "measured"
    #: Set for the control variant that never substitutes, so the renderer can
    #: say "this row is the unmodified policy" rather than implying a hold.
    holds: bool = field(default=True)

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "source": self.source,
            "holds": self.holds,
            "handoff_steps": self.handoff_steps,
            "ramp_steps": self.ramp_steps,
            "actions": list(self.actions),
        }


@dataclass(frozen=True)
class RootImpulse:
    """A shove applied to the animal partway through an episode.

    Stage 1 contains **no in-episode disturbance** -- the only perturbation
    anywhere is joint-angle noise at reset -- so a policy that learns active
    postural correction earns nothing over one that stands still, and the
    zero-action statue is the objective's optimum. That makes "does this policy
    actively correct?" unanswerable from any training artifact: the task never
    asks.

    This asks. The impulse is a step change in the root's linear velocity,
    which is exactly a shove to the torso and needs no force/duration
    bookkeeping to reproduce -- ``delta_v`` fully specifies it, and the
    equivalent momentum is ``total_mass * delta_v``.

    The **statue is the control**, and it is what makes the measurement mean
    something. It commands a constant, so it cannot respond to anything;
    whatever horizon it survives is the plant's *passive* robustness. Only the
    margin above that line is active correction. A policy that matches the
    statue has learned to stand, not to recover.

    Attributes:
        step: Step at which the impulse lands, normally the settle window so
            the policy has established its own state first.
        delta_v: Velocity change in m/s, world frame, applied to the root.
        axis_label: Human name for the direction, for the artifact.
    """

    step: int
    delta_v: tuple[float, float, float]
    axis_label: str = "lateral"

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(np.asarray(self.delta_v, dtype=float)))

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": "root_impulse/v1",
            "step": self.step,
            "delta_v": list(self.delta_v),
            "speed": self.speed,
            "axis_label": self.axis_label,
        }


def _apply_root_impulse(env: Any, impulse: RootImpulse) -> float:
    """Add the impulse to the root's linear velocity. Returns the momentum in N*s.

    ``qvel[0:3]`` is the free joint's world-frame linear velocity, and every
    child body's velocity composes from it, so this translates the whole animal
    rather than only the root body -- a shove, not a tug on one link.
    """
    data = env.unwrapped.data
    model = env.unwrapped.model
    data.qvel[0:3] += np.asarray(impulse.delta_v, dtype=data.qvel.dtype)
    return float(np.sum(model.body_mass)) * impulse.speed


@dataclass(frozen=True)
class SpawnYaw:
    """The spawn turned about vertical right after reset: the heading probe's one modification.

    The stance stage always spawns facing +x with the prey within about 11
    degrees of it, and up to policy interface r13 the trex observation
    carried heading through the world-frame pelvis quaternion (and the
    world-frame velocity and prey direction), so a stance could depend on
    the world heading without any v2 criterion seeing it: both physics-r8
    trex stances fell when spawned 90 degrees off, where the statue stands.
    The velociraptor and the compsognathus robot still observe those
    world-frame channels; the trex from policy interface r14 and the
    anatomical compsognathus from r4 observe no world heading (D-D31,
    ``bipedal-target-heading-free/v1``: projected gravity, the root velocity
    in the gravity-aligned heading frame and the prey direction in pelvis
    coordinates, :mod:`environments.shared.heading_observation`).  Turning the spawn
    changes nothing physical because every plant declares the elliptic
    friction cone (``cone="elliptic"``: T. rex physics r9, D-D29; the
    velociraptor, brachiosaurus and dibothrosuchus, D-D30;
    ``test_friction_cone.py``), whose friction has no direction fixed to the
    world, on a level floor where a rotation about vertical moves no
    height -- so the zero-action statue's survival is the control, and its
    turned rows repeat its unturned ones to solver tolerance.  A level,
    isotropic floor alone was not enough: under MuJoCo's default pyramidal
    cone a box or ellipsoid foot's friction pyramid is aligned with the
    world axes, so on r8 T. rex (and the brachiosaurus and dibothrosuchus
    before D-D30) a 45 degree turn was a small physical perturbation as
    well, and only multiples of 90 degrees were neutral.

    ``whole_scene`` decides what turns with the animal.  ``False`` turns the
    animal alone: the prey stays where it spawned, so the reward terms that
    read the prey bearing (the stance task's heading term) now pay a policy
    that turns toward it, and the reading mixes heading dependence with a
    prey-bearing response.  ``True`` turns the prey about the root with it
    and the episode's reference direction (``_initial_prey_dir_2d``) too, so
    physics and reward are identical for a heading-invariant controller and
    only a world-frame observation differs -- exactly the question of
    whether the policy reads its heading.  On a heading-free observation
    (trex r14, anatomical compsognathus r4) nothing differs: the turned
    observation equals the unturned one, so a seed's turned episodes repeat
    its unturned one up to floating-point rounding, which an unstable closed
    loop can still amplify into a different outcome; the animal-only turn
    moves only the observed prey bearing there.  The probe reports both.

    The probe is report-only (decision D-D27).  The certificate is a
    separate gate kind: ``stance_quality_heading/v1`` (D-D32) rolls this
    same whole-scene turn over the full certification panel at every heading
    of a fixed grid (:func:`build_stance_heading_report`) and blocks handoff,
    reuse and publication on it.  A stage that declares the certificate
    does not roll this probe (config load refuses the key there); on a
    stage without it (the compsognathus robot and the velociraptor) the probe
    stays report-only, with the handoff warning of 2026-10-11.
    """

    yaw_deg: float
    whole_scene: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {"kind": "spawn_yaw/v1", "yaw_deg": float(self.yaw_deg), "whole_scene": bool(self.whole_scene)}


def _root_yaw(env: Any) -> float:
    """The free root's yaw about world z, in radians (from ``qpos[3:7]``, w-first)."""
    w, x, y, z = (float(value) for value in env.unwrapped.data.qpos[3:7])
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _apply_spawn_yaw(env: Any, spawn_yaw: SpawnYaw) -> np.ndarray:
    """Turn the freshly reset spawn about vertical through the root and return the rebuilt observation.

    Pre-multiplies the root quaternion by a world-z rotation and turns the
    root's world-frame linear velocity with it (the free joint's angular
    velocity is body-frame, so it is already invariant); under
    ``whole_scene`` every mocap body (the prey) turns about the root too, and
    an env that keeps an episode reference direction (``_initial_prey_dir_2d``)
    has it turned to match.  Then the derived state is recomputed, the
    step-tagged substep aggregates are dropped (the state was posed out of
    band) and the observation is rebuilt from the turned state.
    """
    import mujoco

    base = env.unwrapped
    model, data = base.model, base.data
    theta = math.radians(float(spawn_yaw.yaw_deg))
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    rotation_2d = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    turn = np.array([math.cos(theta / 2.0), 0.0, 0.0, math.sin(theta / 2.0)])
    turned = np.empty(4)
    mujoco.mju_mulQuat(turned, turn, np.asarray(data.qpos[3:7], dtype=np.float64))
    data.qpos[3:7] = turned
    data.qvel[0:2] = rotation_2d @ np.asarray(data.qvel[0:2], dtype=np.float64)
    if spawn_yaw.whole_scene:
        root_xy = np.asarray(data.qpos[0:2], dtype=np.float64).copy()
        for index in range(model.nmocap):
            data.mocap_pos[index, :2] = root_xy + rotation_2d @ (data.mocap_pos[index, :2] - root_xy)
            mocap_turned = np.empty(4)
            mujoco.mju_mulQuat(mocap_turned, turn, np.asarray(data.mocap_quat[index], dtype=np.float64))
            data.mocap_quat[index] = mocap_turned
        reference = getattr(base, "_initial_prey_dir_2d", None)
        if reference is not None:
            base._initial_prey_dir_2d = rotation_2d @ np.asarray(reference, dtype=np.float64)
    mujoco.mj_forward(model, data)
    base._invalidate_substep_aggregates()
    return np.asarray(base._get_obs())


def _hold_constant_predict(predict: Any, hold: ConstantHold) -> Any:
    """Wrap *predict* so the plant receives a constant command after handoff.

    Feedback is cut, not attenuated: past ``handoff_steps`` the returned action
    no longer depends on the observation at all. That is the point -- a
    low-pass still lets the policy respond slowly, so it cannot distinguish
    "needs bandwidth" from "needs feedback", and this can.

    The step counter is per episode and reset by the caller through
    :func:`reset`, exactly like the low-pass probe: without it the second
    episode would start already past handoff and the panel would measure
    something different from the first.
    """
    constant = np.asarray(hold.actions, dtype=np.float64)
    state: dict[str, Any] = {"step": 0, "last": None}

    def held(obs: np.ndarray) -> np.ndarray:
        step = state["step"]
        state["step"] = step + 1
        if step < hold.handoff_steps:
            action = np.asarray(predict(obs), dtype=np.float64).ravel()
            state["last"] = action
            return np.asarray(action, dtype=np.float32)
        elapsed = step - hold.handoff_steps
        if hold.ramp_steps > 0 and elapsed < hold.ramp_steps:
            # Blend from whatever was last commanded. At handoff that is the
            # policy's own last action; from reset there is none, and zero is
            # the right start because `action = 0` commands the named home
            # control -- the targets the episode's reset pose settles under --
            # so the ramp starts from the actual stance rather than jumping
            # to one.
            previous = state["last"]
            if previous is None or previous.shape != constant.shape:
                previous = np.zeros_like(constant)
            weight = (elapsed + 1) / hold.ramp_steps
            return np.asarray((1.0 - weight) * previous + weight * constant, dtype=np.float32)
        return np.asarray(constant, dtype=np.float32)

    held.reset = lambda: state.update(step=0, last=None)  # type: ignore[attr-defined]
    return held


def constant_hold_actions(report: dict[str, Any]) -> tuple[float, ...]:
    """The per-actuator DC a report measured, as a constant command.

    DC is the time-mean of the commanded action over the post-settle window --
    under ``home-keyframe-residual/v1`` the static target the policy holds,
    since ``action = 0`` is the named home control. Holding it is what makes
    the probe a test of the pose rather than of some nearby pose: the animal
    is asked to stand where it was already standing, with the variation about
    it removed and nothing else changed.

    One deliberate approximation, worth knowing when reading the results: the
    held control is ``f(mean(action))``, not the ``mean(f(action))`` the
    policy actually commanded -- the same distinction schema v2 draws for the
    report's degree statistics. Under the piecewise-affine map the two differ
    only for an actuator whose action distribution crosses zero AND whose
    ctrlrange spans differ about its home control; on the current trex plant
    that is neck/head pitch only, where the measured crossings put the bias
    near 0.01 degrees. Under ``home-keyframe-residual-softcubic/v1`` (the
    anatomical compsognathus legs) they differ wherever a shaped action
    varies, by ``2.7 * mean * variance`` of the residual before the span (for
    a symmetric action distribution). If a plant makes the gap material,
    derive the hold by inverting the recorded per-actuator ``ctrl_mean``
    through the probed mapping instead.
    """
    per_actuator = (report.get("action") or {}).get("per_actuator") or []
    if not per_actuator:
        raise StanceGateReportError(
            "the report carries no per-actuator action statistics, so there is no "
            "measured pose to hold. It was produced before that block existed, or "
            "from a panel in which no episode survived the settling window."
        )
    return tuple(float(entry["dc"]) for entry in sorted(per_actuator, key=lambda e: e["index"]))


def build_stance_gate_report(
    species: str,
    stage: "int | str",
    *,
    stage_config: dict[str, Any],
    model_path: str | None = None,
    vecnorm_path: str | None = None,
    zero_action: bool = False,
    episodes: int | None = None,
    seed: int = PUBLICATION_SEED_START,
    allow_legacy_plant: bool = False,
    filter_actions_hz: float | None = None,
    hold_constant: ConstantHold | None = None,
    impulse: RootImpulse | None = None,
    spawn_yaw: SpawnYaw | None = None,
) -> dict[str, Any]:
    """Roll a policy and return its gate verdict as a serializable dict.

    Importable so the training pipeline can emit the same report it would
    produce by hand, rather than a second implementation that could disagree.

    The checkpoint and the rollout environment are validated against the
    species' current plant identity: a verdict measured on a different plant
    than the one in the tree is not a verdict about this stage, and every
    other artifact path in the repository already refuses that pairing. Pass
    *allow_legacy_plant* to score a checkpoint that predates the contract --
    which is a real use for this script, since it exists partly to judge
    already-finished runs -- and the report records that it was done.

    ``episodes`` of ``None`` means the stage's own ``min_eval_episodes``,
    the panel size the bound's power is specified at. ``0`` is rejected
    rather than silently treated as absent.

    A stage declaring ``stance_quality/v2`` is scored under THAT kind
    (:func:`_stance_v2_report`): strict thresholds, the floor-truth recorder
    on the panel, a statue panel on the same seeds when a statue-relative
    criterion is declared, and a ``mesozoic.stance-gate-report/v3`` report
    that records which code scored it (``scored_gate_kind``), the handoff
    digests and the measurement manifest, so the judge can refuse a report
    that does not describe this gate, this checkpoint or this measurement.
    A PROBE (``filter_actions_hz`` / ``hold_constant`` / ``impulse`` /
    ``spawn_yaw``) on a v2 stage keeps the v1-shaped report it always produced, with no
    recorder and no statue panel: a probe annotates, it never certifies,
    and its readers (the sweep and table writers) read that shape.

    A stage declaring ``stance_quality_heading/v1`` (D-D32) is certified by
    :func:`build_stance_heading_report` alone: a non-probe call for one is a
    ``ValueError`` naming it, because one panel at the training heading is
    exactly the evidence that kind exists to refuse.  Probes on such a
    stage stay v1-shaped, as on a v2 stage.
    """
    if episodes is not None and episodes < 1:
        raise ValueError(f"episodes must be at least 1 if given, got {episodes}")
    if filter_actions_hz is not None and filter_actions_hz <= 0:
        raise ValueError(f"filter_actions_hz must be positive if given, got {filter_actions_hz}")
    curriculum = stage_config["curriculum_kwargs"]
    env_kwargs = dict(stage_config["env_kwargs"])
    horizon = int(env_kwargs.get("max_episode_steps", 1000))
    control_dt = float(env_kwargs.get("timestep", 0.002)) * int(env_kwargs.get("frame_skip", 5))
    probing = filter_actions_hz is not None or hold_constant is not None or impulse is not None or spawn_yaw is not None
    if curriculum.get("gate_kind") == STANCE_HEADING_GATE_KIND and not probing:
        raise ValueError(
            f"stage {stage} declares {STANCE_HEADING_GATE_KIND}, whose certificate is the six-heading grid: build it "
            "with build_stance_heading_report. A single panel at the training heading certifies nothing for it."
        )
    v2_thresholds: StanceV2Thresholds | None = None
    if curriculum.get("gate_kind") == STANCE_GATE_V2_KIND and not probing:
        # Strict, and checked before anything is rolled: a v2 block with a
        # missing bar, or a settle window inside the spawn grace, would cost
        # a whole panel to produce a report the judge then refuses.
        v2_thresholds = StanceV2Thresholds.from_curriculum(curriculum)
        v2_thresholds.validate_settle_window(control_dt)
    # Lenient on purpose (require_criteria=False): a stage that does not gate
    # on stance still gets a readable report rather than a spuriously strict one.
    thresholds = StanceGateThresholds.from_curriculum(curriculum, require_criteria=False)
    default_episodes = thresholds.min_eval_episodes if v2_thresholds is None else v2_thresholds.min_eval_episodes
    panel_episodes = default_episodes if episodes is None else episodes

    from environments.shared.plant_contract import current_plant_identity

    plant_identity = current_plant_identity(species)

    env_class = SPECIES_FACTORIES[species]().env_class
    if zero_action:
        predict = _zero_action_predict(env_class, env_kwargs)
        description = _ZERO_ACTION_DESCRIPTION
    else:
        if model_path is None:
            raise ValueError("model_path is required unless zero_action is set")
        predict, note = _load_policy(
            model_path,
            vecnorm_path,
            lambda: env_class(**env_kwargs),
            plant_identity=plant_identity,
            allow_legacy_plant=allow_legacy_plant,
        )
        description = f"{Path(model_path).name} — {note}"

    if filter_actions_hz is not None:
        predict = _low_pass_predict(predict, filter_actions_hz, control_dt)
        description += f" — actions low-passed at {filter_actions_hz:g} Hz"

    if hold_constant is not None:
        expected = int(plant_identity.action_dim)
        if len(hold_constant.actions) != expected:
            # Checked before the rollout rather than at the first step: numpy
            # broadcasts a mismatched length silently, which would score a
            # policy nobody asked for and report it as this one.
            raise ValueError(
                f"hold_constant carries {len(hold_constant.actions)} actions but {species} has action_dim {expected}"
            )
        predict = _hold_constant_predict(predict, hold_constant)
        description += (
            f" — {hold_constant.label}"
            if not hold_constant.holds
            else f" — commanded action held constant from step {hold_constant.handoff_steps}"
        )

    if v2_thresholds is not None:
        return _stance_v2_report(
            species,
            stage,
            curriculum=curriculum,
            env_class=env_class,
            env_kwargs=env_kwargs,
            thresholds=v2_thresholds,
            predict=predict,
            description=description,
            zero_action=zero_action,
            model_path=model_path,
            vecnorm_path=vecnorm_path,
            episodes=panel_episodes,
            seed=seed,
            horizon=horizon,
            control_dt=control_dt,
            plant_identity=plant_identity,
            allow_legacy_plant=allow_legacy_plant,
        )

    result = run_panel(
        species,
        predict=predict,
        episodes=panel_episodes,
        seed=seed,
        settle_steps=thresholds.settle_steps,
        horizon=horizon,
        env_kwargs=env_kwargs,
        plant_identity=plant_identity,
        control_dt=control_dt,
        impulse=impulse,
        spawn_yaw=spawn_yaw,
    )
    panel = result["panel"]
    passed, failures = evaluate_stance_gate(panel, thresholds)

    return {
        "schema": REPORT_SCHEMA,
        "species": species,
        "stage": stage,
        "gate_kind": curriculum.get("gate_kind"),
        # Which criteria THIS code scored, whatever the stage declares: the
        # report echoes the declared kind above, and a v1-scored report
        # labelled stance_quality/v2 (a probe on a v2 stage, or a report
        # from before the kind existed) must never read as a v2 verdict.
        "scored_gate_kind": STANCE_GATE_KIND,
        "policy": description,
        "episodes": panel_episodes,
        "seed": seed,
        "settle_steps": thresholds.settle_steps,
        "horizon": horizon,
        "passed": passed,
        "failures": failures,
        "thresholds": {
            "min_full_horizon_fraction": thresholds.min_full_horizon_fraction,
            "max_unsupported_duty": thresholds.max_unsupported_duty,
            "max_unsupported_duty_ucb": thresholds.max_unsupported_duty_ucb,
            "min_avg_reward": thresholds.min_avg_reward,
            "min_eval_episodes": thresholds.min_eval_episodes,
        },
        "metrics": {
            "reward_mean": float(result["rewards"].mean()),
            "reward_std": float(result["rewards"].std()),
            "episode_length_mean": float(result["lengths"].mean()),
            "full_horizon_fraction": panel.full_horizon_fraction,
            "mean_unsupported_duty": panel.mean_unsupported_duty,
            "unsupported_duty_ucb": panel.unsupported_duty_ucb,
            "n_duty_episodes": panel.n_duty_episodes,
            # Not gated. Reported because the three shares sum to 1, so a
            # falling unsupported duty does not by itself mean the feet are
            # being planted -- the time can land in single support instead.
            "bilateral_support_duty": result["bilateral_duty"],
            "single_support_duty": result["single_duty"],
        },
        # Whether the CHECKPOINT's plant identity was checked against the
        # tree. `None` for --zero-action, which has no checkpoint to check.
        # The rollout ENVIRONMENT is always validated regardless, so this
        # field is deliberately narrow rather than a blanket "plant_validated"
        # that would claim something false in the zero-action case.
        "checkpoint_plant_validated": None if zero_action else not allow_legacy_plant,
        # `None` for an ordinary run. Recorded because a filtered rollout is
        # NOT a verdict about the policy: it is a probe of a modified one, and
        # a PASS produced this way must never be mistakable for a real PASS in
        # a stored report. `render_stance_gate_report` prints it for the same
        # reason.
        "filter_actions_hz": filter_actions_hz,
        # `None` for an ordinary run. Same contract as `filter_actions_hz`: a
        # held rollout scored a policy whose feedback was cut, so it is a probe
        # and never a verdict, and `_PROBE_MARKERS` keys the filenames off this.
        "hold_constant": None if hold_constant is None else hold_constant.as_dict(),
        # `None` for an ordinary run. A perturbed rollout scores the UNMODIFIED
        # policy on a MODIFIED task, which is the opposite direction from the
        # other two probes but has the same consequence: the gate is defined
        # without a disturbance, so this verdict is not that verdict.
        "impulse": None if impulse is None else impulse.as_dict(),
        # `None` for an ordinary run. A turned spawn scores the unmodified
        # policy on a modified start, like the impulse; `episode_yaw_change_deg`
        # is its reading (per episode, in the same order as `episode_evidence`).
        "spawn_yaw": None if spawn_yaw is None else spawn_yaw.as_dict(),
        **({"episode_yaw_change_deg": list(result["yaw_changes"])} if spawn_yaw is not None else {}),
        # Where the commanded pose sits and what the policy does around it,
        # per actuator. `{}` when no post-settle step was measured.
        "action": result.get("action", {}),
        "terminations": result["terminations"],
        "reward_components": result["components"],
        # The per-episode measurements the panel was reduced from. This is
        # the duty evidence `result_bundle.evidence` refuses stance-gated
        # bundles for lacking -- note it does NOT by itself lift that
        # refusal, which reads `evaluation_selected.csv`; teaching that
        # function to consume this is the remaining step.
        "episode_evidence": result["episodes"],
    }


#: The zero-action policy's description, in every report shape: the v2
#: judge refuses a statue block that records any other
#: (``stance_gate_v2.STATUE_POLICY``, the one definition).
_ZERO_ACTION_DESCRIPTION = STATUE_POLICY


def _zero_action_predict(env_class: Any, env_kwargs: dict[str, Any]) -> Any:
    """The statue: a predict that commands ``action = 0`` (the named home control) every step."""
    probe = env_class(**env_kwargs)
    zero = np.zeros(probe.action_space.shape[0], dtype=np.float32)
    probe.close()

    def predict(_obs: np.ndarray) -> np.ndarray:
        return zero

    return predict


def _stance_v2_episode_rows(
    metrics: list[StanceEpisodeMetrics],
    reasons: "tuple[tuple[str, ...], ...] | None",
    touch_rows: list[dict[str, Any]],
    *,
    seed: int,
    horizon: int,
) -> list[dict[str, Any]]:
    """One evidence row per episode: seed, classification, every floor-truth metric, the touch duties.

    *reasons* is ``None`` for the statue panel, which is a reference and is
    not classified.  The touch duties (v1's measure, from the same rollout)
    ride along report-only, so the two readings of one episode sit side by
    side.
    """
    rows = []
    for index, episode in enumerate(metrics):
        touch = touch_rows[index] if index < len(touch_rows) else {}
        row: dict[str, Any] = {
            "episode": index,
            "seed": seed + index,
            "reached_horizon": episode.length >= horizon,
            "unsupported_duty": touch.get("unsupported_duty"),
            "bilateral_support_duty": touch.get("bilateral_support_duty"),
            "single_support_duty": touch.get("single_support_duty"),
        }
        if reasons is not None:
            row["clean"] = not reasons[index]
            row["reasons"] = list(reasons[index])
        row.update(episode.as_row())
        rows.append(row)
    return rows


def _stance_v2_report(
    species: str,
    stage: "int | str",
    *,
    curriculum: dict[str, Any],
    env_class: Any,
    env_kwargs: dict[str, Any],
    thresholds: StanceV2Thresholds,
    predict: Any,
    description: str,
    zero_action: bool,
    model_path: str | None,
    vecnorm_path: str | None,
    episodes: int,
    seed: int,
    horizon: int,
    control_dt: float,
    plant_identity: Any,
    allow_legacy_plant: bool,
) -> dict[str, Any]:
    """Roll and score the ``stance_quality/v2`` panel, and the statue panel its ratios need.

    The policy panel runs through the floor-truth recorder (``run_panel(...,
    floor_truth=True)``).  When the gate declares a statue-relative
    criterion, a zero-action panel is rolled through the same recorder on
    the same env kwargs and seeds, and reduced to the
    :class:`~environments.shared.curriculum.stance_gate_v2.StatueReference`
    the ratios divide by -- except when the policy IS the statue
    (``zero_action``), whose own panel is that reference, rolled once.  The
    verdict is :func:`~environments.shared.curriculum.stance_gate_v2.evaluate_stance_v2_gate`,
    the same function the judge re-derives the verdict with and publication
    re-derives it from the panel CSV with.

    Recorded so a reader can refuse what does not describe this gate:
    ``scored_gate_kind``; ``thresholds`` as the gate view projects them
    (``settle_steps`` included, unlike the v1 report); the ``measurement``
    manifest and its sha256; the ``handoff`` digests of the checkpoint and
    VecNormalize sidecar that were scored (``None`` for the statue); the
    ``task_sha256`` of the ``[env]`` block and plant the panel rolled,
    derived the one way training derives the fingerprint it records
    (``task_fingerprint.stage_task_fingerprint``), so the judge can refuse a
    panel rolled under another task than the stage ran; the ``statue``
    block with its own episode rows; and every episode's metrics and
    reasons.
    """
    from ..gait.constants import measurement_sha256
    from ..result_bundle.hashing import sha256_file
    from ..species_names import resolve_species_id
    from ..task_fingerprint import stage_task_fingerprint

    # Derived before anything is rolled, under the manifest id training
    # derives it under (``raptor`` is ``velociraptor``): a task that cannot be
    # fingerprinted (a live command mode without its manifest) could only
    # produce a report the judge refuses.
    task_sha256 = stage_task_fingerprint(
        resolve_species_id(species), stage, env_kwargs=env_kwargs, plant_identity=plant_identity
    )["task_sha256"]
    result = run_panel(
        species,
        predict=predict,
        episodes=episodes,
        seed=seed,
        settle_steps=thresholds.settle_steps,
        horizon=horizon,
        env_kwargs=env_kwargs,
        plant_identity=plant_identity,
        control_dt=control_dt,
        floor_truth=True,
    )
    metrics: list[StanceEpisodeMetrics] = result["stance_metrics"]
    measurement: dict[str, Any] = result["measurement"]

    statue = None
    statue_block: dict[str, Any] | None = None
    if thresholds.declares_statue_criteria():
        if zero_action:
            statue_result = result
        else:
            statue_result = run_panel(
                species,
                predict=_zero_action_predict(env_class, env_kwargs),
                episodes=episodes,
                seed=seed,
                settle_steps=thresholds.settle_steps,
                horizon=horizon,
                env_kwargs=env_kwargs,
                plant_identity=plant_identity,
                control_dt=control_dt,
                floor_truth=True,
            )
        statue_metrics: list[StanceEpisodeMetrics] = statue_result["stance_metrics"]
        statue = statue_reference(statue_metrics, horizon=horizon)
        statue_block = {
            "policy": _ZERO_ACTION_DESCRIPTION,
            # The statue's panel is the policy's own when the policy commands
            # zero: the same env kwargs, seeds and recorder, rolled once.  The
            # judge refuses a checkpoint report that says True here.
            "reused_policy_panel": zero_action,
            "seed": seed,
            "reference": statue.as_dict(),
            "episode_evidence": _stance_v2_episode_rows(
                statue_metrics, None, statue_result["episodes"], seed=seed, horizon=horizon
            ),
        }

    verdict = evaluate_stance_v2_gate(metrics, thresholds, horizon=horizon, statue=statue, control_dt=control_dt)
    panel = result["panel"]
    return {
        "schema": STANCE_V2_REPORT_SCHEMA,
        "species": species,
        "stage": stage,
        "gate_kind": curriculum.get("gate_kind"),
        "scored_gate_kind": STANCE_GATE_V2_KIND,
        "policy": description,
        "episodes": episodes,
        "seed": seed,
        "settle_steps": thresholds.settle_steps,
        "horizon": horizon,
        "control_dt": control_dt,
        "passed": verdict.passed,
        "failures": list(verdict.failures),
        # The gate view's projection of the declared block, which is exactly
        # what reporting.gates compares against the block the stage is judged
        # under (gate_config_differences) -- so a report scored under any
        # other threshold, settle_steps included, is refused there.
        "thresholds": gate_config_view(curriculum)["thresholds"],
        "measurement": measurement,
        "measurement_sha256": measurement_sha256(measurement),
        # The pair that was scored, by content: the judge binds these to the
        # handoff select_handoff_checkpoint picks, and the panel CSV stamps
        # them on every row for publication to bind to the certified pair.
        "handoff": {
            "checkpoint": None if zero_action or model_path is None else Path(model_path).name,
            "checkpoint_sha256": None if zero_action or model_path is None else sha256_file(model_path),
            "normalization_sha256": None if zero_action or vecnorm_path is None else sha256_file(vecnorm_path),
        },
        "task_sha256": task_sha256,
        "statue": statue_block,
        "result": verdict.as_dict(),
        "metrics": {
            "reward_mean": float(result["rewards"].mean()),
            "reward_std": float(result["rewards"].std()),
            "episode_length_mean": float(result["lengths"].mean()),
            "full_horizon_fraction": verdict.full_horizon_fraction,
            "n_clean": verdict.n_clean,
            "clean_fraction": verdict.clean_fraction,
            "clean_lcb": verdict.clean_lcb,
            # Not gated: the v1 touch reading of the same rollout, kept so a
            # reader can see where touch and floor truth disagree.
            "mean_unsupported_duty": panel.mean_unsupported_duty,
            "bilateral_support_duty": result["bilateral_duty"],
            "single_support_duty": result["single_duty"],
        },
        "checkpoint_plant_validated": None if zero_action else not allow_legacy_plant,
        # Always None: a probe on a v2 stage takes the v1 path above, so a
        # v3 report is never a probe -- but the keys stay (every
        # `_PROBE_MARKERS` key), so probe_stem and every reader of them
        # answer the same way for both shapes.
        "filter_actions_hz": None,
        "hold_constant": None,
        "impulse": None,
        "spawn_yaw": None,
        "action": result.get("action", {}),
        "terminations": result["terminations"],
        "reward_components": result["components"],
        "episode_evidence": _stance_v2_episode_rows(
            metrics, verdict.episode_reasons, result["episodes"], seed=seed, horizon=horizon
        ),
    }


#: Report keys that mark a rollout as a PROBE rather than a verdict, mapped to
#: the filename stem that probe writes under. Each names a modification applied
#: between the policy and the plant, so a report carrying any of them scored
#: something other than the checkpoint -- and must never land on the real
#: report's filenames or supply the evidence a bundle is certified from.
#:
#: A registry rather than a chain of ``if``s because the failure mode is
#: forgetting to extend the check: the filter probe's guard was written as a
#: single ``is not None`` test, and a second probe added beside it would have
#: silently inherited the certification filenames.
_PROBE_MARKERS: dict[str, str] = {
    "filter_actions_hz": "stance_gate_probe_filtered",
    "hold_constant": "stance_gate_probe_constant",
    "impulse": "stance_gate_probe_impulse",
    "spawn_yaw": "stance_gate_probe_heading",
}


def probe_stem(report: dict[str, Any]) -> str | None:
    """The filename stem a probe report writes under, or ``None`` if it is a verdict."""
    for key, stem in _PROBE_MARKERS.items():
        if report.get(key) is not None:
            return stem
    return None


def _probe_banner(report: dict[str, Any]) -> list[str]:
    """The warning a probe report must carry above its verdict line."""
    if report.get("filter_actions_hz") is not None:
        what = f"actions low-passed at {report['filter_actions_hz']:g} Hz before reaching the plant."
    elif report.get("hold_constant") is not None:
        hold = report["hold_constant"]
        what = (
            "the policy ran the whole episode; this row is the unmodified control."
            if not hold.get("holds", True)
            else (
                f"the commanded action was frozen to a constant from step {hold['handoff_steps']}"
                + (f", ramped in over {hold['ramp_steps']} steps." if hold.get("ramp_steps") else ".")
            )
        )
    elif report.get("impulse") is not None:
        shove = report["impulse"]
        return [
            f"PROBE: a {shove['speed']:.2f} m/s {shove['axis_label']} impulse at step {shove['step']}.",
            "This scores the unmodified policy on a MODIFIED TASK -- stage 1 has no disturbance,",
            "so the gate below was never defined with one. It is not a gate result.",
            "",
        ]
    elif report.get("spawn_yaw") is not None:
        turn = report["spawn_yaw"]
        return [
            f"PROBE: the spawn turned {turn['yaw_deg']:+g} deg about vertical"
            + (
                " with the prey and the episode's reference direction."
                if turn["whole_scene"]
                else " (the animal only)."
            ),
            "This scores the unmodified policy from a MODIFIED START -- the stance stage always spawns",
            "facing +x, so the gate below was never defined from another heading. It is not a gate result.",
            "",
        ]
    else:
        return []
    return [
        f"PROBE: {what}",
        "This scores a MODIFIED policy. The verdict below is not a gate result for this checkpoint.",
        "",
    ]


def _kind_note(report: dict[str, Any]) -> list[str]:
    """The NOTE a report carries when the criteria it scored are not the ones its stage advances on."""
    declared = report["gate_kind"]
    scored = report.get("scored_gate_kind", STANCE_GATE_KIND)
    if declared == scored:
        return []
    if declared in STANCE_GATE_KINDS:
        return [
            f"NOTE: {species_display_name(report['species'])} {stage_display_name(report['stage'])} "
            f"(stage {report['stage']}) declares gate_kind {declared!r}; this "
            f"report scored the {scored!r} criteria, which are reported but are not what this stage "
            "advances on.",
            "",
        ]
    return [
        f"NOTE: {species_display_name(report['species'])} {stage_display_name(report['stage'])} "
        f"(stage {report['stage']}) declares gate_kind {declared!r}, not a "
        "stance kind. The stance criteria below are reported but are not what this stage advances on.",
        "",
    ]


def render_stance_gate_report(report: dict[str, Any]) -> str:
    """Render a report dict as the human-readable text form."""
    if report.get("schema") == STANCE_HEADING_REPORT_SCHEMA:
        return render_stance_heading_report(report)
    if report.get("scored_gate_kind") == STANCE_GATE_V2_KIND:
        return _render_stance_v2_report(report)
    thresholds = report["thresholds"]
    metrics = report["metrics"]
    lines: list[str] = _kind_note(report)
    # Before the verdict, not after: a probe rollout scores a MODIFIED policy,
    # so its PASS/FAIL is not a statement about the checkpoint and a reader must
    # not reach the verdict line without knowing that.
    lines += _probe_banner(report)
    measured = report["horizon"] - report["settle_steps"]
    lines += [
        f"policy              {report['policy']}",
        f"stage               {species_display_name(report['species'])} — {stage_display_name(report['stage'])} "
        f"(stage {report['stage']}, {report['gate_kind']})",
        f"panel               {report['episodes']} episodes, "
        f"seeds {report['seed']}-{report['seed'] + report['episodes'] - 1}",
        f"settle_steps        {report['settle_steps']} (duty measured over the remaining {measured})",
        "",
        f"reward                 {metrics['reward_mean']:9.1f} +/- {metrics['reward_std']:.1f}",
        f"episode length         {metrics['episode_length_mean']:9.1f}",
        f"full_horizon_fraction  {metrics['full_horizon_fraction']:9.4f}   "
        f"(>= {thresholds['min_full_horizon_fraction']:.4f})",
        f"mean_unsupported_duty  {metrics['mean_unsupported_duty']:9.4f}   "
        f"(<= {thresholds['max_unsupported_duty']:.4f})",
        f"unsupported_duty_ucb   {metrics['unsupported_duty_ucb']:9.4f}   "
        f"(<= {thresholds['max_unsupported_duty_ucb']:.4f})",
        f"duty episodes          {metrics['n_duty_episodes']:9d}   (full-horizon episodes only)",
        f"  bilateral support    {metrics['bilateral_support_duty']:9.4f}   (statue 0.998, not gated)",
        f"  single support       {metrics['single_support_duty']:9.4f}   (statue 0.002, not gated)",
        f"terminations           {report['terminations']}",
    ]
    components = report["reward_components"]
    if components:
        lines += ["", "reward per episode, by term (largest magnitude first):"]
        for key in sorted(components, key=lambda k: -abs(components[k])):
            if abs(components[key]) > 0.05:
                lines.append(f"  {key:28s} {components[key]:10.2f}")
    action = report.get("action") or {}
    if action:
        summary = f"DC {action['dc_rms']:.3f} rms, AC {action['ac_rms']:.3f} rms, ~{action['effective_freq_hz']:.1f} Hz"
        if "dc_rms_deg" in action:
            summary += f"; {action['dc_rms_deg']:.1f}deg rms across angular position controls"
        lines += [
            "",
            f"commanded action, post-settle ({summary}):",
            "  Normalised DC is the mean policy action; degree DC is the applied control",
            "  target relative to action zero. AC is variation around each mean.",
        ]
        per_actuator = action.get("per_actuator") or []
        has_degrees = any("dc_deg" in entry for entry in per_actuator)
        if has_degrees:
            lines += [
                "  Ordered by DEGREES, not by normalised action: |action| = 1 maps onto each",
                "  actuator's own control span, so saturation can move a target 8deg or 37.5deg",
                "  from its nominal target. Ranking by the normalised value pools those and",
                "  hides which targets moved. Degree fields are emitted for verified angular",
                "  position controls only; motors retain native control units in the JSON.",
            ]
        # Largest physical offset first when it can be computed. That ordering
        # is the point: sorted by |dc| this table put twelve joints at exactly
        # +/-1.000 at the top, of which the most extreme-looking (the tail, at
        # 12deg of travel) was later measured to be mechanically inert.
        key = (lambda e: -abs(e.get("dc_deg", 0.0))) if has_degrees else (lambda e: -abs(e["dc"]))
        for entry in sorted(per_actuator, key=key)[:12]:
            line = f"  {entry['joint']:22s} DC {entry['dc']:+7.3f}   AC {entry['ac_rms']:6.3f}"
            if "dc_deg" in entry:
                line += (
                    f"   {entry['dc_deg']:+7.1f}deg   +/-{entry['ac_rms_deg']:5.1f}deg"
                    f"   of {entry['range_deg']:5.1f}deg"
                )
            lines.append(line)
        if len(per_actuator) > 12:
            lines.append(f"  ... {len(per_actuator) - 12} more (full list in the JSON)")
        # Contract check: under the home-keyframe residual mapping action zero
        # must be the named home CONTROL.  Keep that distinct from a deliberate
        # control-vs-qpos preload, which is a plant property rather than policy
        # displacement or an action-mapping error.
        offset = [entry for entry in per_actuator if abs(entry.get("zero_offset_deg", 0.0)) > 0.05]
        if offset:
            worst = max(offset, key=lambda e: abs(e["zero_offset_deg"]))
            lines.append(
                f"  NOTE: action = 0 differs from the named home control for {len(offset)} of "
                f"{len(per_actuator)} actuators (worst {worst['joint']} "
                f"{worst['zero_offset_deg']:+.1f}deg); inspect the action-mapping contract."
            )
        preload = [entry for entry in per_actuator if abs(entry.get("home_preload_deg", 0.0)) > 0.05]
        if preload:
            worst = max(preload, key=lambda e: abs(e["home_preload_deg"]))
            lines.append(
                f"  Nominal control preload differs from reset joint pose for {len(preload)} of "
                f"{len(per_actuator)} actuators (worst {worst['joint']} "
                f"{worst['home_preload_deg']:+.1f}deg); this is not policy displacement."
            )
    lines += ["", f"GATE: {'PASS' if report['passed'] else 'FAIL'}"]
    lines += [f"  - {failure}" for failure in report["failures"]]
    return "\n".join(lines)


def _finite_values(rows: list[dict[str, Any]], field_name: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(field_name)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
            values.append(float(value))
    return values


#: How the text report names each hop-or-fall key in the rail's line.
_HOP_OR_FALL_WORDS: dict[str, str] = {
    "min_all_feet_support": "support",
    "max_touchdown_rate": "touchdowns",
    "max_window_displacement_m": "drift",
    "max_actuator_saturation_fraction": "saturation",
    "max_window_airborne_substeps": "window airborne substeps",
    "max_window_peak_floor_force_bw": "window peak floor force",
}


def _render_stance_v2_report(report: dict[str, Any]) -> str:
    """The text form of a ``stance_quality/v2`` report: the bound, then each criterion, then the episodes.

    One line per declared criterion with its bar, the episodes it failed and
    the panel's median and worst value, so a near miss and a gross failure
    read differently; then the statue reference, the report-only floor-truth
    readings, and the first unclean episodes with every reason.
    """
    thresholds = report["thresholds"]
    result = report["result"]
    metrics = report["metrics"]
    rows = report.get("episode_evidence") or []
    n = int(result["n_episodes"])
    measured = report["horizon"] - report["settle_steps"]
    lines = _kind_note(report)
    lines += [
        f"policy              {report['policy']}",
        f"stage               {species_display_name(report['species'])} — {stage_display_name(report['stage'])} "
        f"(stage {report['stage']}, {report['scored_gate_kind']}, floor truth)",
        f"panel               {report['episodes']} episodes, "
        f"seeds {report['seed']}-{report['seed'] + report['episodes'] - 1}",
        f"settle_steps        {report['settle_steps']} (window criteria over the remaining {measured}; settle "
        f"criteria after the {report['measurement']['constants']['spawn_grace_s']:g} s spawn grace)",
        f"measurement         {report['measurement']['constants']['measurement_version']} "
        f"({report['measurement_sha256']})",
        f"task                {report.get('task_sha256')}",
        "",
        f"clean episodes         {result['n_clean']:4d} / {n}   LCB95 {_fmt(result['clean_lcb'], '.4f')}   "
        f"(>= {float(thresholds['min_clean_stance_lcb']):.4f}, min_eval_episodes "
        f"{thresholds['min_eval_episodes']})",
        f"full_horizon_fraction  {_fmt(result['full_horizon_fraction'], '9.4f')}",
        f"reward                 {metrics['reward_mean']:9.1f} +/- {metrics['reward_std']:.1f}",
        f"terminations           {report['terminations']}",
        "",
        "per-episode criteria (fails = episodes failing it; median and worst over the panel):",
    ]
    failing = result.get("criterion_failures") or {}
    lines.append(
        f"  {HORIZON_REASON:36s} reach {report['horizon']} steps         fails {failing.get(HORIZON_REASON, 0):3d}/{n}"
    )
    for key, metric, direction in EPISODE_CRITERIA:
        if key not in thresholds:
            continue
        values = _finite_values(rows, metric)
        median = float(np.median(values)) if values else float("nan")
        worst = (min(values) if direction == "min" else max(values)) if values else float("nan")
        unmeasured = len(rows) - len(values)
        lines.append(
            f"  {key:36s} {'>=' if direction == 'min' else '<='} {float(thresholds[key]):<12.6g} fails "
            f"{failing.get(key, 0):3d}/{n}   median {median:.4g}, worst {worst:.4g}"
            + (f", {unmeasured} unmeasured" if unmeasured else "")
        )
    if "min_foot_load_share_statue_ratio" in thresholds:
        key = "min_foot_load_share_statue_ratio"
        lines.append(f"  {key:36s} >= {float(thresholds[key]):<12.6g} fails {failing.get(key, 0):3d}/{n}")
    for key in ("min_avg_reward", "min_avg_reward_statue_ratio", "min_full_horizon_fraction"):
        if key in thresholds:
            lines.append(f"  {key:36s} {float(thresholds[key]):g} (panel rail)")
    if "max_hop_or_fall_episodes" in thresholds:
        key = "max_hop_or_fall_episodes"
        # The hop keys the block declares: an undeclared one fails no episode.
        hop_words = [_HOP_OR_FALL_WORDS.get(hop, hop) for hop in HOP_OR_FALL_KEYS if hop in thresholds]
        lines.append(
            f"  {key:36s} <= {int(thresholds[key])} (panel rail: {result.get('hop_or_fall_episodes')} episodes end "
            f"early or fail {', '.join(hop_words[:-1])} or {hop_words[-1]})"
        )
    statue = report.get("statue")
    if statue:
        reference = statue["reference"]
        shares = ", ".join(_fmt(share, ".3f") for share in reference["foot_load_share"])
        lines += [
            "",
            f"statue reference ({'the policy panel itself' if statue.get('reused_policy_panel') else 'zero action'}, "
            f"same seeds): {reference['n_full_horizon']}/{reference['n_episodes']} full-horizon, mean reward "
            f"{_fmt(reference['mean_reward'], '.1f')}, foot load shares [{shares}]",
        ]
    # A reading a declared criterion gates is listed with the criteria above, not here.
    gated_metrics = {metric for key, metric, _ in EPISODE_CRITERIA if key in thresholds}
    report_only = tuple(
        (metric, label)
        for metric, label in (
            ("episode_yaw_change_deg", "yaw change over the episode (deg)"),
            ("touch_floor_agreement", "touch agrees with floor truth"),
            ("spawn_peak_floor_force_bw", "spawn-grace peak floor force (BW)"),
            ("settle_touchdowns", "settle-window touchdowns"),
            ("max_sole_cop_fore_aft", "pad CoP fore-aft (|.| / half-length)"),
            ("window_airborne_substeps", "window both-feet-unloaded substeps"),
            ("window_peak_floor_force_bw", "window peak floor force (BW)"),
        )
        if metric not in gated_metrics
    )
    lines += ["", "report-only (not gated; median and max over the panel):"]
    for metric, label in report_only:
        values = _finite_values(rows, metric)
        if values:
            lines.append(f"  {label:36s} {float(np.median(values)):.4g}, max {max(values):.4g}")
    lines.append(f"  {'touch unsupported duty (v1 measure)':36s} {_fmt(metrics.get('mean_unsupported_duty'), '.4f')}")
    unclean = [row for row in rows if not row.get("clean", True)]
    if unclean:
        lines += ["", f"unclean episodes ({len(unclean)}; first {min(len(unclean), 5)} shown):"]
        for row in unclean[:5]:
            lines.append(f"  seed {row['seed']}: " + "; ".join(row.get("reasons") or ()))
    lines += ["", f"GATE: {'PASS' if report['passed'] else 'FAIL'}"]
    lines += [f"  - {failure}" for failure in report["failures"]]
    return "\n".join(lines)


def _fmt(value: Any, spec: str) -> str:
    """*value* formatted with *spec*, or ``nan`` for an unmeasured (``None``/non-finite) value."""
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "nan"
    return format(float(value), spec)


def _json_safe(value: Any) -> Any:
    """Replace non-finite floats with ``null``, recursively.

    ``json.dumps`` writes bare ``NaN`` and ``Infinity`` tokens, which Python
    reads back but RFC 8259 does not permit -- ``jq``, ``JSON.parse`` and Go's
    ``encoding/json`` all reject them.  That bit exactly when it hurt most:
    the gate's "unmeasurable" sentinel is ``+inf`` and the ungated shares are
    ``NaN``, so it was the FAILING panel, the one worth inspecting, that
    produced the unparseable file -- while the docstring below promises this
    form exists so later tooling can read it without parsing prose.

    ``null`` rather than a string, so a consumer's numeric parse fails loudly
    instead of silently comparing ``"inf"``.  The human-readable text form
    still prints ``inf``, and ``failures`` says which criterion was
    unmeasurable.
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def write_stance_gate_report(stage_dir: "str | Path", report: dict[str, Any]) -> dict[str, Path]:
    """Write the report beside the stage's other artifacts.

    Two forms on purpose: the text is what a human opens next to
    ``stage_summary.txt``, and the JSON is what later tooling can read without
    parsing prose.

    Three forms, and the third is load-bearing. ``stance_panel_selected.csv``
    is the per-episode duty evidence ``result_bundle.evidence`` needs to
    certify a stance-gated stage: it used to refuse such bundles outright
    because ``evaluation_selected.csv`` records reward and length but no duty,
    and certifying on the reward rail alone would pass the statue. The refusal
    was correct and it also made the milestone unreachable -- a run that
    genuinely cleared the stance gate still could not produce a publishable
    bundle. This file is what lifts it, and it is a CSV rather than the JSON's
    ``episode_evidence`` so the auditor recomputes the panel from the same
    kind of per-episode record it already uses for reward and length.

    The panel is a different evaluation from ``evaluation_selected.csv``: 40
    episodes at the panel seeds versus 30 at the publication seed. That is
    deliberate -- ``min_eval_episodes`` is the sample size the bound's power
    is specified at -- so it gets its own file rather than columns bolted
    onto a file with a different episode count.
    """
    if report.get("schema") == STANCE_HEADING_REPORT_SCHEMA:
        # Decision 7 of 2026-10-11 (D-D32): a heading-kind stage writes its certificate's files only, never a
        # stance_gate_report.* or stance_panel_selected.csv a reader could take for a yaw-zero verdict.
        raise ValueError(
            "a six-heading certificate is written by write_stance_heading_report, not as a stance gate report"
        )
    directory = Path(stage_dir)
    directory.mkdir(parents=True, exist_ok=True)
    # A filtered rollout scored a MODIFIED policy, so it gets its own filenames
    # and never the certification ones. Derived here rather than passed in so a
    # caller cannot land a probe on top of the real verdict by forgetting an
    # argument -- and `stance_panel_selected.csv` is skipped entirely below,
    # because that file is the per-episode evidence `result_bundle.evidence`
    # certifies a stance-gated stage from. Writing a filtered panel there would
    # certify a policy that was never run.
    stem = probe_stem(report)
    probe = stem is not None
    stem = stem or "stance_gate_report"
    text_path = directory / f"{stem}.txt"
    json_path = directory / f"{stem}.json"
    atomic_write_text(text_path, render_stance_gate_report(report) + "\n")
    # allow_nan=False turns any non-finite value _json_safe missed into a
    # ValueError here rather than an unparseable artifact on Drive.
    atomic_write_json(json_path, _json_safe(report), sort_keys=True, allow_nan=False)
    written = {"stance_gate_report_txt": text_path, "stance_gate_report_json": json_path}
    if probe:
        return written
    panel_path = write_stance_panel_evidence(directory, report)
    if panel_path is not None:
        written["stance_panel_csv"] = panel_path
    return written


# ── the six-heading certificate (``stance_quality_heading/v1``, D-D32) ──────


def heading_protocol_refusals(env_kwargs: Mapping[str, Any], model: Any = None) -> list[str]:
    """Every input of the stage the whole-scene turn would not turn; empty when the protocol applies.

    :func:`_apply_spawn_yaw` turns the animal, every mocap body and the
    episode's reference direction (protocol ``whole_scene_yaw/v1``).  It does
    not turn a live command controller's world heading, a scheduled push or
    a heightfield, so a stage with any of them would score a different task
    at each heading: ``command_mode`` must be ``"none"``,
    ``perturbation_capture_velocity_multiple`` 0, and the compiled *model*
    must hold no heightfield (``model.nhfield``; skipped when ``None``, so
    the ``[env]`` checks can run before an environment is built).  Refused
    before anything is rolled.
    """
    refusals: list[str] = []
    command_mode = env_kwargs.get("command_mode", "none")
    if command_mode != "none":
        refusals.append(
            f"command_mode is {command_mode!r}: the turn does not rotate the command controller's world heading"
        )
    pushes = env_kwargs.get("perturbation_capture_velocity_multiple", 0.0)
    if pushes:
        refusals.append(
            f"perturbation_capture_velocity_multiple is {pushes!r}: the turn does not rotate the world push schedule"
        )
    heightfields = 0 if model is None else int(getattr(model, "nhfield", 0) or 0)
    if heightfields:
        refusals.append(f"the model holds {heightfields} heightfield(s): the turn does not rotate the terrain")
    return refusals


def _normalization_record(predict: Any, *, zero_action: bool) -> dict[str, Any]:
    """How the scored policy's observations were normalised, as :func:`_load_policy` recorded it."""
    if zero_action:
        return {"mode": "none (zero action)"}
    record = getattr(predict, "normalization", None)
    return dict(record) if isinstance(record, Mapping) else {"mode": "unrecorded"}


def build_stance_heading_report(
    species: str,
    stage: "int | str",
    *,
    stage_config: dict[str, Any],
    model_path: str | None = None,
    vecnorm_path: str | None = None,
    zero_action: bool = False,
    episodes: int | None = None,
    seed: int = PUBLICATION_SEED_START,
    allow_legacy_plant: bool = False,
) -> dict[str, Any]:
    """Roll and score the six-heading stance certificate (``stance_quality_heading/v1``, D-D32).

    For each heading of :data:`~environments.shared.curriculum.stance_heading_gate.HEADING_GRID_DEG`
    the policy panel is rolled through the floor-truth recorder on a fresh
    environment with the whole scene turned (``run_panel(...,
    floor_truth=True, spawn_yaw=SpawnYaw(yaw, whole_scene=True))``; at 0
    degrees the turn is the identity and the cell is the v2 panel bit for
    bit), and, when the gate declares a statue-relative criterion, the
    zero-action statue on the same seeds at the same heading -- the policy's
    own panel when the policy IS the statue (``zero_action``).  Each heading
    is judged against its own statue; the verdict is
    :func:`~environments.shared.curriculum.stance_heading_gate.evaluate_stance_heading_gate`,
    the function the judge, publication and reuse re-derive it with.

    Checked before anything is rolled: the stage declares the kind, its
    thresholds read strictly with a settle window past the spawn grace, and
    the turn applies to its task (:func:`heading_protocol_refusals`).  A
    rollout that raises at any heading raises :class:`StanceGateReportError`
    naming the heading, and nothing is returned: a heading that was not
    measured fails the certificate, never falls back to another.

    Records everything a reader binds: the identity (``schema``,
    ``scored_gate_kind``, ``protocol``, ``heading_grid_deg``, ``seeds``),
    the rollout (``policy_deterministic``, ``normalization``,
    ``settle_steps``, ``horizon``, ``control_dt``), the verdict
    (``passed``, ``failures``, ``result``) and the binding (``thresholds``
    as the gate view projects them, ``gate_sha256``, the ``measurement``
    manifest -- identical at every heading -- and its digest, the
    ``handoff`` digests, ``task_sha256``, the plant identity and
    ``checkpoint_plant_validated``), every probe marker as ``None``, and per
    heading the v2 result, the action and termination summaries and every
    episode's row (``seed`` and ``yaw_deg`` included) with the statue block
    and its rows.
    """
    from ..curriculum.stance_gate_v2 import StanceV2Result
    from ..gait.constants import measurement_sha256
    from ..plant_contract import current_plant_identity
    from ..result_bundle.hashing import sha256_file
    from ..species_names import resolve_species_id
    from ..task_fingerprint import stage_task_fingerprint

    curriculum = stage_config["curriculum_kwargs"]
    if curriculum.get("gate_kind") != STANCE_HEADING_GATE_KIND:
        raise ValueError(
            f"stage {stage} declares gate_kind {curriculum.get('gate_kind')!r}, not {STANCE_HEADING_GATE_KIND!r}; "
            "build_stance_gate_report scores it"
        )
    if episodes is not None and episodes < 1:
        raise ValueError(f"episodes must be at least 1 if given, got {episodes}")
    env_kwargs = dict(stage_config["env_kwargs"])
    horizon = int(env_kwargs.get("max_episode_steps", 1000))
    control_dt = float(env_kwargs.get("timestep", 0.002)) * int(env_kwargs.get("frame_skip", 5))
    thresholds = StanceV2Thresholds.from_curriculum(curriculum)
    thresholds.validate_settle_window(control_dt)
    n = thresholds.min_eval_episodes if episodes is None else episodes
    grid = validate_heading_grid(HEADING_GRID_DEG)
    seeds = [seed + index for index in range(n)]

    env_class = SPECIES_FACTORIES[species]().env_class
    refusals = heading_protocol_refusals(env_kwargs)
    if not refusals:
        probe = env_class(**env_kwargs)
        try:
            refusals = heading_protocol_refusals(env_kwargs, probe.unwrapped.model)
        finally:
            probe.close()
    if refusals:
        raise StanceGateReportError(
            f"stage {stage} cannot be certified under {HEADING_PROTOCOL}: {'; '.join(refusals)}. The certificate "
            "turns the animal, the prey and the episode's reference direction; a task with other world-frame "
            "inputs needs a protocol that turns them too."
        )
    plant_identity = current_plant_identity(species)
    task_sha256 = stage_task_fingerprint(
        resolve_species_id(species), stage, env_kwargs=env_kwargs, plant_identity=plant_identity
    )["task_sha256"]

    if zero_action:
        predict = _zero_action_predict(env_class, env_kwargs)
        description = _ZERO_ACTION_DESCRIPTION
    else:
        if model_path is None:
            raise ValueError("model_path is required unless zero_action is set")
        predict, note = _load_policy(
            model_path,
            vecnorm_path,
            lambda: env_class(**env_kwargs),
            plant_identity=plant_identity,
            allow_legacy_plant=allow_legacy_plant,
        )
        description = f"{Path(model_path).name} — {note}"
    statue_predict = None if zero_action else _zero_action_predict(env_class, env_kwargs)

    def panel(policy: Any, yaw: float) -> dict[str, Any]:
        return run_panel(
            species,
            predict=policy,
            episodes=n,
            seed=seed,
            settle_steps=thresholds.settle_steps,
            horizon=horizon,
            env_kwargs=env_kwargs,
            plant_identity=plant_identity,
            control_dt=control_dt,
            floor_truth=True,
            spawn_yaw=SpawnYaw(yaw, whole_scene=True),
        )

    rolled: list[tuple[float, dict[str, Any], dict[str, Any] | None]] = []
    measurement: dict[str, Any] | None = None
    for index, yaw in enumerate(grid):
        logger.info(
            "Heading certificate for %s stage %s: heading %+g deg (%d of %d), %d seeds%s",
            species,
            stage,
            yaw,
            index + 1,
            len(grid),
            n,
            " and the statue" if thresholds.declares_statue_criteria() and not zero_action else "",
        )
        try:
            result = panel(predict, yaw)
            statue_result: dict[str, Any] | None = None
            if thresholds.declares_statue_criteria():
                statue_result = result if zero_action else panel(statue_predict, yaw)
        except StanceGateReportError:
            raise
        except Exception as exc:  # noqa: BLE001 - re-raised as the report's own error, naming the heading
            raise StanceGateReportError(
                f"the heading certificate's rollout at {yaw:+g} deg raised {type(exc).__name__}: {exc}; no "
                "certificate was produced, and a heading that was not measured is never filled from another"
            ) from exc
        if measurement is None:
            measurement = result["measurement"]
        elif result["measurement"] != measurement:
            raise StanceGateReportError(
                f"the measurement manifest at {yaw:+g} deg differs from the one at {grid[0]:+g} deg; one "
                "certificate is measured under one definition"
            )
        rolled.append((yaw, result, statue_result))
    if measurement is None:  # unreachable: the protocol grid is never empty
        raise StanceGateReportError("the heading certificate rolled no heading")

    policy_rows = [
        HeadingEpisode(seed + index, yaw, metrics)
        for yaw, result, _ in rolled
        for index, metrics in enumerate(result["stance_metrics"])
    ]
    statue_by_heading: dict[float, StatueReference | None] = {
        yaw: None if statue is None else statue_reference(statue["stance_metrics"], horizon=horizon)
        for yaw, _, statue in rolled
    }
    verdict = evaluate_stance_heading_gate(
        policy_rows,
        thresholds,
        horizon=horizon,
        seeds=seeds,
        statue_by_heading=statue_by_heading,
        control_dt=control_dt,
        grid=grid,
    )

    headings: list[dict[str, Any]] = []
    for yaw, result, statue in rolled:
        heading_verdict: StanceV2Result | None = verdict.heading_result(yaw)
        reasons = None if heading_verdict is None else heading_verdict.episode_reasons
        reference = statue_by_heading[yaw]
        statue_block: dict[str, Any] | None = None
        if statue is not None and reference is not None:
            statue_block = {
                "policy": _ZERO_ACTION_DESCRIPTION,
                "reused_policy_panel": zero_action,
                "seed": seed,
                "reference": reference.as_dict(),
                "episode_evidence": _heading_rows(
                    _stance_v2_episode_rows(
                        statue["stance_metrics"], None, statue["episodes"], seed=seed, horizon=horizon
                    ),
                    yaw,
                ),
            }
        panel_summary = result["panel"]
        headings.append(
            {
                "yaw_deg": yaw,
                "turn": SpawnYaw(yaw, whole_scene=True).as_dict(),
                "result": None if heading_verdict is None else heading_verdict.as_dict(),
                "metrics": {
                    "reward_mean": float(result["rewards"].mean()),
                    "reward_std": float(result["rewards"].std()),
                    "episode_length_mean": float(result["lengths"].mean()),
                    "full_horizon_fraction": None if heading_verdict is None else heading_verdict.full_horizon_fraction,
                    "n_clean": None if heading_verdict is None else heading_verdict.n_clean,
                    "clean_fraction": None if heading_verdict is None else heading_verdict.clean_fraction,
                    "clean_lcb": None if heading_verdict is None else heading_verdict.clean_lcb,
                    "mean_unsupported_duty": panel_summary.mean_unsupported_duty,
                    "bilateral_support_duty": result["bilateral_duty"],
                    "single_support_duty": result["single_duty"],
                },
                "action": result.get("action", {}),
                "terminations": result["terminations"],
                "reward_components": result["components"],
                "episode_evidence": _heading_rows(
                    _stance_v2_episode_rows(
                        result["stance_metrics"], reasons, result["episodes"], seed=seed, horizon=horizon
                    ),
                    yaw,
                ),
                "statue": statue_block,
            }
        )

    view = gate_config_view(curriculum)
    return {
        "schema": STANCE_HEADING_REPORT_SCHEMA,
        "species": species,
        "stage": stage,
        "gate_kind": curriculum.get("gate_kind"),
        "scored_gate_kind": STANCE_HEADING_GATE_KIND,
        "protocol": HEADING_PROTOCOL,
        "heading_grid_deg": list(grid),
        "seeds": seeds,
        "seed": seed,
        "episodes": n,
        "policy": description,
        # SB3's deterministic predict (_load_policy) or the constant statue: every turn of a seed sees the same
        # policy, so the rows of one seed differ only by the turn.
        "policy_deterministic": True,
        "normalization": _normalization_record(predict, zero_action=zero_action),
        "settle_steps": thresholds.settle_steps,
        "horizon": horizon,
        "control_dt": control_dt,
        "passed": verdict.passed,
        "failures": list(verdict.failures),
        "result": verdict.as_dict(),
        "thresholds": view["thresholds"],
        "gate_sha256": gate_config_sha256(view),
        "measurement": measurement,
        "measurement_sha256": measurement_sha256(measurement),
        "handoff": {
            "checkpoint": None if zero_action or model_path is None else Path(model_path).name,
            "checkpoint_sha256": None if zero_action or model_path is None else sha256_file(model_path),
            "normalization_sha256": None if zero_action or vecnorm_path is None else sha256_file(vecnorm_path),
        },
        "task_sha256": task_sha256,
        # Recorded for the reader; bound through task_sha256, which fingerprints the plant.
        "plant_identity": plant_identity.to_dict(),
        "checkpoint_plant_validated": None if zero_action else not allow_legacy_plant,
        # Every probe marker, None: a certificate is never a probe, and the readers of these keys answer the same
        # way for every report shape.
        **dict.fromkeys(_PROBE_MARKERS),
        "headings": headings,
    }


def _heading_rows(rows: list[dict[str, Any]], yaw: float) -> list[dict[str, Any]]:
    """*rows* (one heading's evidence rows) with the heading they were rolled at, after the seed."""
    return [{"yaw_deg": yaw, **row} for row in rows]


def heading_zero_panel(report: Mapping[str, Any]) -> dict[str, Any]:
    """The certificate's 0 degree cell, in the ``stance_quality/v2`` report's shape.

    What the stance probes annotate on a heading-kind stage (the constant
    hold reads its ``action`` DC, the impulse its ``settle_steps``) and what
    :func:`render_stance_heading_report` renders in detail.  Never written as
    a report: it carries ``heading_cell_of`` naming the certificate, and its
    ``gate_kind`` is the stage's heading kind, which the v2 judge refuses
    and the heading judge refuses for its schema.  Raises ``ValueError`` on a
    report with no 0 degree cell.
    """
    cell = next(
        (
            heading
            for heading in report.get("headings") or ()
            if isinstance(heading, Mapping) and heading.get("yaw_deg") == 0.0
        ),
        None,
    )
    if cell is None:
        raise ValueError("the heading certificate holds no 0 deg cell")
    result = cell.get("result") or {}
    return {
        "schema": STANCE_V2_REPORT_SCHEMA,
        "heading_cell_of": STANCE_HEADING_REPORT_SCHEMA,
        "heading_cell_deg": 0.0,
        "species": report.get("species"),
        "stage": report.get("stage"),
        "gate_kind": report.get("gate_kind"),
        "scored_gate_kind": STANCE_GATE_V2_KIND,
        "policy": report.get("policy"),
        "episodes": report.get("episodes"),
        "seed": report.get("seed"),
        "settle_steps": report.get("settle_steps"),
        "horizon": report.get("horizon"),
        "control_dt": report.get("control_dt"),
        "passed": bool(result.get("passed")),
        "failures": list(result.get("failures") or ()),
        "thresholds": report.get("thresholds"),
        "measurement": report.get("measurement"),
        "measurement_sha256": report.get("measurement_sha256"),
        "handoff": report.get("handoff"),
        "task_sha256": report.get("task_sha256"),
        "statue": cell.get("statue"),
        "result": result,
        "metrics": cell.get("metrics"),
        "checkpoint_plant_validated": report.get("checkpoint_plant_validated"),
        **dict.fromkeys(_PROBE_MARKERS),
        "action": cell.get("action", {}),
        "terminations": cell.get("terminations", {}),
        "reward_components": cell.get("reward_components", {}),
        "episode_evidence": cell.get("episode_evidence") or [],
    }


def render_stance_heading_report(report: dict[str, Any]) -> str:
    """The text form of a heading certificate: the grouped bound, a row per heading, the seeds that failed, the 0 deg cell.

    The pooled episode count is printed and labelled as not the statistic;
    the 0 degree cell is rendered in full through the v2 renderer, without
    its own verdict line (it is one heading's v2 decision, not the
    certificate's), and the certificate's verdict closes the report.
    """
    result = report["result"]
    thresholds = report["thresholds"]
    seeds = report.get("seeds") or []
    headings = report.get("headings") or []
    grid_text = ", ".join(f"{float(yaw):+g}" for yaw in report.get("heading_grid_deg") or ())
    lines = [
        f"policy              {report['policy']}",
        f"stage               {species_display_name(report['species'])} — {stage_display_name(report['stage'])} "
        f"(stage {report['stage']}, {report['scored_gate_kind']}, floor truth)",
        f"protocol            {report['protocol']}: the whole scene turned about vertical through the root "
        "(the animal, the prey and the reference direction)",
        f"grid                {grid_text} deg; {len(seeds)} seeds "
        + (f"{seeds[0]}-{seeds[-1]}" if seeds else "")
        + " at every heading",
        f"settle_steps        {report['settle_steps']}; horizon {report['horizon']}; control step "
        f"{report['control_dt']:g} s",
        f"measurement         {report['measurement']['constants']['measurement_version']} "
        f"({report['measurement_sha256']})",
        f"task                {report.get('task_sha256')}",
        f"gate                {report.get('gate_sha256')}",
        "",
        f"seeds clean at every heading   {result['n_clean_seeds']:4d} / {result['n_seeds']}   LCB95 "
        f"{_fmt(result['clean_seed_lcb'], '.4f')}   (>= {float(thresholds['min_clean_stance_lcb']):.4f}, "
        f"min_eval_episodes {thresholds['min_eval_episodes']}: the statistic)",
    ]
    pooled = result.get("pooled_episodes") or {}
    if pooled:
        lines.append(
            f"episodes clean, pooled         {pooled.get('n_clean'):4d} / {pooled.get('n_episodes')}   LCB95 "
            f"{_fmt(pooled.get('clean_lcb'), '.4f')}   (NOT the statistic: the turns of a seed are not independent)"
        )
    lines += [
        "",
        "each heading's full v2 decision, against the statue rolled at that heading:",
        f"  {'heading':>8} {'clean':>7} {'LCB95':>7} {'full-hz':>8} {'reward':>9} {'statue fh':>10} "
        f"{'statue reward':>14}  v2",
    ]
    for heading in headings:
        cell = heading.get("result") or {}
        statue = (heading.get("statue") or {}).get("reference") or {}
        statue_fh = f"{statue['n_full_horizon']}/{statue['n_episodes']}" if "n_full_horizon" in statue else "-"
        lines.append(
            f"  {float(heading['yaw_deg']):>+8g} {cell.get('n_clean', 0):>3d}/{cell.get('n_episodes', 0):<3d} "
            f"{_fmt(cell.get('clean_lcb'), '7.4f')} {_fmt(cell.get('full_horizon_fraction'), '8.4f')} "
            f"{_fmt((heading.get('metrics') or {}).get('reward_mean'), '9.1f')} {statue_fh:>10} "
            f"{_fmt(statue.get('mean_reward'), '14.1f')}  {'PASS' if cell.get('passed') else 'FAIL'}"
        )
    failing = result.get("seed_failures") or []
    if failing:
        lines += ["", f"seeds not clean at every heading ({len(failing)}; first {min(len(failing), 10)} shown):"]
        for entry in failing[:10]:
            lines.append(
                f"  seed {entry['seed']}: "
                + ", ".join(f"{float(yaw):+g}" for yaw in entry.get("headings_deg") or ())
                + " deg"
            )
    try:
        zero = heading_zero_panel(report)
    except ValueError:
        zero = None
    if zero is not None:
        # Rendered as the v2 decision it is: no kind note (the stage's kind is this certificate), no verdict line.
        detail = _render_stance_v2_report({**zero, "gate_kind": STANCE_GATE_V2_KIND}).split("\nGATE: ", 1)[0]
        lines += [
            "",
            "the 0 deg cell in detail (that heading's v2 decision; the certificate's verdict closes this report):",
            "",
            detail,
        ]
    lines += ["", f"GATE: {'PASS' if report['passed'] else 'FAIL'}"]
    lines += [f"  - {failure}" for failure in report["failures"]]
    return "\n".join(lines)


def write_stance_heading_report(stage_dir: "str | Path", report: dict[str, Any]) -> dict[str, Path]:
    """Write the certificate beside the stage's other artifacts: the text, the JSON, then the grid CSV.

    ``stance_heading_report.{txt,json}`` and ``stance_heading_panel_selected.csv``,
    each atomically, in that order: the CSV is publication's evidence, so a
    write that stops part-way leaves a certificate publication refuses for
    want of it -- never a CSV without the report it came from.  Never
    ``stance_gate_report.*`` or ``stance_panel_selected.csv``.
    """
    if report.get("schema") != STANCE_HEADING_REPORT_SCHEMA:
        raise ValueError(f"not a {STANCE_HEADING_REPORT_SCHEMA} certificate ({report.get('schema')!r})")
    directory = Path(stage_dir)
    directory.mkdir(parents=True, exist_ok=True)
    text_path = directory / STANCE_HEADING_REPORT_TXT
    json_path = directory / STANCE_HEADING_REPORT_JSON
    atomic_write_text(text_path, render_stance_heading_report(report) + "\n")
    atomic_write_json(json_path, _json_safe(report), sort_keys=True, allow_nan=False)
    csv_path = _write_stance_heading_panel_evidence(directory, report)
    return {
        "stance_heading_report_txt": text_path,
        "stance_heading_report_json": json_path,
        "stance_heading_panel_csv": csv_path,
    }


def write_action_filter_sweep(
    stage_dir: "str | Path", reports: list[dict[str, Any]], *, probe_episodes: int
) -> dict[str, Path]:
    """Write the action-filter probe sweep as one curve, not N verdicts.

    Each entry scored a MODIFIED policy, so none of them is a gate result and
    the artifact says so before it says anything else. What is worth reading is
    the trend: how long the filtered policy survives against cutoff. On T-Rex
    stage 1 today that is 96 steps at 5 Hz rising to 351 at 35 Hz, against a
    1000-step horizon -- the policy needs essentially its full command
    bandwidth to stand (issue #491), and a future policy that reaches the
    horizon at a low cutoff is one that could survive a real actuator.

    Deliberately does not write ``stance_panel_selected.csv``: that file is the
    per-episode evidence a stance-gated bundle is certified from, and none of
    these panels scored the policy that would be certified.
    """
    directory = Path(stage_dir)
    directory.mkdir(parents=True, exist_ok=True)
    rows = sorted(
        (
            {
                "filter_actions_hz": report["filter_actions_hz"],
                "episode_length_mean": report["metrics"]["episode_length_mean"],
                "full_horizon_fraction": report["metrics"]["full_horizon_fraction"],
                "reward_mean": report["metrics"]["reward_mean"],
                "mean_unsupported_duty": report["metrics"]["mean_unsupported_duty"],
                "terminations": report["terminations"],
            }
            for report in reports
        ),
        key=lambda row: row["filter_actions_hz"],
    )
    horizon = reports[0]["horizon"]
    # Each entry's `policy` carries its own cutoff suffix, so the first one
    # would label the whole sweep "low-passed at 5 Hz" -- wrong for every other
    # row. Strip it; the per-cutoff values are the table's first column.
    policy = str(reports[0]["policy"]).split(" — actions low-passed at ")[0]
    payload = {
        "schema": "mesozoic.action-filter-sweep/v1",
        "species": reports[0]["species"],
        "stage": reports[0]["stage"],
        "policy": policy,
        "horizon": horizon,
        "episodes_per_cutoff": probe_episodes,
        "note": (
            "Each row scored a MODIFIED policy (actions low-passed before reaching the "
            "plant). No row is a gate verdict. The metric to read is episode_length_mean "
            "against filter_actions_hz."
        ),
        "sweep": rows,
        "reports": reports,
    }
    lines = [
        "PROBE: actions low-passed before reaching the plant, at several cutoffs.",
        "Every row scores a MODIFIED policy. None of them is a gate verdict.",
        "",
        f"policy              {policy}",
        f"stage               {species_display_name(reports[0]['species'])} — "
        f"{stage_display_name(reports[0]['stage'])} (stage {reports[0]['stage']})",
        f"panel               {probe_episodes} episodes per cutoff, horizon {horizon}",
        "",
        f"  {'cutoff':>9}{'ep length':>12}{'full-horiz':>12}{'reward':>10}   terminations",
    ]
    for row in rows:
        lines.append(
            f"  {row['filter_actions_hz']:>7.4g} Hz{row['episode_length_mean']:>12.1f}"
            f"{row['full_horizon_fraction']:>12.4f}{row['reward_mean']:>10.1f}   {row['terminations']}"
        )
    lines += [
        "",
        "Read the trend, not the rows: episode_length_mean against cutoff measures how",
        "much of its own high-frequency command content the policy needs in order to",
        "stand. Balance correction on this plant is ~1.1-1.4 Hz, which every cutoff here",
        "passes essentially untouched, so a short episode means the policy depends on",
        "content well above the frequency band the task itself requires.",
    ]
    text_path = directory / "stance_gate_probe_filtered.txt"
    json_path = directory / "stance_gate_probe_filtered.json"
    atomic_write_text(text_path, "\n".join(lines) + "\n")
    atomic_write_json(json_path, _json_safe(payload), sort_keys=True, allow_nan=False)
    return {"action_filter_sweep_txt": text_path, "action_filter_sweep_json": json_path}


def render_constant_hold_probe(reports: list[dict[str, Any]], *, probe_episodes: int) -> tuple[str, dict[str, Any]]:
    """Reduce the constant-hold variants to their comparison table and payload.

    Split from the writer so the CLI can print the table without depositing
    files in whatever directory it was run from.

    The rows are only meaningful against each other, which is why they share a
    file. Two of them are controls that bracket the measurement:

    * the **unmodified policy**, which must reach the horizon or the probe is
      measuring a broken harness rather than a property of the pose, and
    * the **zero hold**, the statue -- a constant that is already known to
      stand all 1000 steps in 40 of 40 episodes.

    Between them sit the held variants. Read them as a single yes/no:

    * held rollouts reach the horizon -> the pose is holdable by a constant.
      The tremor is not stabilisation; the action penalties simply failed to
      find the smooth solution, and this is an optimisation problem.
    * held rollouts fall while both controls stand -> the pose the policy chose
      genuinely requires feedback. The question then moves to *why* it chose a
      pose it has to fight, which points at the actuators no stage-1 term
      constrains rather than at the action penalties.

    The ramped variant exists to close the obvious objection to the hard
    switch -- that a step transient, not the loss of feedback, is what knocked
    the animal over -- with a measurement instead of an argument. If the hard
    switch falls and the ramped one stands, the transient was the cause and
    neither row says anything about the pose.

    Deliberately does not write ``stance_panel_selected.csv``: no row here
    scored the policy that would be certified.
    """
    rows = [
        {
            "label": report["hold_constant"]["label"],
            "holds": report["hold_constant"]["holds"],
            "source": report["hold_constant"]["source"],
            "handoff_steps": report["hold_constant"]["handoff_steps"],
            "ramp_steps": report["hold_constant"]["ramp_steps"],
            "episode_length_mean": report["metrics"]["episode_length_mean"],
            "full_horizon_fraction": report["metrics"]["full_horizon_fraction"],
            "reward_mean": report["metrics"]["reward_mean"],
            "mean_unsupported_duty": report["metrics"]["mean_unsupported_duty"],
            "terminations": report["terminations"],
        }
        for report in reports
    ]
    horizon = reports[0]["horizon"]
    policy = str(reports[0]["policy"]).split(" — commanded action held constant")[0].split(" — ")[0]
    # The rows that hold the POLICY'S measured pose. Deliberately not "every
    # row that holds something": `hold_zero` holds a constant too, and it is a
    # control whose standing is already known and proves nothing about this
    # policy. Folding it in would make an all-falling result read as mixed.
    held = [row for row in rows if row["source"] == "measured"]
    hold_actions: "list[float] | tuple[float, ...]" = next(
        (report["hold_constant"]["actions"] for report in reports if report["hold_constant"]["source"] == "measured"),
        [],
    )
    payload = {
        "schema": "mesozoic.constant-hold-probe/v1",
        "species": reports[0]["species"],
        "stage": reports[0]["stage"],
        "policy": policy,
        "horizon": horizon,
        "episodes_per_variant": probe_episodes,
        "hold_actions": hold_actions,
        "note": (
            "Each row scored a MODIFIED policy (its commanded action replaced by a "
            "constant partway through the episode). No row is a gate verdict. The "
            "question is whether the held rows reach the horizon: if they do, the "
            "pose is holdable without feedback and the tremor is waste; if they "
            "fall while both controls stand, the pose requires active stabilisation."
        ),
        "variants": rows,
        "reports": reports,
    }
    lines = [
        "PROBE: the policy's commanded action replaced by a constant partway through the episode.",
        "Every held row scores a MODIFIED policy. None of them is a gate verdict.",
        "",
        f"policy              {policy}",
        f"stage               {species_display_name(reports[0]['species'])} — "
        f"{stage_display_name(reports[0]['stage'])} (stage {reports[0]['stage']})",
        f"panel               {probe_episodes} episodes per variant, horizon {horizon}",
        f"hold                the policy's own post-settle mean action, {len(hold_actions)} actuators",
        "",
        f"  {'variant':<26}{'handoff':>8}{'ramp':>6}{'ep length':>11}{'full-horiz':>12}{'reward':>10}   terminations",
    ]
    for row in rows:
        handoff = "never" if not row["holds"] else str(row["handoff_steps"])
        lines.append(
            f"  {row['label']:<26}{handoff:>8}{row['ramp_steps']:>6}"
            f"{row['episode_length_mean']:>11.1f}{row['full_horizon_fraction']:>12.4f}"
            f"{row['reward_mean']:>10.1f}   {row['terminations']}"
        )
    lines += ["", "How to read it:"]
    if held and all(row["full_horizon_fraction"] >= 0.95 for row in held):
        lines += [
            "  Every held variant reached the horizon. The pose the policy holds does NOT",
            "  need feedback to stand -- a constant command suffices. The tremor is therefore",
            "  not closed-loop stabilisation, and the gap to the statue is an optimisation",
            "  failure rather than a missing reward term.",
        ]
    elif held and all(row["full_horizon_fraction"] <= 0.0 for row in held):
        lines += [
            "  No held variant reached the horizon. The pose the policy holds REQUIRES",
            "  continuous feedback. The tremor is stabilisation, not waste, and penalising",
            "  it harder would remove the only thing keeping the animal up. The question",
            "  becomes why the policy settled on a pose it has to fight -- look at the",
            "  actuators no stage-1 reward term constrains.",
        ]
    else:
        lines += [
            "  Mixed. Compare the ramped variant against the hard switch: if only the hard",
            "  switch falls, the step transient caused it and neither row is evidence about",
            "  the pose. Re-run with more episodes before drawing a conclusion.",
        ]
    lines += [
        "",
        "  The two controls bracket the measurement. 'policy' is the unmodified checkpoint",
        "  and must reach the horizon; 'hold_zero' is the statue, a constant already known",
        "  to stand. If either misbehaves, the harness is at fault, not the pose.",
    ]
    return "\n".join(lines) + "\n", payload


#: One-sided level of every comparison in the release ablation, rejecting at
#: ``p <= alpha``. A Fraction, and the p-values exact rationals, so the
#: boundary is decided exactly: three episodes per arm reach p = 1/20 and no
#: lower. Not adjusted across groups -- the ablation is a probe, not a gate --
#: and both the payload and the text say so.
_ABLATION_ALPHA = Fraction(1, 20)
#: The statue control must stand in at least this share of its episodes. A
#: sanity check that the known standing reference stood, not an inference
#: about any group, so it stays a fixed share rather than a test.
_ABLATION_STATUE_STANDS = 0.95
#: How an inconclusive verdict says where a variant sits, by its position.
_ABLATION_POSITION_TEXT = {
    "between": "between the endpoints, separated from both",
    "unresolved": "separated from neither endpoint",
    "insufficient episodes": f"insufficient episodes to reach p <= {float(_ABLATION_ALPHA):g}",
    "unmeasured": "unmeasured or invalid",
}


def _fisher_exact_greater(successes_a: int, n_a: int, successes_b: int, n_b: int) -> Fraction:
    """One-sided Fisher exact p that arm *a* succeeds more often than arm *b*.

    The upper tail of arm a's successes under the hypergeometric distribution
    with both margins fixed, as an exact rational. It is what
    ``scipy.stats.fisher_exact([[successes_a, n_a - successes_a],
    [successes_b, n_b - successes_b]], alternative="greater")`` returns,
    without a runtime dependency on scipy, which only the test extra installs.
    """
    if n_a < 1 or n_b < 1 or not 0 <= successes_a <= n_a or not 0 <= successes_b <= n_b:
        raise ValueError(f"not a 2x2 table: {successes_a}/{n_a} against {successes_b}/{n_b}")
    total, successes = n_a + n_b, successes_a + successes_b
    tail = sum(
        math.comb(successes, x) * math.comb(total - successes, n_a - x)
        for x in range(successes_a, min(n_a, successes) + 1)
    )
    return Fraction(tail, math.comb(total, n_a))


def _fisher_exact_min_p(n_a: int, n_b: int) -> Fraction:
    """The smallest one-sided Fisher p any table with these arm sizes reaches: ``1 / C(n_a + n_b, n_a)``.

    Above alpha no outcome could be significant, so such a comparison reports
    too few episodes rather than a negative finding.
    """
    return Fraction(1, math.comb(n_a + n_b, n_a))


def _ablation_share(row: dict[str, Any] | None) -> float | None:
    """A variant's full-horizon share, or None when it is missing or not a valid share."""
    if row is None:
        return None
    value = row.get("full_horizon_fraction")
    if value is None or isinstance(value, bool):
        return None
    try:
        share = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(share) or not 0.0 <= share <= 1.0:
        return None
    return share


def _full_horizon_count(report: dict[str, Any]) -> tuple[int, int] | None:
    """(episodes that reached the horizon, episodes) for one probe report, or None.

    Fisher needs counts and the report records a share, but the share is
    ``k / n`` by construction (``StancePanel.full_horizon_fraction``), so k is
    recovered from it and the panel size. A share that is not a multiple of
    ``1 / n`` is refused rather than rounded, and so is per-episode evidence,
    when the report carries it, that does not agree.
    """
    episodes = report.get("episodes")
    if isinstance(episodes, bool) or not isinstance(episodes, int) or episodes < 1:
        return None
    share = _ablation_share(report.get("metrics"))
    if share is None:
        return None
    count = round(share * episodes)
    if abs(share * episodes - count) > 1e-9:
        return None
    evidence = report.get("episode_evidence")
    if evidence is not None:
        if not isinstance(evidence, list) or len(evidence) != episodes:
            return None
        reached = [entry.get("reached_horizon") if isinstance(entry, dict) else None for entry in evidence]
        if any(value is None for value in reached) or sum(bool(value) for value in reached) != count:
            return None
    return count, episodes


def _ablation_counts(row: dict[str, Any] | None) -> tuple[int, int] | None:
    """A variant row's (full-horizon episodes, episodes), or None when missing or refused."""
    if row is None or row.get("full_horizon_episodes") is None:
        return None
    return int(row["full_horizon_episodes"]), int(row["episodes"])


def _ablation_comparison(
    variant: tuple[int, int] | None, control: tuple[int, int], *, against: str, stands_more: bool
) -> dict[str, Any]:
    """One one-sided Fisher exact test of a variant's full-horizon count against an endpoint.

    ``stands_more`` asks whether the variant reaches the horizon more often
    than *control* (the test against ``hold_all``); otherwise, whether it
    does so less often (the test against ``hold_zero``).
    """
    entry: dict[str, Any] = {"against": against, "alternative": "greater" if stands_more else "less"}
    if variant is None:
        return {**entry, "p": None, "status": "unmeasured"}
    p = _fisher_exact_greater(*variant, *control) if stands_more else _fisher_exact_greater(*control, *variant)
    if _fisher_exact_min_p(variant[1], control[1]) > _ABLATION_ALPHA:
        status = "insufficient episodes"
    else:
        status = "significant" if p <= _ABLATION_ALPHA else "not significant"
    return {**entry, "p": float(p), "status": status}


def _ablation_position(vs_hold_all: dict[str, Any], vs_hold_zero: dict[str, Any]) -> str:
    """Where a variant sits on the path between the endpoints, from its two comparisons.

    ``stands`` needs it significantly above ``hold_all`` and not
    significantly below ``hold_zero``, ``falls`` the mirror image; separated
    from both is ``between``, which is neither a rescue nor a fall. A
    comparison that could not be made -- an unmeasured variant, or too few
    episodes -- leaves the position unknown even when the other is
    significant, because it is the one that could have shown ``between``.
    """
    statuses = (vs_hold_all["status"], vs_hold_zero["status"])
    if "unmeasured" in statuses:
        return "unmeasured"
    if "insufficient episodes" in statuses:
        return "insufficient episodes"
    above, below = (status == "significant" for status in statuses)
    if above and below:
        return "between"
    if above:
        return "stands"
    if below:
        return "falls"
    return "unresolved"


def _ablation_endpoints(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[str]]:
    """The endpoint controls' counts and separation, and every reason classification is refused.

    ``hold_zero`` must stand in :data:`_ABLATION_STATUE_STANDS` of its
    episodes. ``hold_all`` must reach the horizon significantly less often
    than it (one-sided Fisher exact) -- a partial fall counts when it is
    significant -- or there is no full-pose fall for any group to explain.
    The test also guarantees that the per-group comparisons, on panels of
    the same size, can reach alpha at all.
    """
    refusals: list[str] = []
    counts: dict[str, tuple[int, int] | None] = {}
    for control, labels in (("hold_all", ("hold_all",)), ("hold_zero", ("hold_zero", "hold_zero (statue control)"))):
        matches = [row for row in rows if row["label"] in labels]
        counts[control] = _ablation_counts(matches[0]) if len(matches) == 1 else None
        if len(matches) != 1:
            refusals.append(f"{control}: missing or duplicated endpoint control; expected one, found {len(matches)}")
        elif counts[control] is None:
            required = (
                "a full pose that falls significantly more often than the statue"
                if control == "hold_all"
                else "a standing endpoint"
            )
            refusals.append(f"{control}: missing or invalid measurement; required {required}")
    full, statue = counts["hold_all"], counts["hold_zero"]
    statue_stands = statue is not None and statue[0] / statue[1] >= _ABLATION_STATUE_STANDS
    if statue is not None and not statue_stands:
        refusals.append(
            f"hold_zero: the statue control stands in only {statue[0]}/{statue[1]} episodes, below "
            f"{_ABLATION_STATUE_STANDS:g} of them; a standing reference was not established"
        )
    separation = None
    if full is not None and statue is not None:
        separation = _ablation_comparison(full, statue, against="hold_zero", stands_more=False)
        alpha = float(_ABLATION_ALPHA)
        if statue_stands and separation["status"] == "insufficient episodes":
            refusals.append(
                f"hold_all: insufficient episodes ({full[1]} against the statue's {statue[1]}): "
                f"no outcome reaches one-sided Fisher p <= {alpha:g}"
            )
        elif statue_stands and separation["status"] == "not significant":
            refusals.append(
                f"hold_all: the full held pose stands in {full[0]}/{full[1]} episodes, not significantly "
                f"fewer than the statue's {statue[0]}/{statue[1]} (one-sided Fisher p = {separation['p']:.4f} "
                f"> {alpha:g}); there is no full-pose fall to explain"
            )
    endpoints = {
        "hold_all": None if full is None else list(full),
        "hold_zero": None if statue is None else list(statue),
        "separation": separation,
    }
    return endpoints, refusals


def _ablation_variant_count_text(side: dict[str, Any]) -> str:
    """``release_tail 3/8``, or the bare label when the variant has no valid count."""
    if side["episodes"] is None:
        return str(side["variant"])
    return f"{side['variant']} {side['full_horizon_episodes']}/{side['episodes']}"


def _ablation_verdicts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Classify each ablated group by necessity and sufficiency.

    Necessity and sufficiency are asked separately because either alone
    misleads. A group that is merely *correlated* with the fall -- the
    saturated set is the obvious candidate, since saturation is visible and
    dramatic -- shows up as necessary-and-not-sufficient or as neither, and
    only the pair distinguishes that from a cause. Neither question is
    assessed unless the endpoint controls establish the fall being explained
    (:func:`_ablation_endpoints`). Each is then answered by one-sided Fisher
    exact tests of the variant against both endpoints: ``release_G`` is
    necessary when it stands significantly more often than ``hold_all`` and
    not significantly less often than ``hold_zero``, and not necessary in the
    mirror case; ``only_G`` is sufficient when it falls significantly more
    often than ``hold_zero`` and not significantly less often than
    ``hold_all``. Every other outcome -- between the endpoints, separated from
    neither, too few episodes, unmeasured -- leaves the claim None rather
    than a false claim of non-necessity.
    """
    by_label = {row["label"]: row for row in rows}
    endpoints, refusals = _ablation_endpoints(rows)
    applicable = not refusals
    verdicts = []
    groups = dict.fromkeys(label.split("_", 1)[1] for label in by_label if label.startswith(("release_", "only_")))
    for group in groups:
        release, only = by_label.get(f"release_{group}"), by_label.get(f"only_{group}")
        sides: dict[str, Any] = {"necessity": None, "sufficiency": None}
        necessary = sufficient = None
        if applicable:
            full, statue = tuple(endpoints["hold_all"]), tuple(endpoints["hold_zero"])
            for side, label in (("necessity", f"release_{group}"), ("sufficiency", f"only_{group}")):
                counts = _ablation_counts(by_label.get(label))
                vs_hold_all = _ablation_comparison(counts, full, against="hold_all", stands_more=True)
                vs_hold_zero = _ablation_comparison(counts, statue, against="hold_zero", stands_more=False)
                sides[side] = {
                    "variant": label,
                    "full_horizon_episodes": None if counts is None else counts[0],
                    "episodes": None if counts is None else counts[1],
                    "vs_hold_all": vs_hold_all,
                    "vs_hold_zero": vs_hold_zero,
                    "position": _ablation_position(vs_hold_all, vs_hold_zero),
                }
            necessary = {"stands": True, "falls": False}.get(sides["necessity"]["position"])
            sufficient = {"falls": True, "stands": False}.get(sides["sufficiency"]["position"])
        if not applicable:
            verdict = "not applicable (endpoint controls do not establish a fall to explain)"
        elif release is None or only is None:
            verdict = "incomplete (a variant did not run)"
        elif necessary is None or sufficient is None:
            unknown = [sides["necessity"]] if necessary is None else []
            unknown += [sides["sufficiency"]] if sufficient is None else []
            reasons = (
                f"{_ablation_variant_count_text(side)}: {_ABLATION_POSITION_TEXT[side['position']]}" for side in unknown
            )
            verdict = f"inconclusive ({'; '.join(reasons)})"
        elif necessary and sufficient:
            verdict = "CAUSE — necessary and sufficient"
        elif necessary:
            verdict = "contributes — necessary, not sufficient alone"
        elif sufficient:
            verdict = "sufficient alone, but not necessary"
        else:
            verdict = "bystander — neither"
        verdicts.append(
            {
                "group": group,
                "necessary": necessary,
                "sufficient": sufficient,
                "verdict": verdict,
                "release_full_horizon": None if release is None else release["full_horizon_fraction"],
                "only_full_horizon": None if only is None else only["full_horizon_fraction"],
                **sides,
            }
        )
    return verdicts


#: Lateral impulses, in m/s of root velocity, swept by the recovery probe.
#: Lateral because that is the axis a biped is least able to catch itself on --
#: a fore/aft shove can be absorbed by pitching, a sideways one cannot without
#: moving a foot. Both signs are run, because a learned stance is generally
#: asymmetric -- the r6 checkpoints splayed their toes differently per foot,
#: and the r7 passive-toes checkpoint tremors far harder on one leg -- so a
#: one-sided sweep would measure the easy direction or the hard one and
#: report it as the answer.
_IMPULSE_SPEEDS = (0.5, 1.0, 2.0)


def impulse_variants(speeds: "tuple[float, ...] | list[float]", *, step: int) -> list[RootImpulse]:
    """One :class:`RootImpulse` per (speed, direction), plus the zero control."""
    variants = [RootImpulse(step=step, delta_v=(0.0, 0.0, 0.0), axis_label="none (control)")]
    for speed in sorted(set(float(value) for value in speeds if float(value) > 0)):
        variants.append(RootImpulse(step=step, delta_v=(0.0, speed, 0.0), axis_label="lateral +y"))
        variants.append(RootImpulse(step=step, delta_v=(0.0, -speed, 0.0), axis_label="lateral -y"))
    return variants


#: A row counts as a recovery when this share of episodes reach the horizon.
#: Matches the gate's own `min_full_horizon_fraction`, so "recovered" means the
#: same thing here as "stood" does there.
_RECOVERY_SHARE = 0.95


def _recovery_envelopes(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Largest impulse each side fully recovers from, per direction.

    This, not a mean margin, is the number the probe exists to produce. Mean
    margin over directions pools a full recovery with a total failure: on the
    T-Rex checkpoint the policy reaches the horizon at 0.5 m/s in one direction
    and falls at 0.5 m/s in the other, which averages to a healthy-looking
    +135 steps and describes neither.
    """
    envelopes: dict[str, dict[str, float]] = {}
    for row in rows:
        if row["speed"] <= 0:
            continue
        entry = envelopes.setdefault(row["axis_label"], {"policy": 0.0, "statue": 0.0})
        for who in ("policy", "statue"):
            cell = row[who]
            if cell is not None and cell["full_horizon_fraction"] >= _RECOVERY_SHARE:
                entry[who] = max(entry[who], row["speed"])
    return envelopes


def _recovery_reading(
    rows: list[dict[str, Any]],
    envelopes: dict[str, dict[str, float]],
    disturbed: list[dict[str, Any]],
    horizon: int,
) -> list[str]:
    """The prose verdict, keyed off the envelopes rather than a pooled margin."""
    lines: list[str] = []
    policy_best = max((entry["policy"] for entry in envelopes.values()), default=0.0)
    statue_best = max((entry["statue"] for entry in envelopes.values()), default=0.0)
    mean_margin = sum(row["margin_steps"] for row in disturbed) / len(disturbed)

    for direction, entry in sorted(envelopes.items()):
        policy_text = "none" if entry["policy"] == 0 else f"{entry['policy']:.2f} m/s"
        statue_text = "none" if entry["statue"] == 0 else f"{entry['statue']:.2f} m/s"
        lines.append(f"  recovery envelope, {direction:<16} policy {policy_text:>9}   statue {statue_text:>9}")
    lines.append("")

    if policy_best > statue_best:
        lines += [
            f"  ACTIVE CORRECTION EXISTS: the policy fully recovers from {policy_best:.2f} m/s where a",
            "  constant command never recovers from anything. Stage 1 never measured this,",
            "  because the task has no disturbance to measure it with -- a metric gap, not a",
            "  reward one.",
        ]
        sides = sorted(entry["policy"] for entry in envelopes.values())
        if len(sides) > 1 and sides[0] < sides[-1]:
            worst = "none" if sides[0] == 0 else f"{sides[0]:.2f} m/s"
            lines += [
                "",
                f"  But it is ASYMMETRIC: {policy_best:.2f} m/s one way, {worst} the other, at the same",
                "  magnitude. The envelope is the WEAKER side -- a disturbance does not ask which",
                "  direction the policy prefers. Treat the smaller number as the capability.",
            ]
        beyond = [row["speed"] for row in disturbed if row["speed"] > policy_best]
        if beyond:
            lines += [
                "",
                f"  Above {policy_best:.2f} m/s it stops mattering: at {min(beyond):.2f} m/s and up the policy and",
                "  the statue are within noise of each other, so the recovery envelope is narrow",
                "  rather than merely asymmetric.",
            ]
    elif mean_margin < -0.1 * horizon:
        lines += [
            f"  The policy does WORSE than the statue by {mean_margin:.0f} steps on average.",
            "  Standing still beats what this policy does under load, which is a finding about",
            "  the pose and the tremor rather than about the task.",
        ]
    else:
        lines += [
            "  Neither side recovers from anything, and the policy's survival is within noise",
            f"  of the statue's ({mean_margin:+.0f} steps on average). Nothing has learned to recover --",
            "  exactly what the task structure predicts, since no reward term can pay for a",
            "  correction that is never needed. No reweighting fixes this; the disturbance has",
            "  to enter the TASK. See STAGE1_SPLIT_PLAN's 1a stance / 1b recovery split.",
        ]
    lines += [
        "",
        f"  (Mean step margin over disturbed rows is {mean_margin:+.0f}, reported second because it",
        "  pools a full recovery with a total failure and can describe neither.)",
    ]
    return lines


def render_impulse_probe(
    policy_reports: list[dict[str, Any]],
    statue_reports: list[dict[str, Any]],
    *,
    probe_episodes: int,
) -> tuple[str, dict[str, Any]]:
    """Reduce the recovery sweep to a policy-against-statue table.

    The statue is not decoration here, it is the entire measurement. It
    commands a constant and therefore cannot respond to anything, so whatever
    horizon it survives is the plant's **passive** robustness -- how much of a
    shove this body shrugs off with no control at all. The policy's margin
    above that line is the only part attributable to active correction.

    Three readings, and they lead to different work:

    * policy well above statue -> active correction exists and stage 1 simply
      never measured it. The gap is a *metric* gap, not a reward one.
    * policy indistinguishable from statue -> nothing has learned to recover,
      which is what the task structure predicts, and no reweighting fixes it.
      Adding a disturbance to the task is the change (1a/1b).
    * policy *below* statue -> its pose or its tremor is actively worse than
      standing still under load, which would be a finding about the pose.
    """

    def _rows(reports: list[dict[str, Any]]) -> dict[tuple[float, str], dict[str, Any]]:
        return {
            (report["impulse"]["speed"], report["impulse"]["axis_label"]): {
                "speed": report["impulse"]["speed"],
                "axis_label": report["impulse"]["axis_label"],
                "episode_length_mean": report["metrics"]["episode_length_mean"],
                "full_horizon_fraction": report["metrics"]["full_horizon_fraction"],
                "reward_mean": report["metrics"]["reward_mean"],
                "terminations": report["terminations"],
            }
            for report in reports
        }

    policy_rows, statue_rows = _rows(policy_reports), _rows(statue_reports)
    horizon = policy_reports[0]["horizon"]
    step = policy_reports[0]["impulse"]["step"]
    policy = str(policy_reports[0]["policy"]).split(" — ")[0]
    keys = sorted(set(policy_rows) | set(statue_rows))
    rows: list[dict[str, Any]] = []
    for key in keys:
        here, there = policy_rows.get(key), statue_rows.get(key)
        rows.append(
            {
                "speed": key[0],
                "axis_label": key[1],
                "policy": here,
                "statue": there,
                "margin_full_horizon": (
                    None
                    if here is None or there is None
                    else here["full_horizon_fraction"] - there["full_horizon_fraction"]
                ),
                "margin_steps": (
                    None
                    if here is None or there is None
                    else here["episode_length_mean"] - there["episode_length_mean"]
                ),
            }
        )
    disturbed = [row for row in rows if row["speed"] > 0 and row["margin_steps"] is not None]
    payload = {
        "schema": "mesozoic.impulse-recovery-probe/v1",
        "species": policy_reports[0]["species"],
        "stage": policy_reports[0]["stage"],
        "policy": policy,
        "horizon": horizon,
        "impulse_step": step,
        "episodes_per_variant": probe_episodes,
        "note": (
            "Each row applies a root-velocity impulse mid-episode and scores the UNMODIFIED "
            "policy on a MODIFIED task. Stage 1 declares no disturbance, so no row is a gate "
            "verdict. The statue columns are the control: it commands a constant and cannot "
            "respond, so its survival is the plant's passive robustness and only the policy's "
            "margin above it is active correction."
        ),
        "rows": rows,
        "policy_reports": policy_reports,
        "statue_reports": statue_reports,
    }
    lines = [
        "PROBE: a lateral impulse applied to the root partway through each episode.",
        "Scores the unmodified policy on a MODIFIED TASK. No row is a gate verdict.",
        "",
        f"policy              {policy}",
        f"stage               {species_display_name(policy_reports[0]['species'])} — "
        f"{stage_display_name(policy_reports[0]['stage'])} (stage {policy_reports[0]['stage']})",
        f"panel               {probe_episodes} episodes per row, horizon {horizon}, impulse at step {step}",
        "control             the zero-action statue, which commands a constant and CANNOT respond",
        "",
        f"  {'impulse':>9} {'direction':<16}{'policy len':>11}{'policy fh':>11}"
        f"{'statue len':>11}{'statue fh':>11}{'margin':>9}",
    ]

    def _length(cell: "dict[str, Any] | None") -> str:
        return "-" if cell is None else format(cell["episode_length_mean"], ".1f")

    def _share(cell: "dict[str, Any] | None") -> str:
        return "-" if cell is None else format(cell["full_horizon_fraction"], ".4f")

    for row in rows:
        here, there = row["policy"], row["statue"]
        speed = "none" if row["speed"] == 0 else f"{row['speed']:.2f} m/s"
        margin = "-" if row["margin_steps"] is None else format(row["margin_steps"], "+.0f")
        lines.append(
            f"  {speed:>9} {row['axis_label']:<16}"
            f"{_length(here):>11}{_share(here):>11}"
            f"{_length(there):>11}{_share(there):>11}{margin:>9}"
        )
    envelopes = _recovery_envelopes(rows)
    payload["recovery_envelopes"] = envelopes
    lines += ["", "How to read it:"]
    if not disturbed:
        lines.append("  No disturbed row completed; nothing to conclude.")
    else:
        lines += _recovery_reading(rows, envelopes, disturbed, horizon)
    lines += [
        "",
        "  Both signs are run because a learned stance is generally asymmetric --",
        "  commanded pose and tremor differ between the left and right legs -- so a",
        "  one-sided sweep would measure whichever direction it happens to be good at",
        "  and report that as the answer.",
        "  The zero-impulse row is the harness check: both columns must reach the horizon.",
    ]
    return "\n".join(lines) + "\n", payload


def write_impulse_probe(
    stage_dir: "str | Path",
    policy_reports: list[dict[str, Any]],
    statue_reports: list[dict[str, Any]],
    *,
    probe_episodes: int,
) -> dict[str, Path]:
    """Write the recovery sweep beside the stage's other artifacts."""
    directory = Path(stage_dir)
    directory.mkdir(parents=True, exist_ok=True)
    text, payload = render_impulse_probe(policy_reports, statue_reports, probe_episodes=probe_episodes)
    text_path = directory / "stance_gate_probe_impulse.txt"
    json_path = directory / "stance_gate_probe_impulse.json"
    atomic_write_text(text_path, text)
    atomic_write_json(json_path, _json_safe(payload), sort_keys=True, allow_nan=False)
    return {"impulse_probe_txt": text_path, "impulse_probe_json": json_path}


#: The heading probe's JSON file and the schema it declares (``write_heading_probe``).
HEADING_PROBE_JSON = "stance_heading_probe.json"
HEADING_PROBE_SCHEMA = "mesozoic.stance-heading-probe/v1"

#: The full-horizon fraction below which a whole-scene heading-probe row warns: v2's
#: per-panel rail (``min_full_horizon_fraction``, 0.95 on every v2 stance).  Warn only,
#: by the maintainer's decision of 2026-10-11 (docs/HEADING_INVARIANCE_DESIGN_2026_10.md):
#: the probe stays report only (D-D27) until the heading certificate replaces it.
HEADING_PROBE_WARN_BELOW = 0.95


def spawn_yaw_variants(offsets_deg: "tuple[float, ...] | list[float]") -> list[SpawnYaw]:
    """The heading probe's rows: the unturned control, then each non-zero offset turned alone and with the scene."""
    variants = [SpawnYaw(0.0)]
    for offset in sorted({float(value) for value in offsets_deg if float(value) != 0.0}):
        variants += [SpawnYaw(offset, whole_scene=False), SpawnYaw(offset, whole_scene=True)]
    return variants


def render_heading_probe(
    policy_reports: list[dict[str, Any]],
    statue_reports: list[dict[str, Any]],
    *,
    probe_episodes: int,
) -> tuple[str, dict[str, Any]]:
    """The heading probe as text and JSON: per turned spawn, the policy and the statue side by side.

    Each row reports the full-horizon fraction, the mean reward and the
    mean and largest |episode yaw change|.  The statue commands a constant,
    so its rows are the plant's own heading neutrality: it should stand at
    every offset, and under ``whole_scene`` score its unturned reward.
    """

    def _row(report: dict[str, Any]) -> dict[str, Any]:
        yaw = [float(value) for value in report.get("episode_yaw_change_deg", [])]
        return {
            "full_horizon_fraction": report["metrics"]["full_horizon_fraction"],
            "reward_mean": report["metrics"]["reward_mean"],
            "episode_length_mean": report["metrics"]["episode_length_mean"],
            "yaw_change_mean_deg": float(np.mean(yaw)) if yaw else float("nan"),
            "yaw_change_max_deg": float(np.max(yaw)) if yaw else float("nan"),
            "terminations": report["terminations"],
        }

    rows = []
    for policy, statue in zip(policy_reports, statue_reports):
        turn = policy["spawn_yaw"]
        rows.append(
            {
                "yaw_deg": turn["yaw_deg"],
                "whole_scene": turn["whole_scene"],
                "policy": _row(policy),
                "statue": _row(statue),
            }
        )
    horizon = policy_reports[0]["horizon"] if policy_reports else None
    first_seed = int(policy_reports[0]["seed"]) if policy_reports else 0
    payload: dict[str, Any] = {
        "schema": HEADING_PROBE_SCHEMA,
        "policy": policy_reports[0]["policy"] if policy_reports else None,
        "probe_episodes": probe_episodes,
        "seed": first_seed if policy_reports else None,
        "horizon": horizon,
        "note": (
            "REPORT ONLY (decision D-D27). Each row turns the spawn about vertical right after reset and rolls the "
            "UNMODIFIED policy; 'animal' turns the animal alone (the prey stays, so the heading reward term pays a "
            "turn toward it), 'scene' turns the prey and the episode's reference direction with it, so the physics "
            "is the same at every heading (every plant declares the elliptic friction cone) and only a world-frame "
            "observation differs: the velociraptor's and the compsognathus robot's; the trex (policy interface "
            "r14) and the anatomical compsognathus (r4) observe no world heading (D-D31), so their 'scene' rows "
            "differ from the unturned one only by rounding the closed loop amplifies. The statue is the control. "
            "Not a gate verdict; the six-heading certificate (stance_quality_heading/v1, D-D32) is the gate, on the "
            "stages that declare it."
        ),
        "rows": rows,
    }
    lines = [
        "PROBE: the spawn turned about vertical (report only, decision D-D27; the six-heading certificate, D-D32,",
        "is the gate on the stages that declare it).",
        "Scores the unmodified policy from a MODIFIED START; not a gate result.",
        f"policy              {payload['policy']}",
        *(
            [
                f"stage               {species_display_name(policy_reports[0]['species'])} — "
                f"{stage_display_name(policy_reports[0]['stage'])} (stage {policy_reports[0]['stage']})"
            ]
            if policy_reports and "species" in policy_reports[0] and "stage" in policy_reports[0]
            else []
        ),
        f"panel               {probe_episodes} episodes per row, seeds {first_seed}-"
        f"{first_seed + probe_episodes - 1}, horizon {horizon}",
        "turned              animal = the animal alone (the prey stays; the heading term pays a turn toward it);",
        "                    scene  = the prey and the reference direction too (every plant's elliptic friction cone",
        "                             makes the physics the same at every heading; only a world-frame observation",
        "                             differs, and the trex r14 and anatomical compsognathus r4 observe none)",
        "",
        f"  {'yaw':>6} {'turned':<7}{'policy fh':>10}{'reward':>9}{'|dyaw| mean/max':>17}"
        f"{'statue fh':>11}{'reward':>9}{'|dyaw| max':>11}",
    ]
    for row in rows:
        policy, statue = row["policy"], row["statue"]
        turned = "-" if row["yaw_deg"] == 0.0 else ("scene" if row["whole_scene"] else "animal")
        lines.append(
            f"  {row['yaw_deg']:>+6.0f} {turned:<7}{policy['full_horizon_fraction']:>10.2f}"
            f"{policy['reward_mean']:>9.1f}{policy['yaw_change_mean_deg']:>8.1f}/{policy['yaw_change_max_deg']:<8.1f}"
            f"{statue['full_horizon_fraction']:>11.2f}{statue['reward_mean']:>9.1f}{statue['yaw_change_max_deg']:>11.1f}"
        )
    handoff_warnings = heading_probe_warnings(payload)
    if handoff_warnings:
        lines += [
            "",
            "  HANDOFF WARNING (the maintainer's decision of 2026-10-11: it changes no verdict and blocks nothing)",
        ]
        for message in handoff_warnings:
            lines += textwrap.wrap(
                message, width=112, initial_indent="    - ", subsequent_indent="      ", break_on_hyphens=False
            )
    lines += [
        "",
        "  The unturned row is the harness check: both columns should match the gate panel's first seeds.",
        "  A policy that stands at 0 deg and falls when turned reads its world heading (on a heading-free",
        "  observation, rounding its closed loop amplifies); compare its 'scene' rows (the question the",
        "  observation interface poses) with its 'animal' rows (which add the prey bearing).",
    ]
    return "\n".join(lines) + "\n", payload


def write_heading_probe(
    stage_dir: "str | Path",
    policy_reports: list[dict[str, Any]],
    statue_reports: list[dict[str, Any]],
    *,
    probe_episodes: int,
) -> dict[str, Path]:
    """Write the heading probe beside the stage's other artifacts."""
    directory = Path(stage_dir)
    directory.mkdir(parents=True, exist_ok=True)
    text, payload = render_heading_probe(policy_reports, statue_reports, probe_episodes=probe_episodes)
    text_path = directory / "stance_heading_probe.txt"
    json_path = directory / HEADING_PROBE_JSON
    atomic_write_text(text_path, text)
    atomic_write_json(json_path, _json_safe(payload), sort_keys=True, allow_nan=False)
    return {"heading_probe_txt": text_path, "heading_probe_json": json_path}


def _finite_number(value: Any) -> float | None:
    """*value* as a finite float, or None when it is not a number (a bool included) or overflows a float."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except OverflowError:  # a JSON integer past float range
        return None
    return number if math.isfinite(number) else None


def _probe_fraction(cell: Any) -> float | None:
    """A row cell's ``full_horizon_fraction``, or None when it is missing or not a finite share in [0, 1]."""
    share = _finite_number(cell.get("full_horizon_fraction")) if isinstance(cell, Mapping) else None
    return share if share is not None and 0.0 <= share <= 1.0 else None


def heading_probe_warnings(payload: Any, *, threshold: float = HEADING_PROBE_WARN_BELOW) -> list[str]:
    """The handoff warnings a heading-probe record raises; empty when it raises none.  Never raises.

    Only whole-scene rows (``whole_scene: true``) are read: turning the prey and
    the reference direction with the animal leaves a heading-free controller's
    task unchanged, so a fall there is the policy reading its world heading
    (or, on an observation with no world heading, rounding its closed loop
    amplifies).
    The unturned control (``yaw_deg`` 0, recorded ``whole_scene: false``) is the
    first seeds of the gate panel, which the gate itself judges; the
    animal-only rows add a response to the prey's bearing.  A statue cell
    below *threshold*, or not a share, is reported apart: the control did not
    stand, or is unmeasured, so that row does not isolate heading.  A record
    that cannot be read warns that heading robustness is unmeasured; it never
    passes silently.
    """
    unmeasured = "heading robustness is unmeasured"
    if not isinstance(payload, Mapping) or payload.get("schema") != HEADING_PROBE_SCHEMA:
        found = payload.get("schema") if isinstance(payload, Mapping) else type(payload).__name__
        return [f"the heading probe is not a {HEADING_PROBE_SCHEMA} record ({found!r}); {unmeasured}"]
    episodes = payload.get("probe_episodes")
    if isinstance(episodes, bool) or not isinstance(episodes, int) or episodes < 1:
        return [f"the heading probe records no episodes (probe_episodes {episodes!r}); {unmeasured}"]
    rows = payload.get("rows")
    if not isinstance(rows, list):
        return [f"the heading probe records no rows; {unmeasured}"]
    low: list[str] = []
    missing: list[str] = []
    statue_low: list[str] = []
    statue_missing: list[str] = []
    scene_rows = 0
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or row.get("whole_scene") is not True:
            continue
        scene_rows += 1
        yaw = _finite_number(row.get("yaw_deg"))
        label = f"{yaw:+.0f} deg" if yaw is not None else f"row {index}"
        policy, statue = _probe_fraction(row.get("policy")), _probe_fraction(row.get("statue"))
        if policy is None:
            missing.append(label)
        elif policy < threshold:
            low.append(f"{label} {policy:.2f}" + ("" if statue is None else f" (statue {statue:.2f})"))
        if statue is None:
            statue_missing.append(label)
        elif statue < threshold:
            statue_low.append(f"{label} {statue:.2f}")
    if scene_rows == 0:
        return [f"the heading probe has no whole-scene row; {unmeasured}"]
    messages = []
    if low:
        messages.append(
            f"the policy's full-horizon fraction is below {threshold:.2f}, the v2 per-panel rail, on {len(low)} of "
            f"{scene_rows} whole-scene rows ({'; '.join(low)}; {episodes} episodes per row): this stance falls when "
            "the whole scene is turned. Report only: no verdict changes and nothing is blocked (the maintainer's "
            "decision of 2026-10-11, docs/HEADING_INVARIANCE_DESIGN_2026_10.md)"
        )
    if missing:
        messages.append(
            f"the policy's full-horizon fraction is not a share in [0, 1] on whole-scene rows {', '.join(missing)}; "
            "those headings are unmeasured"
        )
    if statue_low:
        messages.append(
            f"the zero-action statue is itself below {threshold:.2f} on whole-scene rows {', '.join(statue_low)}: the "
            "control did not stand, so those rows do not isolate heading"
        )
    if statue_missing:
        messages.append(
            "the zero-action statue's full-horizon fraction is not a share in [0, 1] on whole-scene rows "
            f"{', '.join(statue_missing)}; those controls are unmeasured, so those rows do not isolate heading"
        )
    return messages


def declares_heading_probe(curriculum: Any) -> bool:
    """Whether a ``[curriculum]`` block asks for the heading probe: a stance gate kind and a non-zero offset.

    Never under the six-heading certificate (``stance_quality_heading/v1``,
    D-D32), which replaces the probe: config load refuses the key there, and
    a recorded block that carries it anyway does not make the probe due.
    """
    if not isinstance(curriculum, Mapping) or curriculum.get("gate_kind") not in STANCE_GATE_KINDS:
        return False
    if curriculum.get("gate_kind") == STANCE_HEADING_GATE_KIND:
        return False
    raw = curriculum.get("stance_probe_spawn_yaw_deg")
    values: list[Any] = list(raw) if isinstance(raw, (list, tuple)) else [raw]
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(number) and number != 0.0:
            return True
    return False


def stage_heading_probe_warnings(
    stage_dir: "str | Path",
    *,
    declared: bool | None = None,
    threshold: float = HEADING_PROBE_WARN_BELOW,
) -> list[str]:
    """:func:`heading_probe_warnings` for the probe *stage_dir* holds.  Never raises.

    An absent probe warns only when the stage declares it (*declared*; None
    reads the stage's recorded ``stage_config.json``): the command-line
    curriculum rolls no probe, and a probe that failed was logged where it
    failed, so the handoff says the heading went unmeasured.

    A stage whose recorded block declares the six-heading certificate
    (``stance_quality_heading/v1``, D-D32) reads no probe:
    :func:`stage_heading_certificate_warnings` stands in for it, silent
    exactly when the certificate re-derives to a pass.  The curriculum's
    handoff and reuse already refuse a failed one, but a manual load
    (``train(..., task_load_mode="initialize_next_stage")``) reads no
    verdict, so there the warning is the only notice.
    """
    record = read_recorded_stage_config(stage_dir) or {}
    recorded_curriculum = record.get("curriculum", record.get("curriculum_kwargs"))
    if isinstance(recorded_curriculum, Mapping) and recorded_curriculum.get("gate_kind") == STANCE_HEADING_GATE_KIND:
        return stage_heading_certificate_warnings(stage_dir, recorded_curriculum, stage=record.get("stage", "?"))
    path = Path(stage_dir) / HEADING_PROBE_JSON
    if not path.exists():
        if declared is None:
            declared = declares_heading_probe(recorded_curriculum)
        if not declared:
            return []
        # Not --out-dir <stage_dir>: it also writes a gate report there, and a complete bundle is immutable.
        return [
            f"{path} is missing although the stage declares the heading probe; heading robustness is unmeasured "
            "(stance_gate_report.py --spawn-yaw-offsets=<offsets> --out-dir <a scratch directory> rolls it; "
            "never the stage directory, where --out-dir also writes a gate report)"
        ]
    payload = read_json_object(path)
    if payload is None:
        return [f"{path} is not a readable JSON object; heading robustness is unmeasured"]
    return heading_probe_warnings(payload, threshold=threshold)


def stage_heading_certificate_warnings(
    stage_dir: "str | Path", curriculum: Mapping[str, Any], *, stage: Any = "?"
) -> list[str]:
    """One warning unless *stage_dir*'s six-heading certificate re-derives to a pass under *curriculum*.  Never raises.

    The handoff warning of a stage that declares ``stance_quality_heading/v1``
    (D-D32), in place of the probe's: ``stance_heading_report.json`` judged
    through ``reporting.gates._stance_heading_stage_gate`` (admission, then
    the re-derivation from every (seed, heading) row) under the block the
    stage recorded.  Empty when it passes; otherwise one message naming
    the certificate as missing, unreadable, refused or failed, so a parent
    handed on without its verdict being read (a manual load) is not handed
    on silently.  A warning, never a refusal.
    """
    path = Path(stage_dir) / STANCE_HEADING_REPORT_JSON
    uncertified = f"heading robustness is uncertified ({STANCE_HEADING_GATE_KIND}, D-D32)"
    if not path.exists():
        return [f"{path} is missing although the stage declares the six-heading certificate; {uncertified}"]
    report = read_json_object(path)
    if report is None:
        return [f"{path} is not a readable JSON object; {uncertified}"]
    try:
        from .gates import _stance_heading_stage_gate

        passed, failures = _stance_heading_stage_gate(curriculum, report, stage=stage, stage_dir=stage_dir)
    except Exception as exc:  # noqa: BLE001 - a warning must never stop a handoff
        return [f"{path} could not be judged ({type(exc).__name__}: {exc}); {uncertified}"]
    if passed:
        return []
    shown = failures[0] if failures else "no reason recorded"
    more = f" (and {len(failures) - 1} more)" if len(failures) > 1 else ""
    return [
        f"{path} does not re-derive to a passing certificate under the stage's recorded gate: {shown}{more}; "
        f"{uncertified}"
    ]


def render_constant_hold_ablation(reports: list[dict[str, Any]], *, probe_episodes: int) -> tuple[str, dict[str, Any]]:
    """Reduce the release ablation to its table, per-group verdicts, and payload.

    The ablation walks the path between two already-measured endpoints: the
    policy's full held pose and the statue. Only when the latter stands and
    the former reaches the horizon significantly less often -- a partial fall
    is still a fall -- is there a fall to attribute to an actuator group. Each
    row moves one actuator group between them, so the table asks which
    coordinate carries the effect rather than merely restating that the pose
    is unholdable. Every comparison is a one-sided Fisher exact test on the
    rows' full-horizon counts, recovered from each report's share and panel
    size.
    """
    rows = []
    for report in reports:
        counted = _full_horizon_count(report)
        rows.append(
            {
                "label": report["hold_constant"]["label"],
                "holds": report["hold_constant"]["holds"],
                "source": report["hold_constant"]["source"],
                "episode_length_mean": report["metrics"]["episode_length_mean"],
                "full_horizon_fraction": report["metrics"]["full_horizon_fraction"],
                "reward_mean": report["metrics"]["reward_mean"],
                "terminations": report["terminations"],
                "n_released": sum(1 for value in report["hold_constant"]["actions"] if value == 0.0),
                "episodes": report.get("episodes"),
                "full_horizon_episodes": None if counted is None else counted[0],
            }
        )
    endpoints, control_refusals = _ablation_endpoints(rows)
    applicable = not control_refusals
    verdicts = _ablation_verdicts(rows)
    alpha = float(_ABLATION_ALPHA)
    horizon = reports[0]["horizon"]
    policy = str(reports[0]["policy"]).split(" — ")[0]
    payload = {
        "schema": "mesozoic.constant-hold-ablation/v3",
        "species": reports[0]["species"],
        "stage": reports[0]["stage"],
        "policy": policy,
        "horizon": horizon,
        "episodes_per_variant": probe_episodes,
        "statistics": {
            "test": "fisher-exact",
            "sided": "one",
            "alpha": alpha,
            "multiplicity": "none",
            "statue_stands_share": _ABLATION_STATUE_STANDS,
        },
        "classification": {"applicable": applicable, "reasons": control_refusals, "endpoints": endpoints},
        "note": (
            "Each row scored a MODIFIED policy: its commanded action frozen to a constant "
            "with one actuator group returned to the home control (released) or with only "
            "that group held. No row is a gate verdict. release_G tests whether G is "
            "NECESSARY for the fall; only_G tests whether G is SUFFICIENT to cause it. "
            "These claims require a standing hold_zero control and a hold_all control that "
            "reaches the horizon significantly less often (a partial or complete full-pose "
            "fall to explain). Each claim rests on one-sided Fisher exact tests of the "
            "variant's full-horizon count against both endpoints at p <= alpha, unadjusted "
            "across groups; the variants share reset seeds and the test ignores that "
            "pairing, which errs conservative when a seed's outcomes move together across "
            "variants. "
            "Unassessed necessity/sufficiency claims are null, not false."
        ),
        "variants": rows,
        "verdicts": verdicts,
        "reports": reports,
    }
    lines = [
        "PROBE: a held constant pose, ablated one actuator group at a time.",
        "Every row scores a MODIFIED policy. None of them is a gate verdict.",
        "",
        f"policy              {policy}",
        f"stage               {species_display_name(reports[0]['species'])} — "
        f"{stage_display_name(reports[0]['stage'])} (stage {reports[0]['stage']})",
        f"panel               {probe_episodes} episodes per variant, horizon {horizon}",
        "released            commanded 0, the named home control under home-keyframe-residual/v1",
        "",
        f"  {'variant':<28}{'released':>9}{'ep length':>11}{'full-horiz':>12}{'reward':>10}   terminations",
    ]

    def _cell(value: Any, spec: str) -> str:
        # An unmeasured cell renders as '-' so the row survives to the unknown-handling below.
        return "-" if value is None else format(value, spec)

    def _count_text(counts: "tuple[int, int] | list[int] | None") -> str:
        return "-" if counts is None else f"{counts[0]}/{counts[1]}"

    def _p_text(p: float) -> str:
        # Below 1e-3 four decimals keep at most one significant digit (1/12870 would print as
        # 0.0001), so switch to scientific notation there.
        return f"{p:.4f}" if p >= 0.001 else f"{p:.1e}"

    def _p_cell(side: dict[str, Any] | None, against: str) -> str:
        # '*' marks p <= alpha; the marker column stays blank otherwise so the digits align.
        comparison = None if side is None else side[f"vs_{against}"]
        if comparison is None or comparison["status"] == "unmeasured":
            return "- "
        if comparison["status"] == "insufficient episodes":
            return "n/a "
        return _p_text(comparison["p"]) + ("*" if comparison["status"] == "significant" else " ")

    for row in rows:
        released = "-" if not row["holds"] else str(row["n_released"])
        lines.append(
            f"  {row['label']:<28}{released:>9}{_cell(row['episode_length_mean'], '.1f'):>11}"
            f"{_cell(row['full_horizon_fraction'], '.4f'):>12}{_cell(row['reward_mean'], '.1f'):>10}"
            f"   {row['terminations']}"
        )
    if not applicable:
        lines += ["", "Classification: not applicable."]
        lines.extend(f"  {reason}" for reason in control_refusals)
        lines.append("  Raw variant measurements are retained; causal labels are suppressed.")
    if verdicts:
        counts_by_label = {row["label"]: _ablation_counts(row) for row in rows}
        separation = endpoints["separation"]
        endpoint_line = (
            f"  endpoints     hold_all {_count_text(endpoints['hold_all'])}, "
            f"hold_zero {_count_text(endpoints['hold_zero'])} full-horizon"
        )
        if separation is not None and separation["status"] == "insufficient episodes":
            endpoint_line += f"; too few episodes for one-sided Fisher p <= {alpha:g}"
        elif separation is not None:
            relation = "<=" if separation["status"] == "significant" else ">"
            endpoint_line += f"; one-sided Fisher p = {_p_text(separation['p'])} {relation} {alpha:g}"
        lines += [
            "",
            "Per group — release_G asks if G is NECESSARY (does removing it rescue the pose?),",
            "only_G asks if G is SUFFICIENT (does it alone break the statue?). Each variant's",
            "full-horizon episodes, then the one-sided Fisher exact p that it reaches the",
            f"horizon more (>) or less (<) often than each endpoint: '*' marks p <= {alpha:g},",
            "'n/a' too few episodes for any outcome to reach it, '-' unmeasured or unassessed.",
            "",
            endpoint_line,
            "",
            f"  {'group':<14}{'release':>9}{'>hold_all ':>13}{'<hold_zero ':>13}"
            f"{'only':>9}{'<hold_zero ':>13}{'>hold_all ':>13}  verdict",
        ]
        for entry in verdicts:
            group, necessity, sufficiency = entry["group"], entry["necessity"], entry["sufficiency"]
            lines.append(
                f"  {group:<14}{_count_text(counts_by_label.get(f'release_{group}')):>9}"
                f"{_p_cell(necessity, 'hold_all'):>13}{_p_cell(necessity, 'hold_zero'):>13}"
                f"{_count_text(counts_by_label.get(f'only_{group}')):>9}"
                f"{_p_cell(sufficiency, 'hold_zero'):>13}{_p_cell(sufficiency, 'hold_all'):>13}"
                f"  {entry['verdict']}"
            )
    if (
        applicable
        and verdicts
        and all(entry["necessary"] is False and entry["sufficient"] is not None for entry in verdicts)
    ):
        sufficient = [entry["group"] for entry in verdicts if entry["sufficient"]]
        lines += ["", "Overall:"]
        if sufficient:
            lines += [
                f"  No group is necessary, but {', '.join(sufficient)} " + ("is" if len(sufficient) == 1 else "are"),
                "  sufficient. The pose is OVER-DETERMINED: more than one subset of it breaks",
                "  the animal independently, so removing any single group leaves another still",
                "  able to. 'Which joint is the cause' is therefore the wrong question -- but the",
                "  sufficient groups are where the effect is concentrated, and a group that holds",
                "  the full horizon on its own is exonerated no matter how extreme its DC looks.",
            ]
        else:
            lines += [
                "  No group is necessary and none is sufficient. Whatever makes the pose",
                "  unholdable is not separated by these groupings -- it may be a combination",
                "  that crosses them, or a joint outside every group. Regroup before concluding.",
            ]
    if applicable:
        lines += [
            "",
            "Read the pair, never one side. A group can look implicated because it is",
            "conspicuous -- twelve actuators pinned at exactly +/-1.000 are hard to ignore --",
            "and still be a bystander. Only necessary AND sufficient identifies a cause;",
            "necessary alone means it contributes with others, and neither means the",
            "stabilisation load lives somewhere this ablation did not separate.",
        ]
    lines += [
        "",
        "Causal classification requires both endpoint controls: 'hold_zero' must stand",
        f"(in at least {_ABLATION_STATUE_STANDS:g} of its episodes), and 'hold_all' must reach the horizon",
        "significantly less often -- a partial fall is still a fall to explain.",
        "Otherwise the experiment does not establish a held-pose fall to explain; the",
        "measurements remain useful as observations. Each label rests on one-sided",
        f"Fisher exact tests at p <= {alpha:g} against the endpoint the variant moves away",
        "from, unadjusted across groups; the variants share reset seeds and the test",
        "ignores that pairing, which errs conservative when a seed's outcomes move",
        "together across variants.",
    ]
    return "\n".join(lines) + "\n", payload


def write_constant_hold_ablation(
    stage_dir: "str | Path", reports: list[dict[str, Any]], *, probe_episodes: int
) -> dict[str, Path]:
    """Write the release ablation beside the stage's other artifacts.

    Its own filenames, and never ``stance_panel_selected.csv`` -- same
    discipline as the other two probes, for the same reason.
    """
    directory = Path(stage_dir)
    directory.mkdir(parents=True, exist_ok=True)
    text, payload = render_constant_hold_ablation(reports, probe_episodes=probe_episodes)
    text_path = directory / "stance_gate_probe_release.txt"
    json_path = directory / "stance_gate_probe_release.json"
    atomic_write_text(text_path, text)
    atomic_write_json(json_path, _json_safe(payload), sort_keys=True, allow_nan=False)
    return {"constant_hold_ablation_txt": text_path, "constant_hold_ablation_json": json_path}


def write_constant_hold_probe(
    stage_dir: "str | Path", reports: list[dict[str, Any]], *, probe_episodes: int
) -> dict[str, Path]:
    """Write the constant-hold probe beside the stage's other artifacts.

    Its own filenames, and never ``stance_panel_selected.csv`` -- the same
    discipline the filtered probe follows, for the same reason: no row here
    scored the policy a bundle would be certified from.
    """
    directory = Path(stage_dir)
    directory.mkdir(parents=True, exist_ok=True)
    text, payload = render_constant_hold_probe(reports, probe_episodes=probe_episodes)
    text_path = directory / "stance_gate_probe_constant.txt"
    json_path = directory / "stance_gate_probe_constant.json"
    atomic_write_text(text_path, text)
    atomic_write_json(json_path, _json_safe(payload), sort_keys=True, allow_nan=False)
    return {"constant_hold_probe_txt": text_path, "constant_hold_probe_json": json_path}


#: Anatomical actuator groups, as substrings matched against joint names. Used
#: to ablate a held pose one region at a time. Substrings rather than indices so
#: the same groups resolve on any species whose joints follow the naming
#: convention, and so adding a joint does not silently fall out of its group.
_ACTUATOR_GROUPS: dict[str, tuple[str, ...]] = {
    "tail": ("tail",),
    "toes": ("toe",),
    "head_neck": ("head", "neck"),
    "hip_rolls": ("hip_roll",),
    # Added for the passive-toes plant (physics r7): the 10M checkpoint's
    # ablation returned "no group necessary, none sufficient" over the groups
    # above while its degree table concentrated the tremor in the knees,
    # ankles and hip rolls -- the distal chain the old grouping never tested
    # as a unit. Per-side groups because that checkpoint's tremor is strongly
    # asymmetric (l_ankle +/-39.2deg vs r_ankle far less), so a symmetric
    # grouping can hide a one-legged mechanism.
    "knees_ankles": ("knee", "ankle"),
    "left_leg": ("l_hip", "l_knee", "l_ankle"),
    "right_leg": ("r_hip", "r_knee", "r_ankle"),
}


def actuator_indices_matching(report: dict[str, Any], patterns: "tuple[str, ...] | list[str]") -> tuple[int, ...]:
    """Indices whose joint name contains any of *patterns*."""
    per_actuator = (report.get("action") or {}).get("per_actuator") or []
    return tuple(
        int(entry["index"])
        for entry in sorted(per_actuator, key=lambda e: e["index"])
        if any(pattern in str(entry["joint"]) for pattern in patterns)
    )


def saturated_actuator_indices(report: dict[str, Any], threshold: float = 0.99) -> tuple[int, ...]:
    """Indices whose commanded pose sits at or beyond *threshold* of the action range.

    A saturated actuator has no headroom in one direction, so it cannot
    contribute correction on that axis -- and on the T-Rex stage-1 checkpoint
    the saturated set is also exactly the set with zero AC, i.e. the joints the
    policy pins and then never moves.
    """
    per_actuator = (report.get("action") or {}).get("per_actuator") or []
    return tuple(
        int(entry["index"])
        for entry in sorted(per_actuator, key=lambda e: e["index"])
        if abs(float(entry["dc"])) >= threshold
    )


def constant_hold_released(hold: tuple[float, ...], release: "tuple[int, ...] | set[int]") -> tuple[float, ...]:
    """*hold* with the given actuators returned to the home control.

    "Released" means commanded ``0``, which under ``home-keyframe-residual/v1``
    is exactly the named home control -- so releasing every actuator gives the
    statue, and releasing none gives the policy's own pose. Every ablation
    below is a point on the path between a pose that falls and one that stands.
    """
    released = set(int(index) for index in release)
    return tuple(0.0 if index in released else value for index, value in enumerate(hold))


def constant_hold_release_variants(
    hold: tuple[float, ...],
    report: dict[str, Any],
    *,
    horizon: int,
    ramp_steps: int = 50,
) -> list[ConstantHold]:
    """Ablate the held pose one actuator group at a time, both ways round.

    Every ablated variant is commanded **from reset**, not from a handoff
    partway through. That is load-bearing and was got wrong the first time:
    handing off at ``settle_steps`` means the policy first establishes its own
    displaced pose and velocities, and the ablation then snaps some subset of
    joints from ``±1.000`` to ``0`` in one step. The transient dominates, every
    variant lands within noise of every other (311-349 steps across all
    thirteen), and -- decisively -- the all-released endpoint falls too, at 323
    steps, when the statue it is supposed to be stands for 1000. Both endpoints
    on the same side of the answer means the path has no signal to locate.

    From reset the endpoints are the two known measurements: releasing every
    actuator commands ``0`` throughout, which *is* the statue and stands the
    full horizon, and releasing none reproduces the probe's own
    ``hold_from_reset`` at 133 steps. The path then genuinely runs from falling
    to standing, and an ablation on it locates which coordinate carries the
    effect.

    Each group gets two variants, because "which joints cause the fall" needs
    two different questions answered and one of them alone is not evidence:

    * ``release_G`` holds everything *except* G, testing **necessity** -- does
      taking G away rescue a pose that otherwise falls?
    * ``only_G`` holds *only* G, testing **sufficiency** -- does adding G alone
      break the statue, which is known to stand?

    When the controls establish a standing statue and a full pose that falls
    significantly more often (a partial fall counts), a group that is both
    necessary and sufficient is the cause. A group that is necessary but not sufficient
    contributes with others. A group that is neither is a bystander, whatever
    its DC looks like -- and the saturated joints look dramatic (twelve pinned
    at exactly +/-1.000) precisely because saturation is visible, not because
    it was shown to matter.

    The two endpoints of the path are included as controls: ``hold_all`` is the
    full pose, which must fall significantly more often than the statue for
    the rows to be classified, and ``hold_zero`` is the statue, which must
    stand. Every ablation sits between them.
    """
    zero = tuple(0.0 for _ in hold)
    groups: dict[str, tuple[int, ...]] = {"saturated": saturated_actuator_indices(report)}
    for name, patterns in _ACTUATOR_GROUPS.items():
        indices = actuator_indices_matching(report, patterns)
        if indices:
            groups[name] = indices
    everything = tuple(range(len(hold)))

    variants = [
        ConstantHold(hold, handoff_steps=horizon, label="policy (control)", source="unmodified", holds=False),
        ConstantHold(hold, handoff_steps=0, ramp_steps=ramp_steps, label="hold_all"),
    ]
    for name, indices in groups.items():
        if not indices or len(indices) == len(hold):
            # Releasing everything is the statue and releasing nothing is
            # `hold_all`; neither is an ablation, and both are already rows.
            continue
        complement = tuple(index for index in everything if index not in set(indices))
        variants.append(
            ConstantHold(
                constant_hold_released(hold, indices),
                handoff_steps=0,
                ramp_steps=ramp_steps,
                label=f"release_{name}",
            )
        )
        variants.append(
            ConstantHold(
                constant_hold_released(hold, complement),
                handoff_steps=0,
                ramp_steps=ramp_steps,
                label=f"only_{name}",
            )
        )
    variants.append(
        ConstantHold(zero, handoff_steps=0, ramp_steps=ramp_steps, label="hold_zero (statue control)", source="zeros")
    )
    return variants


def constant_hold_variants(
    hold: tuple[float, ...], *, settle_steps: int, horizon: int, ramp_steps: int = 50
) -> list[ConstantHold]:
    """The standard variant set: two controls bracketing three held rollouts.

    ``handoff_steps = horizon`` is how the unmodified control is expressed. It
    rolls the real policy for the whole episode yet still carries the
    ``hold_constant`` marker, so it can never be written to the certification
    filenames by a caller who forgets which report is which.
    """
    zero = tuple(0.0 for _ in hold)
    return [
        ConstantHold(hold, handoff_steps=horizon, label="policy (control)", source="unmodified", holds=False),
        ConstantHold(hold, handoff_steps=settle_steps, ramp_steps=0, label="hold_after_settle"),
        ConstantHold(hold, handoff_steps=settle_steps, ramp_steps=ramp_steps, label="hold_after_settle_ramped"),
        ConstantHold(hold, handoff_steps=0, ramp_steps=ramp_steps, label="hold_from_reset"),
        ConstantHold(zero, handoff_steps=0, ramp_steps=0, label="hold_zero (statue control)", source="zeros"),
    ]


#: Column order of ``stance_panel_selected.csv``.  ``unsupported_duty`` is
#: empty for an episode whose duty could not be measured (one shorter than the
#: settling window); the auditor treats empty as unmeasured rather than zero,
#: because zero is the statue's value and the best possible score.
STANCE_PANEL_FIELDNAMES = (
    "episode",
    "panel_seed",
    "length",
    "reward",
    "reached_horizon",
    "unsupported_duty",
    "bilateral_support_duty",
    "single_support_duty",
)


def write_stance_panel_evidence(stage_dir: "str | Path", report: dict[str, Any]) -> "Path | None":
    """Write the panel's per-episode measurements as publication evidence.

    Returns ``None`` when the report carries no ``episode_evidence`` -- an
    older report, or one built before the field existed. The caller records
    only what was written, and the auditor's own refusal covers the absence,
    so a missing file fails the bundle rather than passing it.
    """
    episodes = report.get("episode_evidence")
    if not episodes:
        return None
    if report.get("scored_gate_kind") == STANCE_GATE_V2_KIND:
        return _write_stance_v2_panel_evidence(Path(stage_dir), report)
    rows = []
    for episode in episodes:
        duty = episode.get("unsupported_duty")
        rows.append(
            {
                "episode": episode["episode"],
                "panel_seed": episode["seed"],
                "length": int(episode["length"]),
                "reward": float(episode["reward"]),
                "reached_horizon": bool(episode["reached_horizon"]),
                "unsupported_duty": "" if duty is None else float(duty),
                "bilateral_support_duty": _optional_float_cell(episode.get("bilateral_support_duty")),
                "single_support_duty": _optional_float_cell(episode.get("single_support_duty")),
            }
        )
    # Every row is built before the file is touched, and the table is
    # published atomically: a bad episode or a reclaim leaves the previous
    # file, never a partial one (CU-3; the bytes are unchanged).
    return atomic_write_csv(Path(stage_dir) / "stance_panel_selected.csv", STANCE_PANEL_FIELDNAMES, rows)


def _optional_float_cell(value: Any) -> str | float:
    return "" if value is None else float(value)


def _float_cell(value: Any) -> str:
    """A float column's cell: ``repr`` round-trips exactly; an unmeasured value is ``nan``, never blank-as-zero."""
    if value is None:
        return "nan"
    return repr(float(value))


def _floats_cell(values: Any) -> str:
    return "[" + ", ".join(_float_cell(value) for value in values) + "]"


def _write_stance_v2_panel_evidence(stage_dir: Path, report: dict[str, Any]) -> Path:
    """The ``stance_quality/v2`` panel CSV: every episode's floor-truth row, stamped with the panel's binding.

    Columns are :data:`~environments.shared.curriculum.stance_gate_v2.STANCE_V2_PANEL_FIELDNAMES`.
    Every metric is written so ``StanceEpisodeMetrics.from_row`` reads it back
    bit for bit (``repr`` floats, ``nan`` for unmeasured), and every row
    carries the scored pair's digests, the measurement version and digest
    and the statue reference, so publication can re-derive the verdict from
    this file alone and bind it to the certified checkpoint, the task the
    stage recorded and this checkout's measurement definition -- the
    re-derivation ``result_bundle.evidence`` performs.
    """
    from ..gait.constants import measurement_definition_sha256

    handoff = report.get("handoff") or {}
    measurement = report.get("measurement") or {}
    statue = (report.get("statue") or {}).get("reference")
    stamps = {
        "checkpoint_sha256": handoff.get("checkpoint_sha256") or "",
        "normalization_sha256": handoff.get("normalization_sha256") or "",
        "task_sha256": report.get("task_sha256") or "",
        "measurement_version": (measurement.get("constants") or {}).get("measurement_version") or "",
        "measurement_sha256": report.get("measurement_sha256") or "",
        "measurement_definition_sha256": measurement_definition_sha256(measurement) if measurement else "",
        **_statue_stamps(statue),
    }
    rows = []
    for episode in report["episode_evidence"]:
        row: dict[str, Any] = {
            **_panel_episode_cells(episode),
            "clean": bool(episode["clean"]),
            "reasons": "; ".join(episode.get("reasons") or ()),
        }
        row.update(stamps)
        rows.append(row)
    return atomic_write_csv(stage_dir / "stance_panel_selected.csv", STANCE_V2_PANEL_FIELDNAMES, rows)


def _panel_episode_cells(episode: Mapping[str, Any]) -> dict[str, Any]:
    """One evidence row's panel CSV cells: its seed, horizon flag, touch duties and every floor-truth metric.

    Every metric is written so ``StanceEpisodeMetrics.from_row`` reads it
    back bit for bit (``repr`` floats, ``nan`` for unmeasured).
    """
    row: dict[str, Any] = {
        "episode": episode["episode"],
        "panel_seed": episode["seed"],
        "reached_horizon": bool(episode["reached_horizon"]),
        "unsupported_duty": _optional_float_cell(episode.get("unsupported_duty")),
        "bilateral_support_duty": _optional_float_cell(episode.get("bilateral_support_duty")),
        "single_support_duty": _optional_float_cell(episode.get("single_support_duty")),
    }
    for name in STANCE_METRIC_FIELDS:
        value = episode[name]
        if name in ("length", "settle_steps"):
            row[name] = int(value)
        elif name == "terminated":
            row[name] = bool(value)
        else:
            row[name] = _float_cell(value)
    for name in STANCE_METRIC_FOOT_FIELDS:
        row[name] = _floats_cell(episode[name])
    return row


def _statue_stamps(reference: Mapping[str, Any] | None) -> dict[str, Any]:
    """The four statue-reference columns of a panel row: blank when no statue panel was rolled."""
    return {
        "statue_n_episodes": "" if reference is None else int(reference["n_episodes"]),
        "statue_n_full_horizon": "" if reference is None else int(reference["n_full_horizon"]),
        "statue_mean_reward": "" if reference is None else _float_cell(reference["mean_reward"]),
        "statue_foot_load_share": "" if reference is None else _floats_cell(reference["foot_load_share"]),
    }


def _write_stance_heading_panel_evidence(stage_dir: Path, report: Mapping[str, Any]) -> Path:
    """The six-heading certificate's grid CSV: every policy and statue row, stamped with the certificate's binding.

    Columns are :data:`~environments.shared.curriculum.stance_heading_gate.STANCE_HEADING_PANEL_FIELDNAMES`.
    Per heading, its policy rows (``role`` ``policy``, classified) and its
    statue rows (``role`` ``statue``, the reference, never classified), each
    stamping that heading's statue reference; every row stamps the handoff
    digests, the task, the measurement version and digests, the gate digest,
    the protocol and its grid, the seed block, the horizon, the control
    step and the policy's determinism -- everything publication binds and
    re-derives from (``result_bundle.evidence``) without the JSON.
    """
    from ..gait.constants import measurement_definition_sha256

    handoff = report.get("handoff") or {}
    measurement = report.get("measurement") or {}
    seeds = list(report.get("seeds") or ())
    stamps = {
        "checkpoint_sha256": handoff.get("checkpoint_sha256") or "",
        "normalization_sha256": handoff.get("normalization_sha256") or "",
        "task_sha256": report.get("task_sha256") or "",
        "measurement_version": (measurement.get("constants") or {}).get("measurement_version") or "",
        "measurement_sha256": report.get("measurement_sha256") or "",
        "measurement_definition_sha256": measurement_definition_sha256(dict(measurement)) if measurement else "",
        "gate_sha256": report.get("gate_sha256") or "",
        "protocol": report.get("protocol") or "",
        "heading_grid_deg": heading_grid_cell(report.get("heading_grid_deg") or ()),
        "seed_start": seeds[0] if seeds else "",
        "n_seeds": len(seeds),
        "horizon": int(report["horizon"]),
        "control_dt": _float_cell(report.get("control_dt")),
        "policy_deterministic": bool(report.get("policy_deterministic")),
    }
    rows: list[dict[str, Any]] = []
    for heading in report["headings"]:
        statue = heading.get("statue") or None
        heading_stamps = {
            "yaw_deg": _float_cell(heading["yaw_deg"]),
            **_statue_stamps(None if statue is None else statue["reference"]),
            **stamps,
        }
        for episode in heading["episode_evidence"]:
            rows.append(
                {
                    "role": POLICY_ROLE,
                    **_panel_episode_cells(episode),
                    "clean": bool(episode["clean"]),
                    "reasons": "; ".join(episode.get("reasons") or ()),
                    **heading_stamps,
                }
            )
        for episode in [] if statue is None else statue["episode_evidence"]:
            rows.append(
                {"role": STATUE_ROLE, **_panel_episode_cells(episode), "clean": "", "reasons": "", **heading_stamps}
            )
    return atomic_write_csv(stage_dir / STANCE_HEADING_PANEL_CSV, STANCE_HEADING_PANEL_FIELDNAMES, rows)
