"""Stage-aware Stable-Baselines3 evaluation diagnostics.

The rollout diagnostics in :mod:`environments.shared.diagnostics` describe
what the training workers are doing.  Plateau decisions belong here instead:
they use deterministic evaluation episodes and the same configured metrics
that define each curriculum stage.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from .curriculum.manager import DEFAULT_MIN_EVAL_EPISODES
from .curriculum.stance_gate import required_duty_episodes
from .curriculum.task_success_gate import TASK_SUCCESS_GATE_KIND
from .model_id import ModelId
from .stance_diagnostics import derive_stance_info

logger = logging.getLogger(__name__)

#: The sagittal leg joints whose left-minus-right DC the plateau callback
#: records (``leg_dc_<left>_minus_<right>_<joint>_deg``): the offsets that,
#: with both feet planted, move the OTHER pad's centre of pressure (0.19-0.23
#: pad half-lengths per degree on the T. rex anatomical stance).
SAGITTAL_LEG_JOINTS: tuple[str, ...] = ("hip_pitch", "knee", "ankle")

#: The prefix of ``gate_progress.npz``'s per-evaluation termination-cause counts: ``termination_<cause>``
#: (``termination_truncated`` for the episodes that reached their time limit).
TERMINATION_KEY_PREFIX = "termination_"

#: A sagittal joint's axis must lie within about 45 degrees of the animal's
#: lateral axis at the home pose (``|axis . lateral| >= 0.7``) for its pair to
#: be read: the leg DC is a rotation about that axis, and a joint far off it
#: does not pitch the leg.  Every manifest model's hip pitch, knee and ankle
#: read 0.99 or more.
SAGITTAL_AXIS_MIN_LATERAL = 0.7


class SagittalLegPair(NamedTuple):
    """One left/right sagittal joint pair of the leg DC (:func:`sagittal_leg_dc_pairs`).

    ``left`` and ``right`` are actuator indices (an ``action_dc_per_actuator``
    column).  ``left_sign`` and ``right_sign`` are each joint axis's sign
    along the animal's leftward axis at the home pose, +1 or -1: ``sign *
    angle`` is then a rotation about the leftward axis by the right-hand rule
    on either side, so a mirror-symmetric posture reads the same on both and
    their difference is 0.  The compsognathus robot needs them: its left hip
    pitch axis points right and its right one left, while its knees and
    ankles both point right.
    """

    key: str
    left: int
    right: int
    left_sign: float
    right_sign: float


def _in_process_envs(vec_env: Any) -> list[Any] | None:
    """The sub-envs of an in-process VecEnv (a DummyVecEnv, through VecNormalize), or ``None``.

    The floor-truth recorder hooks the env object itself, so a
    SubprocVecEnv's envs, which live in other processes, cannot be recorded.
    """
    current = vec_env
    for _ in range(16):
        if current is None:
            return None
        envs = current.__dict__.get("envs") if hasattr(current, "__dict__") else None
        if isinstance(envs, (list, tuple)) and envs:
            return list(envs)
        current = current.__dict__.get("venv") if hasattr(current, "__dict__") else None
    return None


def sagittal_leg_dc_pairs(env: Any, morphology: Any) -> tuple[SagittalLegPair, ...]:
    """The sagittal joint pairs of every left/right foot pair: :class:`SagittalLegPair`, in foot then joint order.

    The pairs are the feet whose labels differ only in the side letter
    (``l``/``r``; ``fl``/``fr`` and ``rl``/``rr`` on a quadruped), and a
    side's joints are spelled with its reference site's prefix
    (``FOOT_SUPPORT``'s ``foot_site``: ``r_foot`` -> ``r_knee``,
    ``right_foot_touch_volume`` -> ``right_knee``), the rule
    ``stance_diagnostics._biped_side_names`` reads the leg-home columns by.

    A pair is read only when both joints are driven by a direct, unit-gear
    hinge position servo (the stance report's
    ``_is_angular_position_control``: a control value is then a joint angle
    in radians) and both axes lie within :data:`SAGITTAL_AXIS_MIN_LATERAL`
    of the lateral axis; any other joint is left out.  The lateral axis is
    the horizontal direction from the right foot's reference site to the
    left one's in the model's ``home`` keyframe (``qpos0`` without one),
    computed on a scratch ``MjData``: the env's own state is never touched.
    """
    import mujoco

    from .reporting.stance_report import _is_angular_position_control

    unwrapped = getattr(env, "unwrapped", env)
    model = unwrapped.model
    servo_of_joint: dict[str, int] = {}
    for actuator in range(int(model.nu)):
        joint_id = int(model.actuator_trnid[actuator, 0])
        if _is_angular_position_control(model, actuator, joint_id):
            servo_of_joint.setdefault(str(model.joint(joint_id).name), actuator)
    feet = {foot.label: foot for foot in morphology.feet}
    prefix = {label: str(model.site(foot.primary_site).name).split("_", 1)[0] for label, foot in feet.items()}
    data = mujoco.MjData(model)
    home = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if home >= 0:
        mujoco.mj_resetDataKeyframe(model, data, home)
    mujoco.mj_kinematics(model, data)
    pairs: list[SagittalLegPair] = []
    for left in prefix:
        right = left[:-1] + "r"
        if not left.endswith("l") or right not in prefix:
            continue
        lateral = np.array(data.site_xpos[feet[left].primary_site] - data.site_xpos[feet[right].primary_site])
        lateral[2] = 0.0
        norm = float(np.linalg.norm(lateral))
        if not norm > 0.0:
            continue
        lateral /= norm
        for joint in SAGITTAL_LEG_JOINTS:
            left_actuator = servo_of_joint.get(f"{prefix[left]}_{joint}")
            right_actuator = servo_of_joint.get(f"{prefix[right]}_{joint}")
            if left_actuator is None or right_actuator is None:
                continue
            components = [
                float(np.dot(data.xaxis[int(model.actuator_trnid[actuator, 0])], lateral))
                for actuator in (left_actuator, right_actuator)
            ]
            if min(abs(component) for component in components) < SAGITTAL_AXIS_MIN_LATERAL:
                continue
            pairs.append(
                SagittalLegPair(
                    key=f"leg_dc_{left}_minus_{right}_{joint}_deg",
                    left=left_actuator,
                    right=right_actuator,
                    left_sign=math.copysign(1.0, components[0]),
                    right_sign=math.copysign(1.0, components[1]),
                )
            )
    return tuple(pairs)


def leg_dc_degrees(env: Any, pairs: "tuple[SagittalLegPair, ...]", action_dc: Any) -> dict[str, float]:
    """Each pair's left-minus-right commanded joint offset, in degrees, about the leftward axis.

    *action_dc* is one evaluation's per-actuator action DC
    (``action_dc_per_actuator``).  It is mapped through the env's own
    ``_scale_action`` (as the stance report probes it), less the control of
    action zero, the home control: that is the joint target offset the DC
    commands, in radians on the servos :func:`sagittal_leg_dc_pairs` keeps.
    Each side's offset is signed by its axis along the leftward axis, and the
    right one is subtracted from the left one, so a mirror-symmetric offset
    reads 0 on every model and the zero action reads exactly 0.  The DC is
    mapped, not each step's action, so under a curved map (the anatomical
    compsognathus legs' softcubic residual) this is the target of the mean
    action rather than the mean target, which differ by the curvature times
    the tremor's variance.  A DC that is empty, not finite or not one value
    per actuator gives NaN for every key.  The function reads the map and
    the model only.
    """
    out = {pair.key: float("nan") for pair in pairs}
    if not pairs:
        return out
    unwrapped = getattr(env, "unwrapped", env)
    dc = np.asarray(action_dc, dtype=np.float64).reshape(-1)
    if dc.size != int(unwrapped.model.nu) or not np.isfinite(dc).all():
        return out
    target = np.asarray(unwrapped._scale_action(dc), dtype=np.float64).reshape(-1)
    home = np.asarray(unwrapped._scale_action(np.zeros_like(dc)), dtype=np.float64).reshape(-1)
    offset = target - home
    for pair in pairs:
        out[pair.key] = math.degrees(pair.left_sign * offset[pair.left] - pair.right_sign * offset[pair.right])
    return out


try:
    from stable_baselines3.common.callbacks import BaseCallback as _BaseCallback
    from stable_baselines3.common.callbacks import EvalCallback as _EvalCallback

    _SB3_AVAILABLE = True
except ImportError:
    _BaseCallback = object  # type: ignore[misc,assignment]
    _EvalCallback = object  # type: ignore[misc,assignment]
    _SB3_AVAILABLE = False


def success_metric_applicable(stage_config: dict[str, Any]) -> bool:
    """Return whether task success is an active gate for *stage_config*.

    Stage number alone is deliberately not used: the TOML curriculum config
    is the authority, and a genuine zero remains meaningful whenever a
    positive success-rate threshold is configured — or whenever the stage
    declares ``task_success/v1``, whose bar is ``min_success_lcb`` (plan
    §4.4) and which consumes no ``min_success_rate`` key at all.
    """

    curriculum = stage_config.get("curriculum_kwargs", {})
    if curriculum.get("gate_kind") == TASK_SUCCESS_GATE_KIND:
        return True
    return float(curriculum.get("min_success_rate", 0.0)) > 0.0


class StageAwareEvalCallback(_EvalCallback):  # type: ignore[misc]
    """SB3 ``EvalCallback`` with stage-aware success and velocity capture.

    Stable-Baselines3 prints a success rate whenever an environment emits
    terminal ``info["is_success"]``.  The dinosaur environments intentionally
    keep emitting that standard field, but success is only an evaluation
    metric in stages whose config has a positive success-rate gate.  This
    callback suppresses collection in other stages and prints an explicit
    ``N/A`` instead of a misleading ``0.00%``.

    Mean forward velocity is also captured for every evaluation episode so
    the plateau diagnostic can follow the Stage 2 locomotion gate without
    launching another evaluation pass.

    Per-episode **unsupported duty** is captured the same way, for the
    ``stance_quality/v1`` advancement gate.  It has to come from this pass
    rather than from the small supplementary evaluation: the gate's bound is
    specified at the full evaluation panel size, and reducing it to the
    10-episode supplementary sample would throw away most of the power the
    bound exists to provide.

    The duty signal is :func:`~environments.shared.stance_diagnostics.derive_stance_info`'s
    ``unsupported_duty`` at its default 0.1 N contact threshold -- deliberately
    *not* the reward's 42 N ``min_support_force``.  Every calibration point the
    gate's ceiling rests on is measured at 0.1 N: the statue's 0.000 and run
    ``20260801_021545``'s 0.319.  Switching thresholds here would silently
    shift duty upward and invalidate the 0.02 ceiling.

    Given its *model_id*, it also records **per-foot pad telemetry** for each
    evaluation episode: the floor-truth recorder
    (:class:`~environments.shared.gait.recorder.SubstepContactRecorder`) is
    attached to the evaluation env for the evaluation only -- never to the
    training envs, whose rollouts it would slow by about a third -- and each
    episode is reduced by
    :func:`~environments.shared.gait.pad_telemetry.episode_pad_telemetry`
    (each box-soled foot's signed fore-aft centre of pressure, the gate's own
    per-foot statistic, and the digits' share of the foot's load; on a foot
    that declares a support span, the whole-foot centre of pressure, its
    rocking and its moment margins; and each digit's load share and, on a
    hinged digit, where along it the load sits).  It reads
    the physics and never writes it, so the evaluation is the one an
    unrecorded callback runs, and anything that goes wrong with it costs the
    telemetry, not the run.  The model is also described once for the
    **sagittal leg DC** (:attr:`leg_dc_pairs`), and after each evaluation
    its per-actuator action DC is mapped through the env's action map into
    each pair's left-minus-right commanded joint offset, in degrees about
    the leftward axis (:func:`leg_dc_degrees`, :attr:`evaluations_leg_dc`).
    The same recorded episodes give each box sole's **roll rails**
    (:func:`~environments.shared.gait.pad_telemetry.episode_rail_telemetry`:
    its window-mean tilt and corner lift, whose maxima over the feet are the
    gate's ``max_sole_tilt_deg`` and ``max_sole_corner_lift_m``), kept with
    the pad values (:attr:`foot_rail_keys`): the pad feet, so no model
    records them without a recorder it already attaches.

    Every evaluation episode's **termination cause** is kept too
    (:attr:`evaluations_termination_causes`): the env's
    ``termination_reason`` for an episode it terminated (``"terminated"``
    when it gave none), and ``"truncated"`` for one that reached its time
    limit, read from the episode's last ``info`` as the stance report reads
    it.
    """

    #: Declared at class scope so an instance built without ``__init__`` (the
    #: suite's ``object.__new__`` callbacks) still has them: the model whose
    #: feet the pad telemetry reads (``None``: none is recorded), and what its
    #: first evaluation resolved -- the pad keys
    #: (:meth:`~environments.shared.gait.pad_telemetry.PadLayout.episode_keys`,
    #: empty for a model with no box sole) and the sagittal leg DC pairs
    #: (:class:`SagittalLegPair`).  Both stay empty for good when the model
    #: cannot be described, so every evaluation of a run carries the same
    #: keys.
    model_id: "ModelId | None" = None
    foot_pad_keys: tuple[str, ...] = ()
    #: The roll-rail keys of the model's box-soled feet
    #: (:meth:`~environments.shared.gait.pad_telemetry.RailLayout.episode_keys`),
    #: resolved with the pad keys; their per-episode values are kept in the
    #: same :attr:`evaluations_foot_pad` records.
    foot_rail_keys: tuple[str, ...] = ()
    leg_dc_pairs: tuple[SagittalLegPair, ...] = ()
    _foot_resolved = False
    _warned_foot_telemetry = False

    def __init__(
        self,
        eval_env: Any,
        *,
        stage: "int | str",
        success_applicable: bool,
        settle_steps: int = 0,
        model_id: "ModelId | None" = None,
        **kwargs: Any,
    ) -> None:
        if not _SB3_AVAILABLE:
            raise ImportError(
                "stable-baselines3 is required for StageAwareEvalCallback. Install with: pip install stable-baselines3"
            )
        super().__init__(eval_env, **kwargs)
        self.stage = stage
        self.success_applicable = success_applicable
        self.settle_steps = int(settle_steps)
        self.model_id = model_id
        # Per evaluation, the pad telemetry's per-episode values by key
        # (`foot_pad_keys`, then the roll rails' `foot_rail_keys`),
        # positionally aligned with `evaluations_length` like the duties; `{}`
        # for a model with no box sole.
        self.evaluations_foot_pad: list[dict[str, list[float]]] = []
        # Per evaluation, each episode's termination cause, aligned with `evaluations_length`.
        self.evaluations_termination_causes: list[list[str]] = []
        self._current_eval_termination_causes: list[str] = []
        # Per evaluation, each sagittal pair's left-minus-right commanded joint offset in degrees
        # (`leg_dc_pairs`, `leg_dc_degrees`); `{}` for a model with no pair.
        self.evaluations_leg_dc: list[dict[str, float]] = []
        self._current_eval_foot_pad: dict[str, list[float]] = {}
        # Per sub-env: (unwrapped env, its model, Morphology, PadLayout), and
        # during an evaluation the attached recorders and the episodes each
        # sub-env still owes SB3's evaluate_policy.
        self._foot_setup: list[tuple[Any, Any, Any, Any]] = []
        # Per sub-env, its RailLayout (the box-soled feet), beside `_foot_setup`.
        self._rail_layouts: list[Any] = []
        self._pad_recorders: dict[int, Any] = {}
        self._pad_quota: dict[int, int] = {}
        self._pad_failed = False
        self.evaluations_forward_velocities: list[list[float]] = []
        self.evaluations_unsupported_duties: list[list[float]] = []
        # Not a gate criterion -- captured so a run GENERATES the data needed
        # to set one honestly. The three stance shares sum to 1, so a falling
        # unsupported duty says nothing about whether the feet are being
        # planted; that time can land in single support instead. Bounding
        # bilateral duty is the obvious fix, but the only reference point
        # available today is the statue's 0.998, and the statue never shifts
        # weight. A competent active policy legitimately spends some time in
        # single support, and how much is unmeasured. Recording it every
        # evaluation is what turns that into evidence instead of a guess.
        self.evaluations_bilateral_duties: list[list[float]] = []
        # Action statistics for the DETERMINISTIC policy, which is what the
        # gate scores. `diagnostics.npz` records actions from training
        # rollouts, so every number in it carries exploration noise; the
        # quantities that matter here -- how far the commanded target sits from
        # the home control (action = 0), and how much the policy shakes about it -- are
        # properties of the mean action and are unrecoverable from a noisy
        # sample. Issue #489 had to invert them out of per-episode reward
        # totals under a narrowband assumption. These measure them.
        #
        # Per actuator, not per group: grouping is an analysis decision, and
        # resolving joint names through the VecEnv wrappers is exactly the kind
        # of best-effort lookup that silently returns nothing. Recording the
        # full vector keeps the grouping offline and impossible to get wrong
        # here.
        self.evaluations_action_dc: list[list[float]] = []
        self.evaluations_action_ac_rms: list[list[float]] = []
        self.evaluations_action_delta: list[float] = []
        self.evaluations_action_jerk: list[float] = []
        # Per-episode reward decomposition. Only `mean_reward` was kept, so
        # comparing two runs meant comparing their final checkpoints and
        # nothing in between -- there was no way to ask WHEN a policy adopted
        # the pose it ended up with.
        self.evaluations_reward_terms: list[dict[str, float]] = []
        self._episode_forward_sums: dict[int, float] = {}
        self._episode_forward_counts: dict[int, int] = {}
        self._current_eval_forward_velocities: list[float] = []
        self._episode_steps: dict[int, int] = {}
        self._episode_unsupported: dict[int, int] = {}
        self._episode_bilateral: dict[int, int] = {}
        self._episode_measured: dict[int, int] = {}
        self._current_eval_unsupported_duties: list[float] = []
        self._current_eval_bilateral_duties: list[float] = []
        # Action accumulators, over the post-settle window so the reset
        # transient does not inflate the AC term -- the same window the duty
        # uses, and for the same reason.
        self._action_sum: Any = None
        self._action_sq_sum: Any = None
        self._action_delta_sum = 0.0
        self._action_jerk_sum = 0.0
        self._action_count = 0
        # Counted separately from the samples: a difference needs two and a
        # second difference three, and episode boundaries reset the history.
        # Dividing all three by the sample count biases their ratio, and that
        # ratio is what becomes a frequency.
        self._action_delta_count = 0
        self._action_jerk_count = 0
        self._prev_action: dict[int, Any] = {}
        self._prev_prev_action: dict[int, Any] = {}
        # Reward terms accumulate over the WHOLE episode: they are episode
        # returns, and truncating them to the post-settle window would make
        # them incomparable with `stance_gate_report.py`'s breakdown.
        self._reward_term_sums: dict[str, float] = {}
        self._reward_term_episodes = 0

    def _reset_forward_velocity_capture(self) -> None:
        self._episode_forward_sums.clear()
        self._episode_forward_counts.clear()
        self._current_eval_forward_velocities = []
        self._episode_steps.clear()
        self._episode_unsupported.clear()
        self._episode_bilateral.clear()
        self._episode_measured.clear()
        self._current_eval_unsupported_duties = []
        self._current_eval_bilateral_duties = []
        self._current_eval_termination_causes = []
        self._action_sum = None
        self._action_sq_sum = None
        self._action_delta_sum = 0.0
        self._action_jerk_sum = 0.0
        self._action_count = 0
        self._action_delta_count = 0
        self._action_jerk_count = 0
        self._prev_action.clear()
        self._prev_prev_action.clear()
        self._reward_term_sums.clear()
        self._reward_term_episodes = 0

    def _capture_action(self, locals_: dict[str, Any], env_index: int, after_settle: bool) -> None:
        """Accumulate deterministic-policy action statistics for one step.

        Wrapped defensively because ``actions`` is a local of SB3's
        ``evaluate_policy``, not part of any documented callback contract: a
        version that renames it must cost this diagnostic, not the run.
        """
        try:
            actions = locals_.get("actions")
            if actions is None:
                return
            action = np.asarray(actions)
            action = action[env_index] if action.ndim > 1 else action
            action = np.asarray(action, dtype=np.float64).ravel()
            if not action.size:
                return
            prev = self._prev_action.get(env_index)
            prev_prev = self._prev_prev_action.get(env_index)
            if after_settle:
                if self._action_sum is None or self._action_sum.shape != action.shape:
                    self._action_sum = np.zeros_like(action)
                    self._action_sq_sum = np.zeros_like(action)
                self._action_sum += action
                self._action_sq_sum += action * action
                self._action_count += 1
                # Both differences are summed over actuators, matching the
                # `Sum` in `reward_action_smoothness` / `reward_action_jerk`
                # so the numbers invert straight back through those formulas.
                if prev is not None and prev.shape == action.shape:
                    self._action_delta_sum += float(np.sum((action - prev) ** 2))
                    self._action_delta_count += 1
                    if prev_prev is not None and prev_prev.shape == action.shape:
                        self._action_jerk_sum += float(np.sum((action - 2.0 * prev + prev_prev) ** 2))
                        self._action_jerk_count += 1
            self._prev_prev_action[env_index] = prev
            self._prev_action[env_index] = action
        except Exception:  # noqa: BLE001 - a diagnostic must not sink a run
            return

    # ------------------------------------------------------------------ pad telemetry
    def _warn_foot_telemetry(self, message: str, *, exc_info: bool = False) -> None:
        """Warn once per callback: the same cause fails identically on every evaluation of a run."""
        if not self.__dict__.get("_warned_foot_telemetry", False):
            self._warned_foot_telemetry = True
            logger.warning(
                "Pad telemetry: %s; this run's gate_progress.npz carries no pad, roll-rail or leg DC fields.",
                message,
                exc_info=exc_info,
            )

    def _resolve_foot_telemetry(self) -> None:
        """Describe the evaluation env's model, once: its pad and roll-rail layouts and its sagittal leg DC pairs.

        Nothing is resolved without a ``model_id``.  With one, an env that
        cannot be described -- a SubprocVecEnv, an env with no substep hook,
        a model the morphology refuses -- leaves both empty for the whole
        run and warns once, rather than recording a field on some
        evaluations and not others.
        """
        if self.__dict__.get("_foot_resolved", False):
            return
        self._foot_resolved = True
        self._foot_setup = []
        model_id = self.model_id
        if model_id is None:
            return
        envs = _in_process_envs(self.__dict__.get("eval_env"))
        if envs is None:
            self._warn_foot_telemetry("the evaluation env is not an in-process VecEnv, so it cannot be recorded")
            return
        try:
            from .gait.morphology import Morphology
            from .gait.pad_telemetry import PadLayout, RailLayout

            setup = []
            rails = []
            for env in envs:
                unwrapped = getattr(env, "unwrapped", env)
                morphology = Morphology.from_env(unwrapped, model_id)
                setup.append((unwrapped, unwrapped.model, morphology, PadLayout.from_morphology(morphology)))
                rails.append(RailLayout.from_morphology(morphology))
            pairs = sagittal_leg_dc_pairs(setup[0][0], setup[0][2])
        except Exception:  # noqa: BLE001 - a diagnostic must not sink a run
            self._warn_foot_telemetry(f"cannot describe the {model_id} evaluation env", exc_info=True)
            return
        self._foot_setup = setup
        self._rail_layouts = rails
        self.foot_pad_keys = setup[0][3].episode_keys()
        self.foot_rail_keys = rails[0].episode_keys()
        self.leg_dc_pairs = pairs

    def _foot_keys(self) -> tuple[str, ...]:
        """Every per-episode foot key: the pad keys, then the roll-rail keys."""
        return tuple(self.foot_pad_keys or ()) + tuple(self.foot_rail_keys or ())

    def _begin_pad_capture(self) -> None:
        """Attach a recorder to each evaluation sub-env, for this evaluation only (a no-op without pad feet)."""
        self._resolve_foot_telemetry()
        keys = self._foot_keys()
        self._current_eval_foot_pad = {key: [] for key in keys}
        self._pad_recorders = {}
        self._pad_quota = {}
        self._pad_failed = False
        model_id = self.model_id
        if not keys or model_id is None:
            return
        try:
            from .gait.morphology import Morphology
            from .gait.pad_telemetry import PadLayout, RailLayout
            from .gait.recorder import SubstepContactRecorder

            setup = self._foot_setup
            rails = self.__dict__.setdefault("_rail_layouts", [])
            for index, (env, model, morphology, layout) in enumerate(setup):
                if env.model is not model:
                    # An env that swapped its model (a terrain rebuild) gets a fresh description.
                    morphology = Morphology.from_env(env, model_id)
                    setup[index] = (env, env.model, morphology, PadLayout.from_morphology(morphology))
                    if index < len(rails):
                        rails[index] = RailLayout.from_morphology(morphology)
                # SB3's evaluate_policy splits the panel this way and stops calling back for a
                # sub-env that has finished its share, so its recorder is detached right then.
                self._pad_quota[index] = (int(self.n_eval_episodes) + index) // len(setup)
                if self._pad_quota[index] > 0:
                    self._pad_recorders[index] = SubstepContactRecorder(env, setup[index][2]).attach()
        except Exception:  # noqa: BLE001 - a diagnostic must not sink a run
            self._fail_pad_capture()

    def _detach_pad_recorder(self, env_index: int) -> None:
        recorder = self._pad_recorders.pop(env_index, None)
        if recorder is not None:
            try:
                recorder.detach()
            except Exception:  # noqa: BLE001 - a diagnostic must not sink a run
                logger.warning(
                    "Pad telemetry: could not detach the recorder from eval env %d.", env_index, exc_info=True
                )

    def _fail_pad_capture(self) -> None:
        """Give up this evaluation's pad telemetry (it publishes NaN) and take every recorder off its env."""
        self._pad_failed = True
        for env_index in list(self._pad_recorders):
            self._detach_pad_recorder(env_index)
        if not self.__dict__.get("_warned_pad_capture", False):
            self._warned_pad_capture = True
            logger.warning("Pad telemetry: an evaluation could not be recorded; it reads NaN.", exc_info=True)

    def _end_pad_capture(self) -> None:
        for env_index in list(self.__dict__.get("_pad_recorders", {})):
            self._detach_pad_recorder(env_index)

    def _capture_pad(self, locals_: dict[str, Any], env_index: int, done: bool) -> None:
        """Close this step on the sub-env's recorder; at an episode end, reduce it into one value per foot key."""
        keys = self._foot_keys()
        if not keys:
            return
        values: dict[str, float] = {}
        recorder = self._pad_recorders.get(env_index)
        if recorder is not None and not self._pad_failed:
            try:
                from .gait.pad_telemetry import episode_pad_telemetry

                actions = locals_.get("actions")
                action = None
                if actions is not None:
                    batch = np.asarray(actions)
                    action = batch[env_index] if batch.ndim > 1 else batch
                reward = locals_.get("reward")
                recorder.end_step(action, float("nan") if reward is None else float(reward))
                if done:
                    trace = recorder.end_episode()
                    values = episode_pad_telemetry(
                        trace, self._foot_setup[env_index][3], settle_steps=self.settle_steps
                    )
                    rails = self.__dict__.get("_rail_layouts") or []
                    if env_index < len(rails):
                        from .gait.pad_telemetry import episode_rail_telemetry

                        values = {
                            **values,
                            **episode_rail_telemetry(trace, rails[env_index], settle_steps=self.settle_steps),
                        }
            except Exception:  # noqa: BLE001 - a diagnostic must not sink a run
                self._fail_pad_capture()
                values = {}
        if done:
            # One value per key per episode, NaN when unmeasured, so every list stays aligned with
            # the episode lengths SB3 records -- the duties' contract.
            for key in keys:
                self._current_eval_foot_pad.setdefault(key, []).append(values.get(key, float("nan")))
            remaining = self._pad_quota.get(env_index, 0) - 1
            self._pad_quota[env_index] = remaining
            if remaining <= 0:
                self._detach_pad_recorder(env_index)

    def _publish_pad_telemetry(self) -> None:
        """Append this evaluation's per-episode pad values (all NaN when its capture failed)."""
        current = self.__dict__.get("_current_eval_foot_pad", {})
        failed = self.__dict__.get("_pad_failed", False)
        self.__dict__.setdefault("evaluations_foot_pad", []).append(
            {key: [float("nan")] * len(values) if failed else list(values) for key, values in current.items()}
        )

    def _publish_leg_dc(self) -> None:
        """Append this evaluation's sagittal leg DC in degrees, from the action DC just published.

        Mapped through the first evaluation sub-env's action map (every
        sub-env runs the same model); NaN for every key when the DC is
        unmeasured or the map cannot be read.
        """
        pairs = tuple(self.__dict__.get("leg_dc_pairs", ()) or ())
        values: dict[str, float] = {pair.key: float("nan") for pair in pairs}
        setup = self.__dict__.get("_foot_setup") or []
        series = self.__dict__.get("evaluations_action_dc") or []
        if pairs and setup and series and series[-1]:
            try:
                values = leg_dc_degrees(setup[0][0], pairs, series[-1])
            except Exception:  # noqa: BLE001 - a diagnostic must not sink a run
                if not self.__dict__.get("_warned_leg_dc", False):
                    self._warned_leg_dc = True
                    logger.warning(
                        "Leg DC: the action DC could not be mapped to joint targets; it reads NaN.", exc_info=True
                    )
        self.__dict__.setdefault("evaluations_leg_dc", []).append(values)

    @staticmethod
    def termination_cause(info: Mapping[str, Any]) -> str:
        """The cause of the episode whose last ``info`` this is: its ``termination_reason``, or how it ended.

        A VecEnv marks a time-limit end ``TimeLimit.truncated``; a terminated
        episode carries the env's ``termination_reason`` (``"terminated"``
        when the env gave none), the stance report's reading.
        """
        if info.get("TimeLimit.truncated"):
            return "truncated"
        reason = info.get("termination_reason")
        return str(reason) if reason else "terminated"

    def _log_success_callback(self, locals_: dict[str, Any], globals_: dict[str, Any]) -> None:
        """Capture evaluation velocity and duty, conditionally delegate success."""

        info = locals_["info"]
        env_index = int(locals_.get("i", 0))
        self._capture_pad(locals_, env_index, bool(locals_["done"]))
        for key, value in info.items():
            if not key.startswith("reward_"):
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(number):
                self._reward_term_sums[key] = self._reward_term_sums.get(key, 0.0) + number
        forward_vel = info.get("forward_vel")
        if forward_vel is not None:
            value = float(forward_vel)
            if math.isfinite(value):
                self._episode_forward_sums[env_index] = self._episode_forward_sums.get(env_index, 0.0) + value
                self._episode_forward_counts[env_index] = self._episode_forward_counts.get(env_index, 0) + 1

        # Duty is accumulated only after the settling window, because a policy
        # that corrects reset randomisation may legitimately move a foot doing
        # it -- and settling is the capability stage 1a exists to certify.
        step_index = self._episode_steps.get(env_index, 0)
        self._episode_steps[env_index] = step_index + 1
        self._capture_action(locals_, env_index, after_settle=step_index >= self.settle_steps)
        if step_index >= self.settle_steps:
            stance = derive_stance_info(info)
            if stance:
                self._episode_measured[env_index] = self._episode_measured.get(env_index, 0) + 1
                self._episode_unsupported[env_index] = self._episode_unsupported.get(env_index, 0) + int(
                    stance["unsupported_duty"]
                )
                self._episode_bilateral[env_index] = self._episode_bilateral.get(env_index, 0) + int(
                    stance["bilateral_support_duty"]
                )

        if locals_["done"]:
            count = self._episode_forward_counts.pop(env_index, 0)
            total = self._episode_forward_sums.pop(env_index, 0.0)
            self._current_eval_forward_velocities.append(total / count if count else float("nan"))

            measured = self._episode_measured.pop(env_index, 0)
            unsupported = self._episode_unsupported.pop(env_index, 0)
            bilateral = self._episode_bilateral.pop(env_index, 0)
            self._episode_steps.pop(env_index, None)
            # NaN for an episode with no measurable tail (shorter than the
            # settling window, or an env that reports no foot contacts). The
            # panel builder drops those rather than scoring them as zero.
            self._current_eval_unsupported_duties.append(unsupported / measured if measured else float("nan"))
            self._current_eval_bilateral_duties.append(bilateral / measured if measured else float("nan"))
            self.__dict__.setdefault("_current_eval_termination_causes", []).append(self.termination_cause(info))
            self._reward_term_episodes += 1
            self._prev_action.pop(env_index, None)
            self._prev_prev_action.pop(env_index, None)

        if self.success_applicable:
            super()._log_success_callback(locals_, globals_)

    def seed_prior_history(self, n_evals: int) -> None:
        """Pad the parallel per-eval capture series for *n_evals* seeded evals.

        On a same-stage resume, ``curriculum.seed_resume_eval_state`` restores
        SB3's ``evaluations_*`` lists from the stage's published
        ``evaluations.npz`` — but the prior session's per-eval captures
        (duties, velocities, action statistics, reward terms) are not in that
        file.  ``StageGatePlateauCallback`` resolves these series by the
        ABSOLUTE index into ``evaluations_results``; without padding, every
        post-resume evaluation's index points past the end of each series,
        the stance panel resolves to ``None``, and the gate telemetry
        (``gate_progress.npz`` appends, ``diagnostics/eval_*`` scalars)
        silently stops for the rest of the session.  The placeholders are the
        exact values ``_on_step``/``_publish_action_statistics`` append for an
        unmeasured evaluation, which every consumer already treats as
        "no data for this eval".  The seeded evaluations' rows of
        ``gate_progress.npz``, their per-actuator matrices among them, come
        from the prior session's file
        (``StageGatePlateauCallback.seed_prior_gate_progress``), not from
        these placeholders.
        """
        for _ in range(n_evals):
            self.evaluations_forward_velocities.append([])
            self.evaluations_unsupported_duties.append([])
            self.evaluations_bilateral_duties.append([])
            self.evaluations_action_dc.append([])
            self.evaluations_action_ac_rms.append([])
            self.evaluations_action_delta.append(float("nan"))
            self.evaluations_action_jerk.append(float("nan"))
            self.evaluations_reward_terms.append({})
            self.__dict__.setdefault("evaluations_foot_pad", []).append({})
            self.__dict__.setdefault("evaluations_leg_dc", []).append({})
            self.__dict__.setdefault("evaluations_termination_causes", []).append([])

    def _publish_action_statistics(self) -> None:
        """Reduce one evaluation's accumulators into the published series.

        The DC/AC split is the whole point. ``mean(a)`` per actuator is the
        static target the policy commands -- the distance from the home
        control, which under ``home-keyframe-residual/v1`` is exactly
        ``action = 0`` (the home keyframe's control targets, not necessarily
        its joint pose: the trex ankles carry an authored preload).
        ``sqrt(mean(a^2) - mean(a)^2)`` is what it does *around* that pose. The
        two answer different questions and a pooled standard deviation over
        actuators and time (which is what ``diagnostics.action_std`` computes)
        cannot separate them.
        """
        count = self._action_count
        if count and self._action_sum is not None:
            mean = self._action_sum / count
            var = np.maximum(self._action_sq_sum / count - mean * mean, 0.0)
            self.evaluations_action_dc.append([float(v) for v in mean])
            self.evaluations_action_ac_rms.append([float(v) for v in np.sqrt(var)])
            # Per differencing opportunity, so they invert directly through
            # the reward formulas without an edge-effect bias.
            self.evaluations_action_delta.append(
                self._action_delta_sum / self._action_delta_count if self._action_delta_count else float("nan")
            )
            self.evaluations_action_jerk.append(
                self._action_jerk_sum / self._action_jerk_count if self._action_jerk_count else float("nan")
            )
        else:
            self.evaluations_action_dc.append([])
            self.evaluations_action_ac_rms.append([])
            self.evaluations_action_delta.append(float("nan"))
            self.evaluations_action_jerk.append(float("nan"))
        episodes = self._reward_term_episodes
        self.evaluations_reward_terms.append(
            {key: value / episodes for key, value in self._reward_term_sums.items()} if episodes else {}
        )

    def _on_step(self) -> bool:
        evaluation_due = self.eval_freq > 0 and self.n_calls % self.eval_freq == 0
        if evaluation_due:
            self._reset_forward_velocity_capture()
            self._begin_pad_capture()

        try:
            continue_training = super()._on_step()
        finally:
            # Off the env before any other callback steps it (the curriculum's own panels), and
            # whether or not the evaluation raised.
            if evaluation_due:
                self._end_pad_capture()

        if evaluation_due:
            self.evaluations_forward_velocities.append(list(self._current_eval_forward_velocities))
            self.evaluations_unsupported_duties.append(list(self._current_eval_unsupported_duties))
            self.evaluations_bilateral_duties.append(list(self._current_eval_bilateral_duties))
            self.__dict__.setdefault("evaluations_termination_causes", []).append(
                list(self.__dict__.get("_current_eval_termination_causes", []))
            )
            self._publish_action_statistics()
            self._publish_pad_telemetry()
            self._publish_leg_dc()
            if not self.success_applicable and self.verbose >= 1:
                print(f"Success rate: N/A (not an active Stage {self.stage} gate)")

        return bool(continue_training)


def _episode_max_over_feet(series: "list[Any]") -> list[float] | None:
    """Per episode, the maximum over the feet of their finite values (NaN where none is); ``None`` unless aligned.

    The gate's ``max_sole_cop_fore_aft`` is the maximum over the box feet it
    measured, so a foot with no value in an episode is skipped, not read as 0.
    """
    if not series or any(values is None for values in series) or len({len(values) for values in series}) != 1:
        return None
    out = []
    for episode in zip(*series):
        finite = [float(value) for value in episode if math.isfinite(float(value))]
        out.append(max(finite) if finite else float("nan"))
    return out


def _declared_bar(curriculum_kwargs: Any, key: str) -> float | None:
    """The stage's declared bar for *key* as a finite float, or ``None`` when it declares none."""
    value = curriculum_kwargs.get(key) if isinstance(curriculum_kwargs, Mapping) else None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return None
    return float(value)


@dataclass(frozen=True)
class _GateMetric:
    key: str
    label: str
    threshold: float
    value: float | None
    sample_count: int
    percent: bool = False
    #: Samples this metric needs before it can be followed.  ``None`` means
    #: the stage's ``min_eval_episodes``.  The stance duty overrides it: the
    #: gate certifies on ``ceil(min_eval_episodes x min_full_horizon_fraction)``
    #: duty episodes, so requiring the full panel size here made a 39-of-40
    #: panel report ``eval_gate_met = 0`` while the curriculum advanced on it.
    min_samples: int | None = None
    #: True when the threshold is an upper bound the metric must stay under
    #: (``stance_quality/v1``'s duty ceilings) rather than a floor it must
    #: clear. Without the distinction a ceiling read as a floor inverts:
    #: a policy chattering at duty 0.3 against a 0.02 ceiling would be
    #: reported as comfortably passing.
    ceiling: bool = False

    def is_failing(self) -> bool:
        if self.value is None:
            return False
        return self.value > self.threshold if self.ceiling else self.value < self.threshold


class StageGatePlateauCallback(_BaseCallback):  # type: ignore[misc]
    """Warn once when the currently blocking evaluation gate stops moving.

    A plateau is a small range over the latest ``plateau_window`` deterministic
    evaluations, normalized by that metric's configured gate.  The callback
    follows the most behaviorally specific failing gate: task success, forward
    velocity, episode length, then reward.  It disarms after meaningful
    movement or a gate pass and can warn again only after a later plateau.

    This callback is diagnostic only.  It never stops training and is separate
    from both curriculum advancement and collapse early stopping.
    """

    # Ordered most behaviourally specific first. The stance metrics sit above
    # episode length and reward because on a stance-gated stage they ARE the
    # gate: without them the callback would follow mean_reward and hand out
    # reward-tuning guidance for a stage whose blocking criterion is foot
    # contact, which is worse than silence.
    _METRIC_PRIORITY = (
        "mean_success_rate",
        "mean_forward_vel",
        "mean_unsupported_duty",
        "full_horizon_fraction",
        "mean_episode_length",
        "mean_reward",
    )

    _GUIDANCE = {
        "mean_success_rate": (
            "Inspect target reachability, contact/proximity signals, and the Stage 3 reward ramp before tuning the optimizer."
        ),
        "mean_forward_vel": (
            "Inspect gait stability, actuator saturation, and locomotion reward balance before tuning the optimizer."
        ),
        "mean_episode_length": (
            "Inspect termination reasons, posture, and balance stability before tuning the optimizer."
        ),
        "mean_unsupported_duty": (
            "Inspect foot contact forces, the action smoothness and jerk penalties, and exploration noise: "
            "the policy is spending too much time with neither foot loaded."
        ),
        "full_horizon_fraction": (
            "Inspect termination reasons and the reset distribution before tuning the optimizer: "
            "episodes are ending early rather than degrading in quality."
        ),
        "mean_reward": (
            "Inspect reward components and policy diagnostics to identify which behavior has stopped improving."
        ),
    }

    #: Declared at class scope so an instance built without ``__init__``
    #: still has them. The suite constructs this callback with
    #: ``object.__new__`` to exercise the plateau logic without
    #: stable-baselines3, and a new instance attribute only ``__init__`` set
    #: would turn every one of those tests into an AttributeError.
    _gate_progress_dir: "Path | None" = None
    _gate_progress_timesteps: list[int] = []
    _gate_progress: dict[str, list[float]] = {}
    _gate_progress_seeded = 0
    _warned_gate_progress_unwritable = False

    #: ``gate_progress.npz``'s per-actuator matrices, each with the capture
    #: callback's series its rows come from.
    _GATE_PROGRESS_MATRICES = (
        ("action_dc_per_actuator", "evaluations_action_dc"),
        ("action_ac_rms_per_actuator", "evaluations_action_ac_rms"),
    )

    def __init__(
        self,
        eval_callback: StageAwareEvalCallback,
        *,
        stage: "int | str",
        curriculum_kwargs: dict[str, Any],
        plateau_window: int = 5,
        min_relative_variation: float = 0.05,
        horizon: int = 1000,
        gate_progress_dir: "str | Path | None" = None,
        control_dt: float = 0.01,
        verbose: int = 0,
    ) -> None:
        if not _SB3_AVAILABLE:
            raise ImportError(
                "stable-baselines3 is required for StageGatePlateauCallback. "
                "Install with: pip install stable-baselines3"
            )
        if plateau_window < 2:
            raise ValueError("plateau_window must be at least 2 evaluations")
        if min_relative_variation < 0.0:
            raise ValueError("min_relative_variation must be non-negative")
        super().__init__(verbose)
        self.eval_callback = eval_callback
        self.horizon = int(horizon)
        self.stage = stage
        self.curriculum_kwargs = dict(curriculum_kwargs)
        self.plateau_window = plateau_window
        self.min_relative_variation = min_relative_variation
        self._last_seen_n_evals = 0
        self._histories: dict[str, list[float]] = {key: [] for key in self._METRIC_PRIORITY}
        self._plateau_active = False
        self._plateau_metric: str | None = None
        # Seconds per control step: `timestep * frame_skip`, needed to turn the
        # jerk/delta ratio into a frequency rather than a bare ratio.
        self.control_dt = float(control_dt) if control_dt and control_dt > 0 else 0.01
        # Per-evaluation gate criteria, persisted to `gate_progress.npz`.
        self._gate_progress_dir = None if gate_progress_dir is None else Path(gate_progress_dir)
        self._gate_progress_timesteps: list[int] = []
        self._gate_progress: dict[str, list[float]] = {}
        # Per matrix, one per-actuator row per recorded evaluation (`[]`: unmeasured).
        self._gate_progress_rows: dict[str, list[list[float]]] = {}
        # Rows a same-stage resume carried over from the prior session's file.
        self._gate_progress_seeded = 0
        self._warned_gate_progress_unwritable = False

    @staticmethod
    def _finite_mean(values: Any) -> float | None:
        array = np.asarray(values, dtype=float).reshape(-1)
        finite = array[np.isfinite(array)]
        if finite.size == 0:
            return None
        return float(np.mean(finite))

    @staticmethod
    def _finite_count(values: Any) -> int:
        array = np.asarray(values, dtype=float).reshape(-1)
        return int(np.count_nonzero(np.isfinite(array)))

    def _series(self, attribute: str, index: int) -> Any | None:
        """One evaluation's per-episode series from the capture callback."""
        history = getattr(self.eval_callback, attribute, None) or []
        return history[index] if index < len(history) else None

    def _stance_panel(self, index: int) -> Any | None:
        """This evaluation's stance panel, or ``None`` if it cannot be built.

        Built with the same :mod:`~environments.shared.curriculum.stance_gate`
        reduction the curriculum runs, so what this callback plots is what the
        gate decides. Reducing the duties here independently is how the
        TensorBoard curve came to disagree with the gate by 8x: the scalar
        averaged every episode while the gate averages the full-horizon ones
        only, and on a panel with 5 early failures at duty 0.60 among 35 clean
        episodes at 0.01 that reads 0.084 against a 0.02 ceiling the policy
        was actually clearing.
        """
        from .curriculum.stance_gate import stance_panel_from_episode_duties

        duties = self._series("evaluations_unsupported_duties", index)
        lengths = self._series("evaluations_length", index)
        rewards = self._series("evaluations_results", index)
        if duties is None or lengths is None or rewards is None:
            return None
        if not len(duties) or len(duties) != len(lengths) or len(duties) != len(rewards):
            # Positional alignment is the whole contract; a mismatch pairs each
            # duty with the wrong episode's length.
            return None
        try:
            return stance_panel_from_episode_duties(
                episode_lengths=[float(v) for v in lengths],
                episode_duties=[float(v) for v in duties],
                episode_rewards=[float(v) for v in rewards],
                horizon=self.horizon,
            )
        except (TypeError, ValueError):
            return None

    def _full_horizon_mean(self, attribute: str, index: int) -> float | None:
        """Mean of a per-episode series over full-horizon episodes only.

        The set the gate's duty is measured over.  Averaging a stance share
        over a different set than the gated one is what makes the three
        shares stop summing to 1 and the pair stop being readable together.
        """
        return self._full_horizon_mean_of(self._series(attribute, index), self._series("evaluations_length", index))

    def _full_horizon_mean_of(self, series: Any, lengths: Any) -> float | None:
        """:meth:`_full_horizon_mean` of a per-episode *series* aligned with the episode *lengths*."""
        if series is None or lengths is None or len(series) != len(lengths):
            return None
        kept = [
            float(value)
            for value, length in zip(series, lengths)
            if float(length) >= self.horizon and math.isfinite(float(value))
        ]
        return float(np.mean(kept)) if kept else None

    def _foot_telemetry_scalars(self, index: int) -> dict[str, float]:
        """This evaluation's pad telemetry and sagittal leg DC, one float per key (NaN: unmeasured).

        The keys are fixed by the capture callback's first evaluation
        (``foot_pad_keys``, ``leg_dc_pairs``) and by the stage's declared
        ``max_sole_cop_fore_aft`` bar, so every evaluation supplies the same
        set and none is dropped from ``gate_progress.npz`` as a short series.

        * Each pad key is the mean of its per-episode values over the
          full-horizon episodes -- the set the duty and the bilateral share
          are averaged over -- so an early fall's transient does not read as
          a posture.
        * ``max_sole_cop_fore_aft_mean`` (a model with a box pad) is the mean
          over the same episodes of each episode's maximum over the feet of
          ``<label>_sole_cop_fore_aft_gate``: the episode's own
          ``max_sole_cop_fore_aft``, the number the gate's rail reads.  The
          maximum over the feet of the per-foot means is only a lower bound
          of it, an equality only when one foot is the worse one in every
          episode.  With the stage's bar declared,
          ``max_sole_cop_fore_aft_over_bar_fraction`` is the fraction of
          those episodes whose value exceeds the bar (``value > bar``, the
          gate's comparison): the rail's own criterion on this panel.  An
          episode measured on no foot is left out of both.
        * ``max_foot_cop_fore_aft_mean`` (a model whose feet declare a
          support span: the T. rex) is the same for the whole-foot centre of
          pressure, from ``<label>_foot_cop_fore_aft_gate``: the mean of each
          episode's own ``max_foot_cop_fore_aft``.  Its over-bar fraction,
          ``max_foot_cop_fore_aft_over_bar_fraction``, is written only where
          the stage declares that bar, and no stage does: the key is
          report-only.
        * Each roll-rail key (``foot_rail_keys``: ``<label>_sole_tilt_deg``
          and ``<label>_sole_corner_lift_m`` per box-soled foot) is likewise
          the full-horizon mean of its per-episode values, and
          ``max_sole_tilt_deg_mean`` and ``max_sole_corner_lift_m_mean`` are
          the means of each episode's own maximum over the feet, the numbers
          the gate's two roll rails read.  Their over-bar fractions,
          ``max_sole_tilt_deg_over_bar_fraction`` and
          ``max_sole_corner_lift_m_over_bar_fraction``, are written where the
          stage declares the bar (the T. rex and the anatomical
          Compsognathus).
        * Each ``leg_dc_<left>_minus_<right>_<joint>_deg`` is the capture
          callback's :attr:`~StageAwareEvalCallback.evaluations_leg_dc`
          (:func:`leg_dc_degrees`).
        """
        from .gait.pad_telemetry import FOOT_GATE_COP_FIELD, GATE_COP_FIELD, RAIL_EPISODE_FIELDS

        source = getattr(self, "eval_callback", None)
        nan = float("nan")
        out: dict[str, float] = {}
        pads = self._eval_series("evaluations_foot_pad", index) or {}
        lengths = self._series("evaluations_length", index)
        pad_keys = tuple(getattr(source, "foot_pad_keys", ()) or ()) + tuple(
            getattr(source, "foot_rail_keys", ()) or ()
        )
        for key in pad_keys:
            value = self._full_horizon_mean_of(pads.get(key), lengths)
            out[key] = nan if value is None else value
        # Each rail's own per-episode number: the pad CoP's, the whole foot's, and the two roll rails'.
        for field, metric in (
            (GATE_COP_FIELD, "max_sole_cop_fore_aft"),
            (FOOT_GATE_COP_FIELD, "max_foot_cop_fore_aft"),
            *RAIL_EPISODE_FIELDS,
        ):
            gate_keys = tuple(key for key in pad_keys if key.endswith(f"_{field}"))
            if not gate_keys:
                continue
            per_episode = _episode_max_over_feet([pads.get(key) for key in gate_keys])
            mean = self._full_horizon_mean_of(per_episode, lengths)
            out[f"{metric}_mean"] = nan if mean is None else mean
            bar = _declared_bar(getattr(self, "curriculum_kwargs", None), metric)
            if bar is not None:
                above = (
                    None if per_episode is None else [float(v > bar) if math.isfinite(v) else nan for v in per_episode]
                )
                fraction = self._full_horizon_mean_of(above, lengths)
                out[f"{metric}_over_bar_fraction"] = nan if fraction is None else fraction
        leg_dc = self._eval_series("evaluations_leg_dc", index) or {}
        for pair in tuple(getattr(source, "leg_dc_pairs", ()) or ()):
            out[pair.key] = float(leg_dc.get(pair.key, nan))
        return out

    def _metrics_for_evaluation(self, index: int, panel: Any | None = None) -> list[_GateMetric]:
        rewards = self.eval_callback.evaluations_results
        lengths = self.eval_callback.evaluations_length
        velocities = self.eval_callback.evaluations_forward_velocities
        successes = self.eval_callback.evaluations_successes

        samples: dict[str, Any | None] = {
            "mean_reward": rewards[index] if index < len(rewards) else None,
            "mean_episode_length": lengths[index] if index < len(lengths) else None,
            "mean_forward_vel": velocities[index] if index < len(velocities) else None,
            "mean_success_rate": successes[index] if index < len(successes) else None,
        }
        values = {key: self._finite_mean(sample) if sample is not None else None for key, sample in samples.items()}
        counts = {key: self._finite_count(sample) if sample is not None else 0 for key, sample in samples.items()}

        # Duty comes from the panel, not from a second reduction, so the
        # metric the plateau follows is the number the gate compares.
        if panel is not None and panel.n_duty_episodes:
            values["mean_unsupported_duty"] = panel.mean_unsupported_duty
            counts["mean_unsupported_duty"] = panel.n_duty_episodes
        else:
            values["mean_unsupported_duty"] = None
            counts["mean_unsupported_duty"] = 0

        # Derived, not sampled: the fraction of episodes reaching the horizon.
        # It comes from the same per-episode lengths the length gate uses, so
        # no extra capture is needed.
        episode_lengths = samples["mean_episode_length"]
        if episode_lengths is not None and len(episode_lengths):
            finite = [float(v) for v in episode_lengths if math.isfinite(float(v))]
            horizon_fraction = sum(1.0 for v in finite if v >= self.horizon) / len(finite) if finite else None
            horizon_count = len(finite)
        else:
            horizon_fraction, horizon_count = None, 0

        # task_success/v1 gates on a binomial LOWER BOUND, not the raw rate;
        # the plateau follows the raw success rate against the LCB bar (the
        # bound is monotone in the rate at fixed n, so the metric that has
        # to move is the same one) and the label says which bar it is.
        task_success = self.curriculum_kwargs.get("gate_kind") == TASK_SUCCESS_GATE_KIND
        configured = (
            _GateMetric(
                "mean_success_rate",
                "success rate (LCB95 bar)" if task_success else "success rate",
                float(
                    self.curriculum_kwargs.get("min_success_lcb", self.curriculum_kwargs.get("min_success_rate", 0.0))
                )
                if task_success
                else float(self.curriculum_kwargs.get("min_success_rate", 0.0)),
                values["mean_success_rate"],
                counts["mean_success_rate"],
                percent=True,
            ),
            _GateMetric(
                "mean_forward_vel",
                "forward velocity",
                float(self.curriculum_kwargs.get("min_avg_forward_vel", 0.0)),
                values["mean_forward_vel"],
                counts["mean_forward_vel"],
            ),
            _GateMetric(
                "mean_unsupported_duty",
                "unsupported duty",
                # inf when absent, so the isfinite filter drops it for stages
                # that do not gate on stance. A ceiling cannot use 0.0 as its
                # "absent" default the way a floor does -- 0.0 would be the
                # strictest possible bound, not the loosest.
                float(self.curriculum_kwargs.get("max_unsupported_duty", math.inf)),
                values["mean_unsupported_duty"],
                counts["mean_unsupported_duty"],
                # The gate's own rule, not the panel size -- see _GateMetric.
                min_samples=required_duty_episodes(
                    int(self.curriculum_kwargs.get("min_eval_episodes", DEFAULT_MIN_EVAL_EPISODES)),
                    float(self.curriculum_kwargs.get("min_full_horizon_fraction", 0.0)),
                ),
                ceiling=True,
            ),
            _GateMetric(
                "full_horizon_fraction",
                "full-horizon episodes",
                float(self.curriculum_kwargs.get("min_full_horizon_fraction", 0.0)),
                horizon_fraction,
                horizon_count,
                percent=True,
            ),
            _GateMetric(
                "mean_episode_length",
                "episode length",
                float(self.curriculum_kwargs.get("min_avg_episode_length", 0.0)),
                values["mean_episode_length"],
                counts["mean_episode_length"],
            ),
            _GateMetric(
                "mean_reward",
                "mean reward",
                float(self.curriculum_kwargs.get("min_avg_reward", -math.inf)),
                values["mean_reward"],
                counts["mean_reward"],
            ),
        )
        return [
            metric
            for metric in configured
            if math.isfinite(metric.threshold) and (metric.key == "mean_reward" or metric.threshold > 0.0)
        ]

    def _clear_plateau(self, reason: str) -> None:
        if self._plateau_active:
            # %s, not %d: semantic stages ("recovery") reach here, and a %d
            # would turn the diagnostic into a '--- Logging error ---'.
            logger.info(
                "Stage %s evaluation plateau cleared at step %d: %s.",
                self.stage,
                self.num_timesteps,
                reason,
            )
        self._plateau_active = False
        self._plateau_metric = None

    @staticmethod
    def _format_value(metric: _GateMetric, value: float) -> str:
        if metric.percent:
            return f"{value:.1%}"
        if metric.key == "mean_forward_vel":
            return f"{value:.3f} m/s"
        if metric.key == "mean_episode_length":
            return f"{value:.1f} steps"
        if metric.key == "mean_unsupported_duty":
            return f"{value:.4f}"
        return f"{value:.3f}"

    @staticmethod
    def _gate_scale(threshold: float) -> float:
        """Return the gate magnitude, with a stable scale for a zero gate."""

        return abs(threshold) if threshold != 0.0 else 1.0

    def _record_stance_scalars(self, index: int, panel: Any | None) -> None:
        """Emit the quantities the stance gate actually decides on.

        All three gating numbers used to be absent from TensorBoard and W&B
        entirely — only a raw all-episode duty mean was published — so a
        stance-gated stage gave no way to tell whether it was blocked on the
        mean or on panel variance driving the bound, which is the single most
        useful thing to know.

        ``eval_duty_episodes`` is recorded even when it is zero: that is the
        early-training signal that no episode survived the settling window,
        and it reads identically to "not evaluated" if it is simply omitted.
        """
        if panel is None:
            return
        self.logger.record("diagnostics/eval_full_horizon_fraction", panel.full_horizon_fraction)
        self.logger.record("diagnostics/eval_duty_episodes", panel.n_duty_episodes)
        bilateral = self._full_horizon_mean("evaluations_bilateral_duties", index)
        # Recorded to TensorBoard AND accumulated for `gate_progress.npz`.
        # These are the deterministic, panel-estimator gate criteria -- the
        # same numbers the stance gate tests -- and until now they existed
        # only in TensorBoard. Post-hoc analysis of run 20260804_143747 had to
        # substitute the TRAINING-rollout duty from `diagnostics.npz`, which is
        # contaminated by exploration noise; it happened to agree to three
        # decimals there, but that was luck, not a property worth relying on.
        # Also to the SB3 logger, so TensorBoard and W&B carry them live
        # rather than only the npz. A diagnostic that exists solely in a file
        # nobody opens mid-run is how the stance gate criteria stayed invisible
        # for the whole of run 20260804_143747.
        action_stats = self._action_statistics(index)
        for key, value in action_stats.items():
            if math.isfinite(value):
                self.logger.record(f"diagnostics/eval_{key}", value)
        for key, value in self._reward_term_scalars(index).items():
            self.logger.record(f"reward_terms/eval_{key[len('term_') :]}", value)
        # Report-only, like the action statistics: which pad is on which edge, how much load its
        # digits carry, and the left/right leg offset that puts it there -- per evaluation, so a
        # posture the gate will refuse at the end shows up when it forms rather than 11M steps later.
        foot_telemetry = self._foot_telemetry_scalars(index)
        for key, value in foot_telemetry.items():
            if math.isfinite(value):
                self.logger.record(f"diagnostics/eval_{key}", value)
        # Report-only: how the panel's episodes ended (the env's termination reasons, and the time limit).
        terminations = self._termination_scalars(index)
        for key, value in terminations.items():
            if math.isfinite(value):
                self.logger.record(f"terminations/eval_{key[len(TERMINATION_KEY_PREFIX) :]}", value)
        self._record_gate_progress(
            index,
            full_horizon_fraction=panel.full_horizon_fraction,
            duty_episodes=panel.n_duty_episodes,
            unsupported_duty=panel.mean_unsupported_duty if panel.n_duty_episodes else float("nan"),
            unsupported_duty_ucb=(
                panel.unsupported_duty_ucb
                if panel.n_duty_episodes and math.isfinite(panel.unsupported_duty_ucb)
                else float("nan")
            ),
            bilateral_support_duty=float("nan") if bilateral is None else bilateral,
            mean_reward=panel.mean_reward,
            **action_stats,
            **self._reward_term_scalars(index),
            **foot_telemetry,
            **terminations,
        )
        if not panel.n_duty_episodes:
            return
        self.logger.record("diagnostics/eval_unsupported_duty", panel.mean_unsupported_duty)
        if math.isfinite(panel.unsupported_duty_ucb):
            self.logger.record("diagnostics/eval_unsupported_duty_ucb", panel.unsupported_duty_ucb)
        # Not gated. Measured over the SAME full-horizon episodes as the duty
        # above so the two are comparable; averaging it over every episode is
        # what made the three stance shares stop summing to 1.
        if bilateral is not None:
            self.logger.record("diagnostics/eval_bilateral_support_duty", bilateral)

    def _eval_series(self, name: str, index: int) -> Any:
        series = getattr(self.eval_callback, name, None)
        if not series or index >= len(series):
            return None
        return series[index]

    def _action_statistics(self, index: int) -> dict[str, float]:
        """Scalar summaries of the deterministic policy's action.

        ``action_dc_rms`` is the distance from the home control (``action = 0``) and
        ``action_ac_rms`` is the tremor about it; keeping them apart is the
        point, because they have different causes and different fixes
        (issue #489). The per-actuator vectors go to `gate_progress.npz`
        separately -- these scalars exist so the series is readable without
        reducing a matrix.
        """
        out: dict[str, float] = {}
        dc = self._eval_series("evaluations_action_dc", index)
        ac = self._eval_series("evaluations_action_ac_rms", index)
        out["action_dc_rms"] = float(np.sqrt(np.mean(np.square(dc)))) if dc else float("nan")
        out["action_ac_rms"] = float(np.sqrt(np.mean(np.square(ac)))) if ac else float("nan")
        for key, name in (("action_delta", "evaluations_action_delta"), ("action_jerk", "evaluations_action_jerk")):
            series = getattr(self.eval_callback, name, None)
            value = series[index] if series and index < len(series) else float("nan")
            out[key] = float(value)
        # The frequency the two differences imply, for a narrowband action
        # signal: jerk/delta = (2 sin(pi f dt))^2, blind to any DC offset. A
        # value at the white-noise limit means the ratio is saturated and the
        # signal is broadband, not that it sits at that frequency.
        delta, jerk = out["action_delta"], out["action_jerk"]
        if math.isfinite(delta) and math.isfinite(jerk) and delta > 0.0:
            gain = math.sqrt(jerk / delta)
            out["action_freq_hz"] = math.asin(min(1.0, gain / 2.0)) / (math.pi * self.control_dt)
        else:
            out["action_freq_hz"] = float("nan")
        return out

    def _reward_term_scalars(self, index: int) -> dict[str, float]:
        """Per-episode mean of each reward term, flattened into the npz."""
        terms = self._eval_series("evaluations_reward_terms", index)
        if not terms:
            return {}
        return {f"term_{key[len('reward_') :]}": float(value) for key, value in terms.items()}

    def _bind_gate_progress(self) -> None:
        """Bind instance-local ``gate_progress.npz`` containers, so nothing appends to the class-scope defaults."""
        if "_gate_progress_timesteps" not in self.__dict__:
            self._gate_progress_timesteps = []
            self._gate_progress = {}

    def _termination_scalars(self, index: int) -> dict[str, float]:
        """This evaluation's episode count per termination cause, as ``termination_<cause>`` keys.

        Every cause seen so far in the file is supplied (0 when no episode
        ended that way), so each series stays aligned with ``timesteps``.  A
        cause first seen now reads 0 on the earlier rows that counted causes
        and NaN on rows that did not (a resumed run's rows from before this
        telemetry).  An evaluation whose causes were not captured reads NaN
        for every known cause, and no key at all before the first cause.
        """
        self._bind_gate_progress()
        recorded = self._gate_progress
        known = [key for key in recorded if key.startswith(TERMINATION_KEY_PREFIX)]
        causes = self._eval_series("evaluations_termination_causes", index)
        if not causes:
            return dict.fromkeys(known, float("nan"))
        counts = Counter(str(cause) for cause in causes)
        rows = len(self._gate_progress_timesteps)
        counted = [
            any(row < len(recorded[key]) and math.isfinite(recorded[key][row]) for key in known) for row in range(rows)
        ]
        for cause in sorted(counts):
            key = f"{TERMINATION_KEY_PREFIX}{cause}"
            if key not in recorded:
                recorded[key] = [0.0 if counted[row] else float("nan") for row in range(rows)]
                known.append(key)
        return {key: float(counts.get(key[len(TERMINATION_KEY_PREFIX) :], 0)) for key in known}

    def gate_progress_row(self, row: int = -1) -> dict[str, float]:
        """One recorded row of ``gate_progress.npz``'s scalar series, by key (the latest by default).

        Only the series aligned with ``timesteps`` (the ones the file holds)
        are returned, with ``"timesteps"`` itself; ``{}`` before the first
        row.  Read by the training progress report (``curriculum/progress_report.py``).
        """
        timesteps = list(self.__dict__.get("_gate_progress_timesteps", []))
        if not timesteps or not -len(timesteps) <= row < len(timesteps):
            return {}
        out = {"timesteps": float(timesteps[row])}
        for key, series in self.__dict__.get("_gate_progress", {}).items():
            if len(series) == len(timesteps):
                out[key] = float(series[row])
        return out

    def gate_progress_series(self, key: str) -> tuple[list[int], list[float]]:
        """``(timesteps, values)`` of one ``gate_progress.npz`` series recorded so far.

        Copies, aligned by evaluation; a key this run never recorded (or one
        recorded on only some evaluations) returns NaN where it is missing
        rather than a shorter list, so a reader can never pair a value with
        the wrong evaluation.  Read by the hop watch (curriculum/hop_watch.py).
        """
        timesteps = list(self.__dict__.get("_gate_progress_timesteps", []))
        series = list(self.__dict__.get("_gate_progress", {}).get(key, []))
        if len(series) != len(timesteps):
            series = [float("nan")] * len(timesteps)
        return timesteps, series

    def _record_gate_progress(self, index: int | None = None, /, **values: float) -> None:
        """Append one evaluation's gate criteria and rewrite ``gate_progress.npz``.

        A separate file rather than columns in ``diagnostics.npz`` because the
        two are on different clocks: diagnostics is per training rollout, this
        is per evaluation. Merging them would force one series to be padded
        with NaN against the other's timeline, which is exactly the alignment
        trap ``_history_algo`` already has to work around.

        *index* is the evaluation's absolute index into the capture callback's
        series, where its per-actuator rows are read.  ``None`` reads the
        evaluation at this row's position, which is the same one only while
        nothing was seeded and every evaluation was recorded: a same-stage
        resume (:meth:`seed_prior_gate_progress`) pads those series for the
        seeded evaluations, so reading them by row count took the padding.
        """
        # Bind instance-local containers before appending: the class-scope
        # defaults above exist only so `object.__new__` instances have the
        # attribute, and appending to them would share state across every
        # instance in the process.
        self._bind_gate_progress()
        if index is None:
            index = len(self._gate_progress_timesteps)
        # A key first supplied after a resume reads NaN on the seeded rows: nothing measured it there.
        seeded = self._gate_progress_seeded
        self._gate_progress_timesteps.append(int(self.num_timesteps))
        for key, value in values.items():
            self._gate_progress.setdefault(key, [float("nan")] * seeded).append(float(value))
        # Per-actuator DC and AC, for (n_evals, n_actuators) matrices. Which
        # actuators carry the offset decides whether a leg-pose weight can
        # reach it at all -- `leg_home_pose` covers only the leg joints, so
        # a displacement living in the tail or neck is invisible to it and
        # unfixable by it (issue #489). Scalar summaries alone cannot answer
        # that, and the grouping is left to analysis rather than guessed at
        # here.
        # `getattr` with defaults, not `self.eval_callback` directly: these
        # matrices are an enhancement, and an AttributeError here would sink
        # the run -- or, in the writer, take the ENTIRE file down with it,
        # losing the gate criteria to a failure in a side channel.
        source = getattr(self, "eval_callback", None)
        matrices = self.__dict__.setdefault("_gate_progress_rows", {})
        for key, name in self._GATE_PROGRESS_MATRICES:
            series = getattr(source, name, None) or []
            row = series[index] if 0 <= index < len(series) else None
            matrices.setdefault(key, [[] for _ in range(seeded)]).append(list(row) if row else [])
        if self._gate_progress_dir is None:
            return
        try:
            from .file_io import atomic_savez

            n = len(self._gate_progress_timesteps)
            payload = {"timesteps": np.array(self._gate_progress_timesteps)}
            for key, rows in matrices.items():
                # Ragged rows would need padding, which invents values. A run
                # whose actuator count changed mid-flight is not a thing that
                # happens; a run with some empty evaluations is.
                if len(rows) == n and all(rows) and len({len(row) for row in rows}) == 1:
                    payload[key] = np.array(rows, dtype=float)
            # Length-guarded against `timesteps`, like every optional series in
            # `diagnostics.npz`. Today one call site supplies every key on every
            # evaluation, so nothing can diverge -- but a future caller that
            # supplies a key conditionally would otherwise write a short array
            # that silently reads as the FIRST n evaluations rather than the
            # ones it came from. Dropping it is recoverable; misaligning it is
            # the kind of error this file exists to stop.
            for key, series in self._gate_progress.items():
                if len(series) == n:
                    payload[key] = np.array(series)
            atomic_savez(Path(self._gate_progress_dir) / "gate_progress.npz", **payload)
        except Exception:  # noqa: BLE001 - a diagnostic must not sink a run
            # Once. An unwritable directory fails identically on all ~200
            # evaluations of a 10M run, and 200 stack traces would bury the
            # warnings this run is actually meant to surface.
            if not self._warned_gate_progress_unwritable:
                self._warned_gate_progress_unwritable = True
                logger.warning("Could not write gate_progress.npz", exc_info=True)

    def seed_prior_gate_progress(self, path: "str | Path", *, max_timesteps: int | None = None) -> int:
        """Start from a prior session's ``gate_progress.npz`` at *path*, so a same-stage resume appends to it.

        Without it a resumed session rewrote the file with its own
        evaluations only.  Called by ``curriculum.seed_resume_eval_state``
        with the file beside the ``evaluations.npz`` it seeds from.  Every
        row is kept as it was, except those recorded after *max_timesteps*
        (the resumed checkpoint), which are dropped as the evaluation history
        drops them, so the timesteps stay monotonic.  A key this session
        records that the prior file lacks reads NaN on the seeded rows; a
        matrix it lacks stays out of the file, as one unmeasured evaluation
        keeps it out; and a prior key this session stops supplying is left
        out, like any key recorded on only some evaluations.

        Returns the number of rows seeded: 0 when there is no readable file,
        and the file then starts over with this session's evaluations.
        """
        path = Path(path)
        if not path.exists():
            logger.info("Resume: no prior %s; gate_progress.npz starts with this session's evaluations.", path)
            return 0
        try:
            with np.load(path) as data:
                timesteps = [int(value) for value in data["timesteps"]]
                columns = {key: np.asarray(data[key], dtype=float) for key in data.files if key != "timesteps"}
        except Exception:  # noqa: BLE001 - a diagnostic must not sink a run
            logger.warning(
                "Resume: could not read %s; gate_progress.npz starts with this session's evaluations.",
                path,
                exc_info=True,
            )
            return 0
        rows = len(timesteps)
        keep = rows if max_timesteps is None else sum(1 for value in timesteps if value <= max_timesteps)
        matrices = dict(self._GATE_PROGRESS_MATRICES)
        self._gate_progress_timesteps = timesteps[:keep]
        self._gate_progress = {
            key: [float(value) for value in column[:keep]]
            for key, column in columns.items()
            if key not in matrices and column.shape == (rows,)
        }
        self._gate_progress_rows = {}
        for key in matrices:
            column = columns.get(key)
            if column is not None and column.ndim == 2 and len(column) == rows:
                self._gate_progress_rows[key] = [[float(value) for value in row] for row in column[:keep]]
            else:
                self._gate_progress_rows[key] = [[] for _ in range(keep)]
        self._gate_progress_seeded = keep
        return keep

    def _process_evaluation(self, index: int) -> None:
        panel = self._stance_panel(index)
        # Record BEFORE the no-metrics early-out below: a stage whose gate
        # declares none of the legacy scalar thresholds (the recovery stage's
        # recovery_quality/v1 — judged post-stage from its frozen resolution)
        # has no plateau metrics to follow, but its gate_progress.npz appends
        # and diagnostics/eval_* scalars are exactly the mid-run visibility
        # the stage needs.  Gating the recording on the plateau metrics made
        # every recovery run train dark (gap review EE1).
        self._record_stance_scalars(index, panel)

        metrics = self._metrics_for_evaluation(index, panel)
        if not metrics:
            # Nothing to follow for plateau/gate-met tracking; the recording
            # above already happened.
            return

        min_eval_episodes = int(self.curriculum_kwargs.get("min_eval_episodes", DEFAULT_MIN_EVAL_EPISODES))
        reward_metric = next((metric for metric in metrics if metric.key == "mean_reward"), None)
        if reward_metric is not None:
            self.logger.record("diagnostics/eval_episode_count", reward_metric.sample_count)

        # Follow the metrics this evaluation can actually support, rather than
        # abandoning the whole diagnostic when one of them is unmeasurable.
        #
        # This used to `return` if ANY configured metric was short of samples.
        # Adding unsupported duty to the priority list then switched the entire
        # callback off for exactly the case it is most needed: early stage-1
        # training, where episodes end before settle_steps and every duty is
        # NaN. Measured — a flat, dying run produced one "episode length
        # plateaued" warning under reward_and_length/v1 and ZERO under the
        # stance gate, with eval_gate_met and eval_plateau_active never
        # recorded at all.
        usable = [
            metric
            for metric in metrics
            if metric.value is not None
            and metric.sample_count >= (min_eval_episodes if metric.min_samples is None else metric.min_samples)
        ]
        complete = len(usable) == len(metrics)
        self.logger.record("diagnostics/eval_gate_data_complete", 1.0 if complete else 0.0)
        if not usable:
            return

        for metric in usable:
            assert metric.value is not None
            self._histories[metric.key].append(metric.value)

        failing = [metric for metric in usable if metric.is_failing()]
        if not failing:
            # "Met" only when every configured criterion was checkable. A
            # partial panel that happens to clear the criteria it could
            # measure has not met the gate, and publishing 1.0 here would say
            # it had.
            self.logger.record("diagnostics/eval_gate_met", 1.0 if complete else 0.0)
            self.logger.record("diagnostics/eval_plateau_active", 0.0)
            if complete:
                self._clear_plateau("all active stage gates are currently met")
            return

        self.logger.record("diagnostics/eval_gate_met", 0.0)
        blocking = next(metric for key in self._METRIC_PRIORITY for metric in failing if metric.key == key)
        assert blocking.value is not None

        # `blocking` is the most behaviourally specific failing metric AMONG
        # THOSE MEASURABLE. When the panel is partial, a more specific
        # criterion may be the real blocker and simply have no data, so the
        # warning has to say which ones went unmeasured rather than implying
        # the named metric is the whole story.
        unmeasured = [metric.label for metric in metrics if metric not in usable]
        caveat = (
            ""
            if complete
            else (
                f" NOTE: {', '.join(unmeasured)} could not be measured this evaluation, "
                "so a more specific criterion may be the real blocker."
            )
        )

        if self._plateau_metric is not None and self._plateau_metric != blocking.key:
            self._clear_plateau(f"the blocking gate changed to {blocking.label}")
        self._plateau_metric = blocking.key

        history = self._histories[blocking.key]
        if len(history) < self.plateau_window:
            self.logger.record("diagnostics/eval_plateau_active", 0.0)
            return

        recent = history[-self.plateau_window :]
        absolute_range = max(recent) - min(recent)
        gate_scale = self._gate_scale(blocking.threshold)
        relative_range = absolute_range / gate_scale
        self.logger.record("diagnostics/eval_plateau_relative_range", relative_range)
        relative_margin = (blocking.value - blocking.threshold) / gate_scale
        self.logger.record("diagnostics/eval_gate_relative_margin", relative_margin)

        if relative_range >= self.min_relative_variation:
            self._clear_plateau(
                f"{blocking.label} moved by {relative_range:.1%} of its gate over the evaluation window"
            )
        elif not self._plateau_active:
            message = (
                f"STAGE {self.stage} EVALUATION PLATEAU: {blocking.label} stayed within a "
                f"{self._format_value(blocking, absolute_range)} range ({relative_range:.1%} of its gate) "
                f"across the last {self.plateau_window} evaluations; latest "
                f"{self._format_value(blocking, blocking.value)}, required "
                f"{self._format_value(blocking, blocking.threshold)}, step {self.num_timesteps}. "
                f"{self._GUIDANCE[blocking.key]}{caveat} "
                "Training continues; this diagnostic is not the curriculum gate or collapse early-stop."
            )
            logger.warning("%s", message)
            # The training progress report (progress.json) lists the advisories a run has raised.
            self.__dict__.setdefault("progress_warnings", []).append(
                {"timesteps": int(self.num_timesteps), "source": "evaluation_plateau", "message": message}
            )
            self._plateau_active = True

        self.logger.record("diagnostics/eval_plateau_active", float(self._plateau_active))

    def _on_step(self) -> bool:
        results = getattr(self.eval_callback, "evaluations_results", None)
        if results is None:
            return True

        n_evals = len(results)
        if n_evals < self._last_seen_n_evals:
            self._last_seen_n_evals = 0
            self._histories = {key: [] for key in self._METRIC_PRIORITY}
            self._clear_plateau("evaluation history was reset")

        for index in range(self._last_seen_n_evals, n_evals):
            self._process_evaluation(index)
        self._last_seen_n_evals = n_evals
        return True


def build_stage_evaluation_callbacks(
    eval_env: Any,
    *,
    stage: "int | str",
    stage_config: dict[str, Any],
    diagnostics_verbose: int = 0,
    gate_progress_dir: "str | Path | None" = None,
    model_id: "ModelId | None" = None,
    **eval_callback_kwargs: Any,
) -> tuple[StageAwareEvalCallback, StageGatePlateauCallback]:
    """Build the paired evaluation and stage-gate diagnostic callbacks.

    *model_id* names the model *eval_env* runs; without it the evaluations
    record no pad telemetry and no sagittal leg DC
    (:class:`StageAwareEvalCallback`).
    """

    curriculum_kwargs = stage_config.get("curriculum_kwargs", {})
    eval_callback = StageAwareEvalCallback(
        eval_env,
        stage=stage,
        success_applicable=success_metric_applicable(stage_config),
        # Taken from the stage's own gate declaration so the duty this
        # callback measures uses the same settling window the gate compares
        # it against. Absent (any non-stance gate) it is 0, which measures the
        # whole episode and costs nothing.
        settle_steps=int(curriculum_kwargs.get("settle_steps", 0)),
        model_id=model_id,
        **eval_callback_kwargs,
    )
    plateau_callback = StageGatePlateauCallback(
        eval_callback,
        stage=stage,
        curriculum_kwargs=curriculum_kwargs,
        plateau_window=int(curriculum_kwargs.get("diagnostics_plateau_window", 5)),
        min_relative_variation=float(curriculum_kwargs.get("diagnostics_plateau_min_relative_variation", 0.05)),
        # From [env], not [curriculum]: the horizon defines what "full" means
        # for the full-horizon fraction, and it is an environment property.
        horizon=int(stage_config.get("env_kwargs", {}).get("max_episode_steps", 1000)),
        gate_progress_dir=gate_progress_dir,
        # Also from [env]: one control step is `timestep * frame_skip`, and
        # without it the jerk/delta ratio is a bare number rather than a
        # frequency. Defaults match the MuJoCo/repo defaults so a config that
        # declares neither still reports a sane 100 Hz rather than nothing.
        control_dt=(
            float(stage_config.get("env_kwargs", {}).get("timestep", 0.002))
            * int(stage_config.get("env_kwargs", {}).get("frame_skip", 5))
        ),
        verbose=diagnostics_verbose,
    )
    return eval_callback, plateau_callback
