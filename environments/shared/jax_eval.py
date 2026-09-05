"""CPU-based evaluation for JAX-trained policies.

Runs deterministic evaluation episodes on CPU MuJoCo and collects
per-step biomechanical diagnostics. Used for curriculum stage-gate
checks and generating diagnostic plots.

Usage::

    from environments.shared.jax_eval import evaluate_policy_cpu

    results = evaluate_policy_cpu(
        mj_model, params, network, obs_stats,
        get_obs_fn=get_obs, scale_action_fn=scale_action,
        reward_fn=compute_reward, reward_cfg=reward_cfg,
        config=eval_config,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .action_filter import apply_low_pass, low_pass_alpha
from .perturbation import (
    derive_push_parameters,
    external_push_force,
    max_pushes_for,
    push_schedule,
    validate_push_config,
)
from .reward_functions import reward_lean_aware_posture, reward_posture


@dataclass
class EvalConfig:
    """Configuration for CPU evaluation episodes."""

    n_episodes: int = 25
    max_episode_steps: int = 1000
    frame_skip: int = 5
    healthy_z_range: tuple[float, float] = (0.3, 2.0)
    max_tilt_angle: float = 1.047
    natural_forward_z: float = 0.0
    posture_target_forward_z: float | None = None
    nosedive_threshold: float = 0.5
    termination_body_heights: dict[str, float] | None = None
    termination_site_heights: dict[str, float] | None = None
    success_sites: tuple[str, ...] = ()
    success_threshold: float = 0.3
    target_body: str | None = None
    root_body_id: int = 1
    sensor_gyro_start: int = 0
    sensor_quat_start: int = 6
    action_mapping: str = "midpoint/v1"
    # Command low-pass cutoff of the plant interface (0.0 = off).  Eval must
    # drive the same filtered plant the training kernel does, and feed the
    # filtered command to the action-derived reward terms.
    action_filter_cutoff_hz: float = 0.0
    # Scheduled external pushes of the recovery task (0.0 = off).  Eval
    # must roll the same pushed plant the training kernels do -- schedule,
    # force and body from the shared perturbation module -- or a recovery
    # stage certifies push-free episodes.  Names mirror MJXEnvConfig's
    # perturbation_* fields so callers thread them one-to-one.
    perturbation_capture_velocity_multiple: float = 0.0
    perturbation_interval: float = 2.0
    perturbation_jitter: float = 0.5
    perturbation_duration: float = 0.20
    perturbation_direction: str = "uniform_horizontal"
    reset_noise_scale: float = 0.01
    init_qpos_noise: float = 0.0
    init_yaw_noise: float = 0.0
    # ``None`` preserves the historical CPU-eval behavior of leaving the
    # XML-authored target in place.  Stage evaluation supplies the effective
    # training-environment ranges.
    target_distance_range: tuple[float, float] | None = None
    target_lateral_range: tuple[float, float] | None = None
    target_z: float = 0.5
    forward_vel_max: float = 8.0
    # Success bonus added to the reward when a success site reaches the
    # target (stage 3), and penalty applied on fall termination — both
    # mirror the training env so eval episode rewards are comparable.
    success_bonus: float = 0.0
    fall_penalty: float = 0.0
    # Species standing height for the height-reward decomposition (matches
    # the SB3 envs' target_z: 0.90 trex, 1.2 brachio).
    target_standing_z: float = 0.90
    # Seed for reset noise — evaluation is otherwise non-reproducible and
    # curriculum gate decisions would vary run-to-run for borderline policies.
    seed: int = 42


def _posture_reward_for_eval(
    quat: np.ndarray,
    config: EvalConfig,
    weight: float,
) -> float:
    """Mirror the posture primitive selected by JAX training."""
    if config.posture_target_forward_z is None:
        reward, _ = reward_posture(quat, config.max_tilt_angle, weight)
    else:
        reward, _ = reward_lean_aware_posture(
            quat,
            config.max_tilt_angle,
            weight,
            config.posture_target_forward_z,
        )
    return float(reward)


def _apply_eval_reset_randomization(
    mj_model: Any,
    mj_data: Any,
    config: EvalConfig,
    reset_rng: np.random.Generator,
    target_body_id: int,
) -> None:
    """Apply the reset distribution used by ``MJXDinoEnv`` on CPU.

    NumPy's RNG is intentionally used for deterministic CPU evaluation; exact
    bitwise equality with JAX's PRNG is neither required nor expected.
    """
    if config.target_distance_range is not None and config.target_lateral_range is not None:
        distance = reset_rng.uniform(*config.target_distance_range)
        lateral = reset_rng.uniform(*config.target_lateral_range)
        if target_body_id < 0:
            raise ValueError("random target reset requires a valid target body")
        target_mocap_id = int(mj_model.body_mocapid[target_body_id])
        if target_mocap_id < 0:
            raise ValueError("random target reset requires the target body to be mocap-controlled")
        mj_data.mocap_pos[target_mocap_id] = np.array([distance, lateral, config.target_z])

    if config.reset_noise_scale > 0:
        mj_data.qpos[7:] += reset_rng.uniform(
            -config.reset_noise_scale,
            config.reset_noise_scale,
            size=mj_data.qpos[7:].shape,
        )

    if config.init_qpos_noise > 0:
        mj_data.qpos[:2] += reset_rng.uniform(
            -config.init_qpos_noise,
            config.init_qpos_noise,
            size=2,
        )

    if config.init_yaw_noise > 0:
        yaw_angle = reset_rng.uniform(-config.init_yaw_noise, config.init_yaw_noise)
        half_yaw = yaw_angle / 2.0
        yaw_quat = np.array([np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw)])
        w1, x1, y1, z1 = yaw_quat
        w2, x2, y2, z2 = mj_data.qpos[3:7].copy()
        mj_data.qpos[3:7] = np.array(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ]
        )


@dataclass
class EvalResults:
    """Collected evaluation results and per-step diagnostics."""

    # Per-episode metrics
    rewards: list[float] = field(default_factory=list)
    lengths: list[int] = field(default_factory=list)
    forward_vels: list[float] = field(default_factory=list)
    distances: list[float] = field(default_factory=list)
    tilt_angles: list[float] = field(default_factory=list)
    pelvis_heights: list[float] = field(default_factory=list)
    # True for episodes that ended by reaching the success target (stage 3).
    successes: list[bool] = field(default_factory=list)

    # Per-step diagnostics (across all episodes)
    diag_tilt: list[float] = field(default_factory=list)
    diag_fwd_vel: list[float] = field(default_factory=list)
    diag_pelvis_h: list[float] = field(default_factory=list)
    diag_l_foot: list[float] = field(default_factory=list)
    diag_r_foot: list[float] = field(default_factory=list)
    diag_energy: list[float] = field(default_factory=list)
    diag_reward_components: dict[str, list[float]] = field(
        default_factory=lambda: {
            "forward": [],
            "alive": [],
            "energy": [],
            "posture": [],
            "height": [],
            "nosedive": [],
            "drift": [],
            "speed": [],
            "spin": [],
            "smoothness": [],
        }
    )
    # Non-reward state used to explain the reward components (for example,
    # bilateral-support quality and the alive gate).  Kept separate so plots
    # and exported ``reward_*`` series cannot mistake qualities/errors for
    # additive reward terms.
    diag_reward_diagnostics: dict[str, list[float]] = field(default_factory=dict)

    @property
    def mean_reward(self) -> float:
        return float(np.mean(self.rewards)) if self.rewards else 0.0

    @property
    def std_reward(self) -> float:
        return float(np.std(self.rewards)) if self.rewards else 0.0

    @property
    def mean_length(self) -> float:
        return float(np.mean(self.lengths)) if self.lengths else 0.0

    @property
    def std_length(self) -> float:
        return float(np.std(self.lengths)) if self.lengths else 0.0

    @property
    def mean_forward_vel(self) -> float:
        return float(np.mean(self.forward_vels)) if self.forward_vels else 0.0

    @property
    def mean_distance(self) -> float:
        return float(np.mean(self.distances)) if self.distances else 0.0

    @property
    def mean_tilt(self) -> float:
        return float(np.mean(self.tilt_angles)) if self.tilt_angles else 0.0

    @property
    def mean_height(self) -> float:
        return float(np.mean(self.pelvis_heights)) if self.pelvis_heights else 0.0

    @property
    def mean_success_rate(self) -> float:
        return float(np.mean(self.successes)) if self.successes else 0.0


def evaluate_policy_cpu(
    mj_model: Any,
    params: Any,
    network: Any,
    obs_stats: Any,
    *,
    get_obs_fn: Any,
    normalize_obs_fn: Any,
    scale_action_fn: Any,
    reward_fn: Any,
    reward_cfg: dict[str, float],
    config: EvalConfig,
    foot_sensor_indices: tuple[int, ...] = (),
    foot_aux_indices: tuple[tuple[int, ...], ...] = (),
    reward_components_fn: Any | None = None,
) -> EvalResults:
    """Run deterministic evaluation episodes on CPU MuJoCo.

    Args:
        mj_model: MuJoCo model.
        params: JAX network parameters.
        network: Flax ActorCritic network module.
        obs_stats: RunningMeanStd observation statistics.
        get_obs_fn: Function(mjx_data) -> obs array.
        normalize_obs_fn: Function(obs, obs_stats) -> normalized obs.
        scale_action_fn: Function(action) -> scaled ctrl.
        reward_fn: Function(mjx_data, action, reward_cfg) -> scalar reward.
        reward_cfg: Reward weight dict.
        config: Evaluation configuration.
        foot_sensor_indices: Sensor indices for foot contacts (for diagnostics),
            one per foot.
        foot_aux_indices: Extra sensors summed into each foot's reading,
            aligned with ``foot_sensor_indices``.  Species whose toes are
            separate bodies need these, since one touch sensor cannot see
            geoms on child bodies.
        reward_components_fn: Optional canonical detailed reward function
            with the same call shape as ``reward_fn``.  When provided, its
            values replace the legacy hand-calculated component estimates;
            keys prefixed with ``_`` are recorded as non-reward diagnostics.

    Returns:
        ``EvalResults`` with per-episode and per-step metrics.
    """
    import jax.numpy as jnp
    import mujoco
    import mujoco.mjx as mjx

    from .mjx_env import ACTION_MAPPING_HOME_KEYFRAME_RESIDUAL, ACTION_MAPPING_MIDPOINT
    from .mjx_utils import reset_mujoco_data_to_home

    mj_data = mujoco.MjData(mj_model)
    act_dim = mj_model.nu
    results = EvalResults()
    reset_rng = np.random.default_rng(config.seed)

    # Resolve body-height termination checks
    _body_checks: list[tuple[int, float]] = []
    if config.termination_body_heights:
        for bname, z_thresh in config.termination_body_heights.items():
            bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, bname)
            if bid >= 0:
                _body_checks.append((bid, z_thresh))

    # Resolve site-height termination checks
    _site_checks: list[tuple[int, float]] = []
    if config.termination_site_heights:
        for sname, z_thresh in config.termination_site_heights.items():
            sid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, sname)
            if sid >= 0:
                _site_checks.append((sid, z_thresh))

    # Resolve success site IDs and target body
    _success_site_ids: list[int] = []
    for sname in config.success_sites:
        sid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, sname)
        if sid >= 0:
            _success_site_ids.append(sid)
    _target_body_id = -1
    if config.target_body is not None:
        _target_body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, config.target_body)

    # Command low-pass of the plant interface (see action_filter.py).
    _filter_alpha: float | None = None
    if config.action_filter_cutoff_hz > 0.0:
        _filter_alpha = low_pass_alpha(
            config.action_filter_cutoff_hz,
            float(mj_model.opt.timestep) * config.frame_skip,
        )

    # Scheduled pushes of the recovery task (see perturbation.py).  None
    # means off: no derivation, no RNG draw, no xfrc write -- byte-identical
    # to the push-free eval, exactly as both training backends promise.
    _push: dict[str, Any] | None = None
    if config.perturbation_capture_velocity_multiple > 0.0:
        _schedule_steps = validate_push_config(
            capture_velocity_multiple=config.perturbation_capture_velocity_multiple,
            interval_s=config.perturbation_interval,
            jitter_s=config.perturbation_jitter,
            duration_s=config.perturbation_duration,
            direction=config.perturbation_direction,
            control_dt=float(mj_model.opt.timestep) * config.frame_skip,
        )
        # Calibrate from the pose the reset restores, as both training
        # backends do (MJXDinoEnv resolves 'home' by name, BaseDinoEnv passes
        # its _reset_keyframe_id); keyframe 0 only when no 'home' exists.
        _push_keyframe_id = max(mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_KEY, "home"), 0)
        _push_params = derive_push_parameters(
            mj_model,
            capture_velocity_multiple=config.perturbation_capture_velocity_multiple,
            duration_s=config.perturbation_duration,
            keyframe_id=_push_keyframe_id,
        )
        _push = {
            **_schedule_steps,
            "max_pushes": max_pushes_for(
                config.max_episode_steps,
                _schedule_steps["interval_steps"],
                _schedule_steps["jitter_steps"],
            ),
            "root_body_id": int(_push_params["root_body_id"]),
            "force_n": float(_push_params["force_n"]),
        }

    for ep in range(config.n_episodes):
        if config.action_mapping == ACTION_MAPPING_HOME_KEYFRAME_RESIDUAL:
            reset_mujoco_data_to_home(mj_model, mj_data)
        elif config.action_mapping == ACTION_MAPPING_MIDPOINT:
            mujoco.mj_resetData(mj_model, mj_data)
        else:
            raise ValueError(f"unknown JAX action mapping {config.action_mapping!r}")
        _apply_eval_reset_randomization(
            mj_model,
            mj_data,
            config,
            reset_rng,
            _target_body_id,
        )
        push_starts: np.ndarray | None = None
        push_dirs: np.ndarray | None = None
        if _push is not None:
            # New episode, new schedule.  The seed draw is APPENDED to the
            # reset draw sequence (BaseDinoEnv.reset's discipline), so the
            # joint/target draws above are bit-identical to a push-free eval
            # of the same seed, and the same eval seed yields the same
            # schedule for the policy and for every null controller.
            mj_data.xfrc_applied[:] = 0.0
            schedule_seed = np.uint32(reset_rng.integers(0, 2**32, dtype=np.uint64))
            push_starts, push_dirs = push_schedule(
                schedule_seed,
                max_pushes=_push["max_pushes"],
                interval_steps=_push["interval_steps"],
                jitter_steps=_push["jitter_steps"],
            )
        mujoco.mj_forward(mj_model, mj_data)

        ep_reward = 0.0
        ep_fwd_vels = []
        ep_tilts = []
        ep_heights = []
        ep_success = False
        start_pos = mj_data.xpos[config.root_body_id, :2].copy()
        prev_action = None
        # Second action lag for the jerk term; None = zeros, like the MJX
        # kernel's reset carry (see compute_total_reward).
        prev_prev_action = None
        # Per-episode command low-pass carry (None = seed with first action),
        # mirroring the training backends' filter state.
        filtered_action = None
        # Track the target for reward parity with the training env (approach
        # shaping compares against the previous step's distance).
        prev_target_dist: float | None = None
        if _target_body_id >= 0:
            prev_target_dist = float(np.linalg.norm(mj_data.xpos[_target_body_id] - mj_data.xpos[config.root_body_id]))

        for step in range(config.max_episode_steps):
            cpu_data = mjx.put_data(mj_model, mj_data)
            obs_raw = get_obs_fn(cpu_data)
            obs = normalize_obs_fn(obs_raw, obs_stats)

            mean, _log_std, _value = network.apply(params, obs)
            action = jnp.clip(mean, -1.0, 1.0)
            if _filter_alpha is not None:
                # Same seeding and blend as the training kernels; the
                # filtered command feeds ctrl, the reward call, and the
                # prev_action carry below.
                if filtered_action is not None:
                    action = apply_low_pass(filtered_action, action, _filter_alpha)
                filtered_action = action

            ctrl = np.array(scale_action_fn(action))
            mj_data.ctrl[:] = ctrl
            if _push is not None:
                # Scheduled external push, in lockstep with both training
                # backends: the root row is rewritten every control step and
                # the kernel returns zero outside every window, so the force
                # self-clears and never persists past its schedule.  Reward
                # and observation are deliberately untouched.
                mj_data.xfrc_applied[_push["root_body_id"], 0:3] = external_push_force(
                    step,
                    push_starts,
                    push_dirs,
                    duration_steps=_push["duration_steps"],
                    force_newtons=_push["force_n"],
                )
            # Substep-aggregated contact, in lockstep with both training
            # backends (BaseDinoEnv.step and the MJX step_fn carry): per-foot
            # MIN touch force across the frame_skip substeps, and an
            # any-substep OR of the body/site floor-strike checks.  The
            # boundary sample alone let a control-clock-locked hop unload
            # between samples and still read as continuous support -- eval
            # must score the same quantity training optimizes.
            min_foot_forces: "list[float] | None" = None
            substep_struck = False
            for _ in range(config.frame_skip):
                mujoco.mj_step(mj_model, mj_data)
                substep_forces = []
                for i, idx in enumerate(foot_sensor_indices):
                    if len(mj_data.sensordata) <= idx:
                        continue
                    force = float(mj_data.sensordata[idx])
                    if i < len(foot_aux_indices):
                        force += sum(
                            float(mj_data.sensordata[aux])
                            for aux in foot_aux_indices[i]
                            if len(mj_data.sensordata) > aux
                        )
                    substep_forces.append(force)
                if min_foot_forces is None:
                    min_foot_forces = substep_forces
                else:
                    min_foot_forces = [min(prev, cur) for prev, cur in zip(min_foot_forces, substep_forces)]
                if not substep_struck:
                    substep_struck = any(mj_data.xpos[bid, 2] < zt for bid, zt in _body_checks) or any(
                        mj_data.site_xpos[sid, 2] < zt for sid, zt in _site_checks
                    )

            # Per-step diagnostics
            fwd_vel = float(mj_data.qvel[0])
            body_z = float(mj_data.xpos[config.root_body_id, 2])
            root_quat = mj_data.sensordata[config.sensor_quat_start : config.sensor_quat_start + 4]
            tilt = float(np.arccos(np.clip(1.0 - 2.0 * (root_quat[1] ** 2 + root_quat[2] ** 2), -1, 1)))
            energy = float(np.sum(np.square(np.array(action)))) / act_dim

            ep_fwd_vels.append(fwd_vel)
            ep_tilts.append(tilt)
            ep_heights.append(body_z)

            results.diag_fwd_vel.append(fwd_vel)
            results.diag_tilt.append(tilt)
            results.diag_pelvis_h.append(body_z)
            results.diag_energy.append(energy)
            # Any-substep strike, from the aggregation loop above -- the
            # boundary-only checks missed strikes resolving mid-control-step.
            strike_terminated = substep_struck

            # Foot contacts — substep-MIN per foot, from the aggregation loop.
            # Sensor order alternates right/left (bipeds: R, L; quadrupeds:
            # R, L, RR, RL), so split by parity rather than lumping every
            # non-first foot into the left series.  These series feed the
            # stance-duty reconstruction, so they must carry the same
            # aggregated quantity the gate certifies.
            for i, force in enumerate(min_foot_forces or []):
                target_list = results.diag_r_foot if i % 2 == 0 else results.diag_l_foot
                target_list.append(force)

            # Reward decomposition — all active components
            fwd_norm = float(np.clip(fwd_vel / config.forward_vel_max, -1.0, 1.0))
            results.diag_reward_components["forward"].append(reward_cfg.get("forward_vel_weight", 0.0) * fwd_norm)
            results.diag_reward_components["alive"].append(reward_cfg.get("alive_bonus", 0.0))
            results.diag_reward_components["energy"].append(-reward_cfg.get("energy_penalty_weight", 0.0) * energy)
            posture_reward = _posture_reward_for_eval(
                root_quat,
                config,
                reward_cfg.get("posture_weight", 0.2),
            )
            results.diag_reward_components["posture"].append(posture_reward)

            # Height maintenance
            height_w = reward_cfg.get("height_weight", 0.0)
            if height_w > 0:
                healthy_z_min = config.healthy_z_range[0]
                target_z = config.target_standing_z
                height_frac = float(np.clip((body_z - healthy_z_min) / (target_z - healthy_z_min), 0.0, 1.0))
                results.diag_reward_components["height"].append(height_w * height_frac)
            else:
                results.diag_reward_components["height"].append(0.0)

            # Nosedive penalty
            nosedive_w = reward_cfg.get("nosedive_weight", 0.0)
            root_quat_nd = mj_data.sensordata[config.sensor_quat_start : config.sensor_quat_start + 4]
            w_q, x_q, y_q, z_q = root_quat_nd[0], root_quat_nd[1], root_quat_nd[2], root_quat_nd[3]
            forward_z_nd = float(2.0 * (x_q * z_q - w_q * y_q))
            if nosedive_w > 0:
                nosedive_excess = max(0.0, -(forward_z_nd - config.natural_forward_z))
                results.diag_reward_components["nosedive"].append(-nosedive_w * nosedive_excess)
            else:
                results.diag_reward_components["nosedive"].append(0.0)

            # Drift penalty
            drift_w = reward_cfg.get("drift_penalty_weight", 0.0)
            if drift_w > 0:
                drift_dist = float(np.linalg.norm(mj_data.xpos[config.root_body_id, :2] - start_pos))
                drift_norm = drift_dist / 2.0
                results.diag_reward_components["drift"].append(-drift_w * drift_norm**2)
            else:
                results.diag_reward_components["drift"].append(0.0)

            # Speed penalty
            speed_w = reward_cfg.get("speed_penalty_weight", 0.0)
            if speed_w > 0:
                speed_2d = float(np.linalg.norm(mj_data.qvel[:2]))
                speed_thresh = reward_cfg.get("speed_penalty_threshold", 0.10)
                excess = max(0.0, speed_2d - speed_thresh)
                excess_norm = min(excess / 1.0, 1.0)
                results.diag_reward_components["speed"].append(-speed_w * excess_norm)
            else:
                results.diag_reward_components["speed"].append(0.0)

            # Spin penalty
            spin_w = reward_cfg.get("spin_penalty_weight", 0.0)
            if spin_w > 0:
                gyro = mj_data.sensordata[config.sensor_gyro_start : config.sensor_gyro_start + 3]
                angvel_mag = float(np.linalg.norm(gyro))
                angvel_norm = min(angvel_mag / 10.0, 1.0)
                results.diag_reward_components["spin"].append(-spin_w * angvel_norm)
            else:
                results.diag_reward_components["spin"].append(0.0)

            # Action smoothness
            smoothness_w = reward_cfg.get("smoothness_weight", 0.0)
            if smoothness_w > 0 and prev_action is not None:
                action_delta = float(np.sum(np.square(np.array(action) - np.array(prev_action))))
                max_delta = act_dim * 4.0
                delta_norm = min(action_delta / max_delta, 1.0)
                results.diag_reward_components["smoothness"].append(-smoothness_w * delta_norm)
            else:
                results.diag_reward_components["smoothness"].append(0.0)

            # Per-step state for reward parity with the training env:
            # target position/direction, previous-step target distance
            # (approach shaping), the two previous actions (smoothness and
            # jerk), and success site positions (proximity reward + success
            # bonus).  Passed as
            # kwargs so simple 3-arg reward_fns keep working when absent.
            step_kwargs: dict[str, Any] = {
                "initial_pos_2d": jnp.asarray(start_pos),
                # Pelvis/tilt/nosedive termination is handled inside the
                # canonical scalar reward.  Supply only the extra body/site
                # checks (any-substep) so its fall penalty is applied exactly once.
                "additional_terminated": jnp.asarray(strike_terminated),
            }
            if min_foot_forces:
                # Score the substep-MIN forces, not the boundary snapshot's --
                # eval must price the same quantity the training loop does.
                # Truthiness, not `is not None`: with no foot sensors declared
                # the list is empty, and a shape-(0,) override would replace
                # the composer's own foot-force derivation with nothing.
                step_kwargs["aggregated_foot_forces"] = jnp.asarray(min_foot_forces, dtype=jnp.float32)
            if _target_body_id >= 0:
                target_pos_np = np.array(mj_data.xpos[_target_body_id])
                pelvis_np = np.array(mj_data.xpos[config.root_body_id])
                step_kwargs["target_pos"] = jnp.asarray(target_pos_np)
                step_kwargs["prev_target_distance"] = jnp.asarray(prev_target_dist)
                rel_2d = target_pos_np[:2] - pelvis_np[:2]
                rel_norm = float(np.linalg.norm(rel_2d))
                if rel_norm > 1e-8:
                    # Training projects velocity onto the current
                    # agent-to-target direction; mirror it here.
                    step_kwargs["forward_ref_2d"] = jnp.asarray(rel_2d / rel_norm)
            if prev_action is not None:
                step_kwargs["prev_action"] = prev_action
            if prev_prev_action is not None:
                step_kwargs["prev_prev_action"] = prev_prev_action
            if _success_site_ids:
                step_kwargs["success_site_positions"] = jnp.asarray(
                    np.stack([np.array(mj_data.site_xpos[sid]) for sid in _success_site_ids])
                )

            # Reuse a single mjx.put_data call for the post-step data;
            # mj_data has already been advanced by frame_skip mj_step calls.
            post_step_data = mjx.put_data(mj_model, mj_data)
            if reward_components_fn is not None:
                canonical_details = reward_components_fn(
                    post_step_data,
                    action,
                    reward_cfg,
                    **step_kwargs,
                )
                step_count = len(results.diag_fwd_vel)
                for name, value in canonical_details.items():
                    scalar = float(value)
                    if name.startswith("_"):
                        diagnostic_name = name[1:]
                        results.diag_reward_diagnostics.setdefault(diagnostic_name, []).append(scalar)
                        continue

                    component_values = results.diag_reward_components.setdefault(name, [])
                    # Legacy diagnostics above have already appended this
                    # step for their fixed set of keys.  Replace those
                    # estimates with the canonical implementation; append
                    # newly introduced components such as stance rewards.
                    if len(component_values) == step_count:
                        component_values[-1] = scalar
                    else:
                        component_values.append(scalar)
            r = float(reward_fn(post_step_data, action, reward_cfg, **step_kwargs))
            ep_reward += r

            # Update AFTER the reward call: the training env compares the
            # current action/distance against the previous step's values.
            prev_prev_action = prev_action
            prev_action = action
            if _target_body_id >= 0:
                prev_target_dist = float(
                    np.linalg.norm(mj_data.xpos[_target_body_id] - mj_data.xpos[config.root_body_id])
                )

            if body_z < config.healthy_z_range[0] or body_z > config.healthy_z_range[1]:
                break
            if tilt > config.max_tilt_angle:
                break

            # Nosedive termination (excessive forward pitch)
            root_quat = mj_data.sensordata[config.sensor_quat_start : config.sensor_quat_start + 4]
            w, x, y, z = root_quat[0], root_quat[1], root_quat[2], root_quat[3]
            forward_z = float(2.0 * (x * z - w * y))
            if forward_z < config.natural_forward_z - config.nosedive_threshold:
                break

            # Body/site-height floor-strike termination, any-substep
            if strike_terminated:
                break

            # Stage 3 success: proximity-based contact detection
            if _success_site_ids and _target_body_id >= 0:
                target_pos = mj_data.xpos[_target_body_id]
                if any(
                    float(np.linalg.norm(mj_data.site_xpos[sid] - target_pos)) < config.success_threshold
                    for sid in _success_site_ids
                ):
                    ep_success = True
                    break

        ep_length = step + 1
        distance = float(np.linalg.norm(mj_data.qpos[:2] - start_pos))

        results.rewards.append(ep_reward)
        results.lengths.append(ep_length)
        results.forward_vels.append(float(np.mean(ep_fwd_vels)))
        results.distances.append(distance)
        results.tilt_angles.append(float(np.mean(ep_tilts)))
        results.pelvis_heights.append(float(np.mean(ep_heights)))
        results.successes.append(ep_success)

    return results


def check_stage_gate(
    results: EvalResults,
    gate_min_reward: float = -float("inf"),
    gate_min_length: float = 0,
    gate_min_forward_vel: float = 0.0,
    gate_min_success_rate: float = 0.0,
) -> tuple[bool, list[str]]:
    """Check the ``reward_and_length/v1`` criteria against evaluation results.

    Prefer :func:`check_stage_gate_for_config`, which reads the stage's
    declared ``gate_kind`` and dispatches.  This function takes four fixed
    thresholds and knows nothing about gate kinds, so calling it directly for
    a stage on any other kind certifies that stage on whichever of the four
    happens to be set — which is how ``jax_setup`` came to pass a zero-action
    statue on a ``stance_quality/v1`` stage and write it into
    ``publication_gate_passed``.

    Args:
        results: Evaluation results.
        gate_min_reward: Minimum average reward to pass.
        gate_min_length: Minimum average episode length to pass.
        gate_min_forward_vel: Minimum average forward velocity to pass
            (stage-2 TOML ``min_avg_forward_vel``; 0 disables).
        gate_min_success_rate: Minimum success rate to pass (stage-3 TOML
            ``min_success_rate``; 0 disables).

    Returns:
        (passed, failures) tuple where failures is a list of failure descriptions.
    """
    from .curriculum.gate_schema import finite_gate_metric

    failures = []

    # Every comparison below goes through `finite_gate_metric` first, because
    # an unmeasured metric used to CLEAR its threshold rather than fail it:
    # `nan < gate_min_reward` is False, so an evaluation that produced no
    # usable episodes -- a crashed rollout, an ASHA-pruned trial, a stage that
    # ended before its first evaluation -- returned `(True, [])` and
    # `jax_setup` wrote that straight into `publication_gate_passed`.
    # Reproduced against all four thresholds at once before the guard existed.
    # A threshold nothing can fail (an unset -inf rail, a zero floor) asserts
    # nothing, so it stays skipped exactly as before -- "we could not measure
    # it" is only a refusal where a real threshold was declared. That keeps
    # this change confined to the declared thresholds, which is where the
    # fail-open actually mattered.
    # `asserts_nothing` is per-metric because the "unset" value is: the reward
    # rail is unset at -inf, the other three at 0. Both readings are unchanged
    # from the original comparisons -- no length is below 0 and no reward
    # below -inf -- so only genuinely declared thresholds change behaviour.
    checks: list[tuple[str, Any, float, str, bool]] = [
        ("mean reward", results.mean_reward, float(gate_min_reward), ".2f", gate_min_reward == -float("inf")),
        ("mean episode length", results.mean_length, float(gate_min_length), ".1f", gate_min_length <= 0),
        ("mean forward vel", results.mean_forward_vel, float(gate_min_forward_vel), ".2f", gate_min_forward_vel <= 0),
        ("success rate", results.mean_success_rate, float(gate_min_success_rate), ".2f", gate_min_success_rate <= 0),
    ]
    for name, raw, threshold, fmt, asserts_nothing in checks:
        if asserts_nothing:
            continue
        measured = finite_gate_metric(raw)
        if measured is None:
            failures.append(f"no {name} measurement available to check the {threshold:{fmt}} threshold")
        elif measured < threshold:
            failures.append(f"{name} {measured:{fmt}} < {threshold}")
    return len(failures) == 0, failures


#: Contact force above which a foot counts as bearing load, in newtons.
#: Deliberately :func:`~environments.shared.stance_diagnostics.derive_stance_info`'s
#: default rather than the reward's 42 N ``min_support_force``: every
#: calibration point the stance gate's ceiling rests on is measured at 0.1 N
#: (the statue's 0.000 and run ``20260801_021545``'s 0.319), and switching
#: thresholds here would shift duty upward and invalidate the 0.02 ceiling.
STANCE_CONTACT_THRESHOLD_N = 0.1


def stance_panel_from_eval_results(results: "EvalResults", *, horizon: int, settle_steps: int):
    """Reduce CPU-eval foot traces into the stance gate's panel, or explain why not.

    Returns ``(panel, reason)`` — exactly one is ``None``.

    ``diag_r_foot`` / ``diag_l_foot`` are flat per-step lists spanning every
    episode, so per-episode boundaries are recovered from ``cumsum(lengths)``.
    That reconstruction is only valid when each step contributed exactly one
    reading per side, which is the **biped** case: ``evaluate_policy_cpu``
    routes foot sensors by ``i % 2``, so a species with four foot sensors
    appends two readings per side per step and the alternation no longer
    means right/left (see the ``diag_r_foot``/``diag_l_foot`` interleaving
    defect in KNOWN_ISSUES).

    The length check below detects exactly that: a biped gives
    ``len(diag_r_foot) == sum(lengths)``, a quadruped twice that. It is a
    measurement of the data in hand rather than a species allow-list, so it
    cannot go stale when a species is added.
    """
    from .curriculum.stance_gate import episode_unsupported_duty, stance_panel_from_episode_duties

    lengths = [int(value) for value in results.lengths]
    if not lengths:
        return None, "the evaluation produced no episodes"

    total_steps = sum(lengths)
    right = list(results.diag_r_foot)
    left = list(results.diag_l_foot)
    if not right or not left:
        return None, "the evaluation collected no foot-contact traces (diag_r_foot/diag_l_foot are empty)"
    if len(right) != total_steps or len(left) != total_steps:
        return None, (
            f"foot traces do not align with episode boundaries: {len(right)} right and {len(left)} left "
            f"readings for {total_steps} steps across {len(lengths)} episodes. Per-episode duty cannot be "
            "reconstructed. A ratio of 2 is the known diag_r_foot/diag_l_foot interleaving defect, which "
            "makes this gate unusable for species with four foot sensors until it is fixed"
        )

    duties: list[float | None] = []
    cursor = 0
    for length in lengths:
        window = slice(cursor, cursor + length)
        unsupported = [
            float(
                max(right_force, 0.0) <= STANCE_CONTACT_THRESHOLD_N
                and max(left_force, 0.0) <= STANCE_CONTACT_THRESHOLD_N
            )
            for right_force, left_force in zip(right[window], left[window])
        ]
        duties.append(episode_unsupported_duty(unsupported, settle_steps=settle_steps))
        cursor += length

    return (
        stance_panel_from_episode_duties(
            episode_lengths=[float(value) for value in lengths],
            episode_duties=duties,
            episode_rewards=[float(value) for value in results.rewards],
            horizon=horizon,
        ),
        None,
    )


def check_stage_gate_for_config(
    results: "EvalResults",
    stage_config: dict[str, Any],
) -> "tuple[bool, list[str]]":
    """Check a stage's declared gate against CPU-eval results.

    The gate-kind-aware entry point.  :func:`check_stage_gate` below reads
    four fixed thresholds and knows nothing about ``gate_kind``; calling it
    for a ``stance_quality/v1`` stage silently certified the stage on
    whichever of those four happened to be set. On trex stage 1 that is
    ``min_avg_reward = 1950`` alone — a rail the zero-action statue clears at
    3271.8 and the chattering policy the gate exists to reject clears at
    2133.4 — and ``jax_setup`` writes the verdict straight into
    ``publication_gate_passed``.

    A gate kind whose criteria cannot be evaluated from *results* fails
    closed with the reason, rather than passing on the subset that can.
    That includes every schema-valid kind the JAX path has no evaluator for
    (``recovery_quality/v1`` today): it declares none of the four scalar
    thresholds, so falling through to :func:`check_stage_gate` certified a
    policy that fell in every pushed episode as ``(True, [])`` and
    ``jax_setup`` wrote that into ``publication_gate_passed``.  The refusal
    is the same predicate and message as the in-training curriculum's
    (:func:`~environments.shared.jax_curriculum.unevaluable_gate_kind_reason`),
    so the two JAX paths cannot disagree about which kinds they refuse; here
    the reason is the verdict, so it survives into ``gate_failures``.

    Raises:
        GateSchemaError: If the gate declaration is missing, unknown or
            malformed — including ``none/v1``, which the schema rejects
            outright for an advancing run rather than letting a declared
            pilot reach a verdict here.
    """
    from .curriculum.gate_schema import validate_gate_config
    from .curriculum.stance_gate import STANCE_GATE_KIND, StanceGateThresholds, evaluate_stance_gate
    from .jax_curriculum import unevaluable_gate_kind_reason

    curriculum = stage_config.get("curriculum_kwargs", {})
    stage = stage_config.get("stage", "?")
    gate_kind = validate_gate_config(stage, curriculum, advancement_enabled=True)

    # Schema-valid is not judgeable: refuse, with the reason as the verdict,
    # BEFORE any arm below can read a threshold that is not there.
    refusal = unevaluable_gate_kind_reason(stage, gate_kind)
    if refusal is not None:
        return False, [refusal]

    if gate_kind == STANCE_GATE_KIND:
        horizon = int(stage_config.get("env_kwargs", {}).get("max_episode_steps", 1000))
        thresholds = StanceGateThresholds.from_curriculum(curriculum)
        panel, reason = stance_panel_from_eval_results(
            results,
            horizon=horizon,
            settle_steps=thresholds.settle_steps,
        )
        if panel is None:
            return False, [
                f"stance_quality/v1 cannot be evaluated from this evaluation: {reason}. "
                "Refusing to certify the stage on the reward rail alone, which a zero-action "
                "statue clears."
            ]
        return evaluate_stance_gate(panel, thresholds)

    return check_stage_gate(
        results,
        curriculum.get("min_avg_reward", -float("inf")),
        curriculum.get("min_avg_episode_length", 0),
        gate_min_forward_vel=curriculum.get("min_avg_forward_vel", 0.0),
        gate_min_success_rate=curriculum.get("min_success_rate", 0.0),
    )
