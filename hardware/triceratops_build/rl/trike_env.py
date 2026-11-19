"""Gymnasium environment: the 60 cm Triceratops build model walking forward at a commanded speed.

Simulation only. The robot is ``../sim/generated/trike60_C_closed.xml`` (or its passive-roll-ankle variant), built
by ``../sim/build_model60.py`` (model revision 3, about 12.7 kg; everything here is read from the model file), 1 ms
physics step, 12 position-servo leg drives plus four small head and tail servos. This module is standalone (it does
not subclass ``environments.shared.base_env.BaseDinoEnv``); the README says why.

Task, per episode: walk forward along the heading the robot had at reset (+x of the robot frame), at a commanded
speed drawn from ``[0, 0.27]`` m/s (the museum target is about 0.25 m/s), and hold that heading.

Control: 50 Hz (``frame_skip`` 20 physics steps of 1 ms). The policy outputs 12 numbers in [-1, 1], one per leg
drive (order: ``LEG_ACTUATORS``). They pass a first-order low-pass filter (``action_filter_hz``; part of the policy
interface, to be run on the robot too) and are read as residual joint targets around the neutral standing pose:
``target = neutral + filtered_action * scale[axis]``, clipped to the actuator's ctrlrange. The neck and tail servos
are held at their neutral target (0). The leg targets pass through a per-physics-step FIFO, so a new target reaches
``data.ctrl`` exactly ``k`` physics steps (``k`` ms) after it was set; ``k`` is drawn per episode from
``delay_steps_range`` or fixed by ``delay_steps_fixed``.

Everything random (domain randomisation, delay, command, reset noise, observation noise, heading drift) is drawn from
the env's ``np_random``, so an episode is a function of its reset seed.
"""

from __future__ import annotations

import dataclasses
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import gymnasium as gym
import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE.parent / "sim" / "generated"
DEFAULT_MODEL = MODEL_DIR / "trike60_C_closed.xml"
ANKLE_ROLL_MODEL = MODEL_DIR / "trike60_C_closed_ankleroll.xml"
MODELS = {"trike60_C_closed": DEFAULT_MODEL, "trike60_C_closed_ankleroll": ANKLE_ROLL_MODEL}

CORNERS = ("left_fore", "right_fore", "left_hind", "right_hind")
AXES = ("roll", "pitch", "flexion")
LEG_ACTUATORS = tuple(f"{c}_{a}" for c in CORNERS for a in AXES)
MUSEUM_TARGET_SPEED_MPS = 0.25

REWARD_TERMS = ("r_track", "r_track_linear", "r_alive", "r_upright", "r_height", "r_lateral", "r_yaw_rate",
                "r_heading", "r_energy", "r_action_rate", "r_slip", "r_diagonal", "r_heat", "r_saturation",
                "r_air_time")
REWARD_BONUSES = ("r_track", "r_alive")  # >= 0; r_diagonal and r_air_time may take either sign; every other is <= 0
RAMPED_TERMS = ("r_energy", "r_slip", "r_heat", "r_saturation")  # scaled by penalty_scale (train.py's ramp)
DRIVE_GROUPS = ("roll", "pitch", "front elbow", "hind knee")


@dataclass
class TrikeEnvConfig:
    """Every setting of the environment, with its default. All ranges are closed intervals ``(low, high)``.

    The reward weights, action scales and filter are untuned starting points, not results."""

    model_path: str = str(DEFAULT_MODEL)

    # --- timing ---
    frame_skip: int = 20  # physics steps per control step: 20 x 1 ms = 50 Hz control
    episode_seconds: float = 20.0  # truncation (time limit)

    # --- command ---
    speed_range: tuple[float, float] = (0.0, 0.27)  # m/s, drawn uniformly per episode
    command_zero_prob: float = 0.1  # share of episodes commanded to stand (speed exactly 0)

    # --- action ---
    # Residual joint target per unit action, rad, by drive axis. Targets are also clipped to each ctrlrange.
    action_scale_roll: float = 0.15
    action_scale_pitch: float = 0.40
    action_scale_flexion: float = 0.40
    # First-order low-pass on the action before it becomes a target (y += alpha * (a - y) every control step,
    # alpha = 1 - exp(-2 pi f_c dt) = 0.395 at 4 Hz and 50 Hz), cutoff in Hz (0: off). A 1.5 Hz gait passes at 0.94
    # of its amplitude but about 29 ms late, three times the 10 ms servo delay; step-to-step jitter (25 Hz) passes at
    # 0.25, white noise at 0.50 of its standard deviation. Part of the policy interface: the robot must run this
    # identical filter at the identical 50 Hz control rate.
    action_filter_hz: float = 4.0

    # --- servo delay (per-physics-step FIFO of the 12 leg targets) ---
    delay_steps_range: tuple[int, int] = (0, 10)  # physics steps (= ms at 1 ms), drawn uniformly per episode
    delay_steps_fixed: int | None = None  # when set, every episode uses this delay (evaluation: 10)

    # --- domain randomisation (per episode, from saved nominal copies) ---
    domain_randomization: bool = True  # False: the model is exactly the nominal model every episode
    friction_range: tuple[float, float] = (0.6, 1.0)  # sliding friction of the floor and the four soles (one draw)
    mass_scale_range: tuple[float, float] = (0.9, 1.1)  # per body, mass and inertia scaled together
    kp_scale_range: tuple[float, float] = (0.9, 1.1)  # per actuator, position gain (gainprm[0] and biasprm[1])
    force_limit_scale_range: tuple[float, float] = (0.85, 1.0)  # per actuator, forcerange

    # --- observation noise (Gaussian, standard deviations) ---
    observation_noise: bool = True
    noise_joint_pos: float = 0.005  # rad (leg joints and ankle angles)
    noise_joint_vel: float = 0.05  # rad/s
    noise_gyro: float = 0.02  # rad/s
    noise_orientation: float = 0.01  # per quaternion component, then renormalised (about 1 deg)
    noise_lin_vel: float = 0.02  # m/s, on the privileged base velocity
    noise_foot_load: float = 0.02  # on the load / weight channels (2 % of body weight, about 2.5 N)
    # The robot has no magnetometer (parts list), so its heading is a gyro-integrated estimate that drifts. The
    # heading observation carries a random walk of this intensity, rad per sqrt(s) (0: off), reset to 0 each episode:
    # about 2.6 deg standard deviation after 20 s. The heading REWARD uses the true yaw.
    heading_drift_rw: float = 0.01

    # --- observation content ---
    obs_ankle_angles: bool = True  # the 4 passive ankle pitch angles (the model has jointpos sensors on them)
    obs_privileged_lin_vel: bool = True  # base linear velocity from qvel (simulation only; no sensor gives it)
    # sin/cos gait clock in the observation (Hz); 0 removes it. Off by default since 20 Nov 2026: no reward term uses
    # it and the pipe1 policy ignored it (it stepped at 2.39 Hz against the clock's 1.5 Hz). Runs that trained with it
    # (trike60-pipe1: 1.5) keep it through the env config in their run_record.json, so their checkpoints still load.
    phase_clock_hz: float = 0.0

    # --- reset ---
    reset_joint_noise: float = 0.02  # rad, uniform, on the 12 leg joints
    reset_sole_clearance: float = 0.002  # m: the root is placed so the lowest sole corner is this far above floor

    # --- termination ---
    max_tilt_deg: float = 35.0
    min_height_frac: float = 0.6  # of the nominal standing chassis height
    terminate_on_body_contact: bool = True  # any geom other than the four soles touching the floor

    # --- reward weights (per control step; untuned starting points) ---
    w_track: float = 1.5  # x exp(-(v_fwd - v_cmd)^2 / track_sigma^2)
    track_sigma: float = 0.06  # m/s (22 % of the command range)
    w_track_linear: float = 0.8  # x |v_fwd - v_cmd|: a slope far from the command, where the exp term is flat
    w_alive: float = 0.3
    w_upright: float = 1.0  # x (gx^2 + gy^2), gravity direction in the body frame
    w_height: float = 20.0  # x (z - z_nominal)^2, m^2
    w_lateral: float = 2.0  # x v_lat^2
    w_yaw_rate: float = 0.1  # x yaw_rate^2
    w_heading: float = 0.5  # x heading_error^2 (rad^2, true yaw)
    w_energy: float = 0.001  # x mean over the step of sum |torque x joint speed| (W) of the 12 leg drives
    w_action_rate: float = 0.04  # x sum (a - a_prev)^2 of the raw policy outputs
    w_slip: float = 1.0  # x sum over feet in contact of the sole's horizontal speed^2 (m^2/s^2); 0.5 until pipe1
    w_diagonal: float = 0.0  # optional diagonal (trot) contact symmetry; off by default
    termination_penalty: float = 5.0
    contact_threshold_n: float = 2.0  # a foot is "in contact" above this touch load

    # --- drive heat and torque limits (penalties ramped with energy and slip; see RAMPED_TERMS) ---
    # Continuous torque ratings at the joint, N m: the mass ledger's joint_thermal_eq_nm (X4 roll and pitch 7.5,
    # X6 front elbow 10.8, X4 hind knee through its 1:1 crank 6.75); a test holds them to the ledger.
    cont_torque_roll: float = 7.5
    cont_torque_pitch: float = 7.5
    cont_torque_fore_flexion: float = 10.8
    cont_torque_hind_flexion: float = 6.75
    # x mean over the 12 leg drives and the step's physics steps of (torque / continuous rating)^2. Standing still
    # scores about 0.06 (cost 0.03); the pipe1 gait about 0.9 (cost about 0.45); every drive exactly at its rating, 1.
    w_heat: float = 0.5
    saturation_frac: float = 0.95  # a drive "saturates" at this fraction of its (randomised) force limit
    # x fraction of the 12 drives x physics steps that saturate, against the EPISODE'S randomised force limit (that
    # is the limit the drive would hit; 0.85-1.0 x nominal under domain randomisation)
    w_saturation: float = 5.0  # the pipe1 gait: 2.9 % saturated, cost about 0.14; standing still: 0

    # --- clean steps: debounced contact and air time at touchdown ---
    # A foot counts as lifted only after its load has stayed at or below contact_threshold_n for this long. The
    # ~2 ms breaks seen within pipe1's stances are a sub-millimetre contact-model rebound (sole lowest corner median
    # 0.03 mm above the floor, carrying about 0 N), not steps; debouncing merges them into the stance. They are not
    # penalised: that would train against the contact solver.
    contact_debounce_s: float = 0.010
    # legged_gym-style air-time reward, at each debounced touchdown: + w_air_time x (air time - air_time_target_s),
    # summed over feet, only when the command exceeds air_time_min_command. 0.20 s: at 0.25 m/s a stride of 10-12 cm
    # takes about 0.45 s; with a trot's duty factor near 0.6 the swing lasts about 0.18 s (pipe1's real swings:
    # 140-215 ms). Swings shorter than 0.2 s (shuffles, hops) cost, longer ones pay.
    air_time_target_s: float = 0.20
    w_air_time: float = 1.0
    air_time_min_command: float = 0.05

    def replace(self, **changes: Any) -> "TrikeEnvConfig":
        return dataclasses.replace(self, **changes)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TrikeEnvConfig":
        names = {f.name for f in dataclasses.fields(cls)}
        clean = {}
        for k, v in d.items():
            if k not in names:
                raise ValueError(f"unknown TrikeEnvConfig field {k!r}")
            clean[k] = tuple(v) if isinstance(v, list) else v
        return cls(**clean)


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def drive_group(actuator: str) -> str:
    """'roll', 'pitch', 'front elbow' or 'hind knee' for a leg drive's name."""
    if actuator.endswith("_roll"):
        return "roll"
    if actuator.endswith("_pitch"):
        return "pitch"
    return "front elbow" if "_fore_" in actuator else "hind knee"


def continuous_rating(cfg: "TrikeEnvConfig", actuator: str) -> float:
    """A leg drive's continuous torque rating at the joint, N m (``cont_torque_*``)."""
    return {"roll": cfg.cont_torque_roll, "pitch": cfg.cont_torque_pitch, "front elbow": cfg.cont_torque_fore_flexion,
            "hind knee": cfg.cont_torque_hind_flexion}[drive_group(actuator)]


def diagonal_score(contact) -> float:
    """+1 for a trot (diagonal pairs together, left and right apart), -1 for a bound (front pair and hind pair each
    together, diagonals apart), 0 for a pace, a pronk, all four down or all four up. Order: ``CORNERS``."""
    c = [bool(x) for x in contact]
    return 0.5 * (float(c[0] == c[3]) + float(c[1] == c[2])) - 0.5 * (float(c[0] == c[1]) + float(c[2] == c[3]))


def reward_terms(cfg: TrikeEnvConfig, *, v_fwd: float, v_cmd: float, v_lat: float, yaw_rate: float, yaw_err: float,
                 grav_b, z: float, z_nominal: float, mean_power: float, action, prev_action, slip: float, contact,
                 penalty_scale: float = 1.0, heat_index: float = 0.0, saturated_fraction: float = 0.0,
                 air_time_excess: float = 0.0) -> dict[str, float]:
    """One control step's reward terms (``REWARD_TERMS``), a pure function of the measured quantities.

    ``penalty_scale`` (0-1) multiplies the energy, slip, heat and saturation penalties (``RAMPED_TERMS``): train.py
    ramps it up over the first steps so that early exploration is not punished into standing still.
    ``heat_index``: mean of (torque / continuous rating)^2 over the leg drives and physics steps; ``saturated_fraction``:
    share of drive-physics-steps at or above ``saturation_frac`` of the episode's (randomised) force limit;
    ``air_time_excess``: sum over this step's debounced touchdowns of (air time - ``air_time_target_s``)."""
    err = v_fwd - v_cmd
    a, ap = np.asarray(action, dtype=np.float64), np.asarray(prev_action, dtype=np.float64)
    terms = {
        "r_track": cfg.w_track * math.exp(-((err / cfg.track_sigma) ** 2)),
        "r_track_linear": -cfg.w_track_linear * abs(err),
        "r_alive": cfg.w_alive,
        "r_upright": -cfg.w_upright * float(grav_b[0] ** 2 + grav_b[1] ** 2),
        "r_height": -cfg.w_height * (z - z_nominal) ** 2,
        "r_lateral": -cfg.w_lateral * v_lat ** 2,
        "r_yaw_rate": -cfg.w_yaw_rate * yaw_rate ** 2,
        "r_heading": -cfg.w_heading * yaw_err ** 2,
        "r_energy": -cfg.w_energy * penalty_scale * mean_power,
        "r_action_rate": -cfg.w_action_rate * float(np.sum((a - ap) ** 2)),
        "r_slip": -cfg.w_slip * penalty_scale * slip,
        "r_diagonal": 0.0,
        "r_heat": -cfg.w_heat * penalty_scale * heat_index,
        "r_saturation": -cfg.w_saturation * penalty_scale * saturated_fraction,
        "r_air_time": cfg.w_air_time * air_time_excess if v_cmd > cfg.air_time_min_command else 0.0,
    }
    if cfg.w_diagonal and v_cmd > 0.05:
        terms["r_diagonal"] = cfg.w_diagonal * diagonal_score(contact)
    return terms


class TrikeEnv(gym.Env):
    """The 60 cm Triceratops walking task. See the module docstring; settings in :class:`TrikeEnvConfig`."""

    metadata = {"render_modes": [], "render_fps": 50}

    def __init__(self, config: TrikeEnvConfig | None = None, **overrides: Any):
        super().__init__()
        cfg = config if config is not None else TrikeEnvConfig()
        if overrides:
            cfg = cfg.replace(**overrides)
        self.cfg = cfg
        self.model = mujoco.MjModel.from_xml_path(str(cfg.model_path))
        self.data = mujoco.MjData(self.model)
        m = self.model

        self.physics_dt = float(m.opt.timestep)
        self.dt = self.physics_dt * cfg.frame_skip
        self.max_episode_steps = int(round(cfg.episode_seconds / self.dt))
        self._filter_alpha = (1.0 - math.exp(-2.0 * math.pi * cfg.action_filter_hz * self.dt)
                              if cfg.action_filter_hz > 0 else 1.0)
        self.penalty_scale = 1.0  # see reward_terms; set by train.py through set_penalty_scale

        # --- ids ---
        self._leg_act = np.array([m.actuator(n).id for n in LEG_ACTUATORS], dtype=np.int64)
        self._other_act = np.array([a for a in range(m.nu) if a not in set(self._leg_act.tolist())], dtype=np.int64)
        leg_jnt = m.actuator_trnid[self._leg_act, 0]
        for a, j in zip(LEG_ACTUATORS, leg_jnt):
            if m.joint(int(j)).name != a:
                raise ValueError(f"actuator {a} drives joint {m.joint(int(j)).name}, expected a joint named {a}")
        self._leg_qadr = m.jnt_qposadr[leg_jnt].astype(np.int64)
        self._leg_dadr = m.jnt_dofadr[leg_jnt].astype(np.int64)
        self._ankle_qadr = np.array([m.jnt_qposadr[m.joint(f"{c}_ankle").id] for c in CORNERS], dtype=np.int64)
        key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
        self._key = int(key) if key >= 0 else 0
        self._chassis = m.body("chassis").id
        self._floor = m.geom("floor").id
        self._soles = np.array([m.geom(f"{c}_sole").id for c in CORNERS], dtype=np.int64)
        self._is_sole = np.zeros(m.ngeom, dtype=bool)
        self._is_sole[self._soles] = True
        # Lookup table over contact pairs (g1 * ngeom + g2): True for floor with any geom but a sole. One matmul and
        # one gather per physics step instead of half a dozen small-array numpy calls.
        not_sole = ~self._is_sole
        not_sole[self._floor] = False
        table = np.zeros((m.ngeom, m.ngeom), dtype=bool)
        table[self._floor, not_sole] = True
        table[not_sole, self._floor] = True
        self._bad_pair = table.ravel()
        self._pair_weight = np.array([m.ngeom, 1], dtype=np.int32)

        def sensor_adr(name: str) -> int:
            return int(m.sensor_adr[m.sensor(name).id])

        self._quat_adr = sensor_adr("imu_orientation")
        self._gyro_adr = sensor_adr("imu_angular_velocity")
        self._touch_adr = np.array([sensor_adr(f"{c}_normal_load") for c in CORNERS], dtype=np.int64)

        # --- action scaling ---
        scale_by_axis = {"roll": cfg.action_scale_roll, "pitch": cfg.action_scale_pitch,
                         "flexion": cfg.action_scale_flexion}
        self._action_scale = np.array([scale_by_axis[a] for _ in CORNERS for a in AXES])
        self._ctrl_lo = m.actuator_ctrlrange[self._leg_act, 0].copy()
        self._ctrl_hi = m.actuator_ctrlrange[self._leg_act, 1].copy()
        self._ctrl_neutral = m.key_ctrl[self._key, self._leg_act].copy() if m.nkey else np.zeros(12)
        self._other_neutral = (m.key_ctrl[self._key, self._other_act].copy() if m.nkey
                               else np.zeros(len(self._other_act)))
        self._q_neutral = m.key_qpos[self._key, self._leg_qadr].copy() if m.nkey else np.zeros(12)
        leg_slice = (self._leg_act == np.arange(self._leg_act[0], self._leg_act[0] + 12)).all()
        self._leg_ctrl_index: Any = (slice(int(self._leg_act[0]), int(self._leg_act[0]) + 12) if leg_slice
                                     else self._leg_act)

        # --- nominal copies for domain randomisation ---
        self._nominal = {
            "geom_friction": m.geom_friction.copy(),
            "body_mass": m.body_mass.copy(),
            "body_inertia": m.body_inertia.copy(),
            "actuator_gainprm": m.actuator_gainprm.copy(),
            "actuator_biasprm": m.actuator_biasprm.copy(),
            "actuator_forcerange": m.actuator_forcerange.copy(),
        }
        self._pos_servo = (m.actuator_gaintype == mujoco.mjtGain.mjGAIN_FIXED) & (
            m.actuator_biastype == mujoco.mjtBias.mjBIAS_AFFINE)
        # drive heat and torque limits; debounced contact
        self.continuous_torque = np.array([continuous_rating(cfg, n) for n in LEG_ACTUATORS])
        self._inv_cont2 = 1.0 / self.continuous_torque ** 2
        self._sat_limit = cfg.saturation_frac * m.actuator_forcerange[self._leg_act, 1]
        self._debounce_steps = max(1, int(round(cfg.contact_debounce_s / self.physics_dt)))
        self.nominal_mass = float(m.body_mass.sum())
        self.nominal_weight = self.nominal_mass * float(-m.opt.gravity[2])
        self.nominal_height = self._measure_standing_height()
        self.factors: dict[str, Any] = {}

        # --- spaces ---
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(12,), dtype=np.float32)
        self.obs_layout = self._obs_layout()
        self._reset_episode_state(command=0.0, delay=0)
        mujoco.mj_resetDataKeyframe(m, self.data, self._key)
        mujoco.mj_forward(m, self.data)
        obs = self._get_obs(noise=False)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=obs.shape, dtype=np.float32)

        # Called after every physics step (tests and diagnostics); data.ctrl then holds the ctrl of that step.
        self.substep_hook: Callable[["TrikeEnv"], None] | None = None

    # ------------------------------------------------------------------ setup helpers
    def _obs_layout(self) -> tuple[tuple[str, int], ...]:
        cfg = self.cfg
        parts = [("leg_joint_pos", 12), ("leg_joint_vel", 12)]
        if cfg.obs_ankle_angles:
            parts.append(("ankle_pitch_pos", 4))
        parts += [("gravity_body", 3), ("heading_error_sin_cos_ESTIMATE", 2), ("gyro", 3)]
        if cfg.obs_privileged_lin_vel:
            parts.append(("base_lin_vel_body_PRIVILEGED", 3))
        # both: the action-rate penalty is on the raw outputs, the targets follow the filtered ones
        parts += [("foot_load_over_weight", 4), ("previous_action_filtered", 12), ("previous_action_raw", 12),
                  ("command_speed_mps", 1)]
        if cfg.phase_clock_hz > 0:
            parts.append(("phase_sin_cos", 2))
        return tuple(parts)

    def obs_slots(self, obs) -> dict[str, np.ndarray]:
        """An observation split into its named slots (``obs_layout``)."""
        out, i = {}, 0
        for name, width in self.obs_layout:
            out[name] = np.asarray(obs[i:i + width])
            i += width
        return out

    def _measure_standing_height(self) -> float:
        """Chassis height after 0.5 s standing at the neutral pose on the nominal model (ctrl at the keyframe's)."""
        m, d = self.model, mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(m, d, self._key)
        for _ in range(int(round(0.5 / self.physics_dt))):
            mujoco.mj_step(m, d)
        return float(d.xpos[self._chassis, 2])

    def _reset_episode_state(self, command: float, delay: int) -> None:
        self.command_speed = float(command)
        self.delay_steps = int(delay)
        self._step_count = 0
        self._prev_action = np.zeros(12)
        self._action_filtered = np.zeros(12)
        self._heading = 0.0
        self._heading_drift = 0.0
        self._distance = 0.0
        self._target = self._ctrl_neutral.copy()
        self._fifo: deque = deque([self._target] * self.delay_steps, maxlen=self.delay_steps + 1)
        self._badqacc0 = 0
        self._lifted = np.zeros(4, dtype=bool)  # debounced: lifted only after contact_debounce_s unloaded
        self._unloaded = np.zeros(4, dtype=np.int64)  # consecutive physics steps each foot has been unloaded
        self.touchdowns = 0  # debounced touchdowns this episode (diagnostic)

    def set_penalty_scale(self, scale: float) -> None:
        """Scale (0-1) of the ramped penalties (``RAMPED_TERMS``); train.py ramps it over the first training steps."""
        self.penalty_scale = float(min(1.0, max(0.0, scale)))

    # ------------------------------------------------------------------ domain randomisation
    def _restore_nominal(self) -> None:
        m, n = self.model, self._nominal
        m.geom_friction[:] = n["geom_friction"]
        m.body_mass[:] = n["body_mass"]
        m.body_inertia[:] = n["body_inertia"]
        m.actuator_gainprm[:] = n["actuator_gainprm"]
        m.actuator_biasprm[:] = n["actuator_biasprm"]
        m.actuator_forcerange[:] = n["actuator_forcerange"]

    def _draw_factors(self, rng: np.random.Generator) -> dict[str, Any]:
        """One episode's physics factors (a fixed number of draws in a fixed order, whatever the config)."""
        cfg, m = self.cfg, self.model
        return {
            "friction": float(rng.uniform(*cfg.friction_range)),
            "mass_scale": rng.uniform(*cfg.mass_scale_range, size=m.nbody),
            "kp_scale": rng.uniform(*cfg.kp_scale_range, size=m.nu),
            "force_limit_scale": rng.uniform(*cfg.force_limit_scale_range, size=m.nu),
        }

    def _apply_factors(self, f: dict[str, Any]) -> None:
        """The model back to nominal, then the factors (``f`` empty: nominal), then ``mj_setConst``."""
        m, n = self.model, self._nominal
        self._restore_nominal()
        if f:
            mu = f["friction"]
            m.geom_friction[self._floor, 0] = mu
            m.geom_friction[self._soles, 0] = mu
            s = np.array(f["mass_scale"], dtype=np.float64)
            s[0] = 1.0  # the world body
            m.body_mass[:] = n["body_mass"] * s
            m.body_inertia[:] = n["body_inertia"] * s[:, None]
            kp = np.where(self._pos_servo, f["kp_scale"], 1.0)
            # position actuator: force = kp*ctrl - kp*q - kv*qdot  ->  gainprm[0] = kp, biasprm[1] = -kp
            m.actuator_gainprm[:, 0] = n["actuator_gainprm"][:, 0] * kp
            m.actuator_biasprm[:, 1] = n["actuator_biasprm"][:, 1] * kp
            m.actuator_forcerange[:] = n["actuator_forcerange"] * np.asarray(f["force_limit_scale"])[:, None]
        # Constants derived from masses (subtree masses, invweights, ...) recomputed; bit-identical when nominal.
        mujoco.mj_setConst(m, self.data)

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        cfg, m, d = self.cfg, self.model, self.data
        rng = self.np_random
        opts = options or {}

        # 1. physics factors: drawn every reset (fixed draw count keeps the stream aligned), applied if enabled
        drawn = self._draw_factors(rng)
        self.factors = drawn if cfg.domain_randomization else {}
        self._apply_factors(self.factors)
        self._sat_limit = cfg.saturation_frac * m.actuator_forcerange[self._leg_act, 1]  # this episode's limits

        # 2. servo delay
        delay_draw = int(rng.integers(cfg.delay_steps_range[0], cfg.delay_steps_range[1] + 1))
        delay = cfg.delay_steps_fixed if cfg.delay_steps_fixed is not None else delay_draw
        delay = int(opts.get("delay_steps", delay))
        if delay < 0:
            raise ValueError(f"delay_steps must be >= 0, got {delay}")

        # 3. command
        zero = rng.random() < cfg.command_zero_prob
        speed = float(rng.uniform(*cfg.speed_range))
        command = 0.0 if zero else speed
        if "command_speed" in opts:
            command = float(opts["command_speed"])

        # 4. state: home keyframe, leg joint noise, root placed so the lowest sole corner sits at the clearance
        mujoco.mj_resetDataKeyframe(m, d, self._key)
        d.qpos[self._leg_qadr] += rng.uniform(-cfg.reset_joint_noise, cfg.reset_joint_noise, size=12)
        mujoco.mj_kinematics(m, d)
        d.qpos[2] += cfg.reset_sole_clearance - self._lowest_sole_z()
        d.ctrl[:] = 0.0
        d.ctrl[self._other_act] = self._other_neutral
        d.ctrl[self._leg_ctrl_index] = self._ctrl_neutral
        mujoco.mj_forward(m, d)

        self._reset_episode_state(command=command, delay=delay)
        R = d.xmat[self._chassis].reshape(3, 3)
        self._heading = math.atan2(R[1, 0], R[0, 0])
        self._badqacc0 = int(d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number)

        obs = self._get_obs(noise=True)
        info = {"command_speed": self.command_speed, "delay_steps": self.delay_steps,
                "domain_randomization": bool(self.factors)}
        if self.factors:
            info["friction"] = self.factors["friction"]
        return obs, info

    def step(self, action):
        cfg, m, d = self.cfg, self.model, self.data
        a = np.clip(np.asarray(action, dtype=np.float64).reshape(12), -1.0, 1.0)
        if cfg.action_filter_hz > 0:
            self._action_filtered = self._action_filtered + self._filter_alpha * (a - self._action_filtered)
        else:
            self._action_filtered = a
        target = np.clip(self._ctrl_neutral + self._action_filtered * self._action_scale, self._ctrl_lo,
                         self._ctrl_hi)
        self._target = target
        xy0 = d.xpos[self._chassis, :2].copy()

        fifo, idx, hook = self._fifo, self._leg_ctrl_index, self.substep_hook
        energy = heat = 0.0
        saturated = touchdowns = 0
        air_excess = 0.0
        floor_hit = -1
        inv_c2, sat_limit, touch_adr, thr = self._inv_cont2, self._sat_limit, self._touch_adr, cfg.contact_threshold_n
        n_db, dt_phys, t_air = self._debounce_steps, self.physics_dt, cfg.air_time_target_s
        for _ in range(cfg.frame_skip):
            fifo.append(target)
            d.ctrl[idx] = fifo[0]
            mujoco.mj_step(m, d)
            tau = d.actuator_force[idx]
            # |torque x joint speed| of the 12 leg drives (actuator_velocity = joint speed, gear 1)
            energy += float(np.abs(tau * d.actuator_velocity[idx]).sum())
            heat += float(np.dot(tau * tau, inv_c2))
            saturated += int(np.count_nonzero(np.abs(tau) >= sat_limit))
            # debounced contact: a touchdown ends a lift; a break shorter than the debounce time stays in the stance
            raw = d.sensordata[touch_adr] > thr
            prev = self._unloaded
            down_now = raw & self._lifted
            if down_now.any():
                touchdowns += int(down_now.sum())
                air_excess += float((prev[down_now] * dt_phys - t_air).sum())
                self._lifted &= ~down_now
            self._unloaded = np.where(raw, 0, prev + 1)
            self._lifted |= self._unloaded >= n_db
            if floor_hit < 0 and d.ncon:
                floor_hit = self._non_foot_floor_contact()
            if hook is not None:
                hook(self)
        self._step_count += 1
        self.touchdowns += touchdowns
        mean_power = energy / cfg.frame_skip
        heat_index = heat / (12 * cfg.frame_skip)
        saturated_fraction = saturated / (12 * cfg.frame_skip)

        # --- measurements ---
        R = d.xmat[self._chassis].reshape(3, 3)
        xy1 = d.xpos[self._chassis, :2]
        disp = (xy1 - xy0) / self.dt
        h = np.array([math.cos(self._heading), math.sin(self._heading)])
        v_fwd = float(disp @ h)
        v_lat = float(disp @ np.array([-h[1], h[0]]))
        self._distance += v_fwd * self.dt
        yaw = math.atan2(R[1, 0], R[0, 0])
        yaw_err = _wrap(yaw - self._heading)
        yaw_rate = float((R @ d.qvel[3:6])[2])
        grav_b = -R[2, :]
        z = float(d.xpos[self._chassis, 2])
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(R[2, 2])))))
        loads = d.sensordata[self._touch_adr]
        contact = loads > cfg.contact_threshold_n
        slip = 0.0
        vel6 = np.zeros(6)
        for i in np.flatnonzero(contact):
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_GEOM, int(self._soles[i]), vel6, 0)
            slip += float(vel6[3] ** 2 + vel6[4] ** 2)

        terms = reward_terms(cfg, v_fwd=v_fwd, v_cmd=self.command_speed, v_lat=v_lat, yaw_rate=yaw_rate,
                             yaw_err=yaw_err, grav_b=grav_b, z=z, z_nominal=self.nominal_height,
                             mean_power=mean_power, action=a, prev_action=self._prev_action, slip=slip,
                             contact=contact, penalty_scale=self.penalty_scale, heat_index=heat_index,
                             saturated_fraction=saturated_fraction, air_time_excess=air_excess)

        # --- termination ---
        reason = ""
        if not (np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()) or (
                int(d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number) > self._badqacc0):
            reason = "unstable_simulation"
        elif tilt > cfg.max_tilt_deg:
            reason = "tilt"
        elif z < cfg.min_height_frac * self.nominal_height:
            reason = "height"
        elif cfg.terminate_on_body_contact and floor_hit >= 0:
            reason = f"floor_contact:{m.geom(floor_hit).name}"
        terminated = bool(reason)
        truncated = (not terminated) and self._step_count >= self.max_episode_steps

        reward = float(sum(terms.values()))
        if terminated:
            reward -= cfg.termination_penalty
        self._prev_action = a

        # heading estimate: a gyro-integrated yaw drifts (no magnetometer); a random walk on the observation
        if cfg.observation_noise and cfg.heading_drift_rw > 0:
            self._heading_drift += float(self.np_random.normal(0.0, cfg.heading_drift_rw * math.sqrt(self.dt)))
        obs = self._get_obs(noise=True)
        t = self._step_count * self.dt
        info = dict(terms)
        info.update({
            "reward_total": reward, "forward_speed": v_fwd, "lateral_speed": v_lat, "command_speed": self.command_speed,
            "episode_forward_speed": self._distance / t, "yaw_error": yaw_err, "yaw_rate": yaw_rate,
            "tilt_deg": tilt, "height": z, "mean_power_w": mean_power, "foot_contacts": contact.astype(np.int8),
            "delay_steps": self.delay_steps, "termination": reason, "time_s": t, "penalty_scale": self.penalty_scale,
            "heading_drift": self._heading_drift, "heat_index": heat_index, "saturated_fraction": saturated_fraction,
            "touchdowns": touchdowns, "air_time_excess": air_excess,
        })
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------ pieces
    def _lowest_sole_z(self) -> float:
        d, lowest = self.data, np.inf
        signs = np.array([[sx, sy, -1.0] for sx in (-1.0, 1.0) for sy in (-1.0, 1.0)])
        for g in self._soles:
            R = d.geom_xmat[g].reshape(3, 3)
            pts = d.geom_xpos[g] + (signs * self.model.geom_size[g]) @ R.T
            lowest = min(lowest, float(pts[:, 2].min()))
        return lowest

    def _non_foot_floor_contact(self) -> int:
        """The first geom other than a sole touching the floor in the current contacts, or -1."""
        g = self.data.contact.geom
        hits = self._bad_pair[g @ self._pair_weight]
        if not hits.any():
            return -1
        g1, g2 = (int(x) for x in g[int(np.argmax(hits))])
        return g2 if g1 == self._floor else g1

    def _get_obs(self, noise: bool) -> np.ndarray:
        cfg, d = self.cfg, self.data
        rng = self.np_random if (noise and cfg.observation_noise) else None
        q = d.qpos[self._leg_qadr] - self._q_neutral
        qd = d.qvel[self._leg_dadr].copy()
        ankle = d.qpos[self._ankle_qadr].copy()
        quat = d.sensordata[self._quat_adr:self._quat_adr + 4].copy()
        gyro = d.sensordata[self._gyro_adr:self._gyro_adr + 3].copy()
        loads = d.sensordata[self._touch_adr] / self.nominal_weight
        R_true = d.xmat[self._chassis].reshape(3, 3)
        lin_vel = R_true.T @ d.qvel[0:3]
        if rng is not None:
            q = q + rng.normal(0.0, cfg.noise_joint_pos, 12)
            qd = qd + rng.normal(0.0, cfg.noise_joint_vel, 12)
            ankle = ankle + rng.normal(0.0, cfg.noise_joint_pos, 4)
            gyro = gyro + rng.normal(0.0, cfg.noise_gyro, 3)
            quat = quat + rng.normal(0.0, cfg.noise_orientation, 4)
            lin_vel = lin_vel + rng.normal(0.0, cfg.noise_lin_vel, 3)
            loads = loads + rng.normal(0.0, cfg.noise_foot_load, 4)
        quat /= max(np.linalg.norm(quat), 1e-9)
        Rm = np.zeros(9)
        mujoco.mju_quat2Mat(Rm, quat)
        Rm = Rm.reshape(3, 3)
        grav_b = -Rm[2, :]
        yaw_err_est = _wrap(math.atan2(Rm[1, 0], Rm[0, 0]) + self._heading_drift - self._heading)
        parts = [q, qd]
        if cfg.obs_ankle_angles:
            parts.append(ankle)
        parts += [grav_b, [math.sin(yaw_err_est), math.cos(yaw_err_est)], gyro]
        if cfg.obs_privileged_lin_vel:
            parts.append(lin_vel)
        parts += [loads, self._action_filtered, self._prev_action, [self.command_speed]]
        if cfg.phase_clock_hz > 0:
            ph = 2.0 * math.pi * cfg.phase_clock_hz * self._step_count * self.dt
            parts.append([math.sin(ph), math.cos(ph)])
        return np.concatenate([np.asarray(p, dtype=np.float64).ravel() for p in parts]).astype(np.float32)

    # ------------------------------------------------------------------ helpers for tests and tools
    def leg_targets_for(self, filtered_action) -> np.ndarray:
        """The leg ctrl targets a FILTERED action maps to (before the delay FIFO)."""
        a = np.asarray(filtered_action, dtype=np.float64).reshape(12)
        return np.clip(self._ctrl_neutral + a * self._action_scale, self._ctrl_lo, self._ctrl_hi)

    def leg_ctrl(self) -> np.ndarray:
        return self.data.ctrl[self._leg_act].copy()


def make_env(config: TrikeEnvConfig | dict | None = None, monitor: bool = False) -> gym.Env:
    """A fresh env (picklable by reference, for SubprocVecEnv). ``monitor`` wraps it in SB3's Monitor."""
    if isinstance(config, dict):
        config = TrikeEnvConfig.from_dict(config)
    env: gym.Env = TrikeEnv(config)
    if monitor:
        from stable_baselines3.common.monitor import Monitor

        env = Monitor(env, info_keywords=("episode_forward_speed", "command_speed", "termination"))
    return env
