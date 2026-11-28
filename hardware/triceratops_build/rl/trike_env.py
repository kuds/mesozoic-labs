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
CAPSULE_MODEL = MODEL_DIR / "trike60_C_closed_capsule.xml"  # the owner's foot (D19): the foot study's capsule sole
MODELS = {"trike60_C_closed": DEFAULT_MODEL, "trike60_C_closed_ankleroll": ANKLE_ROLL_MODEL,
          "trike60_C_closed_capsule": CAPSULE_MODEL,
          # the foreleg comparison (D22): PR #697's repackaged forelegs, 128 and 160 mm upper links
          "trike60_C_closed_capsule_fore128": MODEL_DIR / "trike60_C_closed_capsule_fore128.xml",
          "trike60_C_closed_capsule_fore160": MODEL_DIR / "trike60_C_closed_capsule_fore160.xml"}

CORNERS = ("left_fore", "right_fore", "left_hind", "right_hind")
AXES = ("roll", "pitch", "flexion")
LEG_ACTUATORS = tuple(f"{c}_{a}" for c in CORNERS for a in AXES)
MUSEUM_TARGET_SPEED_MPS = 0.25

REWARD_TERMS = ("r_track", "r_track_linear", "r_alive", "r_upright", "r_height", "r_lateral", "r_yaw_rate",
                "r_heading", "r_energy", "r_action_rate", "r_slip", "r_diagonal", "r_heat", "r_saturation",
                "r_air_time", "r_lateral_drift", "r_heat_peak")
STEERING_TERMS = ()  # terms only a steering command_mode adds (none since the review of 23 Nov: r_track tracks both)
REWARD_BONUSES = ("r_track", "r_alive")  # >= 0; r_diagonal and r_air_time may take either sign; every other is <= 0
# Two schedules (train.py sets both in every env before each rollout):
RAMPED_TERMS = ("r_energy", "r_slip")  # x penalty_scale: from the first steps (pipe1's schedule, 0.1 -> 1 over 2 M)
# x late_scale (0 until 1 M), and so is the NEGATIVE part of r_air_time
LATE_TERMS = ("r_heat", "r_saturation", "r_heat_peak")
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
    # (Under a steering command_mode these two are not used, but their draws are still made, so the reset's random
    # stream, and everything drawn after it, is the same in every mode.)

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
    # Whole-robot factors (D26; None = off, the default: no draw, nothing changes). When set, two uniforms are drawn
    # at every reset after every other draw; under domain_randomization they apply on top of the per-body factors:
    # every body's mass and inertia x one factor from mass_scale_whole_range, then a payload from payload_kg_range
    # added to the chassis at its centre of mass, its inertia scaled by the same mass ratio (as the sweep's payload).
    # The per-body draws alone keep the whole robot's mass within about x0.92-1.08 (REPORT_FORELEG_D22.md).
    mass_scale_whole_range: tuple[float, float] | None = None
    payload_kg_range: tuple[float, float] | None = None

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
    obs_foot_loads: bool = True  # the 4 foot touch loads / nominal weight (the core build has no foot load cells)
    # With any of these three off its slot leaves the observation; its noise is still drawn, so every other channel,
    # the rewards and the random stream are as with it on. All three off: the robot's own sensors only (D46).
    # sin/cos gait clock in the observation (Hz); 0 removes it. Off by default since 19 Nov 2026: no reward term uses
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
    w_slip: float = 1.0  # x sum over feet in contact of the sole's horizontal speed^2 (m^2/s^2; slip_at_contact); 0.5 until pipe1
    w_diagonal: float = 0.0  # optional diagonal (trot) contact symmetry; off by default
    termination_penalty: float = 5.0
    contact_threshold_n: float = 2.0  # a foot is "in contact" above this touch load

    # --- drive heat and torque limits (penalties on the LATE schedule; see LATE_TERMS) ---
    # Continuous torque ratings at the joint, N m: the mass ledger's joint_thermal_eq_nm (X4 roll and pitch 7.5,
    # X6 front elbow 10.8, X4 hind knee through its 1:1 crank 6.75); a test holds them to the ledger.
    cont_torque_roll: float = 7.5
    cont_torque_pitch: float = 7.5
    cont_torque_fore_flexion: float = 10.8
    cont_torque_hind_flexion: float = 6.75
    # x mean over the 12 leg drives and the step's physics steps of (torque / continuous rating)^2. Standing still
    # scores about 0.06 (cost 0.02); the pipe1 gait about 0.76 (cost 0.23; 3 episodes at 0.25 m/s), pipe1 with
    # exploration noise (std 0.2) 0.99 (cost 0.30); every drive exactly at its rating, 1. 0.3 rather than 0.5 so that
    # at full scale a noisy walking policy still beats standing by about 0.37 per step (README's reward table).
    w_heat: float = 0.3
    # The hottest drive (D29; 0 = off, the default: the term is then exactly 0 and nothing else changes). From
    # heat_peak_start_s on (F2's window starts at 2 s), each drive's (torque / continuous rating)^2, averaged over the
    # step's physics steps, is low-passed with heat_peak_tau_s (heat_peak_update), starting from heat_peak_margin^2:
    # a hot drive's average rises from the margin, a cool one tends below it. The average weighs every step's heat
    # alike (the second version, bias-corrected, put up to 13x the weight on the steps just after 2 s: README). The
    # cost is w_heat_peak x late_scale x max(0, sqrt(the largest of the 12) - heat_peak_margin), in units of the
    # rating's RMS. On gaits well over the margin it comes to 0.80-0.82 of F2's own price over 2-20 s (900 steps x
    # the excess of F2's 2-20 s RMS); near or under the margin the stride ripple of the 4 s average and the maximum
    # over 12 drives make it charge a little more than F2 would. w_heat's mean over 12 drives prices a single hot
    # drive at a twelfth of its own heat; this prices the drive that fails F2 (RMS torque at or over the continuous
    # rating). The average is not in the observation; while it is over the margin, a step's heat reaches the return
    # (gamma 0.99) with alpha / (1 - 0.99 (1 - alpha)) of the weight of an unfiltered per-step cost: 0.33 at tau 4 s
    # (0.67 at 1 s, 0.5 at 2 s, 0.2 at 8 s). A shorter tau prices sooner but also charges short bursts (a push
    # recovery) that the windings' 4-15 min time constants would not notice. The weight: README, "The steering run's
    # training additions" (exploration noise makes the hottest drive hotter; too high a weight makes standing pay
    # better than a noisy gait).
    w_heat_peak: float = 0.0
    heat_peak_margin: float = 0.8  # RMS / continuous rating below which a drive costs nothing
    heat_peak_tau_s: float = 4.0  # low-pass time constant of each drive's (torque / rating)^2, s
    heat_peak_start_s: float = 2.0  # the average starts this long after reset (the cost is 0 before)
    saturation_frac: float = 0.95  # a drive "saturates" at this fraction of its (randomised) force limit
    # x fraction of the 12 drives x physics steps that saturate, against the EPISODE'S randomised force limit (that
    # is the limit the drive would hit; 0.85-1.0 x nominal under domain randomisation)
    w_saturation: float = 2.0  # pipe1 gait: 3.0 % saturated (cost 0.06), with noise 7.1 % (0.14); standing still: 0

    # --- clean steps: debounced contact and air time at touchdown ---
    # A foot counts as lifted only after its load has stayed at or below contact_threshold_n for this long. The
    # ~2 ms breaks seen within pipe1's stances are a sub-millimetre contact-model rebound (sole lowest corner median
    # 0.03 mm above the floor, carrying about 0 N), not steps; debouncing merges them into the stance. They are not
    # penalised: that would train against the contact solver.
    contact_debounce_s: float = 0.010
    # legged_gym-style air-time reward, at each debounced touchdown, summed over feet, only when the command exceeds
    # air_time_min_command: with d = min(air time, air_time_cap_s) - air_time_target_s, + w_air_time x d when d > 0,
    # and w_air_time x late_scale x d when d < 0 (short swings cost only on the late schedule: exploration noise makes
    # many short lifts). A foot's first landing after reset (the soles start 2 mm up) is not a touchdown. 0.20 s: at
    # 0.25 m/s a stride of 10-12 cm takes about 0.45 s; with a trot's duty factor near 0.6 the swing lasts about
    # 0.18 s (pipe1's real swings: 140-215 ms). The cap, 0.5 s, keeps long hops from paying more.
    air_time_target_s: float = 0.20
    air_time_cap_s: float = 0.5
    w_air_time: float = 1.0
    air_time_min_command: float = 0.05

    # --- rough terrain, pushes, low friction and sideways drift (run 3; every default is OFF, so an older run's
    # recorded config builds the identical model and draws the identical random stream) ---
    # A pool of this many static boxes, each on its own mocap body (a box moved at runtime must be on a mocap body:
    # MuJoCo's collision bounds of world-body geoms are fixed when the model is compiled), laid out at every reset
    # ahead of the robot along its heading: flat floor, a run of small steps up and down, or a ramp up, a plateau and
    # a ramp down. 0: no boxes, flat floor only (the model is then exactly the file's).
    terrain_boxes: int = 0
    terrain_prob_steps: float = 0.35  # the rest of the episodes are flat
    terrain_prob_slopes: float = 0.35
    terrain_step_max_m: float = 0.04  # each step's rise or drop: U(0.25, 1) x this x terrain scale
    terrain_step_len_m: tuple[float, float] = (0.3, 1.0)  # plateau lengths along the heading
    terrain_slope_max_deg: float = 10.0  # each ramp's angle: U(0.25, 1) x this x terrain scale
    terrain_slope_len_m: tuple[float, float] = (0.8, 2.0)  # the up ramp's horizontal length
    terrain_plateau_len_m: tuple[float, float] = (0.4, 1.5)  # the top between the two ramps
    terrain_start_m: tuple[float, float] = (0.4, 0.8)  # first edge ahead of the chassis (front soles at ~0.28 m)
    terrain_half_width_m: float = 2.0  # sideways extent each side of the start line
    # Sideways pushes on the chassis: with this probability per episode, one push, starting at a time drawn from
    # push_window_s and lasting push_duration_s, of impulse robot mass x dv along the chassis's left (+) or right (-)
    # axis at onset; |dv| ~ U(0, push_dv_max_mps x terrain scale), a free-body velocity change (the feet absorb part).
    push_prob: float = 0.0
    push_dv_max_mps: float = 1.0
    push_window_s: tuple[float, float] = (3.0, 17.0)
    push_duration_s: float = 0.1
    # Low friction on the same schedule: when set, the friction draw's lower bound moves from friction_range[0]
    # towards this value with the terrain scale (one uniform draw either way). None: friction_range as given.
    friction_low_hard: float | None = None
    # x (sideways speed, low-passed with time constant lateral_drift_tau_s)^2, (m/s)^2: a cost on steady sideways
    # drift that leaves the trot's step-to-step sway (which w_lateral already prices) almost free. The speed is along
    # the reset heading's left axis, so a turned heading counts too (under a steering command_mode: the robot's own
    # left axis, since its heading is meant to change).
    w_lateral_drift: float = 0.0
    lateral_drift_tau_s: float = 1.0
    # Where w_slip takes a sole's speed: False (every run up to run2b) at the sole geom's origin; True at the mean
    # point of the sole's contacts (the origin's velocity plus omega x r, as gait_metrics' slip_at_contact), so a
    # rounded sole that rolls without sliding is not charged for rolling. In the frozen trot (two episodes per foot,
    # friction 1.0) the cost at the capsule sole's origin is 3.6x the cost at its contacts, the box soles' 1.1-1.2x;
    # less on slippery floors and terrain (D19, rl/analysis/PREREG_RUN3.md).
    slip_at_contact: bool = False

    # --- steering (decision D20; rl/PLAN_STEERING_AND_TELEMETRY.md): follow a requested direction ---
    # "none" (default): the task above, byte-identical (no draw, no extra observation). "heading" or
    # "heading_and_speed": the repository's shared direction-command contract (environments/shared/
    # direction_commands.py, command_frame.py), imported and never edited. A seeded schedule requests a world heading
    # and a speed, changing every cmd_switch_interval_s +- cmd_switch_jitter_s by up to cmd_turn_increment_max; a
    # bounded proportional adapter turns them into a body-frame command (forward speed, 0 lateral, turn rate), which
    # the policy observes as the shared 3-wide segment, pre-scaled to [-1, 1], appended LAST. The heading the
    # adapter uses is the robot's own estimate (true yaw plus the gyro drift), not the simulator's.
    # Extension (this env only, the shared contract stops turning at zero speed): with cmd_turn_in_place, a stop
    # request whose heading differs from the estimate by more than cmd_turn_in_place_deadband turns on the spot, until
    # it is within half of it (hysteresis, so an overshoot does not chatter).
    # The 60 cm values: 0.25 m/s cruise (the museum target), turns up to 0.4 rad/s (23 deg/s; a 0.6 m radius at
    # cruise), heading steps up to 60 deg every 3-5 s; set before any steering run, to revisit with its results.
    command_mode: str = "none"
    cmd_cruise_speed: float = 0.25  # "heading": the requested speed (or a stop)
    cmd_speed_range: tuple[float, float] = (0.1, 0.27)  # "heading_and_speed": drawn per request
    cmd_stop_probability: float = 0.15
    cmd_turn_increment_max: float = math.pi / 3.0
    cmd_straight_probability: float = 0.25
    cmd_switch_interval_s: float = 4.0
    cmd_switch_jitter_s: float = 1.0
    cmd_speed_scale: float = 0.3  # the command segment's scales (m/s, m/s, rad/s): physical / scale is in [-1, 1]
    cmd_lateral_speed_scale: float = 0.3
    cmd_yaw_rate_scale: float = 0.6
    cmd_yaw_gain: float = 1.5  # rad/s of turn per rad of heading error, clipped to cmd_yaw_rate_max
    cmd_yaw_rate_max: float = 0.4
    cmd_turn_slowdown: bool = True  # forward speed x max(min fraction, cos(heading error)) while turning
    cmd_min_turn_speed_fraction: float = 0.25
    cmd_turn_in_place: bool = True
    cmd_turn_in_place_deadband: float = math.radians(5.0)
    # Steering reward (revised after the reviews of 23 Nov, before any steering run):
    # - r_track is the shared contract's gaussian_tracking_reward x w_track: exp(-(e_v / track_sigma)^2 -
    #   (e_yaw / track_yaw_sigma)^2), with e_v the forward speed error (as the straight task's r_track) and e_yaw the
    #   turn-rate error averaged over about one stride (low-passed with track_yaw_tau_s). One Gaussian, so standing
    #   during a walk or a turn request earns about nothing (a separate turn bonus paid standing still). The trot's
    #   step-to-step sideways sway and yaw wobble stay out of it, as in the straight task (r_lateral and r_yaw_rate
    #   price them per step): with the per-step planar error and turn rate, two trained straight gaits scored 0.86 and
    #   0.41 instead of their straight-task 1.44 and 1.32; with these, 1.42 and 1.29 (at a zero turn command);
    # - r_track_linear on the forward error, as in the straight task; r_yaw_rate -w_yaw_rate x e_yaw^2;
    # - r_heading 0: the adapter closes the heading loop (the plan's heading term was dropped before any run);
    # - stepping (r_air_time, r_diagonal) counts when the command asks to walk OR to turn faster than
    #   air_time_min_turn_rate, so turning on the spot is a stepping task too.
    # Speeds are taken along the robot's own heading, not the reset heading.
    track_yaw_sigma: float = 0.2  # rad/s; about twice a trained trot's yaw wobble (run2b: SD 0.09 rad/s)
    track_yaw_tau_s: float = 0.5  # s; the turn-rate error's averaging for r_track (about one stride of a slow trot)
    air_time_min_turn_rate: float = 0.1  # rad/s

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


def _repository_on_path() -> None:
    """Put the repository root on sys.path (last) for the shared command contract, when this file sits in the
    repository (rl/ -> hardware/triceratops_build -> hardware -> root) and the root is not importable yet: tool
    processes that import this env from rl/ alone (the sweep's workers) do not have it."""
    import sys

    root = HERE.parents[2]
    if (root / "environments" / "shared").is_dir() and str(root) not in sys.path:
        sys.path.append(str(root))


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


TERRAIN_KINDS = ("flat", "steps", "slopes")
TERRAIN_DRAWS = 64  # uniforms drawn per reset when terrain is on: a fixed count keeps the random stream aligned
TERRAIN_THICKNESS_M = 0.05  # step boxes reach this far below the floor; ramp boxes are this thick
TERRAIN_PARKED_Z = -2.0  # unused boxes wait here, under the floor, out of reach


def terrain_layout(cfg: "TrikeEnvConfig", scale: float, u) -> dict:
    """One episode's terrain, a pure function of ``TERRAIN_DRAWS`` uniforms ``u`` in [0, 1) and the terrain scale.

    In the start frame (``s`` metres along the reset heading from the chassis, ``z`` up): ``kind``; ``boxes``, each
    ``(s_centre, z_centre, half_length, half_height, pitch)`` with pitch the angle (rad) the box's long axis rises at;
    and the ground profile ``(s_points, z_points)``, linear between points, 0 outside them. Every box spans
    ``cfg.terrain_half_width_m`` to each side."""
    s = min(1.0, max(0.0, float(scale)))
    it = iter(np.asarray(u, dtype=np.float64))

    def draw(lo: float = 0.0, hi: float = 1.0) -> float:
        return lo + (hi - lo) * float(next(it))

    r = draw()
    kind = ("steps" if r < cfg.terrain_prob_steps else
            "slopes" if r < cfg.terrain_prob_steps + cfg.terrain_prob_slopes else "flat")
    start = draw(*cfg.terrain_start_m)
    T = TERRAIN_THICKNESS_M
    boxes: list[tuple[float, float, float, float, float]] = []
    ps, pz = [start], [0.0]

    def plateau(e0: float, e1: float, h: float) -> None:
        if h > 1e-4:
            boxes.append((0.5 * (e0 + e1), 0.5 * (h - T), 0.5 * (e1 - e0), 0.5 * (h + T), 0.0))

    def ramp(s0: float, z0: float, s1: float, z1: float) -> None:
        if abs(z1 - z0) > 1e-4:
            th = math.atan2(z1 - z0, s1 - s0)
            half_len = 0.5 * math.hypot(s1 - s0, z1 - z0)
            # the top face runs from (s0, z0) to (s1, z1); the centre is half a thickness below it, along the normal
            boxes.append((0.5 * (s0 + s1) + 0.5 * T * math.sin(th), 0.5 * (z0 + z1) - 0.5 * T * math.cos(th),
                          half_len, 0.5 * T, th))

    if kind == "steps":
        hmax, h, edge = cfg.terrain_step_max_m * s, 0.0, start
        for i in range(cfg.terrain_boxes):
            rise = draw(0.25, 1.0) * hmax
            up = draw() < 0.5 or i == 0
            if not up and h - rise < 0.0:
                up = True
            if up and h + rise > 3.0 * hmax:
                up = False
            h_new = h + rise if up else h - rise
            length = draw(*cfg.terrain_step_len_m) + (6.0 if i == cfg.terrain_boxes - 1 else 0.0)
            plateau(edge, edge + length, h_new)
            ps += [edge + 1e-6, edge + length]
            pz += [h_new, h_new]
            edge, h = edge + length, h_new
        ps.append(edge + 1e-6)
        pz.append(0.0)
    elif kind == "slopes":
        th_up = math.radians(draw(0.25, 1.0) * cfg.terrain_slope_max_deg * s)
        th_dn = math.radians(draw(0.25, 1.0) * cfg.terrain_slope_max_deg * s)
        run_up, top = draw(*cfg.terrain_slope_len_m), draw(*cfg.terrain_plateau_len_m)
        H = run_up * math.tan(th_up)
        run_dn = H / math.tan(th_dn) if th_dn > 0 else 0.0
        a, b = start, start + run_up
        c, d = b + top, b + top + run_dn
        ramp(a, 0.0, b, H)
        plateau(b, c, H)
        ramp(c, H, d, 0.0)
        ps += [b, c, d]
        pz += [H, H, 0.0]
    return {"kind": kind, "boxes": boxes, "profile": (np.array(ps), np.array(pz))}


FLAT_FLOOR = {"terrain_boxes": 0, "push_prob": 0.0, "friction_low_hard": None, "mass_scale_whole_range": None,
              "payload_kg_range": None}


def flat_floor(cfg: "TrikeEnvConfig") -> "TrikeEnvConfig":
    """``cfg`` with the training-only terrain, pushes, low-friction end and whole-robot mass and payload draws off:
    what evaluations and the analysis tools measure on (the robustness sweep adds its own steps, ramps, pushes, masses
    and payloads), so a randomised measurement draws the per-body factors only, as every run before them did. An
    older run's config is unchanged."""
    return cfg.replace(**FLAT_FLOOR)


def terrain_model(model_path: str, n_boxes: int, half_width: float) -> mujoco.MjModel:
    """The model file plus ``n_boxes`` terrain boxes ``terrain_box_<i>``, each the only geom of its own massless
    mocap body parked under the floor, with the floor's contact settings. Built in memory with MjSpec; nothing on
    disk changes. The robot's bodies, joints, actuators and sensors keep their order and ids."""
    spec = mujoco.MjSpec.from_file(str(model_path))
    ref = mujoco.MjModel.from_xml_path(str(model_path))
    fl = ref.geom("floor").id
    for i in range(n_boxes):
        b = spec.worldbody.add_body()
        b.name = f"terrain_{i}"
        b.mocap = True
        b.pos = [0.0, 0.0, TERRAIN_PARKED_Z]
        g = b.add_geom()
        g.name = f"terrain_box_{i}"
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.size = [0.05, min(0.05, half_width), 0.05]
        g.density = 0.0
        g.contype, g.conaffinity, g.group = int(ref.geom_contype[fl]), int(ref.geom_conaffinity[fl]), 0
        g.condim, g.priority = int(ref.geom_condim[fl]), int(ref.geom_priority[fl])
        g.friction = ref.geom_friction[fl].tolist()
        g.solref, g.solimp = ref.geom_solref[fl].tolist(), ref.geom_solimp[fl].tolist()
        g.solmix, g.margin, g.gap = float(ref.geom_solmix[fl]), float(ref.geom_margin[fl]), float(ref.geom_gap[fl])
        g.material = "floor_mat"
    return spec.compile()


def diagonal_score(contact) -> float:
    """+1 for a trot (diagonal pairs together, left and right apart), -1 for a bound (front pair and hind pair each
    together, diagonals apart), 0 for a pace, a pronk, all four down or all four up. Order: ``CORNERS``."""
    c = [bool(x) for x in contact]
    return 0.5 * (float(c[0] == c[3]) + float(c[1] == c[2])) - 0.5 * (float(c[0] == c[1]) + float(c[2] == c[3]))


def heat_peak_update(ema: np.ndarray, x: np.ndarray, alpha: float) -> np.ndarray:
    """One control step of ``r_heat_peak``'s average, in place on ``ema`` (heat_peak_margin^2 for every drive when the
    average starts): each drive's step mean ``x`` of (torque / rating)^2, low-passed with ``alpha``. Returns each
    drive's RMS over its rating. Linear in ``x``, with the same weight for every step: after n steps of a constant x,
    ``ema`` = x + (start - x) (1 - alpha)^n."""
    ema += alpha * (x - ema)
    return np.sqrt(ema)


def reward_terms(cfg: TrikeEnvConfig, *, v_fwd: float, v_cmd: float, v_lat: float, yaw_rate: float, yaw_err: float,
                 grav_b, z: float, z_nominal: float, mean_power: float, action, prev_action, slip: float, contact,
                 penalty_scale: float = 1.0, late_scale: float = 1.0, heat_index: float = 0.0,
                 saturated_fraction: float = 0.0, air_time_gain: float = 0.0,
                 air_time_shortfall: float = 0.0, v_lat_slow: float = 0.0,
                 yaw_rate_cmd: float | None = None, yaw_err_slow: float | None = None,
                 heat_peak_excess: float = 0.0) -> dict[str, float]:
    """One control step's reward terms (``REWARD_TERMS``), a pure function of the measured quantities.

    ``penalty_scale`` (0-1) multiplies the energy and slip penalties (``RAMPED_TERMS``); ``late_scale`` (0-1) the heat
    and saturation penalties (``LATE_TERMS``) and the negative part of the air-time term. train.py ramps both, the
    late one from 1,000,000 steps on, so that early exploration is not punished into standing still.
    ``heat_index``: mean of (torque / continuous rating)^2 over the leg drives and physics steps; ``saturated_fraction``:
    share of drive-physics-steps at or above ``saturation_frac`` of the episode's (randomised) force limit;
    ``air_time_gain`` (>= 0) and ``air_time_shortfall`` (<= 0): over this step's debounced touchdowns, the sums of the
    positive and of the negative parts of min(air time, ``air_time_cap_s``) - ``air_time_target_s``;
    ``v_lat_slow``: the sideways speed low-passed with ``lateral_drift_tau_s`` (``r_lateral_drift``, 0 by default).
    ``heat_peak_excess``: the hottest drive's low-passed RMS torque over its rating, minus ``heat_peak_margin``, at
    least 0 (``r_heat_peak``, 0 by default).
    ``yaw_rate_cmd``: the steering command's turn rate (a steering ``command_mode``; None for the straight task):
    ``r_track`` becomes the shared contract's Gaussian on the forward speed error and the turn-rate error together
    (``yaw_err_slow``, the error low-passed with ``track_yaw_tau_s``; the per-step error if None), ``r_yaw_rate``
    penalises the per-step turn-rate error, ``r_heading`` is 0, and stepping counts while turning too."""
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
        "r_heat": -cfg.w_heat * late_scale * heat_index,
        "r_saturation": -cfg.w_saturation * late_scale * saturated_fraction,
        "r_air_time": (cfg.w_air_time * (air_time_gain + late_scale * air_time_shortfall)
                       if v_cmd > cfg.air_time_min_command or (yaw_rate_cmd is not None and abs(yaw_rate_cmd) >
                                                                cfg.air_time_min_turn_rate) else 0.0),
        "r_lateral_drift": -cfg.w_lateral_drift * v_lat_slow ** 2,
        "r_heat_peak": -cfg.w_heat_peak * late_scale * heat_peak_excess,
    }
    if yaw_rate_cmd is not None:
        from environments.shared.direction_commands import gaussian_tracking_reward

        e_yaw = yaw_rate - yaw_rate_cmd
        e_track = e_yaw if yaw_err_slow is None else yaw_err_slow
        finite = math.isfinite(err) and math.isfinite(e_track)  # the shared function refuses non-finite input
        terms["r_track"] = cfg.w_track * gaussian_tracking_reward(abs(err), abs(e_track),
                                                                  velocity_sigma=cfg.track_sigma,
                                                                  yaw_rate_sigma=cfg.track_yaw_sigma) if finite else 0.0
        terms["r_yaw_rate"] = -cfg.w_yaw_rate * e_yaw ** 2
        terms["r_heading"] = 0.0
    if cfg.w_diagonal and (v_cmd > 0.05 or (yaw_rate_cmd is not None and abs(yaw_rate_cmd) >
                                            cfg.air_time_min_turn_rate)):
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
        if cfg.command_mode not in ("none", "heading", "heading_and_speed"):
            raise ValueError(f"command_mode must be none, heading or heading_and_speed, got {cfg.command_mode!r}")
        if (cfg.w_heat_peak < 0 or not 0.0 < cfg.heat_peak_margin <= 2.0 or cfg.heat_peak_tau_s <= 0
                or cfg.heat_peak_start_s < 0):
            raise ValueError("w_heat_peak must be >= 0, heat_peak_margin in (0, 2], heat_peak_tau_s > 0 and "
                             "heat_peak_start_s >= 0")
        for name in ("mass_scale_whole_range", "payload_kg_range"):
            r = getattr(cfg, name)
            if r is not None and not (len(r) == 2 and 0.0 <= r[0] <= r[1]):
                raise ValueError(f"{name} must be None or (low, high) with 0 <= low <= high, got {r!r}")
        if cfg.mass_scale_whole_range is not None and cfg.mass_scale_whole_range[0] <= 0.0:
            raise ValueError("mass_scale_whole_range must be positive")
        self._steer = cfg.command_mode != "none"
        if self._steer:  # imported only when steering, so the straight task's imports are as before
            _repository_on_path()
            from environments.shared.command_frame import COMMAND_MODES
            from environments.shared.direction_commands import DirectionCommandConfig, DirectionCommandController

            assert cfg.command_mode in COMMAND_MODES
            self._cmd = DirectionCommandController(DirectionCommandConfig(
                cruise_speed=cfg.cmd_cruise_speed,
                speed_range=tuple(cfg.cmd_speed_range) if cfg.command_mode == "heading_and_speed" else None,
                stop_probability=cfg.cmd_stop_probability, turn_increment_max=cfg.cmd_turn_increment_max,
                straight_probability=cfg.cmd_straight_probability, switch_interval_s=cfg.cmd_switch_interval_s,
                switch_jitter_s=cfg.cmd_switch_jitter_s, speed_scale=cfg.cmd_speed_scale,
                lateral_speed_scale=cfg.cmd_lateral_speed_scale, yaw_rate_scale=cfg.cmd_yaw_rate_scale,
                yaw_gain=cfg.cmd_yaw_gain, yaw_rate_max=cfg.cmd_yaw_rate_max, turn_slowdown=cfg.cmd_turn_slowdown,
                minimum_turn_speed_fraction=cfg.cmd_min_turn_speed_fraction))
            self._cmd_scales = np.array([cfg.cmd_speed_scale, cfg.cmd_lateral_speed_scale, cfg.cmd_yaw_rate_scale])
        if cfg.terrain_boxes > 0 and cfg.terrain_prob_slopes > 0 and cfg.terrain_boxes < 3:
            raise ValueError("terrain_boxes must be at least 3 when slopes are on (ramp, plateau, ramp)")
        if not 0.0 <= cfg.terrain_prob_steps + cfg.terrain_prob_slopes <= 1.0:
            raise ValueError("terrain_prob_steps + terrain_prob_slopes must lie in [0, 1]")
        if 2 + 3 * cfg.terrain_boxes > TERRAIN_DRAWS:  # steps use 2 + 3 draws per box
            raise ValueError(f"terrain_boxes must be at most {(TERRAIN_DRAWS - 2) // 3}")
        self.model = (terrain_model(cfg.model_path, cfg.terrain_boxes, cfg.terrain_half_width_m)
                      if cfg.terrain_boxes > 0 else mujoco.MjModel.from_xml_path(str(cfg.model_path)))
        self.data = mujoco.MjData(self.model)
        m = self.model
        # mj_setConst rewrites the qpos of the data it is given (to qpos_spring), so the whole-robot factors, applied
        # after the reset state is placed, recompute the constants on this scratch copy (the constants are the same)
        self._const_data = mujoco.MjData(self.model)

        self.physics_dt = float(m.opt.timestep)
        self.dt = self.physics_dt * cfg.frame_skip
        self.max_episode_steps = int(round(cfg.episode_seconds / self.dt))
        self._filter_alpha = (1.0 - math.exp(-2.0 * math.pi * cfg.action_filter_hz * self.dt)
                              if cfg.action_filter_hz > 0 else 1.0)
        self.penalty_scale = 1.0  # see reward_terms; set by train.py through set_penalty_scale
        self.late_scale = 1.0  # see reward_terms; set by train.py through set_late_scale
        self.terrain_scale = 1.0  # terrain, pushes and low friction (0-1); set by train.py through set_terrain_scale
        self.terrain_kind = "flat"
        self._ground = None  # (origin, heading, profile s, profile z) of the episode's terrain; None: flat floor
        self._drift_alpha = 1.0 - math.exp(-self.dt / cfg.lateral_drift_tau_s) if cfg.lateral_drift_tau_s > 0 else 1.0
        self._yaw_alpha = 1.0 - math.exp(-self.dt / cfg.track_yaw_tau_s) if cfg.track_yaw_tau_s > 0 else 1.0
        self._heat_alpha = 1.0 - math.exp(-self.dt / cfg.heat_peak_tau_s)
        self._heat_start_step = int(round(cfg.heat_peak_start_s / self.dt))  # control steps before the average starts

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
        # terrain boxes (none unless terrain_boxes > 0): geom, its mocap body and that body's collision-bound node
        self._tgeom = np.array([m.geom(f"terrain_box_{i}").id for i in range(cfg.terrain_boxes)], dtype=np.int64)
        tbody = m.geom_bodyid[self._tgeom]
        self._tmocap = m.body_mocapid[tbody].astype(np.int64)
        self._tbvh = m.body_bvhadr[tbody].astype(np.int64)
        self._is_ground = np.zeros(m.ngeom, dtype=bool)
        self._is_ground[self._floor] = True
        self._is_ground[self._tgeom] = True
        # Lookup table over contact pairs (g1 * ngeom + g2): True for the ground (floor or a terrain box) with any
        # geom but a sole. One matmul and one gather per physics step instead of half a dozen small-array numpy calls.
        not_sole = ~self._is_sole & ~self._is_ground
        table = np.zeros((m.ngeom, m.ngeom), dtype=bool)
        table[np.ix_(self._is_ground, not_sole)] = True
        table[np.ix_(not_sole, self._is_ground)] = True
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
        if cfg.obs_foot_loads:
            parts.append(("foot_load_over_weight", 4))
        # both: the action-rate penalty is on the raw outputs, the targets follow the filtered ones
        parts += [("previous_action_filtered", 12), ("previous_action_raw", 12), ("command_speed_mps", 1)]
        if cfg.phase_clock_hz > 0:
            parts.append(("phase_sin_cos", 2))
        if cfg.command_mode != "none":  # the shared command segment, LAST (environments/shared/command_frame.py)
            parts.append(("command", 3))
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
        self._cmd_exec = {"physical": np.zeros(3), "normalized": np.zeros(3), "heading_error": 0.0,
                          "heading_active": False, "observe_heading": False, "turn_in_place": False,
                          "desired_heading": 0.0, "desired_speed": 0.0, "event_id": 0}
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
        self._loaded_once = np.zeros(4, dtype=bool)  # a foot lifts only after its first landing since reset
        self._unloaded = np.zeros(4, dtype=np.int64)  # consecutive physics steps each foot has been unloaded
        self.touchdowns = 0  # debounced touchdowns this episode (diagnostic)
        self._v_lat_slow = 0.0  # sideways speed, low-passed (r_lateral_drift)
        # each drive's (torque / rating)^2, low-passed (r_heat_peak), from the margin^2 when it starts
        self._heat_ema = np.full(12, self.cfg.heat_peak_margin ** 2)
        self.heat_peak_rms = 0.0  # sqrt of the largest of them, this step (0 before heat_peak_start_s)
        self._yaw_err_slow = 0.0  # steering: turn-rate error, low-passed (r_track)
        self._phys_step = 0  # physics steps since reset (the push's clock)
        self._push_dv, self._push_start, self._push_end = 0.0, -1, -1  # no push unless reset draws one
        self._push_force = np.zeros(3)

    def set_penalty_scale(self, scale: float) -> None:
        """Scale (0-1) of the energy and slip penalties (``RAMPED_TERMS``); train.py ramps it from the first steps."""
        self.penalty_scale = float(min(1.0, max(0.0, scale)))

    def set_terrain_scale(self, scale: float) -> None:
        """Scale (0-1) of the terrain's steps and slopes, the pushes and the low-friction end (from the next reset);
        train.py ramps it from --terrain-ramp-from steps. No effect when terrain, pushes and friction_low_hard are off."""
        self.terrain_scale = float(min(1.0, max(0.0, scale)))

    def set_late_scale(self, scale: float) -> None:
        """Scale (0-1) of the heat and saturation penalties (``LATE_TERMS``) and of short air times; train.py keeps it
        at 0 until --late-ramp-from steps, then ramps it."""
        self.late_scale = float(min(1.0, max(0.0, scale)))

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
        lo, hi = cfg.friction_range
        if cfg.friction_low_hard is not None:
            lo = lo + self.terrain_scale * (cfg.friction_low_hard - lo)
        return {
            "friction": float(rng.uniform(lo, hi)),
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
            m.geom_friction[self._tgeom, 0] = mu
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
        # 5. terrain (only when configured: TERRAIN_DRAWS uniforms, after every older draw), laid ahead of the chassis
        if cfg.terrain_boxes > 0:
            self._place_terrain(rng.random(TERRAIN_DRAWS))
        d.ctrl[:] = 0.0
        d.ctrl[self._other_act] = self._other_neutral
        d.ctrl[self._leg_ctrl_index] = self._ctrl_neutral
        mujoco.mj_forward(m, d)

        self._reset_episode_state(command=command, delay=delay)
        R = d.xmat[self._chassis].reshape(3, 3)
        self._heading = math.atan2(R[1, 0], R[0, 0])
        self._badqacc0 = int(d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number)
        # 6. push (only when configured: 4 uniforms, after the terrain's)
        if cfg.push_prob > 0:
            on, when, size, side = rng.random(4)
            if on < cfg.push_prob:
                self._push_dv = (1.0 if side < 0.5 else -1.0) * size * cfg.push_dv_max_mps * self.terrain_scale
                self._push_start = int(round((cfg.push_window_s[0] + when * (cfg.push_window_s[1] -
                                                                           cfg.push_window_s[0])) / self.physics_dt))
                self._push_end = self._push_start + max(1, int(round(cfg.push_duration_s / self.physics_dt)))

        # 7. steering command (steering modes only: one draw, after every other, seeds the request schedule). An
        # evaluation's "command_speed" holds the reset heading at that speed; "command_target" (heading offset rad,
        # speed m/s) holds another request. Either holds until reset, as the shared controller's set_target does.
        if self._steer:
            st = self._cmd.reset(rng, self._heading)  # the estimate is the true yaw at reset (no drift yet)
            target = opts.get("command_target")
            if target is None and "command_speed" in opts:
                target = (0.0, float(opts["command_speed"]))
            if target is not None:
                if not 0.0 <= float(target[1]) <= cfg.cmd_speed_scale:
                    raise ValueError(f"a steering env takes requested speeds in [0, cmd_speed_scale = "
                                     f"{cfg.cmd_speed_scale}] m/s, got {target[1]}")
                self._cmd.set_target(self._heading + float(target[0]), float(target[1]), time_s=0.0)
                st = self._cmd.update(0.0, self._heading)
            self._set_command(st)

        # 8. whole-robot factors (only when configured: 2 uniforms, after every other draw), applied under domain
        # randomisation on top of the per-body factors, then the derived constants (on the scratch data, so the
        # state placed in step 4 is kept) and the derived quantities of the state recomputed
        if cfg.mass_scale_whole_range is not None or cfg.payload_kg_range is not None:
            u_whole, u_pay = rng.random(2)
            if cfg.domain_randomization:
                lo, hi = cfg.mass_scale_whole_range or (1.0, 1.0)
                whole = lo + (hi - lo) * float(u_whole)
                lo, hi = cfg.payload_kg_range or (0.0, 0.0)
                payload = lo + (hi - lo) * float(u_pay)
                m.body_mass[1:] *= whole
                m.body_inertia[1:] *= whole
                if payload > 0.0:
                    c = self._chassis
                    ratio = (m.body_mass[c] + payload) / m.body_mass[c]
                    m.body_mass[c] += payload
                    m.body_inertia[c] *= ratio
                mujoco.mj_setConst(m, self._const_data)
                mujoco.mj_forward(m, d)
                self.factors["mass_scale_whole"] = whole
                self.factors["payload_kg"] = payload

        obs = self._get_obs(noise=True)
        info = {"command_speed": self.command_speed, "delay_steps": self.delay_steps,
                "domain_randomization": bool(self.factors)}
        if self.factors:
            info["friction"] = self.factors["friction"]
        if cfg.terrain_boxes > 0 or cfg.push_prob > 0 or cfg.friction_low_hard is not None:
            info.update({"terrain_kind": self.terrain_kind, "terrain_scale": self.terrain_scale,
                         "push_dv": self._push_dv,
                         "push_time_s": self._push_start * self.physics_dt if self._push_start >= 0 else None})
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
        peak_on = cfg.w_heat_peak > 0.0
        heat_vec = np.zeros(12) if peak_on else None
        saturated = touchdowns = 0
        air_gain = air_short = 0.0
        floor_hit = -1
        inv_c2, sat_limit, touch_adr, thr = self._inv_cont2, self._sat_limit, self._touch_adr, cfg.contact_threshold_n
        n_db, dt_phys, t_air, t_cap = self._debounce_steps, self.physics_dt, cfg.air_time_target_s, cfg.air_time_cap_s
        push = self._push_start >= 0
        for _ in range(cfg.frame_skip):
            fifo.append(target)
            d.ctrl[idx] = fifo[0]
            if push:
                self._apply_push()
            mujoco.mj_step(m, d)
            self._phys_step += 1
            tau = d.actuator_force[idx]
            # |torque x joint speed| of the 12 leg drives (actuator_velocity = joint speed, gear 1)
            energy += float(np.abs(tau * d.actuator_velocity[idx]).sum())
            heat += float(np.dot(tau * tau, inv_c2))
            if peak_on:
                heat_vec += tau * tau * inv_c2
            saturated += int(np.count_nonzero(np.abs(tau) >= sat_limit))
            # debounced contact: a touchdown ends a lift; a break shorter than the debounce time stays in the stance
            raw = d.sensordata[touch_adr] > thr
            prev = self._unloaded
            down_now = raw & self._lifted
            if down_now.any():
                touchdowns += int(down_now.sum())
                dev = np.minimum(prev[down_now] * dt_phys, t_cap) - t_air
                air_gain += float(dev[dev > 0].sum())
                air_short += float(dev[dev < 0].sum())
                self._lifted &= ~down_now
            self._loaded_once |= raw
            self._unloaded = np.where(raw, 0, prev + 1)
            self._lifted |= (self._unloaded >= n_db) & self._loaded_once
            if floor_hit < 0 and d.ncon:
                floor_hit = self._non_foot_floor_contact()
            if hook is not None:
                hook(self)
        self._step_count += 1
        self.touchdowns += touchdowns
        mean_power = energy / cfg.frame_skip
        heat_index = heat / (12 * cfg.frame_skip)
        heat_peak_excess = 0.0
        if peak_on and self._step_count > self._heat_start_step:  # each drive's step mean of (torque / rating)^2
            rms = heat_peak_update(self._heat_ema, heat_vec / cfg.frame_skip, self._heat_alpha)
            self.heat_peak_rms = float(rms.max())
            heat_peak_excess = max(0.0, self.heat_peak_rms - cfg.heat_peak_margin)
        saturated_fraction = saturated / (12 * cfg.frame_skip)

        # --- measurements ---
        R = d.xmat[self._chassis].reshape(3, 3)
        xy1 = d.xpos[self._chassis, :2]
        disp = (xy1 - xy0) / self.dt
        yaw = math.atan2(R[1, 0], R[0, 0])
        h = np.array([math.cos(yaw if self._steer else self._heading), math.sin(yaw if self._steer else self._heading)])
        v_fwd = float(disp @ h)
        v_lat = float(disp @ np.array([-h[1], h[0]]))
        self._distance += v_fwd * self.dt
        yaw_err = _wrap(yaw - self._heading)
        yaw_rate = float((R @ d.qvel[3:6])[2])
        grav_b = -R[2, :]
        # chassis height above the terrain under it (the floor's 0 without terrain)
        z = float(d.xpos[self._chassis, 2]) - (self.ground_height(xy1) if self._ground is not None else 0.0)
        self._v_lat_slow += self._drift_alpha * (v_lat - self._v_lat_slow)
        if self._steer:  # against the command this step ran under
            e_now = yaw_rate - float(self._cmd_exec["physical"][2])
            self._yaw_err_slow += self._yaw_alpha * (e_now - self._yaw_err_slow)
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(R[2, 2])))))
        loads = d.sensordata[self._touch_adr]
        contact = loads > cfg.contact_threshold_n
        slip = 0.0
        vel6 = np.zeros(6)
        for i in np.flatnonzero(contact):
            g = int(self._soles[i])
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_GEOM, g, vel6, 0)
            if cfg.slip_at_contact and d.ncon:
                gg = d.contact.geom[:d.ncon]
                hit = (gg[:, 0] == g) | (gg[:, 1] == g)
                if hit.any():  # else (a touch load without a contact) the origin's speed, as without the option
                    vel6[3:] += np.cross(vel6[:3], d.contact.pos[:d.ncon][hit].mean(axis=0) - d.geom_xpos[g])
            slip += float(vel6[3] ** 2 + vel6[4] ** 2)

        executed = self._cmd_exec  # the command this step ran under (steering modes)
        v_cmd_executed = self.command_speed
        terms = reward_terms(cfg, v_fwd=v_fwd, v_cmd=self.command_speed, v_lat=v_lat, yaw_rate=yaw_rate,
                             yaw_err=yaw_err, grav_b=grav_b, z=z, z_nominal=self.nominal_height,
                             mean_power=mean_power, action=a, prev_action=self._prev_action, slip=slip,
                             contact=contact, penalty_scale=self.penalty_scale, late_scale=self.late_scale,
                             heat_index=heat_index, saturated_fraction=saturated_fraction, air_time_gain=air_gain,
                             air_time_shortfall=air_short, v_lat_slow=self._v_lat_slow,
                             heat_peak_excess=heat_peak_excess,
                             yaw_rate_cmd=float(self._cmd_exec["physical"][2]) if self._steer else None,
                             yaw_err_slow=self._yaw_err_slow if self._steer else None)

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
        t = self._step_count * self.dt
        if self._steer:  # the next command, from the robot's heading estimate
            self._set_command(self._cmd.update(t, _wrap(yaw + self._heading_drift)))
        obs = self._get_obs(noise=True)
        info = dict(terms)
        info.update({
            "reward_total": reward, "forward_speed": v_fwd, "lateral_speed": v_lat, "command_speed": v_cmd_executed,
            "episode_forward_speed": self._distance / t, "yaw_error": yaw_err, "yaw_rate": yaw_rate,
            "tilt_deg": tilt, "height": z, "mean_power_w": mean_power, "foot_contacts": contact.astype(np.int8),
            "delay_steps": self.delay_steps, "termination": reason, "time_s": t, "penalty_scale": self.penalty_scale,
            "late_scale": self.late_scale,
            "heading_drift": self._heading_drift, "heat_index": heat_index, "saturated_fraction": saturated_fraction,
            "touchdowns": touchdowns, "air_time_gain": air_gain, "air_time_shortfall": air_short,
            "lateral_speed_slow": self._v_lat_slow,
        })
        if peak_on:  # the hottest drive's average (0 before heat_peak_start_s); absent when the term is off
            info["heat_peak_rms"] = self.heat_peak_rms
        if self._steer:  # the executed command and its tracking (the shared contract's field names)
            err_h = _wrap(executed["desired_heading"] - yaw) if executed["observe_heading"] else 0.0
            info.update({
                "command_v_x": float(executed["physical"][0]), "command_v_y": float(executed["physical"][1]),
                "command_yaw_rate": float(executed["physical"][2]), "desired_heading": executed["desired_heading"],
                "desired_speed": executed["desired_speed"], "actual_heading": yaw, "heading_error_rad": err_h,
                "heading_active": executed["heading_active"], "turn_in_place": executed["turn_in_place"],
                "command_event_id": executed["event_id"],
                "base_x": float(d.xpos[self._chassis, 0]), "base_y": float(d.xpos[self._chassis, 1]),
                "tracking_error_v": float(math.hypot(v_fwd - executed["physical"][0], v_lat - executed["physical"][1])),
                "tracking_error_yaw": abs(yaw_rate - float(executed["physical"][2])),
                "yaw_rate_error_slow": self._yaw_err_slow,
                "tracking_error_heading": abs(err_h)})
        if self._ground is not None or self._push_start >= 0:
            info["ground_height"] = float(d.xpos[self._chassis, 2]) - z
            info["push_active"] = bool(self._push_start <= self._phys_step - 1 < self._push_end)  # the last step
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------ steering
    def _set_command(self, st) -> None:
        """The command the policy gets from a shared ``DirectionCommandState``: its body-frame command, or, for a stop
        whose requested heading differs from the estimate by more than the deadband and with ``cmd_turn_in_place``,
        a turn on the spot (forward 0, the same bounded proportional turn rate)."""
        cfg = self.cfg
        phys = np.array(st.physical, dtype=np.float64)
        active, err, turning = bool(st.heading_active), float(st.heading_error), False
        if not active and cfg.cmd_turn_in_place:
            e = _wrap(st.desired_heading - st.current_heading)
            band = cfg.cmd_turn_in_place_deadband * (0.5 if self._cmd_exec["turn_in_place"] else 1.0)
            if abs(e) > band:
                yaw_cmd = float(np.clip(cfg.cmd_yaw_gain * e, -cfg.cmd_yaw_rate_max, cfg.cmd_yaw_rate_max))
                phys, err, turning = np.array([0.0, 0.0, yaw_cmd]), e, True
        # heading_active keeps the shared contract's meaning (a request with a speed); a turn on the spot is
        # reported as turn_in_place, and the observation shows the heading error during either
        self._cmd_exec = {"physical": phys, "normalized": np.clip(phys / self._cmd_scales, -1.0, 1.0),
                          "heading_error": err, "heading_active": active, "observe_heading": active or turning,
                          "turn_in_place": turning,
                          "desired_heading": float(st.desired_heading), "desired_speed": float(st.desired_speed),
                          "event_id": int(st.event_id)}
        self.command_speed = float(phys[0])

    def set_command_target(self, heading_world: float, speed: float) -> np.ndarray:
        """An operator's or a path follower's request (steering modes only): hold this world heading (rad) and speed
        (m/s) until reset. It applies from the next step on; the observation returned here carries it, so act on
        that one (the one the last step returned shows the old command)."""
        if not self._steer:
            raise RuntimeError("set_command_target needs a steering command_mode")
        if not 0.0 <= float(speed) <= self.cfg.cmd_speed_scale:
            raise ValueError(f"requested speed must lie in [0, cmd_speed_scale = {self.cfg.cmd_speed_scale}] m/s")
        t = self._step_count * self.dt
        R = self.data.xmat[self._chassis].reshape(3, 3)
        self._cmd.set_target(heading_world, speed, time_s=t)
        self._set_command(self._cmd.update(t, _wrap(math.atan2(R[1, 0], R[0, 0]) + self._heading_drift)))
        return self._get_obs(noise=True)

    # ------------------------------------------------------------------ pieces
    def _lowest_sole_z(self) -> float:
        """Lowest point of the four soles: a box sole's bottom corners (the as-built foot), or, for the foot study's
        variants (sim/build_feet60.py), a sphere's or a capsule's lowest surface point."""
        d, lowest = self.data, np.inf
        signs = np.array([[sx, sy, -1.0] for sx in (-1.0, 1.0) for sy in (-1.0, 1.0)])
        for g in self._soles:
            R = d.geom_xmat[g].reshape(3, 3)
            size, kind = self.model.geom_size[g], int(self.model.geom_type[g])
            if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
                z = float(d.geom_xpos[g, 2] - size[0])
            elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:  # axis along the geom's z, half length size[1]
                z = float(d.geom_xpos[g, 2] - abs(R[2, 2]) * size[1] - size[0])
            else:
                pts = d.geom_xpos[g] + (signs * size) @ R.T
                z = float(pts[:, 2].min())
            lowest = min(lowest, z)
        return lowest

    def _non_foot_floor_contact(self) -> int:
        """The first geom other than a sole touching the floor in the current contacts, or -1."""
        g = self.data.contact.geom
        hits = self._bad_pair[g @ self._pair_weight]
        if not hits.any():
            return -1
        g1, g2 = (int(x) for x in g[int(np.argmax(hits))])
        return g2 if self._is_ground[g1] else g1

    def _apply_push(self) -> None:
        """Before each physics step: the push's force on the chassis (at its centre of mass) from its first step to
        its last, along the chassis's horizontal left axis at onset; zero after."""
        n, d = self._phys_step, self.data
        if n == self._push_start:
            left = d.xmat[self._chassis].reshape(3, 3)[:, 1].copy()
            left[2] = 0.0
            left /= max(float(np.linalg.norm(left)), 1e-9)
            mass = float(self.model.body_mass.sum())
            self._push_force = left * mass * self._push_dv / ((self._push_end - self._push_start) * self.physics_dt)
            d.xfrc_applied[self._chassis, :3] = self._push_force
        elif n == self._push_end:
            d.xfrc_applied[self._chassis, :3] = 0.0

    def _place_terrain(self, u) -> None:
        """Lays out one episode's terrain (``terrain_layout``) along the chassis's heading from its reset position:
        each box's mocap pose, size and collision bounds; unused boxes parked under the floor."""
        m, d = self.model, self.data
        lay = terrain_layout(self.cfg, self.terrain_scale, u)
        R = d.xmat[self._chassis].reshape(3, 3)
        psi = math.atan2(R[1, 0], R[0, 0])
        h = np.array([math.cos(psi), math.sin(psi)])
        o = d.xpos[self._chassis, :2].copy()
        qz = np.array([math.cos(psi / 2), 0.0, 0.0, math.sin(psi / 2)])
        W = self.cfg.terrain_half_width_m
        q = np.zeros(4)
        for i in range(len(self._tgeom)):
            g, k, bvh = int(self._tgeom[i]), int(self._tmocap[i]), int(self._tbvh[i])
            if i < len(lay["boxes"]):
                sc, zc, hx, hz, pitch = lay["boxes"][i]
                d.mocap_pos[k] = [o[0] + sc * h[0], o[1] + sc * h[1], zc]
                qy = np.array([math.cos(-pitch / 2), 0.0, math.sin(-pitch / 2), 0.0])  # rotation -pitch about y
                mujoco.mju_mulQuat(q, qz, qy)
                d.mocap_quat[k] = q
                size = np.array([hx, W, hz])
            else:
                d.mocap_pos[k] = [o[0], o[1], TERRAIN_PARKED_Z]
                d.mocap_quat[k] = [1.0, 0.0, 0.0, 0.0]
                size = np.array([0.05, 0.05, 0.05])
            # the box's own collision bounds (MuJoCo misses a contact with a box whose geom_aabb or geom_rbound is
            # stale), and its body's bounding-volume node's half-sizes (the node's centre stays as compiled; MuJoCo
            # 3.10 does not use the node for a single-geom body, it is kept consistent)
            m.geom_size[g] = size
            m.geom_rbound[g] = float(np.linalg.norm(size))
            m.geom_aabb[g] = [0.0, 0.0, 0.0, *size]
            m.bvh_aabb[bvh, 3:] = size
        self.terrain_kind = lay["kind"]
        self._ground = (o, h, lay["profile"][0], lay["profile"][1])

    def ground_height(self, xy) -> float:
        """Height of the terrain (0: the floor) under a world point ``xy``, from the episode's profile."""
        if self._ground is None:
            return 0.0
        o, h, ps, pz = self._ground
        r = np.asarray(xy, dtype=np.float64)[:2] - o
        if abs(float(r @ np.array([-h[1], h[0]]))) > self.cfg.terrain_half_width_m:
            return 0.0  # beside the boxes: the floor
        return float(np.interp(float(r @ h), ps, pz, left=0.0, right=0.0))

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
        if self._steer:  # the estimate against the requested heading; 0 while no heading is requested
            c = self._cmd_exec
            yaw_err_est = (_wrap(math.atan2(Rm[1, 0], Rm[0, 0]) + self._heading_drift - c["desired_heading"])
                           if c["observe_heading"] else 0.0)
        else:
            yaw_err_est = _wrap(math.atan2(Rm[1, 0], Rm[0, 0]) + self._heading_drift - self._heading)
        parts = [q, qd]
        if cfg.obs_ankle_angles:
            parts.append(ankle)
        parts += [grav_b, [math.sin(yaw_err_est), math.cos(yaw_err_est)], gyro]
        if cfg.obs_privileged_lin_vel:
            parts.append(lin_vel)
        if cfg.obs_foot_loads:
            parts.append(loads)
        parts += [self._action_filtered, self._prev_action, [self.command_speed]]
        if cfg.phase_clock_hz > 0:
            ph = 2.0 * math.pi * cfg.phase_clock_hz * self._step_count * self.dt
            parts.append([math.sin(ph), math.cos(ph)])
        if self._steer:
            parts.append(self._cmd_exec["normalized"])
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
