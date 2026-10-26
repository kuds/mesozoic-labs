"""
Tyrannosaurus Rex Gymnasium Environment

A large bipedal predator with a massive skull and vestigial forelimbs.
The Stage 3 task uses contact between a fixed head geom and prey as a
"bite" proxy; the model has no articulated jaw.

Observation space (bipedal-target-heading-free/v1 since policy interface r14;
the total dimension is generated in the public species catalog):
    - Joint positions (qpos[7:]) — 21 (21 hinge)
    - Joint velocities (qvel[6:]) — 21 (21 hinge)
    - Pelvis projected gravity (unit down vector in pelvis coordinates) — 3
    - Pelvis angular velocity (gyroscope, its own site frame) — 3
    - Pelvis linear velocity in the gravity-aligned heading frame (forward, left, up) — 3
    - Pelvis acceleration (accelerometer, its own site frame) — 3
    - Foot contact forces (plantar pad + three digit touch sensors per foot) — 2
    - Prey direction in pelvis coordinates (unit vector) — 3
    - Prey distance (scalar) — 1
    - Body-relative command (v_x_cmd, v_y_cmd, yaw_rate_cmd; zeros under command_mode = "none") — 3

No channel carries the world heading: the three root blocks come from the
free root's own quaternion through environments/shared/heading_observation.py
(docs/HEADING_INVARIANCE_DESIGN_2026_10.md), so turning the whole scene about
vertical leaves the observation unchanged.

Action space (total dimension is generated in the public species catalog):
    - Neck/head: neck pitch, neck yaw, head pitch (3)
    - Right leg: hip pitch/roll, knee, ankle (4)
    - Left leg: hip pitch/roll, knee, ankle (4)
    - Tail: pitch 1, yaw 1, pitch 2, pitch 3 (4)

The six toe hinges are passive (spring-loaded at the home pose); they stay
in qpos/qvel so the observation still senses toe posture.

Reward components:
    - Forward velocity (toward prey)
    - Backward velocity penalty
    - Drift penalty (horizontal displacement from spawn)
    - Alive bonus
    - Fall penalty
    - Energy penalty
    - Tail stability
    - Bite-proxy bonus (fixed head geom contacts prey)
    - Approach shaping (distance to prey)
    - Head proximity shaping (reward for positioning head near prey)
    - Posture (continuous tilt penalty)
    - Nosedive penalty
    - Height maintenance
    - Bilateral foot support and load balance
    - Home leg-pose retention (around the authored keyframe, or the statue's
      settled leg pose, mirror-symmetrised)
    - Sole flatness and stance width (around the home keyframe's width, or
      the animal's own width at the end of the settle), optionally paid only
      on a loaded foot
    - Floor impact (summed foot force above a body-weight multiple) and
      airborne substeps, read per physics substep
    - Standing lateral lean (the centre of mass over the feet, left to right;
      off by default)
    - Head clearance and neck posture
    - Gait symmetry (alternating foot contacts)
    - Action smoothness (penalize jerky action changes)
    - Spin penalty (penalize pelvis angular velocity)
    - Heading alignment (facing toward prey)
    - Lateral velocity penalty (anti crab-walk)
    - Speed penalty (penalise absolute speed above threshold)
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from environments.shared.base_env import BaseDinoEnv
from environments.shared.direction_commands import DirectionCommandConfig
from environments.shared.heading_observation import body_to_world, heading_free_root_segments, world_to_heading
from environments.shared.reward_functions import (
    reward_action_saturation as _reward_action_saturation_pure,
)
from environments.shared.reward_functions import (
    reward_bilateral_support as _reward_bilateral_support_pure,
)
from environments.shared.reward_functions import (
    reward_foot_load_balance as _reward_foot_load_balance_pure,
)
from environments.shared.reward_functions import (
    reward_head_clearance as _reward_head_clearance_pure,
)
from environments.shared.reward_functions import (
    reward_lateral_lean as _reward_lateral_lean_pure,
)
from environments.shared.reward_functions import (
    reward_soft_home_pose as _reward_soft_home_pose_pure,
)
from environments.shared.reward_functions import (
    reward_sole_flatness as _reward_sole_flatness_pure,
)
from environments.shared.reward_functions import (
    reward_stance_stagger as _reward_stance_stagger_pure,
)
from environments.shared.reward_functions import (
    reward_stance_width as _reward_stance_width_pure,
)
from environments.shared.reward_functions import (
    reward_target_centered_height as _reward_target_centered_height_pure,
)


class TyrannosaurusRexEnv(BaseDinoEnv):
    """Tyrannosaurus Rex bipedal locomotion and bite-attack environment.

    The species base (decision D-D33): the task, the rewards and the hashed
    policy-interface methods (``_get_obs``, ``_scale_action``,
    ``_cache_ids``), shared by every embodiment of the species.  An
    embodiment subclass (``anatomical/env.py``) sets ``embodiment`` and
    ``MODEL_PATH``; this class is never instantiated itself.
    """

    #: Set by each embodiment subclass: its embodiment and its MJCF.
    embodiment: str
    MODEL_PATH: Path
    #: The species' training labels (species_registry.get_species_config).
    HEIGHT_LABEL = "Pelvis height"
    BEHAVIOR_SECTION_LABEL = "Hunting"
    SUCCESS_KEYS = ("bite_success", "strike_success")

    action_mapping = "home-keyframe-residual/v1"
    # SB3 only since policy interface r14 (configs/plant_versions.toml note
    # 18, decision D-D31): the heading-free observation has no frozen MJX
    # builder, so the T-Rex left the D-D17 core and its mjx_config.py
    # registration was deleted.  The plant contract now hashes _cache_ids
    # (interface_implementations.sb3_sensor_cache) as it does for every
    # SB3-only species, so a token edit there -- reward-target setup
    # included -- is a policy-interface revision.
    supported_training_backends = ("stable-baselines3",)
    # 10 Hz command low-pass (plant interface r11).  Stage-1 balance needs
    # ~1.1-1.4 Hz of closed-loop bandwidth (stance-gate filter probe), but
    # both 2026-08 stage-1 runs converged on statically unstable poses kept
    # up by 16.8-18.7 Hz command chatter that amplitude penalties cannot
    # price out.  The cutoff removes that bandwidth during training, with
    # nearly a decade of margin above the task's needs.  The plant contract
    # hashes the cutoff and the filter arithmetic
    # (interface_implementations.action_low_pass_filter).  See
    # docs/investigations/TREX_STAGE1_NARROW_TOLERANCE_RUN_2026_08.md.
    action_filter_cutoff_hz = 10.0
    # Settled tail pose for the tail_home_pose term, in tail_home_joint_names
    # order (tail_1_pitch, tail_1_yaw, tail_2_pitch, tail_3_pitch); see the
    # provenance comment in _cache_ids.
    _TAIL_SETTLED_QPOS = (0.00002, 0.0, 0.00006, 0.00018)
    # Settled neck pose for the neck_posture term under
    # neck_posture_reference = "settled", in neck_home_joint_names order
    # (neck_pitch, neck_yaw, head_pitch); see the provenance comment in
    # _cache_ids.
    _NECK_SETTLED_QPOS = (0.21401, 0.0, 0.22054)
    _NECK_POSTURE_REFERENCES = ("keyframe", "settled")
    # Settled leg pose for the leg_home_pose term under leg_home_pose_reference
    # = "settled", in leg_home_joint_names order (r_hip_pitch, r_hip_roll,
    # r_knee, r_ankle, l_hip_pitch, l_hip_roll, l_knee, l_ankle).  Measured on
    # the physics-r10 plant as the zero-action statue's mean over the
    # post-settle window (steps 200-999) of the stance config, seeds 3042-3051
    # (all ten full-horizon), then mirror-symmetrised: each left-right pair
    # replaced by its mean, hip roll sign-mapped (the raw pairs agreed to 8e-5
    # rad).  Against the keyframe that is hip pitch -2.60, hip roll -/+0.39,
    # knee -0.25 and ankle +1.95 degrees: the servos do not hold the keyframe
    # under load.  The statues ran at the stance config's reset_noise_scale
    # 0.05; the noise-free statue settles at hip roll 0.000 degrees with the
    # other joints within 0.02 degrees of these, so the -/+0.39 degree hip-roll
    # splay is the noise-averaged settle.  Centred on the keyframe, the term
    # paid a policy for pulling the hips and ankles back toward it, which leans
    # the animal forward; centred here it does not.  First measured on physics
    # r9 (-0.0931906, -/+0.0073721, -0.4397564, 1.3713665; hip pitch -2.74,
    # hip roll -/+0.42, knee -0.40, ankle +1.67 degrees); re-measured the same
    # way on r10 (plant_versions note 20, D-D44), whose level tail moves the
    # ankle most (+4.9 mrad).
    # Re-measure whenever leg masses, springs, gains or the feet change (any
    # plant revision; test_the_settled_leg_pose_is_the_statues checks it).
    _LEG_SETTLED_QPOS = (
        -0.09084374343772902,
        -0.006823512801851125,
        -0.43726937988410725,
        1.3762339038992937,
        -0.09084374343772902,
        0.006823512801851125,
        -0.43726937988410725,
        1.3762339038992937,
    )
    _LEG_HOME_POSE_REFERENCES = ("keyframe", "settled")
    #: Opt-in task options carved out of the task fingerprint and of the
    #: effective config ``stage_config.json`` records while off: each entry is
    #: (switch, off value, the kwargs the switch makes inert).  Off, they change
    #: no reward, info key, trajectory or reset draw, so a stage that leaves
    #: them off keeps its task_sha256; on, every listed kwarg is hashed
    #: (task_fingerprint.inert_opt_in_env_kwargs).
    OPT_IN_TASK_KWARGS = (
        ("lateral_lean_weight", 0.0, ("lateral_lean_weight", "lateral_lean_tolerance_m")),
        ("leg_home_pose_reference", "keyframe", ("leg_home_pose_reference",)),
        # Screen-only (stance4, never merged): the standing foot stagger penalty.
        ("stance_stagger_weight", 0.0, ("stance_stagger_weight", "stance_stagger_tolerance_m")),
        # Scratch prototype (stance4 copband design, never merged, never trained): the per-foot CoP band.
        (
            "stance_cop_band_weight",
            0.0,
            (
                "stance_cop_band_weight",
                "stance_cop_band_lower",
                "stance_cop_band_upper",
                "stance_cop_band_pad_lower",
                "stance_cop_band_pad_upper",
                "stance_cop_band_tolerance",
                "stance_cop_band_full_price_load_share",
                "stance_cop_band_min_load_share",
            ),
        ),
    )
    _STANCE_WIDTH_REFERENCES = ("keyframe", "settled")
    _ACTION_PENALTY_SOURCES = ("filtered", "raw")
    # A foot (or the pair) reads as airborne at or below this touch force, in
    # N: the floor-truth library's contact threshold (gait.constants
    # CONTACT_THRESHOLD_N, which a species env may not import) and the
    # stance-duty diagnostics' default.
    _AIRBORNE_TOUCH_N = 0.1
    _camera_distance = 3.0
    _camera_azimuth = 135
    _camera_elevation = -20
    _camera_track_body = "pelvis"
    # The neck never collides (contype/conaffinity 0); the behavior env ends an
    # episode when it penetrates the terrain.  Inert in this env.
    _terrain_contact_probe_geoms = ("neck_geom",)

    def __init__(
        self,
        render_mode: str | None = None,
        frame_skip: int = 5,
        max_episode_steps: int = 1000,
        # Reward weights
        forward_vel_weight: float = 1.0,
        alive_bonus: float = 0.1,
        energy_penalty_weight: float = 0.001,
        fall_penalty: float = -100.0,
        tail_stability_weight: float = 0.05,
        bite_bonus: float = 10.0,
        bite_approach_weight: float = 1.0,
        bite_head_proximity_weight: float = 0.0,
        posture_weight: float = 0.2,
        nosedive_weight: float = 0.0,
        natural_pitch: float = 0.017,
        height_weight: float = 0.0,
        height_target_tolerance: float = 0.0,
        gait_symmetry_weight: float = 0.0,
        smoothness_weight: float = 0.05,
        heading_weight: float = 0.0,
        lateral_penalty_weight: float = 0.0,
        backward_vel_penalty_weight: float = 0.0,
        drift_penalty_weight: float = 0.0,
        spin_penalty_weight: float = 0.0,
        speed_penalty_weight: float = 0.0,
        speed_penalty_threshold: float = 0.10,
        idle_penalty_weight: float = 0.0,
        idle_velocity_threshold: float = 0.05,
        forward_vel_max: float = 8.0,
        foot_contact_weight: float = 0.0,
        foot_contact_gate: float = 0.0,
        bilateral_support_weight: float = 0.0,
        foot_contact_saturation_force: float = 100.0,
        foot_load_balance_weight: float = 0.0,
        foot_load_balance_min_support_force: float = 0.0,
        foot_load_balance_airborne_penalty: float = 0.0,
        action_jerk_weight: float = 0.0,
        support_conditioned_alive_fraction: float = 0.0,
        leg_home_pose_weight: float = 0.0,
        leg_home_pose_tolerance: float = 0.35,
        head_clearance_weight: float = 0.0,
        head_clearance_target: float = 0.60,
        head_clearance_tolerance: float = 0.48,
        neck_posture_weight: float = 0.0,
        neck_posture_tolerance: float = 0.35,
        neck_posture_reference: str = "keyframe",
        tail_home_pose_weight: float = 0.0,
        tail_home_pose_tolerance: float = 0.10,
        action_saturation_weight: float = 0.0,
        action_saturation_threshold: float = 0.9,
        leg_home_pose_broad_fraction: float = 0.0,
        leg_home_pose_broad_scale: float = 6.0,
        foot_flatness_weight: float = 0.0,
        foot_flatness_tolerance_deg: float = 3.0,
        stance_width_weight: float = 0.0,
        stance_width_tolerance_m: float = 0.05,
        stance_width_reference: str = "keyframe",
        stance_width_settle_steps: int = 200,
        foot_terms_min_support_force: float = 0.0,
        floor_impact_weight: float = 0.0,
        floor_impact_threshold_bw: float = 1.4,
        airborne_substep_weight: float = 0.0,
        action_penalty_source: str = "filtered",
        lateral_lean_weight: float = 0.0,
        lateral_lean_tolerance_m: float = 0.015,
        leg_home_pose_reference: str = "keyframe",
        stance_stagger_weight: float = 0.0,
        stance_stagger_tolerance_m: float = 0.045,
        stance_cop_band_weight: float = 0.0,
        stance_cop_band_lower: float = -0.6,
        stance_cop_band_upper: float = 0.4,
        stance_cop_band_pad_lower: float = -0.65,
        stance_cop_band_pad_upper: float = 0.65,
        stance_cop_band_tolerance: float = 0.357,
        stance_cop_band_full_price_load_share: float = 0.25,
        stance_cop_band_min_load_share: float = 0.35,
        nosedive_termination_threshold: float = 0.62,
        # Environment settings
        prey_distance_range: tuple[float, float] = (3.0, 8.0),
        prey_lateral_range: tuple[float, float] = (-2.0, 2.0),
        # The value the evidence supports.  Both bounds moved down by 0.05 with
        # the theropod stance: the flexed home keyframe settles at 0.9260 m
        # where the columnar one settled at 0.9757 (-0.0497).  Translating the
        # whole band by the measured stance drop is what preserves the three
        # measurements the old (0.75, 1.6) was sized on, rather than
        # re-guessing them:
        #   * The tail is the real backstop.  tail_3/4/5 are unconditional
        #     termination geoms and the tail reaches the floor at a pelvis
        #     height of ~0.55-0.57 in a level squat, so the plant is already
        #     lying down below that.  The tail's pose is fixed relative to the
        #     pelvis and does not move with the legs, so the stance change
        #     left that figure unchanged (0.70 cleared it by ~0.13).  A
        #     kinematic scan of 2026-10-25 (level pelvis, home-keyframe legs,
        #     the settled tail pitches) re-measures it on the settled tail and
        #     supersedes both figures: on physics r9, whose settled tail
        #     drooped, the lowest points of tail_3/4/5 sit 0.46 / 0.55 /
        #     0.60 m below the pelvis, so r9's tail first reaches the floor
        #     at a pelvis height of about 0.60, which 0.70 clears by about
        #     0.10.  Physics r10 (plant_versions note 20, D-D44) preloads the
        #     tail pitch springs and the settled tail is level: 0.13 / 0.11 /
        #     0.075 m below the pelvis, so in a level squat tail_3/4/5 reach
        #     the floor only at a pelvis height of about 0.13, and this floor,
        #     not the tail, is the backstop on r10 (the pushed recovery
        #     statue's 40 ends: tail_contact 14 -> 1, fallen 0 -> 19).  The
        #     bound itself is unchanged (the maintainer kept 0.70 on
        #     2026-10-25).
        #   * Healthy full-horizon episodes bottomed out 0.134 m above the
        #     floor (0.884 against 0.75).  0.834 against 0.70 is the same
        #     margin under a stance that sits 0.0497 lower.
        #   * The retired MJX alive bonus scaled by (z - floor) / (ceiling -
        #     floor), which was 0.266 at the settled stance before and after --
        #     that is why the ceiling moved too -- against the velociraptor's
        #     0.275.
        # The spawn tail is preserved too: 2000 resets at reset_noise_scale
        # = 0.10 put 0.75% below the floor on both plants (0.70 on this one,
        # 0.75 on the columnar one).  The root-height jitter is a 0.10 m
        # normal and dominates the joint noise on either stance, so the
        # translated floor keeps the same share of unrecoverable spawns.
        healthy_z_range: tuple[float, float] = (0.70, 1.55),
        reset_noise_scale: float = 0.01,
        perturbation_capture_velocity_multiple: float = 0.0,
        perturbation_interval: float = 2.0,
        perturbation_jitter: float = 0.5,
        perturbation_duration: float = 0.20,
        perturbation_direction: str = "uniform_horizontal",
        command_mode: str = "none",
        command_config: DirectionCommandConfig | None = None,
    ):
        model_path = str(self.MODEL_PATH)

        # T-Rex-specific reward weights
        self.tail_stability_weight = tail_stability_weight
        self.bite_bonus = bite_bonus
        self.bite_approach_weight = bite_approach_weight
        self.bite_head_proximity_weight = bite_head_proximity_weight
        self.posture_weight = posture_weight
        self.nosedive_weight = nosedive_weight
        self.height_weight = height_weight
        self.height_target_tolerance = height_target_tolerance
        self.gait_symmetry_weight = gait_symmetry_weight
        self.smoothness_weight = smoothness_weight
        self.heading_weight = heading_weight
        self.lateral_penalty_weight = lateral_penalty_weight
        self.backward_vel_penalty_weight = backward_vel_penalty_weight
        self.drift_penalty_weight = drift_penalty_weight
        self.spin_penalty_weight = spin_penalty_weight
        self.speed_penalty_weight = speed_penalty_weight
        self.speed_penalty_threshold = speed_penalty_threshold
        self.idle_penalty_weight = idle_penalty_weight
        self.idle_velocity_threshold = idle_velocity_threshold
        self.forward_vel_max = forward_vel_max
        self.foot_contact_weight = foot_contact_weight
        self.foot_contact_gate = foot_contact_gate
        self.bilateral_support_weight = bilateral_support_weight
        self.foot_contact_saturation_force = foot_contact_saturation_force
        self.foot_load_balance_weight = foot_load_balance_weight
        self.foot_load_balance_min_support_force = foot_load_balance_min_support_force
        self.foot_load_balance_airborne_penalty = foot_load_balance_airborne_penalty
        self.action_jerk_weight = action_jerk_weight
        self.support_conditioned_alive_fraction = support_conditioned_alive_fraction
        self.leg_home_pose_weight = leg_home_pose_weight
        self.leg_home_pose_tolerance = leg_home_pose_tolerance
        self.head_clearance_weight = head_clearance_weight
        self.head_clearance_target = head_clearance_target
        self.head_clearance_tolerance = head_clearance_tolerance
        self.neck_posture_weight = neck_posture_weight
        self.neck_posture_tolerance = neck_posture_tolerance
        self.neck_posture_reference = neck_posture_reference
        self.tail_home_pose_weight = tail_home_pose_weight
        self.tail_home_pose_tolerance = tail_home_pose_tolerance
        self.action_saturation_weight = action_saturation_weight
        self.action_saturation_threshold = action_saturation_threshold
        self.leg_home_pose_broad_fraction = leg_home_pose_broad_fraction
        self.leg_home_pose_broad_scale = leg_home_pose_broad_scale
        self.foot_flatness_weight = foot_flatness_weight
        self.foot_flatness_tolerance_deg = foot_flatness_tolerance_deg
        self.stance_width_weight = stance_width_weight
        self.stance_width_tolerance_m = stance_width_tolerance_m
        self.stance_width_reference = stance_width_reference
        self.stance_width_settle_steps = stance_width_settle_steps
        self.foot_terms_min_support_force = foot_terms_min_support_force
        self.floor_impact_weight = floor_impact_weight
        self.floor_impact_threshold_bw = floor_impact_threshold_bw
        self.airborne_substep_weight = airborne_substep_weight
        self.action_penalty_source = action_penalty_source
        self.lateral_lean_weight = lateral_lean_weight
        self.lateral_lean_tolerance_m = lateral_lean_tolerance_m
        self.leg_home_pose_reference = leg_home_pose_reference
        self.stance_stagger_weight = stance_stagger_weight
        self.stance_stagger_tolerance_m = stance_stagger_tolerance_m
        self.stance_cop_band_weight = stance_cop_band_weight
        self.stance_cop_band_lower = stance_cop_band_lower
        self.stance_cop_band_upper = stance_cop_band_upper
        self.stance_cop_band_pad_lower = stance_cop_band_pad_lower
        self.stance_cop_band_pad_upper = stance_cop_band_pad_upper
        self.stance_cop_band_tolerance = stance_cop_band_tolerance
        self.stance_cop_band_full_price_load_share = stance_cop_band_full_price_load_share
        self.stance_cop_band_min_load_share = stance_cop_band_min_load_share
        # The band's per-step contact reader (floor_contact/step_cop.py), built on first use when the band is on.
        self._cop_band_reader: Any = None
        self.nosedive_termination_threshold = nosedive_termination_threshold

        if self.height_target_tolerance < 0.0:
            raise ValueError("height_target_tolerance must be non-negative")
        if self.foot_contact_saturation_force <= 0.0:
            raise ValueError("foot_contact_saturation_force must be positive")
        if not 0.0 <= self.support_conditioned_alive_fraction <= 1.0:
            raise ValueError("support_conditioned_alive_fraction must be in [0, 1]")
        if self.leg_home_pose_tolerance <= 0.0:
            raise ValueError("leg_home_pose_tolerance must be positive")
        if self.head_clearance_tolerance <= 0.0:
            raise ValueError("head_clearance_tolerance must be positive")
        if self.neck_posture_tolerance <= 0.0:
            raise ValueError("neck_posture_tolerance must be positive")
        if self.neck_posture_reference not in self._NECK_POSTURE_REFERENCES:
            raise ValueError(
                f"neck_posture_reference must be one of {self._NECK_POSTURE_REFERENCES}, "
                f"got {self.neck_posture_reference!r}"
            )
        if self.tail_home_pose_tolerance <= 0.0:
            raise ValueError("tail_home_pose_tolerance must be positive")
        if not 0.0 <= self.action_saturation_threshold < 1.0:
            raise ValueError("action_saturation_threshold must be in [0, 1)")
        if not 0.0 <= self.leg_home_pose_broad_fraction <= 1.0:
            raise ValueError("leg_home_pose_broad_fraction must be in [0, 1]")
        if self.leg_home_pose_broad_scale <= 1.0:
            raise ValueError("leg_home_pose_broad_scale must exceed 1 (it widens the narrow Gaussian)")
        if self.foot_flatness_tolerance_deg <= 0.0:
            raise ValueError("foot_flatness_tolerance_deg must be positive")
        if self.stance_width_tolerance_m <= 0.0:
            raise ValueError("stance_width_tolerance_m must be positive")
        if self.stance_width_reference not in self._STANCE_WIDTH_REFERENCES:
            raise ValueError(
                f"stance_width_reference must be one of {self._STANCE_WIDTH_REFERENCES}, "
                f"got {self.stance_width_reference!r}"
            )
        if (
            isinstance(self.stance_width_settle_steps, bool)
            or not isinstance(self.stance_width_settle_steps, (int, np.integer))
            or self.stance_width_settle_steps < 1
        ):
            raise ValueError("stance_width_settle_steps must be a whole number of steps, at least 1")
        # Not bounded by max_episode_steps: a shortened horizon (a debug or test
        # override) that ends before the capture step simply never pays the
        # settled width term.
        if not 0.0 <= self.foot_terms_min_support_force < np.inf:
            raise ValueError("foot_terms_min_support_force must be a finite force of at least 0 N")
        if not 0.0 <= self.floor_impact_weight < np.inf:
            raise ValueError("floor_impact_weight must be finite and non-negative")
        if not 0.0 < self.floor_impact_threshold_bw < np.inf:
            raise ValueError("floor_impact_threshold_bw must be a positive, finite multiple of body weight")
        if not 0.0 <= self.airborne_substep_weight < np.inf:
            raise ValueError("airborne_substep_weight must be finite and non-negative")
        if self.action_penalty_source not in self._ACTION_PENALTY_SOURCES:
            raise ValueError(
                f"action_penalty_source must be one of {self._ACTION_PENALTY_SOURCES}, "
                f"got {self.action_penalty_source!r}"
            )
        if not 0.0 <= self.lateral_lean_weight < np.inf:
            raise ValueError("lateral_lean_weight must be finite and non-negative")
        if not 0.0 < self.lateral_lean_tolerance_m < np.inf:
            raise ValueError("lateral_lean_tolerance_m must be a positive, finite length")
        if not 0.0 <= self.stance_stagger_weight < np.inf:
            raise ValueError("stance_stagger_weight must be finite and non-negative")
        if not 0.0 < self.stance_stagger_tolerance_m < np.inf:
            raise ValueError("stance_stagger_tolerance_m must be a positive, finite length")
        if not 0.0 <= self.stance_cop_band_weight < np.inf:
            raise ValueError("stance_cop_band_weight must be finite and non-negative")
        for name in ("stance_cop_band", "stance_cop_band_pad"):
            lower, upper = getattr(self, f"{name}_lower"), getattr(self, f"{name}_upper")
            if not -np.inf < lower < upper < np.inf:
                raise ValueError(f"{name}_lower and {name}_upper must be finite with lower < upper")
        if not 0.0 < self.stance_cop_band_tolerance < np.inf:
            raise ValueError("stance_cop_band_tolerance must be positive and finite")
        if not 0.0 <= self.stance_cop_band_full_price_load_share < self.stance_cop_band_min_load_share < 0.5:
            raise ValueError(
                "stance_cop_band_full_price_load_share and stance_cop_band_min_load_share must satisfy "
                "0 <= full_price_load_share < min_load_share < 0.5"
            )
        if self.leg_home_pose_reference not in self._LEG_HOME_POSE_REFERENCES:
            raise ValueError(
                f"leg_home_pose_reference must be one of {self._LEG_HOME_POSE_REFERENCES}, "
                f"got {self.leg_home_pose_reference!r}"
            )
        # The leg target under leg_home_pose_reference = "settled", chosen in
        # _get_reward_info rather than written over _leg_home_qpos: _cache_ids
        # (hashed by the plant contract) rebuilds that from the keyframe
        # whenever an env swaps its compiled model.
        self._leg_home_settled_qpos = np.array(self._LEG_SETTLED_QPOS)

        # Natural forward pitch (~0.95°), measured: the pelvis frame at the home
        # keyframe is level, and under the home controller the plant settles at
        # forward_z ≈ -0.017.  The nosedive penalty and termination are measured
        # relative to that neutral pose.  The former 0.17 rad (9.7°) described
        # the torso capsule's built-in slope, not the pelvis frame the reward
        # actually reads, so it granted ~7 deg of unpenalized nose-down pitch.
        # The theropod stance halved the residual: a flexed limb settles closer
        # to its commanded pose than a columnar one does, so the plant settled
        # 1.55° nose-down where the columnar stance settled 2.9° (0.027 through
        # physics r9).  Physics r10 (plant_versions note 20, D-D44) preloads
        # the tail pitch springs, so the tail holds level instead of drooping
        # with tail_1 on its ventral stop, and the plant settles 0.95°
        # nose-down: the noise-free statue's mean forward_z over steps
        # 200-999 is -0.01666 (r9 -0.02714), asin 0.01666 rad, so 0.017.
        # nosedive_termination_threshold in configs/tyrannosaurus_rex/anatomical/stance.toml is
        # calibrated against this number and moved with it.
        # Unlike the raptor, the T-Rex posture reward stays centred on world
        # vertical (see _get_reward_info).
        self._natural_forward_z = -np.sin(natural_pitch)

        # T-Rex-specific env settings
        self.prey_distance_range = prey_distance_range
        self.prey_lateral_range = prey_lateral_range

        # State tracking for delta-based rewards
        self._prev_prey_distance: float | None = None
        self._prev_action: np.ndarray | None = None
        # The policy's own clipped command of the current step, kept by the
        # _filter_action override for action_penalty_source = "raw", and the
        # step count it belongs to (a stale command is never priced).
        self._policy_action: np.ndarray | None = None
        self._policy_action_step = -1

        # Gait symmetry: track foot touchdown events for alternation reward
        self._init_gait_state()

        # Cached initial direction to prey (set in _spawn_target).
        # Used by forward-velocity and heading rewards so the "forward"
        # reference direction stays fixed for the whole episode, preventing
        # the reward from flipping sign when the T-Rex passes the prey.
        self._initial_prey_dir_2d: np.ndarray = np.array([1.0, 0.0])

        # Cached initial pelvis position (set in _spawn_target).
        # Used by the drift penalty to discourage horizontal displacement.
        self._initial_pos_2d: np.ndarray = np.array([0.0, 0.0])

        super().__init__(
            model_path=model_path,
            render_mode=render_mode,
            frame_skip=frame_skip,
            max_episode_steps=max_episode_steps,
            forward_vel_weight=forward_vel_weight,
            alive_bonus=alive_bonus,
            energy_penalty_weight=energy_penalty_weight,
            fall_penalty=fall_penalty,
            healthy_z_range=healthy_z_range,
            reset_noise_scale=reset_noise_scale,
            perturbation_capture_velocity_multiple=perturbation_capture_velocity_multiple,
            perturbation_interval=perturbation_interval,
            perturbation_jitter=perturbation_jitter,
            perturbation_duration=perturbation_duration,
            perturbation_direction=perturbation_direction,
            command_mode=command_mode,
            command_config=command_config,
        )
        # Scratch prototype: build the CoP band's reader now, so a model it cannot describe fails here.
        if self.stance_cop_band_weight > 0.0:
            self._cop_band()

    def _cop_band(self) -> Any:
        """The CoP band's per-step contact reader, :class:`environments.shared.floor_contact.step_cop.StepFootCop`.

        Scratch prototype (stance4 copband design, never merged).  The reader
        and the morphology it reduces with live in the neutral floor_contact
        package, which the gait recorder imports too, so this env imports
        nothing from the gait package.  Imported here, only when the band is
        on, so an env with it off never loads the reader.  It needs the
        model's foot registry entry with an ankle and a support span: the
        anatomical T. rex declares both in its FOOT_SUPPORT, which is passed
        in from this class rather than looked up through species_registry
        (support_entry), so a band-on env loads neither train_base nor the
        curriculum nor gait at run time either (a test pins it).
        """
        if self._cop_band_reader is None:
            from environments.shared.floor_contact.morphology import Morphology, entry_from_support
            from environments.shared.floor_contact.step_cop import StepFootCop
            from environments.shared.model_id import ModelId

            embodiment = getattr(type(self), "embodiment", None)
            if not embodiment:
                raise ValueError("stance_cop_band_weight > 0 needs an embodiment class (its FOOT_SUPPORT entry)")
            model_id = ModelId("tyrannosaurus_rex", str(embodiment))
            entry = entry_from_support(getattr(type(self), "FOOT_SUPPORT", None))
            morphology = Morphology.from_env(self, model_id, registry={model_id: entry})
            spans = [(foot.label, foot.sole_geom, foot.heel_line_m, foot.toe_line_m) for foot in morphology.feet]
            if [label for label, *_ in spans] != ["r", "l"] or not all(
                sole >= 0 and np.isfinite(heel) and np.isfinite(toe) for _label, sole, heel, toe in spans
            ):
                raise ValueError(f"stance_cop_band_weight > 0 needs two soled feet r, l with support spans: {spans}")
            self._cop_band_reader = StepFootCop(self, morphology)
        return self._cop_band_reader

    def _accumulate_substep(self, substep: int) -> None:
        # Scratch prototype: the CoP band copies each substep's floor contacts (reads only).  The step being
        # taken is _step_count + 1 (step() increments it after the substep loop).  Nothing happens while off.
        super()._accumulate_substep(substep)
        if self.stance_cop_band_weight > 0.0:
            self._cop_band().record(self.model, self.data, self._step_count + 1, substep)

    def _invalidate_substep_aggregates(self) -> None:
        super()._invalidate_substep_aggregates()
        if self._cop_band_reader is not None:
            self._cop_band_reader.forget()

    def _cache_ids(self):
        """Cache MuJoCo IDs for bodies, geoms, and sites."""
        # T-Rex policies command residuals around the complete XML home
        # control vector. Cache the named keyframe so Gymnasium reset and
        # action zero share one nominal state.
        self._cache_home_keyframe("T-Rex")

        # Body IDs
        self.pelvis_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.skull_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "skull")

        # Geom IDs for contact detection
        self.prey_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "prey_geom")
        self.head_bite_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "head_bite")
        self.torso_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "torso")
        self.floor_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

        # Head/skull geom IDs (collision-enabled geoms that should terminate on ground contact)
        # Note: neck_geom and brow_ridge have contype=0, so they never produce floor contacts
        self.skull_upper_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "skull_upper")
        self.snout_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "snout")

        # Tail geom IDs (distal segments that should not contact floor)
        self.tail_3_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "tail_3_geom")
        self.tail_4_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "tail_4_geom")
        self.tail_5_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "tail_5_geom")

        # Geoms that should terminate the episode on ground contact
        self._body_ground_geoms = {
            self.torso_geom_id,
            self.skull_upper_geom_id,
            self.snout_geom_id,
            self.tail_3_geom_id,
            self.tail_4_geom_id,
            self.tail_5_geom_id,
        }
        self._head_ground_geoms = {self.skull_upper_geom_id, self.snout_geom_id}
        self._tail_ground_geoms = {self.tail_3_geom_id, self.tail_4_geom_id, self.tail_5_geom_id}

        # Site IDs for sensors
        self.imu_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "imu")
        self.r_foot_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "r_foot")
        self.l_foot_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "l_foot")
        self.tail_tip_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "tail_tip")
        self.head_tip_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "head_tip")
        # Height terminations sampled at every physics substep by the base
        # step loop; _is_terminated reads the per-check MIN so a head dip
        # between the control-step samples still terminates.  Order is
        # consumed by index there.
        self._substep_height_checks = (
            ("site", self.head_tip_site_id),
            ("body", self.skull_body_id),
        )

        # Prey mocap body
        self.prey_mocap_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "prey")

        # Sensor indices (order matches MJCF sensor definition)
        # pelvis_gyro(3), pelvis_accel(3), pelvis_orientation(4),
        # r_foot_touch(1), l_foot_touch(1)
        # _sensor_gyro_start, _sensor_accel_start, _sensor_quat_start
        # are inherited from BaseDinoEnv (0, 3, 6 respectively).
        self._sensor_r_foot = 10
        self._sensor_l_foot = 11
        # The pad sensors above see the plantar box only: a touch sensor sums
        # contacts on geoms of its site's own body, and the three digits are
        # child bodies on their own passive hinges.  Their own sensors are
        # appended after the tail block, so pad + digits is the force the foot
        # actually transmits -- at the home keyframe 388.4 N + 112.0 N against
        # a measured 500.4 N of floor contact.
        self._sensor_r_foot_digits = (24, 25, 26)
        self._sensor_l_foot_digits = (27, 28, 29)
        # Per-foot groups for the base class's substep MIN aggregation and the
        # observation's two foot channels; the plant contract records each
        # group's sensor names in the foot_contact segment.
        self._foot_sensor_groups = (
            (self._sensor_r_foot, *self._sensor_r_foot_digits),
            (self._sensor_l_foot, *self._sensor_l_foot_digits),
        )

        # Stage-1 pose targets come from the named home keyframe instead of
        # duplicating angles in Python.  This keeps the reward centred on the
        # p5 theropod stance if the XML's authored equilibrium is recalibrated.
        leg_home_joint_names = (
            "r_hip_pitch",
            "r_hip_roll",
            "r_knee",
            "r_ankle",
            "l_hip_pitch",
            "l_hip_roll",
            "l_knee",
            "l_ankle",
        )
        neck_home_joint_names = ("neck_pitch", "neck_yaw", "head_pitch")
        # The tail term prices JOINT positions.  The pre-existing tail
        # stability term reads only tail-tip angular velocity, which a tail
        # parked motionless against a hard stop satisfies perfectly -- the
        # 2026-08 narrow-tolerance run parked all three of tail_1_yaw,
        # tail_2_pitch and tail_3_pitch that way.
        tail_home_joint_names = ("tail_1_pitch", "tail_1_yaw", "tail_2_pitch", "tail_3_pitch")
        self._leg_home_qpos_indices = self._joint_qpos_indices(leg_home_joint_names)
        self._neck_home_qpos_indices = self._joint_qpos_indices(neck_home_joint_names)
        self._tail_home_qpos_indices = self._joint_qpos_indices(tail_home_joint_names)
        home_qpos = self.model.key_qpos[self.home_keyframe_id]
        self._leg_home_qpos = home_qpos[self._leg_home_qpos_indices].copy()
        # The neck target.  "keyframe" (the legacy default) prices the
        # authored zeros, which the neck servos cannot hold: kp 200 against
        # the 42.85 N.m gravity torque about neck_pitch, and kp 80 against
        # 17.55 N.m about head_pitch, settle the statue 0.2143 / 0.2196 rad
        # below them.  The statue then scores only 0.787 quality, and lifting
        # the head back to the keyframe pays +45.6/episode (+41.9 neck, +5.8
        # clearance, -2.1 energy) -- the raised head of every 2026-09 stance.
        # "settled" targets the statue's own pose instead (the precedent is
        # the settled tail below), so the same lift costs 36.4/episode.
        # Measured on the physics-r8 plant as the noise-free statue's mean
        # over the post-settle window (steps 200-1000): neck_pitch 0.21432,
        # neck_yaw 0.0, head_pitch 0.21963 rad.  The 40-seed window means at
        # reset noise 0.05 are 0.21438 / -0.00002 / 0.21955 (sd <= 0.0002),
        # and the long-run equilibrium is 0.21424 / 0.21957.  The r8 hip-roll
        # servo revision did not move it: r7's 40-seed means agree to 1e-4,
        # and r9 carried it (0.21431 / 0.21968).  Re-measured on physics r10
        # (plant_versions note 20: the level tail moves the settled pitch) the
        # same way (seed 0, steps 200-999): 0.21401 / 0.0 / 0.22054 rad, the
        # target since.  Its 40-seed window means (the stance config, seeds
        # 3042-3081, reset noise 0.05) are 0.21407 / -0.00001 / 0.22066 (sd
        # <= 0.00015), and a 5000-step noise-free roll ends (steps 4000-4999)
        # at 0.21401 / 0.22052.
        # Re-measure whenever neck or head masses, springs, or gains change.
        if self.neck_posture_reference == "settled":
            self._neck_home_qpos = np.array(self._NECK_SETTLED_QPOS)
        else:
            self._neck_home_qpos = home_qpos[self._neck_home_qpos_indices].copy()
        # Statue-derived: the noise-free statue's post-settle tail pose (seed
        # 0, steps 200-999, as TestSettledPoseTargetsTrackTheStatue measures
        # it; the precedent is the 0.9260 m settled height target in
        # _get_reward_info).  Through physics r9 the passive tail could not
        # hold the authored zeros against gravity: under the home controller
        # it drooped with tail_1 pitch on its ventral stop (tail_1/tail_2
        # pitch at -0.2107/-0.2029 rad against the -0.2094 stop, so tail_2
        # just above its own, tail_3 at -0.0926) with sub-milliradian spread
        # across 40 seeds at reset noise 0.05, and the term targeted that
        # droop, since pricing the keyframe would have capped the ideal
        # statue at ~0.25 quality.  Physics r10
        # (plant_versions note 20, D-D44) preloads the three tail pitch
        # springs (springref 52.5 / 25.9 / 10.4 deg), and the statue now
        # settles at the keyframe zeros: 0.00002 / 0.0 / 0.00006 / 0.00018
        # rad, the target since (D-D44 re-centred it; the stance tolerance
        # stays 0.05 rad).  Its 40-seed window means (the stance config, seeds
        # 3042-3081, reset noise 0.05) are 0.00016 / 0.00003 / 0.00014 /
        # 0.00022 (sd <= 0.0003).  Re-measure whenever tail masses, springs,
        # or gains change.
        self._tail_home_qpos = np.array(self._TAIL_SETTLED_QPOS)

        # Sole flatness reads the plantar boxes' own frames: each box's local
        # z is its pad normal.  The pads are level at the home keyframe
        # (5.6e-5 deg) and tilt 0.10 deg, all of it pitch, under the
        # noise-free statue, so level is the target without an offset.
        self._plantar_geom_ids = np.asarray(
            [self._geom_id(name) for name in ("r_plantar_geom", "l_plantar_geom")],
            dtype=np.int64,
        )
        # The stance-width target is the authored home keyframe's horizontal
        # foot-site distance (0.280 m; the noise-free statue settles at
        # 0.27998), from the plant rather than a duplicated number: forward
        # kinematics on a scratch MjData, so self.data is untouched.
        home_data = mujoco.MjData(self.model)
        home_data.qpos[:] = home_qpos
        mujoco.mj_kinematics(self.model, home_data)
        self._home_stance_width = self._stance_width(home_data)
        # The width term's current target: always the keyframe's under
        # stance_width_reference = "keyframe"; under "settled" the keyframe's
        # until _get_reward_info captures the animal's own width at
        # stance_width_settle_steps, and the keyframe's again from each
        # _spawn_target.  Set here too, so a reward read before the first
        # reset is defined.
        self._stance_width_target = self._home_stance_width
        # The animal's weight in N, for the floor-impact term: the pelvis
        # subtree is the whole animal (the prey is a separate mocap body), so
        # this is the 85.72 kg x 9.81 = 840.9 N the floor-truth library and
        # stance.toml's foot_load_balance_min_support_force divide by.
        self._body_weight_n = float(self.model.body_subtreemass[self.pelvis_id] * abs(self.model.opt.gravity[2]))

    def _geom_id(self, geom_name: str) -> int:
        """Resolve a geom id, failing on model drift."""
        geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        if geom_id < 0:
            raise ValueError(f"T-Rex model must define geom {geom_name!r}")
        return int(geom_id)

    def _stance_width(self, data: mujoco.MjData) -> float:
        """Planar (x-y) distance between the two foot sites, free of heading.

        Not the lateral separation alone: a fore-aft stagger adds to it, so
        the width term prices a staggered stance as a width error too.
        ``stance_diagnostics`` records a different quantity under the same
        name, the world-frame lateral offset ``|dy|`` of the foot sites.
        """
        foot_offset = data.site_xpos[self.r_foot_site_id, :2] - data.site_xpos[self.l_foot_site_id, :2]
        return float(np.linalg.norm(foot_offset))

    def _com_lateral_offset(self) -> float:
        """The whole-body centre of mass's lateral offset from the foot-site midpoint, in metres, + left.

        Along the left axis of the gravity-aligned pelvis heading frame (the
        observation's own, ``world_to_heading``), from the kinematics of the
        step's last forward pass: ``subtree_com`` of the pelvis (the whole
        animal; the prey is a separate mocap body), the two foot sites and
        the pelvis orientation.
        """
        com = np.asarray(self.data.subtree_com[self.pelvis_id], dtype=np.float64)
        midpoint = 0.5 * (self.data.site_xpos[self.r_foot_site_id] + self.data.site_xpos[self.l_foot_site_id])
        heading = world_to_heading(body_to_world(self.data.xquat[self.pelvis_id]))
        return float((heading @ (com - midpoint))[1])

    def _foot_stagger(self) -> float:
        """The right foot site's fore-aft lead over the left, in metres, + = right foot ahead (screen-only).

        Along the forward axis of the gravity-aligned pelvis heading frame (the
        observation's own, ``world_to_heading``), from the step's last forward
        pass: the quantity the 2026-10-19 split diagnosis calls the stagger
        (r_foot site minus l_foot site along the pelvis heading).  It flips
        sign under the left-right mirror.
        """
        offset = self.data.site_xpos[self.r_foot_site_id] - self.data.site_xpos[self.l_foot_site_id]
        heading = world_to_heading(body_to_world(self.data.xquat[self.pelvis_id]))
        return float((heading @ offset)[0])

    def _sole_tilts_deg(self) -> np.ndarray:
        """Each plantar pad's tilt from level, in degrees (right, left).

        ``geom_xmat`` is row-major, so element 8 is the world-z component of
        the box's local z axis: the cosine of the angle between the pad
        normal and vertical, whichever way the pad rolls or pitches.
        """
        cos_tilt = np.clip(self.data.geom_xmat[self._plantar_geom_ids, 8], -1.0, 1.0)
        return np.asarray(np.degrees(np.arccos(cos_tilt)))

    def _joint_qpos_indices(self, joint_names: tuple[str, ...]) -> np.ndarray:
        """Resolve scalar hinge-joint qpos addresses, failing on model drift."""
        indices = []
        for name in joint_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                raise ValueError(f"T-Rex model must define joint {name!r}")
            indices.append(int(self.model.jnt_qposadr[joint_id]))
        return np.asarray(indices, dtype=np.int32)

    def _bilateral_support_quality(self, right_force: float, left_force: float) -> float:
        """Return bounded support quality, requiring load on both feet."""
        _, quality = _reward_bilateral_support_pure(
            np.asarray((right_force, left_force)),
            self.foot_contact_saturation_force,
            1.0,
        )
        return float(quality)

    def _foot_load_balance(self, right_force: float, left_force: float) -> tuple[float, float]:
        """Return ``(reward, normalized load imbalance)`` for the foot pair.

        An unsupported pair reports imbalance ``1.0``, not ``0.0`` -- airborne
        is maximally imbalanced, not perfectly balanced -- and additionally
        pays ``foot_load_balance_airborne_penalty`` so that airborne is
        strictly worse than honest single support rather than tied with it.
        """
        reward, imbalance = _reward_foot_load_balance_pure(
            np.asarray((right_force, left_force)),
            self.foot_load_balance_weight,
            self.foot_load_balance_min_support_force,
            self.foot_load_balance_airborne_penalty,
        )
        return float(reward), float(imbalance)

    def _home_pose_quality(
        self,
        qpos_indices: np.ndarray,
        target_qpos: np.ndarray,
        tolerance: float,
        broad_fraction: float = 0.0,
        broad_scale: float = 6.0,
    ) -> tuple[float, float]:
        """Return RMS joint error and a smooth bounded home-pose quality."""
        _, rms_error, quality = _reward_soft_home_pose_pure(
            self.data.qpos[qpos_indices],
            target_qpos,
            tolerance,
            1.0,
            broad_fraction,
            broad_scale,
        )
        return float(rms_error), float(quality)

    def _head_clearance_quality(self, head_tip_z: float) -> float:
        """Reward safe clearance smoothly from zero to the target height."""
        _, quality = _reward_head_clearance_pure(
            np.asarray(head_tip_z),
            self.head_clearance_target,
            self.head_clearance_tolerance,
            1.0,
        )
        return float(quality)

    def _scale_action(self, action: np.ndarray) -> np.ndarray:
        """Map normalized residual actions around the XML home controls.

        The two halves are scaled independently so the mapping preserves the
        actuator endpoints while making action zero exactly the named home
        control, even when that control is not the range midpoint.
        """
        residual = np.clip(action, -1.0, 1.0)
        ctrl_range = self.model.actuator_ctrlrange
        ctrl_min = ctrl_range[:, 0]
        ctrl_max = ctrl_range[:, 1]

        below_home = residual * (self._home_ctrl - ctrl_min)
        above_home = residual * (ctrl_max - self._home_ctrl)
        scaled = self._home_ctrl + np.where(residual < 0.0, below_home, above_home)
        return np.asarray(scaled)

    def _filter_action(self, action: np.ndarray) -> np.ndarray:
        """Keep the policy's own clipped command, then low-pass it as the base class does.

        ``BaseDinoEnv.step`` replaces the command with its filtered value
        before the reward sees it, so this is the only point the pre-filter
        command exists.  Under ``action_penalty_source = "raw"`` the jerk,
        smoothness and saturation terms price it; the filtered command still
        drives the plant and the energy term.  The clip is the base class's
        own (and the floor-truth recorder's), so the reward prices exactly
        the command stance_quality/v2's saturation criterion reads.  Tagged
        with the step count the reward will see (``step`` increments it
        after the filter).
        """
        self._policy_action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        self._policy_action_step = self._step_count + 1
        return super()._filter_action(action)

    def _get_obs(self) -> np.ndarray:
        """Construct the bipedal-target-heading-free/v1 observation vector (63 values).

        Since policy interface r14 no channel carries the world heading: the
        pelvis orientation is projected gravity, the pelvis linear velocity is
        expressed in the gravity-aligned heading frame and the prey direction
        in pelvis coordinates, all three from the free root's own quaternion
        qpos[3:7] (environments/shared/heading_observation.py), never from
        the orientation sensor, which is still zero when the constructor
        calls this before any forward pass.
        """
        # Joint positions (exclude root freejoint: first 7 values)
        qpos = self.data.qpos[7:].copy()

        # Joint velocities (exclude root freejoint: first 6 values)
        qvel = self.data.qvel[6:].copy()

        # Pelvis state from sensors, already in their own site frame (never rotated again)
        pelvis_gyro = self.data.sensordata[self._sensor_gyro_start : self._sensor_gyro_start + 3].copy()
        pelvis_accel = self.data.sensordata[self._sensor_accel_start : self._sensor_accel_start + 3].copy()

        # Foot contact (from touch sensors)
        foot_contact = np.array(self._foot_contact_forces())

        # Prey info (relative to pelvis)
        pelvis_pos = self.data.xpos[self.pelvis_id]
        prey_pos = self.data.mocap_pos[0]
        prey_rel = prey_pos - pelvis_pos
        prey_distance = np.linalg.norm(prey_rel)

        # Normalize prey direction (world frame; turned into pelvis coordinates below)
        prey_direction = prey_rel / (prey_distance + 1e-8)

        # Heading-free root block: projected gravity, the root's world linear
        # velocity (free joint qvel[0:3]) in the heading frame, the prey
        # direction in pelvis coordinates.
        gravity, heading_velocity, prey_direction_body = heading_free_root_segments(
            self.data.qpos, self.data.qvel[0:3], prey_direction
        )

        obs = np.concatenate(
            [
                qpos,  # Joint positions
                qvel,  # Joint velocities
                gravity,  # Projected gravity (unit down vector, pelvis frame)
                pelvis_gyro,  # Angular velocity
                heading_velocity,  # Linear velocity (forward, left, up)
                pelvis_accel,  # Accelerometer
                foot_contact,  # Foot contacts (2)
                prey_direction_body,  # Direction to prey (unit vector, pelvis frame)
                [prey_distance],  # Distance to prey (scalar)
                self._command,  # Body-relative command (v_x, v_y, yaw_rate), pre-scaled; zeros unless command_mode != "none"
            ]
        ).astype(np.float32)

        return obs

    def _get_reward_info(self, action: np.ndarray) -> tuple[float, dict[str, float]]:
        """Compute reward and breakdown for logging."""
        info: dict[str, Any] = {}

        pelvis_pos = self.data.xpos[self.pelvis_id]
        prey_pos = self.data.mocap_pos[0]
        forward_ref_2d = self._initial_prey_dir_2d
        vel_2d = self.data.qvel[0:2]

        # 1-1c. Forward velocity (toward prey), backward velocity penalty and drift penalty
        reward_forward, reward_backward, reward_drift = self._progress_terms(
            info, vel_2d, forward_ref_2d, pelvis_pos[:2]
        )

        # 1d. Bilateral support and load balance.  Touch sensors report the
        # full plantar-plus-digit load for each foot.  Saturation prevents
        # impact spikes from being more valuable than quiet support, while the
        # minimum requires both feet to carry load.
        # AGGREGATED across the control step's physics substeps (per-foot MIN):
        # the last-substep read let the seed-43 bounce unload for 4 of every 5
        # substeps and still collect full support reward, and the same sample
        # feeds the r/l_foot_contact info keys the stance-duty gate certifies
        # from.  Falls back to the instantaneous read on un-stepped states.
        r_contact, l_contact = self._aggregated_foot_contact_forces()
        info["r_foot_contact"] = r_contact
        info["l_foot_contact"] = l_contact

        bilateral_support_quality = self._bilateral_support_quality(r_contact, l_contact)
        reward_bilateral_support = self.bilateral_support_weight * bilateral_support_quality
        info["bilateral_support_quality"] = bilateral_support_quality
        info["reward_bilateral_support"] = reward_bilateral_support

        # Take the pure function's REWARD, not just its diagnostic: the
        # airborne penalty that makes the ordering monotone lives in the
        # reward, so recomputing `-weight * imbalance` here would silently drop
        # it (and did, for the imbalance-only version this replaces).
        reward_foot_load_balance, foot_load_imbalance = self._foot_load_balance(r_contact, l_contact)
        info["foot_load_imbalance"] = foot_load_imbalance
        info["reward_foot_load_balance"] = reward_foot_load_balance

        # 2. Alive bonus (shared helper), optionally conditioned in part on
        # bilateral support.  A fraction below 1 leaves recovery headroom
        # after a noisy reset; the zero default is exactly the legacy bonus.
        raw_alive = self._reward_alive()
        alive_fraction = self.support_conditioned_alive_fraction
        alive_gate = (1.0 - alive_fraction) + alive_fraction * bilateral_support_quality
        reward_alive = raw_alive * alive_gate
        info["raw_alive"] = raw_alive
        info["alive_gate"] = alive_gate
        info["reward_alive"] = reward_alive

        # 3. Energy penalty (shared helper).  Always the APPLIED command, under
        # either action_penalty_source: it is an actuation cost, and the
        # actuators track the filtered command.
        reward_energy = self._reward_energy(action)
        info["reward_energy"] = reward_energy

        # 4. Tail stability
        reward_tail, tail_instability = self._compute_tail_stability(self.tail_tip_site_id, self.tail_stability_weight)
        info["tail_instability"] = tail_instability
        info["reward_tail"] = reward_tail

        # 5. Bite bonus (check head_bite-prey contact)
        bitten = self._contact_geom({self.head_bite_geom_id}, self.prey_geom_id) is not None
        info["bite_success"] = 1.0 if bitten else 0.0
        reward_bite = self.bite_bonus if bitten else 0.0
        info["reward_bite"] = reward_bite

        # 6. Approach shaping
        prey_distance = float(np.linalg.norm(prey_pos - pelvis_pos))
        reward_approach, approach_delta = self._compute_approach_shaping(
            prey_distance, self._prev_prey_distance, self.bite_approach_weight, 8.0
        )
        self._prev_prey_distance = prey_distance
        info["prey_distance"] = prey_distance
        info["approach_delta"] = approach_delta
        info["reward_approach"] = reward_approach

        # 6b. Head-to-prey proximity shaping
        # Uses the head tip position (not the pelvis) to give the agent a
        # gradient for aiming its head toward the prey.  Analogous to the
        # raptor's claw proximity reward.
        head_tip_pos = self.data.site_xpos[self.head_tip_site_id]
        head_prey_dist = float(np.linalg.norm(prey_pos - head_tip_pos))
        head_proximity_max_dist = 3.0
        head_proximity = max(0.0, 1.0 - head_prey_dist / head_proximity_max_dist)
        reward_head_proximity = self.bite_head_proximity_weight * head_proximity
        info["head_prey_distance"] = head_prey_dist
        info["head_proximity"] = head_proximity
        info["reward_head_proximity"] = reward_head_proximity

        # 6c. Head clearance and neck posture.  Clearance is a smoothstep:
        # zero one configured tolerance below the target and saturated at the
        # target so lifting the head ever higher cannot farm reward.  The neck
        # target is the authored home keyframe, or the statue's settled neck
        # under neck_posture_reference = "settled" (see _cache_ids).
        head_tip_z = self._clearance(head_tip_pos)
        head_clearance_quality = self._head_clearance_quality(head_tip_z)
        reward_head_clearance = self.head_clearance_weight * head_clearance_quality
        info["head_tip_z"] = head_tip_z
        info["head_pelvis_rel_z"] = float(head_tip_pos[2]) - float(pelvis_pos[2])
        info["head_clearance_quality"] = head_clearance_quality
        info["reward_head_clearance"] = reward_head_clearance

        neck_posture_error, neck_posture_quality = self._home_pose_quality(
            self._neck_home_qpos_indices,
            self._neck_home_qpos,
            self.neck_posture_tolerance,
        )
        reward_neck_posture = self.neck_posture_weight * neck_posture_quality
        info["neck_posture_error"] = neck_posture_error
        info["neck_posture_quality"] = neck_posture_quality
        info["reward_neck_posture"] = reward_neck_posture

        # 6b. Tail home pose.  The tail stability term (6) reads only the
        # tail-tip gyro, so a tail frozen against a hard stop is optimal for
        # it; this prices the joint positions themselves.
        tail_home_pose_error, tail_home_pose_quality = self._home_pose_quality(
            self._tail_home_qpos_indices,
            self._tail_home_qpos,
            self.tail_home_pose_tolerance,
        )
        reward_tail_home_pose = self.tail_home_pose_weight * tail_home_pose_quality
        info["tail_home_pose_error"] = tail_home_pose_error
        info["tail_home_pose_quality"] = tail_home_pose_quality
        info["reward_tail_home_pose"] = reward_tail_home_pose

        # 7. Continuous posture reward
        pelvis_quat = self.data.sensordata[self._sensor_quat_start : self._sensor_quat_start + 4]
        reward_posture, tilt_angle = self._compute_posture_reward(pelvis_quat, self.posture_weight)
        info["tilt_angle"] = tilt_angle
        info["reward_posture"] = reward_posture

        # 8. Nosedive penalty
        reward_nosedive = self._nosedive_term(info, pelvis_quat)

        # 8b. Pelvis height (for LocomotionMetrics tracking)
        pelvis_height = self._clearance(self.data.xpos[self.pelvis_id])
        info["pelvis_height"] = pelvis_height

        # 8c. Height maintenance reward (smooth gradient toward staying upright)
        min_z = self.healthy_z_range[0]
        # Settled pelvis height under the home controller, measured (zero action,
        # zero reset noise, 600 steps: 0.9260 m, sd 0.00001, on physics r8; 0.9262
        # on r9, which carries it: plant_versions note 16; 0.9261 on r10, which
        # carries it too: note 20).  This must track the
        # plant: the previous 0.90 was the root height of the PRE-repair plant,
        # and after the July home-equilibrium fix raised the stance to 0.9757 the
        # term saturated at 1.0 everywhere in the healthy band -- a flat constant
        # worth 39% of stage-1 and 10% of stage-2 return that shaped nothing.
        # ``TestHeightTargetTracksStance`` pins it to the measured stance, which
        # is how the 0.9757 -> 0.9260 move under the theropod stance correction
        # was caught rather than left to drift.
        target_z = 0.9260
        if self.height_target_tolerance > 0.0:
            reward_height, height_error, height_quality = _reward_target_centered_height_pure(
                np.asarray(pelvis_height),
                target_z,
                self.height_target_tolerance,
                self.height_weight,
            )
            reward_height = float(reward_height)
            height_error = float(height_error)
            height_quality = float(height_quality)
        else:
            # Legacy one-sided reward: retain it exactly unless the explicit
            # target-centred tolerance switch is enabled.
            height_error = abs(pelvis_height - target_z)
            height_quality = float(np.clip((pelvis_height - min_z) / (target_z - min_z), 0.0, 1.0))
            reward_height = self.height_weight * height_quality
        info["height_error"] = height_error
        info["height_quality"] = height_quality
        info["reward_height"] = reward_height

        # 8d. Soft leg-pose retention around the authored p5 home stance.
        # Gaussian quality gives corrective signal without hard-locking a
        # joint, and averaging prevents one joint from dominating the term.
        leg_home_pose_error, leg_home_pose_quality = self._home_pose_quality(
            self._leg_home_qpos_indices,
            self._leg_home_settled_qpos if self.leg_home_pose_reference == "settled" else self._leg_home_qpos,
            self.leg_home_pose_tolerance,
            self.leg_home_pose_broad_fraction,
            self.leg_home_pose_broad_scale,
        )
        reward_leg_home_pose = self.leg_home_pose_weight * leg_home_pose_quality
        info["leg_home_pose_error"] = leg_home_pose_error
        info["leg_home_pose_quality"] = leg_home_pose_quality
        info["reward_leg_home_pose"] = reward_leg_home_pose

        # 8e. Sole flatness.  The ankle has no roll joint, so hip roll alone
        # sets each pad's roll: the 2026-09 stances splayed one leg by
        # 2.5-4.4 deg and stood that foot on the pad's front-inner corner
        # (the 2026-10 audit: back-outer corner 12-18 mm up, outer toe off
        # the floor on 77-95% of steps), which the touch sensors, the support
        # terms above and the leg pose term (about 21/episode for that hip
        # roll) all priced as a sound stance.  Per-pad exp(-(tilt/tol)^2), averaged over the feet,
        # so one rolled foot forfeits at most half the term.
        # Under foot_terms_min_support_force > 0 a foot earns its share only
        # while its substep-MIN touch carries that load: a lifted foot is
        # level in the air, and the two physics-r8 runs' hopping seed earned
        # ~80% of this term on its airborne steps.
        sole_tilts = self._sole_tilts_deg()
        feet_supported = (
            r_contact >= self.foot_terms_min_support_force,
            l_contact >= self.foot_terms_min_support_force,
        )
        if self.foot_terms_min_support_force > 0.0:
            reward_foot_flatness, foot_flatness_quality = _reward_sole_flatness_pure(
                sole_tilts,
                self.foot_flatness_tolerance_deg,
                self.foot_flatness_weight,
                foot_weights=np.asarray(feet_supported, dtype=np.float64),
            )
        else:
            reward_foot_flatness, foot_flatness_quality = _reward_sole_flatness_pure(
                sole_tilts, self.foot_flatness_tolerance_deg, self.foot_flatness_weight
            )
        info["r_foot_terms_supported"] = float(feet_supported[0])
        info["l_foot_terms_supported"] = float(feet_supported[1])
        info["r_sole_tilt_deg"] = float(sole_tilts[0])
        info["l_sole_tilt_deg"] = float(sole_tilts[1])
        info["foot_flatness_quality"] = float(foot_flatness_quality)
        info["reward_foot_flatness"] = float(reward_foot_flatness)

        # 8f. Stance width.  A Gaussian on the planar foot-site distance
        # (_stance_width: a fore-aft stagger counts too) around the home
        # keyframe's own (_cache_ids).  The same 2026-09 stances settled
        # 0.344-0.347 m wide on average against the statue's 0.279; the
        # propped leg is what widens the stance, so width and flatness price
        # one exploit through two independent measurements.
        # Under stance_width_reference = "settled" the target is the
        # animal's OWN width at step stance_width_settle_steps, and the term
        # pays only after it: centred on the keyframe it paid a policy for
        # re-seating its feet in the settle (both physics-r8 runs), which a
        # reference the settle itself sets cannot reward.  Under
        # foot_terms_min_support_force > 0 it pays only while both feet carry
        # that load.
        stance_width = self._stance_width(self.data)
        settled_reference = self.stance_width_reference == "settled"
        if settled_reference and self._step_count == self.stance_width_settle_steps:
            self._stance_width_target = stance_width
        reward_stance_width, stance_width_error, stance_width_quality = _reward_stance_width_pure(
            np.asarray(stance_width),
            self._stance_width_target,
            self.stance_width_tolerance_m,
            self.stance_width_weight,
        )
        if settled_reference and self._step_count <= self.stance_width_settle_steps:
            reward_stance_width = 0.0
        if self.foot_terms_min_support_force > 0.0 and not all(feet_supported):
            reward_stance_width = 0.0
        info["stance_width"] = stance_width
        info["stance_width_target"] = float(self._stance_width_target)
        info["stance_width_error"] = float(stance_width_error)
        info["stance_width_quality"] = float(stance_width_quality)
        info["reward_stance_width"] = float(reward_stance_width)

        # 8g. Standing lateral lean (off unless lateral_lean_weight > 0; its
        # info keys exist only then).  A Gaussian bonus on the whole-body
        # centre of mass's offset from the foot-site midpoint along the
        # heading frame's left axis (_com_lateral_offset), paid only after the
        # settle (step > stance_width_settle_steps), only while the command is
        # zero and only while both feet carry foot_terms_min_support_force.
        # The 2026-10 stances that failed the pad rails leaned onto one pad's
        # edge with both feet down, which the support and width terms price
        # as a sound stance.  The offset flips sign under the left-right
        # mirror and enters squared, and every gate is mirror-invariant, so
        # the term is exactly mirror-invariant.  Calibrated on stored stance
        # rollouts (physics r9): weight 0.25 and tolerance 0.015 m; the
        # statue then forfeits 1.7 per episode on average (0.3 median, 16.4 on
        # its most-leaning spawn) against 105-141 more on episodes that
        # stood on a rolled pad, and the window mean offset separates the two
        # with AUC 0.99.  A bonus adds about 0.25 x 800 = 200 to every
        # standing episode's return: re-measure stance.toml's
        # collapse_peak_floor_reference before enabling it there, and add
        # reward_lateral_lean to DiagnosticsCallback.REWARD_KEYS and to
        # visualization's reward-decomposition weight map in the same change
        # (left out while no stage pays it, so no run records an empty series).
        if self.lateral_lean_weight > 0.0:
            com_lateral_offset = self._com_lateral_offset()
            reward_lateral_lean, lateral_lean_quality = _reward_lateral_lean_pure(
                np.asarray(com_lateral_offset), self.lateral_lean_tolerance_m, self.lateral_lean_weight
            )
            lean_gate = (
                self._step_count > self.stance_width_settle_steps and not np.any(self._command) and all(feet_supported)
            )
            info["com_lateral_offset_m"] = com_lateral_offset
            info["lateral_lean_quality"] = float(lateral_lean_quality)
            info["reward_lateral_lean"] = float(reward_lateral_lean) if lean_gate else 0.0

        # 8h. Standing foot stagger (screen-only, never merged; off unless
        # stance_stagger_weight > 0, its info keys exist only then).  A
        # Gaussian PENALTY on the fore-aft stagger of the two foot sites along
        # the pelvis heading (_foot_stagger), centred on 0 (the statue's
        # window mean |stagger| is 2.0 mm), paid only after the settle (step >
        # stance_width_settle_steps) and only while the command is zero.  Not
        # gated on support: a penalty that a lifted foot escaped would pay the
        # policy to unload a foot.  -w (1 - exp(-(s/tol)^2)); the stagger
        # enters squared and both gates are mirror-invariant, so the term is
        # exactly mirror-invariant.  Ported unchanged from the 2026-10-19 night
        # continuation (SP/foot/stance3/night/design/spec.md section 3.1) for
        # the stance4 screen (w 0.30, tol 0.045 m).  Like reward_lateral_lean,
        # it is not in DiagnosticsCallback.REWARD_KEYS.
        if self.stance_stagger_weight > 0.0:
            foot_stagger = self._foot_stagger()
            reward_stance_stagger, stagger_quality = _reward_stance_stagger_pure(
                np.asarray(foot_stagger), self.stance_stagger_tolerance_m, self.stance_stagger_weight
            )
            stagger_gate = self._step_count > self.stance_width_settle_steps and not np.any(self._command)
            info["foot_stagger_m"] = foot_stagger
            info["stance_stagger_quality"] = float(stagger_quality)
            info["reward_stance_stagger"] = float(reward_stance_stagger) if stagger_gate else 0.0

        # 8i. Per-foot centre-of-pressure band (scratch prototype for the
        # stance4 fallback, never merged, never trained; off unless
        # stance_cop_band_weight > 0, its info keys exist only then).  A
        # Gaussian PENALTY on each foot's whole-foot CoP (the judge's K2b
        # reading for this step: every floor contact of the limb over the
        # step's substeps, in span half-lengths, floor_contact/step_cop.py)
        # outside [stance_cop_band_lower, stance_cop_band_upper], with the pad's own
        # CoP held to [stance_cop_band_pad_lower, stance_cop_band_pad_upper]
        # as well, so load moved onto the digits cannot pay the band (guard 2).
        # Each foot's quality is scaled by a guard factor (guard 1, a ramp): 0
        # for a foot that is unloaded or stands off its pad, or carries at
        # most stance_cop_band_full_price_load_share of the two limbs' floor
        # load, 1 from stance_cop_band_min_load_share up, linear between.  A
        # foot whose guard is 0 pays the per-foot maximum, so lifting a foot
        # never lowers the price, and at fixed readings a lighter foot never
        # pays less; the price has no step at the bar.  The ramp leaves a
        # bounded door: a lighter foot whose CoP is back in band pays 1 - g of
        # the per-foot maximum, less than a loaded foot u out of band when
        # g > exp(-(u / tolerance)^2) (at the defaults from share 0.30 for
        # u 0.3, never at 0.25 or below).  Paid only after the settle and only
        # while the command is zero, like the stagger term.  -w mean_feet
        # (1 - q); every reading is fore-aft, the guard is per foot and the
        # mean is symmetric, so the term is exactly mirror-invariant.
        # Defaults (spans of 70 mm): band [-0.6, +0.4] (the split diagnosis'
        # tight band, AUC 1.000 on both failing panels), pad band
        # [-0.65, +0.65] (the band widened by the digits' observed
        # whole-foot-minus-pad shift, p1..p99 -0.25..+0.04), tolerance 0.357
        # (25 mm), the guard's ramp from share 0.25 to 0.35 (the v2 gate's
        # windowed load-share bar; the maintainer's 2026-10-20 decision 3a).
        # The proposed dose is w 0.30, the stagger term's maximum (stance4
        # copband design.md).  Info per foot: <s>_cop_band_guard is the guard
        # factor and <s>_cop_band_counted is 1 only at a guard of 1 (the
        # design's counted rule), so 1 - counted is how often a guard fires
        # and 1 - guard how much.  The arithmetic is
        # reward_functions.cop_band_load_guard's and reward_stance_cop_band's,
        # operation for operation, on the two feet as Python floats (their
        # 2-element numpy calls cost about 35 us a step), keeping their one
        # np.exp over both feet (math.exp moves some values by an ulp): the
        # band's tests pin every value to theirs byte for byte.  Like
        # reward_stance_stagger, it is not in DiagnosticsCallback.REWARD_KEYS:
        # a listed key that no info carries is still written to every run's
        # diagnostics.npz as a NaN series, so listing it would change the
        # band-off runs' records.
        if self.stance_cop_band_weight > 0.0:
            reading = self._cop_band().reading(self._step_count, self.model, self.data)
            feet = list(
                zip(
                    reading.foot_cop.tolist(),
                    reading.foot_loaded.tolist(),
                    reading.pad_cop.tolist(),
                    reading.pad_loaded.tolist(),
                    reading.leg_force_mean.tolist(),
                    reading.digit_share.tolist(),
                )
            )
            limb_total = float(reading.leg_force_mean.sum())
            full_price = self.stance_cop_band_full_price_load_share
            ramp = self.stance_cop_band_min_load_share - full_price  # > 0: the validation's full_price < min
            lower, upper = self.stance_cop_band_lower, self.stance_cop_band_upper
            pad_lower, pad_upper = self.stance_cop_band_pad_lower, self.stance_cop_band_pad_upper
            tolerance = max(self.stance_cop_band_tolerance, 1e-8)
            shares, guards, exponents = [], [], []
            for foot_cop, foot_loaded, pad_cop, pad_loaded, limb, _digit_share in feet:
                share = limb / limb_total if limb_total > 0.0 else 0.0
                loaded = foot_loaded and pad_loaded and math.isfinite(share)
                guard = min(max((share - full_price) / ramp, 0.0), 1.0) if loaded else 0.0
                excess = 0.0  # a foot whose guard is 0 scores 0: its readings (NaN when unloaded) are not read
                if guard > 0.0:
                    whole = max(lower - foot_cop, 0.0) + max(foot_cop - upper, 0.0)
                    pad = max(pad_lower - pad_cop, 0.0) + max(pad_cop - pad_upper, 0.0)
                    excess = max(whole, pad)
                scaled = excess / tolerance
                shares.append(share)
                guards.append(guard)
                exponents.append(-(scaled * scaled))
            gaussian = np.exp(exponents).tolist()
            band_quality = [guard * value if guard > 0.0 else 0.0 for guard, value in zip(guards, gaussian)]
            band_gate = self._step_count > self.stance_width_settle_steps and not np.any(self._command)
            for index, label in enumerate(("r", "l")):
                foot_cop, foot_loaded, pad_cop, pad_loaded, _limb, digit_share = feet[index]
                info[f"{label}_cop_band_foot_cop"] = foot_cop if foot_loaded else 0.0
                info[f"{label}_cop_band_pad_cop"] = pad_cop if pad_loaded else 0.0
                info[f"{label}_cop_band_load_share"] = shares[index]
                info[f"{label}_cop_band_digit_share"] = digit_share if math.isfinite(digit_share) else 0.0
                info[f"{label}_cop_band_guard"] = guards[index]
                info[f"{label}_cop_band_counted"] = 1.0 if guards[index] >= 1.0 else 0.0
                info[f"{label}_cop_band_quality"] = band_quality[index]
            info["stance_cop_band_quality"] = (band_quality[0] + band_quality[1]) / 2.0
            reward_stance_cop_band = -self.stance_cop_band_weight * (
                ((1.0 - band_quality[0]) + (1.0 - band_quality[1])) / 2.0
            )
            info["reward_stance_cop_band"] = reward_stance_cop_band if band_gate else 0.0

        # 9. Gait symmetry (reward alternating foot contacts, shared helper)
        reward_gait, alternation_ratio = self._compute_gait_symmetry(
            float(r_contact), float(l_contact), self.gait_symmetry_weight
        )
        info["alternation_ratio"] = alternation_ratio
        info["contact_asymmetry"] = alternation_ratio  # backward compat with metrics
        info["reward_gait"] = reward_gait

        # 9b. Floor impact and airborne substeps, from every physics substep's
        # per-foot touch (BaseDinoEnv.step keeps the block beside the MIN the
        # support terms read).  Touch is pad plus digits, and nothing but the
        # feet touches the floor while the animal is standing, so the summed
        # touch is the animal's floor force: it equals floor truth to 1e-12 N
        # on every recorded step of the physics-r8 panels.  The impact term
        # prices the step's PEAK above floor_impact_threshold_bw body
        # weights; the airborne term the share of the step's substeps on
        # which every foot is unloaded.  Both run over the whole episode,
        # settle included, where the support terms' step-level MIN and the
        # unscored-in-v1 settle let a stomp or a two-foot hop go unpriced.
        foot_force_block = self._substep_foot_force_block()
        peak_foot_force_bw = float(foot_force_block.sum(axis=1).max()) / self._body_weight_n
        airborne_substeps = int(np.count_nonzero(np.all(foot_force_block <= self._AIRBORNE_TOUCH_N, axis=1)))
        reward_floor_impact = 0.0
        if self.floor_impact_weight > 0.0:
            reward_floor_impact = -self.floor_impact_weight * max(
                0.0, peak_foot_force_bw - self.floor_impact_threshold_bw
            )
        reward_airborne_substeps = 0.0
        if self.airborne_substep_weight > 0.0:
            reward_airborne_substeps = -self.airborne_substep_weight * airborne_substeps / foot_force_block.shape[0]
        info["peak_foot_force_bw"] = peak_foot_force_bw
        info["airborne_substeps"] = float(airborne_substeps)
        info["reward_floor_impact"] = float(reward_floor_impact)
        info["reward_airborne_substeps"] = float(reward_airborne_substeps)

        # 10. Action smoothness (shared helper)
        # Jerk BEFORE smoothness: _reward_action_smoothness rotates the action
        # history, so calling it first would leave the jerk term reading this
        # step's own action as its first lag.
        # Under action_penalty_source = "raw" jerk, smoothness and saturation
        # price the policy's own command (_filter_action), and the action
        # history they keep holds raw commands; the filtered one still drives
        # the plant and the energy term.  The first-order 10 Hz filter passes
        # 24-54% of 12.5-50 Hz content, so a +-1 bang-bang that hops the
        # animal was priced about 9-11x too cheaply on the filtered command.
        # Without the filter (or before any step) the raw command IS
        # ``action``.
        penalty_action = action
        if (
            self.action_penalty_source == "raw"
            and self._policy_action is not None
            and self._policy_action_step == self._step_count
        ):
            penalty_action = self._policy_action
        reward_action_jerk, action_jerk = self._reward_action_jerk(penalty_action)
        info["action_jerk"] = action_jerk
        info["reward_action_jerk"] = reward_action_jerk

        reward_smoothness, action_delta = self._reward_action_smoothness(penalty_action)
        info["action_delta"] = action_delta
        info["reward_smoothness"] = reward_smoothness

        # 10b. Saturation cost.  A command pinned at a range limit is
        # invisible to the smoothness/jerk penalties above -- it cannot
        # oscillate -- so parking joints at stops was the cheapest way to
        # be smooth.  Price the parked fraction directly.
        reward_action_saturation, action_saturation = _reward_action_saturation_pure(
            penalty_action, self.action_saturation_weight, self.action_saturation_threshold
        )
        info["action_saturation"] = float(action_saturation)
        info["reward_action_saturation"] = float(reward_action_saturation)

        # 11-12. Heading alignment and lateral velocity penalty
        reward_heading, reward_lateral = self._heading_terms(info, pelvis_quat, forward_ref_2d, vel_2d)

        # Pelvis angular velocity (for spinning detection in shared diagnostics)
        pelvis_angular_vel, pelvis_yaw_vel = self._compute_pelvis_diagnostics()
        info["pelvis_angular_vel"] = pelvis_angular_vel
        info["pelvis_yaw_vel"] = pelvis_yaw_vel

        # 13. Spin penalty
        reward_spin, spin_instability = self._compute_angular_velocity_penalty(self.spin_penalty_weight)
        info["spin_instability"] = spin_instability
        info["reward_spin"] = reward_spin

        # 14-14b. Speed penalty above its threshold, idle penalty below its own
        reward_speed, reward_idle = self._speed_terms(info, vel_2d)

        # Total reward
        total_reward = (
            reward_forward
            + reward_backward
            + reward_drift
            + reward_alive
            + reward_energy
            + reward_tail
            + reward_bite
            + reward_approach
            + reward_head_proximity
            + reward_head_clearance
            + reward_neck_posture
            + reward_tail_home_pose
            + reward_posture
            + reward_nosedive
            + reward_height
            + reward_bilateral_support
            + reward_foot_load_balance
            + reward_leg_home_pose
            + reward_foot_flatness
            + reward_stance_width
            + reward_gait
            + reward_smoothness
            + reward_action_jerk
            + reward_action_saturation
            + reward_heading
            + reward_lateral
            + reward_spin
            + reward_speed
            + reward_idle
        )
        # Added only when enabled, so the legacy sum keeps its exact order.
        if self.floor_impact_weight > 0.0:
            total_reward += reward_floor_impact
        if self.airborne_substep_weight > 0.0:
            total_reward += reward_airborne_substeps
        if self.lateral_lean_weight > 0.0:
            total_reward += info["reward_lateral_lean"]
        if self.stance_stagger_weight > 0.0:
            total_reward += info["reward_stance_stagger"]
        if self.stance_cop_band_weight > 0.0:
            total_reward += info["reward_stance_cop_band"]
        info["reward_total"] = total_reward

        return total_reward, info

    def _is_terminated(self) -> tuple[bool, dict[str, Any]]:
        """Check if episode should terminate."""
        # Height/tilt, then nosedive termination (shared)
        terminated, info = self._root_termination(self.pelvis_id, "pelvis_height", self.nosedive_termination_threshold)
        if terminated:
            return True, info

        # Site-height termination: snout tip must stay above threshold
        # This catches nose-balancing that geom contact detection may miss.
        # The check reads the substep MIN (indices match the
        # _substep_height_checks declaration) so a dip that recovers between
        # control-boundary samples still terminates; the info key keeps the
        # boundary sample.
        head_tip_z = self._clearance(self.data.site_xpos[self.head_tip_site_id])
        info["head_tip_z"] = head_tip_z
        if self._aggregated_min_height(0, head_tip_z) < 0.12:
            info["termination_reason"] = "head_contact"
            return True, info

        # Body-height termination: skull origin must stay above threshold
        skull_z = self._clearance(self.data.xpos[self.skull_body_id])
        if self._aggregated_min_height(1, skull_z) < 0.45:
            info["termination_reason"] = "skull_low"
            return True, info

        # Check contacts: head-prey (success)
        if self.bite_bonus > 0 and self._contact_geom({self.head_bite_geom_id}, self.prey_geom_id) is not None:
            info["termination_reason"] = "bite_success"
            info["success"] = True
            return True, info

        # Floor contact termination (shared)
        terminated, reason = self._check_floor_contact(
            self._body_ground_geoms,
            self.floor_geom_id,
            geom_categories={
                "tail": self._tail_ground_geoms,
                "head": self._head_ground_geoms,
                "torso": {self.torso_geom_id},
            },
        )
        if terminated:
            info["termination_reason"] = reason
            return True, info

        return False, info

    def _spawn_target(self):
        """Spawn prey at random location ahead of T-Rex."""
        prey_pos = self._spawn_target_2d(self.prey_distance_range, self.prey_lateral_range, 0.5)
        self._initial_prey_dir_2d = self._compute_initial_direction_2d(prey_pos)
        self._initial_pos_2d = self.data.qpos[0:2].copy()

        # Reset delta-based tracking (first step will produce zero deltas)
        self._prev_prey_distance = None
        self._prev_action = None
        self._prev_prev_action = None
        self._policy_action = None
        self._policy_action_step = -1
        # The settled width target is the episode's own (see 8f).
        self._stance_width_target = self._home_stance_width

        # Reset gait symmetry tracking
        self._reset_gait_state()
