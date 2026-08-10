"""T-Rex MJX species configuration.

Registers the T-Rex species with the MJX environment so that
``MJXDinoEnv("trex", ...)`` works out of the box.
"""

from __future__ import annotations

import math

from environments.shared.mjx_env import register_species_mjx

# Must track ``TRexEnv.__init__``'s ``natural_pitch`` default (trex_env.py).
# The pelvis frame is level at the home keyframe and settles ~1.55 deg
# nose-down under the home controller; the nosedive penalty and termination
# are both measured relative to that pose.  Kept as the angle rather than a
# rounded forward_z so the two paths derive the same number from the same
# quantity -- test_mjx_env.test_trex_creation pins them equal.
_NATURAL_PITCH = 0.027

# Sensor indices match the MJCF sensor definition order:
# pelvis_gyro(3), pelvis_accel(3), pelvis_orientation(4),
# r_foot_touch(1), l_foot_touch(1)
_SENSOR_R_FOOT = 10
_SENSOR_L_FOOT = 11
# The two sensors above cover the plantar pad only.  Each digit is its own
# body (it articulates on a passive hinge), and a touch sensor cannot see geoms on
# child bodies, so the digits carry their own sensors -- appended after the
# tail block so every index above keeps its position.  Summed per foot, they
# restore the reading to the force the foot actually transmits.
_SENSOR_R_FOOT_DIGITS = (24, 25, 26)
_SENSOR_L_FOOT_DIGITS = (27, 28, 29)
# Tail tip gyro starts after: gyro(3) + accel(3) + quat(4) + touch(2) + head_pos(3) + tail_pos(3) + tail_linvel(3) = 21
_SENSOR_TAIL_GYRO_START = 21

register_species_mjx(
    species="trex",
    action_mapping="home-keyframe-residual/v1",
    frame_skip=5,
    max_episode_steps=1000,
    healthy_z_range=(0.70, 1.55),
    # Matches the Gymnasium env (the BaseDinoEnv default), which is the value
    # the evidence supports.  max_tilt_angle is the absolute backstop; nosedive
    # is the per-stage tunable pitch gate.  Stages 2 and 3 leave
    # nosedive_termination_threshold at the 0.62 default, which allows 0.703 rad
    # of forward pitch (asin(sin(0.027) + 0.62); it was 0.734 before the
    # theropod stance moved natural_pitch 0.05 -> 0.027) -- so a 0.700 cap would
    # terminate first and silently override that calibration in exactly the
    # stages whose configs ask for a head-forward running posture.  It also
    # normalises the posture penalty as (tilt / max_tilt_angle)**2, and
    # posture_weight = 1.5 is a swept constant shared by all four species
    # at 1.047.  The 0.62 the paragraph above relies on is only true because
    # this registration now declares it -- see the reward_weights block.
    max_tilt_angle=1.047,
    sensor_foot_indices=(_SENSOR_R_FOOT, _SENSOR_L_FOOT),
    sensor_foot_aux_indices=(_SENSOR_R_FOOT_DIGITS, _SENSOR_L_FOOT_DIGITS),
    sensor_gyro_start=0,
    sensor_accel_start=3,
    sensor_quat_start=6,
    natural_forward_z=-math.sin(_NATURAL_PITCH),
    sensor_tail_gyro_start=_SENSOR_TAIL_GYRO_START,
    forward_vel_max=8.0,
    fall_penalty=-100.0,
    target_standing_z=0.9260,  # SB3 height-maintenance target (trex_env.py); settled stance
    target_distance_range=(3.0, 8.0),
    target_lateral_range=(-2.0, 2.0),
    target_z=0.5,
    body_ids={"pelvis": 2},  # MuJoCo body ID for pelvis (world=0, prey=1)
    leg_home_pose_joint_names=(
        "r_hip_pitch",
        "r_hip_roll",
        "r_knee",
        "r_ankle",
        "l_hip_pitch",
        "l_hip_roll",
        "l_knee",
        "l_ankle",
    ),
    neck_posture_joint_names=("neck_pitch", "neck_yaw", "head_pitch"),
    # Joint-position targets for the tail home-pose term; the tail stability
    # term reads only the tail-tip gyro, which a tail parked at a stop
    # satisfies perfectly (2026-08 narrow-tolerance run).
    tail_home_pose_joint_names=("tail_1_pitch", "tail_1_yaw", "tail_2_pitch", "tail_3_pitch"),
    # Statue-derived settled droop, NOT the keyframe zeros: the passive tail
    # rests on its ventral stops under gravity (sub-milliradian spread, 40
    # seeds, noise 0.05).  Must match TRexEnv._TAIL_SETTLED_QPOS; the stage-1
    # factory test pins the agreement.
    tail_home_pose_targets=(-0.2107, 0.0, -0.2029, -0.0926),
    head_clearance_site="head_tip",
    termination_body_heights={
        "skull": 0.45,  # raised from 0.15: skull body origin must stay above ~half standing height
        "torso": 0.25,  # torso capsule radius=0.18 + margin
        "r_thigh": 0.20,  # prevent lying on thighs
        "l_thigh": 0.20,  # prevent lying on thighs
        "tail_3": 0.10,  # tail_3 capsule radius=0.08 + margin
        "tail_4": 0.08,  # tail_4 capsule radius=0.06 + margin
        "tail_5": 0.05,  # tail_5 capsule radius=0.035 + margin
    },
    termination_site_heights={
        "head_tip": 0.12,  # snout tip — terminates nose-balancing (exploit z ~0.06, natural droop reaches ~0.15 at step 100)
    },
    # Stage 3 success: head_tip site proximity to prey.
    # Gated by reward_weights["bite_bonus"] — inactive when 0 (stages 1-2).
    success_sites=("head_tip",),
    success_threshold=0.35,  # head_bite box (0.06) + prey sphere (0.25) + margin
    success_bonus_key="bite_bonus",
    reward_weights={
        "forward_vel_weight": 1.0,
        "alive_bonus": 0.1,
        "energy_penalty_weight": 0.001,
        "posture_weight": 0.2,
        # Canonical key: registry weights bypass canonicalize_env_kwargs, so a
        # legacy "bite_approach_weight" here would shadow the TOML-configured
        # approach_weight in every stage.
        "approach_weight": 1.0,
        "tail_stability_weight": 0.05,
        "smoothness_weight": 0.05,
        "bite_bonus": 10.0,
        # Not a weight: the per-stage pitch gate mjx_env reads out of this
        # dict.  Only stage 1 sets it in TOML, and without an entry here the
        # MJX path fell back to mjx_env's generic 0.5 while the Gymnasium env
        # used TRexEnv.__init__'s 0.62 -- 8.5 deg stricter, in exactly the two
        # stages whose configs ask for a head-forward running posture.
        # Registry weights merge first and TOML overlays them, so stage 1's
        # calibrated 0.493 still wins.  Must track the SB3 default; pinned by
        # TestSB3MJXEnvelopeParity.
        "nosedive_termination_threshold": 0.62,
        # Stage-1 stance refinements are opt-in.  Keeping every weight/fraction
        # at zero preserves the historical MJX rewards for stages/configs that
        # do not explicitly enable them.
        "bilateral_support_weight": 0.0,
        "foot_contact_saturation_force": 100.0,
        "foot_load_balance_weight": 0.0,
        "support_conditioned_alive_fraction": 0.0,
        "leg_home_pose_weight": 0.0,
        "leg_home_pose_tolerance": 0.35,
        "head_clearance_weight": 0.0,
        "head_clearance_target": 0.60,
        "head_clearance_tolerance": 0.48,
        "neck_posture_weight": 0.0,
        "neck_posture_tolerance": 0.35,
        "tail_home_pose_weight": 0.0,
        "tail_home_pose_tolerance": 0.10,
        "action_saturation_weight": 0.0,
        "action_saturation_threshold": 0.9,
        "leg_home_pose_broad_fraction": 0.0,
        "leg_home_pose_broad_scale": 6.0,
        "height_target_tolerance": 0.0,
    },
)
