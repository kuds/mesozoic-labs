"""T-Rex MJX species configuration.

Registers the T-Rex species with the MJX environment so that
``MJXDinoEnv("trex", ...)`` works out of the box.
"""

from __future__ import annotations

import math

from environments.shared.mjx_env import register_species_mjx

# Must track ``TRexEnv.__init__``'s ``natural_pitch`` default (trex_env.py).
# The pelvis frame is level at the home keyframe and settles ~2.9 deg
# nose-down under the home controller; the nosedive penalty and termination
# are both measured relative to that pose.  Kept as the angle rather than a
# rounded forward_z so the two paths derive the same number from the same
# quantity -- test_mjx_env.test_trex_creation pins them equal.
_NATURAL_PITCH = 0.05

# Sensor indices match the MJCF sensor definition order:
# pelvis_gyro(3), pelvis_accel(3), pelvis_orientation(4),
# r_foot_touch(1), l_foot_touch(1)
_SENSOR_R_FOOT = 10
_SENSOR_L_FOOT = 11
# The two sensors above cover the plantar pad only.  Each digit is its own
# body (it carries an actuated hinge), and a touch sensor cannot see geoms on
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
    healthy_z_range=(0.75, 1.6),
    # Matches the Gymnasium env (the BaseDinoEnv default), which is the value
    # the evidence supports.  max_tilt_angle is the absolute backstop; nosedive
    # is the per-stage tunable pitch gate.  Stages 2 and 3 leave
    # nosedive_termination_threshold at the 0.62 default, which allows 0.734 rad
    # of forward pitch -- so a 0.700 cap would terminate first and silently
    # override that calibration in exactly the stages whose configs ask for a
    # head-forward running posture.  It also normalises the posture penalty as
    # (tilt / max_tilt_angle)**2, and posture_weight = 1.5 is a swept constant
    # shared by all four species at 1.047.
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
    target_standing_z=0.90,  # SB3 height-maintenance target (trex_env.py)
    target_distance_range=(3.0, 8.0),
    target_lateral_range=(-2.0, 2.0),
    target_z=0.5,
    body_ids={"pelvis": 2},  # MuJoCo body ID for pelvis (world=0, prey=1)
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
    },
)
