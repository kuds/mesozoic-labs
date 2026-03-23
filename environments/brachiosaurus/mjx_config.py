"""Brachiosaurus MJX species configuration.

Registers the Brachiosaurus species with the MJX environment so that
``MJXDinoEnv("brachiosaurus", ...)`` works out of the box.
"""

from __future__ import annotations

from environments.shared.mjx_env import register_species_mjx

# Sensor indices match the MJCF sensor definition order:
# torso_gyro(3), torso_accel(3), torso_orientation(4),
# fr_foot_touch(1), fl_foot_touch(1), rr_foot_touch(1), rl_foot_touch(1)
_SENSOR_FR_FOOT = 10
_SENSOR_FL_FOOT = 11
_SENSOR_RR_FOOT = 12
_SENSOR_RL_FOOT = 13

register_species_mjx(
    species="brachiosaurus",
    frame_skip=5,
    max_episode_steps=1000,
    healthy_z_range=(1.0, 3.5),
    max_tilt_angle=1.047,
    sensor_foot_indices=(_SENSOR_FR_FOOT, _SENSOR_FL_FOOT, _SENSOR_RR_FOOT, _SENSOR_RL_FOOT),
    sensor_gyro_start=0,
    sensor_accel_start=3,
    sensor_quat_start=6,
    natural_forward_z=0.0,  # Upright quadruped
    forward_vel_max=1.0,
    fall_penalty=-100.0,
    target_distance_range=(3.0, 8.0),
    target_lateral_range=(-2.0, 2.0),
    target_z=3.0,  # Brachiosaurus browses high
    body_ids={"torso": 1},  # MuJoCo body ID for torso
    reward_weights={
        "forward_vel_weight": 1.0,
        "alive_bonus": 0.1,
        "energy_penalty_weight": 0.001,
        "posture_weight": 0.0,
        "approach_weight": 1.0,
        "gait_stability_weight": 0.05,
        "food_reach_bonus": 10.0,
    },
)
