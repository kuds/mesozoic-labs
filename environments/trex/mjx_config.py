"""T-Rex MJX species configuration.

Registers the T-Rex species with the MJX environment so that
``MJXDinoEnv("trex", ...)`` works out of the box.
"""

from __future__ import annotations

from environments.shared.mjx_env import register_species_mjx

# Sensor indices match the MJCF sensor definition order:
# pelvis_gyro(3), pelvis_accel(3), pelvis_orientation(4),
# r_foot_touch(1), l_foot_touch(1)
_SENSOR_R_FOOT = 10
_SENSOR_L_FOOT = 11

register_species_mjx(
    species="trex",
    frame_skip=5,
    max_episode_steps=1000,
    healthy_z_range=(0.5, 1.6),
    max_tilt_angle=1.047,
    sensor_foot_indices=(_SENSOR_R_FOOT, _SENSOR_L_FOOT),
    sensor_gyro_start=0,
    sensor_accel_start=3,
    sensor_quat_start=6,
    natural_forward_z=-0.169,  # -sin(0.17)  ~10° natural pitch
    forward_vel_max=8.0,
    fall_penalty=-100.0,
    target_distance_range=(3.0, 8.0),
    target_lateral_range=(-2.0, 2.0),
    target_z=0.5,
    body_ids={"pelvis": 1},  # MuJoCo body ID for pelvis
    reward_weights={
        "forward_vel_weight": 1.0,
        "alive_bonus": 0.1,
        "energy_penalty_weight": 0.001,
        "posture_weight": 0.2,
        "approach_weight": 1.0,
        "tail_stability_weight": 0.05,
        "smoothness_weight": 0.05,
        "bite_bonus": 10.0,
    },
)
