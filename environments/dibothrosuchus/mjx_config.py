"""Dibothrosuchus MJX species configuration.

Registers the Dibothrosuchus species with the MJX environment so that
``MJXDinoEnv("dibothrosuchus", ...)`` works out of the box.
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
# Tail tip gyro starts after: gyro(3)+accel(3)+quat(4)+touch(4)+snout_pos(3)+tail_pos(3)+tail_linvel(3)=23
_SENSOR_TAIL_GYRO_START = 23

register_species_mjx(
    species="dibothrosuchus",
    action_mapping="home-keyframe-residual/v1",
    frame_skip=5,
    max_episode_steps=1000,
    # Ceiling 0.55: the settled stance is 0.313m, and reset height jitter is
    # normal(0, reset_noise_scale) metres on every species alike.  A 0.45
    # ceiling is only 44% above stance on this small plant, so stage-1 reset
    # noise spawned a fifth of episodes already out of bounds; 0.55 is 75%
    # above stance, still unambiguously airborne.
    healthy_z_range=(0.18, 0.55),
    max_tilt_angle=0.9,
    sensor_foot_indices=(_SENSOR_FR_FOOT, _SENSOR_FL_FOOT, _SENSOR_RR_FOOT, _SENSOR_RL_FOOT),
    sensor_gyro_start=0,
    sensor_accel_start=3,
    sensor_quat_start=6,
    # Measured: the trunk frame settles level under the home controller
    # (forward_z +0.003), so both nosedive and posture keep world vertical.
    natural_forward_z=0.0,
    sensor_tail_gyro_start=_SENSOR_TAIL_GYRO_START,
    forward_vel_max=3.0,
    fall_penalty=-100.0,
    target_standing_z=0.3129,  # SB3 height-maintenance target (dibothrosuchus_env.py)
    target_distance_range=(1.5, 4.0),
    target_lateral_range=(-1.0, 1.0),
    target_z=0.15,  # prey underside at 0.09m, clear of the 0.04m snout-prop gate
    body_ids={"torso": 2},  # MuJoCo body ID for torso (world=0, prey=1)
    termination_body_heights={
        # The trunk itself is already covered by healthy_z_range, which is the
        # sprawl-collapse gate: a belly-down modern-crocodilian sprawl sits far
        # below the erect stance this species is trained to hold.
        "skull": 0.10,  # skull capsule radius=0.028 + margin
        "tail_3": 0.04,  # tail_3 capsule radius=0.018 + margin
        "tail_4": 0.03,  # tail_4 capsule radius=0.011 + margin
    },
    termination_site_heights={
        "snout_tip": 0.04,  # terminates snout-propping the way the T-Rex head_tip gate does
    },
    # Stage 3 success: snout_tip site proximity to prey.
    # Gated by reward_weights["snap_bonus"] — inactive when 0 (stages 1-2).
    success_sites=("snout_tip",),
    success_threshold=0.12,  # snout_snap box (0.022) + prey sphere (0.06) + margin
    success_bonus_key="snap_bonus",
    reward_weights={
        "forward_vel_weight": 1.0,
        "alive_bonus": 0.1,
        "energy_penalty_weight": 0.001,
        "posture_weight": 0.2,
        # Canonical key: registry weights bypass canonicalize_env_kwargs, so a
        # species-flavoured "snap_approach_weight" here would shadow the
        # TOML-configured approach_weight in every stage.
        "approach_weight": 1.0,
        "gait_stability_weight": 0.05,
        "tail_stability_weight": 0.05,
        "smoothness_weight": 0.05,
        "snap_bonus": 10.0,
    },
)
