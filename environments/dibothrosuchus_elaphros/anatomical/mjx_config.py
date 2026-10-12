"""FROZEN (D-D17): dibothrosuchus's MJX plant registration. Nothing trains on it.

The plant contract imports this module by name (a policy-interface digest
input), checks it against ``DibothrosuchusEnv`` and executes ``build_mjx_observation``
on it in its MJX probe on every SB3 training and evaluation run. It holds only
the keys the probe reads. Do not edit, reformat or move it; see
``environments/shared/mjx_env.py``.
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
    species="dibothrosuchus",
    action_mapping="home-keyframe-residual/v1",
    frame_skip=5,
    sensor_foot_indices=(_SENSOR_FR_FOOT, _SENSOR_FL_FOOT, _SENSOR_RR_FOOT, _SENSOR_RL_FOOT),
    sensor_gyro_start=0,
    sensor_accel_start=3,
    sensor_quat_start=6,
    body_ids={"torso": 2},  # MuJoCo body ID for torso (world=0, prey=1)
)
