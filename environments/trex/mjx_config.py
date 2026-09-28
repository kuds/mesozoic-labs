"""FROZEN (D-D17): trex's MJX plant registration. Nothing trains on it.

The plant contract imports this module by name (a policy-interface digest
input), checks it against ``TRexEnv`` and executes ``build_mjx_observation``
on it in its MJX probe on every SB3 training and evaluation run. It holds only
the keys the probe reads. Do not edit, reformat or move it; see
``environments/shared/mjx_env.py``.
"""

from __future__ import annotations

from environments.shared.mjx_env import register_species_mjx

# Sensor indices match the MJCF sensor definition order:
# pelvis_gyro(3), pelvis_accel(3), pelvis_orientation(4),
# r_foot_touch(1), l_foot_touch(1)
_SENSOR_R_FOOT = 10
_SENSOR_L_FOOT = 11
# The pads above miss the digits (child bodies on passive hinges), so each
# digit has its own touch sensor, appended after the tail block; summed per foot.
_SENSOR_R_FOOT_DIGITS = (24, 25, 26)
_SENSOR_L_FOOT_DIGITS = (27, 28, 29)

register_species_mjx(
    species="trex",
    action_mapping="home-keyframe-residual/v1",
    # 10 Hz command low-pass (plant interface r11); must equal
    # TRexEnv.action_filter_cutoff_hz (the plant contract asserts it).
    action_filter_cutoff_hz=10.0,
    frame_skip=5,
    sensor_foot_indices=(_SENSOR_R_FOOT, _SENSOR_L_FOOT),
    sensor_foot_aux_indices=(_SENSOR_R_FOOT_DIGITS, _SENSOR_L_FOOT_DIGITS),
    sensor_gyro_start=0,
    sensor_accel_start=3,
    sensor_quat_start=6,
    body_ids={"pelvis": 2},  # MuJoCo body ID for pelvis (world=0, prey=1)
)
