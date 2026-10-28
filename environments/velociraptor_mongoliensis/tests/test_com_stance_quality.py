"""The Velociraptor's stance quality, tracked through full standing episodes (``com_stance_helpers``).

The static-balance tests check the home keyframe and that the neutral stance survives.  These track the whole-body
centre of mass through the full standing episode and fail on a stance that stands but is a poor one: off its support
(to the nearest edge, sideways or fore-aft), leaning or drifting sideways, off its fore-aft posture, sagging, turning
or twisting on its feet (in the window or from the spawn), or swaying and rocking.  The pelvis's natural 20 deg
forward lean is the statue's own reference.

Measured on physics r4 (2026-10-28, mujoco 3.10.0, the tests' own sampler; window 3-10 s): the scripted 1 deg
hip-roll lean reaches 9.8x a lean bar (the CoM 9.8 mm to one side of the feet), the 0.05 m g w yaw torque 22.4x the
pivot bar and the 0.5 % m g fore-aft rock at 0.75 Hz 7.0x a sway bar, all standing the full horizon.  What the
noisy-spawn bars allow on the current plant (80 spawns at reset noise 0.02): the body turns up to 5.7 deg from its
spawn, up to 0.8 deg of it inside the window, and a foot slides up to 7.3 mm from where it spawned.
"""

from environments.shared.model_id import ModelId
from environments.shared.tests.com_stance_helpers import ComStanceQualityBase


class TestComStanceQuality(ComStanceQualityBase):
    model_id = ModelId("velociraptor_mongoliensis", "anatomical")
    species_label = "Velociraptor"
    physics_revision = 4
    window_start_s = 3.0
    # The noise-free statue's window values, physics r4, mujoco 3.10.0, read on 2026-10-28 with
    # com_stance_calibration and rounded to 0.01 mm and 1e-4 deg and of the load (under 1 % of a width).
    reference = {
        "com_margin_min_m": 0.04921,
        "xcom_margin_min_m": 0.04923,
        "com_margin_lateral_min_m": 0.09563,
        "xcom_margin_lateral_min_m": 0.09566,
        "com_margin_fore_aft_min_m": 0.04921,
        "xcom_margin_fore_aft_min_m": 0.04923,
        "com_fore_aft_mean_m": -0.00072,
        "com_fore_aft_drift_m": 8e-05,
        "root_pitch_mean_deg": -20.0791,
        "com_height_mean_m": 0.46306,
        "com_height_drift_m": -1e-05,
        "foot_turn_max_deg": 0.017,
        "foot_turn_from_spawn_max_deg": 0.0273,
        "foot_splay_change_max_deg": 0.0273,
        "foot_travel_max_m": 0.00041,
        "foot_travel_from_spawn_max_m": 0.00066,
        "com_speed_rms_mps": 6e-05,
        "com_speed_end_rms_mps": 1e-05,
        "com_fore_aft_sd_m": 1e-05,
        "com_height_sd_m": 0.0,
        "root_pitch_sd_deg": 0.0018,
    }
    # Twice the worst of calibration seeds 0-29 at reset noise 0.02, never under twice the floor (SI, deg).
    # Held-out seeds 100-149: none breaks a width; the largest use is 0.51 (root_turn_from_spawn_deg, seed
    # 131).
    spread_width = {
        "com_margin_min_m": 0.0036,
        "xcom_margin_min_m": 0.0034,
        "com_margin_lateral_min_m": 0.016,
        "xcom_margin_lateral_min_m": 0.016,
        "com_margin_fore_aft_min_m": 0.0036,
        "xcom_margin_fore_aft_min_m": 0.0034,
        "com_lateral_mean_m": 0.001,
        "root_roll_mean_deg": 0.11,
        "left_load_share_mean": 0.01,
        "lightest_foot_share_min": 0.013,
        "com_lateral_drift_m": 0.0028,
        "root_roll_drift_deg": 0.36,
        "com_fore_aft_mean_m": 0.0027,
        "com_fore_aft_drift_m": 0.0015,
        "root_pitch_mean_deg": 0.38,
        "com_height_mean_m": 0.001,
        "com_height_drift_m": 0.001,
        "root_turn_deg": 1.6,
        "root_turn_end_deg": 0.1,
        "root_turn_from_spawn_deg": 11.0,
        "root_minus_feet_yaw_from_spawn_deg": 0.73,
        "foot_turn_max_deg": 1.4,
        "foot_turn_from_spawn_max_deg": 11.0,
        "foot_splay_change_max_deg": 0.37,
        "foot_travel_max_m": 0.0056,
        "foot_travel_from_spawn_max_m": 0.014,
        "com_speed_rms_mps": 0.001,
        "com_speed_end_rms_mps": 0.001,
        "com_lateral_sd_m": 0.001,
        "com_fore_aft_sd_m": 0.001,
        "com_height_sd_m": 0.001,
        "root_roll_sd_deg": 0.1,
        "root_pitch_sd_deg": 0.1,
        "root_yaw_sd_deg": 0.19,
    }
    # Held-out seeds: the largest turn from the spawn (-5.66 deg) and the largest lateral offset
    # (0.409 mm) among those holding every width; they use 0.51 and 0.50 of a width.
    noisy_seeds = (131, 110)
