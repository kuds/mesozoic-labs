"""The Dibothrosuchus's stance quality, tracked through full standing episodes (``com_stance_helpers``).

The static-balance tests check the home keyframe and that the neutral stance survives.  These track the whole-body
centre of mass through the full standing episode and fail on a stance that stands but is a poor one: off its support
(to the nearest edge, sideways or fore-aft), leaning or drifting sideways, off its fore-aft posture, sagging, turning
or twisting on its feet (in the window or from the spawn), or swaying and rocking.  On four feet the lightest foot's
share is not fixed by symmetry, so it holds the statue's own (0.144), and the front/rear split is a fore-aft key.
The static-balance tests run 100 steps; these run the full 1,000.

Measured on physics r2 (2026-10-28, mujoco 3.10.0, the tests' own sampler; window 2-10 s): the scripted 1 deg
hip-roll lean reaches 4.7x a lean bar (the load split 55/45), the 0.05 m g w yaw torque 16.6x the pivot bar and the 8
% m g fore-aft rock at 0.75 Hz 5.1x a sway bar, all standing the full horizon.  What the noisy-spawn bars allow on
the current plant (80 spawns at reset noise 0.02): a turn of up to 1.4 deg from the spawn and a foot that slides up
to 11 mm from where it spawned.
"""

from environments.shared.model_id import ModelId
from environments.shared.tests.com_stance_helpers import ComStanceQualityBase


class TestComStanceQuality(ComStanceQualityBase):
    model_id = ModelId("dibothrosuchus_elaphros", "anatomical")
    species_label = "Dibothrosuchus"
    quadruped = True
    physics_revision = 2
    window_start_s = 2.0
    # The noise-free statue's window values, physics r2, mujoco 3.10.0, read on 2026-10-28 with
    # com_stance_calibration and rounded to 0.01 mm and 1e-4 deg and of the load (under 1 % of a width).
    reference = {
        "com_margin_min_m": 0.06407,
        "xcom_margin_min_m": 0.06407,
        "com_margin_lateral_min_m": 0.06409,
        "xcom_margin_lateral_min_m": 0.06409,
        "com_margin_fore_aft_min_m": 0.09193,
        "xcom_margin_fore_aft_min_m": 0.09193,
        "lightest_foot_share_min": 0.1442,
        "com_fore_aft_mean_m": -0.0782,
        "com_fore_aft_drift_m": -3e-05,
        "root_pitch_mean_deg": 0.1109,
        "front_load_share_mean": 0.2886,
        "com_height_mean_m": 0.26214,
        "com_height_drift_m": -1e-05,
        "foot_turn_max_deg": 0.0,
        "foot_turn_from_spawn_max_deg": 0.0,
        "foot_splay_change_max_deg": 0.0,
        "foot_travel_max_m": 0.00027,
        "foot_travel_from_spawn_max_m": 0.00099,
        "com_speed_rms_mps": 2e-05,
        "com_speed_end_rms_mps": 0.0,
        "com_fore_aft_sd_m": 0.0,
        "com_height_sd_m": 0.0,
        "root_pitch_sd_deg": 0.001,
    }
    # Twice the worst of calibration seeds 0-29 at reset noise 0.02, never under twice the floor (SI, deg).
    # Held-out seeds 100-149: none breaks a width; the largest use is 0.52 (foot_turn_max_deg, seed
    # 139).
    spread_width = {
        "com_margin_min_m": 0.0045,
        "xcom_margin_min_m": 0.0044,
        "com_margin_lateral_min_m": 0.0045,
        "xcom_margin_lateral_min_m": 0.0045,
        "com_margin_fore_aft_min_m": 0.001,
        "xcom_margin_fore_aft_min_m": 0.001,
        "com_lateral_mean_m": 0.001,
        "root_roll_mean_deg": 0.1,
        "left_load_share_mean": 0.01,
        "lightest_foot_share_min": 0.051,
        "com_lateral_drift_m": 0.001,
        "root_roll_drift_deg": 0.15,
        "com_fore_aft_mean_m": 0.001,
        "com_fore_aft_drift_m": 0.001,
        "root_pitch_mean_deg": 0.1,
        "front_load_share_mean": 0.01,
        "com_height_mean_m": 0.001,
        "com_height_drift_m": 0.001,
        "root_turn_deg": 0.12,
        "root_turn_end_deg": 0.1,
        "root_turn_from_spawn_deg": 2.8,
        "root_minus_feet_yaw_from_spawn_deg": 0.1,
        "foot_turn_max_deg": 0.12,
        "foot_turn_from_spawn_max_deg": 2.9,
        "foot_splay_change_max_deg": 0.1,
        "foot_travel_max_m": 0.0044,
        "foot_travel_from_spawn_max_m": 0.02,
        "com_speed_rms_mps": 0.001,
        "com_speed_end_rms_mps": 0.001,
        "com_lateral_sd_m": 0.001,
        "com_fore_aft_sd_m": 0.001,
        "com_height_sd_m": 0.001,
        "root_roll_sd_deg": 0.1,
        "root_pitch_sd_deg": 0.1,
        "root_yaw_sd_deg": 0.1,
    }
    # Held-out seeds: the largest turn from the spawn (1.20 deg) and the largest lateral offset
    # (-0.091 mm) among those holding every width; they use 0.43 and 0.52 of a width.
    noisy_seeds = (122, 139)
    # A 0.5 % m g push barely moves this low, wide stance (0.30x): the rock is 8 % of the weight.
    rock_force_bw = 0.08
