"""The Brachiosaurus's stance quality, tracked through full standing episodes (``com_stance_helpers``).

The static-balance tests check the home keyframe and that the neutral stance survives.  These track the whole-body
centre of mass through the full standing episode and fail on a stance that stands but is a poor one: off its support
(to the nearest edge, sideways or fore-aft), leaning or drifting sideways, off its fore-aft posture, sagging, turning
or twisting on its feet (in the window or from the spawn), or swaying and rocking.  On four feet the lightest foot's
share is not fixed by symmetry, so it holds the statue's own (0.187), and the front/rear split is a fore-aft key.

The noise-free statue bounces after the reset (20 mm of CoM height at 2-3 s, a horizontal CoM speed of 14-26 mm/s RMS
at 2-6 s, under 1 mm/s from 10 s), so the episode runs 2,000 steps (20 s) and the window starts at the 12 s settle.
Its nearest support edge is a side (0.300 m, the half-track): the fore-aft margin (0.498 m) is the one a fore-aft
creep moves.

Measured on physics r5 (2026-10-28, mujoco 3.10.0, the tests' own sampler; window 12-20 s): the scripted 1 deg
hip-roll lean reaches 37.6x a lean bar (the CoM 37.6 mm to one side of the feet), the 0.05 m g w yaw torque 13.6x the
pivot bar and the 0.5 % m g fore-aft rock at 0.75 Hz 13.0x a sway bar, all standing the full horizon.  What the
noisy-spawn bars allow on the current plant (80 spawns at reset noise 0.02): a lateral offset of up to 23 mm with 0.8
deg of roll, kept to the end, a turn of up to 1.5 deg from the spawn, and a foot that slides up to 48 mm from where
it spawned.
"""

from environments.shared.model_id import ModelId
from environments.shared.tests.com_stance_helpers import ComStanceQualityBase


class TestComStanceQuality(ComStanceQualityBase):
    model_id = ModelId("brachiosaurus_altithorax", "anatomical")
    species_label = "Brachiosaurus"
    quadruped = True
    physics_revision = 5
    # The settle (12 s) leaves under 5 s of the env's 1,000 steps: twice that, so the window is
    # the stance it settles into.
    horizon_steps = 2000
    window_start_s = 12.0
    # The noise-free statue's window values, physics r5, mujoco 3.10.0, read on 2026-10-28 with
    # com_stance_calibration and rounded to 0.01 mm and 1e-4 deg and of the load (under 1 % of a width).
    reference = {
        "com_margin_min_m": 0.3,
        "xcom_margin_min_m": 0.3,
        "com_margin_lateral_min_m": 0.3,
        "xcom_margin_lateral_min_m": 0.3,
        "com_margin_fore_aft_min_m": 0.49838,
        "xcom_margin_fore_aft_min_m": 0.49834,
        "lightest_foot_share_min": 0.1866,
        "com_fore_aft_mean_m": -0.20715,
        "com_fore_aft_drift_m": -0.00137,
        "root_pitch_mean_deg": -1.1495,
        "front_load_share_mean": 0.3736,
        "com_height_mean_m": 0.94257,
        "com_height_drift_m": -8e-05,
        "foot_turn_max_deg": 0.0,
        "foot_turn_from_spawn_max_deg": 0.0,
        "foot_splay_change_max_deg": 0.0,
        "foot_travel_max_m": 0.00202,
        "foot_travel_from_spawn_max_m": 0.01363,
        "com_speed_rms_mps": 0.00022,
        "com_speed_end_rms_mps": 0.00013,
        "com_fore_aft_sd_m": 3e-05,
        "com_height_sd_m": 2e-05,
        "root_pitch_sd_deg": 0.001,
    }
    # Twice the worst of calibration seeds 0-29 at reset noise 0.02, never under twice the floor (SI, deg).
    # Held-out seeds 100-149: none breaks a width; the largest use is 0.73 (com_margin_min_m, seed
    # 148).
    spread_width = {
        "com_margin_min_m": 0.038,
        "xcom_margin_min_m": 0.038,
        "com_margin_lateral_min_m": 0.038,
        "xcom_margin_lateral_min_m": 0.038,
        "com_margin_fore_aft_min_m": 0.017,
        "xcom_margin_fore_aft_min_m": 0.017,
        "com_lateral_mean_m": 0.045,
        "root_roll_mean_deg": 1.6,
        "left_load_share_mean": 0.063,
        "lightest_foot_share_min": 0.017,
        "com_lateral_drift_m": 0.0021,
        "root_roll_drift_deg": 0.1,
        "com_fore_aft_mean_m": 0.011,
        "com_fore_aft_drift_m": 0.0046,
        "root_pitch_mean_deg": 0.45,
        "front_load_share_mean": 0.01,
        "com_height_mean_m": 0.0012,
        "com_height_drift_m": 0.001,
        "root_turn_deg": 0.1,
        "root_turn_end_deg": 0.1,
        "root_turn_from_spawn_deg": 2.9,
        "root_minus_feet_yaw_from_spawn_deg": 0.57,
        "foot_turn_max_deg": 0.22,
        "foot_turn_from_spawn_max_deg": 3.6,
        "foot_splay_change_max_deg": 0.98,
        "foot_travel_max_m": 0.0065,
        "foot_travel_from_spawn_max_m": 0.051,
        "com_speed_rms_mps": 0.001,
        "com_speed_end_rms_mps": 0.001,
        "com_lateral_sd_m": 0.001,
        "com_fore_aft_sd_m": 0.001,
        "com_height_sd_m": 0.001,
        "root_roll_sd_deg": 0.1,
        "root_pitch_sd_deg": 0.1,
        "root_yaw_sd_deg": 0.1,
    }
    # Held-out seeds: the largest turn from the spawn (-1.48 deg) and the largest lateral offset
    # (-22.792 mm) among those holding every width; they use 0.51 and 0.62 of a width.
    noisy_seeds = (129, 134)
