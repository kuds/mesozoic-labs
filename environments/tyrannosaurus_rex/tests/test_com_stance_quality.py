"""The T. rex's stance quality, tracked through full standing episodes (``com_stance_helpers``).

The static-balance tests check the home keyframe and that the neutral stance survives.  These track the whole-body
centre of mass through the full standing episode and fail on a stance that stands but is a poor one: off its support
(to the nearest edge, sideways or fore-aft), leaning or drifting sideways, off its fore-aft posture, sagging, turning
or twisting on its feet (in the window or from the spawn), or swaying and rocking.

The noise-free statue rocks fore-aft at 0.5 Hz after the reset, the motion falling by a factor e about every 2.6 s
(8.7 mm peak to peak at 2-3 s, 0.5 mm at 9-10 s), so the episode runs 2,000 steps (20 s) and the window starts at the
9 s settle.

Measured on physics r10 (2026-10-28, mujoco 3.10.0, the tests' own sampler; window 9-20 s): the scripted 1 deg
hip-roll lean reaches 5.3x a lean bar (the load split 55/45), the 0.05 m g w yaw torque 16.5x the pivot bar and the
0.5 % m g fore-aft rock at 0.75 Hz 4.9x a sway bar, all standing the full horizon.  What the noisy-spawn bars allow
on the current plant (80 spawns at reset noise 0.02): the body turns up to 4.9 deg from its spawn and a foot slides
up to 17 mm from where it spawned, nearly all of it before the window (the window turn stays under 0.17 deg).
"""

from environments.shared.model_id import ModelId
from environments.shared.tests.com_stance_helpers import ComStanceQualityBase


class TestComStanceQuality(ComStanceQualityBase):
    model_id = ModelId("tyrannosaurus_rex", "anatomical")
    # The static-balance tests' constructor.
    env_kwargs = {"reset_noise_scale": 0.0, "nosedive_termination_threshold": 0.35}
    species_label = "T. rex"
    physics_revision = 10
    # The settle (9 s) leaves under 5 s of the env's 1,000 steps: twice that, so the window is
    # the stance it settles into.
    horizon_steps = 2000
    window_start_s = 9.0
    # The noise-free statue's window values, physics r10, mujoco 3.10.0, read on 2026-10-28 with
    # com_stance_calibration and rounded to 0.01 mm and 1e-4 deg and of the load (under 1 % of a width).
    reference = {
        "com_margin_min_m": 0.06298,
        "xcom_margin_min_m": 0.06288,
        "com_margin_lateral_min_m": 0.18499,
        "xcom_margin_lateral_min_m": 0.18499,
        "com_margin_fore_aft_min_m": 0.06298,
        "xcom_margin_fore_aft_min_m": 0.06288,
        "com_fore_aft_mean_m": -0.02679,
        "com_fore_aft_drift_m": -6e-05,
        "root_pitch_mean_deg": -0.9606,
        "com_height_mean_m": 0.8669,
        "com_height_drift_m": 0.0,
        "foot_turn_max_deg": 0.0001,
        "foot_turn_from_spawn_max_deg": 0.0002,
        "foot_splay_change_max_deg": 0.0002,
        "foot_travel_max_m": 0.0,
        "foot_travel_from_spawn_max_m": 3e-05,
        "com_speed_rms_mps": 0.00027,
        "com_speed_end_rms_mps": 2e-05,
        "com_fore_aft_sd_m": 0.0001,
        "com_height_sd_m": 1e-05,
        "root_pitch_sd_deg": 0.011,
    }
    # Twice the worst of calibration seeds 0-29 at reset noise 0.02, never under twice the floor (SI, deg).
    # Held-out seeds 100-149: none breaks a width; the largest use is 0.59 (foot_turn_max_deg, seed
    # 121).
    spread_width = {
        "com_margin_min_m": 0.001,
        "xcom_margin_min_m": 0.001,
        "com_margin_lateral_min_m": 0.039,
        "xcom_margin_lateral_min_m": 0.039,
        "com_margin_fore_aft_min_m": 0.001,
        "xcom_margin_fore_aft_min_m": 0.001,
        "com_lateral_mean_m": 0.001,
        "root_roll_mean_deg": 0.1,
        "left_load_share_mean": 0.01,
        "lightest_foot_share_min": 0.01,
        "com_lateral_drift_m": 0.001,
        "root_roll_drift_deg": 0.1,
        "com_fore_aft_mean_m": 0.001,
        "com_fore_aft_drift_m": 0.001,
        "root_pitch_mean_deg": 0.1,
        "com_height_mean_m": 0.001,
        "com_height_drift_m": 0.001,
        "root_turn_deg": 0.28,
        "root_turn_end_deg": 0.1,
        "root_turn_from_spawn_deg": 9.8,
        "root_minus_feet_yaw_from_spawn_deg": 0.56,
        "foot_turn_max_deg": 0.34,
        "foot_turn_from_spawn_max_deg": 9.8,
        "foot_splay_change_max_deg": 0.45,
        "foot_travel_max_m": 0.0092,
        "foot_travel_from_spawn_max_m": 0.034,
        "com_speed_rms_mps": 0.001,
        "com_speed_end_rms_mps": 0.001,
        "com_lateral_sd_m": 0.001,
        "com_fore_aft_sd_m": 0.001,
        "com_height_sd_m": 0.001,
        "root_roll_sd_deg": 0.1,
        "root_pitch_sd_deg": 0.1,
        "root_yaw_sd_deg": 0.1,
    }
    # Held-out seeds: the largest turn from the spawn (4.21 deg) and the largest lateral offset
    # (-0.006 mm) among those holding every width; they use 0.59 and 0.47 of a width.
    noisy_seeds = (121, 106)
