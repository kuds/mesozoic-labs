"""The Compsognathus's stance quality, anatomical and robot, tracked through full standing episodes.

``test_balance.py`` checks the home keyframe's support and that small reset errors recover.  These run each model's
env for the full 1,000-step horizon (20 s at its 0.02 s step) and track the whole-body centre of mass
(``com_stance_helpers``), failing on a stance that stands but is a poor one: off its support (to the nearest edge,
sideways or fore-aft), leaning or drifting sideways, off its fore-aft posture, sagging, turning or twisting on its
feet (in the window or from the spawn), or swaying and rocking.

On the robot, the same reduction (``stance_track_helpers``) can later read the hardware, key by key as its sensors
allow: the CoM against the feet (lean, fore-aft, height, their drifts and spreads) from joint encoders and forward
kinematics; roll and pitch and their spreads from the pelvis IMU (``sensors.onboard_readings``); the twist over the
feet from forward kinematics; the load shares and the loaded support polygon only with foot force pads, which have
not been selected, though the load split is the strongest lean signal here (the 1 deg lean moves the CoM 0.57 mm,
under its 1 mm bar, but splits the load 64/36); the world yaw, the turn from the spawn and the foot travel only with
a world pose estimate.  ``imu_lean_keys`` holds the scripted lean to its bar on the IMU's roll alone.

Measured on 2026-10-28 (mujoco 3.10.0, the tests' own sampler): the scripted 1 deg hip-roll lean reaches 15.6x a lean
bar on the anatomical model (window 3-20 s) and 13.8x on the robot (window 2-20 s; the lighter foot's share 0.344 and
0.362; the robot's roll alone 4.8x), the 0.05 m g w yaw torque 12.5x and 8.6x the pivot bar, and the 4 % m g fore-aft
rock at 0.75 Hz 6.8x and 4.5x a sway bar, all standing the full horizon.  What the noisy-spawn bars allow on the
current plants (80 spawns each at reset noise 0.02): a turn from the spawn of up to 5.1 deg (anatomical) and 3.2 deg
(robot), almost all before the window.  Ten of the robot's 280 measured spawns (held-out seeds 111 and 143, and 8 of
seeds 200-399) end in a poor stance the bars fail: the robot buzzes on its feet and creeps backwards for the whole
episode, 58 mm over 18 s on nine of them and 79 mm on seed 111
(``test_known_poor_spawn_seed111_stays_within_the_calibrated_spread``, a strict expected failure).
"""

import pytest

from environments.shared.model_id import ModelId
from environments.shared.tests.com_stance_helpers import ComStanceQualityBase


class TestAnatomicalComStanceQuality(ComStanceQualityBase):
    model_id = ModelId("compsognathus_longipes", "anatomical")
    species_label = "Compsognathus"
    physics_revision = 2
    window_start_s = 3.0
    # The noise-free statue's window values, physics r2, mujoco 3.10.0, read on 2026-10-28 with
    # com_stance_calibration and rounded to 0.01 mm and 1e-4 deg and of the load (under 1 % of a width).
    reference = {
        "com_margin_min_m": 0.02932,
        "xcom_margin_min_m": 0.02932,
        "com_margin_lateral_min_m": 0.06352,
        "xcom_margin_lateral_min_m": 0.06352,
        "com_margin_fore_aft_min_m": 0.02932,
        "xcom_margin_fore_aft_min_m": 0.02932,
        "com_fore_aft_mean_m": -0.00868,
        "com_fore_aft_drift_m": 0.0,
        "root_pitch_mean_deg": 0.0073,
        "com_height_mean_m": 0.22831,
        "com_height_drift_m": 0.0,
        "foot_turn_max_deg": 0.0,
        "foot_turn_from_spawn_max_deg": 0.0,
        "foot_splay_change_max_deg": 0.0,
        "foot_travel_max_m": 0.0,
        "foot_travel_from_spawn_max_m": 0.0,
        "com_speed_rms_mps": 0.0,
        "com_speed_end_rms_mps": 0.0,
        "com_fore_aft_sd_m": 0.0,
        "com_height_sd_m": 0.0,
        "root_pitch_sd_deg": 0.0001,
    }
    # Twice the worst of calibration seeds 0-29 at reset noise 0.02, never under twice the floor (SI, deg).
    # Held-out seeds 100-149: none breaks a width; the largest use is 0.57 (foot_turn_max_deg, seed
    # 111).
    spread_width = {
        "com_margin_min_m": 0.0053,
        "xcom_margin_min_m": 0.0054,
        "com_margin_lateral_min_m": 0.045,
        "xcom_margin_lateral_min_m": 0.045,
        "com_margin_fore_aft_min_m": 0.0053,
        "xcom_margin_fore_aft_min_m": 0.0054,
        "com_lateral_mean_m": 0.001,
        "root_roll_mean_deg": 0.1,
        "left_load_share_mean": 0.01,
        "lightest_foot_share_min": 0.016,
        "com_lateral_drift_m": 0.001,
        "root_roll_drift_deg": 0.1,
        "com_fore_aft_mean_m": 0.001,
        "com_fore_aft_drift_m": 0.001,
        "root_pitch_mean_deg": 0.1,
        "com_height_mean_m": 0.001,
        "com_height_drift_m": 0.001,
        "root_turn_deg": 0.39,
        "root_turn_end_deg": 0.1,
        "root_turn_from_spawn_deg": 11.0,
        "root_minus_feet_yaw_from_spawn_deg": 0.1,
        "foot_turn_max_deg": 0.39,
        "foot_turn_from_spawn_max_deg": 11.0,
        "foot_splay_change_max_deg": 0.1,
        "foot_travel_max_m": 0.0018,
        "foot_travel_from_spawn_max_m": 0.012,
        "com_speed_rms_mps": 0.001,
        "com_speed_end_rms_mps": 0.001,
        "com_lateral_sd_m": 0.001,
        "com_fore_aft_sd_m": 0.001,
        "com_height_sd_m": 0.001,
        "root_roll_sd_deg": 0.1,
        "root_pitch_sd_deg": 0.1,
        "root_yaw_sd_deg": 0.1,
    }
    # Held-out seeds: the largest turn from the spawn (-4.93 deg) and the largest lateral offset
    # (-0.003 mm) among those holding every width; they use 0.45 and 0.57 of a width.
    noisy_seeds = (145, 111)
    # A 0.5 % m g push barely moves this small, stiff stance (0.85x): the rock is 4 % of the weight.
    rock_force_bw = 0.04


class TestRobotComStanceQuality(ComStanceQualityBase):
    model_id = ModelId("compsognathus_longipes", "robot")
    species_label = "Compsognathus robot"
    physics_revision = 1
    window_start_s = 2.0
    # The noise-free statue's window values, physics r1, mujoco 3.10.0, read on 2026-10-28 with
    # com_stance_calibration and rounded to 0.01 mm and 1e-4 deg and of the load (under 1 % of a width).
    reference = {
        "com_margin_min_m": 0.03492,
        "xcom_margin_min_m": 0.03492,
        "com_margin_lateral_min_m": 0.11,
        "xcom_margin_lateral_min_m": 0.11,
        "com_margin_fore_aft_min_m": 0.03492,
        "xcom_margin_fore_aft_min_m": 0.03492,
        "com_fore_aft_mean_m": 0.01508,
        "com_fore_aft_drift_m": 0.0,
        "root_pitch_mean_deg": -0.0138,
        "com_height_mean_m": 0.19639,
        "com_height_drift_m": 0.0,
        "foot_turn_max_deg": 0.0,
        "foot_turn_from_spawn_max_deg": 0.0002,
        "foot_splay_change_max_deg": 0.0002,
        "foot_travel_max_m": 0.0,
        "foot_travel_from_spawn_max_m": 0.0,
        "com_speed_rms_mps": 0.0,
        "com_speed_end_rms_mps": 0.0,
        "com_fore_aft_sd_m": 0.0,
        "com_height_sd_m": 0.0,
        "root_pitch_sd_deg": 0.0,
    }
    # Twice the worst of calibration seeds 0-29 at reset noise 0.02, never under twice the floor (SI, deg).
    # Held-out seeds 100-149: 2 break a width (111, 143: see known_poor_spawns); the largest
    # use among the rest is 0.64 (foot_travel_max_m, seed 141).
    spread_width = {
        "com_margin_min_m": 0.001,
        "xcom_margin_min_m": 0.001,
        "com_margin_lateral_min_m": 0.0017,
        "xcom_margin_lateral_min_m": 0.0017,
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
        "root_turn_deg": 0.1,
        "root_turn_end_deg": 0.1,
        "root_turn_from_spawn_deg": 6.5,
        "root_minus_feet_yaw_from_spawn_deg": 1.9,
        "foot_turn_max_deg": 0.23,
        "foot_turn_from_spawn_max_deg": 8.5,
        "foot_splay_change_max_deg": 2.1,
        "foot_travel_max_m": 0.0017,
        "foot_travel_from_spawn_max_m": 0.013,
        "com_speed_rms_mps": 0.001,
        "com_speed_end_rms_mps": 0.001,
        "com_lateral_sd_m": 0.001,
        "com_fore_aft_sd_m": 0.001,
        "com_height_sd_m": 0.001,
        "root_roll_sd_deg": 0.1,
        "root_pitch_sd_deg": 0.1,
        "root_yaw_sd_deg": 0.1,
    }
    # Held-out seeds: the largest turn from the spawn (1.92 deg) and the largest lateral offset
    # (-0.001 mm) among those holding every width; they use 0.64 and 0.49 of a width.
    noisy_seeds = (141, 102)
    # A 0.5 % m g push barely moves this small, stiff stance (0.56x): the rock is 4 % of the weight.
    rock_force_bw = 0.04
    # Without foot force pads the robot reads its lean from the pelvis IMU's roll alone.
    imu_lean_keys = ("root_roll_mean_deg",)

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "a known poor stance on physics r1 (2026-10-28): after this spawn at reset noise 0.02 the robot buzzes "
            "on its feet (floor load 0-22 N, under 5 % of its weight on 33 window samples) and creeps 79 mm back "
            "over 2-20 s, 4.4 mm/s, turning 2.4 deg; 9 more of 280 spawns creep about 58 mm at 3.2 mm/s (seeds "
            "143, 218, 249, 251, 271, 304, 330, 388, 390). Strict: when a plant change stops it, this passes and "
            "fails, so move the seed to noisy_seeds"
        ),
    )
    def test_known_poor_spawn_seed111_stays_within_the_calibrated_spread(self):
        """Held-out seed 111 breaks the spread bars today: kept as a reproducer of the robot's creep."""
        self.check_noisy_spawn(111)
