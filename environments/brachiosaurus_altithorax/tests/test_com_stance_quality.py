"""Track the centre of mass and all four support contacts through 20 s of standing.

On physics r6 and r7 (whose removal of the food mocap body left the dynamics bit-identical) the
noiseless stance carries 56.15% of weight on the forelimbs. Its smallest lateral and fore-aft
support margins over 8–20 s are 0.300 and 0.505 m. The calibration's 30 noisy resets set the
spread. All 50 held-out resets fit, with the largest use of a width at 0.78. The scripted lean,
yaw torque and fore-aft rock each exceeded a stance bar by at least 21 times. Absolute targets
below track four residual recovery limitations separately.
"""

import pytest

from environments.shared.model_id import ModelId
from environments.shared.tests.com_stance_helpers import ComStanceQualityBase, TargetMissed


class TestComStanceQuality(ComStanceQualityBase):
    model_id = ModelId("brachiosaurus_altithorax", "anatomical")
    species_label = "Brachiosaurus"
    quadruped = True
    physics_revision = 7
    # The 8 s settle leaves only 2 s in a standard episode. Measure the settled pose to 20 s.
    horizon_steps = 2000
    window_start_s = 8.0
    # The noise-free statue's window values, physics r6, mujoco 3.10.0, read on 2026-11-20 with
    # com_stance_calibration and rounded to 0.01 mm and 1e-4 deg and of the load (under 1 % of a width).
    # Re-read on physics r7 on 2026-11-23: every pin equal (the calibration's ratchet found nothing to change).
    reference = {
        "com_margin_min_m": 0.3,
        "xcom_margin_min_m": 0.3,
        "com_margin_lateral_min_m": 0.3,
        "xcom_margin_lateral_min_m": 0.3,
        "com_margin_fore_aft_min_m": 0.50485,
        "xcom_margin_fore_aft_min_m": 0.50485,
        "lightest_foot_share_min": 0.2192,
        "com_fore_aft_mean_m": 0.07094,
        "com_fore_aft_drift_m": 0.0,
        "root_pitch_mean_deg": -0.0259,
        "front_load_share_mean": 0.5615,
        "com_height_mean_m": 1.12112,
        "com_height_drift_m": 0.0,
        "foot_turn_max_deg": 0.0,
        "foot_turn_from_spawn_max_deg": 0.0,
        "foot_splay_change_max_deg": 0.0,
        "foot_travel_max_m": 2e-05,
        "foot_travel_from_spawn_max_m": 8e-05,
        "com_speed_rms_mps": 1e-05,
        "com_speed_end_rms_mps": 0.0,
        "com_fore_aft_sd_m": 0.0,
        "com_height_sd_m": 0.0,
        "root_pitch_sd_deg": 0.0,
    }
    # Twice the worst of calibration seeds 0-29 at reset noise 0.02, never under twice the floor (SI, deg).
    # Held-out seeds 100-149: none breaks a width; the largest use is 0.78 (com_fore_aft_mean_m, seed 107).
    spread_width = {
        "com_margin_min_m": 0.018,
        "xcom_margin_min_m": 0.018,
        "com_margin_lateral_min_m": 0.018,
        "xcom_margin_lateral_min_m": 0.018,
        "com_margin_fore_aft_min_m": 0.013,
        "xcom_margin_fore_aft_min_m": 0.013,
        "com_lateral_mean_m": 0.0022,
        "root_roll_mean_deg": 0.1,
        "left_load_share_mean": 0.01,
        "lightest_foot_share_min": 0.039,
        "com_lateral_drift_m": 0.0019,
        "root_roll_drift_deg": 0.1,
        "com_fore_aft_mean_m": 0.0017,
        "com_fore_aft_drift_m": 0.0012,
        "root_pitch_mean_deg": 0.1,
        "front_load_share_mean": 0.01,
        "com_height_mean_m": 0.001,
        "com_height_drift_m": 0.001,
        "root_turn_deg": 0.39,
        "root_turn_end_deg": 0.1,
        "root_turn_from_spawn_deg": 3.5,
        "root_minus_feet_yaw_from_spawn_deg": 0.27,
        "foot_turn_max_deg": 0.51,
        "foot_turn_from_spawn_max_deg": 4.2,
        "foot_splay_change_max_deg": 0.3,
        "foot_travel_max_m": 0.017,
        "foot_travel_from_spawn_max_m": 0.04,
        "com_speed_rms_mps": 0.001,
        "com_speed_end_rms_mps": 0.001,
        "com_lateral_sd_m": 0.001,
        "com_fore_aft_sd_m": 0.001,
        "com_height_sd_m": 0.001,
        "root_roll_sd_deg": 0.1,
        "root_pitch_sd_deg": 0.1,
        "root_yaw_sd_deg": 0.1,
    }
    # Held-out seeds: the largest turn from the spawn (-1.80 deg) and the largest lateral offset (-0.806 mm) among
    # those holding every width; they use 0.55 and 0.45 of a width.
    noisy_seeds = (140, 112)

    # Seed 137 has the largest post-landing foot excursion in the 80-seed panel.
    target_seeds = (137,)

    @pytest.mark.xfail(
        strict=True,
        raises=TargetMissed,
        reason="Physics r6 and r7: 7/80 noisy resets turn more than 1 degree, up to 1.80 degrees (seed 140).",
    )
    def test_target_turn_from_spawn(self, statue):
        self.check_target(statue, "turn_from_spawn")

    @pytest.mark.xfail(
        strict=True,
        raises=TargetMissed,
        reason="Physics r6 and r7: all 80 noisy resets exceed the 1 mm/s speed target after 3 s; worst 9.48 mm/s (140).",
    )
    def test_target_still_from_3s(self, statue):
        self.check_target(statue, "still_from_3s")

    @pytest.mark.xfail(
        strict=True,
        raises=TargetMissed,
        reason="Physics r6 and r7: all 80 noisy resets take over 2 s to settle (median 7 s, worst 9 s).",
    )
    def test_target_settled_by_2s(self, statue):
        self.check_target(statue, "settled_by_2s")

    def test_target_lean(self, statue):
        self.check_target(statue, "lean")

    def test_target_kept_roll(self, statue):
        self.check_target(statue, "kept_roll")

    @pytest.mark.xfail(
        strict=True,
        raises=TargetMissed,
        reason="Physics r6 and r7: 27/80 noisy resets exceed the 11.33 mm post-landing target; worst 20.63 mm (137).",
    )
    def test_target_foot_slip_after_landing(self, statue):
        self.check_target(statue, "foot_slip_after_landing")
