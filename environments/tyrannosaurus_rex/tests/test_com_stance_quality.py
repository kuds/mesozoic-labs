"""The T. rex's stance quality, tracked through full standing episodes (``com_stance_helpers``).

The static-balance tests check the home keyframe and that the neutral stance survives.  These track the whole-body
centre of mass through the full standing episode and fail on a stance that stands but is a poor one: off its support
(to the nearest edge, sideways or fore-aft), leaning or drifting sideways, off its fore-aft posture, sagging, turning
or twisting on its feet (in the window or from the spawn), or swaying and rocking.

The noise-free statue rocks fore-aft at 0.5 Hz after the reset, the motion falling by a factor e about every 2.6 s
(8.7 mm peak to peak at 2-3 s, 0.5 mm at 9-10 s), so the episode runs 2,000 steps (20 s) and the window starts at the
9 s settle.

On physics r11 (2026-11-21, MuJoCo 3.10.0), every one of the 80 noisy spawns stands for the full 20 s.
The neutral statue's settled minimum CoM margin is 63.1 mm. The worst spawn turns 4.77 deg from its
initial heading. Four strict expected failures track remaining targets for turn, rocking, settling
and foot slip after landing.
"""

import pytest

from environments.shared.model_id import ModelId
from environments.shared.tests.com_stance_helpers import ComStanceQualityBase, TargetMissed


class TestComStanceQuality(ComStanceQualityBase):
    model_id = ModelId("tyrannosaurus_rex", "anatomical")
    # The static-balance tests' constructor.
    env_kwargs = {"reset_noise_scale": 0.0, "nosedive_termination_threshold": 0.35}
    species_label = "T. rex"
    physics_revision = 11
    # The settle (9 s) leaves under 5 s of the env's 1,000 steps: twice that, so the window is
    # the stance it settles into.
    horizon_steps = 2000
    window_start_s = 9.0
    # The noise-free statue's window values, physics r11, MuJoCo 3.10.0, read on 2026-11-21 with
    # com_stance_calibration and rounded to 0.01 mm and 1e-4 deg and of the load (under 1 % of a width).
    # Keep the stricter r10 ceilings for CoM speed RMS and pitch SD (r11: 0.00028 m/s and 0.0114 deg).
    reference = {
        "com_margin_min_m": 0.06307,
        "xcom_margin_min_m": 0.06298,
        "com_margin_lateral_min_m": 0.19058,
        "xcom_margin_lateral_min_m": 0.19057,
        "com_margin_fore_aft_min_m": 0.06307,
        "xcom_margin_fore_aft_min_m": 0.06298,
        "com_fore_aft_mean_m": -0.02668,
        "com_fore_aft_drift_m": -6e-05,
        "root_pitch_mean_deg": -0.9615,
        "com_height_mean_m": 0.86512,
        "com_height_drift_m": 1e-05,
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
    # Keep r10's widths on r11: all 30 calibration and 50 held-out spawns continue to meet them.
    # The new digital spread increases the reference lateral support margin by 5.6 mm, and the
    # redistributed hindlimb mass lowers the CoM by 1.8 mm. The tolerances on both remain fixed.
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
    # The held-out seeds with the greatest turn from spawn and lateral offset, respectively.
    noisy_seeds = (121, 106)

    @pytest.mark.xfail(
        strict=True,
        raises=TargetMissed,
        reason="r11, 2026-11-21: 49 of 80 noisy spawns exceed the 1 deg turn target. Worst 4.77 deg, median 1.57 deg.",
    )
    def test_target_turn_from_spawn(self, statue):
        """Absolute target: the body ends the episode turned under 1 deg from its spawn."""
        self.check_target(statue, "turn_from_spawn")

    @pytest.mark.xfail(
        strict=True,
        raises=TargetMissed,
        reason="r11, 2026-11-21: all 80 noisy spawns exceed the 1 mm/s worst-second RMS target from 3 s. The statue reaches 6.43 mm/s and the worst noisy spawn 9.48 mm/s.",
    )
    def test_target_still_from_3s(self, statue):
        """Absolute target: no rocking after 3 s (every second under 1 mm/s RMS, under 1 mm fore-aft)."""
        self.check_target(statue, "still_from_3s")

    @pytest.mark.xfail(
        strict=True,
        raises=TargetMissed,
        reason="r11, 2026-11-21: all 80 noisy spawns take longer than 2 s to settle. Median 8 s, worst 9 s.",
    )
    def test_target_settled_by_2s(self, statue):
        """Absolute target: the stance has settled by 2 s."""
        self.check_target(statue, "settled_by_2s")

    @pytest.mark.xfail(
        strict=True,
        raises=TargetMissed,
        reason="r11, 2026-11-21: 61 of 80 noisy spawns exceed the 2.44 mm foot-slip target after landing. Worst 10.85 mm, median 3.79 mm.",
    )
    def test_target_foot_slip_after_landing(self, statue):
        """Absolute target: from 1 s no foot slips farther than a 1 deg turn about the feet's centre slides it."""
        self.check_target(statue, "foot_slip_after_landing")
