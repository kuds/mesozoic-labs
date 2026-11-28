"""Standing balance on physics r4, measured using the MuJoCo 3.10.0 stance sampler.

The calibrated bars track stability and the fore/hind load split over 2–10 s. The neutral stance puts
13.15% of body weight on its lightest foot (13.36% on physics r3). Across 80 starts at reset noise 0.02 the
animal settled within 2 s. Four starts missed the 1 degree yaw target and 17 missed the post-landing foot
travel target. The scripted lean, yaw torque and fore-aft rock reached 3.2, 15.0 and 5.3 times a respective
stance bar.

Physics r4 removes the 20 g snout_snap box from the snout tip with the prey (decision D-D59): the lighter
head stands the statue's CoM 1.2 mm further back (fore-aft margin 84.13 -> 82.92 mm) and moves 0.4% of the
weight off the forefeet (front load share 0.2673 -> 0.2632); the lateral margin and the height barely move.
"""

import pytest

from environments.shared.model_id import ModelId
from environments.shared.tests.com_stance_helpers import ComStanceQualityBase, TargetMissed


class TestComStanceQuality(ComStanceQualityBase):
    model_id = ModelId("dibothrosuchus_elaphros", "anatomical")
    species_label = "Dibothrosuchus"
    quadruped = True
    physics_revision = 4
    window_start_s = 2.0
    # The noise-free statue's window values, physics r4, mujoco 3.10.0, read on 2026-11-23 with
    # com_stance_calibration and rounded to 0.01 mm and 1e-4 deg and of the load (under 1 % of a width).
    reference = {
        "com_margin_min_m": 0.0583,
        "xcom_margin_min_m": 0.0583,
        "com_margin_lateral_min_m": 0.05832,
        "xcom_margin_lateral_min_m": 0.05832,
        "com_margin_fore_aft_min_m": 0.08292,
        "xcom_margin_fore_aft_min_m": 0.08292,
        "lightest_foot_share_min": 0.1315,
        "com_fore_aft_mean_m": -0.09006,
        "com_fore_aft_drift_m": -3e-05,
        "root_pitch_mean_deg": -0.0086,
        "front_load_share_mean": 0.2632,
        "com_height_mean_m": 0.26175,
        "com_height_drift_m": 0.0,
        "foot_turn_max_deg": 0.0002,
        "foot_turn_from_spawn_max_deg": 0.0003,
        "foot_splay_change_max_deg": 0.0003,
        "foot_travel_max_m": 0.00018,
        "foot_travel_from_spawn_max_m": 0.00059,
        "com_speed_rms_mps": 1e-05,
        "com_speed_end_rms_mps": 0.0,
        "com_fore_aft_sd_m": 0.0,
        "com_height_sd_m": 0.0,
        "root_pitch_sd_deg": 0.0009,
    }
    # Twice the worst of calibration seeds 0-29 at reset noise 0.02, never under twice the floor (SI, deg).
    # Held-out seeds 100-149: none breaks a width; the largest use is 0.59 (foot_travel_max_m, seed 138).
    # Re-read on physics r4: the rule tightens lightest_foot_share_min (0.032 -> 0.029) and root_turn_deg
    # (0.12 -> 0.11), which this takes, and would widen com_margin_fore_aft_min_m 0.0024 -> 0.0025, which it
    # keeps at r3's 0.0024: twice the worst calibration deviation is at most 0.0025, so no spawn uses more
    # than about 0.52 of the kept width.
    spread_width = {
        "com_margin_min_m": 0.0039,
        "xcom_margin_min_m": 0.0039,
        "com_margin_lateral_min_m": 0.0039,
        "xcom_margin_lateral_min_m": 0.0039,
        "com_margin_fore_aft_min_m": 0.0024,
        "xcom_margin_fore_aft_min_m": 0.0024,
        "com_lateral_mean_m": 0.001,
        "root_roll_mean_deg": 0.12,
        "left_load_share_mean": 0.01,
        "lightest_foot_share_min": 0.029,
        "com_lateral_drift_m": 0.0011,
        "root_roll_drift_deg": 0.3,
        "com_fore_aft_mean_m": 0.001,
        "com_fore_aft_drift_m": 0.001,
        "root_pitch_mean_deg": 0.1,
        "front_load_share_mean": 0.01,
        "com_height_mean_m": 0.001,
        "com_height_drift_m": 0.001,
        "root_turn_deg": 0.11,
        "root_turn_end_deg": 0.1,
        "root_turn_from_spawn_deg": 2.4,
        "root_minus_feet_yaw_from_spawn_deg": 0.1,
        "foot_turn_max_deg": 0.12,
        "foot_turn_from_spawn_max_deg": 2.5,
        "foot_splay_change_max_deg": 0.1,
        "foot_travel_max_m": 0.0045,
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
    # Held-out seeds: the largest turn from the spawn (1.22 deg) and the largest lateral offset (-0.196 mm) among
    # those holding every width; they use 0.52 and 0.49 of a width.
    noisy_seeds = (122, 146)
    # A 0.5 % m g push barely moves this low, wide stance (0.30x): the rock is 8 % of the weight.
    rock_force_bw = 0.08

    @pytest.mark.xfail(
        strict=True,
        raises=TargetMissed,
        reason=(
            "Physics r4, MuJoCo 3.10.0: 4 of 80 starts at noise 0.02 turned more than 1 degree "
            "from spawn. The worst was 1.223 degrees (seed 122), with a median of 0.3815 degrees."
        ),
    )
    def test_target_turn_from_spawn(self, statue):
        """Absolute target: the body ends the episode turned under 1 deg from its spawn."""
        self.check_target(statue, "turn_from_spawn")

    @pytest.mark.xfail(
        strict=True,
        raises=TargetMissed,
        reason=(
            "Physics r4, MuJoCo 3.10.0: 17 of 80 starts at noise 0.02 exceeded 2.629 mm of foot "
            "travel after landing. The worst was 3.743 mm (seed 138), with a median of 2.176 mm."
        ),
    )
    def test_target_foot_slip_after_landing(self, statue):
        """Absolute target: from 1 s no foot slips farther than a 1 deg turn about the feet's centre slides it."""
        self.check_target(statue, "foot_slip_after_landing")
