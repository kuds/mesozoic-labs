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

import pytest

from environments.shared.model_id import ModelId
from environments.shared.tests.com_stance_helpers import ComStanceQualityBase, TargetMissed


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

    # Spawns the absolute targets read besides noisy_seeds: of the 80 spawns, the worst that misses a target both test
    # seeds meet (measured 2026-10-29): 126 (foot_slip_after_landing).
    target_seeds = (126,)

    @pytest.mark.xfail(
        strict=True,
        raises=TargetMissed,
        reason=(
            "physics r5 (2026-10-29): the turn read at the episode's end (the yaw at its end less at the spawn): "
            "root_turn_from_spawn_deg seed 129 1.48 deg (6 of the 80 spawns at reset noise 0.02 miss; worst 1.48 deg, "
            "seed 129; median 0.34 deg). Target: |root_turn_from_spawn_deg| <= 1.00 deg. Strict: when a model change "
            "meets it this passes and so fails; run com_stance_calibration, check the 80 spawns, then drop the mark "
            "(the target becomes a real bar)"
        ),
    )
    def test_target_turn_from_spawn(self, statue):
        """Absolute target: the body ends the episode turned under 1 deg from its spawn."""
        self.check_target(statue, "turn_from_spawn")

    @pytest.mark.xfail(
        strict=True,
        raises=TargetMissed,
        reason=(
            "physics r5 (2026-10-29): the worst whole second's CoM speed RMS from 3 s and the fore-aft spread from 3 "
            "s: com_speed_worst_second_rms_mps the statue 25.77 mm/s, seed 129 22.42 mm/s, seed 134 27.63 mm/s, seed "
            "126 28.16 mm/s (80 of the 80 spawns at reset noise 0.02 miss; worst 30.45 mm/s, seed 26; median 25.02 "
            "mm/s); com_fore_aft_sd_m the statue 1.61 mm, seed 129 1.57 mm, seed 134 1.51 mm, seed 126 1.57 mm (80 of "
            "the 80 spawns at reset noise 0.02 miss; worst 1.68 mm, seed 142; median 1.57 mm). Target: "
            "|com_speed_worst_second_rms_mps| <= 1.00 mm/s, |com_fore_aft_sd_m| <= 1.00 mm. Strict: when a model "
            "change meets it this passes and so fails; run com_stance_calibration, check the 80 spawns, then drop the "
            "mark (the target becomes a real bar)"
        ),
    )
    def test_target_still_from_3s(self, statue):
        """Absolute target: no rocking after 3 s (every second under 1 mm/s RMS, under 1 mm fore-aft)."""
        self.check_target(statue, "still_from_3s")

    @pytest.mark.xfail(
        strict=True,
        raises=TargetMissed,
        reason=(
            "physics r5 (2026-10-29): settle_s the statue 10 s, seed 129 10 s, seed 134 13 s, seed 126 10 s (80 of the "
            "80 spawns at reset noise 0.02 miss; worst 13 s, seed 134; median 11 s). Target: |settle_s| <= 2 s. "
            "Strict: when a model change meets it this passes and so fails; run com_stance_calibration, check the 80 "
            "spawns, then drop the mark (the target becomes a real bar)"
        ),
    )
    def test_target_settled_by_2s(self, statue):
        """Absolute target: the stance has settled by 2 s."""
        self.check_target(statue, "settled_by_2s")

    @pytest.mark.xfail(
        strict=True,
        raises=TargetMissed,
        reason=(
            "physics r5 (2026-10-29): com_lateral_mean_m seed 129 5.56 mm, seed 134 22.79 mm (52 of the 80 spawns at "
            "reset noise 0.02 miss; worst 22.79 mm, seed 134; median 6.47 mm). Target: |com_lateral_mean_m| <= 5.00 "
            "mm. Strict: when a model change meets it this passes and so fails; run com_stance_calibration, check the "
            "80 spawns, then drop the mark (the target becomes a real bar)"
        ),
    )
    def test_target_lean(self, statue):
        """Absolute target: the settled CoM sits within 5 mm of the feet's centre sideways."""
        self.check_target(statue, "lean")

    @pytest.mark.xfail(
        strict=True,
        raises=TargetMissed,
        reason=(
            "physics r5 (2026-10-29): root_roll_mean_deg seed 134 0.80 deg (35 of the 80 spawns at reset noise 0.02 "
            "miss; worst 0.80 deg, seed 134; median 0.23 deg). Target: |root_roll_mean_deg| <= 0.30 deg. Strict: when "
            "a model change meets it this passes and so fails; run com_stance_calibration, check the 80 spawns, then "
            "drop the mark (the target becomes a real bar)"
        ),
    )
    def test_target_kept_roll(self, statue):
        """Absolute target: the settled pelvis keeps under 0.3 deg of roll."""
        self.check_target(statue, "kept_roll")

    @pytest.mark.xfail(
        strict=True,
        raises=TargetMissed,
        reason=(
            "physics r5 (2026-10-29): foot_travel_max_m seed 126 19.00 mm (14 of the 80 spawns at reset noise 0.02 "
            "miss; worst 19.00 mm, seed 126; median 9.04 mm). Target: |foot_travel_max_m| <= 13.58 mm: the slide of a "
            "1 deg turn about the feet's centre, the farthest foot standing 778.3 mm from it. Strict: when a model "
            "change meets it this passes and so fails; run com_stance_calibration, check the 80 spawns, then drop the "
            "mark (the target becomes a real bar)"
        ),
    )
    def test_target_foot_slip_after_landing(self, statue):
        """Absolute target: from 1 s no foot slips farther than a 1 deg turn about the feet's centre slides it."""
        self.check_target(statue, "foot_slip_after_landing")
