"""The Velociraptor's stance quality, tracked through full standing episodes (``com_stance_helpers``).

The static-balance tests check the home keyframe and that the neutral stance survives.  These track the whole-body
centre of mass through the full standing episode and fail on a stance that stands but is a poor one: off its support
(to the nearest edge, sideways or fore-aft), leaning or drifting sideways, off its fore-aft posture, sagging, turning
or twisting on its feet (in the window or from the spawn), or swaying and rocking.  The pelvis's natural pitch
(0.13 deg nose-down on the level-trunk plant of physics r5; 20 deg on r4) is the statue's own reference.

Measured on physics r5 (2026-11-09, mujoco 3.10.0, the tests' own sampler; window 2-10 s; com_stance_calibration
--write --allow-loosen): the scripted 1 deg hip-roll lean reaches 4.3x a lean bar (r4 9.8x), the 0.05 m g w yaw
torque 30.6x the pivot bar (r4 22.4x) and the 0.5 % m g fore-aft rock at 0.75 Hz 7.5x a sway bar (r4 7.0x), all
standing the full horizon.  What the noisy-spawn bars allow on the current plant (80 spawns at reset noise 0.02):
the body ends turned up to 4.2 deg from its spawn (seed 143; r4 5.7), and the spread widths, twice the calibration
spawns' worst, admit 2.7 deg of turn inside the window and 26 mm of foot travel from the spawn.
"""

import pytest

from environments.shared.model_id import ModelId
from environments.shared.tests.com_stance_helpers import ComStanceQualityBase, TargetMissed


class TestComStanceQuality(ComStanceQualityBase):
    model_id = ModelId("velociraptor_mongoliensis", "anatomical")
    species_label = "Velociraptor"
    physics_revision = 5
    window_start_s = 2.0
    # The noise-free statue's window values, physics r5, mujoco 3.10.0, read on 2026-11-09 with
    # com_stance_calibration and rounded to 0.01 mm and 1e-4 deg and of the load (under 1 % of a width).
    reference = {
        "com_margin_min_m": 0.06013,
        "xcom_margin_min_m": 0.06012,
        "com_margin_lateral_min_m": 0.10944,
        "xcom_margin_lateral_min_m": 0.10944,
        "com_margin_fore_aft_min_m": 0.06013,
        "xcom_margin_fore_aft_min_m": 0.06012,
        "com_fore_aft_mean_m": 0.06017,
        "com_fore_aft_drift_m": -1e-05,
        "root_pitch_mean_deg": -0.1341,
        "com_height_mean_m": 0.51692,
        "com_height_drift_m": 0.0,
        "foot_turn_max_deg": 0.0,
        "foot_turn_from_spawn_max_deg": 0.0,
        "foot_splay_change_max_deg": 0.0,
        "foot_travel_max_m": 0.0,
        "foot_travel_from_spawn_max_m": 1e-05,
        "com_speed_rms_mps": 3e-05,
        "com_speed_end_rms_mps": 0.0,
        "com_fore_aft_sd_m": 1e-05,
        "com_height_sd_m": 0.0,
        "root_pitch_sd_deg": 0.0011,
    }
    # Twice the worst of calibration seeds 0-29 at reset noise 0.02, never under twice the floor (SI, deg).
    # Held-out seeds 100-149: none breaks a width; the largest use is 0.51 (root_turn_from_spawn_deg, seed 143).
    spread_width = {
        "com_margin_min_m": 0.003,
        "xcom_margin_min_m": 0.0028,
        "com_margin_lateral_min_m": 0.019,
        "xcom_margin_lateral_min_m": 0.019,
        "com_margin_fore_aft_min_m": 0.003,
        "xcom_margin_fore_aft_min_m": 0.0028,
        "com_lateral_mean_m": 0.001,
        "root_roll_mean_deg": 0.1,
        "left_load_share_mean": 0.01,
        "lightest_foot_share_min": 0.023,
        "com_lateral_drift_m": 0.0027,
        "root_roll_drift_deg": 0.26,
        "com_fore_aft_mean_m": 0.0021,
        "com_fore_aft_drift_m": 0.0014,
        "root_pitch_mean_deg": 0.26,
        "com_height_mean_m": 0.0011,
        "com_height_drift_m": 0.001,
        "root_turn_deg": 2.7,
        "root_turn_end_deg": 0.1,
        "root_turn_from_spawn_deg": 8.4,
        "root_minus_feet_yaw_from_spawn_deg": 0.1,
        "foot_turn_max_deg": 2.7,
        "foot_turn_from_spawn_max_deg": 8.5,
        "foot_splay_change_max_deg": 0.1,
        "foot_travel_max_m": 0.0083,
        "foot_travel_from_spawn_max_m": 0.026,
        "com_speed_rms_mps": 0.001,
        "com_speed_end_rms_mps": 0.001,
        "com_lateral_sd_m": 0.001,
        "com_fore_aft_sd_m": 0.001,
        "com_height_sd_m": 0.001,
        "root_roll_sd_deg": 0.1,
        "root_pitch_sd_deg": 0.1,
        "root_yaw_sd_deg": 0.33,
    }
    # Held-out seeds: the largest turn from the spawn (4.25 deg) and the largest lateral offset (0.309 mm) among those
    # holding every width; they use 0.51 and 0.46 of a width.
    noisy_seeds = (143, 140)

    @pytest.mark.xfail(
        strict=True,
        raises=TargetMissed,
        reason=(
            "physics r5 (2026-11-09): the turn read at the episode's end (the yaw at its end less at the spawn): "
            "root_turn_from_spawn_deg seed 143 4.25 deg, seed 140 -3.89 deg (55 of the 80 spawns at reset noise 0.02 "
            "miss; worst 4.25 deg, seed 143; median 1.45 deg; r4 52 of 80, worst 5.66 deg). Target: "
            "|root_turn_from_spawn_deg| <= 1.00 deg. Strict: "
            "when a model change meets it this passes and so fails; run com_stance_calibration, check the 80 spawns, "
            "then drop the mark (the target becomes a real bar)"
        ),
    )
    def test_target_turn_from_spawn(self, statue):
        """Absolute target: the body ends the episode turned under 1 deg from its spawn."""
        self.check_target(statue, "turn_from_spawn")

    @pytest.mark.xfail(
        strict=True,
        raises=TargetMissed,
        reason=(
            "physics r5 (2026-11-09): settle_s seed 140 3 s, seed 143 2 s (1 of the 80 spawns at reset noise 0.02 "
            "misses: seed 140, a noisy seed; the statue and the median 2 s; r4 60 of 80 at 3 s). Target: "
            "|settle_s| <= 2 s. Strict: when a model change meets it "
            "this passes and so fails; run com_stance_calibration, check the 80 spawns, then drop the mark (the target "
            "becomes a real bar)"
        ),
    )
    def test_target_settled_by_2s(self, statue):
        """Absolute target: the stance has settled by 2 s."""
        self.check_target(statue, "settled_by_2s")

    @pytest.mark.xfail(
        strict=True,
        raises=TargetMissed,
        reason=(
            "physics r5 (2026-11-09): foot_travel_max_m seed 143 2.50 mm, seed 140 3.77 mm (59 of the 80 spawns at "
            "reset noise 0.02 miss; worst 5.51 mm, seed 3; median 2.27 mm; r4 44 of 80, worst 4.46 mm). Target: "
            "|foot_travel_max_m| <= 1.48 mm: the slide of a 1 deg turn about the feet's centre, the farthest foot "
            "standing 84.6 mm from it (r4 80.5 mm, 1.40 mm). On r4 the foot "
            "site under-read the toes' slip: on the 14 measured spawns whose site moves over 1 mm, a point of the "
            "sole under the most loaded contact slips a median 1.4x as far (0.72-2.34x; seed 110: site 2.21 mm, sole "
            "4.30 mm), so the slip was larger than it read. Strict: when a model change meets it this passes and so "
            "fails; run com_stance_calibration, check the 80 spawns, then drop the mark (the target becomes a real "
            "bar)"
        ),
    )
    def test_target_foot_slip_after_landing(self, statue):
        """Absolute target: from 1 s no foot slips farther than a 1 deg turn about the feet's centre slides it."""
        self.check_target(statue, "foot_slip_after_landing")
