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

import pytest

from environments.shared.model_id import ModelId
from environments.shared.tests.com_stance_helpers import ComStanceQualityBase, TargetMissed


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

    @pytest.mark.xfail(
        strict=True,
        raises=TargetMissed,
        reason=(
            "physics r4 (2026-10-29): the turn read at the episode's end (the yaw at its end less at the spawn): "
            "root_turn_from_spawn_deg seed 131 5.66 deg, seed 110 5.19 deg (52 of the 80 spawns at reset noise 0.02 "
            "miss; worst 5.66 deg, seed 131; median 1.32 deg). Target: |root_turn_from_spawn_deg| <= 1.00 deg. Strict: "
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
            "physics r4 (2026-10-29): settle_s seed 131 3 s, seed 110 3 s (60 of the 80 spawns at reset noise 0.02 "
            "miss; worst 3 s, seed 0; median 3 s). Target: |settle_s| <= 2 s. Strict: when a model change meets it "
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
            "physics r4 (2026-10-29): foot_travel_max_m seed 131 1.49 mm, seed 110 2.21 mm (44 of the 80 spawns at "
            "reset noise 0.02 miss; worst 4.46 mm, seed 20; median 1.52 mm). Target: |foot_travel_max_m| <= 1.40 mm: "
            "the slide of a 1 deg turn about the feet's centre, the farthest foot standing 80.5 mm from it. The foot "
            "site under-reads the toes' slip: on the 14 measured spawns whose site moves over 1 mm, a point of the "
            "sole under the most loaded contact slips a median 1.4x as far (0.72-2.34x; seed 110: site 2.21 mm, sole "
            "4.30 mm), so the slip is larger than this reads. Strict: when a model change meets it this passes and so "
            "fails; run com_stance_calibration, check the 80 spawns, then drop the mark (the target becomes a real "
            "bar)"
        ),
    )
    def test_target_foot_slip_after_landing(self, statue):
        """Absolute target: from 1 s no foot slips farther than a 1 deg turn about the feet's centre slides it."""
        self.check_target(statue, "foot_slip_after_landing")
