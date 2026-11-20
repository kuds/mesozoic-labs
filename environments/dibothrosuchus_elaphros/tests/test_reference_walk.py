"""Require observed four-beat footfalls and clearance under the normal actuator path."""

from environments.dibothrosuchus_elaphros.scripts.reference_walk import measure, run_walk


def test_slow_walk_loads_and_clears_each_foot():
    # Longer than the default preview: unbounded forward position feedback
    # can preserve the first six strides yet disrupt later weight transfers.
    report = measure(*run_walk(strides=20))
    assert report["completed"]
    assert 0.09 < report["forward_speed_mps"] < 0.21
    assert 0.75 < report["median_stride_seconds"] < 0.85
    assert all(0.55 < duty < 0.82 for duty in report["duty_factor"].values())
    assert all(0.07 < stride < 0.16 for stride in report["median_stride_length_m"].values())
    assert min(report["peak_swing_clearance_m"].values()) > 0.008
    assert all(3 < angle < 13 for angle in report["loaded_heel_up_pitch_p95_deg"].values())
    assert min(report["minimum_track_width_m"].values()) > 0.07
    assert all(0.15 < share < 0.90 for share in report["digital_load_fraction"].values())
    assert report["minimum_loaded_feet"] >= 2
    assert report["airborne_substeps"] == 0
    assert report["nonleg_peak_load_n"] < 0.1
    assert report["peak_off_sole_step_mean_n"] < 0.1
    assert report["loaded_slip_p95_mps"] < 0.035
    assert max(report["max_abs_pitch_roll_deg"]) < 3
    cycle = ["rl", "fl", "rr", "fr"]
    strikes = report["touchdowns"]
    assert 67 <= len(strikes) <= 69  # 17 measured strides with four beats each
    first = cycle.index(strikes[0])
    assert strikes == [cycle[(first + i) % 4] for i in range(len(strikes))]
