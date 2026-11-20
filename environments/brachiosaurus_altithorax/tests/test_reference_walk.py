"""The reference must actually step, load the soles and advance on a freely moving body."""

from environments.brachiosaurus_altithorax.scripts.reference_walk import measure, run_walk


def test_reference_walk_transfers_weight_without_dragging_the_feet():
    report = measure(*run_walk())
    assert report["completed"]
    assert 0.25 < report["forward_speed_mps"] < 0.40
    assert report["minimum_loaded_feet"] >= 2
    assert report["airborne_substeps"] == 0
    assert report["nonleg_peak_load_n"] < 0.1
    assert report["peak_off_sole_step_mean_n"] < 0.1
    assert report["loaded_slip_p95_mps"] < 0.04
    assert min(report["peak_swing_clearance_m"].values()) > 0.03
    assert max(report["max_abs_pitch_roll_deg"]) < 4.0
    # A stable simulator can also walk in a deep crouch. Check the limb
    # posture while it actually carries weight, independently of the reset.
    assert all(5 < angle < 30 for angle in report["median_loaded_flexion_deg"].values())

    # The extraction window can start mid-stride. Its entire observed
    # sequence should proceed in order, with each foot landing repeatedly.
    cycle = ("rl", "fl", "rr", "fr")
    strikes = report["touchdowns"]
    assert len(strikes) >= 20
    first = cycle.index(strikes[0])
    assert strikes == [cycle[(first + i) % 4] for i in range(len(strikes))]
    assert 0.28 < report["median_ipsilateral_phase"] < 0.46
    assert all(0.5 < duty < 0.85 for duty in report["duty_factor"].values())
