"""The free-root walk must advance through alternating, load-bearing steps."""

from argparse import Namespace

import pytest

from environments.velociraptor_mongoliensis.scripts.audit_motion import run


@pytest.mark.parametrize(("noise", "seed"), [(0.0, 0), (0.05, 1)])
def test_reference_walk_transfers_weight_and_clears_the_claw(noise, seed):
    report = run(Namespace(mode="walk", xml=None, noise=noise, seconds=14, warmup=6), seed)
    assert not report["terminated"]
    assert report["pelvis_x_travel_m"] > 1.6
    assert abs(report["pelvis_y_drift_m"]) < 0.10
    assert report["alternating_touchdown_fraction"] == 1.0
    assert report["airborne_substep_fraction"] == 0.0
    assert report["unregistered_floor_load_fraction"] == 0.0

    duties = []
    for side, foot in report["feet"].items():
        assert foot["touchdowns"] >= 9
        assert 0.50 < foot["complete_cycle_duty_median"] < 0.80
        assert foot["swing_peak_ground_clearance_m_median"] > 0.012
        assert foot["loaded_patch_slip_mps_mean"] < 0.05
        assert report["min_geom_floor_clearance_m"][f"{side}_sickle"] > 0.010
        assert all(report["floor_impulse_share"][f"{side}_toe_d{digit}_geom"] > 0.03 for digit in (3, 4))
        duties.append(foot["complete_cycle_duty_median"])
    assert abs(duties[0] - duties[1]) < 0.10
