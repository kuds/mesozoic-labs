"""Negative controls for the juvenile design's independent geometry gate.

Small analytic fixtures make failures observable without relying on the final
candidate's layout, contact masks, mesh colors or presentation images.
"""

import math

import mujoco
import numpy as np
import pytest

from examples.mini_tyrannosaurus_rex.juvenile import validate


def packing_model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(
        """<mujoco><compiler angle="radian"/><worldbody>
          <body name="torso" pos="0 0 1"><freejoint/>
            <geom name="a" type="box" size=".01 .01 .01" contype="0" conaffinity="0"/>
            <geom name="b" type="box" size=".01 .01 .01" pos=".03 0 0" contype="0" conaffinity="0"/>
            <geom name="part" type="box" size=".005 .005 .005" pos=".1 0 0" euler="0 0 .7853981633974483"/>
            <body name="fixed_cavity" pos=".1 0 0">
              <geom name="envelope" type="box" size=".01 .01 .01" contype="0" conaffinity="0"/>
            </body>
          </body>
        </worldbody><keyframe><key name="home" qpos="0 0 1 1 0 0 0"/></keyframe></mujoco>"""
    )


def moving_model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(
        """<mujoco><compiler angle="radian"/><worldbody>
          <body name="torso" pos="0 0 1"><freejoint/>
            <geom name="a" type="box" size=".01 .01 .01" pos=".055 0 0" contype="0" conaffinity="0"/>
            <body name="head"><joint name="head_yaw" axis="0 0 1" range="-1.5707963267948966 1.5707963267948966"/>
              <geom name="b" type="box" size=".01 .01 .01" pos=".05 0 0" contype="0" conaffinity="0"/>
            </body>
          </body>
        </worldbody><actuator><position joint="head_yaw" ctrlrange="-1.5707963267948966 1.5707963267948966"/></actuator>
        <keyframe><key name="home" qpos="0 0 1 1 0 0 0 1.5707963267948966"/></keyframe></mujoco>"""
    )


def test_contact_disabled_overlap_and_submargin_gap_are_detected():
    model = packing_model()
    manifest = {"hardware_envelopes": ["a", "b"]}
    assert validate.clearance_screen(model, manifest, samples=0)["passed"]
    for position, expected_distance in ((0.015, -0.005), (0.021, 0.001)):
        model.geom_pos[model.geom("b").id, 0] = position
        report = validate.clearance_screen(model, manifest, samples=0)
        assert report["counts"]["fail"] == 1
        assert not report["passed"]
        witness = report["pairs"][0]["failure_witness"]
        assert witness["min_distance_m"] == pytest.approx(expected_distance)
        assert not witness["distance_is_lower_bound"]


def test_moving_parent_child_overlap_is_not_skipped_or_declared_rigid():
    model = moving_model()
    report = validate.clearance_screen(model, {"hardware_envelopes": ["a", "b"]}, samples=0, joint_points=3)
    pair = report["pairs"][0]
    assert pair["status"] == "fail"
    assert not pair["rigidly_co_mounted"]
    assert pair["evaluated_pose_count"] == report["pose_count"]
    assert pair["failure_witness"]["joints"]["head_yaw"] == pytest.approx(0)
    assert pair["failure_witness"]["min_distance_m"] == pytest.approx(-0.015)


def test_welded_child_pairs_have_one_query_without_losing_a_pair():
    model = packing_model()
    report = validate.clearance_screen(model, {"hardware_envelopes": ["a", "b", "envelope"]}, samples=7)
    assert report["passed"]
    assert report["pair_count"] == report["queries"] == 3
    assert report["pose_count"] == 8
    assert all(row["evaluated_pose_count"] == 1 for row in report["pairs"])


def test_far_pair_uses_an_explicit_conservative_certificate():
    model = packing_model()
    report = validate.clearance_screen(model, {"hardware_envelopes": ["a", "envelope"]}, samples=0)
    assert report["passed"]
    assert report["native_queries"] == 0
    assert report["broad_phase_certificates"] == 1
    witness = report["pairs"][0]["witness"]
    assert witness["native_distance_m"] is None
    assert witness["distance_is_lower_bound"]
    # The actual face-to-face gap is 80mm; the sphere bound may understate it.
    assert 0.002 < witness["min_distance_m"] < 0.080


def test_installation_envelope_is_checked_against_a_nonmotor_mechanism():
    model = packing_model()
    model.geom_pos[model.geom("b").id, 0] = 0.015
    report = validate.clearance_screen(model, {"hardware_envelopes": ["a"], "mechanism_geoms": ["b"]}, samples=0)
    assert report["counts"]["fail"] == 1
    assert report["pairs"][0]["categories"] == ["installation_vs_mechanism"]


def test_explicit_exclusion_requires_and_preserves_physical_reason():
    model = packing_model()
    spec = {
        "hardware_envelopes": ["a", "b", "envelope"],
        "excluded_pairs": [{"first": "a", "second": "b", "reason": "  "}],
    }
    with pytest.raises(ValueError, match="physical-interface reason"):
        validate.clearance_screen(model, spec, samples=0)
    spec["excluded_pairs"][0]["reason"] = "Analytic fixture intentionally declares this pair unqualified."
    report = validate.clearance_screen(model, spec, samples=0)
    assert report["pair_count"] == 2
    assert report["excluded_pairs"] == spec["excluded_pairs"]


@pytest.mark.parametrize("bad_dimension", [0, -0.001, math.nan, math.inf])
def test_degenerate_declared_dimensions_cannot_pass(bad_dimension):
    model = packing_model()
    model.geom_size[model.geom("b").id, 0] = bad_dimension
    with pytest.raises(ValueError, match="finite, positive dimensions"):
        validate.clearance_screen(model, {"hardware_envelopes": ["a", "b"]}, samples=0)


def test_containment_uses_rotated_shape_support_and_accepts_a_fixed_child_frame():
    model = packing_model()
    manifest = {"containment": [{"inner_geom": "part", "cavity_geom": "envelope", "allowance_m": 0.002}]}
    report = validate.containment_screen(model, manifest)
    assert report["passed"]
    assert report["checks"][0]["minimum_face_gap_m"] == pytest.approx(0.01 - 0.005 * np.sqrt(2))
    model.geom_size[model.geom("part").id] = 0.008
    report = validate.containment_screen(model, manifest)
    assert not report["passed"]
    assert report["checks"][0]["minimum_face_gap_m"] < 0


def test_dimension_contract_rejects_rescaling_an_installed_part():
    model = packing_model()
    manifest = {
        "hardware": [
            {"name": "fixture", "body_geom": "part", "envelope_geom": "envelope", "raw_dimensions_xyz_mm": [10, 10, 10]}
        ]
    }
    assert validate.integrity_screen(model, manifest)["passed"]
    model.geom_size[model.geom("part").id, 0] = 0.004
    assert not validate.integrity_screen(model, manifest)["passed"]
    manifest["hardware"][0]["raw_dimensions_xyz_mm"][0] = -1
    with pytest.raises(ValueError, match="three positive dimensions"):
        validate.integrity_screen(model, manifest)


def test_mass_contract_rejects_an_unaccounted_component():
    model = packing_model()
    manifest = {"mass_contract": {"total_mass_kg": float(model.body_mass.sum()), "expected_motor_count": 0}}
    assert validate.integrity_screen(model, manifest)["passed"]
    model.body_mass[model.body("torso").id] += 0.010
    assert not validate.integrity_screen(model, manifest)["passed"]


def test_below_margin_lower_bound_is_incomplete_and_cannot_become_a_pass():
    assert validate._distance_status({"min_distance_m": 0.001, "distance_is_lower_bound": True}, 0.002) == "incomplete"
    assert validate._distance_status({"min_distance_m": 0.0, "distance_is_lower_bound": False}, 0.002) == "incomplete"


def test_nanometer_roundoff_allowance_does_not_approve_contact_or_penetration():
    def status(distance, required=0.002):
        return validate._distance_status({"min_distance_m": distance, "distance_is_lower_bound": False}, required)

    assert status(0.002 - 3e-11) == "pass"
    assert status(0.002 - 2e-9) == "fail"
    assert status(0.0, required=1e-12) == "incomplete"
    assert status(-1e-12, required=1e-12) == "fail"


def test_seeded_joint_samples_are_reproducible_and_include_endpoints():
    model = moving_model()
    first = validate.sampled_poses(model, samples=4, seed=10, joint_points=3)
    assert first == validate.sampled_poses(model, samples=4, seed=10, joint_points=3)
    assert first != validate.sampled_poses(model, samples=4, seed=11, joint_points=3)
    individual = [row["joints"]["head_yaw"] for row in first if row["group"] == "individual:head_yaw"]
    assert individual == pytest.approx([-math.pi / 2, 0, math.pi / 2])


def test_compact_output_preserves_acceptance_and_every_nonpassing_pair():
    pairs = [{"status": "pass", "minimum_distance_m": 0.01 + i / 1000} for i in range(120)]
    pairs += [{"status": status, "minimum_distance_m": 0.001} for status in ["fail", "incomplete"]]
    report = {
        "status": "fail",
        "scoped_simulation_pass": False,
        "source_sha256": {"model.xml": "unchanged"},
        "clearance": {
            "pairs": pairs,
            "pair_count": 122,
            "pose_count": 500,
            "counts": {"pass": 120, "fail": 1, "incomplete": 1},
        },
    }
    compact = validate.compact_report(report, worst_pairs=3)
    assert compact["status"] == "fail"
    assert not compact["scoped_simulation_pass"]
    assert compact["source_sha256"] == report["source_sha256"]
    assert compact["clearance"]["counts"] == report["clearance"]["counts"]
    assert compact["clearance"]["pair_count"] == 122
    assert {row["status"] for row in compact["clearance"]["pairs"]} == {"pass", "fail", "incomplete"}
    assert compact["report_detail"]["omitted_passing_pair_count"] == 119
    assert len(report["clearance"]["pairs"]) == 122
