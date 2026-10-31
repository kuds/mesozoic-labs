"""Protect sourced dimensions, installation space and mass accounting."""

import json
from copy import deepcopy
from pathlib import Path
from xml.etree.ElementTree import Element, SubElement, tostring

import mujoco
import numpy as np
import pytest

from examples.tyrannosaurus_rex_50cm import hardware_layout
from examples.tyrannosaurus_rex_50cm.hardware_checks import check_hardware
from examples.tyrannosaurus_rex_50cm.mechanical_checks import EXPRESSION_JOINTS, LEG_JOINTS

HERE = Path(hardware_layout.__file__).parent


def _text(values):
    return " ".join(str(value) for value in values)


@pytest.fixture
def hardware_fixture():
    """Small independent model: two bays, one motor, all sampled joint names."""
    catalog = {
        "compute": {"dimensions_m": [0.1, 0.08, 0.04], "tolerance_m": [0.002] * 3, "mass_kg": 0.1},
        "battery": {"dimensions_m": [0.04, 0.03, 0.02], "tolerance_m": [0, 0, 0], "mass_kg": 0.15},
        "motor": {"dimensions_m": [0.04, 0.03, 0.02], "tolerance_m": [0, 0, 0], "mass_kg": 0.09},
    }
    for name, part in catalog.items():
        part.update(
            part=name,
            mass_tolerance_kg=0.005,
            dimension_basis="vendor_complete",
            shape="box",
            source_url="https://example.invalid/fixture",
            notes="Synthetic source evidence for a fit test.",
        )
    assemblies = []
    for name, pos, size, budget in (
        ("compute", [0, 0, 0.2], [0.12, 0.11, 0.065], 0.4),
        ("battery", [0.22, 0, 0.2], [0.08, 0.08, 0.06], 0.3),
    ):
        assemblies.append(
            {
                "id": name,
                "component": name,
                "geometry": name + "_installed",
                "container": name + "_container",
                "body": "torso",
                "position_m": pos,
                "installed_size_m": size,
                "mass_budget_kg": budget,
                "wall_clearance_m": 0.003,
                "contents": [
                    {
                        "id": name + "_unit",
                        "catalog": name,
                        "geometry": name + "_actual",
                        "offset_m": [0, 0, 0],
                        "catalog_axes": [0, 1, 2],
                        "size_m": (np.array(catalog[name]["dimensions_m"]) + catalog[name]["tolerance_m"]).tolist(),
                    }
                ],
            }
        )
    manifest = {
        "schema_version": 1,
        "catalog": catalog,
        "assemblies": assemblies,
        "mounted_hardware": [
            {"component": "motor", "geometry": "motor_actual", "catalog": "motor", "catalog_axes": [0, 1, 2]}
        ],
        "expected_quantities": {name: 1 for name in catalog},
        "uncertainties": ["Synthetic fixture."],
    }
    components = [
        {"name": "structure", "body": "torso", "mass_kg": 1.2, "pos_m": [0, 0, 0]},
        {"name": "motor", "body": "torso", "mass_kg": 0.1, "pos_m": [1.2, 0, 0.2]},
    ]
    components.extend(
        {"name": part["component"], "body": "torso", "mass_kg": part["mass_budget_kg"], "pos_m": part["position_m"][:]}
        for part in assemblies
    )
    xml = Element("mujoco")
    SubElement(xml, "compiler", angle="radian")
    SubElement(xml, "option", gravity="0 0 -9.81")
    world = SubElement(xml, "worldbody")
    torso = SubElement(world, "body", name="torso", pos="0 0 1")
    SubElement(torso, "freejoint", name="root")
    mass = sum(part["mass_kg"] for part in components)
    center = sum(part["mass_kg"] * np.array(part["pos_m"]) for part in components) / mass
    SubElement(torso, "inertial", mass=str(mass), pos=_text(center), diaginertia="1 1 1")
    for side, lateral in (("left", 0.2), ("right", -0.2)):
        SubElement(torso, "geom", name=side + "_sole_collision", type="box", size=".1 .05 .01", pos=f"0 {lateral} -.5")
    for assembly in assemblies:
        SubElement(
            torso,
            "geom",
            name=assembly["geometry"],
            type="box",
            group="5",
            size=_text(np.array(assembly["installed_size_m"]) / 2),
            pos=_text(assembly["position_m"]),
        )
        SubElement(
            torso,
            "geom",
            name=assembly["container"],
            type="ellipsoid",
            size=".16 .13 .1",
            pos=_text(assembly["position_m"]),
        )
        unit = assembly["contents"][0]
        SubElement(
            torso,
            "geom",
            name=unit["geometry"],
            type="box",
            group="4",
            size=_text(np.array(unit["size_m"]) / 2),
            pos=_text(assembly["position_m"]),
        )
    SubElement(torso, "geom", name="motor_actual", type="box", size=".02 .015 .01", pos="1.2 0 .2")
    for index, joint in enumerate((*LEG_JOINTS, *EXPRESSION_JOINTS)):
        body = SubElement(torso, "body", name=joint + "_body", pos=f"{3 + index} 0 0")
        SubElement(body, "joint", name=joint, type="hinge", axis="0 1 0", range="-1 1")
        SubElement(body, "inertial", mass=".01", pos="0 0 0", diaginertia=".001 .001 .001")
        components.append({"name": joint, "body": joint + "_body", "mass_kg": 0.01, "pos_m": [0, 0, 0]})
    SubElement(SubElement(xml, "keyframe"), "key", name="home")
    model = mujoco.MjModel.from_xml_string(tostring(xml, encoding="unicode"))
    ledger = {"components": components, "mass_kg": sum(part["mass_kg"] for part in components)}
    return model, ledger, manifest


def _row(report, name):
    return next(row for row in report["checks"] if row["name"] == name)


def test_hardware_check_is_read_only_and_preserves_incomplete_evidence(hardware_fixture):
    model, ledger, manifest = hardware_fixture
    before = deepcopy((ledger, manifest))
    model_before = {
        name: getattr(model, name).copy() for name in ("body_mass", "body_ipos", "geom_pos", "geom_size", "qpos0")
    }
    report = check_hardware(model, ledger, manifest)
    assert report["status"] == "incomplete"
    assert report["known_limits_passed"]
    assert report["coverage"]["packing_poses"] == 790
    assert json.loads(json.dumps(report, allow_nan=False))["status"] == "incomplete"
    assert (ledger, manifest) == before
    for name, values in model_before.items():
        np.testing.assert_array_equal(getattr(model, name), values)


def test_original_compute_reservation_is_too_small(hardware_fixture):
    model, ledger, manifest = hardware_fixture
    manifest["assemblies"][0]["installed_size_m"] = [0.060, 0.100, 0.042]
    model.geom("compute_installed").size[:] = np.array([0.060, 0.100, 0.042]) / 2
    report = check_hardware(model, ledger, manifest)
    assert report["status"] == "failed"
    assert _row(report, "compute_unit:installed_edge")["status"] == "failed"


def test_catalog_tolerance_and_orientation_are_part_of_fit(hardware_fixture):
    model, ledger, manifest = hardware_fixture
    unit = manifest["assemblies"][0]["contents"][0]
    unit["catalog_axes"] = [1, 0, 2]
    report = check_hardware(model, ledger, manifest)
    assert _row(report, "compute_unit")["status"] == "failed"
    unit["size_m"] = [0.082, 0.102, 0.042]
    model.geom("compute_actual").size[:] = np.array(unit["size_m"]) / 2
    assert _row(check_hardware(model, ledger, manifest), "compute_unit")["status"] == "passed"
    model.geom("compute_actual").size[:] -= 0.0001
    assert _row(check_hardware(model, ledger, manifest), "compute_unit")["status"] == "failed"


def test_cylinder_fit_uses_circle_and_catalog_cylinder_shape(hardware_fixture):
    model, ledger, manifest = hardware_fixture
    motor = model.geom("motor_actual")
    model.geom_type[motor.id] = mujoco.mjtGeom.mjGEOM_CYLINDER
    motor.size[:] = [0.025, 0.02, 0]
    manifest["catalog"]["motor"]["dimensions_m"] = [0.04, 0.04, 0.02]
    assert _row(check_hardware(model, ledger, manifest), "motor")["status"] == "failed"
    manifest["catalog"]["motor"].update(shape="cylinder", dimensions_m=[0.05, 0.05, 0.04])
    assert _row(check_hardware(model, ledger, manifest), "motor")["status"] == "passed"
    manifest["mounted_hardware"][0]["catalog_axes"] = [2, 0, 1]
    assert _row(check_hardware(model, ledger, manifest), "motor")["status"] == "failed"


def test_fitting_motor_box_does_not_hide_an_incorrect_shaft_direction(hardware_fixture):
    model, ledger, manifest = hardware_fixture
    model.geom("motor_actual").size[:] = 0.05
    manifest["catalog"]["motor"].update(shaft_axis=2, shaft_offset_m=[0, 0, 0])
    mount = manifest["mounted_hardware"][0]
    mount.update(joint=LEG_JOINTS[0], catalog_axes=[0, 2, 1])
    assert _row(check_hardware(model, ledger, manifest), "motor:shaft_alignment")["status"] == "passed"
    mount["catalog_axes"] = [0, 1, 2]
    report = check_hardware(model, ledger, manifest)
    assert _row(report, "motor")["status"] == "passed"
    assert _row(report, "motor:shaft_alignment")["status"] == "failed"


def test_parallel_motor_shaft_must_also_pass_through_the_joint_axis(hardware_fixture):
    model, ledger, manifest = hardware_fixture
    model.geom("motor_actual").size[:] = 0.05
    model.body(LEG_JOINTS[0] + "_body").pos[:] = model.geom("motor_actual").pos
    manifest["catalog"]["motor"].update(shaft_axis=2, shaft_offset_m=[0, 0, 0])
    manifest["mounted_hardware"][0].update(joint=LEG_JOINTS[0], catalog_axes=[0, 2, 1])
    assert _row(check_hardware(model, ledger, manifest), "motor:shaft_line")["status"] == "passed"
    model.geom("motor_actual").pos[0] += 0.01
    report = check_hardware(model, ledger, manifest)
    assert _row(report, "motor:shaft_alignment")["status"] == "passed"
    assert _row(report, "motor:shaft_line")["status"] == "failed"
    assert _row(report, "motor:shaft_line")["perpendicular_distance_m"] == pytest.approx(0.01)


def test_single_container_checks_all_box_corners_and_wall_allowance(hardware_fixture):
    model, ledger, manifest = hardware_fixture
    # Every box axis individually fits, but its corners penetrate the ellipsoid.
    model.geom("battery_container").size[:] = [0.05, 0.05, 0.04]
    report = check_hardware(model, ledger, manifest)
    assert _row(report, "battery:container")["status"] == "failed"
    assert _row(report, "battery:container")["maximum_ellipsoid_score"] > 1


def test_installed_assembly_overlap_is_a_failed_packing_check(hardware_fixture):
    model, ledger, manifest = hardware_fixture
    model.geom("battery_installed").pos[:] = model.geom("compute_installed").pos
    report = check_hardware(model, ledger, manifest)
    assert report["packing"]["status"] == "failed"
    assert report["status"] == "failed"


def test_hidden_motor_geometry_cannot_move_away_from_its_mass_allocation(hardware_fixture):
    model, ledger, manifest = hardware_fixture
    model.geom("motor_actual").pos[0] += 0.02
    assert _row(check_hardware(model, ledger, manifest), "motor:placement")["status"] == "failed"


@pytest.mark.parametrize(
    "change, failed_row",
    [
        ("quantity", "catalog_quantities"),
        ("double_mass", "compiled_body_mass"),
        ("center", "compiled_body_centers"),
        ("budget", "battery:mass_budget"),
    ],
)
def test_quantity_and_mass_accounting_failures(hardware_fixture, change, failed_row):
    model, ledger, manifest = hardware_fixture
    if change == "quantity":
        manifest["expected_quantities"]["battery"] = 2
    elif change == "double_mass":
        ledger["components"].append(
            {"name": "battery_added_twice", "body": "torso", "mass_kg": 0.15, "pos_m": [0.22, 0, 0.2]}
        )
    elif change == "center":
        model.body("torso").ipos[0] += 0.01
    else:
        manifest["assemblies"][1]["mass_budget_kg"] = 0.15  # Catalog mass + 5 g tolerance exceeds this.
    assert _row(check_hardware(model, ledger, manifest), failed_row)["status"] == "failed"


def test_explicit_planning_evidence_stays_unknown(hardware_fixture):
    model, ledger, manifest = hardware_fixture
    manifest["catalog"]["compute"].update(dimensions_m=None, mass_kg=None, source_url=None, dimension_basis="planning")
    report = check_hardware(model, ledger, manifest)
    assert report["known_limits_passed"]
    assert _row(report, "compute_unit")["status"] == "incomplete"
    assert _row(report, "compute:unweighed_contents")["status"] == "incomplete"


@pytest.mark.parametrize(
    "change", ["axes", "duplicate_geometry", "duplicate_component", "nan", "missing_evidence", "missing_source"]
)
def test_malformed_declarations_raise_value_error(hardware_fixture, change):
    model, ledger, manifest = hardware_fixture
    if change == "axes":
        manifest["mounted_hardware"][0]["catalog_axes"] = [0, 0, 2]
    elif change == "duplicate_geometry":
        manifest["mounted_hardware"][0]["geometry"] = "compute_actual"
    elif change == "duplicate_component":
        ledger["components"].append(deepcopy(ledger["components"][0]))
    elif change == "nan":
        manifest["catalog"]["compute"]["dimensions_m"][0] = float("nan")
    elif change == "missing_evidence":
        del manifest["catalog"]["compute"]["mass_kg"]
    else:
        manifest["catalog"]["compute"]["source_url"] = None
    with pytest.raises(ValueError):
        check_hardware(model, ledger, manifest)


@pytest.fixture
def candidate_hardware():
    model = mujoco.MjModel.from_xml_path(str(HERE / "long_body/model.xml"))
    ledger = json.loads((HERE / "long_body/components.json").read_text())
    return model, ledger, hardware_layout.load_manifest()


def test_committed_candidate_keeps_complete_hardware_coverage_and_mass_reserve(candidate_hardware):
    model, ledger, manifest = candidate_hardware
    # Corner containment is a static proof only when the bay and shell share
    # a rigid frame; a future moving-shell reference must fail this contract.
    for assembly in manifest["assemblies"]:
        body = model.body(assembly["body"]).id
        assert model.geom_bodyid[model.geom(assembly["container"]).id] == body
        assert model.geom_bodyid[model.geom(assembly["geometry"]).id] == body
    report = check_hardware(model, ledger, manifest)
    assert report["known_limits_passed"], [row for row in report["checks"] if row["status"] == "failed"]
    assert report["status"] == "incomplete"
    assert report["coverage"] == {
        "assemblies": 4,
        "content_units": 12,
        "mounted_units": 19,
        "packing_poses": 790,
        "packing_pairs": 187,
        "internal_edge_clearance_m": 0.001,
        "pair_clearance_m": 0.002,
    }
    assert report["model_mass_kg"] == pytest.approx(8.207)
    assert ledger["categories_kg"]["unallocated_integration_reserve"] == pytest.approx(0.6)
    assert manifest["expected_quantities"] == {
        "jetson": 1,
        "battery": 1,
        "teensy": 1,
        "can": 2,
        "regulator_12v": 3,
        "regulator_5v": 2,
        "u2d2": 2,
        "camera": 1,
        "xm430": 2,
        "xl330": 4,
        "rmd": 12,
    }
    # A coarse home balance/height contract, not measured or dynamic stability.
    assert _com_in_design_window(report["home_com_m"])
    assert len([row for row in report["checks"] if row["category"] == "catalog_fit"]) == 31
    assert len([row for row in report["checks"] if row["category"] == "shaft_orientation"]) == 18
    assert len([row for row in report["checks"] if row["category"] == "shaft_location"]) == 18
    assert sum(row["status"] == "incomplete" for row in report["checks"] if row["category"] == "catalog_fit") >= 13


def _com_in_design_window(com):
    return -0.020 < com[0] < 0.025 and abs(com[1]) < 0.003 and 0.240 < com[2] < 0.280


def test_consistent_mass_relocation_still_changes_the_balance_contract(candidate_hardware):
    model, ledger, manifest = candidate_hardware
    battery = next(part for part in ledger["components"] if part["name"] == "battery")
    battery["pos_m"][0] += 0.25
    torso = model.body("torso")
    torso_parts = [part for part in ledger["components"] if part["body"] == "torso"]
    torso.ipos[:] = sum(part["mass_kg"] * np.array(part["pos_m"]) for part in torso_parts) / torso.mass[0]
    report = check_hardware(model, ledger, manifest)
    assert _row(report, "home_center_of_mass")["status"] == "passed"
    assert not _com_in_design_window(report["home_com_m"])


def test_multiple_real_regulators_cannot_share_one_placeholder(candidate_hardware):
    model, ledger, manifest = candidate_hardware
    power = next(assembly for assembly in manifest["assemblies"] if assembly["id"] == "power")
    first, second = power["contents"][:2]
    model.geom(second["geometry"]).pos[:] = model.geom(first["geometry"]).pos
    report = check_hardware(model, ledger, manifest)
    assert any(row["status"] == "failed" for row in report["checks"] if row["category"] == "internal_clearance")


@pytest.mark.parametrize("former_conflict", ["battery_tail", "jaw_nod"])
def test_candidate_detects_former_actual_hardware_intersections(candidate_hardware, former_conflict):
    model, ledger, manifest = candidate_hardware
    if former_conflict == "battery_tail":
        model.geom("battery_envelope").pos[2] += 0.018
    else:
        jaw = next(item for item in manifest["mounted_hardware"] if item["component"] == "jaw_motor")
        model.geom(jaw["geometry"]).pos[1] = 0
    report = check_hardware(model, ledger, manifest)
    assert report["packing"]["status"] == "failed"
    assert not report["known_limits_passed"]
