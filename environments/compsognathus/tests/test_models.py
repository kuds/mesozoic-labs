"""Protect the physical assumptions of these untrained prototypes."""

import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import pytest

from environments.compsognathus import MODEL_PATHS
from environments.compsognathus.model import PARAMETERS, load_model, model_bounds, robot_body_ids
from environments.compsognathus.scripts.validate_models import geometry, hold_trial


@pytest.mark.parametrize("variant,nq,nv,nu,mass", [("biological", 24, 23, 14, 1.0), ("robot", 19, 18, 12, 1.5856)])
def test_model_contract(variant, nq, nv, nu, mass):
    model, data = load_model(variant)
    assert (model.nq, model.nv, model.nu) == (nq, nv, nu)
    assert model.body_mass[robot_body_ids(model)].sum() == pytest.approx(mass)
    assert model.joint("root").type == mujoco.mjtJoint.mjJNT_FREE
    assert model.camera("head_camera").id >= 0
    assert model.site("tail_tip").id >= 0
    assert np.all(model.actuator_forcelimited)
    assert np.all(model.actuator_ctrllimited)
    for j in range(1, model.njnt):
        q = data.qpos[model.jnt_qposadr[j]]
        assert model.jnt_range[j, 0] <= q <= model.jnt_range[j, 1]
    for body in robot_body_ids(model):
        if model.body_mass[body] > 0:
            inertia = model.body_inertia[body]
            assert np.all(inertia > 0)
            assert 2 * max(inertia) <= sum(inertia) + 1e-12


def test_robot_preserves_rev_b_mechanics():
    root = Path(__file__).resolve().parents[1]
    source = ET.parse(root / "references/compso_rev_b.xml")
    source.find("compiler").set("meshdir", str(root / "assets/meshes"))
    reference = mujoco.MjModel.from_xml_string(ET.tostring(source.getroot(), encoding="unicode"))
    model, _ = load_model("robot")
    for body in range(1, reference.nbody):
        other = model.body(reference.body(body).name).id
        for field in ("body_pos", "body_quat", "body_mass", "body_ipos", "body_iquat", "body_inertia"):
            if reference.body(body).name == "fixed_head" and field in ("body_ipos", "body_iquat", "body_inertia"):
                # The styled head has an explicit new component COM/inertia,
                # validated separately, inside the original 45 g allowance.
                continue
            np.testing.assert_allclose(getattr(model, field)[other], getattr(reference, field)[body], atol=1e-12)
    for j in range(1, reference.njnt):
        other = model.joint(reference.joint(j).name).id
        np.testing.assert_allclose(model.jnt_range[other], reference.jnt_range[j])
        np.testing.assert_allclose(model.jnt_axis[other], reference.jnt_axis[j])
    # A finish revision must not silently shrink/remove a bracket or cover.
    # The source core/head/tail are aggregate envelopes, checked separately.
    for g in range(reference.ngeom):
        if reference.body(int(reference.geom_bodyid[g])).name in ("world", "core", "fixed_head", "passive_tail"):
            continue
        other = model.geom(reference.geom(g).name).id
        for field in ("geom_type", "geom_pos", "geom_quat", "geom_size"):
            np.testing.assert_allclose(getattr(model, field)[other], getattr(reference, field)[g], atol=1e-12)
        if reference.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            actual_mesh, expected_mesh = model.geom_dataid[other], reference.geom_dataid[g]
            assert model.mesh(int(actual_mesh)).name == reference.mesh(int(expected_mesh)).name
            for field, adr, count in (
                ("mesh_vert", "mesh_vertadr", "mesh_vertnum"),
                ("mesh_face", "mesh_faceadr", "mesh_facenum"),
            ):
                a, b = getattr(model, adr)[actual_mesh], getattr(reference, adr)[expected_mesh]
                na, nb = getattr(model, count)[actual_mesh], getattr(reference, count)[expected_mesh]
                np.testing.assert_array_equal(getattr(model, field)[a : a + na], getattr(reference, field)[b : b + nb])
    assert model.body_jntnum[model.body("passive_tail").id] == 0
    assert model.body_jntnum[model.body("fixed_head").id] == 0
    dimensions = geometry("robot")
    assert all(length == pytest.approx(0.07865) for length in dimensions["parallel_shaft_distances_m"].values())
    np.testing.assert_allclose(dimensions["sole_size_m"], [0.1, 0.09, 0.008])
    assert dimensions["sole_center_spacing_m"] == pytest.approx(0.13)
    _, data = load_model("robot")
    _, high = model_bounds(model, data)
    assert high[2] == pytest.approx(0.353)


@pytest.mark.parametrize("variant", ["biological", "robot"])
def test_standing_contacts_and_motor_limits(variant):
    trial = hold_trial(variant, seconds=10)
    assert trial["stood"], trial
    assert all(trial["standing_checks"].values()), trial
    assert trial["maximum_torque_cap_fraction"] <= 1.000001
    assert trial["final_support_margin_m"] > 0.005
    assert trial["foot_sensor_load_N"] == pytest.approx(trial["ground_normal_load_N"], rel=0.005)
    assert trial["ground_normal_load_N"] == pytest.approx(trial["body_weight_N"], rel=0.01)


def test_robot_stands_with_mass_growth_without_increasing_motor_caps():
    model, _ = load_model("robot")
    heavy, _ = load_model("robot", 1.15)
    assert heavy.body_mass[robot_body_ids(heavy)].sum() == pytest.approx(1.82344)
    np.testing.assert_array_equal(model.actuator_forcerange, heavy.actuator_forcerange)
    assert hold_trial("robot", mass_scale=1.15, seconds=5)["stood"]


@pytest.mark.parametrize("variant", ["biological", "robot"])
def test_standing_is_not_a_hidden_weld_or_passive_statue(variant):
    trial = hold_trial(variant, motors_off=True, seconds=5)
    assert not trial["stood"]
    assert max(trial["peak_torque_Nm"].values()) == 0


def test_robot_uses_derated_rated_torque_not_stall_torque():
    model, _ = load_model("robot")
    p = PARAMETERS["robot"]
    for i in range(model.nu):
        rated = (
            p["hip_roll_rated_torque_nm"]
            if model.actuator(i).name.endswith("hip_roll_act")
            else p["other_rated_torque_nm"]
        )
        cap = rated * p["regulated_voltage_assumption"] / p["rating_voltage"] * p["capacity_derating"]
        np.testing.assert_allclose(model.actuator_forcerange[i], [-cap, cap])


@pytest.mark.parametrize("scale", [0, -1, float("nan"), float("inf")])
def test_reject_invalid_mass_scale(scale):
    with pytest.raises(ValueError):
        load_model("robot", scale)


def test_registered_model_files_are_self_contained():
    for path in MODEL_PATHS.values():
        root = ET.parse(path)
        assert not root.findall(".//include")
        for mesh in root.findall("./asset/mesh"):
            if mesh.get("file"):
                assert (path.parent / root.find("compiler").get("meshdir") / mesh.get("file")).is_file()
            else:
                assert mesh.get("vertex") and mesh.get("face"), "Inline meshes must be self-contained"
