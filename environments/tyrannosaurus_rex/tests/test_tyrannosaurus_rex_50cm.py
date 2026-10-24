"""Protect the medium concept's mechanics and honest free-base stance checks."""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from examples.tyrannosaurus_rex_50cm import build_model, demo, validate


def test_components_reconcile_with_detailed_parts_mass_budget():
    model, _ = demo.load_model()
    ledger = json.loads(demo.MODEL_PATH.with_name("components.json").read_text())
    budget_path = Path(__file__).resolve().parents[3] / "docs/hardware/studies/trex_50cm/parts_budget.json"
    budget = json.loads(budget_path.read_text())
    assert ledger["categories_kg"] == pytest.approx({k: v / 1000 for k, v in budget["mass_allowances_g"].items()})
    assert sum(c["mass_kg"] for c in ledger["components"]) == pytest.approx(7.617)
    assert model.body_mass.sum() == pytest.approx(7.617)
    for b in range(1, model.nbody):
        expected = sum(c["mass_kg"] for c in ledger["components"] if c["body"] == model.body(b).name)
        assert model.body_mass[b] == pytest.approx(expected)
        inertia = model.body_inertia[b]
        assert np.all(inertia > 0)
        assert 2 * max(inertia) <= sum(inertia) + 1e-12


def test_interface_and_physics_have_no_hidden_support():
    model, data = demo.load_model()
    assert (model.nq, model.nv, model.nu, model.nsensor) == (25, 24, 18, 48)
    assert model.joint("root").type == mujoco.mjtJoint.mjJNT_FREE
    assert model.opt.gravity == pytest.approx([0, 0, -9.81])
    assert model.neq == 1
    assert not model.eq_active0.any() and not data.eq_active.any()
    assert model.nmocap == 0
    assert not model.body_gravcomp.any()
    assert not data.xfrc_applied.any()
    assert np.count_nonzero(model.geom_bodyid == 0) == 1
    assert model.actuator_forcelimited.all()
    assert model.actuator_forcerange[:12] == pytest.approx(np.tile([-4, 4], (12, 1)))
    for j in range(1, model.njnt):
        assert model.jnt_limited[j]
        assert model.jnt_range[j, 0] <= data.qpos[model.jnt_qposadr[j]] <= model.jnt_range[j, 1]
    assert not validate.internal_contacts(model, data)
    assert data.ncon == 8
    for c in data.contact[: data.ncon]:
        assert model.geom(c.geom1).name == "floor"
        assert model.geom(c.geom2).name in ("left_sole_collision", "right_sole_collision")
        assert c.dist > -0.0001


def test_size_knee_direction_and_twelve_provisional_motor_envelopes():
    model, data = demo.load_model()
    low, high = validate.visual_bounds(model, data)
    assert 0.48 <= high[2] - low[2] <= 0.52
    assert high[1] - low[1] < 0.35
    cases = [model.geom(i) for i in range(model.ngeom) if model.geom(i).name.endswith("_case_collision")]
    assert len(cases) == 12
    for case in cases:
        assert case.size[:2] == pytest.approx([0.030, 0.0275])
        assert case.contype[0] and case.conaffinity[0]
    for side in ("left", "right"):
        hip, knee, ankle = [data.xanchor[model.joint(f"{side}_{j}").id] for j in ("hip_pitch", "knee", "ankle_pitch")]
        fraction = (knee[2] - hip[2]) / (ankle[2] - hip[2])
        assert knee[0] - (hip[0] + fraction * (ankle[0] - hip[0])) > 0.02


def test_cameras_are_in_the_head_snout_and_foot_sensors_are_on_feet():
    model, data = demo.load_model()
    for name in ("snout_left", "snout_rgb", "snout_right"):
        camera = model.camera(name).id
        assert model.cam_bodyid[camera] == model.body("head").id
        assert model.cam_pos[camera, 0] > 0.13
        # MuJoCo cameras look down their local negative Z axis.
        assert -data.cam_xmat[camera].reshape(3, 3)[:, 2] == pytest.approx([1, 0, 0])
    for side in ("left", "right"):
        for i in range(4):
            site = model.site(f"{side}_contact_{i}").id
            assert model.site_bodyid[site] == model.body(side + "_foot").id
            assert model.sensor(f"{side}_contact_force_{i}").type == mujoco.mjtSensor.mjSENS_TOUCH


@pytest.mark.parametrize("gesture", [False, True])
def test_ten_second_free_stance_and_character_regressions(gesture):
    model, _ = demo.load_model()
    result = validate.run_trial(model, duration=10, gesture=gesture)
    assert result["passed"], result
    assert not result["external_support_active"]
    assert max(result["rms_torque_nm"][:12]) < 2.8
    assert sum(result["final_foot_contact_force_n"]) == pytest.approx(model.body_mass.sum() * 9.81, rel=0.05)
    if gesture:
        for name, minimum in (
            ("head_yaw", 0.3),
            ("head_nod", 0.06),
            ("jaw_pitch", 0.10),
            ("tail_yaw", 0.30),
            ("left_shoulder_pitch", 0.45),
            ("right_shoulder_pitch", 0.35),
        ):
            idx = model.joint(name).id - 1
            assert result["joint_excursion_rad"][idx] > minimum, name


def test_display_weld_is_explicit_and_optional():
    model, data = demo.load_model()
    demo.set_support(model, data, True)
    assert data.eq_active[0]
    demo.set_support(model, data, False)
    assert not data.eq_active[0]
    result = validate.run_trial(model, duration=3, supported=True, gesture=True)
    assert result["passed"] and result["external_support_active"]


def test_stance_check_rejects_insufficient_leg_torque():
    model, _ = demo.load_model()
    model.actuator_forcerange[:12, :] = [-0.1, 0.1]
    result = validate.run_trial(model, duration=3)
    assert not result["passed"]
    assert result["min_height_ratio"] < 0.9 or result["max_torso_tilt_deg"] > 10


def test_source_generator_reproduces_committed_model_and_assets(tmp_path):
    build_model.build(tmp_path)
    for name in ("model.xml", "components.json", "assets/rounded.obj", "assets/soft.obj", "assets/tail.obj"):
        assert (tmp_path / name).read_bytes() == (demo.MODEL_PATH.parent / name).read_bytes(), name


def test_generator_canonicalizes_cross_version_float_sum_residue():
    # This difference was observed between Python 3.11 and 3.12 in Linux CI.
    assert build_model.values(3.095999999999999) == build_model.values(3.096) == "3.096"
    assert build_model.values((0.0, -3.5e-20, 1.2)) == "0 0 1.2"


@pytest.mark.parametrize("duration", [0, -1, float("nan"), float("inf")])
def test_invalid_trial_duration_is_rejected(duration):
    model, _ = demo.load_model()
    with pytest.raises(ValueError):
        validate.run_trial(model, duration=duration)
