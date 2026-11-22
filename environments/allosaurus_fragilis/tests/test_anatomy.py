"""Anatomical dimensions, bilateral foot layout and visible tissue in the compiled plant."""

import mujoco
import numpy as np
import pytest

from environments.allosaurus_fragilis.anatomical.env import AllosaurusFragilisAnatomicalEnv
from environments.allosaurus_fragilis.scripts import build_anatomy


@pytest.fixture(scope="module")
def home():
    model = mujoco.MjModel.from_xml_path(str(AllosaurusFragilisAnatomicalEnv.MODEL_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
    return model, data


def test_leg_lengths_follow_adult_fossil_material(home):
    model, data = home
    # Persons & Currie (2016), Table 2, UUVP 6000r: femur, tibia, metatarsal III.
    for side in ("l", "r"):
        joints = [model.joint(f"{side}_{name}").id for name in ("hip_pitch", "knee", "ankle", "mtp")]
        lengths = np.linalg.norm(np.diff(data.xanchor[joints], axis=0), axis=1)
        np.testing.assert_allclose(lengths, [0.880, 0.730, 0.375], rtol=0.025)
    offset = data.subtree_com[model.body("pelvis").id] - data.body("pelvis").xpos
    assert 0.05 < offset[0] < 0.375 and abs(offset[1]) < 1e-6 and offset[2] < 0


def test_three_splayed_toes_and_elevated_hallux_on_mirrored_feet(home):
    model, data = home
    suffixes = ("toe_d2", "toe_d3", "toe_d4", "hallux")
    left = np.array([data.site("l_" + suffix).xpos for suffix in suffixes])
    right = np.array([data.site("r_" + suffix).xpos for suffix in suffixes])
    np.testing.assert_allclose(left, right * [1, -1, 1], atol=1e-8)
    toe_2, toe_3, toe_4, hallux = left
    assert toe_2[1] < toe_3[1] < toe_4[1]
    assert toe_3[0] > max(toe_2[0], toe_4[0])
    assert hallux[1] < toe_3[1] and hallux[2] > 0.08
    floor = model.geom("floor").id
    supports = {next(g for g in contact.geom if g != floor) for contact in data.contact if floor in contact.geom}
    assert supports == {
        model.geom(f"{s}_{pad}").id for s in ("r", "l") for pad in ("pad", "pad_d2", "pad_d3", "pad_d4")
    }


def test_visible_foot_skin_meets_the_floor_in_the_reference_pose(home):
    model, data = home
    scene = mujoco.MjvScene(model, maxgeom=model.ngeom + 100)
    options = mujoco.MjvOption()
    options.skingroup[:] = 1
    mujoco.mjv_updateScene(model, data, options, None, mujoco.MjvCamera(), mujoco.mjtCatBit.mjCAT_ALL, scene)
    clearance = np.minimum.reduceat(scene.skinvert.reshape(-1, 3)[:, 2], model.skin_vertadr)
    for side in ("l", "r"):
        assert clearance[model.skin(side + "_hallux_claw").id] > 0.08
        for digit in (2, 3, 4):
            assert 0 <= clearance[model.skin(f"{side}_digit_{digit}").id] < 0.01
            assert clearance[model.skin(f"{side}_ungual_{digit}").id] > 0
        plantar = model.geom(side + "_plantar_hide")
        zmin = data.geom_xpos[plantar.id, 2] - plantar.size[2]
        assert -0.001 < zmin <= 0


def test_rebuilt_geometry_and_pigment_match_packaged_model(home, tmp_path):
    model, _ = home
    build_anatomy.write(tmp_path)
    rebuilt = mujoco.MjModel.from_xml_path(str(tmp_path / build_anatomy.MODEL_NAME))
    # Compare decoded data so PNG compression differences across platforms are immaterial.
    np.testing.assert_array_equal(rebuilt.tex_data, model.tex_data)
    np.testing.assert_allclose(rebuilt.skin_vert, model.skin_vert, atol=1e-6)
    np.testing.assert_array_equal(rebuilt.skin_face, model.skin_face)
    np.testing.assert_allclose(rebuilt.skin_bonevertweight, model.skin_bonevertweight, atol=1e-6)
    np.testing.assert_allclose(rebuilt.body_pos, model.body_pos, atol=1e-8)
    np.testing.assert_array_equal(rebuilt.key_ctrl, model.key_ctrl)
