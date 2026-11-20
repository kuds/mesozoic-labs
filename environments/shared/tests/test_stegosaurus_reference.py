"""Guard the published proportions, artifact and measured motion of the free-root example."""

import mujoco
import numpy as np

from examples.stegosaurus_stenops.build import MODEL_PATH, write_model
from examples.stegosaurus_stenops.validate import FEET, FloorProbe, run_validation, touchdowns


def test_published_model_matches_builder(tmp_path):
    write_model(tmp_path)
    for name in ("model.xml", "limb_skins.xml"):
        assert (tmp_path / name).read_bytes() == MODEL_PATH.with_name(name).read_bytes()


def test_specimen_geometry_and_distinct_footprints():
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    probe = FloorProbe(model)
    length = data.site("snout").xpos[0] - data.site("tail_tip").xpos[0]
    assert 5.5 < length < 6.0
    plate_positions = [data.geom(f"plate_{i:02}").xpos for i in range(1, 20)]
    np.testing.assert_array_less(np.diff([p[0] for p in plate_positions]), 0)
    sides = np.sign([p[1] for p in plate_positions])
    np.testing.assert_array_equal(sides[:-1], -sides[1:])
    names = [model.geom(i).name for i in range(model.ngeom)]
    assert sum(name.startswith("plate_") for name in names) == 19
    assert sum(name.startswith("caudal_spike_") for name in names) == 4

    areas = {}
    for leg, geom_ids in zip(FEET, probe.soles, strict=True):
        fore = leg.endswith("f")
        proximal = np.linalg.norm(data.body(f"{leg}_upper").xpos - data.body(f"{leg}_lower").xpos)
        distal = np.linalg.norm(data.body(f"{leg}_lower").xpos - data.body(f"{leg}_foot").xpos)
        np.testing.assert_allclose((proximal, distal), (0.450, 0.312) if fore else (0.868, 0.498), atol=0.001)
        # Exact plan-view bounding rectangle of the oriented sole ellipsoids in the neutral pose.
        assert np.all(model.geom_type[geom_ids] == mujoco.mjtGeom.mjGEOM_ELLIPSOID)
        axis = data.geom_xmat[geom_ids].reshape(-1, 3, 3)[:, :2, :]
        radius = np.linalg.norm(axis * model.geom_size[geom_ids, None, :], axis=-1)
        centers = data.geom_xpos[geom_ids, :2]
        toe_to_heel, width = (centers + radius).max(axis=0) - (centers - radius).min(axis=0)
        assert (width > toe_to_heel) if fore else (toe_to_heel > width)
        areas[leg] = toe_to_heel * width
    for side in ("l", "r"):
        assert 0.25 < areas[side + "f"] / areas[side + "h"] < 0.50


def test_reloading_a_planted_foot_does_not_count_a_second_step():
    load = [0.0, 0.02, 0.009, 0.012, 0.0, 0.0, 0.013]
    clearance = [0.04, -0.001, -0.001, 0.0, 0.003, 0.05, -0.001]
    np.testing.assert_array_equal(touchdowns(load, clearance), [1, 6])


def test_stegosaurus_can_stand_and_repeat_a_four_beat_walk():
    result = run_validation(walk_seconds=14)
    assert result["passed"], [label for label, ok in result["checks"].items() if not ok]
